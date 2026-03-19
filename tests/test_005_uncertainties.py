import os
import random

os.environ["PYTHONHASHSEED"] = "42"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
random.seed(42)

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import iddefix


class TestAnalyticalImpedance:
    @classmethod
    def setup_class(cls):
        # Common synthetic case
        cls.parameters = {
            "1": [400, 30, 0.2e9],
            "2": [1000, 10, 1e9],
            "3": [500, 20, 1.75e9],
        }
        cls.frequency = np.linspace(0, 2e9, 1000)
        cls.noise = np.random.normal(0, 20, len(cls.frequency)) * (1 + 1j)
        cls.impedance = (
            iddefix.Impedances.n_Resonator_longitudinal_imp(
                cls.frequency, cls.parameters
            )
            + cls.noise
        )

        cls.N_resonators = 3
        cls.parameterBounds = [
            (0, 2000),
            (1, 1e3),
            (0.1e9, 2e9),
            (0, 2000),
            (1, 1e3),
            (0.1e9, 2e9),
            (0, 2000),
            (1, 1e3),
            (0.1e9, 2e9),
        ]

        cls.rtol = 1e-2
        cls.atol = 1e-6

        # Build + fit DE once for the class
        cls.DE_model = iddefix.EvolutionaryAlgorithm(
            cls.frequency,
            cls.impedance,
            N_resonators=cls.N_resonators,
            parameterBounds=cls.parameterBounds,
            plane="longitudinal",
            objectiveFunction=iddefix.ObjectiveFunctions.sumOfSquaredError,
        )
        cls.DE_model.run_differential_evolution(
            maxiter=2000,
            popsize=45,
            tol=0.01,
            mutation=(0.4, 1.0),
            crossover_rate=0.7,
        )
        cls.DE_model.run_minimization_algorithm()
        print(cls.DE_model.warning)

        cls.margin = [0.1] * 3
        cls.minimizationBounds = [
            sorted(((1 - cls.margin[i % 3]) * p, (1 + cls.margin[i % 3]) * p))
            for i, p in enumerate(cls.DE_model.evolutionParameters)
        ]

        def objective_function(x, *parameters):
            grouped_parameters = iddefix.utils.pars_to_dict(
                np.asarray(parameters)
            )
            predicted_y = cls.DE_model.fitFunction(x, grouped_parameters)
            return np.concatenate([predicted_y.real, predicted_y.imag])

        cls.popt, cls.pcov = curve_fit(
            objective_function,
            cls.frequency,
            np.concatenate([cls.impedance.real, cls.impedance.imag]),
            p0=cls.DE_model.evolutionParameters,
            bounds=np.array(cls.minimizationBounds).T,
            absolute_sigma=False,
        )
        cls.z_cf = cls.DE_model.fitFunction(
            cls.DE_model.frequency_data, cls.popt
        )

    # --- DE -------------------------------------------------------------------

    def test_DE_model(self):
        # Just smoke-checks that training completed
        assert self.DE_model is not None
        assert hasattr(self.DE_model, "minimizationParameters")
        assert hasattr(self.DE_model, "evolutionParameters")
        assert hasattr(self.DE_model, "minimizationParametersUncertainties")
        assert hasattr(self.DE_model, "evolutionParametersUncertainties")
        # Optional: ensure warnings didn't include "error"
        assert (
            "error" not in str(getattr(self.DE_model, "warning", "")).lower()
        )

    def test_abs_DE_impedance(self, plot: bool = False):
        z_true = np.abs(self.impedance)
        z_de = np.abs(self.DE_model.get_impedance(use_minimization=False))
        z_min = np.abs(self.DE_model.get_impedance())
        z_cf = np.abs(self.z_cf)

        if plot:
            plt.figure(figsize=(8, 5))
            plt.plot(
                self.frequency,
                z_true,
                label="Target impedance",
                lw=5,
                color="black",
            )
            plt.plot(self.frequency, z_de, label="DE fit", lw=2)
            plt.plot(
                self.frequency, z_min, label="Minimized DE fit", lw=2, ls="--"
            )
            plt.plot(self.frequency, z_min, label="Curve fit DE fit", ls=":")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("|Z(f)| [Ohm]")
            plt.title("Analytical resonator impedance fitting with DE")
            plt.legend()
            plt.show()

        assert z_de.shape == z_true.shape == z_min.shape == z_cf.shape
        assert (
            np.isfinite(z_true).all()
            and np.isfinite(z_de).all()
            and np.isfinite(z_min).all()
            and np.isfinite(z_cf).all()
        )

        np.testing.assert_allclose(z_min, z_cf, rtol=self.rtol, atol=self.atol)

    def test_uncertainties(self):
        uncertainties_min = self.DE_model.minimizationParametersUncertainties
        uncertainties_cf = np.sqrt(np.diag(self.pcov))

        np.testing.assert_allclose(
            uncertainties_min / uncertainties_cf,
            np.ones_like(uncertainties_min),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_warn_large_uncertainties_sets_flagged_params(self, capsys):
        # Build a tiny synthetic parameter vector: [Rs1, Q1, fres1, Rs2, Q2, fres2]
        params = np.array([100.0, 10.0, 1.0e9, 200.0, 20.0, 2.0e9])
        uncertainties = np.array([30.0, 1.0, 1.0e8, 10.0, 10.0, 1.0e8])

        # Relative uncertainties are [0.30, 0.10, 0.10, 0.05, 0.50, 0.05]
        self.DE_model.uncertainty_warning = 0.2
        self.DE_model._warn_large_uncertainties(params, uncertainties)

        expected = np.array([True, False, False, False, True, False])
        assert self.DE_model.flagged_params is not None
        assert self.DE_model.flagged_params.shape == params.shape
        np.testing.assert_array_equal(self.DE_model.flagged_params, expected)

        # Check that a warning message is emitted when at least one value is flagged.
        out = capsys.readouterr().out
        assert "relative uncertainty >= 0.20" in out


if __name__ == "__main__":
    # Manual run with plots (reusing the same test methods)
    t = TestAnalyticalImpedance()
    # pytest won’t call setup_class in this mode, so do it:
    t.setup_class()
    print(
        "Running analytical impedance fitting and uncertainty"
        " tests with plots..."
    )
    t.test_DE_model()
    t.test_abs_DE_impedance(plot=True)
    t.test_uncertainties()

import os
import numpy as np
import pytest
import iddefix
import matplotlib.pyplot as plt

class TestAnalyticalImpedance:
    # pytest: do NOT define __init__ on test classes

    @classmethod
    def setup_class(cls):
        # Common synthetic case
        cls.parameters = {
            '1': [400, 30, 0.2e9],
            '2': [1000, 10, 1e9],
            '3': [500, 20, 1.75e9],
        }
        cls.frequency = np.linspace(0, 2e9, 1000)
        cls.impedance = iddefix.Impedances.n_Resonator_longitudinal_imp(
            cls.frequency, cls.parameters
        )

        cls.N_resonators = 3
        cls.parameterBounds = [
            (0, 2000), (1, 1e3), (0.1e9, 2e9),
            (0, 2000), (1, 1e3), (0.1e9, 2e9),
            (0, 2000), (1, 1e3), (0.1e9, 2e9),
        ]

        cls.rtol = 1e-2
        cls.atol = 1e-6

        # Build + fit DE once for the class
        cls.DE_model = iddefix.EvolutionaryAlgorithm(
            cls.frequency,
            cls.impedance.real,  # could be complex; keeping your note
            N_resonators=cls.N_resonators,
            parameterBounds=cls.parameterBounds,
            plane="longitudinal",
            objectiveFunction="real",  # or iddefix.ObjectiveFunctions.sumOfSquaredError
        )
        cls.DE_model.run_differential_evolution(
            maxiter=1000, popsize=45, tol=0.01, mutation=(0.4, 1.0), crossover_rate=0.7
        )
        cls.DE_model.run_minimization_algorithm()

        # Build + fit CMA-ES once for the class
        cls.CMAES_model = iddefix.EvolutionaryAlgorithm(
            cls.frequency,
            cls.impedance.real,
            N_resonators=cls.N_resonators,
            parameterBounds=cls.parameterBounds,
            plane="longitudinal",
            objectiveFunction="real",
        )
        cls.CMAES_model.run_cmaes(maxiter=5000, popsize=50, sigma=0.15)
        cls.CMAES_model.run_minimization_algorithm()

    # --- DE -------------------------------------------------------------------

    def test_DE_model(self):
        # Just smoke-checks that training completed
        assert self.DE_model is not None
        assert hasattr(self.DE_model, "minimizationParameters")
        assert hasattr(self.DE_model, "evolutionParameters")
        # Optional: ensure warnings didn't include "error"
        assert "error" not in str(getattr(self.DE_model, "warning", "")).lower()

    def test_abs_DE_impedance(self, plot: bool = False):
        z_true = np.abs(self.impedance)
        z_de   = np.abs(self.DE_model.get_impedance(use_minimization=False))
        z_min  = np.abs(self.DE_model.get_impedance())

        if plot:
            plt.figure(figsize=(8, 5))
            plt.plot(self.frequency, z_true, label="Target impedance", lw=2)
            plt.plot(self.frequency, z_de,   label="DE fit", ls="--")
            plt.plot(self.frequency, z_min,  label="Minimized DE fit", ls=":")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("|Z(f)| [Ohm]")
            plt.title("Analytical resonator impedance fitting with DE")
            plt.legend()
            plt.show()

        assert z_de.shape == z_true.shape == z_min.shape
        assert np.isfinite(z_true).all() and np.isfinite(z_de).all() and np.isfinite(z_min).all()

        np.testing.assert_allclose(z_de,  z_true, rtol=self.rtol, atol=self.atol)
        np.testing.assert_allclose(z_min, z_true, rtol=self.rtol, atol=self.atol)
        np.testing.assert_allclose(z_min, z_de,   rtol=5e-3,     atol=self.atol)

        def rel_rmse(a, b):
            a = np.asarray(a); b = np.asarray(b)
            denom = max(1e-12, np.mean(np.abs(b)))
            return np.sqrt(np.mean((a - b) ** 2)) / denom

        assert rel_rmse(z_de,  z_true) < 0.02
        assert rel_rmse(z_min, z_true) < 0.01
        assert rel_rmse(z_min, z_de)   < 0.005

    def test_reim_DE_impedance(self, plot: bool = False):
        zr_true = np.real(self.impedance)
        zi_true = np.imag(self.impedance)
        zr_de   = np.real(self.DE_model.get_impedance(use_minimization=False))
        zi_de   = np.imag(self.DE_model.get_impedance(use_minimization=False))
        zr_min  = np.real(self.DE_model.get_impedance())
        zi_min  = np.imag(self.DE_model.get_impedance())

        if plot:
            fig, axs = plt.subplots(2, 1, figsize=(8, 10))
            axs[0].plot(self.frequency, zr_true, label="Target Real", lw=2)
            axs[0].plot(self.frequency, zr_de,   label="DE Real", ls="--")
            axs[0].plot(self.frequency, zr_min,  label="DE Real (min)", ls=":")
            axs[0].set_xlabel("Frequency [Hz]")
            axs[0].set_ylabel("Re{Z} [Ohm]")
            axs[0].legend()

            axs[1].plot(self.frequency, zi_true, label="Target Imag", lw=2)
            axs[1].plot(self.frequency, zi_de,   label="DE Imag", ls="--")
            axs[1].plot(self.frequency, zi_min,  label="DE Imag (min)", ls=":")
            axs[1].set_xlabel("Frequency [Hz]")
            axs[1].set_ylabel("Im{Z} [Ohm]")
            axs[1].legend()
            plt.tight_layout()
            plt.show()

        np.testing.assert_allclose(zr_de,  zr_true, rtol=self.rtol, atol=self.atol)
        np.testing.assert_allclose(zi_de,  zi_true, rtol=self.rtol, atol=self.atol)
        np.testing.assert_allclose(zr_min, zr_true, rtol=self.rtol, atol=self.atol)
        np.testing.assert_allclose(zi_min, zi_true, rtol=self.rtol, atol=self.atol)

    # --- CMA-ES ---------------------------------------------------------------

    def test_CMAES_model(self):
        assert self.CMAES_model is not None
        assert hasattr(self.CMAES_model, "minimizationParameters")
        assert hasattr(self.CMAES_model, "evolutionParameters")
        assert "error" not in str(getattr(self.CMAES_model, "warning", "")).lower()

    def test_abs_CMAES_impedance(self, plot: bool = False):
        z_true = np.abs(self.impedance)
        z_cma  = np.abs(self.CMAES_model.get_impedance(use_minimization=False))
        z_min  = np.abs(self.CMAES_model.get_impedance())

        if plot:
            plt.figure(figsize=(8, 5))
            plt.plot(self.frequency, z_true, label="Target impedance", lw=2)
            plt.plot(self.frequency, z_cma,  label="CMA-ES fit", ls="--")
            plt.plot(self.frequency, z_min,  label="Minimized CMA-ES fit", ls=":")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("|Z(f)| [Ohm]")
            plt.title("Analytical resonator impedance fitting with CMA-ES")
            plt.legend()
            plt.show()

        assert z_cma.shape == z_true.shape == z_min.shape
        assert np.isfinite(z_true).all() and np.isfinite(z_cma).all() and np.isfinite(z_min).all()

        np.testing.assert_allclose(z_cma,  z_true, rtol=self.rtol, atol=self.atol)
        np.testing.assert_allclose(z_min,  z_true, rtol=self.rtol, atol=self.atol)
        np.testing.assert_allclose(z_min,  z_cma,  rtol=5e-3,     atol=self.atol)

        def rel_rmse(a, b):
            a = np.asarray(a); b = np.asarray(b)
            denom = max(1e-12, np.mean(np.abs(b)))
            return np.sqrt(np.mean((a - b) ** 2)) / denom

        assert rel_rmse(z_cma, z_true) < 0.02
        assert rel_rmse(z_min, z_true) < 0.01
        assert rel_rmse(z_min, z_cma)  < 0.005

    def test_reim_CMAES_impedance(self, plot: bool = False):
        zr_true = np.real(self.impedance)
        zi_true = np.imag(self.impedance)
        zr_cma  = np.real(self.CMAES_model.get_impedance(use_minimization=False))
        zi_cma  = np.imag(self.CMAES_model.get_impedance(use_minimization=False))
        zr_min  = np.real(self.CMAES_model.get_impedance())
        zi_min  = np.imag(self.CMAES_model.get_impedance())

        if plot:
            fig, axs = plt.subplots(2, 1, figsize=(8, 10))
            axs[0].plot(self.frequency, zr_true, label="Target Real", lw=2)
            axs[0].plot(self.frequency, zr_cma,  label="CMA-ES Real", ls="--")
            axs[0].plot(self.frequency, zr_min,  label="CMA-ES Real (min)", ls=":")
            axs[0].set_xlabel("Frequency [Hz]")
            axs[0].set_ylabel("Re{Z} [Ohm]")
            axs[0].legend()

            axs[1].plot(self.frequency, zi_true, label="Target Imag", lw=2)
            axs[1].plot(self.frequency, zi_cma,  label="CMA-ES Imag", ls="--")
            axs[1].plot(self.frequency, zi_min,  label="CMA-ES Imag (min)", ls=":")
            axs[1].set_xlabel("Frequency [Hz]")
            axs[1].set_ylabel("Im{Z} [Ohm]")
            axs[1].legend()
            plt.tight_layout()
            plt.show()

        np.testing.assert_allclose(zr_cma, zr_true, rtol=self.rtol, atol=self.atol)
        np.testing.assert_allclose(zi_cma, zi_true, rtol=self.rtol, atol=self.atol)
        np.testing.assert_allclose(zr_min, zr_true, rtol=self.rtol, atol=self.atol)
        np.testing.assert_allclose(zi_min, zi_true, rtol=self.rtol, atol=self.atol)

    def test_table_display(self):
        print("For terminal:")
        self.DE_model.display_resonator_parameters(self.DE_model.minimizationParameters)
        print("\nFor Markdown:")
        self.DE_model.display_resonator_parameters(self.DE_model.minimizationParameters, to_markdown=True)

if __name__ == "__main__":
    # Manual run with plots (reusing the same test methods)
    t = TestAnalyticalImpedance()
    # pytest won’t call setup_class in this mode, so do it:
    t.setup_class()
    print("Running analytical impedance fitting tests with plots...")
    t.test_DE_model()
    t.test_abs_DE_impedance(plot=True)
    t.test_reim_DE_impedance(plot=True)
    t.test_CMAES_model()
    t.test_abs_CMAES_impedance(plot=True)
    t.test_reim_CMAES_impedance(plot=True)
    t.test_table_display()

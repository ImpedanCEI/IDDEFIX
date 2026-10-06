"""Analytical resonator workflows across fitting, wake, and impedance domains.

The tests compare fitted spectra with a target and follow resonator wakes through
the library's FFT, Gaussian convolution, deconvolution, and inverse transform.
Both planes, one or several resonators, and full or finite wake lengths are
covered.

Run ``pytest tests/test_003_analytical_checks.py --debug-plots`` to display the
values compared by each numerical assertion.
"""

import os
import random

os.environ["PYTHONHASHSEED"] = "42"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
random.seed(42)

import numpy as np
import pytest
from scipy.constants import c as c_light

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
        cls.impedance = iddefix.Impedances.n_Resonator_longitudinal_imp(
            cls.frequency, cls.parameters
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

        # Build + fit CMA-ES once for the class
        cls.CMAES_model = iddefix.EvolutionaryAlgorithm(
            cls.frequency,
            cls.impedance.real,
            N_resonators=cls.N_resonators,
            parameterBounds=cls.parameterBounds,
            plane="longitudinal",
            objectiveFunction="real",
        )
        cls.CMAES_model.run_cmaes(maxiter=5000, popsize=50, sigma=0.6, verbose=False)
        cls.CMAES_model.run_minimization_algorithm()
        print(cls.CMAES_model.warning)

        # Build + fit DE once for the class
        cls.DE_model = iddefix.EvolutionaryAlgorithm(
            cls.frequency,
            cls.impedance.real,  # could be complex
            N_resonators=cls.N_resonators,
            parameterBounds=cls.parameterBounds,
            plane="longitudinal",
            objectiveFunction="real",  # or iddefix.ObjectiveFunctions.sumOfSquaredErrorReal
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

    # --- DE -------------------------------------------------------------------

    def test_DE_model(self):
        # The solver and local minimizer both compute and store uncertainties.
        assert self.DE_model is not None
        for parameters, uncertainties in (
            (
                self.DE_model.evolutionParameters,
                self.DE_model.evolutionParametersUncertainties,
            ),
            (
                self.DE_model.minimizationParameters,
                self.DE_model.minimizationParametersUncertainties,
            ),
        ):
            assert uncertainties.shape == parameters.shape
            assert np.isfinite(uncertainties).all()
        assert "error" not in str(getattr(self.DE_model, "warning", "")).lower()

    def test_abs_DE_impedance(self, debug_plot):
        z_true = np.abs(self.impedance)
        z_de = np.abs(self.DE_model.get_impedance(use_minimization=False))
        z_min = np.abs(self.DE_model.get_impedance())

        debug_plot(
            self.frequency,
            {"DE fit": z_de, "Minimized DE fit": z_min},
            z_true,
            title="Analytical resonator impedance fitting with DE",
            xlabel="Frequency [Hz]",
            ylabel="|Z(f)| [Ohm]",
            expected_label="Target impedance",
        )

        assert z_de.shape == z_true.shape == z_min.shape
        assert (
            np.isfinite(z_true).all()
            and np.isfinite(z_de).all()
            and np.isfinite(z_min).all()
        )

        np.testing.assert_allclose(z_de, z_true, rtol=self.rtol, atol=self.atol)
        np.testing.assert_allclose(z_min, z_true, rtol=self.rtol, atol=self.atol)
        np.testing.assert_allclose(z_min, z_de, rtol=5e-3, atol=self.atol)

        def rel_rmse(a, b):
            a = np.asarray(a)
            b = np.asarray(b)
            denom = max(1e-12, np.mean(np.abs(b)))
            return np.sqrt(np.mean((a - b) ** 2)) / denom

        assert rel_rmse(z_de, z_true) < 0.02
        assert rel_rmse(z_min, z_true) < 0.01
        assert rel_rmse(z_min, z_de) < 0.005

    def test_reim_DE_impedance(self, debug_plot):
        z_true = self.impedance
        z_de = self.DE_model.get_impedance(use_minimization=False)
        z_min = self.DE_model.get_impedance()
        zr_true, zi_true = z_true.real, z_true.imag
        zr_de, zi_de = z_de.real, z_de.imag
        zr_min, zi_min = z_min.real, z_min.imag

        debug_plot(
            self.frequency,
            {"DE fit": z_de, "Minimized DE fit": z_min},
            z_true,
            title="Complex resonator impedance fitting with DE",
            xlabel="Frequency [Hz]",
            ylabel="Z(f) [Ohm]",
            expected_label="Target impedance",
        )

        np.testing.assert_allclose(zr_de, zr_true, rtol=self.rtol, atol=self.atol)
        np.testing.assert_allclose(zi_de, zi_true, rtol=self.rtol, atol=self.atol)
        np.testing.assert_allclose(zr_min, zr_true, rtol=self.rtol, atol=self.atol)
        np.testing.assert_allclose(zi_min, zi_true, rtol=self.rtol, atol=self.atol)

    # --- CMA-ES ---------------------------------------------------------------

    def test_CMAES_model(self):
        assert self.CMAES_model is not None
        for parameters, uncertainties in (
            (
                self.CMAES_model.evolutionParameters,
                self.CMAES_model.evolutionParametersUncertainties,
            ),
            (
                self.CMAES_model.minimizationParameters,
                self.CMAES_model.minimizationParametersUncertainties,
            ),
        ):
            assert uncertainties.shape == parameters.shape
            assert np.isfinite(uncertainties).all()
        assert "error" not in str(getattr(self.CMAES_model, "warning", "")).lower()

    def test_abs_CMAES_impedance(self, debug_plot):
        z_true = np.abs(self.impedance)
        z_cma = np.abs(self.CMAES_model.get_impedance(use_minimization=False))
        z_min = np.abs(self.CMAES_model.get_impedance())

        debug_plot(
            self.frequency,
            {"CMA-ES fit": z_cma, "Minimized CMA-ES fit": z_min},
            z_true,
            title="Analytical resonator impedance fitting with CMA-ES",
            xlabel="Frequency [Hz]",
            ylabel="|Z(f)| [Ohm]",
            expected_label="Target impedance",
        )

        assert z_cma.shape == z_true.shape == z_min.shape
        assert (
            np.isfinite(z_true).all()
            and np.isfinite(z_cma).all()
            and np.isfinite(z_min).all()
        )

        np.testing.assert_allclose(z_cma, z_true, rtol=self.rtol, atol=self.atol)
        np.testing.assert_allclose(z_min, z_true, rtol=self.rtol, atol=self.atol)
        np.testing.assert_allclose(z_min, z_cma, rtol=5e-3, atol=self.atol)

        def rel_rmse(a, b):
            a = np.asarray(a)
            b = np.asarray(b)
            denom = max(1e-12, np.mean(np.abs(b)))
            return np.sqrt(np.mean((a - b) ** 2)) / denom

        assert rel_rmse(z_cma, z_true) < 0.02
        assert rel_rmse(z_min, z_true) < 0.01
        assert rel_rmse(z_min, z_cma) < 0.005

    def test_reim_CMAES_impedance(self, debug_plot):
        z_true = self.impedance
        z_cma = self.CMAES_model.get_impedance(use_minimization=False)
        z_min = self.CMAES_model.get_impedance()
        zr_true, zi_true = z_true.real, z_true.imag
        zr_cma, zi_cma = z_cma.real, z_cma.imag
        zr_min, zi_min = z_min.real, z_min.imag

        debug_plot(
            self.frequency,
            {"CMA-ES fit": z_cma, "Minimized CMA-ES fit": z_min},
            z_true,
            title="Complex resonator impedance fitting with CMA-ES",
            xlabel="Frequency [Hz]",
            ylabel="Z(f) [Ohm]",
            expected_label="Target impedance",
        )

        np.testing.assert_allclose(zr_cma, zr_true, rtol=self.rtol, atol=self.atol)
        np.testing.assert_allclose(zi_cma, zi_true, rtol=self.rtol, atol=self.atol)
        np.testing.assert_allclose(zr_min, zr_true, rtol=self.rtol, atol=self.atol)
        np.testing.assert_allclose(zi_min, zi_true, rtol=self.rtol, atol=self.atol)

    def test_table_display(self):
        print("For terminal:")
        self.DE_model.display_resonator_parameters(self.DE_model.minimizationParameters)
        print("\nFor Markdown:")
        self.DE_model.display_resonator_parameters(
            self.DE_model.minimizationParameters, to_markdown=True
        )


# Analytical consistency workflows


def _loaded_resonators(plane, resonators):
    frequencies = np.array([0.35, 0.7, 1.0, 1.4, 1.9]) * 1e9
    model = iddefix.EvolutionaryAlgorithm(
        x_data=frequencies,
        y_data=np.zeros_like(frequencies),
        N_resonators=len(resonators),
        parameterBounds=[(0, 200), (0.2, 20), (0.3e9, 2e9)] * len(resonators),
        plane=plane,
        objectiveFunction="Real",
    )
    model.load_resonator_parameters(
        "\n".join(
            f"{number} | {Rs} | {Q} | {fr}"
            for number, (Rs, Q, fr) in enumerate(resonators, start=1)
        )
    )
    return model


def _assert_complex_spectrum_matches(actual, reference, tolerance=1e-3):
    """Compare both components without a relative-error spike at their zeros."""
    scale = np.linalg.norm(reference)
    for component in ("real", "imag"):
        difference = getattr(actual - reference, component)
        assert np.linalg.norm(difference) / scale < tolerance, component


def _fft_of_wake(times, wake, plane):
    # compute_fft integrates over distance. Divide by c for a time-domain transform.
    frequency, impedance = iddefix.compute_fft(
        times, wake / c_light, fmax=1.9e9, samples=190
    )
    if plane == "transverse":
        impedance *= 1j
    return frequency, impedance


def _deconvolve_potential(times, potential, sigma, plane, fmax=1.9e9):
    # The Gaussian used by compute_deconvolution contains a 1/c factor.
    frequency, impedance = iddefix.compute_deconvolution(
        times, potential / c_light, sigma, fmax=fmax, samples=int(fmax / 1e7)
    )
    if plane == "transverse":
        impedance *= 1j
    return frequency, impedance


def _wake_fft_and_analytical_impedance(plane, count, finite):
    if finite:
        # At 0.8 m these high-Q wakes retain substantial amplitude at the cut.
        resonators = [(100, 8, 1e9), (50, 10, 1.4e9)][:count]
        wake_length = 0.8
        step = (wake_length / c_light) / 4200
        times = (np.arange(4200) + 0.5) * step
    else:
        # The full wake supports overdamped, critical, and underdamped Q.
        resonators = (
            [(80, 0.3, 0.9e9)] if count == 1 else [(80, 0.5, 0.9e9), (50, 2, 1.4e9)]
        )
        wake_length = None
        step = 4e-12
        times = (np.arange(10000) + 0.5) * step

    model = _loaded_resonators(plane, resonators)
    frequency, fft_impedance = _fft_of_wake(times, model.get_wake(times), plane)
    useful_band = (frequency > 0.15e9) & (frequency < 1.8e9)
    analytical = model.get_impedance(frequency, wake_length=wake_length)
    if finite:
        full = model.get_impedance(frequency)
        finite_effect = np.linalg.norm(analytical[useful_band] - full[useful_band])
        assert finite_effect / np.linalg.norm(full[useful_band]) > 0.1
        if plane == "longitudinal":
            # The DC value = area of the truncated wake is generally nonzero.
            # (unless it is truncated at n\pi exactly)
            assert analytical[0] != 0
            np.testing.assert_allclose(fft_impedance[0], analytical[0], rtol=1e-5)

    return (
        frequency[useful_band],
        fft_impedance[useful_band],
        analytical[useful_band],
    )


@pytest.mark.parametrize("plane", ["longitudinal", "transverse"])
@pytest.mark.parametrize("count", [1, 2])
def test_full_wake_fft_matches_impedance(plane, count, debug_plot):
    """The full wake FFT reproduces real and imaginary impedance."""
    frequency, transformed, analytical = _wake_fft_and_analytical_impedance(
        plane, count, finite=False
    )
    debug_plot(
        frequency,
        transformed,
        analytical,
        title=f"Full {plane} wake FFT ({count} resonator(s))",
        xlabel="Frequency [Hz]",
        ylabel="Z(f)",
        expected_label="Analytical impedance",
    )
    _assert_complex_spectrum_matches(transformed, analytical)


@pytest.mark.parametrize("count", [1, 2])
def test_partial_transverse_wake_fft_matches_impedance(count, debug_plot):
    """The transverse formula agrees with the FFT of its truncated wake."""
    frequency, transformed, analytical = _wake_fft_and_analytical_impedance(
        "transverse", count, finite=True
    )
    debug_plot(
        frequency,
        transformed,
        analytical,
        title=f"Partial transverse wake FFT ({count} resonator(s))",
        xlabel="Frequency [Hz]",
        ylabel="Z(f)",
        expected_label="Analytical impedance",
    )
    _assert_complex_spectrum_matches(transformed, analytical)


@pytest.mark.parametrize("count", [1, 2])
def test_partial_longitudinal_wake_fft_matches_impedance(count, debug_plot):
    """The longitudinal formula agrees with the FFT of its truncated wake."""
    frequency, transformed, analytical = _wake_fft_and_analytical_impedance(
        "longitudinal", count, finite=True
    )
    debug_plot(
        frequency,
        transformed,
        analytical,
        title=f"Partial longitudinal wake FFT ({count} resonator(s))",
        xlabel="Frequency [Hz]",
        ylabel="Z(f)",
        expected_label="Analytical impedance",
    )
    _assert_complex_spectrum_matches(transformed, analytical)


def test_partial_critical_longitudinal_wake_fft_matches_impedance(debug_plot):
    """The finite longitudinal transform remains valid at critical damping."""
    model = _loaded_resonators("longitudinal", [(100, 0.5, 1e9)])
    wake_length = 0.03
    step = (wake_length / c_light) / 4200
    times = (np.arange(4200) + 0.5) * step
    frequency, transformed = _fft_of_wake(times, model.get_wake(times), "longitudinal")
    useful_band = (frequency > 0.15e9) & (frequency < 1.8e9)
    analytical = model.get_impedance(frequency, wake_length=wake_length)
    debug_plot(
        frequency[useful_band],
        transformed[useful_band],
        analytical[useful_band],
        title="Partial critical longitudinal wake FFT",
        xlabel="Frequency [Hz]",
        ylabel="Z(f)",
        expected_label="Analytical impedance",
    )
    _assert_complex_spectrum_matches(transformed[useful_band], analytical[useful_band])


@pytest.mark.parametrize("plane", ["longitudinal", "transverse"])
def test_partial_overdamped_wake_fft_matches_impedance(plane, debug_plot):
    """The finite transform supports the overdamped resonator branch."""
    model = _loaded_resonators(plane, [(100, 0.3, 1e9)])
    wake_length = 0.12
    step = (wake_length / c_light) / 4200
    times = (np.arange(4200) + 0.5) * step
    frequency, transformed = _fft_of_wake(times, model.get_wake(times), plane)
    useful_band = (frequency > 0.15e9) & (frequency < 1.8e9)
    analytical = model.get_impedance(frequency, wake_length=wake_length)
    full = model.get_impedance(frequency)
    finite_effect = np.linalg.norm(analytical[useful_band] - full[useful_band])
    assert finite_effect / np.linalg.norm(full[useful_band]) > 0.1
    debug_plot(
        frequency[useful_band],
        transformed[useful_band],
        analytical[useful_band],
        title=f"Partial overdamped {plane} wake FFT",
        xlabel="Frequency [Hz]",
        ylabel="Z(f)",
        expected_label="Analytical impedance",
    )
    _assert_complex_spectrum_matches(transformed[useful_band], analytical[useful_band])


@pytest.mark.parametrize("plane", ["longitudinal", "transverse"])
@pytest.mark.parametrize("Q", [0.3, 0.5, 2.0])
def test_partial_impedance_converges_to_full_impedance(plane, Q, debug_plot):
    """A sufficiently long finite wake reproduces the fully decayed impedance."""
    frequency = np.linspace(0, 2e9, 201)
    model = _loaded_resonators(plane, [(100, Q, 1e9)])

    partial = model.get_impedance(frequency, wake_length=30.0)
    full = model.get_impedance(frequency)

    debug_plot(
        frequency,
        partial,
        full,
        title=f"Long finite {plane} impedance at Q={Q}",
        xlabel="Frequency [Hz]",
        ylabel="Z(f)",
        expected_label="Fully decayed impedance",
    )
    np.testing.assert_allclose(partial, full, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("plane", ["longitudinal", "transverse"])
@pytest.mark.parametrize("count", [1, 2])
def test_analytical_wake_potential_deconvolves_to_impedance(plane, count, debug_plot):
    """Deconvolution of the Gaussian wake potential recovers full impedance."""
    model = _loaded_resonators(plane, [(100, 2, 1e9), (50, 3, 1.4e9)][:count])
    sigma = 0.1e-9
    step = 4e-12
    times = (np.arange(7500) - 2500 + 0.5) * step
    potential = model.get_wake_potential(times, sigma=sigma)

    frequency, recovered = _deconvolve_potential(times, potential, sigma, plane)
    useful_band = (frequency > 0.15e9) & (frequency < 1.8e9)
    frequency = frequency[useful_band]
    recovered = recovered[useful_band]
    analytical = model.get_impedance(frequency)
    debug_plot(
        frequency,
        recovered,
        analytical,
        title=f"Full {plane} potential deconvolution ({count} resonator(s))",
        xlabel="Frequency [Hz]",
        ylabel="Z(f)",
        expected_label="Analytical impedance",
    )
    _assert_complex_spectrum_matches(recovered, analytical)


@pytest.mark.parametrize("plane", ["longitudinal", "transverse"])
@pytest.mark.parametrize("count", [1, 2])
def test_full_wake_convolves_to_analytical_potential(plane, count, debug_plot):
    """The analytical Gaussian potential agrees with public convolution."""
    model = _loaded_resonators(plane, [(100, 2, 1e9), (50, 3, 1.4e9)][:count])
    sigma = 0.1e-9
    step = 4e-12
    times = (np.arange(10000) - 5000 + 0.5) * step
    potential_time, numerical_potential = iddefix.compute_convolution(
        times, model.get_wake(times), sigma, kernel="scipy"
    )
    useful_window = (potential_time > -0.4e-9) & (potential_time < 4e-9)
    analytical = model.get_wake_potential(potential_time[useful_window], sigma=sigma)
    debug_plot(
        potential_time[useful_window],
        numerical_potential[useful_window],
        analytical,
        title=f"Full {plane} wake convolution ({count} resonator(s))",
        xlabel="Time [s]",
        ylabel="Wake potential",
        expected_label="Analytical wake potential",
    )
    relative_error = np.linalg.norm(numerical_potential[useful_window] - analytical)
    relative_error /= np.linalg.norm(analytical)
    assert relative_error < 1e-3


def _partial_potential_and_analytical_impedance(plane, count):
    model = _loaded_resonators(plane, [(100, 8, 1e9), (50, 10, 1.4e9)][:count])
    sigma = 0.1e-9
    wake_length = 0.8
    times = (np.arange(10000) - 5000 + 0.5) * 4e-12
    wake = model.get_wake(times)
    wake = np.where(times <= wake_length / c_light, wake, 0.0)
    potential_time, potential = iddefix.compute_convolution(
        times, wake, sigma, kernel="scipy"
    )
    frequency, recovered = _deconvolve_potential(
        potential_time, potential, sigma, plane
    )
    useful_band = (frequency > 0.15e9) & (frequency < 1.8e9)

    analytical = model.get_impedance(frequency, wake_length=wake_length)
    full = model.get_impedance(frequency)
    finite_effect = np.linalg.norm(analytical[useful_band] - full[useful_band])
    assert finite_effect / np.linalg.norm(full[useful_band]) > 0.1
    return frequency[useful_band], recovered[useful_band], analytical[useful_band]


@pytest.mark.parametrize("count", [1, 2])
def test_partial_transverse_potential_deconvolves_to_impedance(count, debug_plot):
    """Convolution then deconvolution preserves finite transverse impedance."""
    frequency, recovered, analytical = _partial_potential_and_analytical_impedance(
        "transverse", count
    )
    debug_plot(
        frequency,
        recovered,
        analytical,
        title=f"Partial transverse potential deconvolution ({count} resonator(s))",
        xlabel="Frequency [Hz]",
        ylabel="Z(f)",
        expected_label="Analytical impedance",
    )
    _assert_complex_spectrum_matches(recovered, analytical)


@pytest.mark.parametrize("count", [1, 2])
def test_partial_longitudinal_potential_deconvolves_to_impedance(count, debug_plot):
    """Convolution then deconvolution preserves finite longitudinal impedance."""
    frequency, recovered, analytical = _partial_potential_and_analytical_impedance(
        "longitudinal", count
    )
    debug_plot(
        frequency,
        recovered,
        analytical,
        title=f"Partial longitudinal potential deconvolution ({count} resonator(s))",
        xlabel="Frequency [Hz]",
        ylabel="Z(f)",
        expected_label="Analytical impedance",
    )
    _assert_complex_spectrum_matches(recovered, analytical)


@pytest.mark.parametrize("plane", ["longitudinal", "transverse"])
def test_wake_potential_deconvolves_back_to_wake(plane, debug_plot):
    """Recover a wake through public deconvolution and inverse integration."""
    pytest.importorskip("neffint")
    model = _loaded_resonators(plane, [(100, 2, 1e9)])
    sigma = 0.1e-9
    times = (np.arange(10000) - 5000 + 0.5) * 4e-12
    potential_time, potential = iddefix.compute_convolution(
        times, model.get_wake(times), sigma, kernel="scipy"
    )
    frequency, impedance = _deconvolve_potential(
        potential_time, potential, sigma, plane, fmax=10e9
    )
    probe_times = np.linspace(0.2e-9, 2e-9, 40)
    _, recovered = iddefix.compute_ineffint(
        frequency, impedance, times=probe_times, adaptative=False, plane=plane
    )
    expected = model.get_wake(probe_times)
    debug_plot(
        probe_times,
        recovered,
        expected,
        title=f"{plane.capitalize()} potential-to-wake round trip",
        xlabel="Time [s]",
        ylabel="Wake",
        expected_label="Analytical wake",
    )
    relative_error = np.linalg.norm(recovered - expected) / np.linalg.norm(expected)
    assert relative_error < 0.07

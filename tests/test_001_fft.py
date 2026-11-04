"""FFT workflows for converting wake data into impedance spectra.

These tests compare reconstructed wakes and impedances with reference data.

Run ``pytest tests/test_001_fft.py --debug-plots`` to display the asserted
values in a dedicated debug plot.
"""

import numpy as np
import pytest
from scipy.constants import c as c_light

import iddefix


# FFT time-origin phase
@pytest.mark.parametrize(
    "time_origin",
    [0.0, 3.5e-11, -4.0e-11, 0.5e-11],
    ids=["zero", "positive", "negative", "midpoint"],
)
def test_fft_preserves_time_origin_phase(time_origin, debug_plot):
    """Shifting the sample times produces the Fourier time-shift phase."""
    step = 1e-11
    time = np.arange(1000) * step
    wake = np.exp(-time / 2e-9) * np.cos(2 * np.pi * 0.8e9 * time)

    frequency, impedance = iddefix.compute_fft(time, wake, fmax=2e9, samples=200)
    shifted_frequency, shifted_impedance = iddefix.compute_fft(
        time + time_origin, wake, fmax=2e9, samples=200
    )

    np.testing.assert_allclose(shifted_frequency, frequency, rtol=1e-14)
    expected = impedance * np.exp(-2j * np.pi * frequency * time_origin)
    debug_plot(
        frequency,
        shifted_impedance,
        expected,
        title=f"FFT with sample origin $t_0={time_origin:.1e}$ s",
        xlabel="Frequency [Hz]",
        ylabel="Z(f)",
        expected_label="Fourier time-shift reference",
    )
    np.testing.assert_allclose(shifted_impedance, expected, rtol=1e-12, atol=1e-12)


# Inverse-transform causality and adaptive integration
def test_ineffint_can_enforce_causality():
    """Causal mode zeros negative times and preserves positive-time samples."""
    pytest.importorskip("neffint")
    # Neffint 1.1's integrator does not converge at exactly t=0.
    time = np.array([-1e-10, 1e-10])
    frequency = np.linspace(0, 2e9, 1001)
    impedance = np.exp(-0.5 * (frequency / 0.5e9) ** 2)

    _, wake = iddefix.compute_ineffint(
        frequency, impedance, times=time, adaptative=False
    )
    _, causal_wake = iddefix.compute_ineffint(
        frequency, impedance, times=time, adaptative=False, enforce_causality=True
    )
    assert np.all(np.isfinite(wake[time < 0]))
    assert np.any(wake[time < 0] != 0)
    np.testing.assert_array_equal(causal_wake[time < 0], 0.0)
    np.testing.assert_allclose(causal_wake[time > 0], wake[time > 0])


@pytest.mark.parametrize("error", ["Abs", "rms"])
@pytest.mark.parametrize("direction", ["impedance_to_wake", "wake_to_impedance"])
def test_adaptive_neffint_paths(error, direction):
    """Adaptive transforms use Neffint and evaluate on the requested grid."""
    pytest.importorskip("neffint")
    source_grid = np.array([0.0, 2.0])
    source_values = np.array([1.0, 3.0])
    target_grid = np.array([0.25, 0.75])

    if direction == "impedance_to_wake":
        returned_grid, result = iddefix.compute_ineffint(
            source_grid,
            source_values,
            times=target_grid,
            adaptative=True,
            interpolation="linear",
            error=error,
        )
    else:
        returned_grid, result = iddefix.compute_neffint(
            source_grid,
            source_values,
            frequencies=target_grid,
            adaptative=True,
            interpolation="linear",
            error=error,
        )

    np.testing.assert_array_equal(returned_grid, target_grid)
    assert result.shape == target_grid.shape
    assert np.all(np.isfinite(result))


def test_interpolation_error_metrics():
    """Interpolation metrics handle scalar complex values and vector outputs."""
    expected = np.array([1.0, 2.0])
    scalar_error = iddefix.utils.interpolation_error_abs(
        np.array([1.0 + 1.0j, 3.0]), np.array([2.0 + 1.0j, 1.0])
    )
    np.testing.assert_array_equal(scalar_error, expected)

    vector_error = iddefix.utils.interpolation_error_rms(
        np.array([[1.0 + 1.0j, 3.0], [0.0, 2.0]]),
        np.array([[2.0 + 1.0j, 1.0], [3.0, 6.0]]),
    )
    np.testing.assert_allclose(vector_error, [np.sqrt(2.5), np.sqrt(12.5)])
    scalar_rms_error = iddefix.utils.interpolation_error_rms(
        np.array([1.0 + 1.0j, 3.0]), np.array([2.0 + 1.0j, 1.0])
    )
    np.testing.assert_array_equal(scalar_rms_error, expected)


# Reference SPS model and data fixture
def _load_data():
    """Load and prepare data once for all tests."""
    data_wake_potential = np.loadtxt(
        "examples/data/004_SPS_model_transitions_q26.txt",
        comments="#",
        delimiter="\t",
    )

    data_wake_time = data_wake_potential[:, 0] * 1e-9  # [s]
    data_wake_dipolar = data_wake_potential[:, 2]
    sigma = 1e-10

    DE_model = iddefix.EvolutionaryAlgorithm(
        wake_potential_x_data=data_wake_time,
        wake_potential_y_data=data_wake_dipolar * c_light,
        parameterBounds=None,
        plane="transverse",
        fitFunction="wake potential",
        sigma=sigma,
    )

    # Preload DE parameters
    data_str = """
        1     |        2.22e+00        |      76.87       |    1.005e+09
        2     |        7.62e+00        |      138.95      |    1.176e+09
        3     |        1.15e+00        |      15.49       |    1.268e+09
        4     |        1.19e+00        |      39.99       |    1.657e+09
        5     |        1.54e+00        |      169.72      |    2.075e+09
        6     |        1.79e+00        |      177.73      |    2.199e+09
        7     |        1.67e+00        |      53.54       |    2.251e+09
        8     |        1.87e+00        |      39.01       |    2.431e+09
        9     |        1.84e+00        |       5.01       |    2.675e+09
        10    |        1.99e+00        |      178.88      |    2.908e+09
        11    |        1.99e+00        |      38.55       |    3.184e+09
    """.strip()

    DE_model.load_resonator_parameters(data_str)

    return DE_model, data_wake_time, data_wake_dipolar, sigma


@pytest.fixture(scope="module")
def load_data():
    return _load_data()


# Wake and impedance reconstruction
def test_compare_wakes(load_data, debug_plot, adaptative=False):
    """Check that neffint wake and DE model wake are consistent."""
    DE_model, _, _, _ = load_data
    time = np.linspace(1e-11, 50e-9, 1000)
    f_fd = np.linspace(0, 5e9, 1000)

    pytest.importorskip("neffint")
    Z_fd = DE_model.get_impedance(frequency_data=f_fd, wake_length=None)
    t, W = iddefix.compute_ineffint(
        f_fd,
        Z_fd,
        times=time,
        plane="transverse",
        adaptative=adaptative,
    )
    W_de = DE_model.get_wake(time)

    # Test numerical similarity (correlation > 0.95)
    corr = np.corrcoef(W_de, W)[0, 1]
    debug_plot(
        time,
        W,
        W_de,
        title="Wake reconstructed with iNeffint",
        xlabel="Time [s]",
        ylabel="Wake [V/C/m]",
        expected_label="DE wake",
    )
    assert corr > 0.95, f"Wake correlation too low: {corr:.3f}"


def test_compare_impedances(load_data, debug_plot, adaptative=False):
    """Compare impedance computed from DE model, FFT, and neffint."""
    pytest.importorskip("neffint")
    DE_model, _, _, _ = load_data
    f_de = np.linspace(1, 5e9, 2000)
    time = np.linspace(1e-11, 50e-9, 1000)
    Z_de = DE_model.get_impedance(frequency_data=f_de, wake_length=None)
    W_de = DE_model.get_wake(time)

    f_fft, Z_fft = iddefix.compute_fft(time, W_de / c_light, fmax=5e9)

    # Reconstruct this test's wake independently of other tests.
    reconstruction_f = np.linspace(0, 5e9, 1000)
    reconstruction_z = DE_model.get_impedance(reconstruction_f)
    _, reconstructed_wake = iddefix.compute_ineffint(
        reconstruction_f,
        reconstruction_z,
        times=time,
        plane="transverse",
        adaptative=adaptative,
    )
    f_inft, Z_inft = iddefix.compute_fft(time, reconstructed_wake / c_light, fmax=5e9)

    f_nft, Z_nft = iddefix.compute_neffint(
        time, DE_model.get_wake(time), frequencies=f_de, adaptative=adaptative
    )

    Z_fft *= 1j  # transverse
    Z_inft *= 1j  # transverse
    Z_nft *= 1j  # transverse

    # Test that |Z| distributions are roughly consistent
    rel_error = np.mean(np.abs(np.abs(Z_nft) - np.abs(Z_de))) / np.mean(np.abs(Z_de))
    debug_plot(
        f_de,
        Z_nft,
        Z_de,
        title="Impedance reconstructed with Neffint",
        xlabel="Frequency [Hz]",
        ylabel="Transverse impedance [Ohm/m]",
        expected_label="Analytical resonator impedance",
    )
    for name, frequency, impedance in (
        ("FFT of the resonator wake", f_fft, Z_fft),
        ("FFT of the reconstructed wake", f_inft, Z_inft),
    ):
        reference = np.interp(frequency, f_de, Z_de.real) + 1j * np.interp(
            frequency, f_de, Z_de.imag
        )
        debug_plot(
            frequency,
            impedance,
            reference,
            title=name,
            xlabel="Frequency [Hz]",
            ylabel="Transverse impedance [Ohm/m]",
            expected_label="Analytical resonator impedance",
        )
    assert rel_error < 0.1, f"Relative impedance error too high: {rel_error:.2%}"

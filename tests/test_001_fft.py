"""FFT workflows for converting wake data into impedance spectra.

These tests compare reconstructed wakes and impedances with reference data.

Run ``pytest tests/test_001_fft.py --debug-plots`` to display the asserted
values in a dedicated debug plot.
"""

import sys
from io import StringIO

import numpy as np
import pytest
from scipy.constants import c as c_light

sys.path.append("../")
import iddefix


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
        data_wake_time,
        data_wake_dipolar * c_light,
        N_resonators=10,
        parameterBounds=None,
        plane="transverse",
        fitFunction="wake potential",
        sigma=sigma,
    )

    # Preload DE parameters
    data_str = StringIO(
        """
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
    )

    DE_model.minimizationParameters = np.loadtxt(
        data_str, skiprows=0, usecols=(1, 2, 3), delimiter="|", dtype=float
    ).flatten()

    return DE_model, data_wake_time, data_wake_dipolar, sigma


# ---------- Pytest fixture ----------
@pytest.fixture(scope="module")
def load_data():
    return _load_data()


def test_compare_wakes(load_data, debug_plot, adaptative=False):
    """Check that neffint wake and DE model wake are consistent."""
    DE_model, _, _, _ = load_data
    time = np.linspace(1e-11, 50e-9, 1000)
    f_fd = np.linspace(0, 5e9, 1000)

    global t, W
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
    DE_model, _, _, _ = load_data
    f_de = np.linspace(1, 5e9, 10000)
    time = np.linspace(0, 50e-9, 1000)
    Z_de = DE_model.get_impedance(frequency_data=f_de, wake_length=None)
    W_de = DE_model.get_wake(time)

    f_fft, Z_fft = iddefix.compute_fft(time, W_de / c_light, fmax=5e9)

    f_inft, Z_inft = iddefix.compute_fft(time, W / c_light, fmax=5e9)

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

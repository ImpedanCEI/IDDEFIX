"""Linear baseline correction checked against Fourier integrals and wake data."""

import numpy as np
import pytest
from scipy.constants import c
from scipy.integrate import quad

import iddefix
from iddefix.baseline import linear_baseline_impedance, remove_baseline


# Baseline physics and domain consistency
@pytest.mark.parametrize("plane", ["longitudinal", "transverse"])
def test_baseline_wake_impedance_consistency(plane):
    """The same physical slope removes a ramp before or after transformation."""
    time = np.linspace(0, 2e-9, 20001)
    slope = 1e18
    wake = slope * time
    corrected, _ = remove_baseline(time, wake, "wake", slope=slope)
    np.testing.assert_allclose(corrected, 0)
    sampled = wake.copy()
    # Trapezoidal endpoint weights approximate the continuous ramp integral.
    sampled[[0, -1]] *= 0.5
    frequency, impedance = iddefix.compute_fft(time, sampled / c, fmax=1e9)
    if plane == "transverse":
        impedance *= 1j
    corrected, _ = remove_baseline(
        frequency,
        impedance,
        "impedance",
        slope=slope,
        wake_length=c * time[-1],
        plane=plane,
    )
    assert np.max(np.abs(corrected)) < 1e-6 * np.max(np.abs(impedance))


@pytest.mark.parametrize("plane", ["longitudinal", "transverse"])
def test_linear_baseline_transform(plane):
    """The correction follows the finite ramp's Fourier integral, including DC."""
    duration = 8e-9
    slope = 1e10
    frequency = np.array([0.0, 0.2e9, 0.7e9])
    baseline = linear_baseline_impedance(frequency, slope, c * duration, plane)
    expected = []
    for value in frequency:
        omega = 2 * np.pi * value
        real = quad(lambda time: time * np.cos(omega * time), 0, duration)[0]
        imag = -quad(lambda time: time * np.sin(omega * time), 0, duration)[0]
        expected.append(slope * (real + 1j * imag))
    expected = np.asarray(expected)
    if plane == "transverse":
        expected *= 1j
    np.testing.assert_allclose(baseline, expected, rtol=1e-10, atol=1e-10)


# SmartBounds integration
@pytest.mark.parametrize("correction", [True, 1e18])
def test_magnitude_baseline_rejected(correction):
    """Magnitude-only spectra cannot be corrected without phase."""
    with pytest.raises(ValueError):
        iddefix.SmartBoundDetermination(
            np.linspace(0, 2e9, 101),
            np.ones(101),
            impedance_type="absolute",
            wake_length=1,
            remove_baseline=correction,
        )


@pytest.mark.parametrize("plane", ["longitudinal", "transverse"])
def test_baseline_correction(plane):
    """Removing a known ramp restores the clean resonance estimates."""
    frequency = np.linspace(0, 2e9, 2001)
    wake_length = 1.0
    slope = 1e11 * c
    formula = getattr(iddefix.Impedances, f"Resonator_{plane}_imp")
    resistance = 100.0 if plane == "longitudinal" else 4e6
    clean = formula(frequency, resistance, 2.0, 0.8e9, wake_length=wake_length)
    baseline = linear_baseline_impedance(frequency, slope, wake_length, plane)
    contaminated = clean + baseline
    options = dict(
        plane=plane, wake_length=wake_length, minimum_peak_height=1.0, use_Hann=True
    )
    reference = iddefix.SmartBoundDetermination(frequency, clean, **options)
    corrected = iddefix.SmartBoundDetermination(
        frequency, contaminated, remove_baseline=slope, **options
    )

    assert corrected.baseline_slope == slope
    assert corrected.N_resonators == reference.N_resonators
    np.testing.assert_allclose(
        corrected.parameterEstimates, reference.parameterEstimates
    )
    np.testing.assert_allclose(corrected.parameterBounds, reference.parameterBounds)


@pytest.mark.parametrize("plane", ["longitudinal", "transverse"])
@pytest.mark.parametrize("complex_input", [False, True])
def test_calculate_baseline_slope(plane, complex_input):
    """The public helper fits a real slope from real or complex impedance."""
    frequency = np.linspace(0, 2e9, 101)
    wake_length = 0.5
    slope = 1e11
    baseline = linear_baseline_impedance(frequency, slope, wake_length, plane)
    impedance = baseline if complex_input else baseline.real

    fitted = iddefix.SmartBoundDetermination.calculate_baseline_slope(
        frequency, impedance, wake_length, plane
    )

    assert fitted == pytest.approx(slope)


def test_baseline_requires_wake_length():
    """A ramp transform needs the finite wake's integration interval."""
    frequency = np.linspace(0, 2e9, 101)
    with pytest.raises(ValueError, match="wake_length"):
        iddefix.SmartBoundDetermination(
            frequency, np.ones_like(frequency), remove_baseline=True
        )

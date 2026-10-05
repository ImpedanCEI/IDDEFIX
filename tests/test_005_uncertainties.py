"""Uncertainty workflows for fitted parameters and impedance envelopes.

These tests compare parameter errors with SciPy, exercise derivative bounds,
and check that uncertainty envelopes contain the nominal impedance. They also
cover real-only measurements, for which imaginary residuals must be ignored.
"""

import numpy as np
import pytest
from scipy.optimize import curve_fit

import iddefix


def _model(frequency, impedance, *, wake_length=None, objective="Complex"):
    return iddefix.EvolutionaryAlgorithm(
        x_data=frequency,
        y_data=impedance,
        N_resonators=1,
        parameterBounds=[(50.0, 150.0), (0.5, 20.0), (0.8e9, 1.2e9)],
        wake_length=wake_length,
        objectiveFunction=objective,
    )


def test_parameter_uncertainties_agree_with_scipy_curve_fit():
    frequency = np.linspace(0.7e9, 1.3e9, 81)
    parameters = [100.0, 3.0, 1e9]
    impedance = iddefix.Impedances.Resonator_longitudinal_imp(frequency, *parameters)
    rng = np.random.default_rng(42)
    measured = impedance + rng.normal(0, 0.2, frequency.size)
    measured += 1j * rng.normal(0, 0.2, frequency.size)

    def stacked_impedance(x, Rs, Q, fr):
        predicted = iddefix.Impedances.Resonator_longitudinal_imp(x, Rs, Q, fr)
        return np.r_[predicted.real, predicted.imag]

    optimum, covariance = curve_fit(
        stacked_impedance,
        frequency,
        np.r_[measured.real, measured.imag],
        p0=parameters,
        bounds=([50, 0.5, 0.8e9], [150, 20, 1.2e9]),
    )
    model = _model(frequency, measured)
    model.load_resonator_parameters(
        f"1 | {optimum[0]:.17g} | {optimum[1]:.17g} | {optimum[2]:.17g}"
    )

    uncertainties = model.get_uncertainties()

    np.testing.assert_allclose(uncertainties, np.sqrt(np.diag(covariance)), rtol=1e-4)


def test_uncertainties_are_finite_at_partial_wake_q_boundary():
    frequency = np.linspace(0.7e9, 1.3e9, 80)
    parameters = [100.0, 0.5, 1e9]
    impedance = iddefix.Impedances.Resonator_longitudinal_imp(
        frequency, *parameters, wake_length=30.0
    )
    model = _model(frequency, impedance, wake_length=30.0)
    model.load_resonator_parameters("1 | 100 | 0.5 | 1e9")

    uncertainties = model.get_uncertainties()

    assert np.isfinite(uncertainties).all()


def test_loaded_parameters_can_recompute_uncertainties_outside_fit_bounds():
    frequency = np.linspace(0.7e9, 1.3e9, 60)
    parameters = [1000.0, 100.0, 1e9]
    impedance = iddefix.Impedances.Resonator_longitudinal_imp(
        frequency, *parameters, wake_length=30.0
    ).real + 0.1 * np.sin(np.linspace(0.0, 2 * np.pi, frequency.size))
    model = _model(frequency, impedance, wake_length=30.0, objective="Real")
    model.load_resonator_parameters("1 | 1000 | 100 | 1e9")
    original_bounds = model.parameterBounds.copy()

    uncertainties = model.get_uncertainties()

    assert np.isfinite(uncertainties).all()
    assert np.all(uncertainties > 0)
    assert model.parameterBounds == original_bounds


@pytest.mark.parametrize("vary", ["R", "Q", "both"])
def test_impedance_envelope_contains_nominal_with_large_errors(vary):
    frequency = np.linspace(0.5e9, 1.5e9, 101)
    model = _model(frequency, np.zeros_like(frequency))
    model.load_resonator_parameters("1 | 300 ± 1000 | 10 ± 1000 | 1e9 ± 1e8")
    nominal = model.get_impedance(frequency).real

    lower, upper = model.get_impedance_uncertainty(frequency, vary=vary)

    assert np.isfinite(lower).all() and np.isfinite(upper).all()
    assert np.all(lower <= nominal + 1e-12)
    assert np.all(nominal <= upper + 1e-12)
    if vary == "R":
        peak = np.argmin(np.abs(frequency - 1e9))
        np.testing.assert_allclose([lower[peak], upper[peak]], [150.0, 600.0])


def test_partial_wake_envelope_contains_nominal():
    frequency = np.linspace(0.5e9, 1.5e9, 101)
    model = _model(
        frequency,
        np.zeros_like(frequency),
        wake_length=30.0,
    )
    model.load_resonator_parameters("1 | 300 ± 1000 | 10 ± 1000 | 1e9 ± 1e8")

    lower, upper = model.get_impedance_uncertainty(frequency, wake_length=30.0)
    nominal = model.get_impedance(frequency, wake_length=30.0).real

    assert np.all(lower <= nominal + 1e-12)
    assert np.all(nominal <= upper + 1e-12)


def test_real_data_uncertainties_ignore_model_imaginary_part():
    frequency = np.linspace(0.7e9, 1.3e9, 31)
    parameters = [100.0, 3.0, 1e9]
    impedance = iddefix.Impedances.Resonator_longitudinal_imp(frequency, *parameters)
    real_data = impedance.real + 0.1 * np.sin(np.linspace(0, 2 * np.pi, 31))
    model = _model(frequency, real_data, objective="Real")
    model.load_resonator_parameters("1 | 100 | 3 | 1e9")

    original_fit = model.fitFunction
    real_uncertainties = model.get_uncertainties()

    def shifted_fit(x, pars):
        return original_fit(x, pars) + 1000j

    model.fitFunction = shifted_fit
    shifted_uncertainties = model.get_uncertainties()
    np.testing.assert_allclose(shifted_uncertainties, real_uncertainties)

    model.y_data = real_data + 1j * impedance.imag
    complex_uncertainties = model.get_uncertainties()
    assert np.all(complex_uncertainties > real_uncertainties)

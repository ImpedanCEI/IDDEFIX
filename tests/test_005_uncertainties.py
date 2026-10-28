"""Uncertainty workflows for fitted parameters and impedance envelopes.

These tests compare parameter errors with SciPy, exercise derivative bounds,
and check that uncertainty envelopes contain the nominal impedance. They also
cover real-only measurements, for which imaginary residuals must be ignored.
"""

import numpy as np
import pytest
from scipy.optimize import curve_fit

import iddefix


def _model(frequency, impedance, *, wake_length=None, objective="Complex", **kwargs):
    return iddefix.EvolutionaryAlgorithm(
        impedance_x_data=frequency,
        impedance_y_data=impedance,
        N_resonators=1,
        parameterBounds=[(50.0, 150.0), (0.5, 20.0), (0.8e9, 1.2e9)],
        wake_length=wake_length,
        objectiveFunction=objective,
        **kwargs,
    )


# Parameter covariance against SciPy
@pytest.mark.parametrize("complex_data", [False, True])
def test_parameter_uncertainties_agree_with_scipy_curve_fit(complex_data):
    """Real and complex measurement uncertainties agree with SciPy's curve fit."""
    frequency = np.linspace(0.7e9, 1.3e9, 81)
    parameters = [100.0, 3.0, 1e9]
    impedance = iddefix.Impedances.Resonator_longitudinal_imp(frequency, *parameters)
    rng = np.random.default_rng(42)
    measured = impedance + rng.normal(0, 0.2, frequency.size)
    if complex_data:
        measured += 1j * rng.normal(0, 0.2, frequency.size)
    else:
        measured = measured.real

    def stacked_impedance(x, Rs, Q, fr):
        predicted = iddefix.Impedances.Resonator_longitudinal_imp(x, Rs, Q, fr)
        return np.r_[predicted.real, predicted.imag] if complex_data else predicted.real

    optimum, covariance = curve_fit(
        stacked_impedance,
        frequency,
        np.r_[measured.real, measured.imag] if complex_data else measured,
        p0=parameters,
        bounds=([50, 0.5, 0.8e9], [150, 20, 1.2e9]),
    )
    model = _model(frequency, measured)
    model.load_resonator_parameters(
        f"1 | {optimum[0]:.17g} | {optimum[1]:.17g} | {optimum[2]:.17g}"
    )

    uncertainties = model.get_uncertainties()

    np.testing.assert_allclose(uncertainties, np.sqrt(np.diag(covariance)), rtol=1e-4)


@pytest.mark.parametrize("complex_data", [False, True])
def test_joint_uncertainties_match_scipy(complex_data):
    """Public joint covariance agrees with an independent weighted curve fit."""
    frequency = np.linspace(0.5e9, 1.5e9, 41)
    time = np.linspace(0, 2e-9, 61)
    parameters = np.array([100.0, 2.0, 1e9])
    impedance = iddefix.Impedances.Resonator_longitudinal_imp(frequency, *parameters)
    wake = iddefix.Wakes.Resonator_longitudinal_wake(time, *parameters)
    impedance += 0.1 * np.sin(np.arange(frequency.size))
    if not complex_data:
        impedance = impedance.real
    wake += 1e9 * np.cos(np.arange(time.size))
    z_norm, w_norm = np.linalg.norm(impedance), np.linalg.norm(wake)
    measured = [impedance.real / z_norm]
    if complex_data:
        measured.append(impedance.imag / z_norm)
    measured.append(wake / w_norm)
    measured = np.concatenate(measured)

    def combined_model(_, Rs, Q, frequency_ghz):
        # GHz keeps SciPy's independent fit parameters on comparable scales.
        p = [Rs, Q, frequency_ghz * 1e9]
        z = iddefix.Impedances.Resonator_longitudinal_imp(frequency, *p)
        w = iddefix.Wakes.Resonator_longitudinal_wake(time, *p)
        prediction = [z.real / z_norm]
        if complex_data:
            prediction.append(z.imag / z_norm)
        return np.concatenate([*prediction, w / w_norm])

    optimum, covariance = curve_fit(
        combined_model,
        np.arange(measured.size),
        measured,
        p0=[100.0, 2.0, 1.0],
        bounds=([80, 1.5, 0.9], [120, 2.5, 1.1]),
    )
    fitted = optimum * [1, 1, 1e9]
    model = iddefix.EvolutionaryAlgorithm(
        impedance_x_data=frequency,
        impedance_y_data=impedance,
        wake_x_data=time,
        wake_y_data=wake,
    )
    model.load_resonator_parameters(
        f"1 | {fitted[0]:.17g} | {fitted[1]:.17g} | {fitted[2]:.17g}"
    )
    expected = np.sqrt(np.diag(covariance)) * [1, 1, 1e9]
    np.testing.assert_allclose(model.get_uncertainties(), expected, rtol=1e-4)
    error = measured - combined_model(None, *optimum)
    assert model.objectiveFunction(fitted) == pytest.approx(error @ error)


# Damping boundaries and loaded parameters
def test_uncertainties_are_finite_at_partial_wake_q_boundary():
    """Uncertainties remain finite for a finite-length wake at critical damping."""
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
    """Loaded parameters outside the original fit bounds yield finite uncertainties."""
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


# Uncertainty calculation options and data requirements
def test_fit_without_automatic_uncertainties():
    """A fit can skip uncertainty work and still compute it on request."""
    frequency = np.linspace(0.7e9, 1.3e9, 81)
    expected = [100.0, 3.0, 1e9]
    impedance = iddefix.Impedances.Resonator_longitudinal_imp(frequency, *expected).real
    model = _model(
        frequency,
        impedance,
        objective="Real",
        parameterEstimates=expected,
        compute_uncertainties=False,
    )

    model.run_minimization_algorithm()

    assert model.minimizationParametersUncertainties is None
    np.testing.assert_allclose(model.minimizationParameters, expected, rtol=0.01)
    assert np.isfinite(model.get_uncertainties()).all()


@pytest.mark.parametrize("complex_data, n_samples", [(False, 3), (True, 1)])
def test_uncertainties_require_residual_degrees_of_freedom(complex_data, n_samples):
    """Real and complex fits need more residuals than fitted parameters."""
    frequency = np.linspace(0.8e9, 1.2e9, n_samples)
    impedance = iddefix.Impedances.Resonator_longitudinal_imp(
        frequency, 100.0, 3.0, 1e9
    )
    model = _model(frequency, impedance if complex_data else impedance.real)
    model.load_resonator_parameters("1 | 100 | 3 | 1e9")

    with pytest.raises(ValueError):
        model.get_uncertainties()


# Impedance uncertainty envelopes
@pytest.mark.parametrize("vary", ["R", "Q", "both"])
def test_impedance_envelope_contains_nominal_with_large_errors(vary):
    """Large parameter errors yield finite envelopes containing the nominal impedance."""
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
    """Finite-wake impedance envelopes contain the nominal spectrum."""
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

"""Regression tests for impedance envelopes from fitted uncertainties."""

import numpy as np

import iddefix


def _model(wake_length=None):
    frequency = np.linspace(0.5e9, 1.5e9, 101)
    model = iddefix.EvolutionaryAlgorithm(
        x_data=frequency,
        y_data=np.zeros_like(frequency),
        N_resonators=1,
        parameterBounds=[(50.0, 150.0), (0.5, 20.0), (0.9e9, 1.1e9)],
        wake_length=wake_length,
    )
    model.evolutionParameters = np.array([300.0, 10.0, 1e9])
    model.evolutionParametersUncertainties = np.array([1000.0, 1000.0, 0.1e9])
    return model, frequency


def test_impedance_envelope_caps_uncertainty_and_contains_nominal():
    model, frequency = _model()
    nominal = model.get_impedance(frequency).real

    for vary in ("R", "Q", "both"):
        lower, upper = model.get_impedance_uncertainty(frequency, vary=vary)
        assert np.isfinite(lower).all()
        assert np.isfinite(upper).all()
        assert np.all(lower <= nominal + 1e-12)
        assert np.all(nominal <= upper + 1e-12)

    peak = np.argmin(np.abs(frequency - 1e9))
    r_lower, r_upper = model.get_impedance_uncertainty(frequency, vary="R")
    np.testing.assert_allclose([r_lower[peak], r_upper[peak]], [150.0, 600.0])


def test_partial_decay_envelope_contains_nominal():
    model, frequency = _model(wake_length=30.0)
    lower, upper = model.get_impedance_uncertainty(frequency, wake_length=30.0)
    nominal = model.get_impedance(frequency, wake_length=30.0).real

    assert np.all(lower <= nominal + 1e-12)
    assert np.all(nominal <= upper + 1e-12)

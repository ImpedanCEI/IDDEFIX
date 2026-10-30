"""Fit workflows: objective selection, solver options, and invalid predictions.

These tests use a small synthetic impedance so failures point to the fitting
path. PyFDE option forwarding is checked separately because it is optional.
"""

import sys
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
import pytest

import iddefix
import iddefix.solvers as solver_module
from iddefix.solvers import Solvers


@pytest.fixture
def resonance():
    frequency = np.linspace(0.8e9, 1.2e9, 31)
    parameters = np.array([100.0, 2.0, 1e9])
    impedance = iddefix.Impedances.Resonator_longitudinal_imp(frequency, *parameters)
    bounds = [(80.0, 120.0), (1.5, 2.5), (0.9e9, 1.1e9)]
    return frequency, parameters, impedance, bounds


@pytest.mark.parametrize("fit_function", ["impedance", "wake"])
def test_large_input_preserves_samples(fit_function):
    """Oversized inputs remain intact so callers control any downsampling."""
    x_data = np.linspace(1.0, 20.0, 2001)
    y_data = np.ones_like(x_data)

    model = iddefix.EvolutionaryAlgorithm(x_data, y_data, fitFunction=fit_function)

    assert model.x_data is x_data
    assert model.y_data is y_data


@pytest.mark.parametrize(
    ("fit_function", "x_alias", "y_alias"),
    [
        ("impedance", "frequency_data", "impedance_data"),
        ("wake", "time_data", "wake_data"),
        ("wake potential", "time_data", "wake_potential_data"),
    ],
)
def test_downsample_keeps_paired_samples(fit_function, x_alias, y_alias):
    """Fitting and domain-specific data use the same regularly spaced subset."""
    x_data = np.arange(11.0)
    y_data = x_data + 1j * x_data
    model = iddefix.EvolutionaryAlgorithm(x_data, y_data, fitFunction=fit_function)

    model.downsample_data(3)

    np.testing.assert_array_equal(model.x_data, x_data[::3])
    np.testing.assert_array_equal(model.y_data, y_data[::3])
    assert getattr(model, x_alias) is model.x_data
    assert getattr(model, y_alias) is model.y_data


@pytest.mark.parametrize("factor", [0, -1, 1.5, True, 11])
def test_downsample_rejects_invalid_factor(factor):
    """Invalid factors leave the stored input untouched."""
    x_data = np.arange(10.0)
    model = iddefix.EvolutionaryAlgorithm(x_data, x_data)

    with pytest.raises(ValueError, match="factor"):
        model.downsample_data(factor)

    assert model.x_data is x_data
    assert model.y_data is x_data


@pytest.mark.parametrize("complex_data", [False, True])
def test_downsample_preview_shows_retained_traces(monkeypatch, complex_data):
    """The comparison uses one panel per real or imaginary data component."""
    x_data = np.linspace(0.5e9, 1.5e9, 10)
    y_data = np.arange(10, dtype=float)
    if complex_data:
        y_data = y_data + 1j * np.arange(10, 20)
    model = iddefix.EvolutionaryAlgorithm(x_data, y_data)
    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))

    model.downsample_data(3, show=True)

    fig = figures[0]
    assert len(fig.axes) == (2 if complex_data else 1)
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_xdata(), x_data)
    np.testing.assert_array_equal(fig.axes[0].lines[1].get_xdata(), x_data[::3])
    np.testing.assert_array_equal(fig.axes[0].lines[1].get_ydata(), y_data.real[::3])
    if complex_data:
        np.testing.assert_array_equal(
            fig.axes[1].lines[1].get_ydata(), y_data.imag[::3]
        )
    plt.close(fig)


@pytest.mark.parametrize("plane", ["longitudinal", "transverse"])
def test_estimate_bounds_from_impedance(plane, capsys, monkeypatch):
    """Detected bounds and estimates prepare an impedance model for fitting."""
    frequency = np.linspace(0.5e9, 1.5e9, 2001)
    parameters = [100.0, 8.0, 1e9]
    formula = (
        iddefix.Impedances.Resonator_longitudinal_imp
        if plane == "longitudinal"
        else iddefix.Impedances.Resonator_transverse_imp
    )
    impedance = formula(frequency, *parameters)
    model = iddefix.EvolutionaryAlgorithm(frequency, impedance, plane=plane)
    capsys.readouterr()
    inspections = []
    monkeypatch.setattr(
        iddefix.SmartBoundDetermination,
        "inspect",
        lambda self, **options: inspections.append((self, options)),
    )

    bounds = model.estimate_bounds(minimum_peak_height=50.0)

    assert "fres [Hz]" in capsys.readouterr().out
    assert inspections == [(bounds, {"show_components": True})]
    assert isinstance(bounds, iddefix.SmartBoundDetermination)
    assert bounds.impedance_type == "real"
    np.testing.assert_allclose(bounds.analysis_impedance_data, impedance.real)
    assert model.N_resonators == 1
    assert model.parameterBounds == bounds.parameterBounds
    assert model.parameterEstimates == bounds.parameterEstimates
    np.testing.assert_allclose(model.parameterEstimates, parameters, rtol=0.02)
    assert len(model._require_fit_bounds()) == 3

    model.estimate_bounds(
        minimum_peak_height=50.0, show_table=False, show_inspect=False
    )
    assert capsys.readouterr().out == ""
    assert inspections == [(bounds, {"show_components": True})]


def test_estimate_bounds_from_magnitude():
    """Longitudinal bounds can still use the impedance magnitude explicitly."""
    frequency = np.linspace(0.5e9, 1.5e9, 2001)
    impedance = iddefix.Impedances.Resonator_longitudinal_imp(
        frequency, 100.0, 8.0, 1e9
    )
    model = iddefix.EvolutionaryAlgorithm(frequency, impedance)

    bounds = model.estimate_bounds(
        impedance_type="absolute", show_table=False, show_inspect=False
    )

    assert bounds.impedance_type == "absolute"
    np.testing.assert_allclose(bounds.analysis_impedance_data, np.abs(impedance))
    assert model.N_resonators == 1


def test_estimate_bounds_requires_detected_impedance_peak():
    """An empty detection leaves any existing fitting setup untouched."""
    frequency = np.linspace(0.5e9, 1.5e9, 101)
    original = [(1.0, 2.0), (1.0, 10.0), (0.5e9, 1.5e9)]
    model = iddefix.EvolutionaryAlgorithm(
        frequency, np.zeros_like(frequency), parameterBounds=original
    )

    with pytest.raises(ValueError, match="No resonance peaks"):
        model.estimate_bounds()

    assert model.N_resonators == 1
    assert model.parameterBounds is original


def test_estimate_bounds_rejects_wake_data():
    """Peak detection needs a frequency-domain impedance trace."""
    time = np.linspace(0.0, 1e-9, 101)
    model = iddefix.EvolutionaryAlgorithm(time, np.zeros_like(time), fitFunction="wake")

    with pytest.raises(ValueError, match="requires impedance"):
        model.estimate_bounds()


@pytest.mark.parametrize("objective", ["Complex", "Real", "Abs"])
def test_named_objective_matches_its_impedance_residual(resonance, objective):
    frequency, parameters, impedance, bounds = resonance
    target = impedance + 3j
    model = iddefix.EvolutionaryAlgorithm(
        frequency,
        target,
        N_resonators=1,
        parameterBounds=bounds,
        objectiveFunction=objective,
    )

    loss = model.objectiveFunction(parameters, model.fitFunction, frequency, target)
    if objective == "Real":
        assert loss == pytest.approx(0)
    elif objective == "Complex":
        assert loss == pytest.approx(9 * frequency.size)
    else:
        assert loss == pytest.approx(np.sum((np.abs(target) - np.abs(impedance)) ** 2))


@pytest.mark.parametrize("solver", ["de", "cmaes"])
def test_solver_fits_complex_impedance_with_public_options(resonance, solver, capsys):
    frequency, parameters, target, bounds = resonance
    initial = np.array([bounds[0][0], bounds[1][1], bounds[2][0]])
    model = iddefix.EvolutionaryAlgorithm(
        frequency,
        target,
        N_resonators=1,
        parameterBounds=bounds,
        parameterEstimates=initial,
        objectiveFunction="Complex",
    )
    initial_loss = model.objectiveFunction(
        initial, model.fitFunction, frequency, target
    )

    if solver == "de":
        model.run_differential_evolution(
            maxiter=20,
            popsize=5,
            workers=1,
            seed=7,
            strategy="best1bin",
            atol=1e-8,
        )
    else:
        model.run_cmaes(
            maxiter=20,
            popsize=8,
            seed=7,
            restarts=0,
            tolfun=1e-9,
        )
    capsys.readouterr()

    fitted = model.get_impedance(frequency, use_minimization=False)
    fitted_loss = np.sum(np.abs(fitted - target) ** 2)
    assert np.isfinite(model.evolutionParameters).all()
    assert fitted_loss < initial_loss


@pytest.mark.parametrize(
    "objective",
    [
        iddefix.ObjectiveFunctions.sumOfSquaredError,
        iddefix.ObjectiveFunctions.sumOfSquaredErrorReal,
        iddefix.ObjectiveFunctions.sumOfSquaredErrorAbs,
        iddefix.ObjectiveFunctions.logsumOfSquaredError,
        iddefix.ObjectiveFunctions.logsumOfSquaredErrorReal,
        iddefix.ObjectiveFunctions.logsumOfSquaredErrorAbs,
    ],
)
@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_invalid_model_values_cannot_appear_to_fit(objective, invalid_value):
    frequency = np.linspace(0.1e9, 2e9, 20)

    def invalid_impedance(x, parameters):
        return np.full(x.shape, invalid_value, dtype=complex)

    # An all-NaN prediction must not turn into zero error during reduction.
    with np.errstate(invalid="ignore"):
        loss = objective(
            [1.0, 0.5, 1e9],
            invalid_impedance,
            frequency,
            np.ones_like(frequency),
        )
    assert loss == float("inf")


@pytest.mark.parametrize("solver_name", ["run_pyfde_solver", "run_pyfde_jade_solver"])
def test_optional_pyfde_receives_seed(monkeypatch, solver_name):
    received = {}
    pyfde = ModuleType("pyfde")

    class FakeDE:
        def __init__(self, function, **options):
            received.update(options)

        def run(self, n_it):
            return np.array([1.0]), 0.0

    pyfde.ClassicDE = FakeDE
    pyfde.JADE = FakeDE
    monkeypatch.setitem(sys.modules, "pyfde", pyfde)
    monkeypatch.setattr(solver_module, "stop_criterion", lambda solver: 0.0)

    getattr(Solvers, solver_name)(
        [(0.0, 2.0)],
        lambda parameters: parameters[0] ** 2,
        maxiter=1,
        seed=7,
    )
    assert received["seed"] == 7

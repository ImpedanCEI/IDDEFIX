"""Fit workflows: objective selection, solver options, and invalid predictions.

These tests use a small synthetic impedance so failures point to the fitting
path. PyFDE option forwarding is checked separately because it is optional.
"""

import sys
from types import ModuleType

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
    model = iddefix.EvolutionaryAlgorithm(
        frequency,
        target,
        N_resonators=1,
        parameterBounds=bounds,
        objectiveFunction="Complex",
    )
    initial = np.array([bounds[0][0], bounds[1][1], bounds[2][0]])
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

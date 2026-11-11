"""Fit workflows: objective selection, solver options, and invalid predictions.

These tests use a small synthetic impedance so failures point to the fitting
path. PyFDE option forwarding is checked separately because it is optional.
"""

import sys
from types import ModuleType, SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.constants import c

import iddefix
import iddefix.solvers as solver_module
from iddefix.baseline import linear_baseline_impedance
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
def test_estimate_bounds_from_impedance(plane):
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

    bounds = model.estimate_bounds(
        minimum_peak_height=50.0, show_table=False, show_inspect=False
    )

    assert model.smart_bounds is bounds
    assert bounds.impedance_type == "real"
    assert model.N_resonators == 1
    assert model.parameterBounds == bounds.parameterBounds
    np.testing.assert_allclose(model.parameterEstimates, parameters, rtol=0.02)


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
    assert model.N_resonators == 1
    np.testing.assert_allclose(model.parameterEstimates, [100.0, 8.0, 1e9], rtol=0.02)


def test_add_peak_after_estimate_bounds():
    """A missed peak updates EA bounds and invalidates earlier fit results."""
    frequency = np.linspace(0.5e9, 2.5e9, 4001)
    impedance = sum(
        iddefix.Impedances.Resonator_longitudinal_imp(frequency, Rs, 8.0, fr)
        for Rs, fr in ((100.0, 1e9), (20.0, 2e9))
    )
    model = iddefix.EvolutionaryAlgorithm(frequency, impedance)

    with pytest.raises(RuntimeError, match="estimate_bounds"):
        model.add_peak(2e9)

    bounds = model.estimate_bounds(
        minimum_peak_height=50.0, show_table=False, show_inspect=False
    )
    assert model.N_resonators == 1
    model.evolutionParameters = np.array([100.0, 8.0, 1e9])

    model.add_peak(2e9, Q=8.0)

    assert model.N_resonators == bounds.N_resonators == 2
    assert model.parameterBounds == bounds.parameterBounds
    np.testing.assert_allclose(model.parameterEstimates, bounds.parameterEstimates)
    assert len(model.parameterBounds) == 6
    assert model.evolutionParameters is None
    np.testing.assert_allclose(model.parameterEstimates[5], 2e9, rtol=0.001)


def test_apply_bounds_from_smartbounds():
    """A separately created estimator supplies bounds and supports peak edits."""
    frequency = np.linspace(0.5e9, 2.5e9, 4001)
    impedance = sum(
        iddefix.Impedances.Resonator_longitudinal_imp(frequency, Rs, 8.0, fr)
        for Rs, fr in ((100.0, 1e9), (20.0, 2e9))
    )
    bounds = iddefix.SmartBoundDetermination(
        frequency, impedance, minimum_peak_height=50.0
    )
    model = iddefix.EvolutionaryAlgorithm(frequency, impedance)
    model.evolutionParameters = np.array([100.0, 8.0, 1e9])
    assert model.smart_bounds is None

    model.apply_bounds(bounds)

    assert model.smart_bounds is bounds
    assert model.N_resonators == 1
    assert model.parameterBounds == bounds.parameterBounds
    np.testing.assert_allclose(model.parameterEstimates, bounds.parameterEstimates)
    assert model.evolutionParameters is None

    model.add_peak(2e9, Q=8.0)
    assert model.N_resonators == bounds.N_resonators == 2

    transverse = iddefix.EvolutionaryAlgorithm(frequency, impedance, plane="transverse")
    with pytest.raises(ValueError, match="plane"):
        transverse.apply_bounds(bounds)


def test_estimate_bounds_hann_from_complex_magnitude():
    """Finite-wake magnitude detection returns a usable resonance estimate."""
    frequency = np.linspace(0, 2e9, 5001)
    impedance = iddefix.Impedances.Resonator_longitudinal_imp(
        frequency, 100.0, 5.0, 0.8e9, wake_length=0.5
    )
    model = iddefix.EvolutionaryAlgorithm(frequency, impedance, wake_length=0.5)

    bounds = model.estimate_bounds(
        impedance_type="absolute", show_table=False, show_inspect=False
    )

    assert bounds.impedance_type == "absolute"
    assert bounds.N_resonators == 1
    assert model.parameterEstimates[2] == pytest.approx(0.8e9, rel=0.1)


def test_estimate_bounds_uses_hann_for_finite_wake():
    """EA passes complex impedance to Hann detection and retains raw estimates."""
    frequency = np.linspace(0.98e9, 1.02e9, 20001)
    impedance = iddefix.Impedances.Resonator_longitudinal_imp(
        frequency, 100.0, 5000.0, 1e9, wake_length=300.0
    )
    model = iddefix.EvolutionaryAlgorithm(frequency, impedance, wake_length=300.0)

    bounds = model.estimate_bounds(
        minimum_peak_height=1.0,
        prominence=0.5,
        show_table=False,
        show_inspect=False,
    )

    assert bounds.N_resonators == 1
    np.testing.assert_allclose(model.parameterEstimates, bounds.parameterEstimates)


def test_estimate_bounds_removes_linear_baseline():
    """EA corrects the stored impedance once before estimating bounds."""
    frequency = np.linspace(0, 2e9, 2001)
    wake_length = 1.0
    slope = 1e14
    clean = iddefix.Impedances.Resonator_longitudinal_imp(
        frequency, 100.0, 2.0, 0.8e9, wake_length=wake_length
    )
    baseline = linear_baseline_impedance(frequency, slope, wake_length)
    contaminated = clean + baseline
    model = iddefix.EvolutionaryAlgorithm(
        frequency, contaminated, wake_length=wake_length
    )

    bounds = model.estimate_bounds(
        remove_baseline=True, show_table=False, show_inspect=False
    )

    assert bounds.baseline_slope == pytest.approx(slope, rel=1e-4)
    assert bounds.N_resonators == 1
    np.testing.assert_allclose(model.y_data, clean, atol=5)
    corrected = model.y_data.copy()
    model.estimate_bounds(remove_baseline=True, show_table=False, show_inspect=False)
    np.testing.assert_array_equal(model.y_data, corrected)


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


def test_estimate_bounds_rejects_peakless_wake_data():
    """A wake spectrum without peaks cannot supply resonator bounds."""
    time = np.linspace(0.0, 1e-9, 101)
    model = iddefix.EvolutionaryAlgorithm(time, np.zeros_like(time), fitFunction="wake")

    with pytest.raises(ValueError, match="No resonance peaks"):
        model.estimate_bounds()


@pytest.mark.parametrize(
    ("fit_function", "plane"),
    [("wake", "transverse"), ("wake potential", "longitudinal")],
)
def test_estimate_bounds_from_corrected_wake(fit_function, plane):
    """Corrected time-domain fitting data produce impedance bounds."""
    sigma = 0.1e-9
    time = (
        np.linspace(-0.5e-9, 12e-9, 5001)
        if fit_function == "wake potential"
        else np.linspace(0, 12e-9, 4801)
    )
    resistance = 100.0 if plane == "longitudinal" else 4e6
    formula = getattr(
        iddefix.Wakes, f"Resonator_{plane}_{fit_function.replace(' ', '_')}"
    )
    clean = (
        formula(time, resistance, 5.0, 0.8e9, sigma=sigma)
        if fit_function == "wake potential"
        else formula(time, resistance, 5.0, 0.8e9)
    )
    slope = 2e19
    contaminated = clean + slope * np.maximum(time, 0)
    model = iddefix.EvolutionaryAlgorithm(
        time,
        contaminated,
        fitFunction=fit_function,
        plane=plane,
        sigma=sigma,
        wake_length=time[-1] * c,
    )

    bounds = model.estimate_bounds(
        remove_baseline=slope,
        fmax=2e9,
        show_table=False,
        show_inspect=False,
    )

    np.testing.assert_allclose(model.y_data, clean, atol=2e-5)
    assert model.baseline_slope == bounds.baseline_slope == slope
    assert bounds.N_resonators == 1
    assert bounds.parameterEstimates[0] == pytest.approx(resistance, rel=0.02)


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
def test_solver_fits_complex_impedance_with_public_options(resonance, solver):
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
    fitted = model.get_impedance(frequency, use_minimization=False)
    fitted_loss = np.sum(np.abs(fitted - target) ** 2)
    assert np.isfinite(model.evolutionParameters).all()
    assert fitted_loss < initial_loss


@pytest.mark.parametrize("maxiter", [None, 10])
@pytest.mark.parametrize(("restarts", "expected_runs"), [(None, 1), (2, 3)])
def test_cmaes_progress(monkeypatch, maxiter, restarts, expected_runs):
    """Finite limits show a budget; convergence-led runs count iterations."""
    import pymoo.algorithms.soo.nonconvex.cmaes as cmaes_module
    import pymoo.optimize as pymoo_optimize

    observed = {}

    class FakeCMAES:
        def __init__(self, **options):
            observed["restarts"] = options["restarts"]
            observed["maxiter"] = options["maxiter"]

    def fake_minimize(problem, solver, **options):
        callback = options["callback"]
        observed["progress_total"] = callback.pbar.total
        for _ in range(expected_runs):
            callback(SimpleNamespace(pop=SimpleNamespace(get=lambda _: [0.0])))
            for _ in range(10):
                callback(SimpleNamespace(pop=SimpleNamespace(get=lambda _: [0.0, 1.0])))
        callback(SimpleNamespace(pop=SimpleNamespace(get=lambda _: [0.0])))
        observed["iterations"] = callback.pbar.n
        return SimpleNamespace(X=np.array([0.5]), algorithm=SimpleNamespace(n_gen=1))

    monkeypatch.setattr(cmaes_module, "CMAES", FakeCMAES)
    monkeypatch.setattr(pymoo_optimize, "minimize", fake_minimize)
    options = {} if restarts is None else {"restarts": restarts}

    Solvers.run_pymoo_cmaes_solver(
        [(0.0, 1.0)], lambda x: x[0] ** 2, maxiter=maxiter, **options
    )

    assert observed["restarts"] == expected_runs - 1
    if maxiter is None:
        assert np.isinf(observed["maxiter"])
        assert observed["progress_total"] is None
    else:
        assert observed["maxiter"] == maxiter
        assert observed["progress_total"] == maxiter * expected_runs
    assert observed["iterations"] == 10 * expected_runs


def test_cmaes_without_iteration_limit(resonance):
    """The public default lets CMA-ES stop at another configured criterion."""
    frequency, parameters, target, bounds = resonance
    model = iddefix.EvolutionaryAlgorithm(
        frequency,
        target,
        N_resonators=1,
        parameterBounds=bounds,
        parameterEstimates=parameters,
        compute_uncertainties=False,
    )

    result = model.run_cmaes(popsize=8, maxfevals=30)

    assert np.isfinite(model.evolutionParameters).all()
    assert result.algorithm.n_gen >= 1


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

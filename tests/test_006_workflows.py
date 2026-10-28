"""Fit workflows: objective selection, solver options, and invalid predictions.

These tests use a small synthetic impedance so failures point to the fitting
path. Optional backends are exercised through the same public fitting workflow.
"""

import numpy as np
import pytest

import iddefix


@pytest.fixture
def resonance():
    frequency = np.linspace(0.8e9, 1.2e9, 31)
    parameters = np.array([100.0, 2.0, 1e9])
    impedance = iddefix.Impedances.Resonator_longitudinal_imp(frequency, *parameters)
    bounds = [(80.0, 120.0), (1.5, 2.5), (0.9e9, 1.1e9)]
    return frequency, parameters, impedance, bounds


# Automatic bounds estimation
@pytest.mark.parametrize("plane", ["longitudinal", "transverse"])
def test_estimate_bounds_from_impedance(plane):
    """Automatic estimates initialize a fit that reproduces the measured spectrum."""
    frequency = np.linspace(0.5e9, 1.5e9, 2001)
    parameters = [100.0, 8.0, 1e9]
    formula = (
        iddefix.Impedances.Resonator_longitudinal_imp
        if plane == "longitudinal"
        else iddefix.Impedances.Resonator_transverse_imp
    )
    impedance = formula(frequency, *parameters)
    model = iddefix.EvolutionaryAlgorithm(
        impedance_x_data=frequency, impedance_y_data=impedance, plane=plane
    )
    model.estimate_bounds(
        minimum_peak_height=50.0, show_table=False, show_inspect=False
    )
    model.run_minimization_algorithm()

    np.testing.assert_allclose(model.minimizationParameters, parameters, rtol=0.02)
    np.testing.assert_allclose(model.get_impedance(), impedance, rtol=0.02)


def test_estimate_bounds_from_magnitude():
    """Longitudinal bounds can still use the impedance magnitude explicitly."""
    frequency = np.linspace(0.5e9, 1.5e9, 2001)
    impedance = iddefix.Impedances.Resonator_longitudinal_imp(
        frequency, 100.0, 8.0, 1e9
    )
    model = iddefix.EvolutionaryAlgorithm(
        impedance_x_data=frequency, impedance_y_data=impedance
    )

    bounds = model.estimate_bounds(
        impedance_type="absolute", show_table=False, show_inspect=False
    )

    np.testing.assert_allclose(bounds.parameterEstimates, [100.0, 8.0, 1e9], rtol=0.05)
    assert model.N_resonators == 1


def test_estimate_bounds_requires_detected_impedance_peak():
    """An empty detection leaves any existing fitting setup untouched."""
    frequency = np.linspace(0.5e9, 1.5e9, 101)
    original = [(1.0, 2.0), (1.0, 10.0), (0.5e9, 1.5e9)]
    model = iddefix.EvolutionaryAlgorithm(
        impedance_x_data=frequency,
        impedance_y_data=np.zeros_like(frequency),
        parameterBounds=original,
    )

    with pytest.raises(ValueError):
        model.estimate_bounds()

    assert model.N_resonators == 1
    assert model.parameterBounds == original


def test_estimate_bounds_rejects_wake_data():
    """Peak detection needs a frequency-domain impedance trace."""
    time = np.linspace(0.0, 1e-9, 101)
    model = iddefix.EvolutionaryAlgorithm(
        wake_x_data=time, wake_y_data=np.zeros_like(time), fitFunction="wake"
    )

    with pytest.raises(ValueError):
        model.estimate_bounds()


# Single-source and custom objectives
@pytest.mark.parametrize("objective", ["Complex", "Real", "Abs"])
def test_named_objective_matches_its_impedance_residual(resonance, objective):
    """Named objectives measure real, complex, or magnitude errors as requested."""
    frequency, parameters, impedance, bounds = resonance
    target = impedance + 3j
    model = iddefix.EvolutionaryAlgorithm(
        impedance_x_data=frequency,
        impedance_y_data=target,
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


@pytest.mark.parametrize("source", ["impedance", "wake_potential", "wake"])
def test_single_source_objective(source):
    """Explicit source pairs retain the available single-source objectives."""
    parameters = [100.0, 2.0, 1e9]
    if source == "impedance":
        x = np.linspace(0.5e9, 1.5e9, 31)
        y = iddefix.Impedances.Resonator_longitudinal_imp(x, *parameters)
    else:
        x = np.linspace(0, 2e-9, 31)
        function = (
            iddefix.Wakes.Resonator_longitudinal_wake
            if source == "wake"
            else iddefix.Wakes.Resonator_longitudinal_wake_potential
        )
        options = {} if source == "wake" else {"sigma": 50e-12}
        y = function(x, *parameters, **options)
    name = source.replace(" ", "_")
    model = iddefix.EvolutionaryAlgorithm(
        **{name + "_x_data": x, name + "_y_data": y + 1},
        objectiveFunction="Complex" if np.iscomplex(y).any() else "Real",
        sigma=50e-12,
    )
    loss = model.objectiveFunction(parameters, model.fitFunction, x, y + 1)
    assert loss == pytest.approx(x.size)


def test_custom_objective(resonance):
    """A supplied objective measures the requested error on model predictions."""
    frequency, parameters, impedance, _ = resonance

    def objective(parameters, fitFunction, x, y):
        prediction = fitFunction(x, iddefix.utils.pars_to_dict(parameters))
        return float(np.mean(np.abs(prediction - y)))

    model = iddefix.EvolutionaryAlgorithm(
        impedance_x_data=frequency,
        impedance_y_data=impedance,
        objectiveFunction=objective,
    )
    assert model.objectiveFunction(
        parameters * [1.1, 1, 1], model.fitFunction, frequency, impedance
    ) == pytest.approx(np.mean(np.abs(0.1 * impedance)))


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
    """Nonfinite model predictions cannot produce a finite fitting loss."""
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


# Optimization workflows
@pytest.mark.parametrize("solver", ["de", "cmaes"])
def test_solver_fits_complex_impedance_with_public_options(resonance, solver):
    """Public solver options produce a fit that reduces the complex impedance error."""
    frequency, parameters, target, bounds = resonance
    initial = np.array([bounds[0][0], bounds[1][1], bounds[2][0]])
    model = iddefix.EvolutionaryAlgorithm(
        impedance_x_data=frequency,
        impedance_y_data=target,
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


@pytest.mark.parametrize("solver", ["pyfde", "pyfde_jade"])
def test_optional_de_fit(resonance, solver):
    """Optional DE backends recover the resonator through the public fit method."""
    pytest.importorskip("pyfde")
    frequency, parameters, impedance, bounds = resonance
    model = iddefix.EvolutionaryAlgorithm(
        impedance_x_data=frequency,
        impedance_y_data=impedance,
        parameterBounds=bounds,
        compute_uncertainties=False,
    )
    model.run_differential_evolution(
        solver=solver,
        maxiter=200,
        popsize=10,
        tol=1e-8,
        seed=7,
    )
    np.testing.assert_allclose(model.get_impedance(), impedance, rtol=0.02)
    np.testing.assert_allclose(model.evolutionParameters, parameters, rtol=0.02)


@pytest.mark.parametrize("solver", ["local", "de", "cmaes"])
def test_joint_solver_and_uncertainties(resonance, solver):
    """All public optimization paths recover a common synthetic resonator."""
    frequency, parameters, impedance, bounds = resonance
    time = np.linspace(-0.2e-9, 2e-9, 61)
    potential = iddefix.Wakes.Resonator_longitudinal_wake_potential(
        time, *parameters, sigma=50e-12
    )
    model = iddefix.EvolutionaryAlgorithm(
        impedance_x_data=frequency,
        impedance_y_data=impedance,
        wake_potential_x_data=time,
        wake_potential_y_data=potential,
        sigma=50e-12,
        parameterBounds=bounds,
        parameterEstimates=parameters * [1.02, 0.98, 1.01],
    )
    if solver == "local":
        model.run_minimization_algorithm()
        fitted = model.minimizationParameters
        uncertainties = model.minimizationParametersUncertainties
    elif solver == "de":
        model.run_differential_evolution(maxiter=250, popsize=8, workers=1, seed=42)
        fitted = model.evolutionParameters
        uncertainties = model.evolutionParametersUncertainties
    else:
        pytest.importorskip("pymoo")
        model.run_cmaes(maxiter=200, popsize=12, restarts=0, seed=42)
        fitted = model.evolutionParameters
        uncertainties = model.evolutionParametersUncertainties
    np.testing.assert_allclose(fitted, parameters, rtol=0.002)
    assert np.all(np.isfinite(uncertainties))
    np.testing.assert_allclose(
        model.get_impedance(),
        impedance,
        rtol=0.02,
        atol=np.max(np.abs(impedance)) * 1e-3,
    )
    np.testing.assert_allclose(
        model.get_wake_potential(),
        potential,
        rtol=0.02,
        atol=np.max(np.abs(potential)) * 1e-3,
    )


# Shared resonator models and source grids
@pytest.mark.parametrize("plane", ["longitudinal", "transverse"])
@pytest.mark.parametrize("quality", [0.3, 0.5, 2.0])
@pytest.mark.parametrize(
    "sources",
    [
        ("impedance",),
        ("wake",),
        ("wake_potential",),
        ("impedance", "wake"),
        ("impedance", "wake_potential"),
        ("wake", "wake_potential"),
        ("impedance", "wake", "wake_potential"),
    ],
)
def test_named_sources_share_parameters(plane, quality, sources):
    """Every available subset uses the corrected formulas and independent grids."""
    parameters = np.array([100.0, quality, 1e9])
    sigma = 50e-12
    data = {}
    expected = {}
    for index, name in enumerate(sources):
        x = (
            np.linspace(0.1e9, 2e9, 41)
            if name == "impedance"
            else np.linspace(-0.2e-9, 2e-9, 51 + index)
        )
        module = iddefix.Impedances if name == "impedance" else iddefix.Wakes
        suffix = "imp" if name == "impedance" else name
        function = getattr(module, f"Resonator_{plane}_{suffix}")
        options = {"sigma": sigma} if name == "wake_potential" else {}
        y = function(x, *parameters, **options)
        data[name + "_x_data"] = x
        data[name + "_y_data"] = y
        expected[name] = y
    model = iddefix.EvolutionaryAlgorithm(plane=plane, sigma=sigma, **data)
    assert model.objectiveFunction(parameters) == pytest.approx(0, abs=1e-24)
    assert model.objectiveFunction(parameters * [1.1, 1, 1]) > 0
    model.load_resonator_parameters(
        f"1 | {parameters[0]:.17g} | {parameters[1]:.17g} | {parameters[2]:.17g}"
    )
    for name, y in expected.items():
        np.testing.assert_allclose(getattr(model, "get_" + name)(), y)
    np.testing.assert_allclose(model.get_uncertainties(), 0, atol=1e-20)


def test_joint_multiple_resonators():
    """A common flattened vector drives every source in a multi-resonator fit."""
    parameters = np.array([100, 2, 1e9, 50, 0.3, 2e9])
    frequency = np.linspace(0.1e9, 3e9, 61)
    time = np.linspace(-0.2e-9, 3e-9, 81)
    impedance = iddefix.Impedances.n_Resonator_transverse_imp(frequency, parameters)
    wake = iddefix.Wakes.n_Resonator_transverse_wake(time, parameters)
    potential = iddefix.Wakes.n_Resonator_transverse_wake_potential(
        time, parameters, sigma=50e-12
    )
    model = iddefix.EvolutionaryAlgorithm(
        N_resonators=2,
        plane="transverse",
        sigma=50e-12,
        impedance_x_data=frequency,
        impedance_y_data=impedance,
        wake_x_data=time,
        wake_y_data=wake,
        wake_potential_x_data=time,
        wake_potential_y_data=potential,
    )
    assert model.objectiveFunction(parameters) == pytest.approx(0, abs=1e-24)
    for index in range(6):
        perturbed = parameters.copy()
        perturbed[index] *= 1.1
        assert model.objectiveFunction(perturbed) > 0


def test_wake_potential_precedes_wake():
    """Without impedance, wake potential supplies the primary grid."""
    potential_time = np.linspace(-0.2e-9, 2e-9, 41)
    wake_time = np.linspace(0, 3e-9, 61)
    parameters = [100.0, 2.0, 1e9]
    potential = iddefix.Wakes.Resonator_longitudinal_wake_potential(
        potential_time, *parameters, sigma=50e-12
    )
    wake = iddefix.Wakes.Resonator_longitudinal_wake(wake_time, *parameters)
    model = iddefix.EvolutionaryAlgorithm(
        wake_x_data=wake_time,
        wake_y_data=wake,
        wake_potential_x_data=potential_time,
        wake_potential_y_data=potential,
        sigma=50e-12,
    )
    model.load_resonator_parameters("1 | 100 | 2 | 1e9")
    np.testing.assert_allclose(model.get_wake_potential(), potential)
    np.testing.assert_allclose(model.get_wake(), wake)
    np.testing.assert_allclose(model.get_resonator_components().sum(axis=0), potential)


def test_data_pair_argument_order():
    """Positional source pairs are impedance, wake potential, then wake."""
    frequency = np.linspace(0.1e9, 2e9, 41)
    potential_time = np.linspace(-0.2e-9, 2e-9, 51)
    wake_time = np.linspace(0, 3e-9, 61)
    parameters = [100.0, 2.0, 1e9]
    impedance = iddefix.Impedances.Resonator_longitudinal_imp(frequency, *parameters)
    potential = iddefix.Wakes.Resonator_longitudinal_wake_potential(
        potential_time, *parameters, sigma=50e-12
    )
    wake = iddefix.Wakes.Resonator_longitudinal_wake(wake_time, *parameters)
    model = iddefix.EvolutionaryAlgorithm(
        frequency,
        impedance,
        potential_time,
        potential,
        wake_time,
        wake,
        N_resonators=1,
        sigma=50e-12,
    )
    assert model.objectiveFunction(parameters) == pytest.approx(0)
    model.load_resonator_parameters("1 | 100 | 2 | 1e9")
    np.testing.assert_allclose(model.get_impedance(), impedance)
    np.testing.assert_allclose(model.get_wake_potential(), potential)
    np.testing.assert_allclose(model.get_wake(), wake)


# Joint loss normalization
def test_joint_normalization(resonance):
    """Scale and sampling density do not change a source's relative influence."""
    frequency, parameters, impedance, _ = resonance
    time = np.linspace(0, 2e-9, 71)
    wake = iddefix.Wakes.Resonator_longitudinal_wake(time, *parameters)
    model = iddefix.EvolutionaryAlgorithm(
        impedance_x_data=frequency,
        impedance_y_data=impedance * 2,
        wake_x_data=time,
        wake_y_data=wake * 2,
    )
    # Both predictions are half their measured trace: normalized MSE = 1/4.
    assert model.objectiveFunction(parameters) == pytest.approx(0.5)
    duplicated = iddefix.EvolutionaryAlgorithm(
        impedance_x_data=np.tile(frequency, 3),
        impedance_y_data=np.tile(impedance * 2, 3),
    )
    assert duplicated.objectiveFunction(parameters) == pytest.approx(0.25)


def test_joint_zero_data(resonance):
    """An all-zero trace yields a finite loss without hiding nonfinite predictions."""
    frequency, parameters, impedance, _ = resonance
    time = np.linspace(0, 2e-9, 71)
    wake = iddefix.Wakes.Resonator_longitudinal_wake(time, *parameters)
    model = iddefix.EvolutionaryAlgorithm(
        impedance_x_data=frequency,
        impedance_y_data=np.zeros(frequency.size),
        wake_x_data=time,
        wake_y_data=np.zeros(time.size),
    )
    expected = np.mean(np.abs(impedance) ** 2) + np.mean(wake**2)
    assert model.objectiveFunction(parameters) == pytest.approx(expected)
    with np.errstate(invalid="ignore"):
        assert not np.isfinite(model.objectiveFunction([np.nan, 2.0, 1e9]))


def test_joint_amplitude_scaling(resonance):
    """Relative fitting errors are unchanged when the physical signal strength scales."""
    frequency, parameters, impedance, _ = resonance
    time = np.linspace(1e-12, 2e-9, 71)
    wake = iddefix.Wakes.Resonator_longitudinal_wake(time, *parameters)
    losses = []
    for strength in (1e-15, 1.0, 1e15):
        model = iddefix.EvolutionaryAlgorithm(
            impedance_x_data=frequency,
            impedance_y_data=impedance * strength,
            wake_x_data=time,
            wake_y_data=wake * strength,
        )
        probe = parameters * [1.1 * strength, 0.9, 1.01]
        losses.append(model.objectiveFunction(probe))
    np.testing.assert_allclose(losses, losses[0], rtol=1e-12)


# Input validation and legacy data filtering
@pytest.mark.parametrize(
    "options",
    [
        {"wake_x_data": [0, 1]},
        {"wake_x_data": [0], "wake_y_data": []},
        {"wake_x_data": [0], "wake_y_data": [np.nan]},
        {"wake_potential_x_data": [0], "wake_potential_y_data": [1]},
        {
            "wake_x_data": [0],
            "wake_y_data": [1],
            "impedance_x_data": [1],
            "impedance_y_data": [1],
            "objectiveFunction": "Real",
        },
    ],
)
def test_named_source_validation(options):
    """Incomplete, invalid, or incompatible measurements are rejected."""
    with pytest.raises(ValueError):
        iddefix.EvolutionaryAlgorithm(**options)


@pytest.mark.parametrize("keyword", ["x_data", "y_data"])
def test_removed_data_keywords(keyword):
    """The constructor accepts explicit source pairs instead of legacy keywords."""
    with pytest.raises(TypeError):
        iddefix.EvolutionaryAlgorithm(**{keyword: np.array([1.0])})


def test_legacy_positive_x_filter():
    """The standalone legacy helper filters matching x and y samples."""
    frequency = np.array([-1e9, 0, 1e9, 2e9])
    impedance = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j])
    model = iddefix.EvolutionaryAlgorithm(
        impedance_x_data=frequency, impedance_y_data=impedance
    )
    model.check_y_data()
    np.testing.assert_array_equal(model.x_data, frequency[2:])
    np.testing.assert_array_equal(model.y_data, impedance[2:])

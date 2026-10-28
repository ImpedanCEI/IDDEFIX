"""Resonator model lifecycle: load, display, edit, and evaluate parameters.

The tests check public model behavior after parameter changes, including the
partial-wake Q boundary and the transverse impedance at critical damping.
"""

import re
from html import unescape
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

import iddefix


def _model(n_resonators=2, parameter_bounds=None, source="impedance", **kwargs):
    x_data = (
        np.linspace(0.5e9, 1.5e9, 61)
        if source == "impedance"
        else np.linspace(0.0, 10e-9, 61)
    )
    if parameter_bounds is None:
        parameter_bounds = [
            (0.0, 1000.0),
            (0.5, 1e5),
            (0.1e9, 2e9),
        ] * n_resonators
    source = source.replace(" ", "_")
    return iddefix.EvolutionaryAlgorithm(
        **{source + "_x_data": x_data, source + "_y_data": np.zeros_like(x_data)},
        N_resonators=n_resonators,
        parameterBounds=parameter_bounds,
        **kwargs,
    )


# Loading parameters
@pytest.mark.parametrize("n_resonators", [None, 2])
def test_load_table_without_count_or_bounds(n_resonators):
    """A loaded table supplies the resonator count without fitting bounds."""
    frequency = np.linspace(0.5e9, 1.5e9, 61)
    model = iddefix.EvolutionaryAlgorithm(
        impedance_x_data=frequency,
        impedance_y_data=np.zeros_like(frequency),
        N_resonators=n_resonators,
        parameterBounds=None,
    )
    model.load_resonator_parameters("1 | 100 | 3 | 1e9\n2 | 200 | 5 | 1.5e9")

    assert model.N_resonators == 2
    assert model.parameterBounds is None
    model.display_resonator_parameters()
    np.testing.assert_allclose(
        model.get_impedance(frequency),
        iddefix.Impedances.n_Resonator_longitudinal_imp(
            frequency, [100, 3, 1e9, 200, 5, 1.5e9]
        ),
    )


def test_empty_table_rejected():
    """Loading a parameter table requires at least one resonator row."""
    frequency = np.linspace(0.5e9, 1.5e9, 61)
    model = iddefix.EvolutionaryAlgorithm(
        impedance_x_data=frequency, impedance_y_data=np.zeros_like(frequency)
    )
    with pytest.raises(ValueError):
        model.load_resonator_parameters("Resonator | Rs | Q | fres")
    assert model.N_resonators is None


def test_infer_count_from_bounds():
    """Existing bounded workflows may omit the redundant resonator count."""
    frequency = np.linspace(0.5e9, 1.5e9, 61)
    model = iddefix.EvolutionaryAlgorithm(
        impedance_x_data=frequency,
        impedance_y_data=np.zeros_like(frequency),
        parameterBounds=[(0, 1000), (0.5, 10), (0.5e9, 1.5e9)],
    )
    assert model.N_resonators == 1


@pytest.mark.parametrize(
    ("escape_start", "escape_end"),
    [
        pytest.param("", "", id="plain"),
        pytest.param("\x1b[31m", "\x1b[0m", id="ansi"),
        pytest.param("␛[31m", "␛[0m", id="visible-ansi"),
    ],
)
def test_load_table(escape_start, escape_end):
    """Load a table with optional ANSI escapes and reproduce the model spectrum."""
    model = _model()
    model.load_resonator_parameters(
        "Resonator | Rs | Q | fres\n"
        f"1 | {escape_start}100{escape_end} | 3 | 1e9\n"
        f"2 | {escape_start}200 ± 20{escape_end} | 5 | 1.5e9"
    )

    expected = [100.0, 3.0, 1e9, 200.0, 5.0, 1.5e9]
    np.testing.assert_allclose(model.minimizationParameters, expected)
    np.testing.assert_allclose(
        model.minimizationParametersUncertainties,
        [0.0, 0.0, 0.0, 20.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        model.get_impedance(model.x_data),
        iddefix.Impedances.n_Resonator_longitudinal_imp(model.x_data, expected),
    )


# Displaying and copying parameter tables
@pytest.mark.parametrize("to_markdown", [False, True])
@pytest.mark.parametrize("display_uncertainties", [False, True])
@pytest.mark.parametrize("display_warnings", [False, True])
def test_display_round_trip(
    capsys, to_markdown, display_uncertainties, display_warnings
):
    """Copied tables preserve physical parameters regardless of presentation options."""
    model = _model()
    parameters = np.array([995.0, 3.0, 1e9, 200.0, 246.63, 1.5e9])
    uncertainties = np.array([30.0, 0.5, 1e7, 20.0, 0.5, 2e7])
    model.display_resonator_parameters(
        params=parameters,
        uncertainties=uncertainties,
        to_markdown=to_markdown,
        display_uncertainties=display_uncertainties,
        display_warnings=display_warnings,
    )

    output = capsys.readouterr().out
    model.load_resonator_parameters(output, use_minimization=False)

    np.testing.assert_allclose(model.evolutionParameters, parameters)
    np.testing.assert_allclose(
        model.evolutionParametersUncertainties,
        uncertainties if display_uncertainties else np.zeros_like(uncertainties),
    )


def test_notebook_table_round_trip(monkeypatch):
    """A table copied from notebook output can be loaded as resonator data."""
    ipython = pytest.importorskip("IPython")
    ipython_display = pytest.importorskip("IPython.display")
    monkeypatch.setattr(
        ipython, "get_ipython", lambda: SimpleNamespace(kernel=object())
    )
    displayed = []
    monkeypatch.setattr(ipython_display, "display", displayed.append)
    model = _model(n_resonators=1)

    model.display_resonator_parameters(
        params=np.array([995.0, 3.0, 1e9]),
        uncertainties=np.array([300.0, 1.0, 1e7]),
        display_warnings=True,
    )

    assert displayed
    copied_text = "\n".join(
        unescape(re.sub(r"<[^>]+>", "", item.data)) for item in displayed
    )
    model.load_resonator_parameters(copied_text)
    np.testing.assert_allclose(model.minimizationParameters, [995.0, 3.0, 1e9])
    np.testing.assert_allclose(
        model.minimizationParametersUncertainties, [300.0, 1.0, 1e7]
    )


# Fitting parameters
def test_missing_bounds_reject_global_fitting():
    """Global fitting requires bounds after parameters are loaded for evaluation."""
    frequency = np.linspace(0.5e9, 1.5e9, 61)
    model = iddefix.EvolutionaryAlgorithm(
        impedance_x_data=frequency, impedance_y_data=np.zeros_like(frequency)
    )
    model.load_resonator_parameters("1 | 100 | 3 | 1e9")

    with pytest.raises(ValueError):
        model.run_differential_evolution()
    with pytest.raises(ValueError):
        model.run_cmaes()


@pytest.mark.parametrize("algorithm", ["differential_evolution", "cmaes"])
def test_global_fit_respects_bounds(algorithm):
    """A stronger measured mode drives the fit to its allowed shunt-resistance limit."""
    if algorithm == "cmaes":
        pytest.importorskip("pymoo")
    frequency = np.linspace(0.8e9, 1.2e9, 41)
    measured = iddefix.Impedances.Resonator_longitudinal_imp(frequency, 150.0, 2.0, 1e9)
    bounds = [(80.0, 100.0), (1.9, 2.1), (0.99e9, 1.01e9)]
    model = iddefix.EvolutionaryAlgorithm(
        impedance_x_data=frequency,
        impedance_y_data=measured,
        parameterBounds=bounds,
        parameterEstimates=[90, 2, 1e9],
        compute_uncertainties=False,
    )
    if algorithm == "differential_evolution":
        model.run_differential_evolution(maxiter=150, popsize=8, workers=1, seed=7)
    else:
        model.run_cmaes(maxiter=150, popsize=8, restarts=0, seed=7)
    fitted = model.evolutionParameters
    assert fitted[0] == pytest.approx(100.0, rel=0.01)
    assert np.all(fitted >= np.asarray(bounds)[:, 0])
    assert np.all(fitted <= np.asarray(bounds)[:, 1])
    assert np.all(np.isfinite(model.get_impedance()))


def test_local_fit_respects_margin():
    """Local refinement cannot exceed the requested resistance margin."""
    frequency = np.linspace(0.8e9, 1.2e9, 41)
    measured = iddefix.Impedances.Resonator_longitudinal_imp(frequency, 150.0, 2.0, 1e9)
    model = iddefix.EvolutionaryAlgorithm(
        impedance_x_data=frequency,
        impedance_y_data=measured,
        N_resonators=1,
        compute_uncertainties=False,
    )
    model.load_resonator_parameters("1 | 100 | 2 | 1e9", use_minimization=False)
    model.run_minimization_algorithm(margin=[0.1, 0.0, 0.0])
    expected = iddefix.Impedances.Resonator_longitudinal_imp(frequency, 110.0, 2.0, 1e9)
    np.testing.assert_allclose(model.get_impedance(), expected, rtol=1e-6)


# Editing resonators
def test_edit_loaded_model_without_bounds():
    """A bounds-free loaded model supports the notebook editing workflow."""
    frequency = np.linspace(0.5e9, 1.5e9, 61)
    model = iddefix.EvolutionaryAlgorithm(
        impedance_x_data=frequency, impedance_y_data=np.zeros_like(frequency)
    )
    model.load_resonator_parameters("1 | 100 | 3 | 1e9\n2 | 200 | 5 | 1.5e9")

    model.remove_resonator(2)
    model.add_resonator(200, 5, 1.5e9)

    assert model.N_resonators == 2
    assert model.parameterBounds is None
    np.testing.assert_allclose(
        model.minimizationParameters, [100, 3, 1e9, 200, 5, 1.5e9]
    )


def test_edit_resonators():
    """Editing, adding, and removing modes updates both fitted stages."""
    model = _model()
    model.load_resonator_parameters(
        "1 | 100 ± 10 | 3 ± 0.3 | 1e9 ± 1e7\n2 | 200 ± 20 | 5 ± 0.5 | 1.5e9 ± 2e7"
    )
    model.evolutionParameters[0] = 90.0  # The two fit stages can differ.

    model.modify_resonator(2, Q=1.2, fr=1.4e9)
    np.testing.assert_allclose(model.evolutionParameters, [90, 3, 1e9, 200, 1.2, 1.4e9])
    np.testing.assert_allclose(
        model.minimizationParameters, [100, 3, 1e9, 200, 1.2, 1.4e9]
    )
    assert model.evolutionParametersUncertainties is None
    assert model.minimizationParametersUncertainties is None

    model.add_resonator(300.0, 7.0, 1.2e9, uncertainty=[30.0, 0.7, 3e7])
    model.remove_resonator(1)

    expected = [200.0, 1.2, 1.4e9, 300.0, 7.0, 1.2e9]
    assert model.N_resonators == 2
    np.testing.assert_allclose(model.minimizationParameters, expected)
    np.testing.assert_allclose(
        model.minimizationParametersUncertainties,
        [0.0, 0.0, 0.0, 30.0, 0.7, 3e7],
    )
    np.testing.assert_allclose(
        model.get_impedance(model.x_data),
        iddefix.Impedances.n_Resonator_longitudinal_imp(model.x_data, expected),
    )


def test_model_after_edits():
    """Manual edits keep the starting vector aligned with the fitted modes."""
    initial = [100.0, 3.0, 1e9, 200.0, 5.0, 1.5e9]
    model = _model(parameterEstimates=initial, objectiveFunction="Real")
    model.load_resonator_parameters("1 | 100 | 3 | 1e9\n2 | 200 | 5 | 1.5e9")

    model.modify_resonator(2, Q=4.0)
    model.add_resonator(
        300.0,
        7.0,
        1.2e9,
        parameter_bounds=[(100.0, 500.0), (1.0, 10.0), (1e9, 1.5e9)],
    )
    model.remove_resonator(1)

    expected = [200.0, 4.0, 1.5e9, 300.0, 7.0, 1.2e9]
    model.y_data = iddefix.Impedances.n_Resonator_longitudinal_imp(
        model.x_data, expected
    ).real
    model.run_minimization_algorithm()
    np.testing.assert_allclose(model.parameterEstimates, expected)
    fitted = model.get_impedance(model.x_data).real
    assert np.max(np.abs(fitted - model.y_data)) / model.y_data.max() < 0.01


def test_edit_q_boundary():
    """Invalid Q leaves the model intact; positive subcritical Q is valid."""
    model = _model(n_resonators=1)
    model.load_resonator_parameters("1 | 100 | 3 | 1e9")
    original = model.minimizationParameters.copy()

    with pytest.raises(ValueError):
        model.modify_resonator(1, Q=0)
    np.testing.assert_array_equal(model.minimizationParameters, original)

    model.modify_resonator(1, Rs=0, Q=0.3)
    np.testing.assert_allclose(model.minimizationParameters, [0, 0.3, 1e9])


# Components and plots
@pytest.mark.parametrize(
    ("fit_function", "plane"),
    [
        ("impedance", "longitudinal"),
        ("impedance", "transverse"),
        ("wake", "longitudinal"),
        ("wake potential", "longitudinal"),
    ],
)
def test_get_resonator_components(fit_function, plane):
    """Resonator components sum to the selected impedance or wake model."""
    model = _model(source=fit_function, plane=plane, sigma=1e-10)
    model.load_resonator_parameters("1 | 100 | 3 | 1e9\n2 | 200 | 5 | 1.5e9")

    components = model.get_resonator_components()
    total_model = getattr(model, "get_" + fit_function.replace(" ", "_"))()

    np.testing.assert_allclose(components.sum(axis=0), total_model)


@pytest.mark.parametrize(
    ("fit_function", "plane", "ylabel"),
    [
        ("wake", "longitudinal", "Wake [V/C]"),
        ("wake potential", "longitudinal", "Wake potential [V/C]"),
        ("wake", "transverse", "Wake [V/C/m]"),
    ],
)
def test_plot_resonator_components(request, monkeypatch, fit_function, plane, ylabel):
    """Component plots label wake amplitudes with the selected plane's units."""
    model = _model(n_resonators=1, source=fit_function, plane=plane, sigma=1e-10)
    model.load_resonator_parameters("1 | 100 | 3 | 1e9")

    if not request.config.getoption("--debug-plots"):
        monkeypatch.setattr(plt, "show", lambda: None)
    fig, axes = model.plot_resonator_components()

    assert axes[0].get_ylabel() == ylabel
    plt.close(fig)


@pytest.mark.parametrize(
    ("n_resonators", "fit_function"),
    [(9, "impedance"), (10, "impedance"), (10, "wake")],
)
def test_plot_many_components(request, monkeypatch, n_resonators, fit_function):
    """Component plots run on both sides of the colorbar threshold."""
    model = _model(n_resonators=n_resonators, source=fit_function)
    model.load_resonator_parameters(
        "\n".join(f"{index + 1} | 100 | 3 | 1e9" for index in range(n_resonators))
    )

    if not request.config.getoption("--debug-plots"):
        monkeypatch.setattr(plt, "show", lambda: None)
    fig, _ = model.plot_resonator_components()
    plt.close(fig)

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


def _model(n_resonators=2, parameter_bounds=None, **kwargs):
    x_data = (
        np.linspace(0.5e9, 1.5e9, 61)
        if kwargs.get("fitFunction", "impedance") == "impedance"
        else np.linspace(0.0, 10e-9, 61)
    )
    if parameter_bounds is None:
        parameter_bounds = [
            (0.0, 1000.0),
            (0.5, 1e5),
            (0.1e9, 2e9),
        ] * n_resonators
    return iddefix.EvolutionaryAlgorithm(
        x_data=x_data,
        y_data=np.zeros_like(x_data),
        N_resonators=n_resonators,
        parameterBounds=parameter_bounds,
        **kwargs,
    )


# Loading and displaying parameters
@pytest.mark.parametrize("n_resonators", [None, 2])
def test_load_table_without_count_or_bounds(capsys, n_resonators):
    """A loaded table supplies the resonator count without fitting bounds."""
    frequency = np.linspace(0.5e9, 1.5e9, 61)
    model = iddefix.EvolutionaryAlgorithm(
        x_data=frequency,
        y_data=np.zeros_like(frequency),
        N_resonators=n_resonators,
        parameterBounds=None,
    )
    model.load_resonator_parameters("1 | 100 | 3 | 1e9\n2 | 200 | 5 | 1.5e9")

    assert model.N_resonators == 2
    assert model.parameterBounds is None
    model.display_resonator_parameters()
    assert "parameter bounds" not in capsys.readouterr().out
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
        x_data=frequency, y_data=np.zeros_like(frequency)
    )
    with pytest.raises(ValueError, match="No resonator rows"):
        model.load_resonator_parameters("Resonator | Rs | Q | fres")
    assert model.N_resonators is None


def test_infer_count_from_bounds():
    """Existing bounded workflows may omit the redundant resonator count."""
    frequency = np.linspace(0.5e9, 1.5e9, 61)
    model = iddefix.EvolutionaryAlgorithm(
        x_data=frequency,
        y_data=np.zeros_like(frequency),
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


@pytest.mark.parametrize("to_markdown", [False, True])
def test_display_round_trip(capsys, to_markdown):
    """Displayed ASCII and Markdown tables preserve parameters and uncertainties."""
    model = _model()
    parameters = np.array([100.0, 3.0, 1e9, 200.0, 5.0, 1.5e9])
    uncertainties = np.array([30.0, 0.5, 1e7, 20.0, 0.5, 2e7])
    model.display_resonator_parameters(
        params=parameters,
        uncertainties=uncertainties,
        to_markdown=to_markdown,
    )

    output = capsys.readouterr().out
    assert "5.0e-01" in output
    assert "100" in output
    if to_markdown:
        assert "100 ± 3.0e+01" in output
    model.load_resonator_parameters(output, use_minimization=False)

    np.testing.assert_allclose(model.evolutionParameters, parameters)
    np.testing.assert_allclose(model.evolutionParametersUncertainties, uncertainties)


@pytest.mark.parametrize("to_markdown", [False, True])
@pytest.mark.parametrize("display_uncertainties", [False, True])
@pytest.mark.parametrize("q", [0.00432198, 0.0143298])
def test_display_round_trip_small_q(q, display_uncertainties, to_markdown, capsys):
    model = _model(
        n_resonators=1,
        parameter_bounds=[(0.0, 1e6), (0.0001, 10.0), (0.1e9, 2e9)],
    )
    capsys.readouterr()
    model.display_resonator_parameters(
        params=np.array([568903.0, q, 1e9]),
        uncertainties=np.array([10.0, 0.0001, 1e7]) if display_uncertainties else None,
        to_markdown=to_markdown,
    )

    output = capsys.readouterr().out
    assert "568903" in output
    assert f"{q:.6g}" in output
    model.load_resonator_parameters(output, use_minimization=False)
    assert model.evolutionParameters[0] == pytest.approx(568903.0, rel=1e-5)
    assert model.evolutionParameters[1] == pytest.approx(q)


@pytest.mark.parametrize(
    ("display_uncertainties", "expected_width"), [(False, 65), (True, 89)]
)
def test_display_table_alignment(capsys, display_uncertainties, expected_width):
    """The header, data rows, and separators have the same visible width."""
    model = _model(n_resonators=1)
    model.display_resonator_parameters(
        params=np.array([995.0, 3.0, 1e9]),
        uncertainties=np.array([300.0, 0.5, 1e7]),
        display_uncertainties=display_uncertainties,
    )
    lines = [
        re.sub(r"\x1b\[[0-9;]*m", "", line)
        for line in capsys.readouterr().out.splitlines()
        if "|" in line or line.startswith("-")
    ]

    assert len(lines) == 5
    assert {len(line) for line in lines} == {expected_width}
    if display_uncertainties:
        assert "5.0e-01" in lines[3]


def test_uncertainty_signs_align_across_rows(capsys):
    """Values with different digit counts keep their ± signs aligned."""
    model = _model()
    model.display_resonator_parameters(
        params=np.array([100.0, 3.0, 1e9, 200.0, 246.63, 1.5e9]),
        uncertainties=np.array([30.0, 0.5, 1e7, 20.0, 230.0, 2e7]),
    )
    rows = [
        re.sub(r"\x1b\[[0-9;]*m", "", line).split("|")
        for line in capsys.readouterr().out.splitlines()
        if line.strip().startswith(("1", "2")) and "|" in line
    ]

    assert len(rows) == 2
    for column in range(1, 4):
        assert rows[0][column].index("±") == rows[1][column].index("±")


def test_notebook_table_uses_monospace_html(monkeypatch, capsys):
    """Notebook output preserves the table spacing and warning colors."""
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

    assert len(displayed) == 1
    html = displayed[0].data
    assert '<pre style="font-family: monospace;' in html
    assert "Resonator" in html and "±" in html
    assert '<span style="color: #3b82f6">' in html
    assert '<span style="color: #ef4444">' in html
    assert "Resonator" not in capsys.readouterr().out

    copied_text = unescape(re.sub(r"<[^>]+>", "", html))
    model.load_resonator_parameters(copied_text)
    np.testing.assert_allclose(model.minimizationParameters, [995.0, 3.0, 1e9])
    np.testing.assert_allclose(
        model.minimizationParametersUncertainties, [300.0, 1.0, 1e7]
    )


def test_display_warnings_are_opt_in(capsys):
    """A bound hit is reported and takes color priority over uncertainty."""
    model = _model(n_resonators=1)
    model.load_resonator_parameters("1 | 995 ± 300 | 2000 ± 500 | 1e9 ± 1e7")
    capsys.readouterr()

    model.display_resonator_parameters(display_uncertainties=True)
    output = capsys.readouterr().out
    assert "Warning:" not in output
    assert "\033[34m" not in output
    assert "\033[31m" not in output

    model.display_resonator_parameters(
        display_uncertainties=True, display_warnings=True
    )
    output = capsys.readouterr().out
    row = next(line for line in output.splitlines() if line.strip().startswith("1"))
    cells = row.split("|")

    assert "parameter bounds" in output
    assert "relative uncertainty >= 0.20" in output
    assert "\033[34m[!] Warning:" in output
    assert "\033[34m" in cells[1]
    assert "\033[31m" not in cells[1]
    assert "\033[31m" in cells[2]


def test_display_multiplicative_bounds(capsys):
    """A central Q is not flagged; a Q close to its multiplier limit is."""
    model = _model(
        n_resonators=1,
        parameter_bounds=[(0.0, 1000.0), (0.1, 10.0), (0.1e9, 2e9)],
    )
    model.load_resonator_parameters("1 | 100 | 1 | 1e9")

    model.display_resonator_parameters(display_warnings=True)
    assert "parameter bounds" not in capsys.readouterr().out

    model.modify_resonator(1, Q=9.9)
    model.display_resonator_parameters(display_warnings=True)
    output = capsys.readouterr().out
    assert "parameter bounds" in output
    assert "\033[34m" in output

    wide_model = _model(
        n_resonators=1,
        parameter_bounds=[(0.0, 1000.0), (1.0, 1e100), (0.1e9, 2e9)],
    )
    wide_model.load_resonator_parameters("1 | 100 | 2e99 | 1e9")
    wide_model.display_resonator_parameters(display_warnings=True)
    assert "parameter bounds" in capsys.readouterr().out


# Fitting parameters
def test_missing_bounds_reject_global_fitting():
    """Global fitting requires bounds after parameters are loaded for evaluation."""
    frequency = np.linspace(0.5e9, 1.5e9, 61)
    model = iddefix.EvolutionaryAlgorithm(
        x_data=frequency, y_data=np.zeros_like(frequency)
    )
    model.load_resonator_parameters("1 | 100 | 3 | 1e9")

    with pytest.raises(ValueError, match="parameterBounds"):
        model.run_differential_evolution()
    with pytest.raises(ValueError, match="parameterBounds"):
        model.run_cmaes()


@pytest.mark.parametrize("algorithm", ["differential_evolution", "cmaes"])
def test_global_fit_bound_warning(monkeypatch, capsys, algorithm):
    """Global fits automatically report proximity to parameter bounds."""
    model = _model(n_resonators=1, compute_uncertainties=False)
    solution = np.array([1000.0, 3.0, 1e9])

    if algorithm == "differential_evolution":
        monkeypatch.setattr(
            model,
            "_solve_differential_evolution",
            lambda *args, **kwargs: (solution, "converged"),
        )
        model.run_differential_evolution()
    else:
        monkeypatch.setattr(
            "iddefix.framework.Solvers.run_pymoo_cmaes_solver",
            lambda *args, **kwargs: (solution, "converged", SimpleNamespace()),
        )
        model.run_cmaes()

    assert "parameter bounds" in capsys.readouterr().out


def test_minimization_bound_warning(monkeypatch, capsys):
    """Local-fit warnings refer to its margin bounds, not global search bounds."""
    model = _model(n_resonators=1, compute_uncertainties=False)
    model.load_resonator_parameters("1 | 100 | 3 | 1e9", use_minimization=False)

    def minimize_at_upper_bound(function, x0, bounds, **kwargs):
        return SimpleNamespace(x=np.array([bounds[0][1], x0[1], x0[2]]))

    monkeypatch.setattr("iddefix.framework.minimize", minimize_at_upper_bound)
    model.run_minimization_algorithm()

    output = capsys.readouterr().out
    np.testing.assert_allclose(model.minimizationBounds[0], [90.0, 110.0])
    assert "minimization bounds" in output


# Editing resonators
def test_edit_loaded_model_without_bounds():
    """A bounds-free loaded model supports the notebook editing workflow."""
    frequency = np.linspace(0.5e9, 1.5e9, 61)
    model = iddefix.EvolutionaryAlgorithm(
        x_data=frequency, y_data=np.zeros_like(frequency)
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


def test_model_after_edits(capsys):
    """Manual edits keep the starting vector aligned with the fitted modes."""
    initial = [100.0, 3.0, 1e9, 200.0, 5.0, 1.5e9]
    model = _model(parameterEstimates=initial)
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
    capsys.readouterr()

    np.testing.assert_allclose(model.parameterEstimates, expected)
    fitted = model.get_impedance(model.x_data).real
    assert np.max(np.abs(fitted - model.y_data)) / model.y_data.max() < 0.01


def test_edit_q_boundary():
    """Invalid Q leaves the model intact; positive subcritical Q is valid."""
    model = _model(n_resonators=1)
    model.load_resonator_parameters("1 | 100 | 3 | 1e9")
    original = model.minimizationParameters.copy()

    with pytest.raises(ValueError, match="Q must be positive"):
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
    model = _model(fitFunction=fit_function, plane=plane, sigma=1e-10)
    model.load_resonator_parameters("1 | 100 | 3 | 1e9\n2 | 200 | 5 | 1.5e9")

    components = model.get_resonator_components()
    total_model = model.fitFunction(model.x_data, model.minimizationParameters)

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
    model = _model(n_resonators=1, fitFunction=fit_function, plane=plane, sigma=1e-10)
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
    model = _model(n_resonators=n_resonators, fitFunction=fit_function)
    model.evolutionParameters = np.tile([100.0, 3.0, 1e9], n_resonators)

    if not request.config.getoption("--debug-plots"):
        monkeypatch.setattr(plt, "show", lambda: None)
    fig, _ = model.plot_resonator_components()
    plt.close(fig)

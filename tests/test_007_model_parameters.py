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
from scipy.optimize import OptimizeResult

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
def test_load_table_without_count_or_bounds(n_resonators):
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


@pytest.mark.parametrize("display_uncertainties", [False, True])
def test_display_table_alignment(capsys, display_uncertainties):
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
    assert len({len(line) for line in lines}) == 1
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
    """Notebook output can be copied back into the parameter loader."""
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
    )

    assert len(displayed) == 1
    html = displayed[0].data
    assert "monospace" in html
    assert "Resonator" in html and "±" in html
    assert "Resonator" not in capsys.readouterr().out

    copied_text = unescape(re.sub(r"<[^>]+>", "", html))
    model.load_resonator_parameters(copied_text)
    np.testing.assert_allclose(model.minimizationParameters, [995.0, 3.0, 1e9])
    np.testing.assert_allclose(
        model.minimizationParametersUncertainties, [300.0, 1.0, 1e7]
    )


# Fitting parameters
def test_loaded_model_default_bounds():
    """A model loaded for evaluation can later fit with data-derived bounds."""
    frequency = np.linspace(0.5e9, 1.5e9, 61)
    target = iddefix.Impedances.Resonator_longitudinal_imp(frequency, 100, 3, 1e9)
    model = iddefix.EvolutionaryAlgorithm(frequency, target)
    model.load_resonator_parameters("1 | 100 | 3 | 1e9")

    model.run_differential_evolution(maxiter=1, popsize=5, workers=1, seed=42)

    np.testing.assert_allclose(
        model.parameterBounds,
        [(0, 100), (0.001, 1000), (frequency[0], 10 * frequency[-1])],
    )
    assert np.isfinite(model.get_impedance(use_minimization=False)).all()


def test_minimization_uses_margin_bounds(monkeypatch):
    """Local refinement constrains parameters to its margin bounds."""
    model = _model(n_resonators=1, compute_uncertainties=False)
    model.load_resonator_parameters("1 | 100 | 3 | 1e9", use_minimization=False)

    def minimize_at_upper_bound(function, x0, bounds, **kwargs):
        return OptimizeResult(
            x=np.array([bounds[0][1], x0[1], x0[2]]), message="Finished", nit=0, nfev=0
        )

    monkeypatch.setattr("iddefix.framework.minimize", minimize_at_upper_bound)
    model.run_minimization_algorithm()

    np.testing.assert_allclose(model.minimizationBounds[0], [90.0, 110.0])


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
    model.impedance_data = iddefix.Impedances.n_Resonator_longitudinal_imp(
        model.x_data, expected
    ).real
    model.run_minimization_algorithm()
    capsys.readouterr()

    np.testing.assert_allclose(model.parameterEstimates, expected)
    fitted = model.get_impedance(model.x_data).real
    assert (
        np.max(np.abs(fitted - model.impedance_data)) / model.impedance_data.max()
        < 0.01
    )


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
        ("impedance", "longitudinal", r"Real part $\Re(Z)$ [$\Omega$]"),
        ("impedance", "transverse", r"Real part $\Re(Z)$ [$\Omega$/m]"),
        ("wake", "longitudinal", "Wake [V/C]"),
        ("wake potential", "longitudinal", "Wake potential [V/C]"),
        ("wake", "transverse", "Wake [V/C/m]"),
    ],
)
def test_plot_resonator_components(request, monkeypatch, fit_function, plane, ylabel):
    """Component plots use the selected plane's impedance or wake units."""
    model = _model(n_resonators=1, fitFunction=fit_function, plane=plane, sigma=1e-10)
    model.load_resonator_parameters("1 | 100 | 3 | 1e9")

    if not request.config.getoption("--debug-plots"):
        monkeypatch.setattr(plt, "show", lambda: None)
    fig, axes = model.plot_resonator_components()

    assert axes[0].get_ylabel() == ylabel
    if fit_function == "impedance":
        units = r"[$\Omega$]" if plane == "longitudinal" else r"[$\Omega$/m]"
        assert axes[1].get_ylabel() == r"Imaginary part $\Im(Z)$ " + units
    plt.close(fig)


@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("plane", ["longitudinal", "transverse"])
def test_plot_absolute_impedance_components(monkeypatch, complex_input, plane):
    """Magnitude-only fits show component and total magnitudes on one panel."""
    frequency = np.linspace(0.5e9, 1.5e9, 61)
    parameters = [100.0, 3.0, 0.9e9, 80.0, 4.0, 1.1e9]
    formula = getattr(iddefix.Impedances, f"n_Resonator_{plane}_imp")
    impedance = formula(frequency, parameters)
    input_data = impedance if complex_input else np.abs(impedance)
    model = iddefix.EvolutionaryAlgorithm(
        frequency, input_data, objectiveFunction="abs", plane=plane
    )
    model.load_resonator_parameters("1 | 100 | 3 | 0.9e9\n2 | 80 | 4 | 1.1e9")
    monkeypatch.setattr(plt, "show", lambda: None)

    fig, axes = model.plot_resonator_components()

    assert len(axes) == 1
    units = r"[$\Omega$]" if plane == "longitudinal" else r"[$\Omega$/m]"
    for axis in axes:
        assert axis.get_ylabel().endswith(units)
    components = model.get_resonator_components()
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(), np.abs(input_data))
    np.testing.assert_allclose(
        axes[0].lines[1].get_ydata(), np.abs(components.sum(axis=0))
    )
    for line, component in zip(axes[0].lines[2:], components):
        np.testing.assert_allclose(line.get_ydata(), np.abs(component))
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


@pytest.mark.parametrize("plane", ["longitudinal", "transverse"])
@pytest.mark.parametrize("part", ["real", "imag", "abs", "both"])
@pytest.mark.parametrize("components", [False, True])
def test_plot_impedance_parts(plane, part, components):
    """Plot selection is independent of complex input and fitting objective."""
    model = _model(plane=plane, objectiveFunction="abs")
    model.load_resonator_parameters("1 | 100 | 3 | 1e9\n2 | 200 | 5 | 1.5e9")
    frequency = model.frequency_data
    parameters = model.minimizationParameters
    formula = getattr(iddefix.Impedances, f"n_Resonator_{plane}_imp")
    data = formula(frequency, parameters) + 1j
    model.impedance_data = data
    fig, axes = model.plot(type=part, show_components=components, show=False)
    selected = (
        [np.real, np.imag]
        if part == "both"
        else [{"real": np.real, "imag": np.imag, "abs": np.abs}[part]]
    )
    assert len(axes) == len(selected)
    for axis, transform in zip(axes, selected):
        assert len(axis.lines) == (4 if components else 2)
        np.testing.assert_allclose(axis.lines[0].get_ydata(), transform(data))
        np.testing.assert_allclose(
            axis.lines[1].get_ydata(), transform(formula(frequency, parameters))
        )
        if components:
            for line, row in zip(axis.lines[2:], parameters.reshape(-1, 3)):
                np.testing.assert_allclose(
                    line.get_ydata(), transform(formula(frequency, row))
                )
    assert model.objectiveFunction is iddefix.ObjectiveFunctions.sumOfSquaredErrorAbs
    plt.close(fig)


@pytest.mark.parametrize("plane", ["longitudinal", "transverse"])
@pytest.mark.parametrize("function", ["impedance", "wake", "wake potential"])
@pytest.mark.parametrize("components", [False, True])
@pytest.mark.parametrize("source", ["impedance", "wake", "wake potential"])
def test_plot_other_functions(plane, function, components, source):
    """Evaluate each domain on a new grid without modifying stored fitting data."""
    model = _model(fitFunction=source, plane=plane, sigma=1e-10)
    model.load_resonator_parameters("1 | 100 | 3 | 1e9\n2 | 200 | 5 | 1.5e9")
    before = {
        name: getattr(model, name)
        for name in [
            "x_data",
            "y_data",
            "time_data",
            "wake_data",
            "frequency_data",
            "impedance_data",
            "wake_potential_data",
            "sigma",
        ]
    }
    x = (
        np.linspace(0.1e9, 3e9, 83)
        if function == "impedance"
        else np.linspace(-1e-9, 30e-9, 83)
    )
    kwargs = (
        {"wake_length": 0.5}
        if function == "impedance"
        else {"sigma": 2e-10}
        if function == "wake potential"
        else {}
    )
    if function == "impedance":
        formula = getattr(iddefix.Impedances, f"n_Resonator_{plane}_imp")
    else:
        formula = getattr(
            iddefix.Wakes, f"n_Resonator_{plane}_{function.replace(' ', '_')}"
        )
    method = model.plot_resonator_components if components else model.plot
    fig, axes = method(x, function=function, type="real", show=False, **kwargs)
    assert len(axes[0].lines) == (3 if components else 1)
    np.testing.assert_allclose(axes[0].lines[0].get_xdata(), x)
    np.testing.assert_allclose(
        axes[0].lines[0].get_ydata(),
        np.real(formula(x, model.minimizationParameters, **kwargs)),
    )
    if components:
        contributions = [line.get_ydata() for line in axes[0].lines[1:]]
        np.testing.assert_allclose(
            np.sum(contributions, axis=0), axes[0].lines[0].get_ydata()
        )
    for name, value in before.items():
        assert getattr(model, name) is value
    plt.close(fig)


def test_plot_stored_grid_and_reference():
    """Matching grids use fitting data; a different domain uses only its prediction."""
    model = _model()
    model.load_resonator_parameters("1 | 100 | 3 | 1e9\n2 | 200 | 5 | 1.5e9")
    fig, axes = model.plot(show=False)
    assert all(len(axis.lines) == 2 for axis in axes)
    plt.close(fig)
    with pytest.raises(ValueError, match="x_data"):
        model.plot(function="wake", show=False)
    model.time_data = np.linspace(0, 10e-9, 101)
    fig, axes = model.plot(function="wake", show=False)
    assert len(axes[0].lines) == 1
    np.testing.assert_array_equal(axes[0].lines[0].get_xdata(), model.time_data)
    plt.close(fig)


def test_plot_explicit_reference_and_styles(monkeypatch):
    """Explicit reference data, stage selection, styles, and deferred display work together."""
    model = _model()
    model.load_resonator_parameters("1 | 100 | 3 | 1e9\n2 | 200 | 5 | 1.5e9")
    model.evolutionParameters = model.minimizationParameters * 1.01
    x = np.linspace(0.2e9, 2e9, 101)
    y = np.ones(x.shape, dtype=complex) * (3 + 4j)
    displayed = []
    monkeypatch.setattr(plt, "show", lambda: displayed.append(True))
    fig, axes = model.plot_resonator_components(
        x,
        y,
        False,
        "viridis",
        (7, 4),
        80,
        {"color": "green"},
        {"linestyle": ":"},
        {"linewidth": 3},
        "log",
        type="abs",
        show=False,
        subplot_kw={"yscale": "log"},
        squeeze=False,
    )
    assert displayed == []
    assert axes.shape == (1,)
    axis = axes[0]
    np.testing.assert_array_equal(axis.lines[0].get_ydata(), np.abs(y))
    np.testing.assert_allclose(
        axis.lines[1].get_ydata(),
        np.abs(model.get_impedance(x, use_minimization=False)),
    )
    assert axis.lines[0].get_color() == "green"
    assert axis.lines[1].get_linewidth() == 3
    assert axis.lines[2].get_linestyle() == ":"
    assert axis.get_xscale() == axis.get_yscale() == "log"
    assert fig.dpi == 80
    np.testing.assert_allclose(fig.get_size_inches(), [7, 4])
    axis.set_title("Customized before display")
    plt.show()
    assert displayed == [True]
    plt.close(fig)
    fig, _ = model.plot(type="abs")
    assert displayed == [True, True]
    plt.close(fig)


@pytest.mark.parametrize("plane", ["longitudinal", "transverse"])
def test_plot_model_defaults(plane):
    """Finite impedance and Gaussian duration defaults follow the model settings."""
    model = _model(plane=plane, wake_length=0.5)
    model.load_resonator_parameters("1 | 100 | 3 | 1e9\n2 | 200 | 5 | 1.5e9")
    fig, axes = model.plot(type="real", show=False)
    formula = getattr(iddefix.Impedances, f"n_Resonator_{plane}_imp")
    np.testing.assert_allclose(
        axes[0].lines[1].get_ydata(),
        formula(
            model.frequency_data, model.minimizationParameters, wake_length=0.5
        ).real,
    )
    plt.close(fig)
    time = np.linspace(-1e-9, 10e-9, 101)
    potential = getattr(iddefix.Wakes, f"n_Resonator_{plane}_wake_potential")
    for sigma in (None, 2e-10):
        model.sigma = sigma
        fig, axes = model.plot(time, function="wake potential", show=False)
        expected = potential(
            time, model.minimizationParameters, sigma=1e-10 if sigma is None else sigma
        )
        np.testing.assert_allclose(axes[0].lines[0].get_ydata(), expected)
        assert model.sigma is sigma
        plt.close(fig)


@pytest.mark.parametrize("components", [False, True])
def test_plot_colorbar(components):
    """Only displayed resonator contributions trigger the many-mode colorbar."""
    model = _model(n_resonators=10)
    model.evolutionParameters = np.tile([100.0, 3.0, 1e9], 10)
    fig, axes = model.plot(type="abs", show_components=components, show=False)
    assert len(axes) == 1
    assert len(fig.axes) == (2 if components else 1)
    plt.close(fig)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"function": "invalid"},
        {"type": "invalid"},
        {"scale": "invalid"},
        {"x_data": []},
        {"x_data": [[1, 2]]},
        {"x_data": [1, 2], "y_data": [1]},
        {"x_data": [0.0, 1e-9], "function": "wake potential", "sigma": 0},
    ],
)
def test_plot_invalid_arguments(kwargs):
    """Reject invalid plotting requests before creating a figure."""
    model = _model()
    model.load_resonator_parameters("1 | 100 | 3 | 1e9\n2 | 200 | 5 | 1.5e9")
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        model.plot(show=False, **kwargs)
    assert plt.get_fignums() == before


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("to_markdown", [False, True])
def test_display_bounds(explicit, to_markdown, capsys):
    """Display supplied or stored bound triples without changing model bounds."""
    model = _model(n_resonators=1)
    stored = list(model.parameterBounds)
    supplied = [(2, 4), (3, 6), (1e9, 2e9)] if explicit else None
    capsys.readouterr()
    model.display_parameter_bounds(supplied, to_markdown=to_markdown)
    output = capsys.readouterr().out
    assert "2 to 4" in output if explicit else "0 to 1000" in output
    assert "fres [Hz]" in output
    assert model.parameterBounds == stored


def test_display_default_bounds(capsys):
    """Displaying an unfitted model can derive and store its default bounds."""
    x = np.linspace(1e8, 2e9, 41)
    y = iddefix.Impedances.Resonator_longitudinal_imp(x, 100, 3, 1e9)
    model = iddefix.EvolutionaryAlgorithm(x, y, N_resonators=1)
    model.display_parameter_bounds(to_markdown=True)
    assert " to " in capsys.readouterr().out
    np.testing.assert_allclose(
        model.parameterBounds, [(0, y.real.max()), (0.001, 1000), (x[0], 10 * x[-1])]
    )


@pytest.mark.parametrize("fit_function", ["impedance", "wake", "wake potential", "abs"])
@pytest.mark.parametrize("scale", ["linear", "log", "loglog"])
def test_component_plot_scales(monkeypatch, fit_function, scale):
    """All plot domains apply scales without changing the plotted values."""
    model = _model(
        n_resonators=1,
        fitFunction="impedance" if fit_function == "abs" else fit_function,
        objectiveFunction="abs" if fit_function == "abs" else "real",
        sigma=1e-10,
    )
    model.load_resonator_parameters("1 | 100 | 3 | 1e9")
    monkeypatch.setattr(plt, "show", lambda: None)
    components = model.get_resonator_components().copy()
    fig, axes = model.plot_resonator_components(scale=scale)
    for axis in axes:
        assert axis.get_xscale() == ("linear" if scale == "linear" else "log")
        assert axis.get_yscale() == ("log" if scale == "loglog" else "linear")
    np.testing.assert_array_equal(model.get_resonator_components(), components)
    plt.close(fig)


def test_component_plot_axis_override(monkeypatch):
    """Explicit subplot axis settings override scale defaults independently."""
    model = _model(n_resonators=1)
    model.load_resonator_parameters("1 | 100 | 3 | 1e9")
    monkeypatch.setattr(plt, "show", lambda: None)
    fig, axes = model.plot_resonator_components(
        scale="loglog",
        subplot_kw={"yscale": "linear"},
    )
    for axis in axes:
        assert axis.get_xscale() == "log"
        assert axis.get_yscale() == "linear"
    plt.close(fig)


@pytest.fixture
def estimated_model():
    frequency = np.linspace(0.1e9, 4e9, 4001)
    impedance = iddefix.Impedances.n_Resonator_longitudinal_imp(
        frequency,
        [100, 20, 1e9, 80, 20, 2e9, 60, 20, 3e9],
    )
    model = iddefix.EvolutionaryAlgorithm(frequency, impedance)
    model.estimate_bounds(show_table=False, show_inspect=False)
    assert model.N_resonators == 3
    return model


@pytest.mark.parametrize("numbers", [2, [1, 3, 1]])
def test_remove_linked_resonators(estimated_model, numbers):
    """Fitted values and uncertainties remain aligned with edited peak metadata."""
    model = estimated_model
    parameters = np.asarray(model.parameterEstimates).copy()
    uncertainty = np.arange(1.0, 10.0)
    for name in ["evolutionParameters", "minimizationParameters"]:
        setattr(model, name, parameters.copy())
        setattr(model, name + "Uncertainties", uncertainty.copy())
    model.minimization_result = OptimizeResult(x=parameters.copy())
    kept = np.delete(np.arange(3), np.unique(np.atleast_1d(numbers)) - 1)
    indices = (3 * kept[:, None] + np.arange(3)).ravel()
    model.remove_resonator(numbers)
    assert model.bounds is model.smart_bounds
    assert model.bounds.N_resonators == model.N_resonators == len(kept)
    assert model.bounds.parameterBounds is model.parameterBounds
    assert model.bounds.parameterEstimates is model.parameterEstimates
    assert model.minimization_result is None
    for name in ["evolutionParameters", "minimizationParameters"]:
        np.testing.assert_array_equal(getattr(model, name), parameters[indices])
        np.testing.assert_array_equal(
            getattr(model, name + "Uncertainties"), uncertainty[indices]
        )
    np.testing.assert_allclose(
        model.get_impedance(), model.get_resonator_components().sum(axis=0)
    )


def test_edit_model_peaks(estimated_model):
    """Peak editing clears fits and supports adding a removed peak again."""
    model = estimated_model
    model.minimizationParameters = model.parameterEstimates.copy()
    model.remove_peak(2)
    assert model.N_resonators == 2
    assert model.minimizationParameters is None
    model.add_peak(2e9, Q=20)
    assert model.N_resonators == model.bounds.N_resonators == 3
    model.bounds.remove_peak([1, 3])
    model.apply_bounds(model.bounds)
    assert model.N_resonators == 1
    assert model.parameterEstimates[2] == pytest.approx(2e9, rel=0.001)


def test_add_resonator_detaches_peaks(estimated_model):
    """Manual modes leave earlier peak estimates unchanged and remain removable."""
    model = estimated_model
    previous_bounds = model.bounds
    model.minimizationParameters = model.parameterEstimates.copy()
    model.add_resonator(
        50, 10, 3.5e9, parameter_bounds=[(0, 100), (1, 100), (3e9, 4e9)]
    )
    assert model.bounds is None
    assert previous_bounds.N_resonators == 3
    model.remove_resonator([1, 4])
    assert model.N_resonators == 2
    assert len(model.parameterBounds) == len(model.parameterEstimates) == 6


@pytest.mark.parametrize("numbers", [0, 4, True, 1.5, [1, 2, 3]])
def test_remove_resonator_invalid(estimated_model, numbers):
    """Invalid removals preserve model parameters and linked peak metadata."""
    model = estimated_model
    before = model.parameterEstimates.copy()
    with pytest.raises(ValueError):
        model.remove_resonator(numbers)
    np.testing.assert_array_equal(model.parameterEstimates, before)
    assert model.N_resonators == model.bounds.N_resonators == 3

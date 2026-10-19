"""Interactive SmartBounds workflows with analytical resonator spectra."""

import asyncio

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import CloseEvent, KeyEvent, MouseEvent

import iddefix

pytestmark = pytest.mark.interactive


def _click(figure, frequency, impedance, button=1):
    axes = figure.axes[0]
    x, y = axes.transData.transform((frequency, impedance))
    event = MouseEvent("button_press_event", figure.canvas, x, y, button=button)
    figure.canvas.callbacks.process("button_press_event", event)


@pytest.mark.parametrize("run_in_ipython", [False, True])
def test_inline_backend_fallback(monkeypatch, capsys, run_in_ipython):
    """Inline Python and IPython runs warn and estimate peaks without prompting."""
    frequency = np.linspace(0.2e9, 1.4e9, 10001)
    expected = np.array([100.0, 8.0, 0.8e9])
    impedance = iddefix.Impedances.Resonator_longitudinal_imp(frequency, *expected).real
    monkeypatch.setattr(
        matplotlib, "get_backend", lambda: "module://matplotlib_inline.backend_inline"
    )
    monkeypatch.setattr(
        "builtins.input", lambda prompt: pytest.fail("Inline selection requested input")
    )

    if run_in_ipython:
        shell = pytest.importorskip(
            "IPython.core.interactiveshell"
        ).InteractiveShell.instance()
        monkeypatch.setitem(shell.user_ns, "iddefix", iddefix)
        monkeypatch.setitem(shell.user_ns, "frequency", frequency)
        monkeypatch.setitem(shell.user_ns, "impedance", impedance)
        monkeypatch.setitem(shell.user_ns, "_smartbounds_test_result", None)
        result = shell.run_cell(
            "_smartbounds_test_result = iddefix.SmartBoundDetermination("
            "frequency, impedance, impedance_type='real', interactive=True)"
        )
        assert result.success
        bounds = shell.user_ns["_smartbounds_test_result"]
    else:
        bounds = iddefix.SmartBoundDetermination(
            frequency, impedance, impedance_type="real", interactive=True
        )

    warning = capsys.readouterr().out
    assert "%matplotlib widget" in warning
    assert "%matplotlib ipympl" in warning
    assert "automatic peak detection" in warning
    assert bounds.N_resonators == 1
    np.testing.assert_allclose(bounds.parameterEstimates, expected, rtol=0.01)
    assert asyncio.run(bounds.wait_for_selection()) == bounds.parameterBounds


def test_desktop_selection(monkeypatch):
    """Click, undo, and finish select two analytical resonators."""
    frequency = np.linspace(0.1e9, 2e9, 10001)
    expected = np.array([[100.0, 8.0, 0.5e9], [80.0, 10.0, 1.5e9]])
    impedance = sum(
        iddefix.Impedances.Resonator_longitudinal_imp(frequency, *resonator)
        for resonator in expected
    ).real
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "agg")
    monkeypatch.setattr(plt, "show", lambda **kwargs: None)

    bounds = iddefix.SmartBoundDetermination(
        frequency, impedance, impedance_type="real", interactive=True
    )
    figure = plt.gcf()
    figure.canvas.draw()
    for selected_frequency, button in (
        (0.5e9, 1),
        (1.0e9, 1),
        (1.0e9, 3),
        (1.5e9, 1),
    ):
        selected_height = np.interp(selected_frequency, frequency, impedance)
        _click(figure, selected_frequency, selected_height, button=button)
    figure.canvas.callbacks.process(
        "key_press_event", KeyEvent("key_press_event", figure.canvas, key="enter")
    )

    assert bounds.N_resonators == 2
    np.testing.assert_allclose(
        np.asarray(bounds.parameterEstimates).reshape(-1, 3), expected, rtol=0.01
    )
    assert asyncio.run(bounds.wait_for_selection()) == bounds.parameterBounds


def test_widget_selection(monkeypatch):
    """Widget controls select an analytical resonator after an undo."""
    pytest.importorskip("ipywidgets")
    display = pytest.importorskip("IPython.display")
    frequency = np.linspace(0.2e9, 1.4e9, 10001)
    expected = np.array([100.0, 8.0, 0.8e9])
    impedance = iddefix.Impedances.Resonator_longitudinal_imp(frequency, *expected).real
    rendered = []
    monkeypatch.setattr(
        matplotlib, "get_backend", lambda: "module://ipympl.backend_nbagg"
    )
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(display, "display", rendered.append)

    bounds = iddefix.SmartBoundDetermination(
        frequency, impedance, impedance_type="real", interactive=True
    )
    figure = plt.gcf()
    figure.canvas.draw()
    controls = {
        button.description: button for button in rendered[0].children[1].children
    }
    pick, undo, done = (controls[name] for name in ("Pick", "Undo", "Done"))
    pick.value = True
    peak_height = np.interp(expected[2], frequency, impedance)
    _click(figure, expected[2], peak_height)
    undo.click()
    _click(figure, expected[2], peak_height)
    done.click()

    assert bounds.N_resonators == 1
    np.testing.assert_allclose(bounds.parameterEstimates, expected, rtol=0.01)
    assert asyncio.run(bounds.wait_for_selection()) == bounds.parameterBounds


def test_cancelled_selection(monkeypatch):
    """Closing the plot before selection reports cancellation."""
    frequency = np.linspace(0.2e9, 1.4e9, 1001)
    impedance = iddefix.Impedances.Resonator_longitudinal_imp(
        frequency, 100.0, 8.0, 0.8e9
    ).real
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "agg")
    monkeypatch.setattr(plt, "show", lambda **kwargs: None)

    bounds = iddefix.SmartBoundDetermination(
        frequency, impedance, impedance_type="real", interactive=True
    )
    figure = plt.gcf()
    figure.canvas.callbacks.process(
        "close_event", CloseEvent("close_event", figure.canvas)
    )
    plt.close(figure)

    with pytest.raises(RuntimeError, match="closed before completion"):
        asyncio.run(bounds.wait_for_selection())

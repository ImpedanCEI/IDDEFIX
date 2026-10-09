"""Automatic and interactive workflows for choosing resonator fit bounds.

The tests cover peak detection, width interpolation, and user selection.
"""

import asyncio
from contextlib import nullcontext
from importlib import import_module

import numpy as np
import pytest
from matplotlib.backend_bases import KeyEvent, MouseEvent

import iddefix


def _two_peaks():
    frequency = np.linspace(0.0, 10.0, 2001)
    impedance = 10.0 * np.exp(-(((frequency - 3.0) / 0.1) ** 2) / 2) + 5.0 * np.exp(
        -(((frequency - 7.0) / 0.15) ** 2) / 2
    )
    return frequency, impedance


def _click(figure, frequency, impedance, button=3):
    axes = figure.axes[0]
    figure.canvas.draw()
    x, y = axes.transData.transform((frequency, impedance))
    event = MouseEvent("button_press_event", figure.canvas, x, y, button=button)
    figure.canvas.callbacks.process("button_press_event", event)


def test_smart_bounds_detects_modes_and_respects_height():
    frequency, impedance = _two_peaks()
    bounds = iddefix.SmartBoundDetermination(
        frequency, impedance, minimum_peak_height=1.0
    )

    assert bounds.N_resonators == 2
    assert len(bounds.parameterBounds) == 6
    assert all(low < high for low, high in bounds.parameterBounds)

    bounds.find(minimum_peak_height=7.0)
    assert bounds.N_resonators == 1
    assert len(bounds.parameterBounds) == 3


def test_smart_bounds_interpolation_uses_exact_longitudinal_crossing():
    frequency = np.linspace(5.0, 15.0, 101)
    impedance = 10.0 / (1.0 + (frequency - 10.0) ** 2)
    bounds = iddefix.SmartBoundDetermination(
        frequency,
        impedance,
        minimum_peak_height=np.full_like(frequency, 5.0),
        samples=1001,
        impedance_type="real",
        Q_bounds=[1.0, 1.0],
    )

    assert len(bounds.analysis_frequency_data) == 1001
    np.testing.assert_array_equal(bounds.frequency_data, frequency)
    assert bounds.N_resonators == 1
    crossing = bounds.crossing_frequencies[0]
    expected_q = crossing * 10.0 / abs(crossing**2 - 10.0**2)
    np.testing.assert_allclose(bounds.parameterBounds[1], [expected_q, expected_q])


def test_smart_bounds_uses_selected_side_for_coupled_modes():
    frequency = np.arange(8.0)
    impedance = np.array([0.0, 0.0, 10.0, 4.0, 8.0, 3.0, 0.0, 0.0])
    left = iddefix.SmartBoundDetermination(
        frequency, impedance, minimum_peak_height=5.0, q_side="left"
    )
    right = iddefix.SmartBoundDetermination(
        frequency, impedance, minimum_peak_height=5.0, q_side=["left", "right"]
    )

    np.testing.assert_array_equal(right.peaks, [2, 4])
    assert right.q_sides_used == ["left", "right"]
    assert right.parameterBounds[4][0] > left.parameterBounds[4][0]


def test_interactive_picker_selects_modes_without_widget_backend(monkeypatch):
    smart_bounds = import_module("iddefix.smartBoundDetermination")
    monkeypatch.setattr(smart_bounds.matplotlib, "get_backend", lambda: "agg")
    monkeypatch.setattr(smart_bounds.plt, "show", lambda **kwargs: None)

    frequency = np.arange(7.0)
    impedance = np.array([0.0, 1.0, 10.0, 1.0, 8.0, 1.0, 0.0])
    bounds = iddefix.SmartBoundDetermination(frequency, impedance, interactive=True)
    figure = bounds._selection_figure

    _click(figure, 2.0, 10.0, button=1)
    _click(figure, 4.0, 8.0, button=1)
    event = KeyEvent("key_press_event", figure.canvas, key="enter")
    figure.canvas.callbacks.process("key_press_event", event)

    np.testing.assert_array_equal(bounds.peaks, [2, 4])
    assert asyncio.run(bounds.wait_for_selection()) is bounds.parameterBounds
    smart_bounds.plt.close(figure)


def test_widget_picker_allows_zoom_then_right_click_and_undo(monkeypatch):
    pytest.importorskip("IPython.display")
    pytest.importorskip("ipywidgets")
    smart_bounds = import_module("iddefix.smartBoundDetermination")
    monkeypatch.setattr(
        smart_bounds.matplotlib,
        "get_backend",
        lambda: "module://ipympl.backend_nbagg",
    )
    monkeypatch.setattr(smart_bounds.plt, "show", lambda **kwargs: None)
    monkeypatch.setattr("IPython.display.display", lambda widget: None)

    frequency = np.arange(7.0)
    impedance = np.array([0.0, 1.0, 10.0, 1.0, 8.0, 1.0, 0.0])
    bounds = iddefix.SmartBoundDetermination(frequency, impedance, interactive=True)
    figure = bounds._selection_figure

    class Toolbar:
        mode = "zoom rect"

        def zoom(self):
            self.mode = ""

        def _wait_cursor_for_draw_cm(self):
            return nullcontext()

    figure.canvas.toolbar = Toolbar()
    _click(figure, 2.0, 10.0)
    assert bounds.peaks is None

    bounds._selection_pick_button.value = True
    assert figure.canvas.toolbar.mode == ""
    _click(figure, 2.0, 10.0)
    _click(figure, 4.0, 8.0)
    bounds._selection_undo_button.click()
    _click(figure, 4.0, 8.0)
    bounds._selection_done_button.click()

    np.testing.assert_array_equal(bounds.peaks, [2, 4])
    assert bounds.N_resonators == 2
    assert asyncio.run(bounds.wait_for_selection()) is bounds.parameterBounds
    smart_bounds.plt.close(figure)


def test_smart_bounds_allows_overdamped_q_bounds():
    frequency = np.array([0.0, 10.0, 1000.0, 2000.0])
    impedance = np.array([10.0, 11.0, 10.0, 0.0])

    bounds = iddefix.SmartBoundDetermination(
        frequency, impedance, minimum_peak_height=1.0
    )

    assert bounds.N_resonators == 1
    peak_frequency = frequency[bounds.peaks[0]]
    crossing = bounds.crossing_frequencies[0]
    estimated_q = crossing * peak_frequency / abs(crossing**2 - peak_frequency**2)
    assert 0 < bounds.parameterBounds[1][0] < 0.5
    assert bounds.parameterBounds[1][0] == pytest.approx(estimated_q * 0.5)


def test_smart_bounds_uses_q_one_fallback_when_crossing_is_missing():
    frequency = np.array([0.0, 1.0, 2.0])
    impedance = np.array([9.0, 10.0, 9.0])

    bounds = iddefix.SmartBoundDetermination(
        frequency, impedance, minimum_peak_height=1.0
    )

    assert bounds.N_resonators == 1
    assert bounds.upper_lower_bounds[0] == 0.0
    assert bounds.parameterBounds[1] == (0.5, 5.0)


def test_frequency_bounds_scale_with_estimated_q():
    frequency = np.linspace(1e6, 2e9, 50001)
    widths = []

    for expected_q in (2.0, 10.0):
        impedance = iddefix.Impedances.Resonator_longitudinal_imp(
            frequency, 100.0, expected_q, 0.8e9
        )
        bounds = iddefix.SmartBoundDetermination(
            frequency,
            impedance.real,
            minimum_peak_height=50.0,
            impedance_type="real",
        )
        fres_min, fres_max = bounds.parameterBounds[2]
        widths.append(fres_max - fres_min)

    assert widths[0] > widths[1]


@pytest.mark.parametrize("q_side", ["left", "right"])
def test_transverse_real_bounds_recover_resonator_parameters(q_side):
    frequency = np.linspace(1e6, 2e9, 50001)
    expected = np.array([4e6, 0.5, 0.8e9])
    impedance = iddefix.Impedances.Resonator_transverse_imp(frequency, *expected)

    bounds = iddefix.SmartBoundDetermination(
        frequency,
        impedance.real,
        minimum_peak_height=1e5,
        Rs_bounds=[1.0, 1.0],
        Q_bounds=[1.0, 1.0],
        fres_bounds=[0.0, 0.0],
        impedance_type="real",
        plane="transverse",
        q_side=q_side,
    )

    np.testing.assert_allclose(bounds.parameterEstimates, expected, rtol=3e-4)


def test_add_transverse_peak_uses_supplied_q_without_crossing():
    frequency = np.linspace(1e6, 2e9, 50001)
    expected = np.array([4e6, 2.0, 0.8e9])
    impedance = iddefix.Impedances.Resonator_transverse_imp(frequency, *expected)
    peak_frequency = frequency[np.argmax(impedance.real)]

    bounds = iddefix.SmartBoundDetermination(
        frequency,
        impedance.real,
        minimum_peak_height=impedance.real.max() + 1.0,
        Rs_bounds=[1.0, 1.0],
        Q_bounds=[1.0, 1.0],
        fres_bounds=[0.0, 0.0],
        impedance_type="real",
        plane="transverse",
    )
    bounds.add_peak(peak_frequency, Q=expected[1])

    assert np.isnan(bounds.crossing_frequencies[0])
    np.testing.assert_allclose(bounds.parameterEstimates, expected, rtol=1e-4)

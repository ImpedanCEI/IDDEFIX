"""SmartBounds estimates checked against analytical resonator impedances."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

import iddefix


def _single_mode_bounds():
    """Create SmartBounds through the analytical workflow used in examples."""
    frequency = np.linspace(0.5e9, 1.5e9, 2001)
    impedance = iddefix.Impedances.Resonator_longitudinal_imp(
        frequency, 100.0, 8.0, 1e9
    ).real
    return iddefix.SmartBoundDetermination(
        frequency,
        impedance,
        impedance_type="real",
        minimum_peak_height=50.0,
    )


# Peak discovery and frequency bounds
def test_detected_modes():
    """Two longitudinal peaks recover their parameters and combined spectrum."""
    frequency = np.linspace(0.1e9, 2e9, 10001)
    expected = np.array([[100.0, 8.0, 0.5e9], [80.0, 10.0, 1.5e9]])
    impedance = sum(
        iddefix.Impedances.Resonator_longitudinal_imp(frequency, *resonator)
        for resonator in expected
    ).real

    bounds = iddefix.SmartBoundDetermination(
        frequency,
        impedance,
        impedance_type="real",
        minimum_peak_height=50.0,
    )

    assert bounds.N_resonators == 2
    np.testing.assert_allclose(
        np.asarray(bounds.parameterEstimates).reshape(-1, 3), expected, rtol=0.01
    )
    model = bounds.get_impedance_components().sum(axis=0)
    assert np.max(np.abs(model.real - impedance)) / impedance.max() < 0.01


def test_find_updates_plane():
    """Re-running find on a transverse trace replaces longitudinal estimates."""
    frequency = np.linspace(1e6, 2e9, 5001)
    longitudinal = iddefix.Impedances.Resonator_longitudinal_imp(
        frequency, 100.0, 2.0, 0.8e9
    ).real
    bounds = iddefix.SmartBoundDetermination(
        frequency, longitudinal, impedance_type="real", minimum_peak_height=50.0
    )

    refined_frequency = np.linspace(1e6, 2e9, 10001)
    expected = np.array([4e6, 0.5, 0.8e9])
    transverse = iddefix.Impedances.Resonator_transverse_imp(
        refined_frequency, *expected
    ).real
    bounds.find(
        frequency_data=refined_frequency,
        impedance_data=transverse,
        minimum_peak_height=1e5,
        impedance_type="real",
        plane="transverse",
        samples=20001,
    )

    assert bounds.N_resonators == 1
    np.testing.assert_allclose(bounds.parameterEstimates, expected, rtol=0.002)


def test_frequency_bounds_scale_with_q():
    """A lower-Q resonator receives a wider frequency interval."""
    frequency = np.linspace(1e6, 2e9, 10001)
    widths = []

    for Q in (2.0, 10.0):
        impedance = iddefix.Impedances.Resonator_longitudinal_imp(
            frequency, 100.0, Q, 0.8e9
        ).real
        bounds = iddefix.SmartBoundDetermination(
            frequency,
            impedance,
            impedance_type="real",
            minimum_peak_height=50.0,
        )
        low, high = bounds.parameterBounds[2]
        widths.append(high - low)

    assert widths[0] > widths[1]


# Longitudinal estimates
@pytest.mark.parametrize("Q", [0.3, 10.0])
@pytest.mark.parametrize("impedance_type", ["real", "absolute"])
def test_longitudinal_frequency_resolution(Q, impedance_type):
    """Real and absolute estimates improve as frequency sampling becomes finer."""
    expected = np.array([100.0, Q, 0.8e9])
    estimates = []

    for samples in (401, 20001):
        frequency = np.linspace(1e6, 3e9, samples)
        impedance = iddefix.Impedances.Resonator_longitudinal_imp(frequency, *expected)
        data = impedance.real if impedance_type == "real" else np.abs(impedance)
        bounds = iddefix.SmartBoundDetermination(
            frequency,
            data,
            impedance_type=impedance_type,
            minimum_peak_height=50.0,
        )
        assert bounds.N_resonators == 1
        estimates.append(np.asarray(bounds.parameterEstimates))

    coarse_error, fine_error = [
        np.max(np.abs((estimate - expected) / expected)) for estimate in estimates
    ]
    assert fine_error < coarse_error
    assert fine_error < 0.003
    assert all(
        low < value < high
        for value, (low, high) in zip(expected, bounds.parameterBounds)
    )


# Transverse estimates and crossings
@pytest.mark.parametrize("q_side", ["left", "right"])
def test_transverse_crossing_sides(q_side):
    """Either crossing recovers a critically damped transverse resonator."""
    frequency = np.linspace(1e6, 2e9, 20001)
    expected = np.array([4e6, 0.5, 0.8e9])
    impedance = iddefix.Impedances.Resonator_transverse_imp(frequency, *expected).real

    bounds = iddefix.SmartBoundDetermination(
        frequency,
        impedance,
        impedance_type="real",
        plane="transverse",
        minimum_peak_height=1e5,
        q_side=q_side,
    )

    assert bounds.N_resonators == 1
    np.testing.assert_allclose(bounds.parameterEstimates, expected, rtol=0.001)


def test_transverse_peak_with_known_q():
    """A manually selected peak and supplied Q recover the transverse parameters."""
    frequency = np.linspace(1e6, 2e9, 20001)
    expected = np.array([4e6, 2.0, 0.8e9])
    impedance = iddefix.Impedances.Resonator_transverse_imp(frequency, *expected).real
    peak_frequency = frequency[np.argmax(impedance)]

    bounds = iddefix.SmartBoundDetermination(
        frequency,
        impedance,
        impedance_type="real",
        plane="transverse",
        minimum_peak_height=impedance.max() + 1.0,
    )
    bounds.add_peak(peak_frequency, Q=expected[1])

    assert bounds.N_resonators == 1
    np.testing.assert_allclose(bounds.parameterEstimates, expected, rtol=0.001)


def test_transverse_absolute_warning(capsys):
    """Absolute transverse data warn and do not receive the real-part correction."""
    frequency = np.linspace(1e6, 2e9, 5001)
    impedance = np.abs(
        iddefix.Impedances.Resonator_transverse_imp(
            frequency, 4e6, 5.0, 0.8e9, wake_length=0.5
        )
    )
    options = dict(
        impedance_type="absolute",
        plane="transverse",
        minimum_peak_height=impedance.max() * 0.6,
    )
    full = iddefix.SmartBoundDetermination(frequency, impedance, **options)
    finite = iddefix.SmartBoundDetermination(
        frequency, impedance, wake_length=0.5, **options
    )

    assert "use impedance_type='real'" in capsys.readouterr().out
    np.testing.assert_allclose(finite.parameterEstimates, full.parameterEstimates)


# Wake-length correction
@pytest.mark.parametrize("plane", ["longitudinal", "transverse"])
def test_wake_length_automatic_estimates(plane):
    """Peak detection recovers a finite-wake resonator in either plane."""
    frequency = np.linspace(1e6, 2e9, 10001)
    expected = np.array([4e6, 0.8, 0.8e9])
    formula = getattr(iddefix.Impedances, f"Resonator_{plane}_imp")
    impedance = formula(frequency, *expected, wake_length=0.5).real

    bounds = iddefix.SmartBoundDetermination(
        frequency,
        impedance,
        impedance_type="real",
        plane=plane,
        wake_length=0.5,
        minimum_peak_height=impedance.max() * 0.6,
    )

    assert bounds.N_resonators == 1
    np.testing.assert_allclose(bounds.parameterEstimates, expected, rtol=0.04)


@pytest.mark.parametrize("plane", ["longitudinal", "transverse"])
@pytest.mark.parametrize("Q", [0.4, 0.5, 5.0])
def test_wake_length_known_q(plane, Q):
    """Supplied Q recovers Rs and the spectrum across damping regimes."""
    frequency = np.linspace(1e6, 2e9, 10001)
    expected = np.array([4e6, Q, 0.8e9])
    formula = getattr(iddefix.Impedances, f"Resonator_{plane}_imp")
    impedance = formula(frequency, *expected, wake_length=0.5).real
    peak_frequency = frequency[np.argmax(impedance)]

    bounds = iddefix.SmartBoundDetermination(
        frequency,
        impedance,
        impedance_type="real",
        plane=plane,
        wake_length=0.5,
        minimum_peak_height=impedance.max() + 1.0,
    )
    bounds.add_peak(peak_frequency, Q=Q)

    assert bounds.parameterEstimates[0] == pytest.approx(expected[0], rel=0.05)
    model = bounds.get_impedance_components()[0].real
    around_peak = (frequency > 0.5e9) & (frequency < 1.2e9)
    peak_scaled_error = (
        np.max(np.abs(model[around_peak] - impedance[around_peak])) / impedance.max()
    )
    assert peak_scaled_error < 0.1


def test_wake_length_longitudinal_absolute():
    """Finite-wake correction improves Rs estimated from the magnitude."""
    frequency = np.linspace(1e6, 2e9, 10001)
    expected = np.array([4e6, 5.0, 0.8e9])
    impedance = iddefix.Impedances.Resonator_longitudinal_imp(
        frequency, *expected, wake_length=0.5
    )
    magnitude = np.abs(impedance)
    peak_frequency = frequency[np.argmax(magnitude)]
    options = dict(
        impedance_type="absolute",
        minimum_peak_height=magnitude.max() + 1.0,
    )
    full = iddefix.SmartBoundDetermination(frequency, magnitude, **options)
    finite = iddefix.SmartBoundDetermination(
        frequency, magnitude, wake_length=0.5, **options
    )
    full.add_peak(peak_frequency, Q=expected[1])
    finite.add_peak(peak_frequency, Q=expected[1])

    assert abs(finite.parameterEstimates[0] - expected[0]) < abs(
        full.parameterEstimates[0] - expected[0]
    )
    assert finite.parameterEstimates[0] == pytest.approx(expected[0], rel=0.07)


# Inspection and table output
@pytest.mark.parametrize(
    ("show_bounds", "show_components"),
    [(False, False), (True, False), (True, True)],
)
def test_inspect(request, monkeypatch, show_bounds, show_components):
    """Plot options run for an analytical finite-wake transverse trace."""
    frequency = np.linspace(1e6, 2e9, 5001)
    expected = np.array([4e6, 0.8, 0.8e9])
    impedance = iddefix.Impedances.Resonator_transverse_imp(
        frequency, *expected, wake_length=0.5
    ).real
    bounds = iddefix.SmartBoundDetermination(
        frequency,
        impedance,
        impedance_type="real",
        plane="transverse",
        wake_length=0.5,
        minimum_peak_height=impedance.max() * 0.6,
    )

    if not request.config.getoption("--debug-plots"):
        monkeypatch.setattr(plt, "show", lambda: None)
    bounds.inspect(show_bounds=show_bounds, show_components=show_components)
    plt.close("all")


@pytest.mark.parametrize("to_markdown", [False, True])
def test_bounds_table_layout(capsys, to_markdown):
    """Computed SmartBounds use aligned text and Markdown table layouts."""
    bounds = _single_mode_bounds()

    bounds.to_table(to_markdown=to_markdown)
    output = capsys.readouterr().out
    rs_bounds, q_bounds, frequency_bounds = bounds.parameterBounds

    if to_markdown:
        expected = (
            f"| 1 | {rs_bounds[0]:.2f} to {rs_bounds[1]:.2f} | "
            f"{q_bounds[0]:.2f} to {q_bounds[1]:.2f} |"
        )
        assert expected in output
    else:
        lines = [
            line for line in output.splitlines() if "|" in line or line.startswith("-")
        ]
        assert len(lines) == 5
        assert len({len(line) for line in lines}) == 1
        assert f"{frequency_bounds[0]:.2e} to {frequency_bounds[1]:.2e}" in output


def test_bounds_table_in_notebook(monkeypatch):
    """Computed bound ranges use the shared monospace notebook display."""
    ipython = pytest.importorskip("IPython")
    ipython_display = pytest.importorskip("IPython.display")
    monkeypatch.setattr(
        ipython, "get_ipython", lambda: SimpleNamespace(kernel=object())
    )
    displayed = []
    monkeypatch.setattr(ipython_display, "display", displayed.append)
    bounds = _single_mode_bounds()

    bounds.to_table()

    assert len(displayed) == 1
    assert '<pre style="font-family: monospace;' in displayed[0].data
    frequency_bounds = bounds.parameterBounds[2]
    assert (
        f"{frequency_bounds[0]:.2e} to {frequency_bounds[1]:.2e}" in displayed[0].data
    )

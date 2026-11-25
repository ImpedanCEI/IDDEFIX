"""SmartBounds estimates checked against analytical resonator impedances."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.constants import c
from scipy.signal import find_peaks

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
def test_prominence_uses_trace_units():
    """Prominence is passed to SciPy in trace units; None leaves it unfiltered."""
    frequency = np.linspace(0.1e9, 0.9e9, 9)
    impedance = np.array([0, 100, 0, 0, 0.5, 0, 0, 4, 0])
    bounds = iddefix.SmartBoundDetermination(frequency, impedance, prominence=1)
    unfiltered = iddefix.SmartBoundDetermination(frequency, impedance)
    scaled = iddefix.SmartBoundDetermination(frequency, impedance * 0.01, prominence=1)

    np.testing.assert_array_equal(bounds.peaks, [1, 7])
    np.testing.assert_array_equal(unfiltered.peaks, [1, 4, 7])
    np.testing.assert_array_equal(scaled.peaks, [1])
    bounds.find(prominence=5)
    np.testing.assert_array_equal(bounds.peaks, [1])
    bounds.find(prominence=0)
    np.testing.assert_array_equal(bounds.peaks, [1, 4, 7])


def test_default_height_does_not_filter_negative_maxima():
    """An omitted height leaves signed local maxima eligible for detection."""
    frequency = np.linspace(0.1e9, 0.6e9, 6)
    impedance = np.array([-3, -1, -3, -5, -2, -5])

    bounds = iddefix.SmartBoundDetermination(frequency, impedance)

    np.testing.assert_array_equal(bounds.peaks, [1, 4])


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


def test_transverse_absolute_estimates():
    """Absolute transverse data do not receive the real-part correction."""
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

    np.testing.assert_allclose(finite.parameterEstimates, full.parameterEstimates)


# Hann peak detection
def test_hann_recovers_resonance_above_dc():
    """A spectrum starting above DC still recovers the isolated resonance."""
    frequency = np.linspace(1e9, 2e9, 2001)
    impedance = iddefix.Impedances.Resonator_longitudinal_imp(frequency, 100, 5, 1.5e9)
    shift = 0.3e9
    bounds = iddefix.SmartBoundDetermination(
        frequency,
        impedance,
        wake_length=c / shift,
        use_Hann=True,
    )
    assert bounds.N_resonators == 1
    resistance, quality, resonance = bounds.parameterEstimates
    assert resistance == pytest.approx(100, rel=0.06)
    assert quality == pytest.approx(5, rel=0.02)
    assert resonance == pytest.approx(1.5e9, rel=0.01)


def test_hann_uses_continuous_band():
    """Ignoring the detached low-frequency band avoids a spurious edge peak."""
    frequency = np.linspace(0.1e9, 2e9, 1901)
    impedance = np.cos(2 * np.pi * frequency / 0.28e9)
    bounds = iddefix.SmartBoundDetermination(
        frequency,
        impedance,
        wake_length=c / 0.3e9,
        use_Hann=True,
    )
    valid = np.isfinite(bounds.hann_trace)
    np.testing.assert_array_equal(valid, (frequency >= 0.25e9) & (frequency <= 1.85e9))
    estimates = np.asarray(bounds.parameterEstimates).reshape(-1, 3)
    np.testing.assert_allclose(estimates[:, 2], np.arange(1, 7) * 0.28e9, atol=1e6)


@pytest.mark.parametrize("plane", ["longitudinal", "transverse"])
@pytest.mark.parametrize("impedance_type", ["real", "absolute"])
def test_hann_trace_matches_windowed_wake(plane, impedance_type):
    """Shifted impedance agrees with an independently windowed wake FFT."""
    time = np.linspace(-0.4e-9, 16.6e-9, 1000)
    causal_time = np.maximum(time, 0.0)
    wake = np.exp(-causal_time / 8e-9) * np.sin(2 * np.pi * 1.1e9 * causal_time)
    frequency, impedance = iddefix.compute_fft(time, wake, fmax=5e9)
    if plane == "transverse":
        impedance *= 1j
    input_data = impedance if impedance_type == "absolute" else impedance.real
    bounds = iddefix.SmartBoundDetermination(
        frequency,
        input_data,
        plane=plane,
        wake_length=time[-1] * c,
        impedance_type=impedance_type,
        use_Hann=True,
    )
    hann_window = 0.5 + 0.5 * np.cos(np.pi * time / time[-1])
    _, windowed_impedance = iddefix.compute_fft(time, wake * hann_window, fmax=5e9)
    if plane == "transverse":
        windowed_impedance *= 1j
    expected = (
        np.abs(windowed_impedance)
        if impedance_type == "absolute"
        else windowed_impedance.real
    )

    valid = np.isfinite(bounds.hann_trace)
    assert valid[0]
    np.testing.assert_allclose(
        bounds.hann_trace[valid],
        expected[valid],
        atol=1e-4 * np.max(np.abs(expected)),
        rtol=0,
    )


def test_hann_detection_uses_original_impedance_for_estimates():
    """Finite-wake ripple peaks are suppressed without changing Rs or Q estimates."""
    frequency = np.linspace(0.98e9, 1.02e9, 20001)
    impedance = iddefix.Impedances.Resonator_longitudinal_imp(
        frequency, 100.0, 5000.0, 1e9, wake_length=300.0
    )
    options = dict(
        impedance_type="real", wake_length=300.0, prominence=0.5, use_Hann=True
    )
    bounds = iddefix.SmartBoundDetermination(
        frequency, impedance, minimum_peak_height=1.0, **options
    )
    real_bounds = iddefix.SmartBoundDetermination(
        frequency, impedance.real, minimum_peak_height=1.0, **options
    )
    unwindowed = iddefix.SmartBoundDetermination(
        frequency,
        impedance.real,
        impedance_type="real",
        minimum_peak_height=40.0,
        prominence=0.5,
    )

    assert len(find_peaks(impedance.real, height=1.0, prominence=0.5)[0]) > 1
    assert bounds.N_resonators == 1
    np.testing.assert_allclose(
        bounds.parameterEstimates, real_bounds.parameterEstimates
    )
    assert bounds.parameterEstimates[1] == pytest.approx(
        unwindowed.parameterEstimates[1]
    )


# Wake-length correction
@pytest.mark.parametrize("plane", ["longitudinal", "transverse"])
def test_wake_length_automatic_estimates(plane):
    """Hann detection finds a short-wake resonator in either plane."""
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
        minimum_peak_height=impedance.max() * 0.2,
        use_Hann=True,
    )

    assert bounds.N_resonators == 1
    np.testing.assert_allclose(bounds.parameterEstimates, expected, rtol=0.2)


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
@pytest.mark.parametrize("plane", ["longitudinal", "transverse"])
def test_inspect(request, monkeypatch, show_bounds, show_components, plane):
    """Finite-wake inspection plots show impedance units for the selected plane."""
    frequency = np.linspace(1e6, 2e9, 5001)
    expected = np.array([4e6, 0.8, 0.8e9])
    formula = getattr(iddefix.Impedances, f"Resonator_{plane}_imp")
    impedance = formula(frequency, *expected, wake_length=0.5).real
    bounds = iddefix.SmartBoundDetermination(
        frequency,
        impedance,
        impedance_type="real",
        plane=plane,
        wake_length=0.5,
        minimum_peak_height=impedance.max() * 0.2,
        use_Hann=True,
    )

    if not request.config.getoption("--debug-plots"):
        monkeypatch.setattr(plt, "show", lambda: None)
    bounds.inspect(show_bounds=show_bounds, show_components=show_components)
    units = r"[$\Omega$]" if plane == "longitudinal" else r"[$\Omega$/m]"
    assert plt.gcf().axes[0].get_ylabel() == "Impedance " + units
    plt.close("all")


@pytest.mark.parametrize(
    ("show_bounds", "show_components"), [(True, False), (False, True)]
)
def test_inspect_many_resonators(monkeypatch, show_bounds, show_components):
    """Inspect renders a model with ten resonators."""
    frequency = np.linspace(0.5e9, 10.5e9, 10001)
    impedance = sum(
        iddefix.Impedances.Resonator_longitudinal_imp(frequency, 100, 30, i * 1e9)
        for i in range(1, 11)
    )
    bounds = iddefix.SmartBoundDetermination(
        frequency, impedance.real, minimum_peak_height=20, prominence=20
    )
    monkeypatch.setattr(plt, "show", lambda: None)

    bounds.inspect(show_bounds=show_bounds, show_components=show_components)

    figure = plt.gcf()
    assert bounds.N_resonators == 10
    assert len(figure.axes) == 2
    assert figure.axes[1].get_ylabel() == "Resonator number"
    plt.close(figure)


@pytest.mark.parametrize("to_markdown", [False, True])
def test_bounds_table_layout(capsys, to_markdown):
    """Computed SmartBounds use aligned text and Markdown table layouts."""
    bounds = _single_mode_bounds()

    bounds.to_table(to_markdown=to_markdown)
    output = capsys.readouterr().out
    rs_bounds, q_bounds, frequency_bounds = bounds.parameterBounds

    if to_markdown:
        expected = (
            f"| 1 | {rs_bounds[0]:.4g} to {rs_bounds[1]:.4g} | "
            f"{q_bounds[0]:.4g} to {q_bounds[1]:.4g} |"
        )
        assert expected in output
    else:
        lines = [
            line for line in output.splitlines() if "|" in line or line.startswith("-")
        ]
        assert len(lines) == 5
        assert len({len(line) for line in lines}) == 1
        assert f"{rs_bounds[0]:.4g} to {rs_bounds[1]:.4g}" in output
        assert f"{frequency_bounds[0]:.2e} to {frequency_bounds[1]:.2e}" in output


@pytest.mark.parametrize("to_markdown", [False, True])
@pytest.mark.parametrize("q_bounds", [(0.00123456, 0.00987654), (0.0143298, 0.0198765)])
def test_bounds_table_displays_small_q_values(capsys, to_markdown, q_bounds):
    bounds = _single_mode_bounds()
    bounds.to_table(
        parameterBounds=[(1.0, 2.0), q_bounds, (1e6, 2e6)],
        to_markdown=to_markdown,
    )

    output = capsys.readouterr().out
    assert "1 to 2" in output
    assert f"{q_bounds[0]:.4g} to {q_bounds[1]:.4g}" in output


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
    assert "monospace" in displayed[0].data
    frequency_bounds = bounds.parameterBounds[2]
    assert (
        f"{frequency_bounds[0]:.2e} to {frequency_bounds[1]:.2e}" in displayed[0].data
    )


@pytest.mark.parametrize("numbers", [2, [1, 3, 1]])
def test_remove_peak(numbers):
    """Removal uses original table numbers and retains explicit Q estimates."""
    frequency = np.linspace(0.1e9, 4e9, 4001)
    impedance = iddefix.Impedances.n_Resonator_longitudinal_imp(
        frequency,
        [100, 20, 1e9, 80, 20, 2e9, 60, 20, 3e9],
    ).real
    bounds = iddefix.SmartBoundDetermination(
        frequency, impedance, minimum_peak_height=200
    )
    for fr, q in [(1e9, 10), (2e9, 15), (3e9, 25)]:
        bounds.add_peak(fr, Q=q)
    before = np.asarray(bounds.parameterEstimates).reshape(-1, 3).copy()
    keep = np.delete(np.arange(3), np.unique(np.atleast_1d(numbers)) - 1)
    bounds.remove_peak(numbers)
    assert bounds.N_resonators == len(keep)
    np.testing.assert_allclose(
        np.asarray(bounds.parameterEstimates).reshape(-1, 3), before[keep]
    )
    assert len(bounds.peaks) == len(bounds.q_sides_used) == len(keep)


@pytest.mark.parametrize("numbers", [0, 2, True, 1.5, [1]])
def test_remove_peak_invalid(numbers):
    """Invalid removal leaves the completed estimate intact."""
    bounds = _single_mode_bounds()
    before = bounds.parameterEstimates.copy()
    with pytest.raises(ValueError):
        bounds.remove_peak(numbers)
    np.testing.assert_array_equal(bounds.parameterEstimates, before)
    assert bounds.N_resonators == 1


@pytest.mark.parametrize("plane", ["longitudinal", "transverse"])
def test_interactive_impedance_units(monkeypatch, plane):
    """The interactive peak picker uses the same units as impedance inspection."""
    frequency = np.linspace(0.5e9, 1.5e9, 101)
    formula = getattr(iddefix.Impedances, f"Resonator_{plane}_imp")
    impedance = formula(frequency, 100, 8, 1e9)
    monkeypatch.setattr("matplotlib.get_backend", lambda: "agg")
    monkeypatch.setattr(plt, "show", lambda **kwargs: None)
    iddefix.SmartBoundDetermination(frequency, impedance, plane=plane, interactive=True)
    figure = plt.gcf()
    units = r"[$\Omega$]" if plane == "longitudinal" else r"[$\Omega$/m]"
    assert figure.axes[0].get_ylabel() == "Impedance " + units
    plt.close(figure)

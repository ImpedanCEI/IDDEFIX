import numpy as np

import iddefix


def _synthetic_two_resonances():
    """Generate a simple spectrum with two clear resonance peaks.

    The peaks are Gaussian-shaped and well separated so that
    ``scipy.signal.find_peaks`` reliably detects two maxima.
    """

    frequency = np.linspace(0.0, 10.0, 2001)
    peak1 = 10.0 * np.exp(-((frequency - 3.0) ** 2) / (2 * 0.1**2))
    peak2 = 5.0 * np.exp(-((frequency - 7.0) ** 2) / (2 * 0.15**2))
    impedance = peak1 + peak2
    return frequency, impedance


def test_smart_bounds_detect_two_resonances():
    """SmartBoundDetermination should detect two resonances and build bounds."""

    frequency, impedance = _synthetic_two_resonances()
    sbd = iddefix.SmartBoundDetermination(
        frequency,
        impedance,
        minimum_peak_height=1.0,
    )

    # Two peaks detected
    assert sbd.peaks is not None
    assert len(sbd.peaks) == 2

    # Each resonance contributes three tuples: (Rs_bounds, Q_bounds, fres_bounds)
    assert len(sbd.parameterBounds) == 3 * 2
    assert sbd.N_resonators == 2

    # Basic sanity on bounds: ordering and positivity where expected
    for i in range(int(sbd.N_resonators)):
        rs_min, rs_max = sbd.parameterBounds[3 * i]
        q_min, q_max = sbd.parameterBounds[3 * i + 1]
        f_min, f_max = sbd.parameterBounds[3 * i + 2]

        assert rs_min > 0
        assert rs_max > rs_min
        assert q_min > 0
        assert q_max > q_min
        assert f_min < f_max


def test_smart_bounds_respect_minimum_peak_height():
    """Raising the minimum_peak_height should reduce the number of resonances."""

    frequency, impedance = _synthetic_two_resonances()
    sbd = iddefix.SmartBoundDetermination(
        frequency,
        impedance,
        minimum_peak_height=1.0,
    )

    assert len(sbd.peaks) == 2

    # Now require a higher peak height so only the tallest peak remains
    sbd.find(minimum_peak_height=7.0)

    assert len(sbd.peaks) == 1
    assert len(sbd.parameterBounds) == 3 * 1
    assert sbd.N_resonators == 1

    rs_min, rs_max = sbd.parameterBounds[0]
    assert rs_min > 0
    assert rs_max > rs_min


def test_smart_bounds_to_table_formats(capsys):
    """to_table should print both ASCII and Markdown representations."""

    frequency, impedance = _synthetic_two_resonances()
    sbd = iddefix.SmartBoundDetermination(
        frequency,
        impedance,
        minimum_peak_height=1.0,
    )

    # ASCII table
    sbd.to_table(to_markdown=False)
    captured_ascii = capsys.readouterr().out
    assert "Resonator" in captured_ascii
    assert "Rs [Ohm/m or Ohm]" in captured_ascii

    # Markdown table
    sbd.to_table(to_markdown=True)
    captured_md = capsys.readouterr().out
    assert "| Resonator |" in captured_md
    assert "Rs [Ohm/m or Ohm]" in captured_md

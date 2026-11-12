"""Linear baseline removal for impedance, wake, and wake-potential data."""

from numbers import Real

import numpy as np
import numpy.typing as npt
from scipy.constants import c

ArrayLike = npt.ArrayLike


def linear_baseline_impedance(
    frequency: ArrayLike,
    slope: float,
    wake_length: float,
    plane: str = "longitudinal",
    sigma: float | None = None,
) -> np.ndarray:
    """Transform a finite ``slope * time`` ramp to impedance."""

    frequency = np.asarray(frequency)
    T = wake_length / c  # Truncation time
    x = -2j * np.pi * frequency * T  # Fourier phase -j*2pifT

    # Evaluate integral
    integral = np.empty_like(x, dtype=complex)
    small = np.abs(x) < 1e-3
    xs = x[small]  # Small -> Taylor expansion
    integral[small] = 0.5 + xs / 3 + xs**2 / 8 + xs**3 / 30 + xs**4 / 144
    xl = x[~small]  # Not small -> Analytic expresion
    integral[~small] = (1 + np.exp(xl) * (xl - 1)) / xl**2
    baseline = c * slope * T**2 * integral

    # Remove baseline amplification from deconvolution
    if sigma is not None:
        bunch_spectrum = np.exp(-0.5 * (2 * np.pi * frequency * sigma) ** 2)
        if np.any(bunch_spectrum == 0):
            raise ValueError("Gaussian bunch spectrum vanishes on this grid")
        baseline = baseline / bunch_spectrum

    return 1j * baseline if plane == "transverse" else baseline


def calculate_baseline_slope(
    x_data: ArrayLike,
    y_data: ArrayLike,
    data_type: str,
    *,
    wake_length: float | None = None,
    plane: str = "longitudinal",
    sigma: float | None = None,
    fit_interval: tuple[float, float] | None = None,
) -> float:
    """Fit a real ramp slope to an impedance spectrum or a wake tail."""

    x = np.asarray(x_data)
    y = np.asarray(y_data)
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError("x_data and y_data must be paired one-dimensional arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x_data and y_data must be finite")

    if data_type == "impedance":
        # Check input params
        if plane not in ("longitudinal", "transverse"):
            raise ValueError("plane must be 'longitudinal' or 'transverse'")
        if wake_length is None or not np.isfinite(wake_length) or wake_length <= 0:
            raise ValueError("wake_length must be a finite positive value")
        if sigma is not None and (not np.isfinite(sigma) or sigma <= 0):
            raise ValueError("sigma must be a finite positive value")
        # Retrieve baseline kernel
        kernel = linear_baseline_impedance(x, 1.0, wake_length, plane, sigma)
        fit_kernel = kernel if np.iscomplexobj(y) else kernel.real
        # Least squares fit
        norm = np.vdot(fit_kernel, fit_kernel).real
        if norm == 0 or not np.isfinite(norm):
            raise ValueError("Cannot fit baseline on this frequency grid")
        slope = float(np.vdot(fit_kernel, y).real / norm)
        return slope

    if data_type not in ("wake", "wake function", "wake potential"):
        raise ValueError(
            "data_type must be impedance, wake function, or wake potential"
        )

    # Fitting to wake or wake potential within the interval
    if fit_interval is None:
        selected = x >= 0.8 * x[-1]  # last 20%
    else:
        start, stop = fit_interval
        if not np.isfinite(start) or not np.isfinite(stop) or start >= stop:
            raise ValueError("fit_interval must contain increasing finite times")
        selected = (x >= start) & (x <= stop)
    selected &= x >= 0
    if np.count_nonzero(selected) < 2 or np.ptp(x[selected]) == 0:
        raise ValueError("Wake baseline fit requires at least two tail samples")
    # Least squares fit
    time = x[selected] - np.mean(x[selected])
    values = y[selected].real - np.mean(y[selected].real)
    slope = float(np.dot(time, values) / np.dot(time, time))
    return slope


def remove_baseline(
    x_data: ArrayLike,
    y_data: ArrayLike,
    data_type: str,
    *,
    slope: float | None = None,
    wake_length: float | None = None,
    plane: str = "longitudinal",
    sigma: float | None = None,
    fit_interval: tuple[float, float] | None = None,
) -> tuple[np.ndarray, float]:
    """Return baseline-corrected samples and the removed slope."""
    # Retrieve slope
    if slope is None:
        slope = calculate_baseline_slope(
            x_data,
            y_data,
            data_type,
            wake_length=wake_length,
            plane=plane,
            sigma=sigma,
            fit_interval=fit_interval,
        )
    elif not isinstance(slope, Real) or not np.isfinite(slope):
        raise ValueError("slope must be a finite real number")

    # Correct y_data
    slope = float(slope)
    x = np.asarray(x_data)
    y = np.asarray(y_data)
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError("x_data and y_data must be paired one-dimensional arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x_data and y_data must be finite")

    # Correct complex or real impedance
    if data_type == "impedance":
        if wake_length is None or not np.isfinite(wake_length) or wake_length <= 0:
            raise ValueError("wake_length must be a finite positive value")
        if sigma is not None and (not np.isfinite(sigma) or sigma <= 0):
            raise ValueError("sigma must be a finite positive value")
        baseline = linear_baseline_impedance(x, slope, wake_length, plane, sigma)
        corrected = y - (baseline if np.iscomplexobj(y) else baseline.real)

    # Correct real wake or wake potential
    elif data_type in ("wake", "wake function", "wake potential"):
        corrected = y - slope * np.maximum(x, 0)
    else:
        raise ValueError(
            "data_type must be impedance, wake function, or wake potential"
        )
    print(f"[*] Removed linear baseline from {data_type} data (slope={slope:.6g})")
    return corrected, slope

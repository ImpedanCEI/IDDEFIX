from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.constants import c as c_light

ArrayLike = npt.ArrayLike
NumericArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
ParameterDict = dict[int, ArrayLike]


def pars_to_dict(pars: ArrayLike) -> dict[int, ArrayLike]:
    """Group a flat resonator parameter sequence into triples.

    Parameters
    ----------
    pars : array_like
        One-dimensional sequence ordered as consecutive
        ``(Rs, Q, resonant_frequency)`` triples.

    Returns
    -------
    dict[int, array_like]
        Mapping from zero-based resonator index to each three-value slice.

    Raises
    ------
    ValueError
        If the number of parameters is not divisible by three.
    """

    if len(pars) % 3 != 0:
        raise ValueError("Input list length must be a multiple of 3")

    grouped_parameters = {}
    for i in range(0, len(pars), 3):
        grouped_parameters[i // 3] = pars[i : i + 3]

    return grouped_parameters


def compute_fft(
    data_time: ArrayLike,
    data_wake: ArrayLike,
    fmax: float = 3e9,
    samples: int = 1001,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute impedance from a uniformly sampled wake using an FFT.

    The wake is zero-padded, transformed, restricted to nonnegative frequencies
    below ``fmax``, and corrected for the time of the first sample. The
    transform includes the spatial step ``c * dt`` in its normalization.

    Parameters
    ----------
    data_time : array-like
        One-dimensional, uniformly spaced time samples in seconds. Must contain
        at least two samples.
    data_wake : array-like
        Wake values corresponding to ``data_time``. Must have the same length.
    fmax : float, optional
        Upper frequency limit in hertz. Returned frequencies satisfy
        ``0 <= f < fmax``. Defaults to 3 GHz.
    samples : int, optional
        Controls the number of zero-padded FFT samples and therefore the
        frequency spacing. Defaults to 1001.

    Returns
    -------
    f : ndarray
        One-dimensional frequency array in hertz, from zero up to but excluding
        ``fmax``.
    Z : ndarray
        Complex impedance values corresponding to ``f``.

    Notes
    -----
    ``data_time`` must be evenly spaced. The phase factor
    ``exp(-2j * pi * f * data_time[0])`` accounts for a nonzero first sample.
    The wake is transformed with NumPy's FFT and scaled by the spatial step
    ``c * dt``.

    Examples
    --------
    >>> import numpy as np
    >>> time = np.linspace(0, 10e-9, 1001)  # seconds
    >>> wake = np.sin(2 * np.pi * 1e9 * time)
    >>> f, Z = compute_fft(time, wake, fmax=2e9, samples=500)
    >>> f.shape == Z.shape
    True
    """

    ds = (data_time[1] - data_time[0]) * c_light
    N = int((c_light / ds) // fmax * samples)
    Z = np.fft.fft(data_wake, n=N)
    f = np.fft.fftfreq(len(Z), ds / c_light)

    # Mask invalid frequencies
    mask = np.logical_and(f >= 0, f < fmax)
    Z = Z[mask] * ds
    f = f[mask]
    Z *= np.exp(-2j * np.pi * f * data_time[0])

    return f, Z


def compute_convolution(
    data_time: ArrayLike,
    data_wake: ArrayLike,
    sigma: float,
    kernel: str = "numpy",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the full discrete convolution of a wake with a Gaussian bunch.

    The Gaussian is sampled on a zero-centered lag grid with the same spacing
    as ``data_time``. Its samples are normalized by their sum before
    convolution, so the operation forms a weighted average of the wake.

    Parameters
    ----------
    data_time : array_like
        One-dimensional, uniformly spaced time samples in seconds. Must contain
        at least two samples.
    data_wake : array_like
        One-dimensional wake samples corresponding to ``data_time``. Must
        have the same length.
    sigma : float
        RMS width of the Gaussian bunch profile in seconds.
    kernel : {'numpy', 'scipy', 'scipy_fft'}, optional
        Backend for the full convolution: ``'numpy'`` uses
        :func:`numpy.convolve`, ``'scipy'`` uses ``scipy.signal.convolve``, and
        ``'scipy_fft'`` uses ``scipy.signal.fftconvolve``. Defaults to
        ``'numpy'``.

    Returns
    -------
    t_convolved : ndarray
        Time axis for the full convolution. It starts at the first input time
        plus the first Gaussian lag and has ``len(data_time) + len(data_wake) - 1``
        samples, assuming ``data_wake`` and ``data_time`` have equal lengths.
    wake_convolved : ndarray
        Wake samples after convolution with the normalized Gaussian bunch.

    Notes
    -----
    ``data_time`` is expected to be uniformly spaced. Both the wake and
    Gaussian lag grid use that spacing, and the returned time axis includes
    the starting times of both arrays. All backends return the full
    convolution.
    """

    if kernel.lower() == "numpy":
        from numpy import convolve
    elif kernel.lower() == "scipy":
        from scipy.signal import convolve
    elif kernel.lower() == "scipy_fft":
        from scipy.signal import fftconvolve as convolve

    data_time = np.asarray(data_time)
    dt = data_time[1] - data_time[0]
    kernel_time = (np.arange(len(data_time)) - (len(data_time) - 1) / 2) * dt
    lambdat = gaussian_bunch(kernel_time, sigma)
    normalization = np.sum(lambdat)

    if not np.isfinite(normalization) or normalization == 0:
        # Keep a representable peak when all physical Gaussian samples
        # underflow (or overflow) before their normalization cancels the scale.
        gaussian_exponent = -0.5 * (kernel_time / sigma) ** 2
        lambdat = np.exp(gaussian_exponent - np.max(gaussian_exponent))
        normalization = np.sum(lambdat)

    # Perform the convolution
    wake_convolved = convolve(data_wake, lambdat) / normalization
    t_convolved = data_time[0] + kernel_time[0] + np.arange(len(wake_convolved)) * dt

    return t_convolved, wake_convolved


def compute_deconvolution(
    data_time: ArrayLike,
    data_wake_potential: ArrayLike,
    sigma: float,
    fmax: float | None = None,
    samples: int = 1001,
    rolloff: float = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate impedance by deconvolving a Gaussian bunch from a potential.

    The potential and Gaussian profile are transformed with an FFT, and their
    spectra are divided. Only nonnegative frequencies below the selected
    cutoff are returned.

    Parameters
    ----------
    data_time : array_like
        One-dimensional, uniformly spaced time samples in seconds. The axis
        should cover the potential on both sides of the bunch center.
    data_wake_potential : array_like
        Wake-potential samples corresponding to ``data_time``, in the
        normalization expected by IDDEFIX. Keep the potential and impedance
        units consistent with the time and spatial scaling used by the model.
    sigma : float
        RMS bunch length (standard deviation) of the Gaussian profile in
        seconds.
    fmax : float, optional
        Frequency limit in hertz. If omitted, it is derived as
        ``1 / (3 * sigma)``.
    samples : int, optional
        Controls the zero-padded FFT length and frequency spacing. Defaults to
        1001.
    rolloff : float, optional
        Nonnegative taper width as a fraction of ``fmax``. Defaults to zero,
        which applies no taper and returns frequencies below ``fmax``. A
        positive value ``r`` tapers the impedance from ``fmax`` to
        ``(1 + r) * fmax``; for example, ``rolloff=0.5`` reaches zero at
        ``1.5 * fmax``.

    Returns
    -------
    f : ndarray
        One-dimensional nonnegative frequency array in hertz. Frequencies are
        below ``fmax`` when ``rolloff=0`` and below ``(1 + rolloff) * fmax``
        when the taper is enabled.
    Z : ndarray
        Complex impedance values corresponding to ``f``.

    Notes
    -----
    For positive ``rolloff`` (denoted by ``r``), the taper factor is 1 through
    ``fmax`` and then follows ``0.5 * (1 + cos(pi * (f - fmax) / (r * fmax)))``
    until it reaches zero at ``(1 + r) * fmax``. Frequencies at or above that
    endpoint are excluded from the returned array. Negative frequencies are
    discarded.

    Raises
    ------
    ValueError
        If ``rolloff`` is negative.

    ``data_time`` is assumed to be uniformly spaced. The FFT uses the spatial
    step ``c * dt`` in its frequency scaling.
    """

    if fmax is None:
        fmax = 1 / (3 * sigma)
    if rolloff < 0:
        raise ValueError("rolloff must be nonnegative")
    cutoff = fmax * (1 + rolloff)

    ds = (data_time[1] - data_time[0]) * c_light
    N = int((c_light / ds) // cutoff * samples)
    N = max(len(data_time), N)

    # Analytical gaussian with given sigma
    lambdat = gaussian_bunch(data_time, sigma)

    Z = np.fft.fft(data_wake_potential, n=N)
    lambdaf = np.fft.fft(lambdat, n=N)
    f = np.fft.fftfreq(len(Z), ds / c_light)

    # Mask invalid frequencies
    mask = np.logical_and(f >= 0, f < cutoff)
    Z = Z[mask] / lambdaf[mask]
    f = f[mask]
    if rolloff > 0:
        tapered = f > fmax
        Z[tapered] *= 0.5 * (1 + np.cos(np.pi * (f[tapered] - fmax) / (cutoff - fmax)))

    return f, Z


def gaussian_bunch(time: ArrayLike, sigma: float) -> np.ndarray:
    """Evaluate a normalized Gaussian bunch profile on a time grid.

    Parameters
    ----------
    time : array_like
        Time samples in seconds, measured from the bunch center.
    sigma : float
        RMS bunch length (standard deviation) in seconds; must be positive.

    Returns
    -------
    ndarray
        Gaussian line-density profile evaluated at ``time``, scaled by the
        speed of light as used by IDDEFIX's time-to-distance convention.
    """
    # Analytical gaussian with given sigma
    return (
        1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(time**2) / (2 * sigma**2)) / c_light
    )


def gaussian_bunch_spectrum(
    time: ArrayLike,
    sigma: float,
    fmax: float,
    samples: int = 1001,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the positive-frequency FFT spectrum of a Gaussian bunch.

    Parameters
    ----------
    time : array_like
        One-dimensional, uniformly spaced time samples in seconds.
    sigma : float
        RMS bunch length (standard deviation) in seconds.
    fmax : float
        Upper frequency limit in hertz; frequencies at or above this limit
        are omitted.
    samples : int, optional
        Controls the zero-padded FFT length and frequency spacing. Defaults to
        1001.

    Returns
    -------
    f : ndarray
        Nonnegative frequency samples below ``fmax``, in hertz.
    lambdaf : ndarray
        FFT values of the sampled Gaussian profile at ``f``.

    Notes
    -----
    ``time`` must be uniformly spaced. The returned values are the discrete
    FFT of ``gaussian_bunch(time, sigma)`` with a phase correction for
    ``time[0]``. They are not additionally scaled by the time step.
    """
    # Analytical gaussian with given sigma
    lambdat = gaussian_bunch(time, sigma)

    # Perform FFT
    ds = (time[1] - time[0]) * c_light
    N = int((c_light / ds) // fmax * samples)
    lambdaf = np.fft.fft(lambdat, n=N)
    f = np.fft.fftfreq(len(lambdaf), ds / c_light)

    # Mask invalid frequencies
    mask = np.logical_and(f >= 0, f < fmax)
    f = f[mask]
    lambdaf = lambdaf[mask] * np.exp(-2j * np.pi * f * time[0])
    return f, lambdaf


# Needed for adaptive neffint routines
def interpolation_error_abs(
    func_output: ArrayLike,
    interpolant_output: ArrayLike,
) -> np.ndarray:
    """Return the absolute interpolation error element by element.

    The inputs must have matching shapes. The leading axis represents samples;
    no reduction is performed over any trailing output dimensions.
    """
    return np.abs(func_output - interpolant_output)


def interpolation_error_rms(
    func_output: ArrayLike,
    interpolant_output: ArrayLike,
) -> np.ndarray:
    """Return an RMS interpolation error for each leading-axis sample.

    The inputs must have matching shapes. For one-dimensional inputs, each
    sample has a scalar output and the absolute error is returned. For
    multidimensional outputs, the RMS magnitude is computed over all axes
    after the leading sample axis. Complex errors use their magnitudes.
    """
    error = np.asarray(interpolant_output) - np.asarray(func_output)
    if error.ndim == 1:
        return np.abs(error)
    axes = tuple(range(1, error.ndim))
    return np.sqrt(np.mean(np.abs(error) ** 2, axis=axes))


def compute_ineffint(
    data_freq: ArrayLike,
    data_impedance: ArrayLike,
    times: ArrayLike = np.linspace(1e-11, 50e-9, 1000),
    adaptative: bool = True,
    interpolation: str = "linear",
    plane: str = "longitudinal",
    error: str = "Abs",
    enforce_causality: bool = False,
) -> tuple[ArrayLike, np.ndarray]:
    """Compute a wake from impedance using Neffint's Fourier integration.

    With ``adaptative=True``, Neffint refines the supplied frequency grid
    before evaluating the transform. Set it to ``False`` to integrate on the
    supplied grid directly. The returned wake is real-valued and follows the
    longitudinal or transverse convention selected by ``plane``.

    Parameters
    ----------
    data_freq : array_like
        One-dimensional, increasing frequency samples in hertz.
    data_impedance : array_like
        Complex impedance samples corresponding to ``data_freq``. Use units
        consistent with the selected plane (typically ohms longitudinally or
        ohms per metre transversely).
    times : array_like, optional
        Time samples in seconds at which to evaluate the wake. Defaults to
        1000 points from 10 ps to 50 ns.
    adaptative : bool, optional
        Whether to refine the supplied grid with Neffint before integration.
        The parameter retains its historical spelling. Defaults to ``True``.
    interpolation : {'linear', 'pchip'}, optional
        Interpolation method used by Neffint for Fourier integration.
        Defaults to ``'linear'``.
    plane : {'longitudinal', 'transverse'}, optional
        Impedance and wake convention to use. Defaults to ``'longitudinal'``.
    error : {'abs', 'rms'}, optional
        Error metric for adaptive refinement; capitalization is ignored.
        Defaults to ``'Abs'``.
    enforce_causality : bool, optional
        If ``True``, set returned wake samples at ``times < 0`` to zero. The
        sample at ``t = 0`` is retained. If ``False``, print a yellow warning
        when any negative-time wake sample is nonzero. Defaults to ``False``.

    Returns
    -------
    times : array_like
        The requested time samples.
    wake : ndarray
        Real-valued wake samples corresponding to ``times``.

    Raises
    ------
    ImportError
        If Neffint is not installed.

    Notes
    -----
    Adaptive refinement adds frequency samples by interpolating between the
    supplied data. It cannot recover impedance features absent from the input
    range or sampling. Causality enforcement clips the computed result and
    therefore changes the direct inverse-transform result.
    """
    try:
        import neffint
    except Exception:
        raise ImportError(
            "This function uses the python package `neffint` \n \
                           > pip install neffint"
        )
    from scipy.interpolate import interp1d

    if not adaptative:
        wake = neffint.fourier_integral_fixed_sampling(
            times=times,
            frequencies=data_freq,
            func_values=data_impedance,
            pos_inf_correction_term=False,
            neg_inf_correction_term=False,
            interpolation=interpolation,  # `pchip gives artificial imag. baseline`
        )

    # Using adaptative freq. refining
    if adaptative:
        if error.lower() == "abs":
            interpolation_error_norm = interpolation_error_abs
        elif error.lower() == "rms":
            interpolation_error_norm = interpolation_error_rms

        func = interp1d(
            data_freq, data_impedance, kind="linear", fill_value="extrapolate"
        )

        frequencies, impedance = neffint.improve_frequency_range(
            initial_frequencies=data_freq,
            func=func,
            interpolation_error_norm=interpolation_error_norm,
            absolute_integral_tolerance=1e0,  # The absolute tolerance the algorithm tries to get the error below
            step_towards_inf_factor=2,  # The multiplicative step size used to scan for higher and lower frequencies to add
            bisection_mode_condition=None,  # None (the default) here gives only logarithmic bisection when adding internal points
            max_iterations=10000,
        )

        wake = neffint.fourier_integral_fixed_sampling(
            times=times,
            frequencies=frequencies,
            func_values=impedance,
            pos_inf_correction_term=False,
            neg_inf_correction_term=False,
            interpolation=interpolation,
        )

    # Normalize
    wake = np.conjugate(wake) / np.pi
    if plane == "longitudinal":
        pass
    elif plane == "transverse":
        wake *= 1j
    wake = np.asarray(wake.real).copy()

    # Enforce causality
    negative_time = np.asarray(times) < 0
    if enforce_causality:
        wake[negative_time] = 0
    elif np.any(wake[negative_time] != 0):
        print(
            "\033[93m[!] Warning: Wake is nonzero for t < 0; "
            "pass enforce_causality=True to zero those samples.\033[0m"
        )

    return times, wake


def compute_neffint(
    data_time: ArrayLike,
    data_wake: ArrayLike,
    frequencies: ArrayLike = np.linspace(1, 5e9, 1000),
    adaptative: bool = True,
    interpolation: str = "linear",
    plane: str = "longitudinal",
    error: str = "abs",
) -> tuple[ArrayLike, np.ndarray]:
    """Compute impedance from a wake using Neffint's Fourier integration.

    With ``adaptative=True``, Neffint refines the supplied time grid before
    evaluating the transform. Set it to ``False`` to integrate on the supplied
    grid directly.

    Parameters
    ----------
    data_time : array_like
        One-dimensional, increasing time samples in seconds.
    data_wake : array_like
        Real or complex wake samples corresponding to ``data_time``.
    frequencies : array_like, optional
        Frequency samples in hertz at which to evaluate the impedance.
        Defaults to 1000 points from 1 Hz to 5 GHz.
    adaptative : bool, optional
        Whether to refine the supplied time grid with Neffint before
        integration. The parameter retains its historical spelling. Defaults
        to ``True``.
    interpolation : {'linear', 'pchip'}, optional
        Interpolation method used by Neffint for Fourier integration.
        Defaults to ``'linear'``.
    plane : {'longitudinal', 'transverse'}, optional
        Wake and impedance convention to use. Defaults to ``'longitudinal'``.
    error : {'abs', 'rms'}, optional
        Error metric for adaptive refinement; capitalization is ignored.
        Defaults to ``'abs'``.

    Returns
    -------
    frequencies : array_like
        The requested frequency samples in hertz.
    impedance : ndarray
        Complex impedance values corresponding to ``frequencies``.

    Raises
    ------
    ImportError
        If Neffint is not installed.

    Notes
    -----
    Adaptive refinement interpolates between supplied wake samples. It cannot
    recover time-domain details absent from the input range or sampling.
    """
    try:
        import neffint
    except Exception:
        raise ImportError(
            "This function uses the python package `neffint` \n \
                           > pip install neffint"
        )
    from scipy.interpolate import interp1d

    if not adaptative:
        impedance = neffint.fourier_integral_fixed_sampling(
            times=frequencies,
            frequencies=data_time,
            func_values=data_wake,
            pos_inf_correction_term=False,
            neg_inf_correction_term=False,
            interpolation=interpolation,  # `pchip gives artificial imag. baseline`
        )

    # Using adaptative freq. refining
    if adaptative:
        if error.lower() == "abs":
            interpolation_error_norm = interpolation_error_abs
        elif error.lower() == "rms":
            interpolation_error_norm = interpolation_error_rms

        func = interp1d(data_time, data_wake, kind="linear", fill_value="extrapolate")

        times, wake = neffint.improve_frequency_range(
            initial_frequencies=data_time,
            func=func,
            interpolation_error_norm=interpolation_error_norm,
            absolute_integral_tolerance=1e0,  # The absolute tolerance the algorithm tries to get the error below
            step_towards_inf_factor=2,  # The multiplicative step size used to scan for higher and lower frequencies to add
            bisection_mode_condition=None,  # None (the default) here gives only logarithmic bisection when adding internal points
            max_iterations=10000,
        )

        impedance = neffint.fourier_integral_fixed_sampling(
            times=frequencies,
            frequencies=times,
            func_values=wake,
            pos_inf_correction_term=False,
            neg_inf_correction_term=False,
            interpolation=interpolation,
        )

    # Normalize
    impedance = np.conjugate(impedance) / np.pi / 2

    if plane == "longitudinal":
        pass
    elif plane == "transverse":
        impedance *= 1j

    return frequencies, impedance

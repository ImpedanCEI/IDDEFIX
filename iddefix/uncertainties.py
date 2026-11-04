from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.linalg import svd
from scipy.optimize._numdiff import approx_derivative

from .utils import pars_to_dict

ArrayLike = npt.ArrayLike
FitCallable = Callable[[ArrayLike, dict[int, ArrayLike]], ArrayLike]
JointSources = dict[str, tuple[ArrayLike, ArrayLike, FitCallable]]


def StackedResiduals(
    parameters: ArrayLike,
    x: ArrayLike | None,
    fitFunction: FitCallable | None,
    y: ArrayLike | None,
    sources: JointSources | None = None,
) -> np.ndarray:
    """Return real residuals, stacking imaginary residuals for complex data.

    This function takes the parameters obtained from the Differential Evolution or
    minimization algorithm and evaluate the stacked residuals at that solution.

    Args:
        parameters: Array of parameters used by the fit_function.
        x: Array of x values for the data.
        fitFunction: Function that takes parameters and x values as input and
            returns predicted y values (including real and imaginary parts).
        y: Array of y values for the data (including real and imaginary parts).
        sources: Optional joint datasets. When supplied, stack each source's
            residuals divided by its measured RMS and square root of sample count.
            An all-zero measured trace uses RMS scale 1.

    Returns:
        residuals: The stacked residuals for a given solution.
    """

    if sources is not None:
        residuals = []
        for source_x, source_y, source_model in sources.values():
            source_y = np.asarray(source_y)
            residual = StackedResiduals(parameters, source_x, source_model, source_y)
            # Scale before squaring to avoid overflow for large measured units.
            peak = float(np.max(np.abs(source_y)))
            scale = (
                peak * np.sqrt(np.mean(np.abs(source_y / peak) ** 2)) if peak else 1.0
            )
            residuals.append(residual / scale / np.sqrt(source_y.size))
        return np.concatenate(residuals)

    y = np.asarray(y)
    grouped_parameters = pars_to_dict(parameters)
    predicted_y = fitFunction(x, grouped_parameters)

    real_residuals = y.real - predicted_y.real
    if not np.iscomplex(y).any():
        return real_residuals

    return np.concatenate([real_residuals, y.imag - predicted_y.imag])


def build_jacobian(
    parameters: ArrayLike,
    fitFunction: FitCallable,
    x: ArrayLike,
    y: ArrayLike,
    parameterBounds: ArrayLike | None = None,
    sources: JointSources | None = None,
) -> np.ndarray:
    """Build the Jacobian of the system.

    This function takes the parameters obtained from the Differential Evolution or
    minimization algorithm and evaluate the Jacobian around the solution. The Jacobian
    is computed using the 3-point method, which is the most accurate method.

    The Jacobian can also be build using the function scipy.optimize.approx_fprime
    However, it only allows using the 2-point method, which is less accurate.

    Args:
        parameters: Array of parameters used by the fit_function.
        fitFunction: Function that takes parameters and x values as input and
            returns predicted y values (including real and imaginary parts).
        x: Array of x values for the data.
        y: Array of y values for the data (including real and imaginary parts).
        parameterBounds: Optional (lower, upper) bounds for each parameter.
            Derivatives at a bound are taken from the allowed side.
        sources: Optional joint datasets, each containing its grid, measured
            values, and model. Uses fixed RMS and sample-count normalization.

    Returns:
        jac: The system Jacobian matrix.
    """

    bounds = (-np.inf, np.inf)
    if parameterBounds is not None:
        bounds = tuple(np.asarray(parameterBounds, dtype=float).T)

    jac = approx_derivative(
        lambda p: StackedResiduals(p, x, fitFunction, y, sources=sources),
        parameters,
        method="3-point",  # central differences (more accurate)
        bounds=bounds,
        # rel_step=1e-6,             # relative step; tunes accuracy
    )
    return jac


def get_uncertainties(
    parameters: ArrayLike,
    fitFunction: FitCallable,
    x: ArrayLike,
    y: ArrayLike,
    parameterBounds: ArrayLike | None = None,
    sources: JointSources | None = None,
) -> np.ndarray:
    r"""Compute the parameter uncertainties of the results of the Differential
    Evolution or minimization algorithm.

    This function takes the parameters obtained from the Differential Evolution or
    minimization algorithm and evaluate the Jacobian using `build_jacobian()`.
    Then, the covariance matrix is approximated as `(J^\top J)^{-1}`, the inverse
    of the Hessian. This approximation is valid in the vicinity of the optimal
    solution and may yield unphysical values close to bounds. The covariance
    matrix is obtained by computing the Moore-Penrose inverse, discarding zero
    singular values.

    The covariance is scaled by the residual sum of squares divided by the
    residual degrees of freedom (`M - N`), where `M` is the number of real
    residuals (including imaginary components only for complex input) and
    `N` is the number of parameters. The returned uncertainties are the
    square roots of the covariance diagonal.
    It is equivalent to leaving the default option absolute_sigma=False in
    scipy.optimize.curve_fit().

    Args:
        parameters: Array of parameters used by the fit_function.
        fitFunction: Function that takes parameters and x values as input and
            returns predicted y values (including real and imaginary parts).
        x: Array of x values for the data.
        y: Array of y values for the data (including real and imaginary parts).
        parameterBounds: Optional (lower, upper) bounds for each parameter.
            These keep numerical derivative probes inside the fit domain.
        sources: Optional joint datasets. Uses the same fixed normalized
            residuals as the joint objective for the Jacobian and variance.

    Returns:
        uncertainties: 1 \sigma standard deviation for each of the input parameters.

    Raises:
        ValueError: If the number of real residuals does not exceed the number
            of fitted parameters.
    """

    residuals = StackedResiduals(parameters, x, fitFunction, y, sources=sources)
    m = residuals.size
    n = np.asarray(parameters).size
    if m <= n:
        raise ValueError(
            "Parameter uncertainties require more real residuals than fitted "
            f"parameters (M > N); got M={m} and N={n}. Provide more data or fit "
            "fewer parameters. To skip automatic uncertainties during fitting, "
            "set compute_uncertainties=False."
        )

    jac = build_jacobian(
        parameters, fitFunction, x, y, parameterBounds, sources=sources
    )

    _, s, VT = svd(jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[: s.size]
    pcov = np.dot(VT.T / s**2, VT)  # absolute_sigma=True in scipy.optimize.curve_fit()

    pcov *= np.dot(residuals, residuals) / (m - n)  # absolute_sigma=False
    return np.sqrt(np.diag(pcov))

from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.linalg import svd
from scipy.optimize._numdiff import approx_derivative

import iddefix

ArrayLike = npt.ArrayLike
FitCallable = Callable[[ArrayLike, dict[int, ArrayLike]], ArrayLike]


def StackedResiduals(
    parameters: ArrayLike,
    x: ArrayLike,
    fitFunction: FitCallable,
    y: ArrayLike,
) -> np.ndarray:
    """Return real residuals, appending imaginary residuals for complex-valued data."""

    y = np.asarray(y)
    grouped_parameters = iddefix.utils.pars_to_dict(parameters)
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
) -> np.ndarray:
    """Differentiate stacked residuals with three-point finite differences.

    Optional bound pairs restrict probes; derivatives at bounds use the
    allowed side. Returns a real matrix with one column per fitted parameter.
    """

    bounds = (-np.inf, np.inf)
    if parameterBounds is not None:
        bounds = tuple(np.asarray(parameterBounds, dtype=float).T)
        values = np.asarray(parameters, dtype=float)
        # Include tiny solver roundoff at a bound; reject larger violations.
        if np.allclose(values, np.clip(values, *bounds), rtol=1e-13, atol=0):
            bounds = (np.minimum(bounds[0], values), np.maximum(bounds[1], values))

    jac = approx_derivative(
        lambda p: StackedResiduals(p, x, fitFunction, y),
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
) -> np.ndarray:
    """Estimate one-standard-deviation errors from the residual Jacobian.

    Parameters
    ----------
    parameters : array_like
        Flat vector in repeated (Rs, Q, fr) order.
    fitFunction : callable
        Model called as fitFunction(x, grouped_parameter_dictionary).
    x : array_like
        Input grid in the model domain: frequency [Hz] or time [s].
    y : array_like
        Data on x in the same units as model predictions. Real input uses real
        residuals; complex input stacks real and imaginary residuals.
    parameterBounds : array_like or None, optional
        Bound pair per parameter for derivative probes. None uses unbounded probes.

    Returns
    -------
    ndarray
        Standard deviations in the same order and units as parameters.

    Raises
    ------
    ValueError
        If the number of real residuals does not exceed the parameter count.

    Notes
    -----
    The local covariance uses an SVD pseudoinverse of J.T@J, scaled by
    SSE/(M-N), where M is the number of real residuals and N the parameter count.
    These local least-squares estimates do not guarantee identifiable parameters
    or valid confidence intervals near bounds.
    """

    residuals = StackedResiduals(parameters, x, fitFunction, y)
    m = residuals.size
    n = np.asarray(parameters).size
    if m <= n:
        raise ValueError(
            "Parameter uncertainties require more real residuals than fitted "
            f"parameters (M > N); got M={m} and N={n}. Provide more data or fit "
            "fewer parameters. To skip automatic uncertainties during fitting, "
            "set compute_uncertainties=False."
        )

    jac = build_jacobian(parameters, fitFunction, x, y, parameterBounds)

    _, s, VT = svd(jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[: s.size]
    pcov = np.dot(VT.T / s**2, VT)  # absolute_sigma=True in scipy.optimize.curve_fit()

    pcov *= np.dot(residuals, residuals) / (m - n)  # absolute_sigma=False
    return np.sqrt(np.diag(pcov))

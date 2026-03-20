import numpy as np
from scipy.linalg import svd
from scipy.optimize._numdiff import approx_derivative

import iddefix


def StackedResiduals(parameters, x, fitFunction, y):
    """The stacked residuals (real and imaginary parts).

    This function takes the parameters obtained from the Differential Evolution or
    minimization algorithm and evaluate the stacked residuals at that solution.

    Args:
        parameters: Array of parameters used by the fit_function.
        x: Array of x values for the data.
        fitFunction: Function that takes parameters and x values as input and
            returns predicted y values (including real and imaginary parts).
        y: Array of y values for the data (including real and imaginary parts).

    Returns:
        residuals: The stacked residuals for a given solution.
    """

    grouped_parameters = iddefix.utils.pars_to_dict(parameters)
    predicted_y = fitFunction(x, grouped_parameters)

    residuals = np.concatenate(
        [
            y.real - predicted_y.real,
            y.imag - predicted_y.imag,
        ]
    )
    return residuals


def build_jacobian(parameters, fitFunction, x, y):
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

    Returns:
        jac: The system Jacobian matrix.
    """

    jac = approx_derivative(
        lambda p: StackedResiduals(p, x, fitFunction, y),
        parameters,
        method="3-point",  # central differences (more accurate)
        # rel_step=1e-6,             # relative step; tunes accuracy
    )
    return jac


def get_uncertainties(parameters, fitFunction, x, y):
    r"""Compute the parameter uncertainties of the results of the Differential
    Evolution or minimization algorithm.

    This function takes the parameters obtained from the Differential Evolution or
    minimization algorithm and evaluate the Jacobian using `build_jacobian()`.
    Then, the covariance matrix is approximated as `(J^\top J)^{-1}`, the inverse
    of the Hessian. This approximation is valid in the vicinity of the optimal
    solution and may yield unphysical values close to bounds. The covariance
    matrix is obtained by computing the Moore-Penrose inverse, discarding zero
    singular values.

    The uncertainties are defined as the `1 \sigma` standard deviation multiplied
    by `\xi^2 / (M - N)`, where `M` is the length of `x`, `N` the length of
    `parameters` and `\xi^2` the reduced chi-squared.
    It is equivalent to leaving the default option absolute_sigma=False in
    scipy.optimize.curve_fit().

    Args:
        parameters: Array of parameters used by the fit_function.
        fitFunction: Function that takes parameters and x values as input and
            returns predicted y values (including real and imaginary parts).
        x: Array of x values for the data.
        y: Array of y values for the data (including real and imaginary parts).

    Returns:
        uncertainties: 1 \sigma standard deviation for each of the input parameters.
    """

    jac = build_jacobian(parameters, fitFunction, x, y)

    _, s, VT = svd(jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[: s.size]
    pcov = np.dot(VT.T / s**2, VT)  # absolute_sigma=True in scipy.optimize.curve_fit()

    residuals = StackedResiduals(parameters, x, fitFunction, y)
    m = residuals.size
    n = parameters.size
    pcov *= np.dot(residuals, residuals) / (m - n)  # absolute_sigma=False
    return np.sqrt(np.diag(pcov))

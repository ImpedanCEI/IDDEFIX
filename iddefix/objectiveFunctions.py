"""
Created on Sat Dec  5 16:34:10 2020
Modified Tue 23 Sep @edelafue

@author: sjoly
"""

from typing import Callable

import numpy as np
import numpy.typing as npt

ArrayLike = npt.ArrayLike
FitCallable = Callable[[ArrayLike, dict[int, ArrayLike]], ArrayLike]

from .utils import pars_to_dict


def _sum_objective_terms(terms: np.ndarray, predicted_y: np.ndarray) -> float:
    """Sum terms, rejecting nonfinite predictions, NaN, and positive-infinite loss."""
    # np.nansum can make an all-NaN model look like a perfect (zero-error) fit.
    if (
        not np.all(np.isfinite(predicted_y))
        or np.any(np.isnan(terms))
        or np.any(np.isposinf(terms))
    ):
        return float("inf")
    return float(np.sum(terms))


class ObjectiveFunctions:
    """Scalar objectives for resonator fitting, called as class functions.

    ``parameters`` is a flat vector of (Rs, Q, fr) triples. Each objective groups
    it into a dictionary before calling ``fitFunction(x, grouped_parameters)``.
    ``x`` and ``y`` are NumPy arrays in the fitting domain. Invalid model values
    produce infinite loss. Log objectives sum logarithms of sample errors;
    they are not the logarithm of the total SSE or a fit in log-amplitude space.
    """

    def sumOfSquaredError(
        parameters: ArrayLike,
        fitFunction: FitCallable,
        x: ArrayLike,
        y: ArrayLike,
    ) -> float:
        """Return sum(abs(y - prediction)**2), including real and imaginary errors.

        Parameters
        ----------
        parameters : array_like
            Flat parameter vector in repeated (Rs, Q, fr) order.
        fitFunction : FitCallable
            Model callable accepting (x, grouped_parameter_dictionary).
        x : array_like
            Input sample locations in the model domain.
        y : array_like
            Reference NumPy data array on x, in the model's units.

        Returns
        -------
        float
            Scalar loss; see Notes for the residual components and logarithm convention.

        Notes
        -----
        See ObjectiveFunctions for the shared parameters and model-call convention.
        """

        grouped_parameters = pars_to_dict(parameters)
        predicted_y = fitFunction(x, grouped_parameters)
        squared_error = _sum_objective_terms(
            (y.real - predicted_y.real) ** 2 + (y.imag - predicted_y.imag) ** 2,
            predicted_y,
        )
        return squared_error

    def sumOfSquaredErrorReal(
        parameters: ArrayLike,
        fitFunction: FitCallable,
        x: ArrayLike,
        y: ArrayLike,
    ) -> float:
        """Return sum((y.real - prediction.real)**2), ignoring imaginary residuals.

        Parameters
        ----------
        parameters : array_like
            Flat parameter vector in repeated (Rs, Q, fr) order.
        fitFunction : FitCallable
            Model callable accepting (x, grouped_parameter_dictionary).
        x : array_like
            Input sample locations in the model domain.
        y : array_like
            Reference NumPy data array on x, in the model's units.

        Returns
        -------
        float
            Scalar loss; see Notes for the residual components and logarithm convention.

        Notes
        -----
        See ObjectiveFunctions for the shared parameters and model-call convention.
        """

        grouped_parameters = pars_to_dict(parameters)
        predicted_y = fitFunction(x, grouped_parameters)
        squared_error = _sum_objective_terms(
            (y.real - predicted_y.real) ** 2, predicted_y
        )
        return squared_error

    def sumOfSquaredErrorAbs(
        parameters: ArrayLike,
        fitFunction: FitCallable,
        x: ArrayLike,
        y: ArrayLike,
    ) -> float:
        """Return sum((abs(y) - abs(prediction))**2), ignoring phase.

        Parameters
        ----------
        parameters : array_like
            Flat parameter vector in repeated (Rs, Q, fr) order.
        fitFunction : FitCallable
            Model callable accepting (x, grouped_parameter_dictionary).
        x : array_like
            Input sample locations in the model domain.
        y : array_like
            Reference NumPy data array on x, in the model's units.

        Returns
        -------
        float
            Scalar loss; see Notes for the residual components and logarithm convention.

        Notes
        -----
        See ObjectiveFunctions for the shared parameters and model-call convention.
        """

        grouped_parameters = pars_to_dict(parameters)
        predicted_y = fitFunction(x, grouped_parameters)
        squared_error = _sum_objective_terms(
            (np.abs(y) - np.abs(predicted_y)) ** 2, predicted_y
        )
        return squared_error

    def logsumOfSquaredError(
        parameters: ArrayLike,
        fitFunction: FitCallable,
        x: ArrayLike,
        y: ArrayLike,
    ) -> float:
        """Return sum(log(abs(y - prediction)**2)).

        Parameters
        ----------
        parameters : array_like
            Flat parameter vector in repeated (Rs, Q, fr) order.
        fitFunction : FitCallable
            Model callable accepting (x, grouped_parameter_dictionary).
        x : array_like
            Input sample locations in the model domain.
        y : array_like
            Reference NumPy data array on x, in the model's units.

        Returns
        -------
        float
            Scalar loss; see Notes for the residual components and logarithm convention.

        Notes
        -----
        Uses both components. Exact zero residuals contribute -inf; no floor is
        added. See ObjectiveFunctions for the shared model-call convention.
        """
        grouped_parameters = pars_to_dict(parameters)
        predicted_y = fitFunction(x, grouped_parameters)
        log_squared_error = _sum_objective_terms(
            np.log((y.real - predicted_y.real) ** 2 + (y.imag - predicted_y.imag) ** 2),
            predicted_y,
        )
        return log_squared_error

    def logsumOfSquaredErrorReal(
        parameters: ArrayLike,
        fitFunction: FitCallable,
        x: ArrayLike,
        y: ArrayLike,
    ) -> float:
        """Return sum(log((y.real - prediction.real)**2)).

        Parameters
        ----------
        parameters : array_like
            Flat parameter vector in repeated (Rs, Q, fr) order.
        fitFunction : FitCallable
            Model callable accepting (x, grouped_parameter_dictionary).
        x : array_like
            Input sample locations in the model domain.
        y : array_like
            Reference NumPy data array on x, in the model's units.

        Returns
        -------
        float
            Scalar loss; see Notes for the residual components and logarithm convention.

        Notes
        -----
        Imaginary residuals are ignored. Exact matches contribute -inf; no floor
        is added. See ObjectiveFunctions for the shared model-call convention.
        """

        grouped_parameters = pars_to_dict(parameters)
        predicted_y = fitFunction(x, grouped_parameters)
        log_squared_error = _sum_objective_terms(
            np.log((y.real - predicted_y.real) ** 2), predicted_y
        )
        return log_squared_error

    def logsumOfSquaredErrorAbs(
        parameters: ArrayLike,
        fitFunction: FitCallable,
        x: ArrayLike,
        y: ArrayLike,
        eps: float = 1e-12,
    ) -> float:
        """Return sum(log((abs(y) - abs(prediction))**2 + eps)).

        Parameters
        ----------
        parameters : array_like
            Flat parameter vector in repeated (Rs, Q, fr) order.
        fitFunction : FitCallable
            Model callable accepting (x, grouped_parameter_dictionary).
        x : array_like
            Input sample locations in the model domain.
        y : array_like
            Reference NumPy data array on x, in the model's units.
        eps : float, optional
            Positive squared-error floor before taking the logarithm. Default is 1e-12.

        Returns
        -------
        float
            Scalar loss; see Notes for the residual components and logarithm convention.

        Notes
        -----
        Ignores phase. ``eps`` defaults to 1e-12 and floors the squared magnitude
        error before taking its natural logarithm. See ObjectiveFunctions for the
        shared parameters and model-call convention.
        """

        grouped_parameters = pars_to_dict(parameters)
        predicted_y = fitFunction(x, grouped_parameters)
        log_squared_error = _sum_objective_terms(
            np.log((np.abs(y) - np.abs(predicted_y)) ** 2 + eps), predicted_y
        )
        return log_squared_error

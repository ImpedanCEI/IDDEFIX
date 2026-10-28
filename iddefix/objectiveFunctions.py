#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
    # np.nansum can make an all-NaN model look like a perfect (zero-error) fit.
    if (
        not np.all(np.isfinite(predicted_y))
        or np.any(np.isnan(terms))
        or np.any(np.isposinf(terms))
    ):
        return float("inf")
    return float(np.sum(terms))


class ObjectiveFunctions:
    def jointSumOfSquaredError(
        parameters: ArrayLike,
        sources: dict[str, tuple[ArrayLike, ArrayLike, FitCallable]],
        fitFunction: FitCallable | None = None,
        x: ArrayLike | None = None,
        y: ArrayLike | None = None,
    ) -> float:
        """Sum each source's mean squared error divided by measured RMS squared.

        ``sources`` maps each active domain to its grid, measurements, and model.
        Sources contribute equally, with real and imaginary errors for complex
        measurements. An all-zero measured trace uses scale 1. The other
        arguments allow use with the existing solver dispatch interface.
        """
        grouped_parameters = pars_to_dict(parameters)
        loss = 0.0
        for source_x, source_y, source_model in sources.values():
            source_y = np.asarray(source_y)
            predicted_y = np.asarray(source_model(source_x, grouped_parameters))
            # Match the fixed normalization used for joint uncertainties.
            peak = float(np.max(np.abs(source_y)))
            scale = (
                peak * np.sqrt(np.mean(np.abs(source_y / peak) ** 2)) if peak else 1.0
            )
            difference = (source_y - predicted_y) / scale
            terms = difference.real**2
            if np.iscomplex(source_y).any():
                terms = terms + difference.imag**2
            loss += _sum_objective_terms(terms, predicted_y) / source_y.size
        return float(loss)

    def sumOfSquaredError(
        parameters: ArrayLike,
        fitFunction: FitCallable,
        x: ArrayLike,
        y: ArrayLike,
    ) -> float:
        """Calculates the sum of squared errors (SSE) for a given fit function.

        This function computes the SSE between the predicted values from a fit
        function and the actual data points. It works with both real and imaginary
        components of the data.

        Args:
            parameters: Array of parameters used by the fit_function.
            fitFunction: Function that takes parameters and x values as input and
                returns predicted y values (including real and imaginary parts).
            x: Array of x values for the data.
            y: Array of y values for the data (including real and imaginary parts).

        Returns:
            The sum of squared errors (SSE).
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
        """Calculates the real sum of squared errors (SSE) for a given fit function.

        This function computes the SSE between the predicted values from a fit
        function and the actual data points. It works only with the real component
        of the data.

        Args:
            parameters: Array of parameters used by the fit_function.
            fitFunction: Function that takes parameters and x values as input and
                returns predicted y values (including only the real part).
            x: Array of x values for the data.
            y: Array of y values for the data (including only the real part).

        Returns:
            The real sum of squared errors (SSE).
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
        """Calculates the magnitude-based sum of squared errors (SSE).

        This function computes the SSE between the magnitudes of the predicted
        values from a fit function and the magnitudes of the actual data points.

        Args:
            parameters: Array of parameters used by the fitFunction.
            fitFunction: Function that takes parameters and x values as input and
                returns predicted y values.
            x: Array of x values for the data.
            y: Array of y values for the data.

        Returns:
            The magnitude-based sum of squared errors (SSE).
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
        """Calculates the sum of log squared errors for a fit function.

        This function computes the log squared errors between the predicted
        values from a fit function and the actual data points. It works with
        both real and imaginary components of the data.

        Args:
            parameters: Array of parameters used by the fit_function.
            fitFunction: Function that takes parameters and x values as input and
                returns predicted y values (including real and imaginary parts).
            x: Array of x values for the data.
            y: Array of y values for the data (including real and imaginary parts).

        Returns:
            The sum of log squared errors.
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
        """Calculates the real sum of log squared errors for a given fit function.

        This function computes the real log squared errors between the predicted values
        from a fit function and the actual data points.
        It works only with the real component of the data.

        Args:
            parameters: Array of parameters used by the fit_function.
            fitFunction: Function that takes parameters and x values as input and
                returns predicted y values (including only the real part).
            x: Array of x values for the data.
            y: Array of y values for the data (including only the real part).

        Returns:
            The real sum of log squared errors.
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
        """Calculates the magnitude-based sum of log squared errors.

        This function computes the log of squared errors between the magnitudes
        of the predicted values from a fit function and the magnitudes of the
        actual data points. A small epsilon is added inside the log to avoid
        issues with ``log(0)``.

        Args:
            parameters: Array of parameters used by the fitFunction.
            fitFunction: Function that takes parameters and x values as input and
                returns predicted y values.
            x: Array of x values for the data.
            y: Array of y values for the data.
            eps: Small positive constant to prevent log(0).

        Returns:
            The magnitude-based sum of log squared errors.
        """

        grouped_parameters = pars_to_dict(parameters)
        predicted_y = fitFunction(x, grouped_parameters)
        log_squared_error = _sum_objective_terms(
            np.log((np.abs(y) - np.abs(predicted_y)) ** 2 + eps), predicted_y
        )
        return log_squared_error

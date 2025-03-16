#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:34:10 2020

@author: sjoly
"""
import numpy as np

from .utils import pars_to_dict

class ObjectiveFunctions:
    def sumOfSquaredError(parameters, fitFunction, x, y):
        """Calculates the sum of squared errors (SSE) for a given fit function.

        This function computes the SSE between the predicted values from a fit function and
        the actual data points. It works with both real and imaginary components of the data.

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
        squared_error = np.nansum((y.real - predicted_y.real)**2 + (y.imag - predicted_y.imag)**2)
        return squared_error

    def sumOfSquaredErrorReal(parameters, fitFunction, x, y):
        """Calculates the real sum of squared errors (SSE) for a given fit function.

        This function computes the SSE between the predicted values from a fit function and
        the actual data points. It works only with the real component of the data.

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
        squared_error = np.nansum((y.real - predicted_y.real)**2)
        return squared_error

    def logsumOfSquaredError(parameters, fitFunction, x, y):
        """Calculates the sum of log squared errors for a given fit function.

        This function computes the log squared errors between the predicted values from a fit function
        and the actual data points. It works with both real and imaginary components of the data.

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
        log_squared_error = np.nansum(np.log((y.real - predicted_y.real)**2 + (y.imag - predicted_y.imag)**2))
        return log_squared_error

    def logsumOfSquaredErrorReal(parameters, fitFunction, x, y):
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
        log_squared_error = np.nansum(np.log((y.real - predicted_y.real)**2))
        return log_squared_error
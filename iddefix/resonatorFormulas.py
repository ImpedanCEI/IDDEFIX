#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:20:11 2020

@author: sjoly
"""
import numpy as np
from scipy import special as sp

from .utils import pars_to_dict

class Wakes:

    # Longitudinal and transverse wake functions
    def Resonator_longitudinal_wake(times, Rs, Q, resonant_frequency):
        """Calculates the longitudinal wake function of a resonator.

        This function calculates the longitudinal wake function of a resonator
        with shunt impedance `Rs`, quality factor `Q`, and resonant frequency
        `resonant_frequency` at different times `times`.

        Args:
            times (np.ndarray): Array of time values in seconds.
            Rs (float): Shunt impedance of the resonator in Ohm.
            Q (float): Quality factor of the resonator.
            resonant_frequency (float): Resonant frequency of the resonator in Hz.

        Returns:
            np.ndarray: Array of longitudinal wake function values [V/C] at the corresponding times.

        Notes:
            This formula uses the generalized wake function formula and
            can be used for any real positive value of `Q`.
            https://cds.cern.ch/record/192684/files/198812060.pdf

            Units for this formula are:
                Rs: Ohm
                Q: dimensionless
                resonant_frequency: Hz

        Examples:
            >>> times = np.linspace(0, 1e-9, 1000)
            >>> Rs = 1e6
            >>> Q = 0.6
            >>> resonant_frequency = 1e9
            >>> wake_function = Resonator_longitudinal_wake(times, Rs, Q, resonant_frequency)
            >>> plt.plot(times, wake_function)
            >>> plt.xlabel("Time [s]")
            >>> plt.ylabel("Longitudinal Wake Function [V/C]")
            >>> plt.show()
        """

        omega_r = 2 * np.pi * resonant_frequency
        if Q == 0.5:
            Wl = 2 * Rs * omega_r * np.exp(-omega_r * times) * (1 - omega_r * times)
        elif Q < 0.5:
            omega_r_bar = omega_r * np.sqrt((1 / (4 * Q**2)) - 1)
            cos_term = np.cosh(omega_r_bar * times)
            sin_term = np.sinh(omega_r_bar * times)
            exp_term = np.exp(-omega_r * times/(2 * Q))
            Wl = Rs * omega_r / Q * exp_term * (cos_term - omega_r / (2 * Q * omega_r_bar) * sin_term)
        else:
            omega_r_bar = omega_r * np.sqrt(1 - (1 / (4 * Q**2)))
            cos_term = np.cos(omega_r_bar * times)
            sin_term = np.sin(omega_r_bar * times)
            exp_term = np.exp(-omega_r * times/(2 * Q))
            Wl = Rs * omega_r / Q * exp_term * (cos_term - omega_r / (2 * Q * omega_r_bar) * sin_term)
        Wl[times == 0.] = omega_r / (2 * Q) # Fundamental theorem of beam loading
        Wl[times < 0] = 0.

        return Wl

    def Resonator_transverse_wake(times, Rs, Q, resonant_frequency):
        """Calculates the longitudinal wake function of a resonator.

        This function calculates the transverse wake function of a resonator
        with shunt impedance `Rs`, quality factor `Q`, and resonant frequency
        `resonant_frequency` at different times `times`.

        Args:
            times (np.ndarray): Array of time values in seconds.
            Rs (float): Shunt impedance of the resonator in Ohm/m.
            Q (float): Quality factor of the resonator.
            resonant_frequency (float): Resonant frequency of the resonator in Hz.

        Returns:
            np.ndarray: Array of longitudinal wake function values [V/C/m] at the corresponding times.

        Notes:
            This formula uses the generalized wake function formula and
            can be used for any real positive value of `Q`.
            https://cds.cern.ch/record/192684/files/198812060.pdf

            Units for this formula are:
                Rs: Ohm/m
                Q: dimensionless
                resonant_frequency: Hz

        Examples:
            >>> times = np.linspace(0, 1e-9, 1000)
            >>> Rs = 1e6
            >>> Q = 0.6
            >>> resonant_frequency = 1e9
            >>> wake_function = Resonator_transverse_wake(times, Rs, Q, resonant_frequency)
            >>> plt.plot(times, wake_function)
            >>> plt.xlabel("Time [s]")
            >>> plt.ylabel("Transverse Wake Function [V/C]")
            >>> plt.show()
        """

        omega_r = 2 * np.pi * resonant_frequency
        exp_term = np.exp(-omega_r * times / 2 / Q)
        if Q == 0.5:
            Wt = omega_r**2 * Rs / Q * exp_term * times
        elif Q < 0.5:
            omega_r_bar = omega_r * np.sqrt((1 / (4 * Q**2)) - 1)
            sqrt_term = np.sqrt((1 /(4 * Q * Q)) - 1)
            sin_term = np.sinh(omega_r * sqrt_term * times)
            Wt = omega_r**2 * Rs / (Q * omega_r_bar) * exp_term * sin_term
        else:
            omega_r_bar = omega_r * np.sqrt(1 - (1 / (4 * Q**2)))
            sqrt_term = np.sqrt(1 - (1 /(4 * Q * Q)))
            sin_term = np.sin(omega_r * sqrt_term * times)
            Wt = omega_r**2 * Rs / (Q * omega_r_bar) * exp_term * sin_term
        Wt[times < 0] = 0.

        return Wt

    def n_Resonator_longitudinal_wake(times, pars):
        """Calculates the combined longitudinal wake function of multiple resonators.

        This function calculates the total longitudinal wake function induced by a system
        consisting of multiple resonators at different times `times`. Each resonator
        is defined by its parameters provided in a dictionary `dict_params`.

        Args:
            times (np.ndarray): Array of time values in seconds.
            dict_params (dict or ndarray): Dictionary containing resonator parameters.
            If dict, Keys are unique identifiers for each resonator, and values are
            lists containing the parameters in the following order:
                    - Rs (float): Shunt impedance of the resonator in Ohm.
                    - Q (float): Quality factor of the resonator.
                    - resonant_frequency (float): Resonant frequency of the resonator in Hz.

        Returns:
            np.ndarray: Array of combined transverse wake function values [V/C] at
                the corresponding times.

        Examples:
            >>> times = np.linspace(0, 1e-9, 1000)
            >>> pars = {
            ...     1: [1e6, 1, 1e9],
            ...     2: [2e6, 1, 5e8],
            ... }
            >>> # Or, use directly the result from the DE solver
            >>> pars = DE_model.evolutionParameters
            >>> wake_function = n_Resonator_longitudinal_wake(times, pars)
            >>> plt.plot(times, wake_function)
            >>> plt.xlabel("Time [s]")
            >>> plt.ylabel("Longitudinal Wake Function [V/C]")
            >>> plt.show()

        Notes:
            - This function assumes all resonators have the same type of longitudinal
                wake function formula implemented in `Resonator_longitudinal_wake`.
            - The combined wake function is calculated by summing the individual
                wake function contributions from each resonator at each time step.
            - Resonator parameters should be positive values (except for the shunt impedance).
            Behavior for invalid values is not defined and may lead to errors.
        """
        if type(pars) is dict:
            dict_params = pars
        else:
            dict_params = pars_to_dict(pars) #takes list or ndarray

        Wl = np.sum(Wakes.Resonator_longitudinal_wake(times, *params) for params in dict_params.values())
        return Wl

    def n_Resonator_transverse_wake(times, pars):
        """Calculates the combined transverse wake function of multiple resonators.

        This function calculates the total transverse wake function induced by a system
        consisting of multiple resonators at different times `times`. Each resonator
        is defined by its parameters provided in a dictionary `dict_params`.

        Args:
            times (np.ndarray): Array of time values in seconds.
            dict_params (dict or ndarray): Dictionary containing resonator parameters. Keys are
                unique identifiers for each resonator, and values are lists containing
                the parameters in the following order:
                    - Rs (float): Shunt impedance of the resonator in Ohm/m.
                    - Q (float): Quality factor of the resonator.
                    - resonant_frequency (float): Resonant frequency of the resonator in Hz.

        Returns:
            np.ndarray: Array of combined transverse wake function values [V/C/m] at
                the corresponding times.

        Examples:
            >>> times = np.linspace(0, 1e-9, 1000)
            >>> pars = {
            ...     1: [1e6, 1, 1e9],
            ...     2: [2e6, 1, 5e8],
            ... }
            >>> # Or, use directly the result from the DE solver
            >>> pars = DE_model.evolutionParameters
            >>> wake_function = n_Resonator_transverse_wake(times, pars)
            >>> plt.plot(times, wake_function)
            >>> plt.xlabel("Time [s]")
            >>> plt.ylabel("Transverse Wake Function [V/C/m]")
            >>> plt.show()

        Notes:
            - This function assumes all resonators have the same type of transverse
                wake function formula implemented in `Resonator_transverse_wake`.
            - The combined wake function is calculated by summing the individual
                wake function contributions from each resonator at each time step.
            - Resonator parameters should be positive values (except for the shunt impedance).
            Behavior for invalid values is not defined and may lead to errors.
        """
        if type(pars) is dict:
            dict_params = pars
        else:
            dict_params = pars_to_dict(pars) #takes list or ndarray

        Wt = np.sum(Wakes.Resonator_transverse_wake(times, *params) for params in dict_params.values())
        return Wt

    # Longitudinal and transverse wake potentials
    def Resonator_longitudinal_wake_potential(times, Rs, Q, resonant_frequency, sigma=1e-10, use_mpmath=False):
        """
        Single resonator wake potential (longitudinal) for a Gaussian bunch of line density.

        Args:
            Rs (float or list): Shunt impedance (Ohm).
            resonant_frequency (float or list): Resonant frequency (Hz).
            Q (float or list): Quality factor.
            sigma (float): RMS bunch length (s) 
            times (array-like): Times (s) where wake is computed (times > 0 behind the source).
            use_mpmath (bool, optional): Use mpmath for calculations. Defaults to False.

        Returns:
            np.ndarray: Wake potential at times `times`.

        Notes:
            The formula is from Chao's Handbook (p. 237, sec 3.2), partly re-derived by
            N. Mounet. Equivalent formula in https://cds.cern.ch/record/192684/files/198812060.pdf
            Q must be different from 0.5!
            Rs, resonant_frequency, and Q must be scalar.
        """
        omegar = 2 * np.pi * resonant_frequency
        kr = omegar * (1 - 1 / (4 * Q**2))**0.5
        alphar = omegar / (2 * Q)
        cstsin = -Rs * omegar**2 / (4 * Q**2 * kr)
        cstcos = Rs * omegar / (2 * Q)

        mpmath_is_installed = True

        if use_mpmath:
            try:
                import mpmath
            except:
                raise ImportError("mpmath is not installed. Please install it to use this function.")
                mpmath_is_installed = False

        if use_mpmath and mpmath_is_installed:
            from mpmath import erfc, exp, matrix, re, im
            cst = exp((alphar**2 - kr**2) * sigma**2 / 2)
            times_mp = matrix(times)
            erfc_arg = -(times_mp - alphar * sigma**2 + 1j * kr * sigma**2) / (np.sqrt(2) * sigma)
            erfc_v = erfc_arg.apply(erfc)
            arg_expo1= -alphar * times_mp
            expo1 = arg_expo1.apply(exp)
            arg_expo2= 1j * (kr * times_mp - alphar * kr * sigma**2)
            expo2 = arg_expo2.apply(exp)
            im_expo_erfc = matrix([im(ex * er) for (ex, er) in zip(expo2, erfc_v)])
            re_expo_erfc = matrix([re(ex * er) for (ex, er) in zip(expo2, erfc_v)])
            W = np.hstack([cst * ex1 * (cstsin * im_exr + cstcos * re_exr)
                            for (ex1, im_exr, re_exr) in zip(expo1, im_expo_erfc, re_expo_erfc)])
        else:
            cst = np.exp((alphar**2 - kr**2) * sigma**2 / 2)
            erfc_v = sp.erfc(-(times - alphar * sigma**2 + 1j * kr * sigma**2) / (np.sqrt(2) * sigma))
            expo1 = np.exp(-alphar * times)
            expo2 = np.exp(1j * (kr * times - alphar * kr * sigma**2))
            W = cst * expo1 * (cstsin * np.imag(expo2 * erfc_v) + cstcos * np.real(expo2 * erfc_v))

        return W

    def n_Resonator_longitudinal_wake_potential(times, pars, sigma=1e-10):

        if type(pars) is dict:
            dict_params = pars
        else:
            dict_params = pars_to_dict(pars) #takes list or ndarray

        Wpl = np.sum(Wakes.Resonator_longitudinal_wake_potential(times, *params, sigma=sigma) for params in dict_params.values())
        return Wpl

    def Resonator_transverse_wake_potential(times, Rs, Q, resonant_frequency, sigma=1e-10, use_mpmath=False):
        """
        Single resonator wake potential (transverse) for a Gaussian bunch of line density.

        Args:
            Rs (float or list): Shunt impedance (Ohm).
            resonant_frequency (float or list): Resonant frequency (Hz).
            Q (float or list): Quality factor.
            sigma (float): RMS bunch length (s) 
            times (array-like): Times (s) where wake is computed (times > 0 behind the source).
            use_mpmath (bool, optional): Use mpmath for calculations. Defaults to False.

        Returns:
            np.ndarray: Wake potential at times `times`.

        Notes:
            The formula is from Chao's Handbook (p. 237, sec 3.2), partly re-derived by
            N. Mounet. Equivalent formula in https://cds.cern.ch/record/192684/files/198812060.pdf
            Q must be different from 0.5!
            Rs, resonant_frequency, and Q must be scalar.
        """

        omegar = 2 * np.pi * resonant_frequency
        kr = omegar * (1 - 1 / (4 * Q**2))**0.5
        alphar = omegar / (2 * Q)

        mpmath_is_installed = True

        if use_mpmath:
            try:
                import mpmath
            except:
                raise ImportError("mpmath is not installed. Please install it to use this function.")
                mpmath_is_installed = False

        if use_mpmath and mpmath_is_installed:
            from mpmath import erfc, exp, matrix, re, im
            times_mp = matrix(times)
            cst = Rs * omegar**2 / (2 * Q * kr) * exp((alphar**2 - kr**2) * sigma**2 / 2)
            erfc_arg = -(times_mp - alphar * sigma**2 + 1j * kr * sigma**2) / (np.sqrt(2) * sigma)
            erfc_v = erfc_arg.apply(erfc)
            arg_expo1 = -alphar * times_mp
            expo1 = arg_expo1.apply(exp)
            arg_expo2= 1j * (kr * times_mp - alphar * kr * sigma**2)
            expo2 = arg_expo2.apply(exp)
            im_expo_erfc = matrix([im(ex * er) for (ex, er) in zip(expo2, erfc_v)])
            W = np.hstack([cst * ex1 * im_exr for (ex1, im_exr) in zip(expo1, im_expo_erfc)])
        else:
            cst = Rs * omegar**2 / (2 * Q * kr) * np.exp((alphar**2 - kr**2) * sigma**2 / 2)
            erf_v = sp.erf((times - alphar * sigma**2 + 1j * kr * sigma**2) / (np.sqrt(2) * sigma))
            expo1 = np.exp(-alphar * times)
            expo2 = np.exp(1j * (kr * times - alphar * kr * sigma**2))
            W = cst * expo1 * np.imag(expo2 * (1 + erf_v))

        return W

    def n_Resonator_transverse_wake_potential(times, pars, sigma=1e-10):

        if type(pars) is dict:
            dict_params = pars
        else:
            dict_params = pars_to_dict(pars) #takes list or ndarray

        Wpt = np.sum(Wakes.Resonator_transverse_wake_potential(times, *params, sigma=sigma) for params in dict_params.values())
        return Wpt


class Impedances:

    # Longitudinal and transverse impedance functions
    def Resonator_longitudinal_imp(frequencies, Rs, Q, resonant_frequency, wake_length=None):
        """Calculates the longitudinal impedance of a resonator.

        This function calculates the longitudinal impedance of a resonator
        with shunt impedance `Rs`, quality factor `Q`, and resonant frequency
        `resonant_frequency` at different frequencies `frequencies`.

        Args:
            frequencies (np.ndarray): Array of frequencies values in Hz.
            Rs (float): Shunt impedance of the resonator in Ohm.
            Q (float): Quality factor of the resonator.
            resonant_frequency (float): Resonant frequency of the resonator in Hz.
            wake_length (float, optional): Additional parameter for the calculation
            of the impedance of a partially decayed wake.
                Defaults to None for the original (fully decayed wake) behavior.

        Returns:
            np.ndarray: Array of longitudinal impedance values [Ohm] at the corresponding frequencies.

        Notes:
            The fully decayed formula uses the generalized impedance formula
            (https://cds.cern.ch/record/192684/files/198812060.pdf) and can
            be used for any real positive value of `Q`.

            The partially decayed formula uses the formula derived in
            (Joly, S. thesis not published yet!)

            Moreover, it sets the impedance value to zero for zero frequencies
            in both cases.

            Units for this formula are:
                Rs: Ohm/m
                Q: dimensionless
                resonant_frequency: Hz
                wake_length: m

        Examples:
            >>> frequencies = np.linspace(0, 2.5e9, 1000)
            >>> Rs = 1e6
            >>> Q = 0.6
            >>> resonant_frequency = 1e9
            >>> impedance = Resonator_longitudinal_imp(frequencies, Rs, Q, resonant_frequency)
            >>> plt.plot(frequencies, impedance)
            >>> plt.xlabel("Frequency [Hz]")
            >>> plt.ylabel("Longitudinal Impedance [Ohm]")
            >>> plt.show()
        """
        if wake_length is None:
            # Fully decayed wake
            zero_index = np.where(frequencies > 0)[0]  # find index of non-zero element
            if zero_index.size < frequencies.size:
                Zl = np.zeros_like(frequencies, dtype=complex)  # initialize Zl as 0
                Zl[zero_index] = Rs / (1 + 1j*Q * (
                        frequencies[zero_index]/resonant_frequency -
                    resonant_frequency/frequencies[zero_index])) # calculate all Zl for non-zero frequencies
            else:
                Zl = Rs / (1 + 1j*Q * (
                        frequencies/resonant_frequency - resonant_frequency/frequencies))

        else:
            # Partially decayed wake
            omega = 2 * np.pi * frequencies
            omega_r = 2 * np.pi * resonant_frequency
            c = 299792458.0 # speed of light in vacuum

            if Q < 0.5:
                raise ValueError("Quality factor Q must be larger than 0.5."
                                "The wake is unlikely to be partially decayed"
                                "for such a low quality factor otherwise.")

            B = omega_r / 2 / Q
            C = omega_r * np.sqrt(1 - 1 / 4 / Q**2)
            T = wake_length / c

            zero_index = np.where(frequencies > 0)[0]  # find index of non-zero element
            if zero_index.size < frequencies.size:
                #A = Rs * omega_r / 2 / Q
                A = Rs * omega[zero_index] / 2 / Q # correct scaling to fit with usual formula
                exp_term = np.exp(-(B - 1j*(C - omega[zero_index])) * T)
                numerator = A * (1 - exp_term)
                denominator = B - 1j*(C - omega[zero_index])
                Zl = np.zeros_like(frequencies, dtype=complex)  # initialize Zl as 0
                # calculate all Zl for non-zero frequencies
                Zl[zero_index] = numerator / denominator

            else:
                #A = Rs * omega_r / 2 / Q
                A = Rs * omega / 2 / Q # correct scaling to fit with usual formula
                exp_term = np.exp(-(B - 1j*(C - omega)) * T)
                numerator = A * (1 - exp_term)
                denominator = B - 1j*(C - omega)
                Zl = numerator / denominator

        return Zl

    def Resonator_transverse_imp(frequencies, Rs, Q, resonant_frequency, wake_length=None):
        """Calculates the transverse impedance of a resonator.

        This function calculates the transverse impedance of a resonator with shunt
        impedance `Rs`, quality factor `Q`, and resonant frequency `resonant_frequency`
        at different frequencies `frequencies`. The `wake_length` argument allows
        computing the impedance of a partially decayed wake over `wake_length`.

        Args:
            frequencies (np.ndarray): Array of frequencies values in Hz.
            Rs (float): Shunt impedance of the resonator in Ohm/m.
            Q (float): Quality factor of the resonator.
            resonant_frequency (float): Resonant frequency of the resonator in Hz.
            wake_length (float, optional): Additional parameter for the calculation
            of the impedance of a partially decayed wake.
                Defaults to None for the original (fully decayed wake) behavior.

        Returns:
            np.ndarray: Array of transverse impedance values [Ohm/m] at the corresponding frequencies.

        Notes:
            The fully decayed formula uses the generalized impedance formula
            (https://cds.cern.ch/record/192684/files/198812060.pdf) and can
            be used for any real positive value of `Q`.

            The partially decayed formula uses the formula derived in
            (Joly, S. thesis not published yet!)

            Moreover, it sets the impedance value to zero for zero frequencies
            in both cases.

            Units for this formula are:
                Rs: Ohm/m
                Q: dimensionless
                resonant_frequency: Hz
                wake_length: m
        """
        if wake_length is None:
            # Fully decayed wake
            zero_index = np.where(frequencies > 0)[0]  # find index of non-zero element
            if zero_index.size < frequencies.size:
                Zt = np.zeros_like(frequencies, dtype=complex)  # initialize Zt as 0
                Zt[zero_index] = resonant_frequency / frequencies[zero_index] * Rs / (1 + 1j*Q * (
                        frequencies[zero_index]/resonant_frequency -
                    resonant_frequency/frequencies[zero_index])) # calculate all Zt for non-zero frequencies

            else:
                Zt = resonant_frequency / frequencies * Rs / (1 + 1j*Q * (
                        frequencies/resonant_frequency - resonant_frequency/frequencies))

        else:
            # Partially decayed wake
            omega = 2 * np.pi * frequencies
            omega_r = 2 * np.pi * resonant_frequency
            c = 299792458.0 # speed of light in vacuum

            if Q < 0.5:
                raise ValueError("Quality factor Q must be larger than 0.5."
                                "The wake is unlikely to be partially decayed"
                                "for such a low quality factor otherwise.")

            A = Rs * omega_r / (Q * np.sqrt(1 - 1 / 4 / Q**2))
            B = omega_r / 2 / Q
            C = omega_r * np.sqrt(1 - 1 / 4 / Q**2)
            T = wake_length / c

            zero_index = np.where(frequencies > 0)[0]  # find index of non-zero element
            if zero_index.size < frequencies.size:
                exp_term = np.exp(-T * (B + 1j * omega[zero_index]))
                cos_term = np.cos(C * T)
                sin_term = (B + 1j * omega[zero_index]) / C * np.sin(C * T)
                denominator = C**2 + (B + 1j * omega[zero_index])**2
                Zt = np.zeros_like(frequencies, dtype=complex)  # initialize Zt as 0
                # calculate all Zt for non-zero frequencies
                Zt[zero_index] = 1j * A * C / denominator * (1 - exp_term * (cos_term + sin_term))

            else:
                exp_term = np.exp(-T * (B + 1j * omega))
                cos_term = np.cos(C * T)
                sin_term = (B + 1j * omega) / C * np.sin(C * T)
                denominator = C**2 + (B + 1j * omega)**2
                Zt = 1j * A * C / denominator * (1 - exp_term * (cos_term + sin_term))

        return Zt

    def n_Resonator_longitudinal_imp(frequencies, pars, wake_length=None):
        """Calculates the combined longitudinal impedance of multiple resonators.

        This function calculates the total longitudinal impedance of a system consisting
        of multiple resonators at different frequencies `frequencies`. Each resonator
        is defined by its parameters provided in a dictionary `dict_params`.

        Args:
            frequencies (np.ndarray): Array of frequencies values in Hz.
            dict_params (dict or ndarray): Dictionary containing resonator parameters. Keys are
                unique identifiers for each resonator, and values are lists containing
                the parameters in the following order:
                    - Rs (float): Shunt impedance of the resonator in Ohm.
                    - Q (float): Quality factor of the resonator.
                    - resonant_frequency (float): Resonant frequency of the resonator in Hz.
            wake_length (float, optional): Additional parameter for the calculation
            of the impedance of a partially decayed wake.
            Defaults to None for the original (fully decayed wake) behavior.

        Returns:
            np.ndarray: Array of combined longitudinal impedance values [Ohm] at
                the corresponding frequencies.

        Examples:
            >>> frequencies = np.linspace(0, 2.5e9, 1000)
            >>> pars = {
            ...     1: [1e6, 1, 1e9],
            ...     2: [2e6, 1, 5e8],
            ... }
            >>> # Or, use directly the result from the DE solver
            >>> pars = DE_model.evolutionParameters
            >>> impedance = n_Resonator_longitudinal_imp(frequencies, pars)
            >>> plt.plot(frequencies, impedance)
            >>> plt.xlabel("Frequency [Hz]")
            >>> plt.ylabel("Longitudinal Impedance [Ohm]")
            >>> plt.show()

        Notes:
            - This function assumes all resonators have the same type of longitudinal
                impedance formula implemented in `Resonator_longitudinal_imp`.
            - The combined impedance is calculated by summing the individual
                impedance contributions from each resonator at each frequency.
            - Resonator parameters should be positive values (except for the shunt impedance).
            Behavior for invalid values is not defined and may lead to errors.
        """
        if type(pars) is dict:
            dict_params = pars
        else:
            dict_params = pars_to_dict(pars) #takes list or ndarray

        if wake_length is None:
            # Fully decayed wake
            Zl = np.sum(Impedances.Resonator_longitudinal_imp(frequencies, *params) for params in dict_params.values())
        else:
            # Partially decayed wake
            Zl = np.sum(Impedances.Resonator_longitudinal_imp(frequencies, *params, wake_length=wake_length) for params in dict_params.values())

        return Zl

    def n_Resonator_transverse_imp(frequencies, pars, wake_length=None):
        """Calculates the combined transverse impedance of multiple resonators.

        This function calculates the total transverse impedance of a system consisting
        of multiple resonators at different frequencies `frequencies`. Each resonator
        is defined by its parameters provided in a dictionary `dict_params`.

        Args:
            frequencies (np.ndarray): Array of frequencies values in Hz.
            dict_params (dict or ndaray): Dictionary containing resonator parameters. Keys are
                unique identifiers for each resonator, and values are lists containing
                the parameters in the following order:
                    - Rs (float): Shunt impedance of the resonator in Ohm/m.
                    - Q (float): Quality factor of the resonator.
                    - resonant_frequency (float): Resonant frequency of the resonator in Hz.
            wake_length (float, optional): Additional parameter for the calculation
            of the impedance of a partially decayed wake.
            Defaults to None for the original (fully decayed wake) behavior.

        Returns:
            np.ndarray: Array of combined transverse impedance values [Ohm/m] at
                the corresponding frequencies.

        Examples:
            >>> frequencies = np.linspace(0, 2.5e9, 1000)
            >>> pars = {
            ...     1: [1e6, 1, 1e9],
            ...     2: [2e6, 1, 5e8],
            ... }
            >>> # Or, use directly the result from the DE solver
            >>> pars = DE_model.evolutionParameters
            >>> impedance = n_Resonator_transverse_imp(frequencies, pars)
            >>> plt.plot(frequencies, impedance)
            >>> plt.xlabel("Frequency [Hz]")
            >>> plt.ylabel("Transverse Impedance [Ohm/m]")
            >>> plt.show()

        Notes:
            - This function assumes all resonators have the same type of transverse
                impedance formula implemented in `Resonator_transverse_imp`.
            - The combined impedance is calculated by summing the individual
                impedance contributions from each resonator at each frequency.
            - Resonator parameters should be positive values (except for the shunt impedance).
            Behavior for invalid values is not defined and may lead to errors.
        """
        if type(pars) is dict:
            dict_params = pars
        else:
            dict_params = pars_to_dict(pars) #takes list or ndarray

        if wake_length is None:
            # Fully decayed wake
            Zt = np.sum(Impedances.Resonator_transverse_imp(frequencies, *params) for params in dict_params.values())
        else:
            # Partially decayed wake
            Zt = np.sum(Impedances.Resonator_transverse_imp(frequencies, *params, wake_length=wake_length) for params in dict_params.values())

        return Zt
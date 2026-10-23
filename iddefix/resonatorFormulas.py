#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:20:11 2020

@author: sjoly
"""

from typing import Sequence

import numpy as np
import numpy.typing as npt
from scipy import special as sp
from scipy.constants import c

from .utils import pars_to_dict

ArrayLike = npt.ArrayLike
ParameterDict = dict[int, Sequence[float]]
ParsType = ParameterDict | ArrayLike


class Wakes:
    # Longitudinal and transverse wake functions
    def Resonator_longitudinal_wake(
        times: ArrayLike,
        Rs: float,
        Q: float,
        resonant_frequency: float,
    ) -> np.ndarray:
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
            np.ndarray: Array of longitudinal wake function values [V/C] at the
            corresponding times.

        Notes:
            This formula uses the generalized wake function formula and can be
            used for any real positive value of `Q`.
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
        B = omega_r / (2 * Q)
        if Q == 0.5:
            Wl = 2 * Rs * omega_r * np.exp(-omega_r * times) * (1 - omega_r * times)
        elif Q < 0.5:
            D = omega_r * np.sqrt((1 / (4 * Q**2)) - 1)
            cos_term = np.cosh(D * times)
            sin_term = np.sinh(D * times)
            exp_term = np.exp(-B * times)
            Wl = Rs * omega_r / Q * exp_term * (cos_term - B / D * sin_term)
        else:
            C = omega_r * np.sqrt(1 - (1 / (4 * Q**2)))
            cos_term = np.cos(C * times)
            sin_term = np.sin(C * times)
            exp_term = np.exp(-B * times)
            Wl = Rs * omega_r / Q * exp_term * (cos_term - B / C * sin_term)
        Wl[times == 0.0] = omega_r / (2 * Q)  # Fundamental theorem of beam loading
        Wl[times < 0] = 0.0

        return Wl

    def Resonator_transverse_wake(
        times: ArrayLike,
        Rs: float,
        Q: float,
        resonant_frequency: float,
    ) -> np.ndarray:
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
        B = omega_r / (2 * Q)
        exp_term = np.exp(-B * times)
        if Q == 0.5:
            Wt = omega_r**2 * Rs / Q * exp_term * times
        elif Q < 0.5:
            D = omega_r * np.sqrt((1 / (4 * Q**2)) - 1)
            sin_term = np.sinh(D * times)
            Wt = omega_r**2 * Rs / (Q * D) * exp_term * sin_term
        else:
            C = omega_r * np.sqrt(1 - (1 / (4 * Q**2)))
            sin_term = np.sin(C * times)
            Wt = omega_r**2 * Rs / (Q * C) * exp_term * sin_term
        Wt[times < 0] = 0.0

        return Wt

    def n_Resonator_longitudinal_wake(times: ArrayLike, pars: ParsType) -> np.ndarray:
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
            dict_params = pars_to_dict(pars)  # takes list or ndarray

        Wl = sum(
            Wakes.Resonator_longitudinal_wake(times, *params)
            for params in dict_params.values()
        )
        return Wl

    def n_Resonator_transverse_wake(times: ArrayLike, pars: ParsType) -> np.ndarray:
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
            dict_params = pars_to_dict(pars)  # takes list or ndarray

        Wt = sum(
            Wakes.Resonator_transverse_wake(times, *params)
            for params in dict_params.values()
        )
        return Wt

    # Longitudinal and transverse wake potentials
    def Resonator_longitudinal_wake_potential(
        times: ArrayLike,
        Rs: float,
        Q: float,
        resonant_frequency: float,
        sigma: float = 1e-10,
        use_mpmath: bool = False,
    ) -> np.ndarray:
        """Calculate the longitudinal resonator wake potential for a Gaussian bunch.

        Args:
            times (array-like): Observation times in seconds.
            Rs (float): Shunt impedance in Ohm.
            Q (float): Positive quality factor.
            resonant_frequency (float): Resonant frequency in Hz.
            sigma (float): RMS bunch duration in seconds.
            use_mpmath (bool): Deprecated compatibility option. SciPy is used
                for all quality factors.

        Returns:
            np.ndarray: Longitudinal wake potential in V/C.

        Notes:
            Uses the critical, overdamped, or underdamped expression according
            to Q. Gaussian convolution can give a nonzero leading tail at
            negative observation times.
        """
        times = np.asarray(times, dtype=float)
        omega_r = 2 * np.pi * resonant_frequency
        gaussian = np.exp(-(times**2) / (2 * sigma**2))

        if Q == 0.5:
            a = omega_r
            argument = (a * sigma**2 - times) / (np.sqrt(2) * sigma)
            with np.errstate(over="ignore", invalid="ignore"):
                I_sigma = np.where(
                    argument >= 0,
                    gaussian * sp.erfcx(argument) / 2,
                    np.exp(a**2 * sigma**2 / 2 - a * times) * sp.erfc(argument) / 2,
                )
            J_sigma = (times - a * sigma**2) * I_sigma + (
                sigma / np.sqrt(2 * np.pi) * gaussian
            )
            W = 2 * Rs * omega_r * (I_sigma - omega_r * J_sigma)

        elif Q < 0.5:
            B = omega_r / (2 * Q)
            D = omega_r * np.sqrt(1 / (4 * Q**2) - 1)
            a_plus = B + D
            a_minus = omega_r**2 / a_plus
            I_values = []
            for a in (a_minus, a_plus):
                argument = (a * sigma**2 - times) / (np.sqrt(2) * sigma)
                with np.errstate(over="ignore", invalid="ignore"):
                    I_sigma = np.where(
                        argument >= 0,
                        gaussian * sp.erfcx(argument) / 2,
                        np.exp(a**2 * sigma**2 / 2 - a * times) * sp.erfc(argument) / 2,
                    )
                I_values.append(I_sigma)
            I_minus, I_plus = I_values
            W = Rs * omega_r / (2 * Q * D) * (a_plus * I_plus - a_minus * I_minus)

        else:
            B = omega_r / (2 * Q)
            C = omega_r * np.sqrt(1 - 1 / (4 * Q**2))
            a = B - 1j * C
            argument = (a * sigma**2 - times) / (np.sqrt(2) * sigma)
            # K_sigma(a, t) in theory.md; erfcx(z) = wofz(i*z).
            with np.errstate(over="ignore", invalid="ignore"):
                K_sigma = np.where(
                    argument.real >= 0,
                    gaussian * sp.wofz(1j * argument) / 2,
                    np.exp(a**2 * sigma**2 / 2 - a * times)
                    - gaussian * sp.wofz(-1j * argument) / 2,
                )
            W = Rs * omega_r / Q * np.real((1 + 1j * B / C) * K_sigma)

        return W

    def n_Resonator_longitudinal_wake_potential(
        times: ArrayLike,
        pars: ParsType,
        sigma: float = 1e-10,
    ) -> np.ndarray:
        if type(pars) is dict:
            dict_params = pars
        else:
            dict_params = pars_to_dict(pars)  # takes list or ndarray

        Wpl = sum(
            Wakes.Resonator_longitudinal_wake_potential(times, *params, sigma=sigma)
            for params in dict_params.values()
        )
        return Wpl

    def Resonator_transverse_wake_potential(
        times: ArrayLike,
        Rs: float,
        Q: float,
        resonant_frequency: float,
        sigma: float = 1e-10,
        use_mpmath: bool = False,
    ) -> np.ndarray:
        """Calculate the transverse resonator wake potential for a Gaussian bunch.

        Args:
            times (array-like): Observation times in seconds.
            Rs (float): Shunt impedance in Ohm/m.
            Q (float): Positive quality factor.
            resonant_frequency (float): Resonant frequency in Hz.
            sigma (float): RMS bunch duration in seconds.
            use_mpmath (bool): Deprecated compatibility option. SciPy is used
                for all quality factors.

        Returns:
            np.ndarray: Transverse wake potential in V/C/m.

        Notes:
            Uses the critical, overdamped, or underdamped expression according
            to Q. Gaussian convolution can give a nonzero leading tail at
            negative observation times.
        """
        times = np.asarray(times, dtype=float)
        omega_r = 2 * np.pi * resonant_frequency
        gaussian = np.exp(-(times**2) / (2 * sigma**2))

        if Q == 0.5:
            a = omega_r
            argument = (a * sigma**2 - times) / (np.sqrt(2) * sigma)
            with np.errstate(over="ignore", invalid="ignore"):
                I_sigma = np.where(
                    argument >= 0,
                    gaussian * sp.erfcx(argument) / 2,
                    np.exp(a**2 * sigma**2 / 2 - a * times) * sp.erfc(argument) / 2,
                )
            J_sigma = (times - a * sigma**2) * I_sigma + (
                sigma / np.sqrt(2 * np.pi) * gaussian
            )
            W = 2 * Rs * omega_r**2 * J_sigma

        elif Q < 0.5:
            B = omega_r / (2 * Q)
            D = omega_r * np.sqrt(1 / (4 * Q**2) - 1)
            a_plus = B + D
            a_minus = omega_r**2 / a_plus
            I_values = []
            for a in (a_minus, a_plus):
                argument = (a * sigma**2 - times) / (np.sqrt(2) * sigma)
                with np.errstate(over="ignore", invalid="ignore"):
                    I_sigma = np.where(
                        argument >= 0,
                        gaussian * sp.erfcx(argument) / 2,
                        np.exp(a**2 * sigma**2 / 2 - a * times) * sp.erfc(argument) / 2,
                    )
                I_values.append(I_sigma)
            I_minus, I_plus = I_values
            W = Rs * omega_r**2 / (2 * Q * D) * (I_minus - I_plus)

        else:
            B = omega_r / (2 * Q)
            C = omega_r * np.sqrt(1 - 1 / (4 * Q**2))
            a = B - 1j * C
            argument = (a * sigma**2 - times) / (np.sqrt(2) * sigma)
            # K_sigma(a, t) in theory.md; erfcx(z) = wofz(i*z).
            with np.errstate(over="ignore", invalid="ignore"):
                K_sigma = np.where(
                    argument.real >= 0,
                    gaussian * sp.wofz(1j * argument) / 2,
                    np.exp(a**2 * sigma**2 / 2 - a * times)
                    - gaussian * sp.wofz(-1j * argument) / 2,
                )
            W = Rs * omega_r**2 / (Q * C) * np.imag(K_sigma)

        return W

    def n_Resonator_transverse_wake_potential(
        times: ArrayLike,
        pars: ParsType,
        sigma: float = 1e-10,
    ) -> np.ndarray:
        if type(pars) is dict:
            dict_params = pars
        else:
            dict_params = pars_to_dict(pars)  # takes list or ndarray

        Wpt = sum(
            Wakes.Resonator_transverse_wake_potential(times, *params, sigma=sigma)
            for params in dict_params.values()
        )
        return Wpt


class Impedances:
    # Longitudinal and transverse impedance functions
    def Resonator_longitudinal_imp(
        frequencies: ArrayLike,
        Rs: float,
        Q: float,
        resonant_frequency: float,
        wake_length: float | None = None,
    ) -> np.ndarray:
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

            The partially decayed formula is the finite Fourier transform of
            the longitudinal wake and supports any real positive value of `Q`.

            The fully decayed formula sets the impedance value to zero at zero
            frequency. A finite wake can have a nonzero value there.

            Units for this formula are:
                Rs: Ohm
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
            positive = frequencies > 0
            Zl = np.zeros_like(frequencies, dtype=complex)
            selected_frequencies = frequencies[positive]
            Zl[positive] = (
                Rs
                * selected_frequencies
                * resonant_frequency
                / (
                    selected_frequencies * resonant_frequency
                    + 1j * Q * (selected_frequencies**2 - resonant_frequency**2)
                )
            )

        else:
            # Partially decayed wake
            omega = 2 * np.pi * frequencies
            omega_r = 2 * np.pi * resonant_frequency

            B = omega_r / 2 / Q
            T = wake_length / c
            p = B + 1j * omega
            if Q < 0.5:
                D = omega_r * np.sqrt(1 / 4 / Q**2 - 1)
                fast_rate = B + D
                # B - D = omega_r**2 / (B + D) avoids cancellation at low Q.
                slow_rate = omega_r**2 / fast_rate
                slow_decay = np.exp(-slow_rate * T)
                fast_to_slow = np.exp(-2 * D * T)
                phase = np.exp(-1j * omega * T)
                decay_cosh = phase * slow_decay * (1 + fast_to_slow) / 2
                decay_sinh_over_D = (
                    phase * slow_decay * (-np.expm1(-2 * D * T)) / (2 * D)
                )
                numerator = (
                    1j * omega
                    - 1j * omega * decay_cosh
                    + (omega_r**2 + 1j * B * omega) * decay_sinh_over_D
                )
                denominator = (slow_rate + 1j * omega) * (fast_rate + 1j * omega)
            else:
                # The formula remains finite at the critical value Q = 0.5.
                C = omega_r * np.sqrt(1 - 1 / 4 / Q**2)
                exp_term = np.exp(-p * T)
                cos_term = np.cos(C * T)
                # sin(C*T)/C has the finite limit T at the critical Q = 0.5.
                sin_over_C = T * np.sinc(C * T / np.pi)
                numerator = 1j * omega + exp_term * (
                    -1j * omega * cos_term + C**2 * sin_over_C + B * p * sin_over_C
                )
                denominator = C**2 + p**2
            Zl = Rs * omega_r / Q * numerator / denominator

        return Zl

    def Resonator_transverse_imp(
        frequencies: ArrayLike,
        Rs: float,
        Q: float,
        resonant_frequency: float,
        wake_length: float | None = None,
    ) -> np.ndarray:
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

            The partially decayed formula is the finite Fourier transform of
            the transverse wake and supports any real positive value of `Q`.

            At zero frequency, the fully decayed impedance has the analytical
            value ``1j * Rs / Q``. The partially decayed value is the finite
            transverse wake area multiplied by ``1j`` and is generally nonzero.

            Units for this formula are:
                Rs: Ohm/m
                Q: dimensionless
                resonant_frequency: Hz
                wake_length: m
        """
        if wake_length is None:
            # Fully decayed wake
            Zt = (
                Rs
                * resonant_frequency**2
                / (
                    frequencies * resonant_frequency
                    + 1j * Q * (frequencies**2 - resonant_frequency**2)
                )
            )

        else:
            # Partially decayed wake
            omega = 2 * np.pi * frequencies
            omega_r = 2 * np.pi * resonant_frequency

            B = omega_r / 2 / Q
            T = wake_length / c
            prefactor = Rs * omega_r**2 / Q
            if Q < 0.5:
                D = omega_r * np.sqrt(1 / 4 / Q**2 - 1)
                fast_rate = B + D
                slow_rate = omega_r**2 / fast_rate
                slow_decay = np.exp(-slow_rate * T)
                fast_to_slow = np.exp(-2 * D * T)
                phase = np.exp(-1j * omega * T)
                decay_cosh = phase * slow_decay * (1 + fast_to_slow) / 2
                decay_sinh_over_D = (
                    phase * slow_decay * (-np.expm1(-2 * D * T)) / (2 * D)
                )
                p = B + 1j * omega
                denominator = (slow_rate + 1j * omega) * (fast_rate + 1j * omega)
                Zt = (
                    1j
                    * prefactor
                    / denominator
                    * (1 - decay_cosh - p * decay_sinh_over_D)
                )

            else:
                C = omega_r * np.sqrt(1 - 1 / 4 / Q**2)
                # The prefactor and sin(C*T)/C have finite limits at Q = 0.5.
                sin_over_C = T * np.sinc(C * T / np.pi)
                p = B + 1j * omega
                exp_term = np.exp(-T * p)
                cos_term = np.cos(C * T)
                sin_term = p * sin_over_C
                denominator = C**2 + p**2
                Zt = (
                    1j
                    * prefactor
                    / denominator
                    * (1 - exp_term * (cos_term + sin_term))
                )

        return Zt

    def n_Resonator_longitudinal_imp(
        frequencies: ArrayLike,
        pars: ParsType,
        wake_length: float | None = None,
    ) -> np.ndarray:
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
            dict_params = pars_to_dict(pars)  # takes list or ndarray

        if wake_length is None:
            # Fully decayed wake
            Zl = sum(
                Impedances.Resonator_longitudinal_imp(frequencies, *params)
                for params in dict_params.values()
            )
        else:
            # Partially decayed wake
            Zl = sum(
                Impedances.Resonator_longitudinal_imp(
                    frequencies, *params, wake_length=wake_length
                )
                for params in dict_params.values()
            )

        return Zl

    def n_Resonator_transverse_imp(
        frequencies: ArrayLike,
        pars: ParsType,
        wake_length: float | None = None,
    ) -> np.ndarray:
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
            dict_params = pars_to_dict(pars)  # takes list or ndarray

        if wake_length is None:
            # Fully decayed wake
            Zt = sum(
                Impedances.Resonator_transverse_imp(frequencies, *params)
                for params in dict_params.values()
            )
        else:
            # Partially decayed wake
            Zt = sum(
                Impedances.Resonator_transverse_imp(
                    frequencies, *params, wake_length=wake_length
                )
                for params in dict_params.values()
            )

        return Zt

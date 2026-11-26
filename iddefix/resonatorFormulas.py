"""
Created on Mon Mar 23 13:20:11 2020

@author: sjoly
"""

import warnings
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
    """Analytical longitudinal and transverse resonator wakes and potentials.

    Call methods on the class, for example Wakes.Resonator_longitudinal_wake(...).
    Time grids are in seconds. Longitudinal Rs is in Ohm and transverse Rs in
    Ohm/m; wake and potential outputs are V/C and V/C/m, respectively.
    Multi-resonator methods accept a flat (Rs, Q, fr) vector or a dictionary
    whose values are those triples. Frequencies fr are in Hz and Q is dimensionless.
    """

    # Longitudinal and transverse wake functions

    def Resonator_longitudinal_wake(
        times: ArrayLike,
        Rs: float,
        Q: float,
        fr: float,
    ) -> np.ndarray:
        """Evaluate the longitudinal wake of one resonator.

        Parameters
        ----------
        times : array_like
            Time samples [s], with positive times behind the source.
        Rs : float
            Shunt impedance [Ohm].
        Q : float
            Positive quality factor, including the critical value 0.5.
        fr : float
            Positive resonant frequency [Hz].

        Returns
        -------
        ndarray
            Wake samples [V/C] on the input grid. Negative-time samples are zero.

        Notes
        -----
        Uses the generalized resonator wake, including overdamped and critical
        limits; see https://cds.cern.ch/record/192684/files/198812060.pdf.
        """

        times = np.asarray(times)
        positive_times = np.maximum(times, 0)
        omega_r = 2 * np.pi * fr
        if Q == 0.5:
            Wl = (
                2
                * Rs
                * omega_r
                * np.exp(-omega_r * positive_times)
                * (1 - omega_r * positive_times)
            )
        elif Q < 0.5:
            # Use decaying exponentials to avoid overflow in cosh and sinh
            B = omega_r / (2 * Q)
            D = omega_r * np.sqrt(1 / (4 * Q**2) - 1)
            fast_rate = B + D
            slow_rate = omega_r**2 / fast_rate
            slow_decay = np.exp(-slow_rate * positive_times)
            fast_to_slow = np.exp(-2 * D * positive_times)
            decay_cosh = slow_decay * (1 + fast_to_slow) / 2
            decay_sinh_over_D = (
                slow_decay * (-np.expm1(-2 * D * positive_times)) / (2 * D)
            )
            Wl = Rs * omega_r / Q * (decay_cosh - B * decay_sinh_over_D)
        else:
            omega_r_bar = omega_r * np.sqrt(1 - (1 / (4 * Q**2)))
            cos_term = np.cos(omega_r_bar * positive_times)
            sin_term = np.sin(omega_r_bar * positive_times)
            exp_term = np.exp(-omega_r * positive_times / (2 * Q))
            Wl = (
                Rs
                * omega_r
                / Q
                * exp_term
                * (cos_term - omega_r / (2 * Q * omega_r_bar) * sin_term)
            )
        Wl[times == 0.0] = omega_r / (2 * Q)  # Fundamental theorem of beam loading
        Wl[times < 0] = 0.0

        return Wl

    def Resonator_transverse_wake(
        times: ArrayLike,
        Rs: float,
        Q: float,
        fr: float,
    ) -> np.ndarray:
        """Evaluate the transverse wake of one resonator.

        Parameters
        ----------
        times : array_like
            Time samples [s], with positive times behind the source.
        Rs : float
            Shunt impedance [Ohm/m].
        Q : float
            Positive quality factor, including the critical value 0.5.
        fr : float
            Positive resonant frequency [Hz].

        Returns
        -------
        ndarray
            Wake samples [V/C/m] on the input grid. Negative-time samples are zero.

        Notes
        -----
        Uses the generalized resonator wake, including overdamped and critical
        limits; see https://cds.cern.ch/record/192684/files/198812060.pdf.
        """

        times = np.asarray(times)
        positive_times = np.maximum(times, 0)
        omega_r = 2 * np.pi * fr
        if Q == 0.5:
            exp_term = np.exp(-omega_r * positive_times / 2 / Q)
            Wt = omega_r**2 * Rs / Q * exp_term * positive_times
        elif Q < 0.5:
            # Use decaying exponentials to avoid overflow in sinh
            B = omega_r / (2 * Q)
            D = omega_r * np.sqrt(1 / (4 * Q**2) - 1)
            fast_rate = B + D
            slow_rate = omega_r**2 / fast_rate
            slow_decay = np.exp(-slow_rate * positive_times)
            decay_sinh_over_D = (
                slow_decay * (-np.expm1(-2 * D * positive_times)) / (2 * D)
            )
            Wt = omega_r**2 * Rs / Q * decay_sinh_over_D
        else:
            omega_r_bar = omega_r * np.sqrt(1 - (1 / (4 * Q**2)))
            sqrt_term = np.sqrt(1 - (1 / (4 * Q * Q)))
            sin_term = np.sin(omega_r * sqrt_term * positive_times)
            exp_term = np.exp(-omega_r * positive_times / 2 / Q)
            Wt = omega_r**2 * Rs / (Q * omega_r_bar) * exp_term * sin_term
        Wt[times < 0] = 0.0

        return Wt

    def n_Resonator_longitudinal_wake(times: ArrayLike, pars: ParsType) -> np.ndarray:
        """Sum longitudinal wake functions for the supplied resonators.

        Parameters
        ----------
        times : array_like
            Time grid [s] about the source or bunch center.
        pars : array_like or dict
            Flat (Rs, Q, fr) vector or dictionary of parameter triples.

        Returns
        -------
        ndarray
            Sum of the resonator contributions, in the units stated in Notes.

        Notes
        -----
        ``times`` is a NumPy time grid [s]. ``pars`` is a flat (Rs, Q, fr) vector
        or a dictionary of triples, with Rs in Ohm, positive Q, and fr in Hz.
        Returns samples in V/C. Each mode uses Resonator_longitudinal_wake,
        including its causal convention and support for any positive Q.
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
        """Sum transverse wake functions for the supplied resonators.

        Parameters
        ----------
        times : array_like
            Time grid [s] about the source or bunch center.
        pars : array_like or dict
            Flat (Rs, Q, fr) vector or dictionary of parameter triples.

        Returns
        -------
        ndarray
            Sum of the resonator contributions, in the units stated in Notes.

        Notes
        -----
        ``times`` is a NumPy time grid [s]. ``pars`` is a flat (Rs, Q, fr) vector
        or a dictionary of triples, with Rs in Ohm/m, positive Q, and fr in Hz.
        Returns samples in V/C/m. Each mode uses Resonator_transverse_wake,
        including its causal convention and support for any positive Q.
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
        fr: float,
        sigma: float = 1e-10,
        use_mpmath: bool = False,
    ) -> np.ndarray:
        """Evaluate one longitudinal resonator's Gaussian wake potential.

        Parameters
        ----------
        times : ndarray
            Times [s] measured from the bunch center; positive times are behind it.
        Rs : float
            Shunt impedance [Ohm].
        Q : float
            Quality factor for the analytic Gaussian expression. Use Q > 0.5;
            its critical-damping limit is not implemented.
        fr : float
            Positive resonant frequency [Hz].
        sigma : float, optional
            RMS Gaussian bunch duration [s]. Default is 1e-10.
        use_mpmath : bool, optional
            Use the optional mpmath backend. Default is False (NumPy/SciPy).

        Returns
        -------
        ndarray
            Wake potential [V/C] on the requested grid. It can extend before
            time zero because the Gaussian bunch has a finite duration.

        Notes
        -----
        The Gaussian closed form follows Chao's Handbook, section 3.2, and the
        resonator formulas in https://cds.cern.ch/record/192684/files/198812060.pdf.
        """
        omegar = 2 * np.pi * fr
        kr = omegar * (1 - 1 / (4 * Q**2)) ** 0.5
        alphar = omegar / (2 * Q)
        cstsin = -Rs * omegar**2 / (4 * Q**2 * kr)
        cstcos = Rs * omegar / (2 * Q)

        mpmath_is_installed = True

        if use_mpmath:
            try:
                pass
            except Exception:
                raise ImportError(
                    "mpmath is not installed. Please install it to use this function."
                )
                mpmath_is_installed = False

        if use_mpmath and mpmath_is_installed:
            from mpmath import erfc, exp, im, matrix, re

            cst = exp((alphar**2 - kr**2) * sigma**2 / 2)
            times_mp = matrix(times)
            erfc_arg = -(times_mp - alphar * sigma**2 + 1j * kr * sigma**2) / (
                np.sqrt(2) * sigma
            )
            erfc_v = erfc_arg.apply(erfc)
            arg_expo1 = -alphar * times_mp
            expo1 = arg_expo1.apply(exp)
            arg_expo2 = 1j * (kr * times_mp - alphar * kr * sigma**2)
            expo2 = arg_expo2.apply(exp)
            im_expo_erfc = matrix([im(ex * er) for (ex, er) in zip(expo2, erfc_v)])
            re_expo_erfc = matrix([re(ex * er) for (ex, er) in zip(expo2, erfc_v)])
            W = np.hstack(
                [
                    cst * ex1 * (cstsin * im_exr + cstcos * re_exr)
                    for (ex1, im_exr, re_exr) in zip(expo1, im_expo_erfc, re_expo_erfc)
                ]
            )
        else:
            cst = np.exp((alphar**2 - kr**2) * sigma**2 / 2)
            erfc_v = sp.erfc(
                -(times - alphar * sigma**2 + 1j * kr * sigma**2) / (np.sqrt(2) * sigma)
            )
            expo1 = np.exp(-alphar * times)
            expo2 = np.exp(1j * (kr * times - alphar * kr * sigma**2))
            W = (
                cst
                * expo1
                * (cstsin * np.imag(expo2 * erfc_v) + cstcos * np.real(expo2 * erfc_v))
            )

        return W

    def n_Resonator_longitudinal_wake_potential(
        times: ArrayLike,
        pars: ParsType,
        sigma: float = 1e-10,
    ) -> np.ndarray:
        """Sum Gaussian wake potentials for longitudinal resonators.

        Parameters
        ----------
        times : array_like
            Time grid [s] about the source or bunch center.
        pars : array_like or dict
            Flat (Rs, Q, fr) vector or dictionary of parameter triples.
        sigma : float, optional
            Shared RMS Gaussian bunch duration [s]. Default is 1e-10.

        Returns
        -------
        ndarray
            Sum of the resonator contributions, in the units stated in Notes.

        Notes
        -----
        ``times`` is a NumPy time grid [s] about the bunch center. ``pars`` is a
        flat (Rs, Q, fr) vector or a dictionary of triples (Rs in Ohm, fr in Hz).
        ``sigma`` is the shared RMS bunch duration [s], default 1e-10.
        Returns samples in V/C. Each mode uses Resonator_longitudinal_wake_potential;
        its quality-factor restrictions apply to every mode.
        """
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
        fr: float,
        sigma: float = 1e-10,
        use_mpmath: bool = False,
    ) -> np.ndarray:
        """Evaluate one transverse resonator's Gaussian wake potential.

        Parameters
        ----------
        times : ndarray
            Times [s] measured from the bunch center; positive times are behind it.
        Rs : float
            Shunt impedance [Ohm/m].
        Q : float
            Quality factor for the analytic Gaussian expression. Use Q > 0.5;
            its critical-damping limit is not implemented.
        fr : float
            Positive resonant frequency [Hz].
        sigma : float, optional
            RMS Gaussian bunch duration [s]. Default is 1e-10.
        use_mpmath : bool, optional
            Use the optional mpmath backend. Default is False (NumPy/SciPy).

        Returns
        -------
        ndarray
            Wake potential [V/C/m] on the requested grid. It can extend before
            time zero because the Gaussian bunch has a finite duration.

        Notes
        -----
        The Gaussian closed form follows Chao's Handbook, section 3.2, and the
        resonator formulas in https://cds.cern.ch/record/192684/files/198812060.pdf.
        """

        omegar = 2 * np.pi * fr
        kr = omegar * (1 - 1 / (4 * Q**2)) ** 0.5
        alphar = omegar / (2 * Q)

        mpmath_is_installed = True

        if use_mpmath:
            try:
                pass
            except Exception:
                raise ImportError(
                    "mpmath is not installed. Please install it to use this function."
                )
                mpmath_is_installed = False

        if use_mpmath and mpmath_is_installed:
            from mpmath import erfc, exp, im, matrix

            times_mp = matrix(times)
            cst = (
                Rs * omegar**2 / (2 * Q * kr) * exp((alphar**2 - kr**2) * sigma**2 / 2)
            )
            erfc_arg = -(times_mp - alphar * sigma**2 + 1j * kr * sigma**2) / (
                np.sqrt(2) * sigma
            )
            erfc_v = erfc_arg.apply(erfc)
            arg_expo1 = -alphar * times_mp
            expo1 = arg_expo1.apply(exp)
            arg_expo2 = 1j * (kr * times_mp - alphar * kr * sigma**2)
            expo2 = arg_expo2.apply(exp)
            im_expo_erfc = matrix([im(ex * er) for (ex, er) in zip(expo2, erfc_v)])
            W = np.hstack(
                [cst * ex1 * im_exr for (ex1, im_exr) in zip(expo1, im_expo_erfc)]
            )
        else:
            cst = (
                Rs
                * omegar**2
                / (2 * Q * kr)
                * np.exp((alphar**2 - kr**2) * sigma**2 / 2)
            )
            erf_v = sp.erf(
                (times - alphar * sigma**2 + 1j * kr * sigma**2) / (np.sqrt(2) * sigma)
            )
            expo1 = np.exp(-alphar * times)
            expo2 = np.exp(1j * (kr * times - alphar * kr * sigma**2))
            W = cst * expo1 * np.imag(expo2 * (1 + erf_v))

        return W

    def n_Resonator_transverse_wake_potential(
        times: ArrayLike,
        pars: ParsType,
        sigma: float = 1e-10,
    ) -> np.ndarray:
        """Sum Gaussian wake potentials for transverse resonators.

        Parameters
        ----------
        times : array_like
            Time grid [s] about the source or bunch center.
        pars : array_like or dict
            Flat (Rs, Q, fr) vector or dictionary of parameter triples.
        sigma : float, optional
            Shared RMS Gaussian bunch duration [s]. Default is 1e-10.

        Returns
        -------
        ndarray
            Sum of the resonator contributions, in the units stated in Notes.

        Notes
        -----
        ``times`` is a NumPy time grid [s] about the bunch center. ``pars`` is a
        flat (Rs, Q, fr) vector or a dictionary of triples (Rs in Ohm/m, fr in Hz).
        ``sigma`` is the shared RMS bunch duration [s], default 1e-10.
        Returns samples in V/C/m. Each mode uses Resonator_transverse_wake_potential;
        its quality-factor restrictions apply to every mode.
        """
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
    """Analytical longitudinal and transverse resonator impedance functions.

    Call methods on the class with NumPy frequency arrays in Hz. Longitudinal
    Rs and impedance are in Ohm; transverse values are in Ohm/m. Q is positive
    and dimensionless, and fr is in Hz. Multi-resonator methods
    accept a flat (Rs, Q, fr) vector or a dictionary of parameter triples.
    A wake_length in metres selects the finite-duration Fourier transform;
    None selects the fully decayed formula.
    """

    @staticmethod  # Temporary deprecation warning
    def _resolve_fr(fr: float | None, resonant_frequency: float | None) -> float:
        """Resolve the preferred frequency keyword and its deprecated alias."""
        if resonant_frequency is not None:
            if fr is not None:
                raise TypeError("Pass only one of 'fr' and 'resonant_frequency'")
            warnings.warn(
                "'resonant_frequency' is deprecated; use 'fr' instead",
                DeprecationWarning,
                stacklevel=3,
            )
            return resonant_frequency
        if fr is None:
            raise TypeError("Missing resonant frequency: pass 'fr'")
        return fr

    # Longitudinal and transverse impedance functions

    def Resonator_longitudinal_imp(
        frequencies: ArrayLike,
        Rs: float,
        Q: float,
        fr: float | None = None,
        wake_length: float | None = None,
        *,
        resonant_frequency: float | None = None,
    ) -> np.ndarray:
        """Evaluate the complex longitudinal impedance of one resonator.

        Parameters
        ----------
        frequencies : ndarray
            Frequency samples [Hz].
        Rs : float
            Shunt impedance [Ohm].
        Q : float
            Positive quality factor; overdamped and critical values are supported.
        fr : float
            Positive resonant frequency [Hz].
        wake_length : float or None, optional
            Wake truncation length [m]. None uses the fully decayed expression;
            a finite length uses the truncated wake transform.
        resonant_frequency : float or None, optional
            [!] Deprecated keyword-only alias for fr. Emits DeprecationWarning.
            Do not pass both names.

        Returns
        -------
        ndarray
            Complex impedance [Ohm] on the requested frequency grid.

        Raises
        ------
        TypeError
            If neither frequency keyword is supplied, or both are supplied.

        Notes
        -----
        The fully decayed value at DC is zero; finite wakes can have nonzero DC.
        See https://cds.cern.ch/record/192684/files/198812060.pdf for the resonator
        convention. Finite-length formulas also support any positive Q.
        """
        fr = Impedances._resolve_fr(fr, resonant_frequency)
        if wake_length is None:
            # Fully decayed wake
            positive = frequencies > 0
            Zl = np.zeros_like(frequencies, dtype=complex)
            selected_frequencies = frequencies[positive]
            Zl[positive] = (
                Rs
                * selected_frequencies
                * fr
                / (
                    selected_frequencies * fr
                    + 1j * Q * (selected_frequencies**2 - fr**2)
                )
            )

        else:
            # Partially decayed wake
            omega = 2 * np.pi * frequencies
            omega_r = 2 * np.pi * fr

            B = omega_r / 2 / Q
            T = wake_length / c
            s = B + 1j * omega
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
                exp_term = np.exp(-s * T)
                cos_term = np.cos(C * T)
                # sin(C*T)/C has the finite limit T at the critical Q = 0.5.
                sin_over_C = T * np.sinc(C * T / np.pi)
                numerator = 1j * omega + exp_term * (
                    -1j * omega * cos_term + C**2 * sin_over_C + B * s * sin_over_C
                )
                denominator = C**2 + s**2
            Zl = Rs * omega_r / Q * numerator / denominator

        return Zl

    def Resonator_transverse_imp(
        frequencies: ArrayLike,
        Rs: float,
        Q: float,
        fr: float | None = None,
        wake_length: float | None = None,
        *,
        resonant_frequency: float | None = None,
    ) -> np.ndarray:
        """Evaluate the complex transverse impedance of one resonator.

        Parameters
        ----------
        frequencies : ndarray
            Frequency samples [Hz].
        Rs : float
            Shunt impedance [Ohm/m].
        Q : float
            Positive quality factor; overdamped and critical values are supported.
        fr : float
            Positive resonant frequency [Hz].
        wake_length : float or None, optional
            Wake truncation length [m]. None uses the fully decayed expression;
            a finite length uses the truncated wake transform.
        resonant_frequency : float or None, optional
            [!] Deprecated keyword-only alias for fr. Emits DeprecationWarning.
            Do not pass both names.

        Returns
        -------
        ndarray
            Complex impedance [Ohm/m] on the requested frequency grid.

        Raises
        ------
        TypeError
            If neither frequency keyword is supplied, or both are supplied.

        Notes
        -----
        The fully decayed value at DC is 1j*Rs/Q; finite wakes can also have nonzero DC.
        See https://cds.cern.ch/record/192684/files/198812060.pdf for the resonator
        convention. Finite-length formulas also support any positive Q.
        """
        fr = Impedances._resolve_fr(fr, resonant_frequency)
        if wake_length is None:
            # Fully decayed wake
            Zt = Rs * fr**2 / (frequencies * fr + 1j * Q * (frequencies**2 - fr**2))

        else:
            # Partially decayed wake
            omega = 2 * np.pi * frequencies
            omega_r = 2 * np.pi * fr

            B = omega_r / 2 / Q
            T = wake_length / c
            AC = Rs * omega_r**2 / Q
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
                s = B + 1j * omega
                denominator = (slow_rate + 1j * omega) * (fast_rate + 1j * omega)
                Zt = 1j * AC / denominator * (1 - decay_cosh - s * decay_sinh_over_D)

            else:
                C = omega_r * np.sqrt(1 - 1 / 4 / Q**2)
                # A*C and sin(C*T)/C have finite limits at the critical Q = 0.5.
                sin_over_C = T * np.sinc(C * T / np.pi)
                s = B + 1j * omega
                exp_term = np.exp(-T * s)
                cos_term = np.cos(C * T)
                sin_term = s * sin_over_C
                denominator = C**2 + s**2
                Zt = 1j * AC / denominator * (1 - exp_term * (cos_term + sin_term))

        return Zt

    def n_Resonator_longitudinal_imp(
        frequencies: ArrayLike,
        pars: ParsType,
        wake_length: float | None = None,
    ) -> np.ndarray:
        """Sum complex longitudinal impedances for the supplied resonators.

        Parameters
        ----------
        frequencies : array_like
            Evaluation frequency grid [Hz].
        pars : array_like or dict
            Flat (Rs, Q, fr) vector or dictionary of parameter triples.
        wake_length : float or None, optional
            Wake truncation length [m]. None uses fully decayed impedance.

        Returns
        -------
        ndarray
            Sum of the resonator contributions, in the units stated in Notes.

        Notes
        -----
        ``frequencies`` is a NumPy grid [Hz]. ``pars`` is a flat (Rs, Q, fr) vector
        or a dictionary of triples, with Rs in Ohm, positive Q, and fr in Hz.
        ``wake_length`` [m] defaults to None (fully decayed); a supplied value
        truncates every mode at that length. Returns complex samples in Ohm.
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
        """Sum complex transverse impedances for the supplied resonators.

        Parameters
        ----------
        frequencies : array_like
            Evaluation frequency grid [Hz].
        pars : array_like or dict
            Flat (Rs, Q, fr) vector or dictionary of parameter triples.
        wake_length : float or None, optional
            Wake truncation length [m]. None uses fully decayed impedance.

        Returns
        -------
        ndarray
            Sum of the resonator contributions, in the units stated in Notes.

        Notes
        -----
        ``frequencies`` is a NumPy grid [Hz]. ``pars`` is a flat (Rs, Q, fr) vector
        or a dictionary of triples, with Rs in Ohm/m, positive Q, and fr in Hz.
        ``wake_length`` [m] defaults to None (fully decayed); a supplied value
        truncates every mode at that length. Returns complex samples in Ohm/m.
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

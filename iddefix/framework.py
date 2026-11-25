#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:20:11 2020

@author: sjoly
@modified by: MaltheRaschke, edelafue
"""

from __future__ import annotations

import re
from copy import copy, deepcopy
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
import numpy.typing as npt
from scipy.constants import c
from scipy.optimize import minimize

from . import baseline
from ._table import display_parameter_bounds, display_resonator_table
from .objectiveFunctions import ObjectiveFunctions as obj
from .resonatorFormulas import Impedances as imp
from .resonatorFormulas import Wakes as wak
from .smartBoundDetermination import SmartBoundDetermination
from .solvers import Solvers
from .uncertainties import get_uncertainties as calculate_uncertainties
from .utils import compute_deconvolution, compute_fft

if TYPE_CHECKING:
    from pymoo.core.result import Result

# type-checking aliases
ArrayLike = npt.ArrayLike
ParameterBounds = list[tuple[float, float]]
ObjectiveCallable = Callable[..., float]


class EvolutionaryAlgorithm:
    def __init__(
        self,
        x_data: ArrayLike,
        y_data: ArrayLike,
        N_resonators: int | None = None,
        parameterBounds: ParameterBounds | None = None,
        plane: str = "longitudinal",
        fitFunction: str = "impedance",
        objectiveFunction: ObjectiveCallable | str | None = None,
        wake_length: float | None = None,
        sigma: float | None = None,
        uncertainty_warning: float = 0.2,
        parameterEstimates: ArrayLike | None = None,
        compute_uncertainties: bool = False,
    ) -> None:
        """Fit longitudinal or transverse resonators to impedance or wake data.

        Parameters
        ----------
        x_data : array_like
            Frequencies [Hz] for impedance, or times [s] for wakes and potentials.
            Time-domain transforms require uniform sampling about the bunch center.
        y_data : array_like
            Paired impedance [Ohm] or wake function/potential [V/C] samples.
            Transverse values also carry inverse-metre displacement normalization
            (the driving offset for dipolar wakes). Convert simulation exports
            such as V/pC to these units before fitting.
        N_resonators : int or None, optional
            Resonator count. Inferred from explicit bounds or a loaded table when
            omitted. Peak detection can also set it; iterative fitting starts at 1
            when no count is available.
        parameterBounds : list of tuple or None, optional
            One (lower, upper) pair per parameter in repeated (Rs, Q, fr) order.
            Missing bounds are derived when needed: Rs from zero to the peak real
            impedance, Q from 0.001 to 1000, and fr from the smallest positive
            frequency to ten times the largest. Wake inputs use FFT/deconvolution
            for bounds only; fitting remains in the input domain.
        plane : {"longitudinal", "transverse"}, optional
            Impedance plane. Default is "longitudinal".
        fitFunction : {"impedance", "wake", "wake potential"}, optional
            Input and fitting domain. Default is "impedance". Use a space in
            "wake potential"; "wake function" is also accepted for wakes.
        objectiveFunction : callable, str, or None, optional
            "real", "complex", or "abs"/"absolute" (case-insensitive), or a callable
            accepting (parameters, fitFunction, x, y). Default is complex SSE when
            any input sample has a nonzero imaginary part, otherwise real SSE.
            Magnitude fitting discards phase information.
        wake_length : float or None, optional
            Wake truncation length [m] for finite-duration impedance and SmartBounds
            analysis. None uses fully decayed impedance. Not required to evaluate
            a time-domain wake or potential.
        sigma : float or None, optional
            RMS Gaussian bunch duration [s], centered at time zero. Used for
            wake-potential evaluation and deconvolution; defaults to 1e-10 s for
            potential fits/getters. For impedance baseline removal, supply it only
            when the spectrum was deconvolved from a Gaussian wake potential.
        uncertainty_warning : float, optional
            Relative standard-deviation threshold for warnings. Default is 0.2.
        parameterEstimates : array_like or None, optional
            Starting values in flattened (Rs, Q, fr) order. Defaults to None.
            Solver x0 overrides these values. Without estimates, CMA-ES and direct
            local fitting use bound midpoints; DE initializes a population.
        compute_uncertainties : bool, optional
            Estimate parameter uncertainties after fitting. Default is False;
            they can also be requested later with get_uncertainties().

        Notes
        -----
        Evolutionary results are stored in ``evolutionParameters``; local and
        selected iterative results in ``minimizationParameters``. Getters prefer
        local results and fall back to evolutionary results. Loading a parameter
        table permits evaluation without fitting or generating bounds.

        Use NumPy arrays for numerical input. Original inputs remain in
        ``x_data`` and ``y_data``. Domain-specific arrays
        such as ``impedance_data`` hold the data currently used for fitting, including
        any baseline correction or downsampling. Resolve narrow resonances when
        reducing the sample count.

        Examples
        --------
        Construct a model, estimate bounds, fit, and evaluate on a frequency grid:

        >>> import numpy as np
        >>> import iddefix
        >>> frequency = np.linspace(0.5e9, 1.5e9, 401)
        >>> Z = iddefix.Impedances.Resonator_longitudinal_imp(frequency, 100, 8, 1e9)
        >>> model = iddefix.EvolutionaryAlgorithm(frequency, Z)
        >>> bounds = model.estimate_bounds(show_table=False, show_inspect=False)
        >>> model.run_cmaes(maxiter=100, popsize=10)
        >>> Z_fit = model.get_impedance(frequency)
        """

        self.x_data = x_data
        self.y_data = y_data

        if len(x_data) > 2000:
            print(
                f"\033[93m[!] Fitting {len(x_data)} samples may be slow. "
                "Consider downsampling "
                "with downsample_data(factor, show=True).\033[0m"
            )

        if N_resonators is None and parameterBounds is not None:
            if len(parameterBounds) == 0 or len(parameterBounds) % 3:
                raise ValueError(
                    "parameterBounds must contain three pairs per resonator"
                )
            N_resonators = len(parameterBounds) // 3
        self.N_resonators = N_resonators
        self.parameterBounds = parameterBounds
        self.parameterEstimates = parameterEstimates
        self.compute_uncertainties = compute_uncertainties
        self.objectiveFunction = objectiveFunction
        self.wake_length = wake_length
        self.plane = plane
        self.sigma = sigma
        self.fitFunctionType = "impedance"

        self.time_data = None
        self.wake_data = None
        self.wake_potential_data = None
        self.frequency_data = None
        self.impedance_data = None
        self.baseline_slope = None
        self._smart_bounds: SmartBoundDetermination | None = None
        self.evolutionParameters = None
        self.evolutionParametersUncertainties = None
        self.minimizationParameters = None
        self.minimizationParametersUncertainties = None
        self.minimizationBounds = None
        self.minimization_result = None
        self.best_trials = {}
        self.trial_history = {}
        self.flagged_params = None

        # Threshold on relative uncertainty (sigma/|value|) above which a
        # warning is printed after evolution/minimization
        self.uncertainty_warning = uncertainty_warning

        if self.objectiveFunction is None:
            if np.iscomplex(y_data).any():
                self.objectiveFunction = obj.sumOfSquaredError
                print(
                    "\033[93m[!] Objective function set to default \
                    `iddefix.objectiveFunctions.sumOfSquaredError`\033[0m"
                )
            else:
                self.objectiveFunction = obj.sumOfSquaredErrorReal
                print(
                    "\033[93m[!] Objective function set to \
                    `iddefix.objectiveFunctions.sumOfSquaredErrorReal` \
                    for real-valued only data\033[0m"
                )
        elif isinstance(self.objectiveFunction, str):
            objective_name = self.objectiveFunction.lower()
            if objective_name == "complex":
                self.objectiveFunction = obj.sumOfSquaredError
            elif objective_name == "real":
                self.objectiveFunction = obj.sumOfSquaredErrorReal
            elif objective_name in ("abs", "absolute"):
                self.objectiveFunction = obj.sumOfSquaredErrorAbs
            else:
                print(
                    "\033[93m[!] Objective function set to default \
                    `iddefix.objectiveFunctions.sumOfSquaredError`\033[0m"
                )
                self.objectiveFunction = obj.sumOfSquaredError

        if fitFunction == "wake" or fitFunction == "wake function":
            self.fitFunctionType = "wake"
            if plane == "longitudinal":
                self.fitFunction = wak.n_Resonator_longitudinal_wake
            elif plane == "transverse":
                self.fitFunction = wak.n_Resonator_transverse_wake
            else:
                raise Exception("Algorithm needs N_resonartors >= 1")
            self.time_data = x_data
            self.wake_data = y_data

        elif fitFunction == "wake potential":
            self.fitFunctionType = "wake potential"
            if self.sigma is None:
                print(
                    "\033[93m[!] sigma not specified, using the default sigma=1e-10 s\033[0m"
                )
                self.sigma = 1e-10

            if plane == "longitudinal":
                self.fitFunction = partial(
                    wak.n_Resonator_longitudinal_wake_potential,
                    sigma=self.sigma,
                )
            elif plane == "transverse":
                self.fitFunction = partial(
                    wak.n_Resonator_transverse_wake_potential, sigma=self.sigma
                )
            else:
                raise Exception("Algorithm needs N_resonartors >= 1")
            self.time_data = x_data
            self.wake_potential_data = y_data

        else:  # Default to "impedance"
            if wake_length is not None:
                print(
                    "[*] Using the partially decayed resonator formalism for impedance"
                )
            else:
                print("[*] Using the fully decayed resonator formalism for impedance")

            if plane == "longitudinal":
                self.fitFunction = partial(
                    imp.n_Resonator_longitudinal_imp, wake_length=wake_length
                )
            elif plane == "transverse":
                self.fitFunction = partial(
                    imp.n_Resonator_transverse_imp, wake_length=wake_length
                )
            else:
                raise Exception("Algorithm needs N_resonartors >= 1")
            self.frequency_data = x_data
            self.impedance_data = y_data

    @property
    def _fit_data(self) -> tuple[ArrayLike, ArrayLike]:
        """Current samples used for fitting in the selected domain."""
        if self.fitFunctionType == "impedance":
            return self.frequency_data, self.impedance_data
        if self.fitFunctionType == "wake":
            return self.time_data, self.wake_data
        return self.time_data, self.wake_potential_data

    def downsample_data(
        self,
        factor: int | None = None,
        show: bool = False,
        scale: str = "linear",
        spacing: str | None = None,
        *,
        samples: int | None = None,
    ) -> None:
        """Downsample paired data, always retaining the first and last samples.

        Parameters
        ----------
        factor : int or None, optional
            Keep every factor-th sample; mutually exclusive with samples.
        show : bool, optional
            Show an input-versus-retained-data preview. Default is False.
        scale : str, optional
            Plot scale: "linear", "log" (x), or "loglog" (both axes).
        spacing : str or None, optional
            Selection spacing: None, "linear", or "log". No interpolation is used.
        samples : int or None, optional
            Approximate retained sample count; mutually exclusive with factor.

        Notes
        -----
        Supply exactly one of ``factor`` (keep every Nth sample) or ``samples``
        (approximate retained count). ``samples`` defaults to linear spacing;
        ``spacing="log"`` selects near logarithmically spaced targets.
        ``factor`` also accepts explicit linear or log spacing.

        Selection uses original samples without interpolation. Duplicate
        selections are removed; preserving endpoints may add points. Log
        spacing also retains a zero sample and requires nonnegative x with
        at least one positive value. FFT/deconvolution requires uniform times.

        With ``show=True``, compare original and retained data. Complex data
        use separate real and imaginary panels. ``scale`` controls plotting:
        ``"linear"`` (default), ``"log"`` (x), or ``"loglog"`` (both).
        Log axes omit nonpositive values without changing stored signs.
        Call before fitting; preserve enough points to resolve narrow modes.
        """
        if (factor is None) == (samples is None):
            raise ValueError("Supply exactly one of factor or samples")
        if factor is not None and (
            isinstance(factor, (bool, np.bool_))
            or not isinstance(factor, (int, np.integer))
            or factor < 1
        ):
            raise ValueError("factor must be a positive integer")

        if samples is not None and (
            isinstance(samples, (bool, np.bool_))
            or not isinstance(samples, (int, np.integer))
            or samples < 2
        ):
            raise ValueError("samples must be an integer of at least two")

        fit_x, fit_y = self._fit_data
        x_data = np.asarray(fit_x)
        y_data = np.asarray(fit_y)
        if (
            x_data.ndim != 1
            or y_data.ndim != 1
            or x_data.shape != y_data.shape
            or len(x_data) < 2
        ):
            raise ValueError(
                "x_data and y_data must be matching one-dimensional arrays with at least two samples"
            )
        if samples is None:
            indices = np.unique(
                np.r_[np.arange(0, len(x_data), factor), len(x_data) - 1]
            )
            count = len(indices)
        else:
            count = min(samples, len(x_data))
            spacing = "linear" if spacing is None else spacing
        if spacing is not None:
            if spacing not in ("linear", "log"):
                raise ValueError("spacing must be None, 'linear', or 'log'")
            order = np.argsort(x_data)
            zeros = np.array([], dtype=int)
            if spacing == "log":
                if np.any(x_data < 0) or not np.any(x_data > 0):
                    raise ValueError(
                        "log spacing requires nonnegative x and positive samples"
                    )
                zeros = order[x_data[order] == 0][:1]
                order = order[x_data[order] > 0]
                count -= len(zeros)
            coordinates = np.log(x_data[order]) if spacing == "log" else x_data[order]
            targets = (
                np.linspace(coordinates[0], coordinates[-1], count)
                if count > 1
                else coordinates[-1:]
            )
            right = np.clip(np.searchsorted(coordinates, targets), 0, len(order) - 1)
            left = np.maximum(right - 1, 0)
            nearest = np.where(
                np.abs(coordinates[left] - targets)
                <= np.abs(coordinates[right] - targets),
                left,
                right,
            )
            indices = np.unique(np.r_[0, len(x_data) - 1, zeros, order[nearest]])
            indices = indices[np.argsort(x_data[indices])]
            if len(indices) < 2:
                raise ValueError("spacing must retain at least two distinct samples")

        if show:
            import matplotlib.pyplot as plt

            xscale, yscale = {
                "linear": ("linear", "linear"),
                "log": ("log", "linear"),
                "loglog": ("log", "log"),
            }[scale]
            components = (
                (("Real", np.real), ("Imaginary", np.imag))
                if np.iscomplexobj(y_data)
                else (("", np.real),)
            )
            fig, axes = plt.subplots(
                len(components),
                1,
                figsize=(10, 5 * len(components)),
                sharex=True,
                squeeze=False,
            )
            for axis, (name, component) in zip(axes[:, 0], components):
                axis.plot(x_data, component(y_data), label="Input")
                axis.plot(
                    x_data[indices],
                    component(y_data[indices]),
                    "o",
                    label="Downsampled",
                )
                axis.set_ylabel(f"{name} y_data" if name else "y_data")
                axis.set(xscale=xscale, yscale=yscale)
                axis.legend()
            axes[-1, 0].set(
                xlabel="Frequency [Hz]"
                if self.fitFunctionType == "impedance"
                else "Time [s]",
            )
            fig.tight_layout()

        if self.fitFunctionType == "impedance":
            self.frequency_data = x_data[indices]
            self.impedance_data = y_data[indices]
        elif self.fitFunctionType == "wake":
            self.time_data = x_data[indices]
            self.wake_data = y_data[indices]
        else:
            self.time_data = x_data[indices]
            self.wake_potential_data = y_data[indices]
        self.evolutionParametersUncertainties = None
        self.minimizationParametersUncertainties = None
        self.flagged_params = None
        self._smart_bounds = None

        print(f"[*] Downsampled data to {len(indices)} points (from {len(x_data)}).")

        if show:
            plt.show()

    def remove_baseline(
        self,
        slope: float | None = None,
        fit_interval: tuple[float, float] | None = None,
    ) -> float:
        """Remove a linear wake baseline from the fitting data and return its slope.

        Parameters
        ----------
        slope : float or None, optional
            Baseline slope in wake units per second. None estimates it from the data.
        fit_interval : tuple[float, float] or None, optional
            Time interval (start, stop) [s] used to fit the wake/potential tail.

        Returns
        -------
        float
            Removed slope in wake units per second.

        Notes
        -----
        With ``slope=None``, fit a ramp to the complex impedance or to the last 20%
        of nonnegative-time wake/potential samples. ``fit_interval=(start, stop)``
        selects a different tail interval in seconds for time-domain input only.
        A supplied slope has wake units per second; the subtracted time-domain
        ramp is slope*maximum(t, 0).

        Impedance correction requires ``wake_length`` [m]. ``sigma`` [s] is used
        only for spectra deconvolved from Gaussian wake potentials. Magnitude-only
        impedance cannot be corrected this way.

        Original inputs are retained. Fitted parameters, starting estimates, and
        the linked SmartBounds estimator are cleared. Repeating the call returns
        the previous slope; supplying a different slope raises ValueError.
        """
        if self.baseline_slope is not None:
            if slope is not None and slope != self.baseline_slope:
                raise ValueError("Baseline has already been removed with another slope")
            return self.baseline_slope

        if (
            self.fitFunctionType == "impedance"
            and self.objectiveFunction is obj.sumOfSquaredErrorAbs
            and not np.iscomplexobj(self.y_data)
        ):
            raise ValueError(
                "Baseline correction requires complex impedance for an absolute "
                "impedance objective"
            )

        # Retrieve corrected y_data and slope
        corrected, fitted_slope = baseline.remove_baseline(
            *self._fit_data,
            self.fitFunctionType,
            slope=slope,
            fit_interval=fit_interval,
            wake_length=self.wake_length,
            plane=self.plane,
            sigma=self.sigma if self.fitFunctionType == "impedance" else None,
        )

        # Update the model inputs
        self.baseline_slope = fitted_slope
        if self.fitFunctionType == "impedance":
            self.impedance_data = corrected
        elif self.fitFunctionType == "wake":
            self.wake_data = corrected
        else:
            self.wake_potential_data = corrected

        # Remove stored bounds
        self.evolutionParameters = None
        self.minimizationParameters = None
        self.evolutionParametersUncertainties = None
        self.minimizationParametersUncertainties = None
        self.minimizationBounds = None
        self.parameterEstimates = None
        self.flagged_params = None
        self._smart_bounds = None
        return fitted_slope

    def estimate_bounds(
        self,
        impedance_type: str | None = None,
        prominence: float | None = 1e-4,
        show_table: bool = True,
        show_inspect: bool = True,
        remove_baseline: bool | float = False,
        use_Hann: bool = True,
        fmax: float | None = None,
        fft_samples: int = 10001,
        rolloff: float = 0.0,
        **kwargs: Any,
    ) -> SmartBoundDetermination:
        """Estimate resonator bounds from impedance or transformed wake data.

        Wake data are transformed with ``compute_fft`` and wake-potential data
        with ``compute_deconvolution`` for peak detection. The model continues
        to fit its original data domain. Detected bounds, estimates, and the
        resonator count are stored on this model.

        Parameters
        ----------
        impedance_type : {"absolute", "real"} or None, optional
            Use the magnitude or real part of the stored impedance. By default,
            use ``"absolute"`` for the built-in magnitude objective and
            ``"real"`` otherwise. An explicit value overrides this choice.
        prominence : float or None, optional
            Minimum peak prominence in the detection trace's units, passed
            directly to ``scipy.signal.find_peaks``. Default is 1e-4; pass
            None to disable this filter.
        show_table : bool, optional
            Display the estimated bounds with ``to_table()``. Default is True.
        show_inspect : bool, optional
            Plot detected peaks and estimated resonator components with
            ``inspect(show_components=True)``. Default is True.
        remove_baseline : bool or float, optional
            If True, fit and remove a linear baseline from the stored fitting
            data before estimating bounds. A numeric value supplies its slope.
            Impedance correction requires ``wake_length`` and uses ``sigma``
            when the spectrum was Gaussian-deconvolved. Default is False.
        use_Hann : bool, optional
            Use a Hann trace to find peaks when the model has a finite wake
            length. Default is True.
        fmax : float, optional
            Upper frequency limit for transforming wake or wake-potential data.
            Defaults to 3 GHz for wakes and ``1 / (3 * sigma)`` for potentials.
        fft_samples : int, optional
            FFT zero-padding resolution for wake-domain inputs. Default is 10001.
            This does not add resolution beyond the measured time window.
        rolloff : float, optional
            Deconvolution taper width for wake potentials. Default is zero.
        **kwargs
            Automatic peak-finding options passed to
            :class:`SmartBoundDetermination`:

            - ``minimum_peak_height``: Minimum peak height; defaults to None,
              meaning no height filter.
            - ``threshold`` and ``distance``: Criteria passed to
              ``scipy.signal.find_peaks``; both default to None. ``distance``
              counts samples on the analysis grid.
            - ``Rs_bounds`` and ``Q_bounds``: Lower and upper multipliers for
              the estimated Rs and Q; each defaults to ``[0.1, 10]``.
            - ``fres_bounds``: Lower and upper multiples of the estimated
              resonator half-width; defaults to ``[-1, 1]``.
            - ``samples``: Number of equally spaced analysis samples, selected
              by linear interpolation. None uses the transformed or stored grid.
              This changes peak analysis only, not the fitting data.
            - ``q_side``: ``"auto"``, ``"left"``, ``"right"``, or a list with
              one choice per peak; defaults to ``"auto"``.

            With a finite ``wake_length`` and ``use_Hann=True``, peak finding
            uses a Hann trace for real-part detection and for absolute
            detection when complex impedance is supplied. Parameter estimates
            use the unwindowed data.

            Interactive selection is not supported by this method.

        Returns
        -------
        SmartBoundDetermination
            Estimator containing the detected peaks and bounds, available for
            inspection with ``inspect()`` or ``to_table()``.

        Raises
        ------
        ValueError
            If no peaks are detected or interactive selection is requested.
        """
        if kwargs.get("interactive", False):
            raise ValueError("estimate_bounds does not support interactive selection")

        if impedance_type is None:
            impedance_type = (
                "absolute"
                if self.objectiveFunction is obj.sumOfSquaredErrorAbs
                else "real"
            )
        if impedance_type not in ("absolute", "real"):
            raise ValueError("impedance_type must be 'absolute' or 'real'")

        # Correct baseline before bounds
        if remove_baseline is not False and remove_baseline is not None:
            if (
                self.fitFunctionType == "impedance"
                and impedance_type == "absolute"
                and not np.iscomplexobj(self.y_data)
            ):
                raise ValueError(
                    "Baseline correction requires complex impedance for "
                    "impedance_type='absolute'"
                )
            self.remove_baseline(
                slope=None if remove_baseline is True else remove_baseline
            )

        # Populate impedance data
        original_impedance_data = None
        original_frequency_data = None
        if self.fitFunctionType == "impedance":
            frequency_data = self.frequency_data
            impedance_data = self.impedance_data
            baseline_sigma = self.sigma
            if self.baseline_slope is not None:
                original_frequency_data = self.x_data
                original_impedance_data = self.y_data
        elif self.fitFunctionType == "wake":
            frequency_data, impedance_data = compute_fft(
                self.time_data,
                np.asarray(self.wake_data) / c,
                fmax=3e9 if fmax is None else fmax,
                samples=fft_samples,
            )
            if self.baseline_slope is not None:
                original_frequency_data, original_impedance_data = compute_fft(
                    self.x_data,
                    np.asarray(self.y_data) / c,
                    fmax=3e9 if fmax is None else fmax,
                    samples=fft_samples,
                )
            baseline_sigma = None
        else:  # Wake potential
            frequency_data, impedance_data = compute_deconvolution(
                self.time_data,
                np.asarray(self.wake_potential_data) / c,
                sigma=self.sigma,
                fmax=fmax,
                samples=fft_samples,
                rolloff=rolloff,
            )
            if self.baseline_slope is not None:
                original_frequency_data, original_impedance_data = (
                    compute_deconvolution(
                        self.x_data,
                        np.asarray(self.y_data) / c,
                        sigma=self.sigma,
                        fmax=fmax,
                        samples=fft_samples,
                        rolloff=rolloff,
                    )
                )
            baseline_sigma = self.sigma
        if self.fitFunctionType != "impedance" and self.plane == "transverse":
            impedance_data = 1j * impedance_data
            if original_impedance_data is not None:
                original_impedance_data = 1j * original_impedance_data
        if original_impedance_data is not None:
            original_impedance_data = np.interp(
                frequency_data, original_frequency_data, original_impedance_data
            )

        # Compute bounds
        bounds = SmartBoundDetermination(
            frequency_data,
            impedance_data,
            impedance_type=impedance_type,
            prominence=prominence,
            plane=self.plane,
            wake_length=self.wake_length,
            sigma=baseline_sigma,
            baseline_slope=self.baseline_slope,
            original_impedance_data=original_impedance_data,
            use_Hann=use_Hann and self.wake_length is not None,
            **kwargs,
        )
        if not bounds.N_resonators:
            raise ValueError("No resonance peaks found in the impedance data")

        # Populate model and show bounds
        self.apply_bounds(bounds)
        if show_inspect:
            bounds.inspect(show_components=True)
        if show_table:
            bounds.to_table()
        return bounds

    @property
    def smart_bounds(self) -> SmartBoundDetermination | None:
        """Most recently applied SmartBounds estimator, if available."""
        return self._smart_bounds

    @property
    def bounds(self) -> SmartBoundDetermination | None:
        """Most recently applied SmartBounds estimator; alias of ``smart_bounds``."""
        return self._smart_bounds

    def apply_bounds(self, bounds: SmartBoundDetermination) -> None:
        """Use a completed SmartBounds estimate to set fitting bounds.

        Parameters
        ----------
        bounds : SmartBoundDetermination
            Completed SmartBounds estimator for the same impedance plane.

        Notes
        -----
        The estimator must describe the same impedance plane and contain at
        least one completed peak. Applying it sets the count, bound pairs,
        and starting estimates, and clears previous fits. It is then available
        through ``bounds`` and ``smart_bounds`` for peak editing.
        """
        if not isinstance(bounds, SmartBoundDetermination):
            raise TypeError("bounds must be a SmartBoundDetermination instance")
        if bounds.plane != self.plane:
            raise ValueError("bounds plane must match the model plane")
        if (
            not bounds.N_resonators
            or bounds.parameterBounds is None
            or bounds.parameterEstimates is None
        ):
            raise ValueError("bounds must contain at least one completed peak")

        self._smart_bounds = bounds
        self.N_resonators = bounds.N_resonators
        self.parameterBounds = bounds.parameterBounds
        self.parameterEstimates = bounds.parameterEstimates
        self.evolutionParameters = None
        self.evolutionParametersUncertainties = None
        self.minimizationParameters = None
        self.minimizationParametersUncertainties = None
        self.minimizationBounds = None
        self.minimization_result = None
        self.flagged_params = None

    def add_peak(
        self,
        frequency: float,
        q_side: str | None = None,
        Q: float | None = None,
    ) -> None:
        """Add a peak to the latest SmartBounds estimate and update fit bounds.

        Parameters
        ----------
        frequency : float
            Frequency [Hz] of a peak to add, mapped to the nearest analysis sample.
        q_side : str or None, optional
            Crossing side: "auto", "left", or "right"; None uses the stored choice.
        Q : float or None, optional
            Positive quality factor. None leaves an existing value unchanged.

        Notes
        -----
        Call ``estimate_bounds`` or ``apply_bounds`` first. A successful
        addition clears fitted parameters because the number of resonators
        has changed.
        """
        bounds = self._smart_bounds
        if bounds is None:
            raise RuntimeError(
                "Call estimate_bounds or apply_bounds before adding a peak"
            )
        if (
            self.parameterBounds is not bounds.parameterBounds
            or self.parameterEstimates is not bounds.parameterEstimates
        ):
            raise RuntimeError("Call estimate_bounds again after editing the model")

        bounds.add_peak(frequency, q_side=q_side, Q=Q)
        self.apply_bounds(bounds)

    def remove_peak(self, peak_number: int | Sequence[int]) -> None:
        """Remove 1-based peaks from the latest estimate and reapply its bounds.

        Parameters
        ----------
        peak_number : int or sequence[int]
            One peak number or a sequence of 1-based table numbers to remove.

        Notes
        -----
        As with ``add_peak``, this clears earlier fitted parameters. To retain
        fitted values for the remaining modes, use ``remove_resonator``.
        """
        bounds = self._smart_bounds
        if bounds is None:
            raise RuntimeError(
                "Call estimate_bounds or apply_bounds before removing a peak"
            )
        if (
            self.parameterBounds is not bounds.parameterBounds
            or self.parameterEstimates is not bounds.parameterEstimates
        ):
            raise RuntimeError("Call estimate_bounds again after editing the model")
        bounds.remove_peak(peak_number)
        self.apply_bounds(bounds)

    def check_y_data(self) -> None:
        """Discard nonpositive x samples from the active fitting arrays.

        Notes
        -----
        Legacy helper; not called automatically. Original x_data/y_data are retained.
        """
        x, y = self._fit_data
        mask = np.asarray(x) > 0.0
        if self.fitFunctionType == "impedance":
            self.frequency_data = np.asarray(x)[mask]
            self.impedance_data = np.asarray(y)[mask]
        else:
            self.time_data = np.asarray(x)[mask]
            if self.fitFunctionType == "wake":
                self.wake_data = np.asarray(y)[mask]
            else:
                self.wake_potential_data = np.asarray(y)[mask]

    def _solve_differential_evolution(
        self,
        parameterBounds: ParameterBounds,
        objectiveFunction: ObjectiveCallable,
        fitFunction: Callable[..., np.ndarray],
        x_values_data: ArrayLike,
        y_values_data: ArrayLike,
        maxiter: int = 2000,
        popsize: int = 150,
        mutation: tuple[float, float] = (0.1, 0.5),
        crossover_rate: float = 0.8,
        tol: float = 0.01,
        solver: str = "scipy",
        **kwargs: Any,
    ) -> tuple[np.ndarray, str]:
        """Dispatch a bound-constrained objective to the selected DE backend."""

        objective_function = partial(
            objectiveFunction,
            fitFunction=fitFunction,
            x=x_values_data,
            y=y_values_data,
        )

        # Map solver names to functions
        solver_functions = {
            "scipy": Solvers.run_scipy_solver,
            "pyfde": Solvers.run_pyfde_solver,
            "pyfde_jade": Solvers.run_pyfde_jade_solver,
        }

        solver_function = solver_functions.get(solver)
        if not solver_function:
            raise ValueError(f"Invalid solver name: {solver}")

        solver_options = dict(maxiter=maxiter, popsize=popsize, tol=tol, **kwargs)
        if solver != "pyfde_jade":
            solver_options.update(mutation=mutation, crossover_rate=crossover_rate)
        solution, message = solver_function(
            parameterBounds, objective_function, **solver_options
        )

        return solution, message

    def _impedance_for_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the input spectrum or estimate it from time-domain data."""
        if self.fitFunctionType == "impedance":
            frequency = np.asarray(self.frequency_data)
            impedance = np.asarray(self.impedance_data)
        else:
            time = np.asarray(self.time_data)
            nyquist = 0.5 / (time[1] - time[0])
            if self.fitFunctionType == "wake":
                frequency, impedance = compute_fft(
                    time, np.asarray(self.wake_data) / c, fmax=nyquist
                )
            else:
                frequency, impedance = compute_deconvolution(
                    time,
                    np.asarray(self.wake_potential_data) / c,
                    sigma=self.sigma,
                    fmax=min(nyquist, 1 / (3 * self.sigma)),
                )
            if self.plane == "transverse":
                impedance = 1j * impedance
        return frequency, impedance

    def _default_parameterBounds(
        self, N_resonators: int | None = None
    ) -> ParameterBounds:
        """Estimate impedance if needed, then return bounds in Rs, Q, fr order."""
        count = self.N_resonators if N_resonators is None else N_resonators
        if count is None:
            raise ValueError("Set N_resonators before deriving default bounds")
        frequency, impedance = self._impedance_for_bounds()
        return [
            (0.0, np.max(np.real(impedance))),
            (0.001, 1000.0),
            (np.min(frequency[frequency > 0]), 10 * np.max(frequency)),
        ] * count

    def _require_fit_bounds(self) -> ParameterBounds:
        """Return validated finite bounds, deriving defaults when absent."""
        if self.N_resonators is None:
            raise ValueError("Set N_resonators before fitting")
        if self.parameterBounds is None:
            self.parameterBounds = self._default_parameterBounds()
        bounds = np.asarray(self.parameterBounds, dtype=float)
        if (
            bounds.shape != (3 * self.N_resonators, 2)
            or not np.all(np.isfinite(bounds))
            or np.any(bounds[:, 0] > bounds[:, 1])
        ):
            raise ValueError(
                "Fitting requires finite parameterBounds for each resonator"
            )
        return self.parameterBounds

    def run_differential_evolution(
        self,
        maxiter: int = 2000,
        popsize: int = 15,
        mutation: tuple[float, float] = (0.1, 0.5),
        crossover_rate: float = 0.8,
        tol: float = 0.01,
        solver: str = "scipy",
        x0: ArrayLike | None = None,
        **kwargs: Any,
    ) -> None:
        """Runs the differential evolution (DE) algorithm to estimate optimal
        resonance parameters.

        This function applies a global optimization technique using a DE solver
        to determine the best-fitting parameters for the given impedance data.
        The results can later be refined using a local minimization algorithm.

        Parameters
        ----------
        maxiter : int, optional
            Maximum number of iterations for the DE solver. Default is 2000.
        popsize : int, optional
            Population multiplier per free parameter. Default is 15.
        mutation : tuple of float, optional
            Range of mutation factors controlling parameter variation.
            Default is (0.1, 0.5).
        crossover_rate : float, optional
            Probability of recombining individuals in the DE algorithm.
            Default is 0.8.
        tol : float, optional
            Solver convergence tolerance; default 0.01. For SciPy DE, controls
            the spread of population objective values relative to their mean;
            it is not a target fitting error.
        solver : str, optional
            Specifies the DE solver to use. Valid options are:
            - `"scipy"`: Uses SciPy's built-in DE solver.
            - `"pyfde"`: Uses `pyfde`, an alternative DE implementation.
            - `"pyfde_jade"`: Uses JADE, a self-adaptive DE variant.
            Default is `"scipy"`.
        x0 : array_like, optional
            Initial parameter vector for SciPy differential evolution. Defaults
            to ``parameterEstimates`` stored on the model. PyFDE backends do not
            currently support an explicit initial vector. If both are ``None``,
            SciPy uses its configured population initializer.
        **kwargs : Any
            Additional options for the selected solver. With ``solver="scipy"``,
            options such as ``strategy``, ``init``, ``atol``, ``workers``, and
            ``seed`` are passed to
            ``scipy.optimize.differential_evolution``. IDDEFIX defaults to
            ``strategy="rand1bin"``, ``workers=-1``, and ``polish=False``;
            supplied keywords override these defaults. ``popsize`` is a
            multiplier of the number of free parameters for SciPy.

            With ``solver="pyfde"`` or ``"pyfde_jade"``, constructor options
            such as ``seed`` are passed to PyFDE. Batch objectives are not
            supported by the IDDEFIX objective functions. See the solver docs:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
            https://pyfde.readthedocs.io/en/latest/tutorial.html

        Notes
        -----
        - Uses `_solve_differential_evolution()` to dispatch the selected
          differential evolution backend.
        - The optimized parameters are stored in `self.evolutionParameters`.
        - Calls `self.display_resonator_parameters()` to present the estimated
          parameters.

        Returns
        -------
        None
            The optimized parameters are stored in `self.evolutionParameters`.
        """
        solver_options = dict(kwargs)
        if solver == "scipy":
            solver_options["x0"] = self.parameterEstimates if x0 is None else x0

        evolutionParameters, warning = self._solve_differential_evolution(
            self._require_fit_bounds(),
            self.objectiveFunction,
            self.fitFunction,
            *self._fit_data,
            maxiter=maxiter,
            popsize=popsize,
            mutation=mutation,
            crossover_rate=crossover_rate,
            tol=tol,
            solver=solver,
            **solver_options,
        )

        self.evolutionParameters = evolutionParameters
        self.evolutionParametersUncertainties = None
        self.flagged_params = None
        if self.compute_uncertainties:
            self.get_uncertainties(
                use_minimization=False, parameterBounds=self.parameterBounds
            )

        self.warning = warning
        self.display_resonator_parameters(
            params=self.evolutionParameters,
            to_markdown=False,
            uncertainties=self.evolutionParametersUncertainties,
            display_warnings=True,
        )

    def run_cmaes(
        self,
        maxiter: int | None = None,
        popsize: int = 50,
        sigma: float = 0.6,
        verbose: bool = False,
        x0: ArrayLike | None = None,
        tol: float = 1e-4,
        **kwargs: Any,
    ) -> Result:
        """Runs the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) algorithm
        from `pymoo` to optimize resonance parameters.

        This function applies the CMA-ES global optimization method to minimize the
        objective function based on the given impedance data and parameter bounds.
        The resulting optimized parameters are stored for further analysis or refinement.

        Parameters
        ----------
        maxiter : int or None, optional
            CMA-ES iteration limit per run. By default, there is no iteration
            limit; CMA-ES stops according to its convergence and other stopping
            criteria. With restarts, the total generation count can exceed a
            specified limit.
        popsize : int, optional
            Population size for the CMA-ES algorithm. Default is 50.
        sigma : float, optional
            Initial standard deviation for the sampling distribution. Default is 0.6.
        verbose : bool, optional
            Show pymoo's progress output instead of IDDEFIX's progress bar.
        x0 : array_like, optional
            Initial parameter vector. Defaults to ``parameterEstimates`` stored
            on the model. If both are ``None``, CMA-ES uses the bounds midpoint.
        tol : float, optional
            Solver convergence tolerance, mapped to CMA-ES ``tolx`` (parameter
            convergence); default 1e-4. Uses normalized coordinates with
            CMA-ES's default ``normalize=True``. Other stopping criteria remain active.
        **kwargs : Any
            Additional options passed to ``pymoo.algorithms.soo.nonconvex.cmaes.CMAES``.
            For example, ``tolx``, ``tolfun``, ``maxfevals``, ``restarts``,
            ``restart_from_best``, and ``seed``. IDDEFIX defaults to zero
            restarts and seed 42; a supplied keyword overrides these defaults.
            Use ``popsize`` above for the population size. Pymoo's
            internal stopping and restarts do not make ``maxiter`` a strict
            outer generation cap.

            Pymoo CMA-ES options are documented at
            https://pymoo.org/_modules/pymoo/algorithms/soo/nonconvex/cmaes.html.
            Other CMA-ES options passed through to its underlying strategy
            depend on the installed pymoo version.

        Returns
        -------
        res : pymoo.core.result.Result
            The optimization result object containing the solution and metadata.

        Notes
        -----
        - Uses `Solvers.run_pymoo_cmaes_solver()` to execute the optimization.
        - The optimized parameters are stored in `self.evolutionParameters`.
        - Calls `self.display_resonator_parameters()` to present the results.
        """

        objective_function = partial(
            self.objectiveFunction,
            fitFunction=self.fitFunction,
            x=self._fit_data[0],
            y=self._fit_data[1],
        )

        solution, message, res = Solvers.run_pymoo_cmaes_solver(
            self._require_fit_bounds(),
            objective_function,
            sigma=sigma,
            maxiter=maxiter,
            popsize=popsize,
            verbose=verbose,
            x0=self.parameterEstimates if x0 is None else x0,
            tol=tol,
            **kwargs,
        )

        self.evolutionParameters = solution
        self.evolutionParametersUncertainties = None
        self.flagged_params = None
        if self.compute_uncertainties:
            self.get_uncertainties(
                use_minimization=False, parameterBounds=self.parameterBounds
            )
        self.warning = message
        self.display_resonator_parameters(
            params=self.evolutionParameters,
            to_markdown=False,
            uncertainties=self.evolutionParametersUncertainties,
            display_warnings=True,
        )

        return res

    def run_minimization_algorithm(
        self,
        margin: float | Sequence[float] | None = [0.1, 0.1, 0.1],
        method: str = "Nelder-Mead",
        verbose: bool = True,
        options: dict[str, Any] | None = None,
        normalize_parameters: bool = False,
        maxiter: int | None = None,
        tol: float | None = None,
    ) -> None:
        """Fit or refine resonators using local minimization.

        Parameters
        ----------
        margin : float, sequence of float, or None, optional
            Relative refinement margins for (Rs, Q, fr), default 10% each.
            Applied around the latest minimization result, or the evolution
            result if no local fit exists. Resonant frequencies retain their
            configured positive lower bound. ``None`` uses full parameter bounds.
        method : str, optional
            SciPy minimization method; default ``"Nelder-Mead"``.
        verbose : bool, optional
            Print termination status and the parameter table; default True.
        options : dict or None, optional
            Advanced method-specific settings, overriding derived budgets
            and tolerances (e.g. xatol, fatol, ftol, gtol).
        normalize_parameters : bool, optional
            Optimize in coordinates scaled by finite bound spans; default
            False preserves the existing search and runtime. When True,
            default ``xatol=1e-6`` is in normalized coordinates; supplied
            ``initial_simplex`` coordinates must also be normalized.
        maxiter : int or None, optional
            Per-fit iteration cap. None uses 1000 per resonator for
            refinement or 5000 per resonator for full-bound fits. Nelder-Mead
            and L-BFGS-B also use this value as their evaluation budget.
        tol : float or None, optional
            Solver convergence tolerance; its meaning depends on the method.
            None uses Nelder-Mead xatol=1 (1e-6 with normalization) and fatol=1,
            or SciPy defaults for other methods.

        Notes
        -----
        - Initial fits use ``parameterEstimates`` or the bounds midpoint.
        - Repeated calls start from the latest fitted parameters.
        - Without a fitted result, full parameter bounds are used regardless
          of margin. Missing bounds are derived from the data.
        - Results and used bounds are stored in ``minimizationParameters``
          and ``minimizationBounds``. The SciPy ``OptimizeResult`` is retained
          in ``minimization_result`` (parameters and final simplex in physical
          units). Uncertainties are computed if requested.
        """
        if verbose:
            print("Method for minimization : " + method)
        objective_function = partial(
            self.objectiveFunction,
            fitFunction=self.fitFunction,
            x=self._fit_data[0],
            y=self._fit_data[1],
        )
        fitted_parameters = (
            self.minimizationParameters
            if self.minimizationParameters is not None
            else self.evolutionParameters
        )
        refine = fitted_parameters is not None and margin is not None
        if refine:
            margins = np.broadcast_to(margin, (3,))
            minimizationBounds = [
                sorted(((1 - margins[i % 3]) * p, (1 + margins[i % 3]) * p))
                for i, p in enumerate(fitted_parameters)
            ]
            for i in range(2, len(minimizationBounds), 3):
                lower = (
                    self.parameterBounds[i][0]
                    if self.parameterBounds is not None
                    else 0.0
                )
                minimizationBounds[i][0] = max(
                    minimizationBounds[i][0], lower, np.nextafter(0.0, 1.0)
                )
        else:
            if self.parameterBounds is None:
                self.parameterBounds = self._default_parameterBounds()
            minimizationBounds = self.parameterBounds
        initial_parameters = fitted_parameters
        if initial_parameters is None:
            initial_parameters = (
                np.mean(minimizationBounds, axis=1)
                if self.parameterEstimates is None
                else self.parameterEstimates
            )
        budget = (
            self.N_resonators * (1000 if refine else 5000)
            if maxiter is None
            else maxiter
        )
        solver_options = {"maxiter": budget, "disp": False}
        if method.lower() == "nelder-mead":
            solver_options.update(
                maxfev=budget,
                adaptive=True,
                xatol=(1e-6 if normalize_parameters else 1.0) if tol is None else tol,
                fatol=1.0 if tol is None else tol,
            )
        elif method.lower() == "l-bfgs-b":
            solver_options["maxfun"] = budget
        solver_options.update(options or {})
        solver_objective = objective_function
        solver_bounds = minimizationBounds
        x0 = np.asarray(initial_parameters)
        if normalize_parameters:
            lower, upper = np.asarray(minimizationBounds, dtype=float).T
            scale = np.where(upper > lower, upper - lower, 1.0)
            x0 = (x0 - lower) / scale
            solver_bounds = list(zip(np.zeros_like(lower), (upper - lower) / scale))

            def solver_objective(values):
                return objective_function(lower + scale * values)

        result = minimize(
            solver_objective,
            x0=x0,
            bounds=solver_bounds,
            method=method,
            tol=tol,
            options=solver_options,
        )
        if normalize_parameters:
            result.x = lower + scale * result.x
            if "final_simplex" in result:
                vertices, values = result.final_simplex
                result.final_simplex = (lower + scale * vertices, values)
        self.minimization_result = result
        self.minimizationParameters = result.x
        self.minimizationBounds = minimizationBounds
        self.minimizationParametersUncertainties = None
        self.flagged_params = None
        if self.compute_uncertainties:
            self.get_uncertainties(parameterBounds=self.minimizationBounds)
        if verbose:
            print(f"Minimization: {result.message}")
            print(
                f"Iterations: {result.nit} / {solver_options['maxiter']} (maxiter); "
                f"evaluations: {result.nfev} / "
                f"{solver_options.get('maxfev', solver_options.get('maxfun', 'solver default'))} "
                f"({'maxfev' if 'maxfev' in solver_options else 'maxfun' if 'maxfun' in solver_options else 'solver default'})"
            )
            self.display_resonator_parameters(
                params=self.minimizationParameters,
                to_markdown=False,
                uncertainties=self.minimizationParametersUncertainties,
                display_warnings=True,
            )

    def run_iterative_minimization(
        self,
        max_resonators: int = 20,
        maxiter: int | None = None,
        initial_q: float = 1.0,
        rms_target: float = 0.01,
        rms_floor: float = 1e-6,
        method: str = "Nelder-Mead",
        verbose: bool = False,
        options: dict[str, Any] | None = None,
    ) -> dict[int, EvolutionaryAlgorithm]:
        """Fit increasing numbers of resonators and keep the lowest-RMS model.

        Start with ``self.N_resonators`` (or 1 if unset). Fit all parameters,
        add a mode in the largest logarithmic frequency gap, and repeat until
        ``rms_target`` is reached or ``max_resonators`` is exhausted.
        Supplied estimates or existing fits seed the first trial; otherwise,
        frequencies are log-spaced and amplitudes come from the input spectrum.
        Added modes use data-derived default bounds. Each trial uses full bounds
        and normalized parameters.

        Parameters
        ----------
        max_resonators : int, optional
            Largest resonator count to try, inclusive. Default is 20.
        maxiter : int or None, optional
            Iteration limit per trial; defaults to 5000 per resonator.
            Function evaluations are unlimited unless capped in ``options``.
        initial_q : float, optional
            Starting Q for automatically seeded modes. Default is 1.
        rms_target : float, optional
            Relative RMS target; 0.01 means 1%. For complex fits, both real
            and imaginary components must meet it. Default is 0.01.
        rms_floor : float, optional
            Relative denominator floor near zero crossings. Default is 1e-6.
        method : str, optional
            SciPy local minimization method. Default is "Nelder-Mead".
        verbose : bool, optional
            Show solver details and parameter tables. Default is False;
            brief trial summaries are always printed.
        options : dict or None, optional
            Solver settings, including ``tol``. Method-specific settings
            override it. Nelder-Mead defaults to normalized ``xatol=1e-3``
            when ``tol`` is absent.

        Returns
        -------
        dict[int, EvolutionaryAlgorithm]
            Best fitted model per resonator count, also stored in ``best_trials``.
            ``trial_history`` retains every attempt. Repeated calls preserve
            these histories. The lowest-RMS model is restored on ``self``.
            A rerun refits the current count before checking its target. Keep
            the data and objective unchanged: saved scores are not recomputed.

        Notes
        -----
        The configured objective drives fitting; relative RMS selects trials
        and controls stopping. RMS is computed per component as
        ``sqrt(mean(((prediction - data) / scale)**2))``, where
        ``scale = maximum(abs(data), rms_floor * max(abs(data)))``.
        Complex fits use the worse component; magnitude fits use absolute
        values. Entirely zero reference components are excluded.

        Saved models provide ``rms_error``, ``rms_components``, and the SciPy
        ``minimization_result``. Uncertainties, if requested, are computed only
        for the selected model. This growth search does not guarantee the
        globally optimal resonator count.
        """

        count = 1 if self.N_resonators is None else self.N_resonators
        if (
            isinstance(count, (bool, np.bool_))
            or not isinstance(count, (int, np.integer))
            or count < 1
            or isinstance(max_resonators, (bool, np.bool_))
            or not isinstance(max_resonators, (int, np.integer))
            or max_resonators < count
        ):
            raise ValueError("Require 1 <= N_resonators <= max_resonators")
        if not np.isfinite(rms_target) or rms_target < 0:
            raise ValueError("rms_target must be finite and nonnegative")
        if not np.isfinite(rms_floor) or rms_floor <= 0:
            raise ValueError("rms_floor must be finite and positive")

        options = dict(options or {})
        tol = options.pop("tol", None)
        evaluation_limit = {"nelder-mead": "maxfev", "l-bfgs-b": "maxfun"}.get(
            method.lower()
        )
        if evaluation_limit is not None:
            options.setdefault(evaluation_limit, np.inf)
        if tol is None and method.lower() == "nelder-mead":
            options = {"xatol": 1e-3, **(options or {})}

        # Retrieve or calculate impedance from input data
        frequency, impedance = self._impedance_for_bounds()
        order = np.argsort(frequency)
        frequency, amplitude = frequency[order], np.abs(impedance[order])
        trial = copy(self)
        trial.N_resonators = count
        trial.compute_uncertainties = False
        trial.best_trials = {}
        trial.trial_history = {}
        trial._smart_bounds = None

        # Start with default parameter bounds if None
        if trial.parameterBounds is None:
            trial.parameterBounds = trial._default_parameterBounds()
        new_bounds = trial.parameterBounds[:3]
        f_lower, f_upper = new_bounds[2]

        # Distribute trial resonators across the frequency grid
        def seed(fr):
            return np.clip(
                [np.interp(fr, frequency, amplitude), initial_q, fr],
                np.asarray(new_bounds)[:, 0],
                np.asarray(new_bounds)[:, 1],
            )

        if trial.parameterEstimates is None:
            if trial.minimizationParameters is not None:
                trial.parameterEstimates = trial.minimizationParameters.copy()
            elif trial.evolutionParameters is not None:
                trial.parameterEstimates = trial.evolutionParameters.copy()
            else:
                frequencies = np.geomspace(f_lower, f_upper, trial.N_resonators + 2)[
                    1:-1
                ]
                estimates = np.array([seed(fr) for fr in frequencies])
                trial.parameterEstimates = estimates.ravel()
        trial.evolutionParameters = None
        trial.evolutionParametersUncertainties = None
        new_bounds = trial._default_parameterBounds(N_resonators=1)
        f_lower, f_upper = new_bounds[2]
        trials = self.best_trials
        history = self.trial_history
        best = (
            min(trials.values(), key=lambda model: model.rms_error) if trials else None
        )
        best_error = best.rms_error if best is not None else np.inf
        x, y = trial._fit_data

        if self.objectiveFunction in (
            obj.sumOfSquaredErrorAbs,
            obj.logsumOfSquaredErrorAbs,
        ):
            diagnostic = "abs"
        elif self.objectiveFunction in (
            obj.sumOfSquaredErrorReal,
            obj.logsumOfSquaredErrorReal,
        ):
            diagnostic = "real"
        elif self.objectiveFunction in (
            obj.sumOfSquaredError,
            obj.logsumOfSquaredError,
        ):
            diagnostic = "complex"
        else:
            diagnostic = "complex" if np.iscomplexobj(y) else "real"

        def relative_rms(predicted, data):
            peak = np.max(np.abs(data))
            if peak == 0:
                return np.nan
            scale = np.maximum(np.abs(data), rms_floor * peak)
            return float(np.sqrt(np.mean(((predicted - data) / scale) ** 2)))

        def rms_summary(errors):
            return " | ".join(
                f"{name} RMS: {error:.2%}" if np.isfinite(error) else f"{name} RMS: n/a"
                for name, error in errors.items()
            )

        while trial.N_resonators <= max_resonators:
            print(f"\nTrial: {trial.N_resonators} resonators")
            trial.run_minimization_algorithm(
                margin=None,
                method=method,
                verbose=verbose,
                options=options,
                normalize_parameters=True,
                maxiter=maxiter,
                tol=tol,
            )
            result = trial.minimization_result
            order = np.argsort(trial.minimizationParameters.reshape(-1, 3)[:, 2])
            indices = (3 * order[:, None] + np.arange(3)).ravel()
            trial.minimizationParameters = trial.minimizationParameters[indices]
            result.x = trial.minimizationParameters.copy()
            if "final_simplex" in result:
                vertices, values = result.final_simplex
                result.final_simplex = (vertices[:, indices], values)
            trial.parameterBounds = [trial.parameterBounds[i] for i in indices]
            trial.minimizationBounds = [trial.minimizationBounds[i] for i in indices]
            parameters = trial.minimizationParameters.copy()
            trial.parameterEstimates = parameters.copy()
            trial.objective_value = float(
                trial.objectiveFunction(
                    parameters, fitFunction=trial.fitFunction, x=x, y=y
                )
            )
            predicted = trial.fitFunction(x, parameters)
            if diagnostic == "abs":
                errors = {"Magnitude": relative_rms(np.abs(predicted), np.abs(y))}
            elif diagnostic == "real":
                errors = {"Real": relative_rms(predicted.real, np.real(y))}
            else:
                errors = {
                    "Real": relative_rms(predicted.real, np.real(y)),
                    "Imaginary": relative_rms(predicted.imag, np.imag(y)),
                }
            error = float(np.nanmax(list(errors.values())))
            trial.rms_error = error
            trial.rms_components = errors
            trial.real_rms_error = errors.get("Real", np.nan)
            trial.imag_rms_error = errors.get("Imaginary", np.nan)
            snapshot = deepcopy(trial)
            history[max(history, default=0) + 1] = snapshot
            previous = trials.get(trial.N_resonators)
            if previous is None or error < previous.rms_error:
                trials[trial.N_resonators] = snapshot
            print(rms_summary(errors))
            if best is None or error < best_error:
                best, best_error = snapshot, error
                print("New best fit.")
            else:
                print(f"Best remains: {best.N_resonators} resonators.")
            if error <= rms_target:
                print(f"Stopping: pointwise RMS at or below {rms_target:.2%}.")
                break
            if trial.N_resonators == max_resonators:
                print(
                    "Stopping: maximum resonator count reached; error target not met."
                )
                break

            existing = parameters.reshape(-1, 3)[:, 2]
            existing = existing[(existing > f_lower) & (existing < f_upper)]
            gaps = np.sort(np.r_[f_lower, existing, f_upper])
            gap = np.argmax(np.diff(np.log(gaps)))
            fr = np.sqrt(gaps[gap] * gaps[gap + 1])
            trial.add_resonator(*seed(fr), parameter_bounds=new_bounds)
            trial.minimizationParameters = None

        for name in (
            "N_resonators",
            "parameterBounds",
            "minimizationParameters",
            "minimizationBounds",
            "minimization_result",
            "rms_error",
            "real_rms_error",
            "imag_rms_error",
            "rms_components",
        ):
            setattr(self, name, deepcopy(getattr(best, name)))
        self.parameterEstimates = self.minimizationParameters.copy()
        self.evolutionParameters = self.evolutionParametersUncertainties = None
        self.minimizationParametersUncertainties = None
        self.best_trials = trials
        self.trial_history = history
        self.objective_value = best.objective_value
        self._smart_bounds = None
        self._update_parameter_flags()
        if self.compute_uncertainties:
            self.get_uncertainties(parameterBounds=self.minimizationBounds)
        print(f"Selected: {self.N_resonators} resonators.")
        print(rms_summary(self.rms_components))
        return trials

    def get_uncertainties(
        self,
        use_minimization: bool = True,
        x_data: ArrayLike | None = None,
        y_data: ArrayLike | None = None,
        parameterBounds: ArrayLike | None = None,
    ) -> np.ndarray:
        """Recompute and store uncertainties for the current resonator parameters.

        Parameters
        ----------
        use_minimization : bool, optional
            Prefer minimization parameters when available. False selects evolution results.
        x_data : array_like or None, optional
            Input or evaluation grid: frequency [Hz] or time [s], matching the domain.
        y_data : array_like or None, optional
            Paired data in the model domain and normalization.
        parameterBounds : array_like or None, optional
            Derivative-bound pairs in (Rs, Q, fr) order. None uses physical bounds.

        Returns
        -------
        ndarray
            Parameter standard deviations in flattened (Rs, Q, fr) order.

        Notes
        -----
        By default, use minimization results when available, otherwise
        evolutionary results. Input data defaults to the data stored on the
        model. Derivatives use physical resonator bounds by default, so loaded
        parameters need not lie within the original fitting bounds. Pass
        ``parameterBounds`` to use different derivative bounds.
        """
        if use_minimization and self.minimizationParameters is not None:
            parameters = self.minimizationParameters
            uncertainty_attribute = "minimizationParametersUncertainties"
        else:
            parameters = self.evolutionParameters
            uncertainty_attribute = "evolutionParametersUncertainties"

        if parameters is None:
            raise ValueError("No resonator parameters are available")

        if x_data is None:
            x_data = self._fit_data[0]
        if y_data is None:
            y_data = self._fit_data[1]
        if parameterBounds is None:
            parameterBounds = [
                bound
                for _ in range(self.N_resonators)
                for bound in ((-np.inf, np.inf), (0.0, np.inf), (0.0, np.inf))
            ]

        uncertainties = calculate_uncertainties(
            np.asarray(parameters),
            self.fitFunction,
            np.asarray(x_data),
            np.asarray(y_data),
            parameterBounds,
        )
        setattr(self, uncertainty_attribute, uncertainties)
        self.flagged_params = self._compute_flagged_params_mask(
            parameters, uncertainties
        )
        return uncertainties

    def display_parameter_bounds(
        self,
        parameterBounds: ParameterBounds | None = None,
        to_markdown: bool = False,
    ) -> None:
        """Display supplied bounds or the model's parameter bounds.

        Parameters
        ----------
        parameterBounds : list of tuple or None, optional
            Bound pairs in repeated (Rs, Q, fr) order. None uses the documented default.
        to_markdown : bool, optional
            Print Markdown instead of the terminal/notebook table. Default is False.

        Notes
        -----
        Missing bounds use the data-derived defaults. Set ``to_markdown=True``
        to print a Markdown table instead of the terminal or notebook table.
        """
        if parameterBounds is None:
            parameterBounds = self._require_fit_bounds()
        display_parameter_bounds(parameterBounds, to_markdown=to_markdown)

    def display_resonator_parameters(
        self,
        params: ArrayLike | None = None,
        to_markdown: bool = False,
        uncertainties: ArrayLike | None = None,
        display_uncertainties: bool = True,
        display_warnings: bool = False,
    ) -> None:
        """Display resonator parameters and their uncertainties.

        Parameters
        ----------
        params : array_like or None, optional
            Flat (Rs, Q, fr) parameter vector. None selects the latest fitted stage.
        to_markdown : bool, optional
            Print Markdown instead of the terminal/notebook table. Default is False.
        uncertainties : array_like or None, optional
            Matching standard deviations. When params is omitted, None uses the
            selected stage's stored standard deviations.
        display_uncertainties : bool, optional
            Include standard deviations when available. Default is True.
        display_warnings : bool, optional
            Print and highlight bound/uncertainty warnings. Default is False.

        Notes
        -----
        Set ``display_warnings=True`` to print uncertainty and bound warnings
        and highlight affected values. Fitting methods enable warnings for
        their automatic parameter display.
        """

        # Select the latest fitted stage and its matching uncertainties.
        if params is None:
            if self.minimizationParameters is not None:
                params = self.minimizationParameters
                print("[*] Displaying parameters after minimization")
                if uncertainties is None:
                    uncertainties = self.minimizationParametersUncertainties
            elif self.evolutionParameters is not None:
                params = self.evolutionParameters
                if uncertainties is None:
                    uncertainties = self.evolutionParametersUncertainties
            else:
                print("\033[93m[!] No parameters to display.\033[0m")
                return

        # Find parameters that need uncertainty or bound warnings.
        flagged_mask = self._compute_flagged_params_mask(params, uncertainties)
        if (
            params is self.minimizationParameters
            and self.minimizationBounds is not None
        ):
            bounds = self.minimizationBounds
            bounds_name = "minimization"
        else:
            bounds = self.parameterBounds
            bounds_name = "parameter"
        lower_mask, upper_mask = self._compute_bound_proximity_masks(params, bounds)
        bound_mask = lower_mask | upper_mask
        yellow = "\033[93m"
        reset = "\033[0m"
        if display_warnings:
            self._warn_large_uncertainties(params, uncertainties)
        else:
            flagged_mask = None
            bound_mask = np.zeros_like(bound_mask)
        if display_warnings and np.any(bound_mask):
            print(
                f"{yellow}[!] Warning: Fit within 1% of the {bounds_name} bounds. "
                f"Consider widening the bounds before refitting.{reset}"
            )

        if not display_uncertainties:
            uncertainties = None
            flagged_mask = None

        # Format each parameter once before adding optional uncertainty text.
        params_reshaped = params.reshape(-1, 3)
        if to_markdown:
            frequency_format = ".6g"
        elif uncertainties is not None:
            frequency_format = ".2e"
        else:
            frequency_format = ".3e"
        value_formats = (".6g", ".6g", frequency_format)
        values = [
            tuple(format(p_row[j], value_formats[j]) for j in range(3))
            for p_row in params_reshaped
        ]

        if uncertainties is not None:
            uncert_reshaped = uncertainties.reshape(-1, 3)
            if to_markdown:
                rows = [
                    tuple(f"{value[j]} ± {u_row[j]:.1e}" for j in range(3))
                    for value, u_row in zip(values, uncert_reshaped)
                ]
            else:
                # Align the ± signs across resonators in the text table.
                value_widths = [
                    max((len(row[j]) for row in values), default=0) for j in range(3)
                ]
                rows = [
                    tuple(
                        f"{value[j]:>{value_widths[j]}} ± {u_row[j]:>9.1e}"
                        for j in range(3)
                    )
                    for value, u_row in zip(values, uncert_reshaped)
                ]
        else:
            rows = values

        # Bound and uncertainty highlighting applies to the text/HTML table.
        if to_markdown:
            display_resonator_table(rows, to_markdown=True)
        else:
            bound_reshaped = bound_mask.reshape(-1, 3)
            minimum_widths = (26, 24, 26) if uncertainties is not None else (22, 12, 18)

            flagged_reshaped = (
                flagged_mask.reshape(-1, 3)
                if flagged_mask is not None
                else np.zeros_like(bound_reshaped)
            )
            colors = [
                tuple(
                    "blue"
                    if bound_reshaped[i, j]
                    else "red"
                    if flagged_reshaped[i, j]
                    else None
                    for j in range(3)
                )
                for i in range(len(rows))
            ]
            display_resonator_table(rows, minimum_widths=minimum_widths, colors=colors)

    def load_resonator_parameters(
        self, table: str, use_minimization: bool = True
    ) -> None:
        """Load parameter values and uncertainties from a displayed table.

        Parameters
        ----------
        table : str
            ASCII or Markdown parameter table, with optional uncertainty columns.
        use_minimization : bool, optional
            Prefer minimization parameters when available. False selects evolution results.

        Notes
        -----
        Accepts the ASCII or Markdown output of
        ``display_resonator_parameters`` with or without uncertainties.
        Missing uncertainties default to zero. Header and separator lines are
        optional. ANSI highlighting, including its visible ``␛``
        representation, is ignored. The number of rows must match
        ``N_resonators`` when specified; otherwise it is inferred from the table.
        Printed values have limited precision, so the
        original fit arrays remain preferable when available.

        By default the values are stored as minimization results, which are
        selected by the impedance getters. Missing evolutionary parameters and
        uncertainties are also populated from the table. Set
        ``use_minimization=False`` to store evolutionary results instead.
        """
        clean_table = re.sub(r"(?:\x1b|␛)\[[0-9;]*m", "", table)
        number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
        parameter_cell = re.compile(rf"\s*({number})(?:\s*±\s*({number}))?\s*")
        values = []
        uncertainties = []

        for line in clean_table.splitlines():
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]
            if not cells or not cells[0].isdigit():
                continue
            if len(cells) != 4:
                raise ValueError(f"Invalid resonator row: {line}")

            row_number = int(cells[0])
            if row_number != len(values) // 3 + 1:
                raise ValueError("Resonator rows must be numbered consecutively")

            row_values = []
            row_uncertainties = []
            for cell in cells[1:]:
                match = parameter_cell.fullmatch(cell)
                if match is None:
                    raise ValueError(f"Invalid parameter cell: {cell}")
                row_values.append(float(match[1]))
                row_uncertainties.append(float(match[2]) if match[2] else 0.0)
            values.extend(row_values)
            uncertainties.extend(row_uncertainties)

        if not values:
            raise ValueError("No resonator rows found in table")
        if self.N_resonators is not None and len(values) != 3 * self.N_resonators:
            raise ValueError(
                f"Expected {self.N_resonators} resonator rows, found {len(values) // 3}"
            )
        if self.parameterBounds is not None and len(self.parameterBounds) != len(
            values
        ):
            raise ValueError(
                "parameterBounds must contain three pairs per loaded resonator"
            )
        if self.N_resonators is None:
            self.N_resonators = len(values) // 3

        parameters = np.asarray(values)
        parameter_uncertainties = np.asarray(uncertainties)
        self.minimizationBounds = None
        if use_minimization:
            self.minimizationParameters = parameters
            self.minimizationParametersUncertainties = parameter_uncertainties
            if self.evolutionParameters is None:
                self.evolutionParameters = parameters.copy()
                self.evolutionParametersUncertainties = parameter_uncertainties.copy()
        else:
            self.evolutionParameters = parameters
            self.evolutionParametersUncertainties = parameter_uncertainties
        self._update_parameter_flags()

    def modify_resonator(
        self,
        resonator_number: int,
        Rs: float | None = None,
        Q: float | None = None,
        fr: float | None = None,
    ) -> None:
        """Overwrite selected parameters of a resonator by its 1-based number.

        Parameters
        ----------
        resonator_number : int
            1-based table number of the resonator to modify.
        Rs : float or None, optional
            Shunt impedance [Ohm or Ohm/m]. None leaves it unchanged.
        Q : float or None, optional
            Positive quality factor. None leaves an existing value unchanged.
        fr : float or None, optional
            Resonant frequency [Hz]. None leaves it unchanged.

        Notes
        -----
        ``None`` leaves a parameter unchanged. Both available fit stages are
        updated; fitted uncertainties are cleared because they describe the
        parameters before this manual edit. Call ``get_uncertainties`` to
        estimate uncertainties for the modified model.
        """
        if (
            isinstance(resonator_number, (bool, np.bool_))
            or not isinstance(resonator_number, (int, np.integer))
            or not 1 <= resonator_number <= self.N_resonators
        ):
            raise ValueError("resonator_number is outside the model")
        if self.evolutionParameters is None and self.minimizationParameters is None:
            raise ValueError("Load or fit resonator parameters before modifying a mode")

        changes = {}
        for offset, (name, value) in enumerate((("Rs", Rs), ("Q", Q), ("fr", fr))):
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a finite real number") from exc
            if not np.isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
            changes[offset] = value

        if not changes:
            return
        if 1 in changes and changes[1] <= 0:
            raise ValueError("Q must be positive")
        if 2 in changes and changes[2] <= 0:
            raise ValueError("fr must be positive")

        first_parameter = 3 * (resonator_number - 1)
        for parameter_name, uncertainty_name in (
            ("evolutionParameters", "evolutionParametersUncertainties"),
            ("minimizationParameters", "minimizationParametersUncertainties"),
        ):
            parameters = getattr(self, parameter_name)
            if parameters is None:
                continue
            parameters = np.asarray(parameters, dtype=float).copy()
            for offset, value in changes.items():
                parameters[first_parameter + offset] = value
            setattr(self, parameter_name, parameters)
            setattr(self, uncertainty_name, None)
        if self.parameterEstimates is not None:
            estimates = np.asarray(self.parameterEstimates, dtype=float).copy()
            for offset, value in changes.items():
                estimates[first_parameter + offset] = value
            self.parameterEstimates = estimates
        self.minimizationBounds = None
        self._update_parameter_flags()

    def remove_resonator(self, resonator_number: int | Sequence[int]) -> None:
        """Remove one or several resonators by their 1-based table numbers.

        Parameters
        ----------
        resonator_number : int or sequence[int]
            One resonator number or a sequence of 1-based table numbers.

        Notes
        -----
        The corresponding values, uncertainties, and fit bounds are removed
        from both evolution and minimization results when present. At least
        one resonator must remain. Lists use the numbering before removal;
        duplicate numbers are removed only once. Linked SmartBounds peak
        metadata, bounds, and estimates are synchronized when present.
        """
        numbers = np.unique(np.atleast_1d(resonator_number))
        if numbers.size == 0:
            return
        if (
            not np.issubdtype(numbers.dtype, np.integer)
            or np.any(numbers < 1)
            or np.any(numbers > self.N_resonators)
        ):
            raise ValueError("resonator_number is outside the model")
        if len(numbers) == self.N_resonators:
            raise ValueError("The model must retain at least one resonator")

        if self.bounds is not None:
            self.bounds.remove_peak(numbers)
        indices = (3 * (numbers[:, None] - 1) + np.arange(3)).ravel()
        for name in (
            "evolutionParameters",
            "evolutionParametersUncertainties",
            "minimizationParameters",
            "minimizationParametersUncertainties",
        ):
            values = getattr(self, name)
            if values is not None:
                setattr(self, name, np.delete(values, indices))
        if self.parameterEstimates is not None:
            self.parameterEstimates = np.delete(self.parameterEstimates, indices)
        self.minimizationBounds = None
        self.minimization_result = None

        if self.parameterBounds is not None:
            removed = set(indices)
            self.parameterBounds = [
                bound
                for index, bound in enumerate(self.parameterBounds)
                if index not in removed
            ]
        self.N_resonators -= len(numbers)
        if self.bounds is not None:
            self.bounds.parameterBounds = self.parameterBounds
            self.bounds.parameterEstimates = self.parameterEstimates
            self.bounds.N_resonators = self.N_resonators
        self._update_parameter_flags()

    def add_resonator(
        self,
        Rs: float,
        Q: float,
        fres: float,
        uncertainty: float | Sequence[float] = 0,
        parameter_bounds: ParameterBounds | None = None,
    ) -> None:
        """Append a resonator to the stored parameters and uncertainties.

        Parameters
        ----------
        Rs : float
            Shunt impedance: Ohm longitudinally or Ohm/m transversely.
        Q : float
            Positive quality factor for the added mode.
        fres : float
            Positive resonant frequency [Hz].
        uncertainty : float or sequence[float], optional
            One standard deviation for all three parameters, or an (Rs, Q, fr) triple.
        parameter_bounds : list of tuple or None, optional
            Three bound pairs for the added (Rs, Q, fr) mode.

        Notes
        -----
        ``uncertainty`` may be one value for all three parameters or a
        sequence of three values in ``(Rs, Q, fres)`` order. The new mode is
        appended to both result stages when both are present. Without explicit
        ``parameter_bounds``, its fit bounds are open apart from Q > 0 and
        fres >= 0; supply finite bounds before running another fit. Adding a
        mode detaches any SmartBounds estimator because its peak list no longer
        describes the model.
        """
        if self.evolutionParameters is None and self.minimizationParameters is None:
            raise ValueError("Load or fit resonator parameters before adding a mode")

        new_parameters = np.array([Rs, Q, fres], dtype=float)
        new_uncertainties = np.broadcast_to(
            np.asarray(uncertainty, dtype=float), (3,)
        ).copy()
        if Q <= 0 or fres <= 0:
            raise ValueError("Q and fres must be positive")
        if np.any(new_uncertainties < 0):
            raise ValueError("uncertainties must be non-negative")
        if parameter_bounds is None:
            new_bounds = [(-np.inf, np.inf), (0.0, np.inf), (0.0, np.inf)]
        else:
            if len(parameter_bounds) != 3:
                raise ValueError("parameter_bounds must contain three pairs")
            new_bounds = list(parameter_bounds)

        for parameter_name, uncertainty_name in (
            ("evolutionParameters", "evolutionParametersUncertainties"),
            ("minimizationParameters", "minimizationParametersUncertainties"),
        ):
            parameters = getattr(self, parameter_name)
            if parameters is None:
                continue
            setattr(
                self,
                parameter_name,
                np.concatenate((parameters, new_parameters)),
            )
            uncertainties = getattr(self, uncertainty_name)
            if uncertainties is None:
                uncertainties = np.zeros_like(parameters, dtype=float)
            setattr(
                self,
                uncertainty_name,
                np.concatenate((uncertainties, new_uncertainties)),
            )
        if self.parameterEstimates is not None:
            self.parameterEstimates = np.concatenate(
                (self.parameterEstimates, new_parameters)
            )
        self.minimizationBounds = None
        self.minimization_result = None
        self._smart_bounds = None

        if self.parameterBounds is not None:
            self.parameterBounds = [*self.parameterBounds, *new_bounds]
        elif parameter_bounds is not None:
            open_bounds = [(-np.inf, np.inf), (0.0, np.inf), (0.0, np.inf)]
            self.parameterBounds = open_bounds * self.N_resonators + new_bounds
        self.N_resonators += 1
        self._update_parameter_flags()

    def _update_parameter_flags(self) -> None:
        """Refresh uncertainty flags for the parameter stage used by getters."""
        if self.minimizationParameters is not None:
            parameters = self.minimizationParameters
            uncertainties = self.minimizationParametersUncertainties
        else:
            parameters = self.evolutionParameters
            uncertainties = self.evolutionParametersUncertainties
        self.flagged_params = self._compute_flagged_params_mask(
            parameters, uncertainties
        )

    def _compute_flagged_params_mask(
        self,
        params: ArrayLike | None,
        uncertainties: ArrayLike | None,
    ) -> np.ndarray | None:
        """Mark parameters whose relative uncertainty exceeds the warning threshold."""

        if params is None or uncertainties is None:
            return None

        threshold = getattr(self, "uncertainty_warning", None)
        if threshold is None:
            return None

        params = np.asarray(params, dtype=float)
        uncertainties = np.asarray(uncertainties, dtype=float)
        if params.shape != uncertainties.shape:
            return None

        with np.errstate(divide="ignore", invalid="ignore"):
            rel_unc = np.abs(uncertainties / params)

        return np.isfinite(rel_unc) & (rel_unc >= threshold)

    def _compute_bound_proximity_masks(
        self, params: ArrayLike, bounds: ParameterBounds | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find lower and upper bound hits, using log position for positive Rs/Q."""
        params = np.asarray(params, dtype=float).ravel()
        if bounds is None:
            empty = np.zeros(params.shape, dtype=bool)
            return empty, empty.copy()
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (params.size, 2):
            empty = np.zeros(params.shape, dtype=bool)
            return empty, empty.copy()

        width = bounds[:, 1] - bounds[:, 0]
        valid = np.all(np.isfinite(bounds), axis=1) & (width > 0)
        position = np.zeros_like(params)
        position[valid] = (params[valid] - bounds[valid, 0]) / width[valid]

        multiplicative = (
            valid & (np.arange(params.size) % 3 < 2) & (bounds[:, 0] > 0) & (params > 0)
        )
        position[multiplicative] = (
            np.log(params[multiplicative]) - np.log(bounds[multiplicative, 0])
        ) / (np.log(bounds[multiplicative, 1]) - np.log(bounds[multiplicative, 0]))
        return valid & (position <= 0.01), valid & (position >= 0.99)

    def _warn_large_uncertainties(
        self,
        params: ArrayLike | None,
        uncertainties: ArrayLike | None,
    ) -> None:
        """Store the uncertainty mask and print a summary when values are flagged."""

        mask = self._compute_flagged_params_mask(params, uncertainties)
        self.flagged_params = mask

        if mask is None:
            return

        threshold = getattr(self, "uncertainty_warning", None)
        n_flagged = int(mask.sum())
        if n_flagged == 0:
            return

        params = np.asarray(params, dtype=float)
        uncertainties = np.asarray(uncertainties, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_unc = np.abs(uncertainties / params)
        max_rel = float(np.nanmax(rel_unc[mask]))
        print(
            f"\033[93m[!] Warning: {n_flagged} parameter(s) have relative uncertainty >= {threshold:.2f} (max {max_rel:.2f}).\033[0m"
        )

    def get_wake(
        self,
        time_data: ArrayLike | None = None,
        use_minimization: bool = True,
    ) -> np.ndarray:
        """Evaluate the fitted wake function on a time grid in seconds.

        Parameters
        ----------
        time_data : array_like or None, optional
            Time grid [s]. None uses the model's stored time grid.
        use_minimization : bool, optional
            Prefer minimization parameters when available. False selects evolution results.

        Returns
        -------
        ndarray
            Fitted wake [V/C longitudinally, V/C/m transversely] on the time grid.

        Notes
        -----
        Uses ``time_data`` stored on the model if no grid is supplied. An impedance
        fit requires an explicit time grid on the first call. Results have units
        V/C longitudinally or V/C/m transversely and vanish before time zero.

        By default, use minimization parameters when available, otherwise evolution
        parameters. Set ``use_minimization=False`` to select the evolutionary fit.
        Returns an array on the requested grid. If absent, the model's ``time_data``
        and ``wake_data`` are populated on the first evaluation.
        """
        # Check for time data
        if time_data is None:
            if self.time_data is None:
                raise AttributeError("Provide time data array")
            time_data = self.time_data
        else:
            if self.time_data is None:
                self.time_data = time_data

        # Which pars to use
        if use_minimization and self.minimizationParameters is not None:
            pars = self.minimizationParameters
        else:
            pars = self.evolutionParameters

        # Which plane and formula
        if self.plane == "longitudinal" and self.N_resonators > 1:
            wake_data = wak.n_Resonator_longitudinal_wake(time_data, pars)
        elif self.plane == "transverse" and self.N_resonators > 1:
            wake_data = wak.n_Resonator_transverse_wake(time_data, pars)
        elif self.plane == "longitudinal" and self.N_resonators == 1:
            wake_data = wak.Resonator_longitudinal_wake(time_data, *pars)
        elif self.plane == "transverse" and self.N_resonators == 1:
            wake_data = wak.Resonator_transverse_wake(time_data, *pars)

        if self.wake_data is None:
            self.wake_data = wake_data

        return wake_data

    def get_wake_potential(
        self,
        time_data: ArrayLike | None = None,
        sigma: float | None = None,
        use_minimization: bool = True,
    ) -> np.ndarray:
        """Evaluate the fitted Gaussian wake potential on a time grid [s].

        Parameters
        ----------
        time_data : array_like or None, optional
            Time grid [s]. None uses the model's stored time grid.
        sigma : float or None, optional
            RMS Gaussian bunch duration [s].
        use_minimization : bool, optional
            Prefer minimization parameters when available. False selects evolution results.

        Returns
        -------
        ndarray
            Fitted Gaussian wake potential [V/C or V/C/m] on the time grid.

        Notes
        -----
        ``sigma`` is the RMS bunch duration in seconds, not metres. If omitted,
        use the model's sigma or 1e-10 s when unset. Times are measured from the
        bunch center; unlike the point-charge wake, a potential can extend to
        negative times. The requested grid defaults to the model's ``time_data``.

        Use minimization parameters when available, otherwise evolution parameters.
        Set ``use_minimization=False`` to select the evolutionary fit. Returns an
        array in V/C longitudinally or V/C/m transversely. This getter does not
        replace ``wake_potential_data`` used for fitting.
        """
        # Check for time data
        if time_data is None:
            if self.time_data is None:
                raise AttributeError("Provide time data array")
            time_data = self.time_data
        else:
            if self.time_data is None:
                self.time_data = time_data

        # Check for sigma
        if sigma is None:
            if self.sigma is None:
                self.sigma = 1e-10
            sigma = self.sigma
            print(
                f"\033[93m[!] sigma not specified, using sigma = {sigma:.2e} s\033[0m"
            )

        # Which pars to use
        if use_minimization and self.minimizationParameters is not None:
            pars = self.minimizationParameters
        else:
            pars = self.evolutionParameters

        # Which plane and formula - TODO check normalization
        if self.plane == "longitudinal" and self.N_resonators > 1:
            wake_potential_data = wak.n_Resonator_longitudinal_wake_potential(
                time_data, pars, sigma=sigma
            )
        elif self.plane == "transverse" and self.N_resonators > 1:
            wake_potential_data = wak.n_Resonator_transverse_wake_potential(
                time_data, pars, sigma=sigma
            )
        elif self.plane == "longitudinal" and self.N_resonators == 1:
            wake_potential_data = wak.Resonator_longitudinal_wake_potential(
                time_data, *pars, sigma=sigma
            )
        elif self.plane == "transverse" and self.N_resonators == 1:
            wake_potential_data = wak.Resonator_transverse_wake_potential(
                time_data, *pars, sigma=sigma
            )

        return wake_potential_data

    def get_impedance_from_fitFunction(
        self,
        frequency_data: ArrayLike | None = None,
        use_minimization: bool = True,
    ) -> np.ndarray:
        """Evaluate the configured fit function using a frequency grid.

        Parameters
        ----------
        frequency_data : array_like or None, optional
            Evaluation frequencies [Hz]. None uses the stored frequency grid.
        use_minimization : bool, optional
            Prefer minimization parameters when available. False selects evolution results.

        Returns
        -------
        ndarray
            Values of the configured impedance fit function on the requested grid.

        Notes
        -----
        Legacy impedance-only helper; it does not transform wake-domain fits.
        For ordinary impedance evaluation, use ``get_impedance``. This helper
        retains the wake length bound into the model's impedance fit function.
        The grid defaults to ``frequency_data``; results prefer minimization
        parameters unless ``use_minimization=False``.
        """
        # Check for frequency data
        if frequency_data is None:
            if self.frequency_data is None:
                raise AttributeError("Provide frequency data array")
            frequency_data = self.frequency_data
        else:
            if self.frequency_data is None:
                self.frequency_data = frequency_data

        # Which pars to use
        if use_minimization and self.minimizationParameters is not None:
            pars = self.minimizationParameters
        else:
            pars = self.evolutionParameters

        impedance_data = self.fitFunction(frequency_data, pars)

        return impedance_data

    def get_impedance(
        self,
        frequency_data: ArrayLike | None = None,
        use_minimization: bool = True,
        wake_length: float | None = None,
    ) -> np.ndarray:
        """Evaluate fitted complex impedance on a frequency grid [Hz].

        Parameters
        ----------
        frequency_data : array_like or None, optional
            Evaluation frequencies [Hz]. None uses the stored frequency grid.
        use_minimization : bool, optional
            Prefer minimization parameters when available. False selects evolution results.
        wake_length : float or None, optional
            Wake truncation length [m]. None uses fully decayed impedance.

        Returns
        -------
        ndarray
            Complex impedance [Ohm or Ohm/m] on the frequency grid.

        Notes
        -----
        ``frequency_data`` defaults to the model's stored frequency grid. Supply
        one explicitly after a time-domain fit. Returns Ohm longitudinally or
        Ohm/m transversely.

        Use minimization parameters when available, otherwise evolution parameters.
        Set ``use_minimization=False`` to select the evolutionary fit.

        ``wake_length`` [m] selects finite-duration impedance. The default None
        always returns fully decayed impedance, even if the model has a configured
        wake length. Pass ``wake_length=model.wake_length`` to reproduce that
        finite-length convention.
        """
        # Check for frequency data
        if frequency_data is None:
            if self.frequency_data is None:
                raise AttributeError("Provide frequency data array")
            frequency_data = self.frequency_data
        else:
            if self.frequency_data is None:
                self.frequency_data = frequency_data

        # Which pars to use
        if use_minimization and self.minimizationParameters is not None:
            pars = self.minimizationParameters
        else:
            pars = self.evolutionParameters

        # Which plane and formula
        if self.plane == "longitudinal" and self.N_resonators > 1:
            impedance_data = imp.n_Resonator_longitudinal_imp(
                frequency_data, pars, wake_length
            )
        elif self.plane == "transverse" and self.N_resonators > 1:
            impedance_data = imp.n_Resonator_transverse_imp(
                frequency_data, pars, wake_length
            )
        elif self.plane == "longitudinal" and self.N_resonators == 1:
            impedance_data = imp.Resonator_longitudinal_imp(
                frequency_data, *pars, wake_length
            )
        elif self.plane == "transverse" and self.N_resonators == 1:
            impedance_data = imp.Resonator_transverse_imp(
                frequency_data, *pars, wake_length
            )

        return impedance_data

    def get_resonator_components(
        self,
        x_data: ArrayLike | None = None,
        use_minimization: bool = True,
    ) -> np.ndarray:
        """Return one fitted model contribution per resonator.

        Parameters
        ----------
        x_data : array_like or None, optional
            Input or evaluation grid: frequency [Hz] or time [s], matching the domain.
        use_minimization : bool, optional
            Prefer minimization parameters when available. False selects evolution results.

        Returns
        -------
        ndarray
            Component values with shape (N_resonators, number_of_samples).

        Notes
        -----
        Each contribution is evaluated independently with the configured
        impedance, wake, or wake-potential ``fitFunction``.
        """

        if x_data is None:
            x_data = self._fit_data[0]
        x_data = np.asarray(x_data)

        if use_minimization and self.minimizationParameters is not None:
            parameters = self.minimizationParameters
        else:
            parameters = self.evolutionParameters
        if parameters is None:
            raise ValueError("No resonator parameters are available")

        return np.asarray(
            [
                self.fitFunction(x_data, resonator_parameters)
                for resonator_parameters in np.asarray(parameters).reshape(-1, 3)
            ]
        )

    def plot(
        self,
        x_data: ArrayLike | None = None,
        y_data: ArrayLike | None = None,
        *,
        function: str | None = None,
        type: str | None = None,
        show_components: bool = False,
        use_minimization: bool = True,
        cmap: str = "rainbow",
        figsize: tuple[float, float] | None = None,
        dpi: float | None = None,
        input_data_kwargs: dict[str, Any] | None = None,
        component_kwargs: dict[str, Any] | None = None,
        total_model_kwargs: dict[str, Any] | None = None,
        sigma: float | None = None,
        wake_length: float | None = None,
        scale: str = "linear",
        show: bool = True,
        **subplot_kwargs: Any,
    ):
        """Plot reference data and the fitted model in the requested domain.

        Parameters
        ----------
        x_data : array_like or None, optional
            Evaluation grid: frequency [Hz] or time [s]. None uses the stored
            grid for the selected function. Values are evaluated directly;
            plotting does not interpolate or modify the fitting data.
        y_data : array_like or None, optional
            Reference data on x_data in the selected function's units. When
            omitted, include fitting data only if both function and grid match.
        function : {"impedance", "wake", "wake potential"} or None, optional
            Function to evaluate. None uses the model's fitting domain.
        type : {"real", "imag", "abs", "both"} or None, optional
            Mathematical part to display. None selects magnitude for impedance
            with an absolute objective, both impedance parts otherwise, and
            real values for wakes/potentials. This does not change the objective.
        show_components : bool, optional
            Include individual resonators. Default is False.
        use_minimization : bool, optional
            Prefer minimization parameters when available. False uses evolution
            results. Default is True.
        cmap : str, optional
            Component colormap. Default is "rainbow".
        figsize : tuple of float or None, optional
            Figure width and height in inches. None uses the panel defaults.
        dpi : float or None, optional
            Figure resolution; None uses the Matplotlib default.
        input_data_kwargs, component_kwargs, total_model_kwargs : dict or None, optional
            Line-style overrides for reference data, resonators, and total model.
        sigma : float or None, optional
            RMS Gaussian bunch duration [s] for wake potentials. None uses the
            model's sigma, or 1e-10 s if unset. The stored sigma is unchanged.
        wake_length : float or None, optional
            Impedance truncation length [m]. None uses the model's wake length;
            if both are None, evaluate fully decayed impedance.
        scale : {"linear", "log", "loglog"}, optional
            Linear axes, logarithmic x, or logarithmic x and y. Explicit
            subplot_kw axis settings take precedence. Default is "linear".
        show : bool, optional
            Call matplotlib.pyplot.show(). Set False to customize the returned
            figure before displaying it. Default is True.
        **subplot_kwargs : object
            Extra arguments for matplotlib.pyplot.subplots. The number of
            panels is determined by type.

        Returns
        -------
        figure : matplotlib.figure.Figure
            Plot containing the selected data and model curves.
        axes : ndarray of matplotlib.axes.Axes
            One-dimensional array of plotting axes, even for a single panel.

        Notes
        -----
        A different function uses the same fitted resonator parameters, not a
        transform of the reference data. Supply a suitable x_data grid if one
        is not stored. Impedance units are Ohm longitudinally and Ohm/m
        transversely; wake/potential units are V/C and V/C/m, respectively.
        Log axes hide nonpositive values without changing their signs.
        Ten or more displayed resonators use a colorbar instead of individual
        legend entries. Magnitudes of individual modes need not sum to the
        magnitude of the total complex model.

        Examples
        --------
        Customize a magnitude plot before displaying it:

        >>> fig, axes = model.plot(type="abs", show=False)
        >>> axes[0].set_title("Fitted impedance")

        Evaluate a wake potential from the same fitted parameters:

        >>> fig, axes = model.plot(time, function="wake potential", sigma=sigma)
        """
        import matplotlib.pyplot as plt

        # Choose which function to plot
        function = self.fitFunctionType if function is None else function
        if function == "wake function":
            function = "wake"
        if function not in ("impedance", "wake", "wake potential"):
            raise ValueError(
                "function must be 'impedance', 'wake', or 'wake potential'"
            )
        if type is None:
            if function == "impedance":
                absolute_objective = self.objectiveFunction in (
                    obj.sumOfSquaredErrorAbs,
                    obj.logsumOfSquaredErrorAbs,
                )
                type = "abs" if absolute_objective else "both"
            else:
                type = "real"
        if type not in ("real", "imag", "abs", "both"):
            raise ValueError("type must be 'real', 'imag', 'abs', or 'both'")
        scales = {
            "linear": ("linear", "linear"),
            "log": ("log", "linear"),
            "loglog": ("log", "log"),
        }
        if scale not in scales:
            raise ValueError("scale must be 'linear', 'log', or 'loglog'")

        # Choose the x_data values and matching input y_data
        if x_data is None:
            x_data = self.frequency_data if function == "impedance" else self.time_data
        if x_data is None:
            raise ValueError(f"Provide x_data for plotting {function}")
        x_data = np.asarray(x_data)
        if x_data.ndim != 1 or x_data.size == 0:
            raise ValueError("x_data must be a nonempty one-dimensional grid")
        if (
            y_data is None
            and function == self.fitFunctionType
            and np.array_equal(x_data, self._fit_data[0])
        ):
            y_data = self._fit_data[1]
        if y_data is not None:
            y_data = np.asarray(y_data)
            if y_data.shape != x_data.shape:
                raise ValueError("x_data and y_data must have the same shape")

        # Calculate the model without changing stored data
        parameters = (
            self.minimizationParameters
            if use_minimization and self.minimizationParameters is not None
            else self.evolutionParameters
        )
        if parameters is None:
            raise ValueError("No resonator parameters are available")
        if function == "impedance":
            evaluate = partial(
                getattr(imp, f"n_Resonator_{self.plane}_imp"),
                wake_length=self.wake_length if wake_length is None else wake_length,
            )
            units = r"[$\Omega$]" if self.plane == "longitudinal" else r"[$\Omega$/m]"
            name, xlabel = "Impedance", "Frequency [Hz]"
        else:
            evaluate = getattr(
                wak, f"n_Resonator_{self.plane}_{function.replace(' ', '_')}"
            )
            if function == "wake potential":
                if sigma is None:
                    sigma = self.sigma if self.sigma is not None else 1e-10
                if not np.isfinite(sigma) or sigma <= 0:
                    raise ValueError("sigma must be finite and positive")
                evaluate = partial(evaluate, sigma=sigma)
            units = "[V/C]" if self.plane == "longitudinal" else "[V/C/m]"
            name, xlabel = (
                ("Wake" if function == "wake" else "Wake potential"),
                "Time [s]",
            )
        total_model = evaluate(x_data, parameters)
        components = (
            [evaluate(x_data, row) for row in np.asarray(parameters).reshape(-1, 3)]
            if show_components
            else []
        )

        # Set up the axes and labels
        selected_parts = ("real", "imag") if type == "both" else (type,)
        transforms = {"real": np.real, "imag": np.imag, "abs": np.abs}
        descriptions = {
            "real": r"Real part $\Re(Z)$",
            "imag": r"Imaginary part $\Im(Z)$",
            "abs": r"Magnitude $|Z|$",
        }
        if function != "impedance":
            descriptions = {
                "real": name,
                "imag": "Imaginary " + name.lower(),
                "abs": "Magnitude " + name.lower(),
            }
        xscale, yscale = scales[scale]
        figure_options = {
            "figsize": (10, 7 if len(selected_parts) == 2 else 5)
            if figsize is None
            else figsize,
            "sharex": True,
            **subplot_kwargs,
        }
        figure_options["subplot_kw"] = {
            "xscale": xscale,
            "yscale": yscale,
            **figure_options.get("subplot_kw", {}),
        }
        if dpi is not None:
            figure_options["dpi"] = dpi
        fig, axes = plt.subplots(len(selected_parts), 1, **figure_options)
        axes = np.asarray(axes).reshape(-1)
        colors = plt.get_cmap(cmap, max(len(components), 1))

        # Draw the curves, with individual resonators behind the data and total
        for axis, part in zip(axes, selected_parts):
            values = transforms[part]
            if y_data is not None:
                input_style = {
                    "linestyle": "-",
                    "color": "blue" if part in ("abs", "imag") else "red",
                    "linewidth": 2.0,
                    "label": "Input data",
                    "alpha": 0.7,
                    "zorder": 3,
                }
                input_style.update(input_data_kwargs or {})
                axis.plot(x_data, values(y_data), **input_style)
            total_style = {
                "linestyle": "--",
                "color": "black",
                "linewidth": 1.8,
                "alpha": 0.9,
                "label": "Total model",
                "zorder": 4,
            }
            total_style.update(total_model_kwargs or {})
            axis.plot(x_data, values(total_model), **total_style)
            for index, contribution in enumerate(components):
                component_style = {
                    "linestyle": "-",
                    "color": colors(index),
                    "linewidth": 1.5,
                    "alpha": 0.7,
                    "label": f"Resonator {index + 1}",
                    "zorder": 2,
                }
                component_style.update(component_kwargs or {})
                axis.plot(x_data, values(contribution), **component_style)
            axis.set_ylabel(descriptions[part] + " " + units)
        axes[-1].set_xlabel(xlabel)

        # Use a colorbar when there are too many resonators for the legend
        if len(components) >= 10:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import BoundaryNorm

            for axis in axes:
                axis.legend(
                    handles=axis.lines[: 2 if y_data is not None else 1],
                    loc="upper right",
                    frameon=False,
                )
            fig.tight_layout()
            boundaries = np.arange(0.5, len(components) + 1.5)
            colorbar = fig.colorbar(
                ScalarMappable(
                    norm=BoundaryNorm(boundaries, len(components)), cmap=colors
                ),
                ax=axes,
                boundaries=boundaries,
                ticks=np.arange(1, len(components) + 1),
                fraction=0.035,
                pad=0.025,
                aspect=30,
            )
            colorbar.set_label("Resonator number")
            colorbar.ax.tick_params(length=0)
            colorbar.ax.invert_yaxis()
        else:
            for axis in axes:
                axis.legend(ncol=3, frameon=False)
            fig.tight_layout()
        if show:
            plt.show()
        return fig, axes

    def plot_resonator_components(
        self,
        x_data: ArrayLike | None = None,
        y_data: ArrayLike | None = None,
        use_minimization: bool = True,
        cmap: str = "rainbow",
        figsize: tuple[float, float] | None = None,
        dpi: float | None = None,
        input_data_kwargs: dict[str, Any] | None = None,
        component_kwargs: dict[str, Any] | None = None,
        total_model_kwargs: dict[str, Any] | None = None,
        scale: str = "linear",
        *,
        function: str | None = None,
        type: str | None = None,
        sigma: float | None = None,
        wake_length: float | None = None,
        show: bool = True,
        **subplot_kwargs: Any,
    ):
        """Plot the fitted model with individual resonators using :meth:`plot`.

        Parameters
        ----------
        x_data, y_data : array_like or None, optional
            Evaluation grid and reference data; defaults follow plot().
        use_minimization : bool, optional
            Prefer minimization results; False selects evolutionary parameters.
        cmap : str, optional
            Colormap for the resonators. Default is "rainbow".
        figsize : tuple of float or None, optional
            Figure size in inches; None uses the panel defaults.
        dpi : float or None, optional
            Resolution; None uses the Matplotlib default.
        input_data_kwargs, component_kwargs, total_model_kwargs : dict or None, optional
            Line styles for reference data, individual modes, and total model.
        scale : {"linear", "log", "loglog"}, optional
            Axis scales. Default is "linear".
        function : {"impedance", "wake", "wake potential"} or None, optional
            Function to evaluate; None uses the fitting domain.
        type : {"real", "imag", "abs", "both"} or None, optional
            Displayed part; the absolute objective defaults to magnitude for
            impedance, including when the reference data are complex.
        sigma : float or None, optional
            RMS Gaussian bunch duration [s] for wake potentials.
        wake_length : float or None, optional
            Impedance truncation length [m]; None uses the model's setting.
        show : bool, optional
            Display immediately. False allows customization before display.
        **subplot_kwargs : object
            Additional arguments for matplotlib.pyplot.subplots.

        Returns
        -------
        figure : matplotlib.figure.Figure
            Figure containing reference data, total model, and resonators.
        axes : ndarray of matplotlib.axes.Axes
            One-dimensional array of data axes, excluding any colorbar.

        See Also
        --------
        plot : Shared plotting implementation and evaluation semantics.
        """
        # Use plot() to draw the model and its resonators
        return self.plot(
            x_data,
            y_data,
            function=function,
            type=type,
            show_components=True,
            use_minimization=use_minimization,
            cmap=cmap,
            figsize=figsize,
            dpi=dpi,
            input_data_kwargs=input_data_kwargs,
            component_kwargs=component_kwargs,
            total_model_kwargs=total_model_kwargs,
            sigma=sigma,
            wake_length=wake_length,
            scale=scale,
            show=show,
            **subplot_kwargs,
        )

    def get_impedance_uncertainty(
        self,
        frequency_data: ArrayLike | None = None,
        use_minimization: bool = True,
        wake_length: float | None = None,
        n_sigma: float = 1.0,
        q_samples: int = 33,
        vary: str = "both",
        max_factor: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return pointwise sensitivity bounds on the real impedance.

        Parameters
        ----------
        frequency_data : array_like or None, optional
            Evaluation frequencies [Hz]. None uses the stored frequency grid.
        use_minimization : bool, optional
            Prefer minimization parameters when available. False selects evolution results.
        wake_length : float or None, optional
            Wake truncation length [m]. None uses fully decayed impedance.
        n_sigma : float, optional
            Number of fitted standard deviations used for parameter variation.
        q_samples : int, optional
            Number of Q samples per mode for finite-length sensitivity estimates.
        vary : str, optional
            Parameters to vary: "r"/"rs", "q", or "both" (case-insensitive).
        max_factor : float, optional
            Multiplicative cap on parameter variation; must be at least 1.

        Returns
        -------
        lower_bound, upper_bound : ndarray
            Pointwise sensitivity bounds on the real impedance, in Ohm or Ohm/m.

        Notes
        -----
        ``vary`` accepts "r"/"rs", "q", or "both" (case-insensitive).
        The selected parameters vary by up to
        ``n_sigma`` fitted standard deviations, limited to a multiplicative
        range between ``value / max_factor`` and ``value * max_factor``.
        Varied Q stays positive, and resonant frequencies stay fixed.
        Fitting search bounds are not used. These are sensitivity scenarios,
        not confidence intervals.

        Extrema are found per resonator and frequency, so the envelopes need
        not represent one parameter set across the frequency range. For
        partially decayed wakes, Q is sampled on ``q_samples`` points and
        the extrema are approximate. The nominal Q is always sampled.

        The nominal complex impedance remains available from ``get_impedance``.
        """
        if n_sigma < 0:
            raise ValueError("n_sigma must be non-negative")
        if max_factor < 1:
            raise ValueError("max_factor must be at least 1")
        if wake_length is not None and q_samples < 2:
            raise ValueError("q_samples must be at least 2")
        vary = vary.lower()
        if vary not in ("r", "rs", "q", "both"):
            raise ValueError("vary must be 'R', 'Q', or 'both'")

        if frequency_data is None:
            if self.frequency_data is None:
                raise AttributeError("Provide frequency data array")
            frequency_data = self.frequency_data

        if use_minimization and self.minimizationParameters is not None:
            parameters = self.minimizationParameters
            uncertainties = self.minimizationParametersUncertainties
        else:
            parameters = self.evolutionParameters
            uncertainties = self.evolutionParametersUncertainties

        if parameters is None or uncertainties is None:
            raise AttributeError("Fit parameters and uncertainties are required")

        parameters = np.asarray(parameters, dtype=float).reshape(-1, 3)
        uncertainties = np.asarray(uncertainties, dtype=float).reshape(-1, 3)
        frequency_data = np.asarray(frequency_data, dtype=float)
        lower_bound = np.zeros_like(frequency_data)
        upper_bound = np.zeros_like(frequency_data)

        resonator_impedance = (
            imp.Resonator_longitudinal_imp
            if self.plane == "longitudinal"
            else imp.Resonator_transverse_imp
        )

        def capped_limits(value: float, uncertainty: float) -> tuple[float, float]:
            factor_limits = sorted((value / max_factor, value * max_factor))
            return (
                max(value - n_sigma * uncertainty, factor_limits[0]),
                min(value + n_sigma * uncertainty, factor_limits[1]),
            )

        for (rs, q, resonant_frequency), (rs_sigma, q_sigma, _) in zip(
            parameters, uncertainties
        ):
            rs_values = capped_limits(rs, rs_sigma) if vary != "q" else (rs,)
            if vary in ("q", "both"):
                q_lower, q_upper = capped_limits(q, q_sigma)
                q_lower = max(q_lower, np.nextafter(0.0, 1.0))
                if wake_length is None:
                    q_values = (q_lower, q_upper)
                else:
                    q_values = np.unique(
                        np.append(np.linspace(q_lower, q_upper, q_samples), q)
                    )
            else:
                q_values = (q,)

            mode_lower = np.full_like(frequency_data, np.inf)
            mode_upper = np.full_like(frequency_data, -np.inf)
            for candidate_q in q_values:
                for candidate_rs in rs_values:
                    real_impedance = resonator_impedance(
                        frequency_data,
                        candidate_rs,
                        candidate_q,
                        resonant_frequency,
                        wake_length,
                    ).real
                    np.minimum(mode_lower, real_impedance, out=mode_lower)
                    np.maximum(mode_upper, real_impedance, out=mode_upper)

            lower_bound += mode_lower
            upper_bound += mode_upper

        return lower_bound, upper_bound

    def get_impedance_from_fft(
        self,
        time_data: ArrayLike | None = None,
        wake_data: ArrayLike | None = None,
        fmax: float = 3e9,
        samples: int = 1001,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform a fitted wake with the legacy FFT convention.

        Parameters
        ----------
        time_data : array_like or None, optional
            Time grid [s]. None uses the model's stored time grid.
        wake_data : array_like or None, optional
            Legacy argument; currently ignored in favor of the fitted wake.
        fmax : float, optional
            Upper transform frequency [Hz].
        samples : int, optional
            Zero-padding resolution. Default is 1001.

        Returns
        -------
        frequency, spectrum : ndarray
            Nonnegative frequencies [Hz] and the legacy FFT spectrum.

        Notes
        -----
        ``time_data`` [s] defaults to the stored grid; it must be uniformly spaced.
        ``fmax`` [Hz] and ``samples`` control the cutoff and zero-padding resolution.
        Returns (frequency, spectrum). The ``wake_data`` argument is currently
        ignored: this method always evaluates the fitted wake on the stored grid.

        This legacy path uses spatial FFT scaling and additional plane-dependent
        sign factors. For analytic impedance in the model's physical units, use
        ``get_impedance``. For an explicit transform, use ``compute_fft`` on
        ``model.get_wake(time) / c`` and multiply by 1j for the transverse convention.
        """
        # Check for time data
        if time_data is None:
            if self.time_data is None:
                raise AttributeError("Provide time data array")
            time_data = self.time_data
        else:
            if self.time_data is None:
                self.time_data = time_data

        wake_data = self.get_wake(self.time_data)

        f, Z = compute_fft(
            data_time=time_data,
            data_wake=wake_data,
            fmax=fmax,
            samples=samples,
        )

        # Apply convention
        if self.plane == "transverse":
            Z *= -1j
        elif self.plane == "longitudinal":
            Z *= -1.0

        return f, Z

    def get_extrapolated_wake(
        self,
        new_end_time: float | None = None,
        dt: float | None = None,
        time_data: ArrayLike | None = None,
        use_minimization: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the fitted wake on an extended time grid.

        Parameters
        ----------
        new_end_time : float or None, optional
            Required end time [s], excluded from the extended grid.
        dt : float or None, optional
            Extension time step [s]. None uses the smallest input time step.
        time_data : array_like or None, optional
            Time grid [s]. None uses the model's stored time grid.
        use_minimization : bool, optional
            Prefer minimization parameters when available. False selects evolution results.

        Returns
        -------
        time, wake : ndarray
            Extended time grid [s] and fitted wake [V/C or V/C/m].

        Notes
        -----
        ``new_end_time`` [s] is required and is excluded from the returned grid.
        ``time_data`` defaults to the stored grid. Keep its samples before the last
        point, then extend from that point with spacing ``dt`` [s]; when omitted,
        use the smallest input time step. Returns (extended_time, fitted_wake).

        Uses minimization parameters when available; ``use_minimization=False``
        selects the evolutionary fit. This evaluates the resonator model beyond
        the original interval rather than extrapolating the measured data.
        """
        # Check for time data
        if time_data is None:
            if self.time_data is None:
                raise AttributeError("Provide `time_data` array")
            time_data = self.time_data
        else:
            if self.time_data is None:
                self.time_data = time_data

        if new_end_time is None:
            raise Exception("Provide `new_end_time` to extrapolate")

        if dt is None:
            dt = np.min(time_data[1:] - time_data[:-1])

        ext_time_data = np.concatenate(
            (time_data[:-1], np.arange(time_data[-1], new_end_time, dt))
        )

        ext_wake_data = self.get_wake(ext_time_data, use_minimization)

        return ext_time_data, ext_wake_data

    def save_txt(
        self,
        f_name: str,
        x_data: ArrayLike | None = None,
        y_data: ArrayLike | None = None,
        x_name: str = "X [-]",
        y_name: str = "Y [-]",
    ) -> None:
        """Export supplied x and y arrays as text columns with a two-line header.

        Parameters
        ----------
        f_name : str
            Output filename; see Notes for the current suffix behavior.
        x_data : array_like or None, optional
            Input or evaluation grid: frequency [Hz] or time [s], matching the domain.
        y_data : array_like or None, optional
            Paired data in the model domain and normalization.
        x_name : str, optional
            Header label for the x column.
        y_name : str, optional
            Header label for the y column.

        Notes
        -----
        Both arrays are required; missing data prints a message and writes nothing.
        ``x_name`` and ``y_name`` label the columns. Values are written using
        ``numpy.savetxt``; this method does not save the fitted parameter table.

        The current implementation appends an extra .txt suffix: f_name="fit"
        and f_name="fit.txt" both write fit.txt.txt. Use numpy.savetxt directly
        when the exact output filename matters.
        """
        if not f_name.endswith(".txt"):
            f_name += ".txt"

        if x_data is not None and y_data is not None:
            np.savetxt(
                f_name + ".txt",
                np.c_[x_data, y_data],
                header="   " + x_name + " " * 20 + y_name + "\n" + "-" * 48,
            )
        else:
            print("txt not saved, please provide x_data and y_data")

    def read_txt(
        self,
        txt: str,
        skiprows: int = 2,
        delimiter: str | None = None,
        usecols: Sequence[int] | None = None,
        as_dict: bool = False,
    ) -> dict[Any, np.ndarray] | tuple[np.ndarray, np.ndarray]:
        """Read numerical columns without changing the model's fitting data.

        Parameters
        ----------
        txt : str
            Path to the numerical text file.
        skiprows : int, optional
            Number of leading rows to skip. Default is 2.
        delimiter : str or None, optional
            Field separator. None uses whitespace.
        usecols : sequence[int] or None, optional
            Zero-based column indices to load. None loads all columns.
        as_dict : bool, optional
            Return a dictionary of selected columns instead of the first two arrays.

        Returns
        -------
        tuple of ndarray or dict
            First two selected columns, or all selected columns when as_dict=True.

        Notes
        -----
        ``skiprows`` defaults to 2; ``delimiter=None`` splits on whitespace.
        ``usecols`` selects columns using zero-based indices. The reader first
        tries floats, then complex numbers if parsing fails.

        With ``as_dict=False``, return the first two selected columns as (x, y).
        With ``as_dict=True``, return all selected columns in a dictionary, keyed
        by bracket-delimited labels parsed from the first line or by column index
        if parsing fails. Provide a table with at least two data rows; tuple output
        also requires at least two selected columns.
        """

        try:
            load = np.loadtxt(
                txt, skiprows=skiprows, delimiter=delimiter, usecols=usecols
            )
        except Exception:
            load = np.loadtxt(
                txt,
                skiprows=skiprows,
                delimiter=delimiter,
                usecols=usecols,
                dtype=complex,
            )

        try:  # keys == header names
            with open(txt) as f:
                header = f.readline()

            header = header.replace(" ", "")
            header = header.replace("#", "")
            header = header.replace("\n", "")
            header = header.split("]")

            d = {}
            for i in range(len(load[0, :])):
                d[header[i] + "]"] = load[:, i]

        except Exception:  # keys == int 0, 1, ...
            d = {}
            for i in range(len(load[0, :])):
                d[i] = load[:, i]

        if as_dict:
            return d
        else:
            x_data = list(d.values())[0]
            y_data = list(d.values())[1]
            return x_data, y_data

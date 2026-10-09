#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:34:10 2020

@author: MaltheRaschke
"""

import asyncio

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks

from .resonatorFormulas import Impedances as imp

ArrayLike = npt.ArrayLike
ParameterBounds = list[tuple[float, float]]


class SmartBoundDetermination:
    def __init__(
        self,
        frequency_data: ArrayLike,
        impedance_data: ArrayLike,
        minimum_peak_height: float = 1.0,
        threshold: float | None = None,
        distance: float | None = None,
        prominence: float | None = None,
        Rs_bounds: list[float] = [0.1, 10],
        Q_bounds: list[float] = [0.5, 5],
        fres_bounds: list[float] = [-1.0, 1.0],
        samples: int | None = None,
        q_side: str | list[str] = "auto",
        impedance_type: str = "absolute",
        interactive: bool = False,
        plane: str = "longitudinal",
    ) -> None:
        """
        Automatically determines parameter bounds for resonance fitting
        by detecting impedance peaks in frequency-domain data.

        This class uses `scipy.signal.find_peaks` to identify resonances and
        estimates the bounds for resistance (Rs), quality factor (Q), and
        resonant frequency (fres) from a one-sided peak width.

        Parameters
        ----------
        frequency_data : numpy.ndarray
            Frequency data in Hz.
        impedance_data : numpy.ndarray
            Impedance magnitude or real impedance data in Ohms.
        minimum_peak_height : float, optional
            Minimum height for a peak to be considered a resonance. Default is 1.0.
        threshold : float, optional
            Required vertical distance between a peak and its neighboring values
            to be considered a peak. Passed to `scipy.signal.find_peaks`.
            Default is None.
        distance : float, optional
            Required minimum horizontal distance (in indices) between peaks.
            Passed to `scipy.signal.find_peaks`. When ``samples`` is set, this
            counts samples on the interpolated grid. Default is None.
        prominence : float, optional
            Required prominence of peaks. The prominence measures how much a peak
            stands out compared to its surrounding values. Passed to
            `scipy.signal.find_peaks`.
            Default is None.
        Rs_bounds : list, optional
            Scaling factors [min, max] for Rs bounds. Default is [0.1, 10].
        Q_bounds : list, optional
            Scaling factors [min, max] for Q bounds. Default is [0.5, 5].
        fres_bounds : list, optional
            Lower and upper factors multiplying the estimated resonator
            half-width ``fres / (2 * Q)``. Default is [-1, 1].
        samples : int, optional
            Number of equally spaced frequency samples used for peak finding and
            Q estimation. If None, use the input samples. Interpolation is linear
            and does not add information beyond the input data.
        q_side : {"auto", "left", "right"} or list of these, optional
            Side of each peak used for its half-power width and Q estimate.
            "auto" uses the nearest crossing. A list chooses a side separately
            for each detected peak, in increasing frequency order. If the
            selected side has no crossing, the Q estimate defaults to 1;
            the configured scaling factors can still produce a bound below 0.5.
        impedance_type : {"absolute", "real"}, optional
            Type of impedance supplied. Absolute impedance uses the 3 dB
            amplitude level (peak / sqrt(2)); real impedance uses the
            half-maximum level (peak / 2). Default is "absolute".
        interactive : bool, optional
            Select resonances by clicking on the impedance plot. With an
            ipympl notebook backend, zoom or pan first, then enable Pick and
            click each peak. Use Undo to remove the last pick and Done
            to finish. With a desktop backend, press Enter to
            finish; with an inline backend, enter frequencies in Hz at the
            prompt. Clicks are mapped to the nearest analysis sample. In a
            widget notebook, await ``wait_for_selection()`` before using bounds.
            Peak-finding settings are ignored when enabled. Default is False.
        plane : {"longitudinal", "transverse"}, optional
            Impedance plane used to convert each detected peak and crossing
            into estimates of Rs, Q, and fres. Transverse estimation currently
            supports real impedance only. Default is "longitudinal".
        Attributes
        ----------
        peaks : numpy.ndarray or None
            Indices of detected peaks on the analysis frequency grid.
        analysis_frequency_data, analysis_impedance_data : numpy.ndarray
            Frequency grid and impedance used for peak finding. These equal the
            inputs when ``samples`` is None.
        peaks_height : dict or None
            Peak properties containing the detected heights under
            ``"peak_heights"``.
        minus_3dB_points : numpy.ndarray or None
            Crossing levels for each detected peak. For real impedance these
            are the half-maximum levels (the name is kept for compatibility).
        upper_lower_bounds : numpy.ndarray or None
            One-sided width to the selected crossing for each peak.
        crossing_frequencies : numpy.ndarray or None
            Interpolated frequency of the selected crossing for each peak.
        q_sides_used : list[str]
            Crossing side used for each peak's Q estimate.
        N_resonators : int or None
            Number of detected resonators.
        parameterBounds : list of tuples
            Computed parameter bounds in the format:
            [(Rs_min, Rs_max), (Q_min, Q_max), (fres_min, fres_max), ...].
            None until an interactive selection is finished.
        parameterEstimates : list of float or None
            Initial resonator estimates in the same flattened order as the
            bounds: [Rs_1, Q_1, fres_1, Rs_2, Q_2, fres_2, ...]. None until an
            interactive selection is finished.

        Methods
        -------
        find(frequency_data=None, impedance_data=None, minimum_peak_height=None,
            threshold=None, distance=None, prominence=None, samples=None,
            q_side=None, impedance_type=None, interactive=None, plane=None)
            Detects impedance peaks and determines fitting parameter bounds
            automatically or from interactive selections.

        inspect(show_bounds=False, show_components=False)
            Plots the impedance data and highlights detected resonance peaks
            along with their selected crossing levels, bounds, and estimated
            resonator components.

        add_peak(frequency, q_side=None, Q=None)
            Adds a peak at the nearest analysis frequency and recomputes all
            estimates and bounds. A supplied Q estimate skips the crossing
            calculation for that peak.

        to_table(to_markdown=False)
            Displays resonance parameters in an ASCII or Markdown-formatted table.

        Notes
        -----
        - The crossing level depends on ``impedance_type``.
        - Peak detection is based on `scipy.signal.find_peaks`.
        - Computed parameter bounds are stored in `self.parameterBounds`.
        - The `inspect()` method visualizes peak detection results.
        - The `to_table()` method prints a structured table of parameter ranges.
        - Detection runs during construction. The public `find()` method remains
          available for legacy workflows that explicitly recompute the bounds.
        """

        # Input data and estimation settings
        self.frequency_data = frequency_data
        self.impedance_data = impedance_data
        self.minimum_peak_height = minimum_peak_height
        self.threshold = threshold
        self.distance = distance
        self.prominence = prominence
        self.Rs_bounds = Rs_bounds
        self.Q_bounds = Q_bounds
        self.fres_bounds = fres_bounds
        self.samples = samples
        self.q_side = q_side
        self.impedance_type = impedance_type
        self.interactive = interactive
        self.plane = plane

        # Results populated by peak detection and bound estimation
        self.peaks = None
        self.peaks_height = None
        self.minus_3dB_points = None
        self.upper_lower_bounds = None
        self.crossing_frequencies = None
        self.N_resonators = None
        self.analysis_frequency_data = None
        self.analysis_impedance_data = None
        self.q_sides_used = None
        # Interactive selection state
        self._selection_done = asyncio.Event()
        self._selection_figure = None
        self._selection_done_button = None
        self._selection_pick_button = None
        self._selection_undo_button = None
        self._selection_cancelled = False
        self._peak_q_estimates = {}

        # Run the initial peak search
        self.parameterEstimates = None
        self.parameterBounds = self._find()

    def find(
        self,
        frequency_data: ArrayLike | None = None,
        impedance_data: ArrayLike | None = None,
        minimum_peak_height: float | None = None,
        threshold: float | None = None,
        distance: float | None = None,
        prominence: float | None = None,
        samples: int | None = None,
        q_side: str | list[str] | None = None,
        impedance_type: str | None = None,
        interactive: bool | None = None,
        plane: str | None = None,
    ) -> ParameterBounds | None:
        """Recompute bounds; retained for compatibility with legacy workflows."""
        return self._find(
            frequency_data=frequency_data,
            impedance_data=impedance_data,
            minimum_peak_height=minimum_peak_height,
            threshold=threshold,
            distance=distance,
            prominence=prominence,
            samples=samples,
            q_side=q_side,
            impedance_type=impedance_type,
            interactive=interactive,
            plane=plane,
        )

    def _find(
        self,
        frequency_data: ArrayLike | None = None,
        impedance_data: ArrayLike | None = None,
        minimum_peak_height: float | None = None,
        threshold: float | None = None,
        distance: float | None = None,
        prominence: float | None = None,
        samples: int | None = None,
        q_side: str | list[str] | None = None,
        impedance_type: str | None = None,
        interactive: bool | None = None,
        plane: str | None = None,
    ) -> ParameterBounds | None:
        """Detect peaks and update the stored parameter estimates and bounds."""

        # Resolve call-specific overrides against the stored configuration
        frequency_data = (
            self.frequency_data if frequency_data is None else frequency_data
        )
        impedance_data = (
            self.impedance_data if impedance_data is None else impedance_data
        )
        minimum_peak_height = (
            self.minimum_peak_height
            if minimum_peak_height is None
            else minimum_peak_height
        )
        threshold = self.threshold if threshold is None else threshold
        distance = self.distance if distance is None else distance
        prominence = self.prominence if prominence is None else prominence
        samples = self.samples if samples is None else samples
        q_side = self.q_side if q_side is None else q_side
        impedance_type = (
            self.impedance_type if impedance_type is None else impedance_type
        )
        interactive = self.interactive if interactive is None else interactive
        plane = self.plane if plane is None else plane

        # Validate and store the impedance interpretation
        if plane not in ("longitudinal", "transverse"):
            raise ValueError("plane must be 'longitudinal' or 'transverse'")
        self.plane = plane
        self.impedance_type = impedance_type
        if plane == "transverse" and impedance_type == "absolute":
            print(
                "[!] Warning: Transverse SmartBounds estimation is derived "
                "for real impedance. Absolute impedance may give inconclusive "
                "bounds; use impedance_type='real' instead."
            )

        # Build the analysis grid used for both peaks and crossings
        frequency_data = np.asarray(frequency_data)
        impedance_data = np.asarray(impedance_data)
        if samples is not None:
            analysis_frequency_data = np.linspace(
                frequency_data[0], frequency_data[-1], samples
            )
            analysis_impedance_data = np.interp(
                analysis_frequency_data, frequency_data, impedance_data
            )
            if (
                isinstance(minimum_peak_height, np.ndarray)
                and minimum_peak_height.shape == frequency_data.shape
            ):
                minimum_peak_height = np.interp(
                    analysis_frequency_data,
                    frequency_data,
                    minimum_peak_height,
                )
        else:
            analysis_frequency_data = frequency_data
            analysis_impedance_data = impedance_data

        self.analysis_frequency_data = analysis_frequency_data
        self.analysis_impedance_data = analysis_impedance_data
        self._peak_q_estimates = {}

        # Select peaks interactively or with scipy.signal.find_peaks
        if interactive:
            self.peaks = None
            self.parameterEstimates = None
            self.parameterBounds = None
            self._selection_done.clear()
            self._selection_cancelled = False
            self._start_interactive_selection(q_side, impedance_type, plane)
            return self.parameterBounds

        peaks, _ = find_peaks(
            analysis_impedance_data,
            height=minimum_peak_height,
            threshold=threshold,
            distance=distance,
            prominence=prominence,
        )
        return self._bounds_from_peaks(peaks, q_side, impedance_type, plane)

    def _bounds_from_peaks(
        self,
        peaks: np.ndarray,
        q_side: str | list[str],
        impedance_type: str,
        plane: str,
    ) -> ParameterBounds:
        """Estimate parameter bounds from selected analysis-grid peaks."""
        # Set the crossing level appropriate to the supplied impedance data
        if impedance_type == "absolute":
            crossing_fraction = np.sqrt(1 / 2)
        elif impedance_type == "real":
            crossing_fraction = 0.5
        else:
            raise ValueError("impedance_type must be 'absolute' or 'real'")

        peak_heights = self.analysis_impedance_data[peaks]
        sides = [q_side] * len(peaks) if isinstance(q_side, str) else q_side
        positive_q_floor = np.finfo(float).eps

        # Collect metadata and flattened fit parameters in peak order
        crossing_levels = []
        crossing_widths = []
        crossing_frequencies = []
        sides_used = []
        parameter_estimates = []
        parameter_bounds = []

        # Estimate one resonator and its bounds from each selected peak
        for i, (peak, height) in enumerate(zip(peaks, peak_heights)):
            peak_frequency = self.analysis_frequency_data[peak]
            crossing_level = height * crossing_fraction
            supplied_q = self._peak_q_estimates.get(peak)

            if supplied_q is not None:
                # No crossing belongs to a peak whose Q was supplied explicitly.
                crossing_frequency = np.nan
                crossing_width = np.nan
                selected_side = "estimate"
                estimated_q = supplied_q
            else:
                crossing_frequency, crossing_width, selected_side = (
                    self._find_peak_crossing(peak, crossing_level, sides[i])
                )

                if crossing_width <= 0.0:
                    # Keep truncated peaks in the fit even when the requested
                    # crossing is outside the supplied frequency range.
                    estimated_q = 1.0
                elif plane == "longitudinal":
                    estimated_q = (
                        crossing_frequency
                        * peak_frequency
                        / abs(crossing_frequency**2 - peak_frequency**2)
                    )
                else:  # transverse estimate to Q based on width
                    estimated_q = peak_frequency / (2 * crossing_width)

            estimated_q = max(positive_q_floor, estimated_q)
            estimated_rs = height
            estimated_fres = peak_frequency

            # Only the transverse real part has a supported peak-shift correction.
            if (
                plane == "transverse"
                and impedance_type == "real"
                and peak_frequency > 0.0
            ):
                estimated_rs, estimated_q, estimated_fres = (
                    self._estimate_transverse_parameters(
                        peak_frequency,
                        height,
                        estimated_q,
                        crossing_frequency,
                        supplied_q,
                    )
                )

            rs_bounds = (
                estimated_rs * self.Rs_bounds[0],
                estimated_rs * self.Rs_bounds[1],
            )
            q_bounds = (
                max(positive_q_floor, estimated_q * self.Q_bounds[0]),
                estimated_q * self.Q_bounds[1],
            )
            # Frequency factors scale with the estimated resonator Q
            estimated_frequency_width = estimated_fres / (2 * estimated_q)
            freq_bounds = (
                max(
                    positive_q_floor,
                    estimated_fres + self.fres_bounds[0] * estimated_frequency_width,
                ),
                estimated_fres + self.fres_bounds[1] * estimated_frequency_width,
            )

            if estimated_rs < 0:
                rs_bounds = (rs_bounds[1], rs_bounds[0])

            crossing_levels.append(crossing_level)
            crossing_widths.append(crossing_width)
            crossing_frequencies.append(crossing_frequency)
            sides_used.append(selected_side)
            parameter_estimates.extend([estimated_rs, estimated_q, estimated_fres])
            parameter_bounds.extend([rs_bounds, q_bounds, freq_bounds])

        self.peaks = peaks
        self.peaks_height = {"peak_heights": peak_heights}
        self.minus_3dB_points = np.asarray(crossing_levels)
        self.upper_lower_bounds = np.asarray(crossing_widths)
        self.crossing_frequencies = np.asarray(crossing_frequencies)
        self.q_sides_used = sides_used
        self.N_resonators = len(peaks)
        self.parameterEstimates = parameter_estimates
        self.parameterBounds = parameter_bounds
        return parameter_bounds

    def _find_peak_crossing(
        self,
        peak: int,
        crossing_level: float,
        selected_side: str,
    ) -> tuple[float, float, str]:
        """Return the selected crossing frequency, width, and side for a peak."""
        frequency_data = self.analysis_frequency_data
        impedance_data = self.analysis_impedance_data
        crossing_indices = np.argwhere(
            np.diff(np.sign(impedance_data - crossing_level))
        ).flatten()

        if len(crossing_indices) == 0:
            return np.nan, 0.0, selected_side

        # Interpolate within each bracketing pair instead of using the nearest
        # sample, which can otherwise produce a zero width at the peak.
        left_frequency = frequency_data[crossing_indices]
        right_frequency = frequency_data[crossing_indices + 1]
        left_impedance = impedance_data[crossing_indices]
        right_impedance = impedance_data[crossing_indices + 1]
        crossing_frequencies = left_frequency + (
            (crossing_level - left_impedance)
            * (right_frequency - left_frequency)
            / (right_impedance - left_impedance)
        )

        peak_frequency = frequency_data[peak]
        left_crossings = crossing_frequencies[crossing_frequencies < peak_frequency]
        right_crossings = crossing_frequencies[crossing_frequencies > peak_frequency]
        left_width = (
            peak_frequency - left_crossings[-1] if len(left_crossings) else np.inf
        )
        right_width = (
            right_crossings[0] - peak_frequency if len(right_crossings) else np.inf
        )

        # Auto chooses the closest available crossing on either side.
        if selected_side == "auto":
            selected_side = "left" if left_width <= right_width else "right"
        if selected_side == "left":
            crossing_frequency = left_crossings[-1] if len(left_crossings) else np.nan
            crossing_width = left_width
        elif selected_side == "right":
            crossing_frequency = right_crossings[0] if len(right_crossings) else np.nan
            crossing_width = right_width
        else:
            raise ValueError("q_side must be 'auto', 'left', or 'right'")

        if not np.isfinite(crossing_width):
            return np.nan, 0.0, selected_side
        return crossing_frequency, crossing_width, selected_side

    @staticmethod
    def _estimate_transverse_parameters(
        peak_frequency: float,
        peak_height: float,
        estimated_q: float,
        crossing_frequency: float,
        supplied_q: float | None,
    ) -> tuple[float, float, float]:
        """Convert a transverse real-impedance peak to Rs, Q, and fres.

        The dimensionless peak location is recovered either from the selected
        crossing or directly from a supplied Q estimate.
        """
        # Invert the transverse peak condition to locate the peak from Q.
        if supplied_q is not None:
            q_squared = supplied_q**2
            eta_peak = (
                2 * q_squared - 1 + np.sqrt(16 * q_squared**2 - 4 * q_squared + 1)
            ) / (6 * q_squared)
        elif np.isfinite(crossing_frequency):
            crossing_ratio = crossing_frequency / peak_frequency
            numerator = 4 * crossing_ratio - crossing_ratio**2 - 1
            denominator = crossing_ratio**4 - 3 * crossing_ratio**2 + 4 * crossing_ratio
            eta_squared = numerator / denominator if denominator != 0.0 else np.nan
            eta_peak = (
                np.sqrt(eta_squared)
                if np.isfinite(eta_squared) and eta_squared > 0.0
                else np.nan
            )
        else:
            return peak_height, estimated_q, peak_frequency

        # Leave the peak estimate unchanged if the correction is not physical.
        if not np.isfinite(eta_peak) or not 0.0 < eta_peak < 1.0:
            return peak_height, estimated_q, peak_frequency

        if supplied_q is None:
            estimated_q = np.sqrt(eta_peak / ((1 - eta_peak) * (1 + 3 * eta_peak)))
        estimated_fres = peak_frequency / np.sqrt(eta_peak)
        estimated_rs = (
            peak_height * 2 * np.sqrt(eta_peak) * (1 + eta_peak) / (1 + 3 * eta_peak)
        )
        return estimated_rs, estimated_q, estimated_fres

    def add_peak(
        self,
        frequency: float,
        q_side: str | None = None,
        Q: float | None = None,
    ):
        """Add a peak at the nearest analysis frequency and recompute bounds.

        Existing peaks retain their selected crossing sides. The new peak uses
        the configured ``q_side`` when it is a single value, otherwise ``auto``;
        pass ``q_side`` to override that choice. If ``Q`` is supplied, it is
        used as the quality-factor estimate and the crossing calculation is
        skipped for the new peak.
        """
        # Peak additions require a completed automatic or interactive search
        if self.peaks is None or self.parameterBounds is None:
            raise RuntimeError("Finish interactive peak selection before adding a peak")

        # A supplied Q replaces the crossing-based estimate for this peak
        if Q is not None:
            try:
                Q = float(Q)
            except (TypeError, ValueError) as error:
                raise ValueError("Q must be a finite positive value") from error
            if not np.isfinite(Q) or Q <= 0.0:
                raise ValueError("Q must be a finite positive value")

        frequency = float(frequency)
        frequency_data = self.analysis_frequency_data
        if frequency < frequency_data[0] or frequency > frequency_data[-1]:
            raise ValueError("frequency is outside the analysis frequency range")

        peak = int(np.abs(frequency_data - frequency).argmin())
        if peak in self.peaks:
            raise ValueError("A peak already exists at this analysis frequency")

        selected_side = q_side
        if selected_side is None:
            selected_side = self.q_side if isinstance(self.q_side, str) else "auto"
        if selected_side not in ("auto", "left", "right"):
            raise ValueError("q_side must be 'auto', 'left', or 'right'")

        # Preserve earlier choices while inserting the new peak in frequency order
        previous_sides = dict(zip(self.peaks, self.q_sides_used))
        peaks = np.sort(np.append(self.peaks, peak)).astype(int)
        sides = [
            selected_side if index == peak else previous_sides[index] for index in peaks
        ]
        if Q is not None:
            self._peak_q_estimates[peak] = Q
        self._bounds_from_peaks(
            peaks,
            sides,
            self.impedance_type,
            self.plane,
        )

    def _start_interactive_selection(
        self,
        q_side: str | list[str],
        impedance_type: str,
        plane: str,
    ) -> None:
        """Pick peaks with controls suited to the active Matplotlib backend."""
        # Set up the selection plot for the active Matplotlib backend
        frequency_data = self.analysis_frequency_data
        impedance_data = self.analysis_impedance_data
        backend = matplotlib.get_backend().lower()
        is_widget = "ipympl" in backend or "nbagg" in backend
        is_inline = "inline" in backend
        fig, ax = plt.subplots()
        ax.plot(frequency_data, impedance_data)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Impedance [Ohm]")
        markers = ax.plot([], [], "rx", markersize=8)[0]
        selection_lines = []
        selected_indices: list[int] = []
        status_label = None
        pick_button = None

        # Shared selection callbacks
        def update_selection() -> None:
            peaks = np.array(sorted(selected_indices), dtype=int)
            markers.set_data(frequency_data[peaks], impedance_data[peaks])
            for line in selection_lines:
                line.remove()
            selection_lines.clear()
            for peak in peaks:
                selection_lines.append(
                    ax.axvline(frequency_data[peak], color="r", linestyle="--")
                )
            if status_label is not None:
                frequencies = ", ".join(
                    f"{frequency_data[peak] * 1e-9:.4f}" for peak in peaks
                )
                status_label.value = (
                    f"Selected: {frequencies} GHz" if len(peaks) else "Selected: none"
                )
            fig.canvas.draw_idle()

        def undo_selection() -> None:
            if selected_indices:
                selected_indices.pop()
                update_selection()

        def finish_selection() -> None:
            if not selected_indices:
                if status_label is not None:
                    status_label.value = "Select at least one peak before clicking Done"
                return
            peaks = np.array(sorted(selected_indices), dtype=int)
            self._bounds_from_peaks(peaks, q_side, impedance_type, plane)
            self._selection_done.set()
            if self._selection_done_button is not None:
                self._selection_done_button.disabled = True
            if pick_button is not None:
                pick_button.disabled = True
            if self._selection_undo_button is not None:
                self._selection_undo_button.disabled = True
            plt.close(fig)

        def on_click(event) -> None:
            if event.inaxes is not ax or event.xdata is None:
                return
            if is_widget:
                if pick_button is None or not pick_button.value:
                    return
            index = int(np.abs(frequency_data - event.xdata).argmin())
            if event.button == 1 or (is_widget and event.button == 3):
                if index not in selected_indices:
                    selected_indices.append(index)
            elif event.button == 3:
                undo_selection()
                return
            else:
                return
            update_selection()

        def on_key(event) -> None:
            if event.key in ("enter", "return"):
                finish_selection()

        def on_close(event) -> None:
            if self.parameterBounds is None:
                self._selection_cancelled = True
                self._selection_done.set()

        fig.canvas.mpl_connect("close_event", on_close)
        self._selection_figure = fig
        self._selection_done_button = None
        self._selection_pick_button = None
        self._selection_undo_button = None

        # Inline backends use a text prompt because the displayed plot is static
        if is_inline:
            ax.set_title("Enter peak frequencies in Hz at the prompt")
            plt.show()
            response = input("Peak frequencies in Hz (spaces or commas): ").strip()
            for value in response.replace(",", " ").split():
                frequency = float(value)
                index = int(np.abs(frequency_data - frequency).argmin())
                if index not in selected_indices:
                    selected_indices.append(index)
            finish_selection()
            return

        # Widget backends provide explicit pick, undo, and done controls
        fig.canvas.mpl_connect("button_press_event", on_click)
        if is_widget:
            try:
                import ipywidgets as widgets
                from IPython.display import display
            except ImportError as error:
                plt.close(fig)
                raise ImportError(
                    "Interactive ipympl selection requires ipywidgets"
                ) from error

            ax.set_title("Zoom/pan, then enable Pick and click peaks")
            status_label = widgets.Label(value="Selected: none")
            pick_button = widgets.ToggleButton(
                value=False, description="Pick", icon="mouse-pointer"
            )
            undo_button = widgets.Button(description="Undo", icon="undo")
            done_button = widgets.Button(
                description="Done", button_style="success", icon="check"
            )

            def on_pick_toggle(change) -> None:
                if not change["new"]:
                    return
                toolbar = getattr(fig.canvas, "toolbar", None)
                if toolbar is None:
                    return
                mode = str(toolbar.mode).lower()
                if "pan" in mode:
                    toolbar.pan()
                elif "zoom" in mode:
                    toolbar.zoom()

            pick_button.observe(on_pick_toggle, names="value")
            undo_button.on_click(lambda button: undo_selection())
            done_button.on_click(lambda button: finish_selection())
            self._selection_pick_button = pick_button
            self._selection_undo_button = undo_button
            self._selection_done_button = done_button
            plt.show()
            controls = widgets.HBox([pick_button, undo_button, done_button])
            display(widgets.VBox([status_label, controls]))
            return

        # Desktop backends use mouse and keyboard events directly
        ax.set_title("Left-click peaks; right-click to undo; press Enter")
        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show(block=False)

    async def wait_for_selection(self) -> ParameterBounds:
        """Wait for interactive picking to finish before using the bounds."""
        if self.parameterBounds is None:
            await self._selection_done.wait()
        if self._selection_cancelled:
            raise RuntimeError("Interactive selection was closed before completion")
        return self.parameterBounds

    def get_impedance_components(
        self,
        frequency_data: ArrayLike | None = None,
    ) -> np.ndarray:
        """Return one estimated impedance array per detected resonance."""
        if self.parameterEstimates is None:
            raise RuntimeError("Finish peak selection before evaluating the model")
        if frequency_data is None:
            frequency_data = self.analysis_frequency_data
        frequency_data = np.asarray(frequency_data)

        fit_function = (
            imp.n_Resonator_longitudinal_imp
            if self.plane == "longitudinal"
            else imp.n_Resonator_transverse_imp
        )
        return np.asarray(
            [
                fit_function(frequency_data, resonator_parameters)
                for resonator_parameters in np.asarray(self.parameterEstimates).reshape(
                    -1, 3
                )
            ]
        )

    def inspect(
        self,
        show_bounds: bool = False,
        show_components: bool = False,
    ) -> None:
        """Plot detected peaks, bounds, and estimated resonator components."""
        # Input trace and optional component data
        plt.figure()
        plt.plot(
            self.analysis_frequency_data,
            self.analysis_impedance_data,
            "-",
            color="tab:blue" if show_components else None,
            linewidth=2.0 if show_components else None,
            label="Input data" if show_components else None,
        )

        if self.peaks is not None:
            color_map = (
                plt.get_cmap("turbo", max(len(self.peaks), 1))
                if show_bounds or show_components
                else None
            )
            components = (
                self.get_impedance_components()
                if show_components and len(self.peaks) > 0
                else None
            )
            component_values = np.real if self.impedance_type == "real" else np.abs
            resonance_handles = []
            # Peak, crossing, and frequency-bound annotations
            for i, (peak, minus_3dB_point, upper_lower_bound) in enumerate(
                zip(self.peaks, self.minus_3dB_points, self.upper_lower_bounds)
            ):
                color = color_map(i) if show_bounds or show_components else None
                peak_handle = plt.plot(
                    self.analysis_frequency_data[peak],
                    self.analysis_impedance_data[peak],
                    "x",
                    color=color if show_bounds or show_components else "black",
                    label=f"#{i + 1}" if show_bounds and not show_components else None,
                )[0]
                if show_bounds and not show_components:
                    resonance_handles.append(peak_handle)
                if np.isfinite(self.crossing_frequencies[i]):
                    plt.vlines(
                        self.analysis_frequency_data[peak],
                        ymin=minus_3dB_point,
                        ymax=self.analysis_impedance_data[peak],
                        color=color if show_bounds or show_components else "r",
                        linestyle="--",
                    )
                    plt.hlines(
                        minus_3dB_point,
                        xmin=(
                            self.analysis_frequency_data[peak] - upper_lower_bound
                            if self.q_sides_used[i] == "left"
                            else self.analysis_frequency_data[peak]
                        ),
                        xmax=(
                            self.analysis_frequency_data[peak] + upper_lower_bound
                            if self.q_sides_used[i] == "right"
                            else self.analysis_frequency_data[peak]
                        ),
                        color=color if show_bounds or show_components else "g",
                        linestyle="--",
                    )
                if show_bounds and np.isfinite(self.crossing_frequencies[i]):
                    plt.plot(
                        self.crossing_frequencies[i],
                        minus_3dB_point,
                        marker="o",
                        markerfacecolor="none",
                        color=color,
                    )
                plt.text(
                    self.analysis_frequency_data[peak],
                    self.analysis_impedance_data[peak],
                    f"#{i + 1}",
                    fontsize=9,
                    color="black",
                )

                if (
                    show_bounds
                    and self.parameterEstimates is not None
                    and self.parameterBounds is not None
                ):
                    estimated_fres = self.parameterEstimates[3 * i + 2]
                    fres_bounds = self.parameterBounds[3 * i + 2]
                    plt.axvspan(
                        fres_bounds[0],
                        fres_bounds[1],
                        color=color,
                        alpha=0.15,
                        zorder=0,
                    )
                    plt.axvline(
                        estimated_fres,
                        color=color,
                        linestyle="-.",
                    )
                if components is not None:
                    component_handle = plt.plot(
                        self.analysis_frequency_data,
                        component_values(components[i]),
                        color=color,
                        linewidth=1.2,
                        linestyle=":",
                        alpha=0.7,
                        label=f"Resonator {i + 1}",
                    )[0]
                    resonance_handles.append(component_handle)
            # Sum the same components to show the complete estimated model
            if components is not None:
                plt.plot(
                    self.analysis_frequency_data,
                    component_values(components.sum(axis=0)),
                    color="black",
                    linewidth=1.2,
                    linestyle=":",
                    alpha=0.8,
                    label="Total model",
                )
        # Axes and legend
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Impedance [Ohm]")
        plt.title("Smart Bound Determination")
        if show_components and self.peaks is not None and len(self.peaks) > 0:
            plt.legend(ncol=3, frameon=False)
        elif show_bounds and self.peaks is not None and len(self.peaks) > 0:
            plt.legend(handles=resonance_handles, title="Resonator")
        plt.show()

        return None

    def to_table(
        self,
        parameterBounds: ParameterBounds | None = None,
        to_markdown: bool = False,
    ) -> None:
        """
        Displays resonance parameters in a formatted ASCII table.

        Args:
            params: A list of tuples containing resonator parameters in the order:
                    (Rs_min, Rs_max), (Q_min, Q_max), (fres_min, fres_max).
            to_markdown: If True, prints the table in Markdown format.

        Example Output:
        ------------------------------------------------------------
        Resonator |   Rs [Ohm/m or Ohm]    |        Q         |    fres [Hz]
        ------------------------------------------------------------
        1      |  31.12 to 311.12       |  88.20 to 180.47 |  4.16e+08 to 6.82e+08
        2      |  85.61 to 864.12       |  120.55 to 200.23|  5.30e+08 to 7.23e+08
        ------------------------------------------------------------
        """
        params = self.parameterBounds if parameterBounds is None else parameterBounds
        N_resonators = len(params) // 3  # Compute number of resonators

        # Define formatting
        header_format = "{:^10}|{:^24}|{:^18}|{:^25}"
        data_format = "{:^10d}|{:^24}|{:^18}|{:^25}"

        if to_markdown:
            # Markdown Table
            print("\n")
            print("| Resonator | Rs [Ohm/m or Ohm] | Q | fres [Hz] |")
            print("|-----------|------------------|---|-----------|")
            for i in range(N_resonators):
                rs_range = f"{params[i * 3][0]:.2f} to {params[i * 3][1]:.2f}"
                q_range = f"{params[i * 3 + 1][0]:.2f} to {params[i * 3 + 1][1]:.2f}"
                fres_range = f"{params[i * 3 + 2][0]:.2e} to {params[i * 3 + 2][1]:.2e}"
                print(f"| {i + 1} | {rs_range} | {q_range} | {fres_range} |")
        else:
            # ASCII Table
            print("\n" + "-" * 80)

            # Print header
            print(
                header_format.format("Resonator", "Rs [Ohm/m or Ohm]", "Q", "fres [Hz]")
            )
            print("-" * 80)

            # Print data
            for i in range(N_resonators):
                rs_range = f"{params[i * 3][0]:.2f} to {params[i * 3][1]:.2f}"
                q_range = f"{params[i * 3 + 1][0]:.2f} to {params[i * 3 + 1][1]:.2f}"
                fres_range = f"{params[i * 3 + 2][0]:.2e} to {params[i * 3 + 2][1]:.2e}"
                print(data_format.format(i + 1, rs_range, q_range, fres_range))

            print("-" * 80)

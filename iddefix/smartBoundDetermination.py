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
        Rs_bounds: list[float] = [0.8, 10],
        Q_bounds: list[float] = [0.5, 5],
        fres_bounds: list[float] = [-0.01e9, +0.01e9],
        samples: int | None = None,
        q_side: str | list[str] = "auto",
        impedance_type: str = "absolute",
        interactive: bool = False,
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
            Scaling factors [min, max] for Rs bounds. Default is [0.8, 10].
        Q_bounds : list, optional
            Scaling factors [min, max] for Q bounds. Default is [0.5, 5].
        fres_bounds : list, optional
            Offset bounds [min, max] for frequency in Hz. Default is [-0.01e9, 0.01e9].
        samples : int, optional
            Number of equally spaced frequency samples used for peak finding and
            Q estimation. If None, use the input samples. Interpolation is linear
            and does not add information beyond the input data.
        q_side : {"auto", "left", "right"} or list of these, optional
            Side of each peak used for its half-power width and Q estimate.
            "auto" uses the nearest crossing. A list chooses a side separately
            for each detected peak, in increasing frequency order. If the
            selected side has no crossing, the Q estimate defaults to 0.5.
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

        Attributes
        ----------
        peaks : numpy.ndarray or None
            Indices of detected peaks on the analysis frequency grid.
        analysis_frequency_data, analysis_impedance_data : numpy.ndarray
            Frequency grid and impedance used for peak finding. These equal the
            inputs when ``samples`` is None.
        peaks_height : numpy.ndarray or None
            Heights of the detected peaks.
        minus_3dB_points : numpy.ndarray or None
            Crossing levels for each detected peak. For real impedance these
            are the half-maximum levels (the name is kept for compatibility).
        upper_lower_bounds : numpy.ndarray or None
            One-sided width to the selected crossing for each peak.
        q_sides_used : list[str]
            Crossing side used for each peak's Q estimate.
        Nres : int or None
            Number of detected resonators.
        parameterBounds : list of tuples
            Computed parameter bounds in the format:
            [(Rs_min, Rs_max), (Q_min, Q_max), (fres_min, fres_max), ...].
            None until an interactive selection is finished.

        Methods
        -------
        find(frequency_data=None, impedance_data=None, minimum_peak_height=None,
            threshold=None, distance=None, prominence=None, samples=None,
            q_side=None, impedance_type=None, interactive=None)
            Detects impedance peaks and determines fitting parameter bounds
            automatically or from interactive selections.

        inspect()
            Plots the impedance data and highlights detected resonance peaks
            along with their selected crossing levels and widths.

        to_table(to_markdown=False)
            Displays resonance parameters in an ASCII or Markdown-formatted table.

        Notes
        -----
        - The crossing level depends on ``impedance_type``.
        - Peak detection is based on `scipy.signal.find_peaks`.
        - Computed parameter bounds are stored in `self.parameterBounds`.
        - The `inspect()` method visualizes peak detection results.
        - The `to_table()` method prints a structured table of parameter ranges.

        Returns
        -------
        parameterBounds : list of tuples
            A list of parameter bounds for fitting. Each resonance contributes
            three sets of bounds:
            - `(Rs_min, Rs_max)`: Bounds for resistance Rs.
            - `(Q_min, Q_max)`: Bounds for quality factor Q.
            - `(freq_min, freq_max)`: Bounds for the resonant frequency.

        Notes
        -----
        - Automatic peak finding uses `scipy.signal.find_peaks`.
        - The selected crossing width is used to estimate initial Q factors.
        - The detected peaks and their heights are stored in instance attributes
        `self.peaks` and `self.peaks_height`, respectively.
        - The number of detected resonances is stored in `self.Nres`.
        """

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

        self.peaks = None
        self.peaks_height = None
        self.minus_3dB_points = None
        self.upper_lower_bounds = None
        self.N_resonators = None
        self.analysis_frequency_data = None
        self.analysis_impedance_data = None
        self.q_sides_used = None
        self._selection_done = asyncio.Event()
        self._selection_figure = None
        self._selection_done_button = None
        self._selection_pick_button = None
        self._selection_undo_button = None
        self._selection_cancelled = False

        self.parameterBounds = self.find()

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
    ) -> ParameterBounds | None:
        """
        Identifies peaks in the impedance data and determines the bounds
        for fitting parameters based on the detected peaks.

        This function uses `scipy.signal.find_peaks` to locate peaks
        in the impedance data and then calculates bounds for
        fitting parameters, including resistance (Rs), quality factor (Q),
        and resonant frequency.

        Parameters
        ----------
        frequency_data : numpy.ndarray, optional
            Array containing the frequency data in Hz.
            If None, the instance attribute `self.frequency_data` is used.
        impedance_data : numpy.ndarray, optional
            Array containing the impedance data in Ohms.
            If None, the instance attribute `self.impedance_data` is used.
        minimum_peak_height : float or numpy.ndarray or 2-item list, optional
            Minimum peak height for the peak-finding algorithm.
            * If numpy.ndarray, it should have the same length as impedance_data
            * If 2-item list, specifies the [min, max] of peak heights
            An array is interpolated with the impedance when ``samples`` is set.
        threshold : float, optional
            Required vertical distance between a peak and its neighboring values
            to be considered a peak. Passed to `scipy.signal.find_peaks`.
            Default is None.
        distance : float, optional
            Required minimum horizontal distance (in indices) between peaks.
            Passed to `scipy.signal.find_peaks`. Default is None.
        prominence : float, optional
            Required prominence of peaks. The prominence measures how much a peak
            stands out compared to its surrounding values. Passed to
            `scipy.signal.find_peaks`.
            Default is None.
        samples : int, optional
            Number of samples on a linearly interpolated frequency grid. Defaults
            to the value supplied to the constructor.
        q_side : str or list of str, optional
            Use the left or right crossing for Q estimation. "auto" uses
            the nearest crossing; a list selects a side for each detected peak.
            If that side has no crossing, Q defaults to 0.5. Defaults to the
            value supplied to the constructor.
        impedance_type : {"absolute", "real"}, optional
            Choose the crossing level for Q estimation. Defaults to the
            value supplied to the constructor.
        interactive : bool, optional
            Select resonances on the plot instead of using peak finding.
            In ipympl, enable Pick, click peaks, click Done, and await
            ``wait_for_selection()`` before using the bounds. Desktop plots
            use Enter; inline plots prompt for frequencies in Hz.

        Returns
        -------
        parameterBounds : list of tuples or None
            A list of parameter bounds for fitting. Each resonance contributes
            three sets of bounds:
            - `(Rs_min, Rs_max)`: Bounds for resistance Rs.
            - `(Q_min, Q_max)`: Bounds for quality factor Q.
            - `(freq_min, freq_max)`: Bounds for the resonant frequency.
            None while interactive selection is pending.

        Notes
        -----
        - Automatic peak finding uses `scipy.signal.find_peaks`.
        - The selected crossing width is used to estimate initial Q factors.
        - The detected peaks and their heights are stored in instance attributes
        `self.peaks` and `self.peaks_height`, respectively.
        - The number of detected resonances is stored in `self.Nres`.

        """

        # Use instance attributes if no arguments are provided
        if frequency_data is None:
            frequency_data = self.frequency_data
        if impedance_data is None:
            impedance_data = self.impedance_data
        if minimum_peak_height is None:
            minimum_peak_height = self.minimum_peak_height
        if threshold is None:
            threshold = self.threshold
        if distance is None:
            distance = self.distance
        if prominence is None:
            prominence = self.prominence
        if samples is None:
            samples = self.samples
        if q_side is None:
            q_side = self.q_side
        if impedance_type is None:
            impedance_type = self.impedance_type
        if interactive is None:
            interactive = self.interactive

        frequency_data = np.asarray(frequency_data)
        impedance_data = np.asarray(impedance_data)
        if samples is not None:
            analysis_frequency_data = np.linspace(
                frequency_data[0], frequency_data[-1], samples
            )
            analysis_impedance_data = np.interp(
                analysis_frequency_data, frequency_data, impedance_data
            )
            if isinstance(minimum_peak_height, np.ndarray):
                if minimum_peak_height.shape == frequency_data.shape:
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

        if interactive:
            self.peaks = None
            self.parameterBounds = None
            self._selection_done.clear()
            self._selection_cancelled = False
            self._start_interactive_selection(q_side, impedance_type)
            return self.parameterBounds

        peaks, _ = find_peaks(
            analysis_impedance_data,
            height=minimum_peak_height,
            threshold=threshold,
            distance=distance,
            prominence=prominence,
        )
        return self._bounds_from_peaks(peaks, q_side, impedance_type)

    def _bounds_from_peaks(
        self,
        peaks: np.ndarray,
        q_side: str | list[str],
        impedance_type: str,
    ) -> ParameterBounds:
        """Estimate parameter bounds from selected analysis-grid peaks."""
        analysis_frequency_data = self.analysis_frequency_data
        analysis_impedance_data = self.analysis_impedance_data
        peaks_height = {"peak_heights": analysis_impedance_data[peaks]}

        minimum_resonator_q = 0.5
        Nres = len(peaks)
        initial_Qs = np.zeros(Nres)
        self.minus_3dB_points = np.zeros(Nres)
        self.upper_lower_bounds = np.zeros(Nres)
        self.q_sides_used = []
        sides = [q_side] * Nres if isinstance(q_side, str) else q_side

        if impedance_type == "absolute":
            crossing_fraction = np.sqrt(1 / 2)
        elif impedance_type == "real":
            crossing_fraction = 0.5
        else:
            raise ValueError("impedance_type must be 'absolute' or 'real'")

        for i, (peak, height) in enumerate(zip(peaks, peaks_height["peak_heights"])):
            crossing_level = height * crossing_fraction
            self.minus_3dB_points[i] = crossing_level
            idx_crossings = np.argwhere(
                np.diff(np.sign(analysis_impedance_data - crossing_level))
            ).flatten()

            if len(idx_crossings) == 0:
                # A truncated resonance may not cross its selected level in the
                # supplied spectrum.  Keep it in the fit with the minimum
                # physically supported Q rather than dropping it.
                upper_lower_bound = 0.0
                selected_side = sides[i]
            else:
                # Locate crossings within their bracketing samples. Using the
                # left sample directly can give zero width when it is the peak.
                left_frequency = analysis_frequency_data[idx_crossings]
                right_frequency = analysis_frequency_data[idx_crossings + 1]
                left_impedance = analysis_impedance_data[idx_crossings]
                right_impedance = analysis_impedance_data[idx_crossings + 1]
                crossing_frequency = left_frequency + (
                    (crossing_level - left_impedance)
                    * (right_frequency - left_frequency)
                    / (right_impedance - left_impedance)
                )
                peak_frequency = analysis_frequency_data[peak]
                left_crossings = crossing_frequency[crossing_frequency < peak_frequency]
                right_crossings = crossing_frequency[
                    crossing_frequency > peak_frequency
                ]
                left_width = (
                    peak_frequency - left_crossings[-1]
                    if len(left_crossings)
                    else np.inf
                )
                right_width = (
                    right_crossings[0] - peak_frequency
                    if len(right_crossings)
                    else np.inf
                )
                selected_side = sides[i]
                if selected_side == "auto":
                    selected_side = "left" if left_width <= right_width else "right"
                if selected_side == "left":
                    upper_lower_bound = left_width
                elif selected_side == "right":
                    upper_lower_bound = right_width
                else:
                    raise ValueError("q_side must be 'auto', 'left', or 'right'")
                if not np.isfinite(upper_lower_bound):
                    upper_lower_bound = 0.0
            self.upper_lower_bounds[i] = upper_lower_bound
            self.q_sides_used.append(selected_side)

            if upper_lower_bound <= 0.0:
                estimated_Q = minimum_resonator_q
            else:
                estimated_Q = analysis_frequency_data[peak] / (upper_lower_bound * 2)

            # Q = 0.5 is the lowest value supported by the resonator
            # formalism.  It also makes the bound estimation robust for very
            # broad or poorly resolved resonances.
            initial_Qs[i] = max(minimum_resonator_q, estimated_Q)

        parameterBounds = []

        for i in range(Nres):
            # Add the fixed bounds
            Rs_bounds = (
                peaks_height["peak_heights"][i] * self.Rs_bounds[0],
                peaks_height["peak_heights"][i] * self.Rs_bounds[1],
            )
            Q_bounds = (
                max(minimum_resonator_q, initial_Qs[i] * self.Q_bounds[0]),
                initial_Qs[i] * self.Q_bounds[1],
            )
            freq_bounds = (
                analysis_frequency_data[peaks[i]] + self.fres_bounds[0],
                analysis_frequency_data[peaks[i]] + self.fres_bounds[1],
            )

            if peaks_height["peak_heights"][i] < 0:
                Rs_bounds = (
                    Rs_bounds[1],
                    Rs_bounds[0],
                )  # Swap for negative peaks
            parameterBounds.extend([Rs_bounds, Q_bounds, freq_bounds])

        # Store peaks and peaks_height as instance attributes
        self.peaks = peaks
        self.peaks_height = peaks_height
        self.N_resonators = len(parameterBounds) / 3
        self.parameterBounds = parameterBounds
        return parameterBounds

    def _start_interactive_selection(
        self, q_side: str | list[str], impedance_type: str
    ) -> None:
        """Pick peaks with controls suited to the active Matplotlib backend."""
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
            self._bounds_from_peaks(peaks, q_side, impedance_type)
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

    def inspect(self) -> None:
        plt.figure()
        plt.plot(self.analysis_frequency_data, self.analysis_impedance_data)

        if self.peaks is not None:
            for i, (peak, minus_3dB_point, upper_lower_bound) in enumerate(
                zip(self.peaks, self.minus_3dB_points, self.upper_lower_bounds)
            ):
                plt.plot(
                    self.analysis_frequency_data[peak],
                    self.analysis_impedance_data[peak],
                    "x",
                    color="black",
                )
                plt.vlines(
                    self.analysis_frequency_data[peak],
                    ymin=minus_3dB_point,
                    ymax=self.analysis_impedance_data[peak],
                    color="r",
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
                    color="g",
                    linestyle="--",
                )
                plt.text(
                    self.analysis_frequency_data[peak],
                    self.analysis_impedance_data[peak],
                    f"#{i + 1}",
                    fontsize=9,
                )
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Impedance [Ohm]")
        plt.title("Smart Bound Determination")
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

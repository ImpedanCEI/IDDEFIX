#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:34:10 2020

@author: MaltheRaschke
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks

class SmartBoundDetermination:

    def __init__(self, frequency_data, impedance_data,
                 minimum_peak_height=1.0,
                 Rs_bounds=[0.8, 10],
                 Q_bounds=[0.5, 5],
                 fres_bounds=[-0.01e9, +0.01e9]):
        """
        Automatically determines parameter bounds for resonance fitting
        by detecting impedance peaks in frequency-domain data.

        This class uses `scipy.signal.find_peaks` to identify resonances and
        estimates the bounds for resistance (Rs), quality factor (Q), and
        resonant frequency (fres) using the 3dB bandwidth method.

        Parameters
        ----------
        frequency_data : numpy.ndarray
            Frequency data in Hz.
        impedance_data : numpy.ndarray
            Impedance magnitude data in Ohms.
        minimum_peak_height : float, optional
            Minimum height for a peak to be considered a resonance. Default is 1.0.
        Rs_bounds : list, optional
            Scaling factors [min, max] for Rs bounds. Default is [0.8, 10].
        Q_bounds : list, optional
            Scaling factors [min, max] for Q bounds. Default is [0.5, 5].
        fres_bounds : list, optional
            Offset bounds [min, max] for frequency in Hz. Default is [-0.01e9, 0.01e9].

        Attributes
        ----------
        peaks : numpy.ndarray or None
            Indices of detected peaks in the impedance data.
        peaks_height : numpy.ndarray or None
            Heights of the detected peaks.
        minus_3dB_points : numpy.ndarray or None
            3dB bandwidth points for each detected peak.
        upper_lower_bounds : numpy.ndarray or None
            Upper and lower frequency bounds for each peak.
        Nres : int or None
            Number of detected resonators.
        parameterBounds : list of tuples
            Computed parameter bounds in the format:
            [(Rs_min, Rs_max), (Q_min, Q_max), (fres_min, fres_max), ...].

        Methods
        -------
        find(frequency_data=None, impedance_data=None, minimum_peak_height=None,
            threshold=None, distance=None, prominence=None)
            Detects impedance peaks and determines fitting parameter bounds
            using `scipy.signal.find_peaks`

        inspect()
            Plots the impedance data and highlights detected resonance peaks
            along with their 3dB bandwidth estimates.

        to_table(to_markdown=False)
            Displays resonance parameters in an ASCII or Markdown-formatted table.

        Notes
        -----
        - The 3dB bandwidth method is used to estimate Q factors and set frequency bounds.
        - Peak detection is based on `scipy.signal.find_peaks`.
        - Computed parameter bounds are stored in `self.parameterBounds`.
        - The `inspect()` method visualizes peak detection results.
        - The `to_table()` method prints a structured table of parameter ranges.
        """

        self.frequency_data = frequency_data
        self.impedance_data = impedance_data
        self.minimum_peak_height = minimum_peak_height
        self.Rs_bounds = Rs_bounds
        self.Q_bounds = Q_bounds
        self.fres_bounds = fres_bounds

        self.peaks = None
        self.peaks_height = None
        self.minus_3dB_points = None
        self.upper_lower_bounds = None
        self.N_resonators = None

        self.parameterBounds = self.find()

    def find(self, frequency_data=None, impedance_data=None,
             minimum_peak_height=None, threshold=None,
             distance=None, prominence=None):
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
        threshold : float, optional
            Required vertical distance between a peak and its neighboring values
            to be considered a peak. Passed to `scipy.signal.find_peaks`. Default is None.
        distance : float, optional
            Required minimum horizontal distance (in indices) between peaks.
            Passed to `scipy.signal.find_peaks`. Default is None.
        prominence : float, optional
            Required prominence of peaks. The prominence measures how much a peak
            stands out compared to its surrounding values. Passed to `scipy.signal.find_peaks`.
            Default is None.

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
        - The peak-finding algorithm is implemented using `scipy.signal.find_peaks`.
        See the official documentation for more details:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
        - The 3dB bandwidth method is used to estimate initial Q factors and
        define frequency bounds.
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


        # Find the peaks of the impedance data
        peaks, peaks_height = find_peaks(impedance_data,
                                         height=minimum_peak_height,
                                         threshold=threshold, distance=distance,
                                         prominence=prominence)

        # Store peaks and peaks_height as instance attributes
        self.peaks = peaks
        self.peaks_height = peaks_height

        Nres = len(peaks)
        initial_Qs = np.zeros(Nres)
        self.minus_3dB_points = np.zeros(Nres)
        self.upper_lower_bounds = np.zeros(Nres)

        for i, (peak, height) in enumerate(zip(peaks, peaks_height['peak_heights'])):
            minus_3dB_point = height * np.sqrt(1/2)
            self.minus_3dB_points[i] = minus_3dB_point
            idx_crossings = np.argwhere(np.diff(np.sign(impedance_data - minus_3dB_point))).flatten()

            upper_lower_bound = np.min(np.abs(frequency_data[idx_crossings] - frequency_data[peak]))
            self.upper_lower_bounds[i] = upper_lower_bound

            initial_Qs[i] = frequency_data[peak]/(upper_lower_bound*2)

        parameterBounds = []

        for i in range(Nres):
            # Add the fixed bounds
            Rs_bounds = (peaks_height['peak_heights'][i]*self.Rs_bounds[0], peaks_height['peak_heights'][i]*self.Rs_bounds[1])
            Q_bounds = (initial_Qs[i]*self.Q_bounds[0] , initial_Qs[i]*self.Q_bounds[1])
            freq_bounds = (frequency_data[peaks[i]]+self.fres_bounds[0], frequency_data[peaks[i]]+self.fres_bounds[1])

            parameterBounds.extend([Rs_bounds, Q_bounds, freq_bounds])

        self.N_resonators = len(parameterBounds)/3
        self.parameterBounds = parameterBounds
        return parameterBounds

    def inspect(self):
        plt.figure()
        plt.plot(self.frequency_data, self.impedance_data)

        if self.peaks is not None:
            for i , (peak, minus_3dB_point, upper_lower_bound) in enumerate(zip(self.peaks, self.minus_3dB_points, self.upper_lower_bounds)):
                plt.plot(self.frequency_data[peak], self.impedance_data[peak], 'x', color='black')
                plt.vlines(self.frequency_data[peak], ymin=minus_3dB_point, ymax=self.impedance_data[peak], color='r', linestyle='--')
                plt.hlines(minus_3dB_point, xmin=self.frequency_data[peak] - upper_lower_bound, xmax=self.frequency_data[peak] + upper_lower_bound, color='g', linestyle='--')
                plt.text(self.frequency_data[peak], self.impedance_data[peak], f'#{i+1}', fontsize=9)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Impedance [Ohm]')
        plt.title('Smart Bound Determination')
        plt.show()

        return None

    def to_table(self, to_markdown=False):
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
        params = self.parameterBounds
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
            print(header_format.format("Resonator", "Rs [Ohm/m or Ohm]", "Q", "fres [Hz]"))
            print("-" * 80)

            # Print data
            for i in range(N_resonators):
                rs_range = f"{params[i * 3][0]:.2f} to {params[i * 3][1]:.2f}"
                q_range = f"{params[i * 3 + 1][0]:.2f} to {params[i * 3 + 1][1]:.2f}"
                fres_range = f"{params[i * 3 + 2][0]:.2e} to {params[i * 3 + 2][1]:.2e}"
                print(data_format.format(i + 1, rs_range, q_range, fres_range))

            print("-" * 80)
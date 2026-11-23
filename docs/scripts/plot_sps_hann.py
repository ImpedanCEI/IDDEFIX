"""Reproduce the SmartBounds half-Hann comparison with SPS transition data.

Run from the repository root with:
    PYTHONPATH=. python docs/scripts/plot_sps_hann.py
"""

from pathlib import Path

import matplotlib
import numpy as np
from scipy.constants import c
from scipy.interpolate import CubicSpline

import iddefix

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[2]
    data = np.loadtxt(root / "examples/data/004_SPS_model_transitions_q26.txt")
    time = data[:, 0] * 1e-9
    frequency, impedance = iddefix.compute_deconvolution(time, data[:, 2], sigma=1e-10)
    impedance *= 1j  # Transverse convention used in example 004b.
    wake_length = time[-1] * c

    # Reproduce the previous full Hann with the same interpolation and symmetry.
    positive = frequency > 0
    extended_frequency = np.r_[-frequency[positive][::-1], frequency]
    extended_impedance = np.r_[-np.conj(impedance[positive][::-1]), impedance]
    interpolate = CubicSpline(extended_frequency, extended_impedance)
    old_shift = c / wake_length
    old_valid = frequency + old_shift <= frequency[-1]
    old_frequency = frequency[old_valid]
    full_hann = (
        0.5 * interpolate(old_frequency)
        - 0.25 * interpolate(old_frequency - old_shift)
        - 0.25 * interpolate(old_frequency + old_shift)
    )
    half_hann, valid = iddefix.SmartBoundDetermination._hann_impedance(
        frequency, impedance, frequency, wake_length, "transverse"
    )

    with plt.rc_context({"font.size": 10}):
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), layout="constrained")
        normalized_time = np.linspace(0, 1, 501)
        axes[0].plot(
            normalized_time,
            0.5 - 0.5 * np.cos(2 * np.pi * normalized_time),
            label="Before: full Hann",
            color="tab:orange",
        )
        axes[0].plot(
            normalized_time,
            0.5 + 0.5 * np.cos(np.pi * normalized_time),
            label="After: descending half Hann",
            color="tab:blue",
        )
        axes[0].set(xlabel="Time / record length", ylabel="Wake weight", xlim=(0, 1))
        for ax in axes[1:]:
            ax.plot(
                frequency / 1e9,
                impedance.real,
                color="0.65",
                lw=1,
                label="Unwindowed impedance",
            )
            ax.plot(
                old_frequency / 1e9,
                full_hann.real,
                color="tab:orange",
                lw=1.5,
                label="Before: full Hann",
            )
            ax.plot(
                frequency[valid] / 1e9,
                half_hann.real,
                color="tab:blue",
                lw=1.5,
                label="After: descending half Hann",
            )
            ax.set(
                xlabel="Frequency [GHz]", ylabel=r"$\mathrm{Re}(Z_\perp)$ [$\Omega$/m]"
            )
        axes[1].set(xlim=(0, 3), title="SPS transitions: detection traces")
        axes[2].set(xlim=(0.95, 1.8), title="Resonance detail")
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.legend(loc="upper right", fontsize=9)
        fig.suptitle("One-sided Hann preserves the beginning of the wake")
        output = root / "docs/images/sps_hann_comparison.png"
        fig.savefig(output, dpi=160)
        plt.close(fig)
        print(output)


if __name__ == "__main__":
    main()

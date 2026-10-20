"""Fit CST lossy-wall impedances obtained from finite wake lengths."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from iddefix.poleResidueFitting import fit_poles_evolutionary
from iddefix.poleResidueFormulas import PoleResidue

PIPE_RADIUS = 20.0e-3
PIPE_LENGTH = 0.5
CONDUCTIVITY = 1.4e6

MINIMUM_FIT_FREQUENCY = 1.0e7
MAXIMUM_FIT_FREQUENCY = 1.0e9

NUMBER_REAL_POLES = 6

WAKE_LENGTHS = {
    0.3: "lossy_wall_WL_300mm.txt",
    1.0: "lossy_wall_WL_1000mm.txt",
    3.0: "lossy_wall_WL_3000mm.txt",
    10.0: "lossy_wall_WL_10000mm.txt",
}


def load_cst_impedance(file_path):
    """Load frequency, real(Z), and imaginary(Z) from a CST export."""
    data = np.loadtxt(file_path, comments="#")

    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"expected at least three columns in {file_path}")

    frequencies = data[:, 0]
    impedance = data[:, 1] + 1j * data[:, 2]

    valid = (
        np.isfinite(frequencies)
        & np.isfinite(impedance.real)
        & np.isfinite(impedance.imag)
        & (frequencies >= MINIMUM_FIT_FREQUENCY)
        & (frequencies <= MAXIMUM_FIT_FREQUENCY)
    )

    return frequencies[valid], impedance[valid]


def longitudinal_lossy_wall_impedance(frequencies):
    """Thick-wall reference for a circular beam pipe."""
    mu_0 = 4.0e-7 * np.pi
    s = 2j * np.pi * frequencies

    coefficient = (
        PIPE_LENGTH / (2.0 * np.pi * PIPE_RADIUS) * np.sqrt(mu_0 / CONDUCTIVITY)
    )

    return coefficient * np.sqrt(s)


def build_real_pole_bounds(number_poles):
    """Distribute stable real-pole bounds across the fit band."""
    minimum_rate = 2.0 * np.pi * MINIMUM_FIT_FREQUENCY / 100.0
    maximum_rate = 2.0 * np.pi * MAXIMUM_FIT_FREQUENCY * 100.0

    edges = np.linspace(
        np.log10(minimum_rate),
        np.log10(maximum_rate),
        number_poles + 1,
    )

    return [(edges[index], edges[index + 1]) for index in range(number_poles)]


def relative_error(predicted, target):
    """Return the pointwise complex relative error."""
    floor = np.max(np.abs(target)) * 1.0e-12
    return np.abs(predicted - target) / np.maximum(
        np.abs(target),
        floor,
    )


def main():
    data_directory = Path(r"D:\Iddefix\Data")
    results = {}

    for index, (wake_length, filename) in enumerate(WAKE_LENGTHS.items()):
        file_path = data_directory / filename

        if not file_path.is_file():
            raise FileNotFoundError(f"CST export not found: {file_path}")

        frequencies, impedance = load_cst_impedance(file_path)

        print(f"Fitting {filename}...")
        print(
            f"  {frequencies.size} samples from "
            f"{frequencies[0]:.3e} to {frequencies[-1]:.3e} Hz"
        )

        optimization = fit_poles_evolutionary(
            frequencies=frequencies,
            impedance=impedance,
            number_real_poles=NUMBER_REAL_POLES,
            number_complex_pairs=0,
            parameter_bounds=build_real_pole_bounds(NUMBER_REAL_POLES),
            wake_length=wake_length,
            fit_direct_term=True,
            amplitude_weighting="relative",
            frequency_weighting="log",
            maxiter=300,
            popsize=12,
            tol=1.0e-7,
            polish=True,
            seed=2026 + index,
            workers=1,
        )

        residue_fit = optimization.residue_fit
        finite_fit = residue_fit.fitted_impedance

        full_impedance = PoleResidue.impedance(
            frequencies=frequencies,
            poles=residue_fit.poles,
            residues=residue_fit.residues,
            direct_term=residue_fit.direct_term,
        )

        error = relative_error(finite_fit, impedance)

        print(f"  objective: {optimization.objective_value:.6e}")
        print(f"  RMS relative error: {np.sqrt(np.mean(error**2)):.6e}")
        print(f"  maximum relative error: {np.max(error):.6e}")
        print(f"  real poles: {optimization.real_poles}")
        print(f"  residues: {residue_fit.residues}")
        print(f"  direct term: {residue_fit.direct_term:.6e}")
        print()

        results[wake_length] = {
            "frequencies": frequencies,
            "impedance": impedance,
            "finite_fit": finite_fit,
            "full_impedance": full_impedance,
            "relative_error": error,
        }

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        sharex=True,
    )

    for axis, (wake_length, result) in zip(
        axes.flat,
        results.items(),
    ):
        frequencies = result["frequencies"]
        impedance = result["impedance"]
        finite_fit = result["finite_fit"]

        axis.semilogx(
            frequencies,
            impedance.real,
            color="black",
            label="CST: real",
        )
        axis.semilogx(
            frequencies,
            impedance.imag,
            color="black",
            linestyle="--",
            label="CST: imaginary",
        )
        axis.semilogx(
            frequencies,
            finite_fit.real,
            label="Finite-wake fit: real",
        )
        axis.semilogx(
            frequencies,
            finite_fit.imag,
            linestyle="--",
            label="Finite-wake fit: imaginary",
        )
        axis.set_title(f"Wake length = {wake_length:g} m")
        axis.set_ylabel("Longitudinal impedance [Ohm]")
        axis.grid(True, which="both")

    axes[-1, 0].set_xlabel("Frequency [Hz]")
    axes[-1, 1].set_xlabel("Frequency [Hz]")
    axes[0, 0].legend()
    figure.tight_layout()

    reconstruction_figure, reconstruction_axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
    )

    reference_frequencies = next(iter(results.values()))["frequencies"]
    analytical_reference = longitudinal_lossy_wall_impedance(reference_frequencies)

    reconstruction_axes[0].loglog(
        reference_frequencies,
        analytical_reference.real,
        color="black",
        linewidth=2.5,
        label="Analytical lossy wall: real",
    )
    reconstruction_axes[0].loglog(
        reference_frequencies,
        analytical_reference.imag,
        color="black",
        linestyle="--",
        linewidth=2.5,
        label="Analytical lossy wall: imaginary",
    )

    for wake_length, result in results.items():
        frequencies = result["frequencies"]
        full_impedance = result["full_impedance"]

        line = reconstruction_axes[0].loglog(
            frequencies,
            np.abs(full_impedance.real),
            label=f"Reconstructed: {wake_length:g} m",
        )[0]
        reconstruction_axes[0].loglog(
            frequencies,
            np.abs(full_impedance.imag),
            color=line.get_color(),
            linestyle="--",
        )
        reconstruction_axes[1].loglog(
            frequencies,
            result["relative_error"],
            label=f"Finite fit: {wake_length:g} m",
        )

    reconstruction_axes[0].set_ylabel("Longitudinal impedance [Ohm]")
    reconstruction_axes[0].grid(True, which="both")
    reconstruction_axes[0].legend()

    reconstruction_axes[1].set_xlabel("Frequency [Hz]")
    reconstruction_axes[1].set_ylabel("Pointwise relative fit error")
    reconstruction_axes[1].grid(True, which="both")
    reconstruction_axes[1].legend()

    reconstruction_figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

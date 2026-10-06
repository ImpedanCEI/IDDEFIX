"""Compare resonator and pole-residue extrapolation on CST cavity data."""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

import iddefix
from iddefix.poleResidueFitting import fit_poles_evolutionary
from iddefix.poleResidueFormulas import PoleResidue

NUMBER_RESONATORS = 2
NUMBER_COMPLEX_PAIRS = 2
WAKE_LENGTH = 25.4  # metres
RANDOM_SEED = 2026


def load_impedance(file_path):
    """Load a CST impedance export and remove the zero-frequency point."""
    data = np.loadtxt(file_path, comments="#", delimiter="\t")
    frequencies = data[:, 0] * 1.0e9
    impedance = data[:, 1] + 1j * data[:, 2]

    positive_frequency = frequencies > 0.0

    return (
        frequencies[positive_frequency],
        impedance[positive_frequency],
    )


def resonator_bounds():
    """Return bounds for the two visible cavity resonances."""
    return [
        (1.0e2, 1.0e4),
        (1.0, 1.0e3),
        (0.45e9, 0.65e9),
        (1.0e2, 1.0e4),
        (1.0, 1.0e3),
        (0.70e9, 0.90e9),
    ]


def pole_bounds():
    """Return log10 bounds for decay rates and damped frequencies."""
    minimum_decay_rate = 2.0 * np.pi * 0.45e9 / (2.0 * 1.0e3)
    maximum_decay_rate = 2.0 * np.pi * 0.90e9 / (2.0 * 1.0)

    decay_bound = (
        np.log10(minimum_decay_rate),
        np.log10(maximum_decay_rate),
    )

    first_frequency_bound = (
        np.log10(2.0 * np.pi * 0.40e9),
        np.log10(2.0 * np.pi * 0.70e9),
    )

    second_frequency_bound = (
        np.log10(2.0 * np.pi * 0.65e9),
        np.log10(2.0 * np.pi * 0.95e9),
    )

    # Parameter order: alpha_1, alpha_2, beta_1, beta_2.
    return [
        decay_bound,
        decay_bound,
        first_frequency_bound,
        second_frequency_bound,
    ]


def normalized_l2_error(predicted, target):
    """Return ||predicted-target||_2 / ||target||_2."""
    return float(np.linalg.norm(predicted - target) / np.linalg.norm(target))


def pointwise_relative_error(predicted, target):
    """Return a safely normalized pointwise complex error."""
    magnitude_floor = np.max(np.abs(target)) * 1.0e-12

    return np.abs(predicted - target) / np.maximum(
        np.abs(target),
        magnitude_floor,
    )


def print_metrics(name, finite_fit, partial_data, full_fit, full_data, runtime):
    """Print finite-fit and fully-decayed extrapolation errors."""
    finite_relative_error = pointwise_relative_error(
        finite_fit,
        partial_data,
    )
    full_relative_error = pointwise_relative_error(
        full_fit,
        full_data,
    )

    print(name)
    print(f"  runtime: {runtime:.2f} s")
    print(
        "  finite normalized L2 error: "
        f"{normalized_l2_error(finite_fit, partial_data):.6e}"
    )
    print(
        f"  finite RMS relative error: {np.sqrt(np.mean(finite_relative_error**2)):.6e}"
    )
    print(
        "  extrapolation normalized L2 error: "
        f"{normalized_l2_error(full_fit, full_data):.6e}"
    )
    print(
        "  extrapolation RMS relative error: "
        f"{np.sqrt(np.mean(full_relative_error**2)):.6e}"
    )
    print()


def main():
    data_directory = Path(__file__).resolve().parent / "data"

    partial_frequency, partial_impedance = load_impedance(
        data_directory / "002_impedance_acceleratorCavity_partially_decayed.txt"
    )
    full_frequency, full_impedance = load_impedance(
        data_directory / "002_impedance_acceleratorCavity_fully_decayed.txt"
    )

    if not np.allclose(partial_frequency, full_frequency):
        raise ValueError(
            "partially and fully decayed data use different frequency grids"
        )

    frequencies = partial_frequency

    # Original resonator fit.
    np.random.seed(RANDOM_SEED)

    resonator_model = iddefix.EvolutionaryAlgorithm(
        x_data=frequencies,
        y_data=partial_impedance,
        N_resonators=NUMBER_RESONATORS,
        parameterBounds=resonator_bounds(),
        plane="longitudinal",
        wake_length=WAKE_LENGTH,
        objectiveFunction=iddefix.ObjectiveFunctions.sumOfSquaredError,
    )

    start_time = perf_counter()
    resonator_model.run_differential_evolution(
        maxiter=1000,
        popsize=30,
        tol=1.0e-7,
        mutation=(0.4, 1.0),
        crossover_rate=0.7,
        solver="scipy",
    )
    resonator_model.run_minimization_algorithm()
    resonator_runtime = perf_counter() - start_time

    resonator_finite = resonator_model.get_impedance(
        frequency_data=frequencies,
        wake_length=WAKE_LENGTH,
    )
    resonator_full = resonator_model.get_impedance(
        frequency_data=frequencies,
        wake_length=None,
    )

    # Pole-residue fit with the same rational pole order.
    start_time = perf_counter()
    pole_result = fit_poles_evolutionary(
        frequencies=frequencies,
        impedance=partial_impedance,
        number_real_poles=0,
        number_complex_pairs=NUMBER_COMPLEX_PAIRS,
        parameter_bounds=pole_bounds(),
        wake_length=WAKE_LENGTH,
        amplitude_weighting="uniform",
        frequency_weighting="samples",
        fit_direct_term=False,
        enforce_zero_dc=True,
        maxiter=1000,
        popsize=30,
        mutation=(0.4, 1.0),
        crossover_rate=0.7,
        tol=1.0e-7,
        polish=True,
        seed=RANDOM_SEED,
        workers=-1,
    )
    pole_runtime = perf_counter() - start_time

    pole_finite = pole_result.residue_fit.fitted_impedance
    pole_full = PoleResidue.impedance(
        frequencies=frequencies,
        poles=pole_result.residue_fit.poles,
        residues=pole_result.residue_fit.residues,
        direct_term=pole_result.residue_fit.direct_term,
    )

    print()
    print_metrics(
        "Resonator fit",
        resonator_finite,
        partial_impedance,
        resonator_full,
        full_impedance,
        resonator_runtime,
    )
    print_metrics(
        "Pole-residue fit",
        pole_finite,
        partial_impedance,
        pole_full,
        full_impedance,
        pole_runtime,
    )

    print("Pole-residue parameters")
    print(f"  poles: {pole_result.residue_fit.poles}")
    print(f"  residues: {pole_result.residue_fit.residues}")
    print()

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(10, 11),
        sharex=True,
    )

    axes[0].plot(
        frequencies,
        partial_impedance.real,
        color="black",
        linewidth=2.5,
        label="Partially decayed CST: real",
    )
    axes[0].plot(
        frequencies,
        partial_impedance.imag,
        color="black",
        linewidth=2.5,
        linestyle="--",
        label="Partially decayed CST: imaginary",
    )

    for fitted_impedance, label, color in [
        (resonator_finite, "Resonator", "tab:blue"),
        (pole_finite, "Pole-residue", "tab:orange"),
    ]:
        axes[0].plot(
            frequencies,
            fitted_impedance.real,
            color=color,
            label=f"{label}: real",
        )
        axes[0].plot(
            frequencies,
            fitted_impedance.imag,
            color=color,
            linestyle="--",
            label=f"{label}: imaginary",
        )

    axes[0].set_ylabel("Finite-wake impedance [Ohm]")
    axes[0].set_title("Fit of the partially decayed CST impedance")
    axes[0].grid(True)
    axes[0].legend(ncol=2)

    axes[1].plot(
        frequencies,
        full_impedance.real,
        color="black",
        linewidth=2.5,
        label="Fully decayed CST: real",
    )
    axes[1].plot(
        frequencies,
        full_impedance.imag,
        color="black",
        linewidth=2.5,
        linestyle="--",
        label="Fully decayed CST: imaginary",
    )

    for fitted_impedance, label, color in [
        (resonator_full, "Resonator", "tab:blue"),
        (pole_full, "Pole-residue", "tab:orange"),
    ]:
        axes[1].plot(
            frequencies,
            fitted_impedance.real,
            color=color,
            label=f"{label}: real",
        )
        axes[1].plot(
            frequencies,
            fitted_impedance.imag,
            color=color,
            linestyle="--",
            label=f"{label}: imaginary",
        )

    axes[1].set_ylabel("Fully decayed impedance [Ohm]")
    axes[1].set_title("Extrapolation compared with fully decayed CST data")
    axes[1].grid(True)
    axes[1].legend(ncol=2)

    resonator_error = pointwise_relative_error(
        resonator_full,
        full_impedance,
    )
    pole_error = pointwise_relative_error(
        pole_full,
        full_impedance,
    )

    axes[2].semilogy(
        frequencies,
        resonator_error,
        label="Resonator extrapolation",
    )
    axes[2].semilogy(
        frequencies,
        pole_error,
        label="Pole-residue extrapolation",
    )
    axes[2].set_xlabel("Frequency [Hz]")
    axes[2].set_ylabel("Pointwise relative error")
    axes[2].grid(True, which="both")
    axes[2].legend()

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

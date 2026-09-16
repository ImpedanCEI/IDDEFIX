"""Compare transverse resonator and pole-residue fits of an SPS transition."""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c_light

import iddefix
from iddefix.poleResidueFitting import (
    build_fit_weights,
    fit_poles_evolutionary,
)
from iddefix.poleResidueFormulas import PoleResidue

REAL_POLE_COUNTS = (0, 1)

BACKGROUND_SAMPLES = 50
SAMPLES_PER_RESONANCE = 10

RANDOM_SEED = 2026

# Differential Evolution optimizes all resonator parameters in the
# legacy model.  In the pole-residue model it optimizes only the poles;
# the residues are determined by linear least squares for every pole
# candidate.  The optimizer budgets are therefore configured separately.
LEGACY_MAXITER = 200
LEGACY_POPSIZE = 10

POLE_RESIDUE_MAXITER = 20
POLE_RESIDUE_POPSIZE = 4
POLE_RESIDUE_POLISH = False


def weighted_complex_error(
    parameters,
    fitFunction,
    x,
    y,
):
    """Use the same weighting as the pole-residue fit."""
    fitted = fitFunction(
        x,
        parameters,
    )

    weights = build_fit_weights(
        frequencies=x,
        impedance=y,
        amplitude_weighting="sqrt_relative",
        frequency_weighting="linear",
    )

    return float(np.sum(np.abs(weights * (fitted - y)) ** 2))


def smart_bounds_to_complex_pole_bounds(
    parameter_bounds,
):
    """Convert SmartBounds (Q, fres) to log10(alpha, beta)."""
    decay_bounds = []
    frequency_bounds = []

    for index in range(
        0,
        len(parameter_bounds),
        3,
    ):
        q_min, q_max = parameter_bounds[index + 1]

        f_min, f_max = parameter_bounds[index + 2]

        q_min = max(
            q_min,
            0.5 + 1.0e-12,
        )

        q_max = max(
            q_max,
            q_min,
        )

        f_min = max(
            f_min,
            np.finfo(float).tiny,
        )

        f_max = max(
            f_max,
            f_min,
        )

        omega_min = 2.0 * np.pi * f_min
        omega_max = 2.0 * np.pi * f_max

        alpha_min = omega_min / (2.0 * q_max)

        alpha_max = omega_max / (2.0 * q_min)

        beta_min = omega_min * np.sqrt(1.0 - 1.0 / (4.0 * q_min**2))

        beta_max = omega_max * np.sqrt(1.0 - 1.0 / (4.0 * q_max**2))

        decay_bounds.append(
            (
                np.log10(alpha_min),
                np.log10(alpha_max),
            )
        )

        frequency_bounds.append(
            (
                np.log10(beta_min),
                np.log10(beta_max),
            )
        )

    return decay_bounds + frequency_bounds


def real_pole_bounds(
    number_real_poles,
    frequencies,
):
    """Distribute real-pole bounds over measured time scales."""
    if number_real_poles == 0:
        return []

    # frequencies[0] is zero.
    minimum_rate = 2.0 * np.pi * frequencies[1] / 100.0

    maximum_rate = 2.0 * np.pi * frequencies[-1] * 100.0

    edges = np.linspace(
        np.log10(minimum_rate),
        np.log10(maximum_rate),
        number_real_poles + 1,
    )

    return list(
        zip(
            edges[:-1],
            edges[1:],
        )
    )


def select_fit_indices(
    frequencies,
    smart_bounds,
):
    """Select broadband and resonance-region samples."""
    background_indices = np.linspace(
        0,
        frequencies.size - 1,
        min(
            BACKGROUND_SAMPLES,
            frequencies.size,
        ),
        dtype=int,
    )

    selected_indices = [background_indices]

    for resonance_index, peak_index in enumerate(smart_bounds.peaks):
        peak_frequency = frequencies[peak_index]

        resonance_width = smart_bounds.upper_lower_bounds[resonance_index]

        inside_window = np.flatnonzero(
            np.abs(frequencies - peak_frequency) <= 2.0 * resonance_width
        )

        if inside_window.size > SAMPLES_PER_RESONANCE:
            local_indices = np.linspace(
                0,
                inside_window.size - 1,
                SAMPLES_PER_RESONANCE,
                dtype=int,
            )

            inside_window = inside_window[local_indices]

        selected_indices.append(inside_window)

        # Always include the exact detected peak.
        selected_indices.append(
            np.asarray(
                [peak_index],
                dtype=int,
            )
        )

    return np.unique(np.concatenate(selected_indices))


def relative_error(
    fitted,
    target,
):
    """Calculate pointwise relative complex error."""
    magnitude_floor = np.max(np.abs(target)) * 1.0e-12

    return np.abs(fitted - target) / np.maximum(
        np.abs(target),
        magnitude_floor,
    )


def print_metrics(
    name,
    fitted,
    target,
    runtime,
):
    """Print common fit metrics."""
    error = relative_error(
        fitted,
        target,
    )

    normalized_l2_error = np.linalg.norm(fitted - target) / np.linalg.norm(target)

    print(name)
    print(f"  runtime: {runtime:.2f} s")

    print(f"  normalized L2 error: {normalized_l2_error:.6e}")

    print(f"  RMS relative error: {np.sqrt(np.mean(error**2)):.6e}")

    print(f"  maximum relative error: {np.max(error):.6e}")

    print()


def main():
    data_path = (
        Path(__file__).resolve().parent / "data" / "004_SPS_model_transitions_q26.txt"
    )

    data = np.loadtxt(
        data_path,
        comments="#",
        delimiter="\t",
    )

    wake_time = data[:, 0] * 1.0e-9

    dipolar_wake_potential = data[:, 2]

    # Fourier transform of the complete wake.
    frequencies, wake_transform = iddefix.compute_fft(
        wake_time,
        dipolar_wake_potential,
    )

    # Transverse convention used by example 004b.
    impedance = -1j * wake_transform

    wake_length = wake_time[-1] * c_light

    # -------------------------------------------------
    # SmartBounds resonance detection
    # -------------------------------------------------
    #
    # The old baseline-removal procedure is used only
    # for detecting resonance positions. The actual fit
    # target remains the full complex impedance.
    # -------------------------------------------------

    early_wake = np.where(
        wake_time <= 1.0e-9,
        dipolar_wake_potential,
        0.0,
    )

    _, early_wake_transform = iddefix.compute_fft(
        wake_time,
        early_wake,
    )

    detection_signal = np.abs(wake_transform) - np.abs(early_wake_transform)

    minimum_peak_heights = np.zeros_like(frequencies)

    minimum_peak_heights[frequencies < 2.0e9] = 0.25

    minimum_peak_heights[frequencies >= 2.0e9] = 0.05

    minimum_peak_heights[frequencies >= 2.5e9] = 0.10

    smart_bounds = iddefix.SmartBoundDetermination(
        frequency_data=frequencies,
        impedance_data=detection_signal,
        minimum_peak_height=(minimum_peak_heights),
    )

    number_complex_pairs = int(smart_bounds.N_resonators)

    print(f"SmartBounds detected {number_complex_pairs} resonances.")

    print("Peak frequencies [GHz]:")

    print(frequencies[smart_bounds.peaks] / 1.0e9)

    # -------------------------------------------------
    # Reduced frequency grid
    # -------------------------------------------------

    fit_indices = select_fit_indices(
        frequencies,
        smart_bounds,
    )

    fit_frequencies = frequencies[fit_indices]

    fit_impedance = impedance[fit_indices]

    print(f"Using {fit_frequencies.size} of {frequencies.size} frequency samples.")

    print()

    fits = {}
    runtimes = {}

    # -------------------------------------------------
    # Complex transverse resonator fit
    # -------------------------------------------------

    np.random.seed(RANDOM_SEED)

    resonator_model = iddefix.EvolutionaryAlgorithm(
        x_data=fit_frequencies,
        y_data=fit_impedance,
        N_resonators=(number_complex_pairs),
        parameterBounds=(smart_bounds.parameterBounds),
        plane="transverse",
        wake_length=wake_length,
        objectiveFunction=(weighted_complex_error),
    )

    start_time = perf_counter()

    resonator_model.run_differential_evolution(
        maxiter=LEGACY_MAXITER,
        popsize=LEGACY_POPSIZE,
        tol=1.0e-4,
        mutation=(0.1, 0.5),
        crossover_rate=0.8,
        solver="scipy",
    )

    runtimes["Resonators"] = perf_counter() - start_time

    # get_impedance_from_fitFunction uses the finite-wake
    # function configured by the model constructor.
    fits["Resonators"] = resonator_model.get_impedance_from_fitFunction(
        frequency_data=frequencies,
        use_minimization=False,
    )

    # -------------------------------------------------
    # Pole-residue fits
    # -------------------------------------------------

    complex_pole_bounds = smart_bounds_to_complex_pole_bounds(
        smart_bounds.parameterBounds
    )

    pole_results = {}

    for number_real_poles in REAL_POLE_COUNTS:
        label = f"Pole-residue: +{number_real_poles} real poles"

        print(f"Fitting {label}...")

        parameter_bounds = (
            real_pole_bounds(
                number_real_poles,
                fit_frequencies,
            )
            + complex_pole_bounds
        )

        start_time = perf_counter()

        result = fit_poles_evolutionary(
            frequencies=fit_frequencies,
            impedance=fit_impedance,
            number_real_poles=(number_real_poles),
            number_complex_pairs=(number_complex_pairs),
            parameter_bounds=(parameter_bounds),
            wake_length=wake_length,
            plane="transverse",
            fit_direct_term=True,
            fit_proportional_term=True,
            enforce_zero_dc=False,
            amplitude_weighting=("sqrt_relative"),
            frequency_weighting="linear",
            residue_solver="least_squares",
            maxiter=POLE_RESIDUE_MAXITER,
            popsize=POLE_RESIDUE_POPSIZE,
            mutation=(0.1, 0.5),
            crossover_rate=0.8,
            tol=1.0e-4,
            polish=POLE_RESIDUE_POLISH,
            seed=(RANDOM_SEED + number_real_poles),
            workers=1,
        )

        runtimes[label] = perf_counter() - start_time

        pole_results[number_real_poles] = result

        fits[label] = PoleResidue.finite_wake_impedance(
            frequencies=frequencies,
            poles=result.residue_fit.poles,
            residues=result.residue_fit.residues,
            wake_length=wake_length,
            direct_term=result.residue_fit.direct_term,
            proportional_term=(result.residue_fit.proportional_term),
            plane="transverse",
        )

    # -------------------------------------------------
    # Results
    # -------------------------------------------------

    print()

    for label, fitted in fits.items():
        print_metrics(
            name=label,
            fitted=fitted,
            target=impedance,
            runtime=runtimes[label],
        )

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------

    figure, axes = plt.subplots(
        4,
        1,
        figsize=(11, 14),
        sharex=True,
    )

    axes[0].plot(
        frequencies,
        impedance.real,
        color="black",
        linewidth=2,
        label="SPS data",
    )

    axes[1].plot(
        frequencies,
        impedance.imag,
        color="black",
        linewidth=2,
        label="SPS data",
    )

    axes[2].plot(
        frequencies,
        np.abs(impedance),
        color="black",
        linewidth=2,
        label="SPS data",
    )

    for label, fitted in fits.items():
        axes[0].plot(
            frequencies,
            fitted.real,
            label=label,
        )

        axes[1].plot(
            frequencies,
            fitted.imag,
            label=label,
        )

        axes[2].plot(
            frequencies,
            np.abs(fitted),
            label=label,
        )

        axes[3].semilogy(
            frequencies,
            relative_error(
                fitted,
                impedance,
            ),
            label=label,
        )

    axes[0].set_ylabel("Real(Z_perp) [Ohm/m]")

    axes[1].set_ylabel("Imag(Z_perp) [Ohm/m]")

    axes[2].set_ylabel("Abs(Z_perp) [Ohm/m]")

    axes[3].set_ylabel("Relative error")

    axes[3].set_xlabel("Frequency [Hz]")

    for axis in axes:
        axis.grid(
            True,
            which="both",
        )
        axis.legend()

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

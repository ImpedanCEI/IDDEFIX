"""Compare SmartBounds resonator and pole-residue fits of BWS data.

SmartBounds detects the resonances and supplies bounds in (Rs, Q, fres).
The Q and fres bounds are converted to bounds for complex poles. The
legacy resonator model and the pole-residue models are fitted to the same
complex impedance samples with the same weighting. The pole-residue fits
use Differential Evolution for the poles and linear least squares for the
residues. Increasing number of real poles are added to the pole-residue model
"""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

import iddefix
from iddefix.poleResidueFitting import (
    build_fit_weights,
    fit_poles_evolutionary,
)
from iddefix.poleResidueFormulas import PoleResidue

USE_REDUCED_FREQUENCY_GRID = True

NUMBER_BACKGROUND_SAMPLES = 100
NUMBER_SAMPLES_PER_RESONANCE = 30
RESONANCE_WINDOW_FACTOR = 2.0

MINIMUM_PEAK_HEIGHT = 2.0
REAL_POLE_COUNTS = (0, 1, 2, 3, 4)
RANDOM_SEED = 2026

# Keep the optimizer budgets separate.  SciPy interprets ``popsize`` as
# a multiplier of the number of nonlinear optimization variables.
LEGACY_MAXITER = 200
LEGACY_POPSIZE = 10
LEGACY_LOCAL_MINIMIZATION = True

POLE_RESIDUE_MAXITER = 200
POLE_RESIDUE_POPSIZE = 4
POLE_RESIDUE_POLISH = True

TOLERANCE = 1.0e-4
MUTATION = (0.3, 0.8)
CROSSOVER_RATE = 0.5


def load_impedance(file_path):
    """Load the complex BWS impedance and remove zero frequency."""
    data = np.loadtxt(
        file_path,
        comments="#",
        delimiter="\t",
    )

    frequencies = data[:, 0] * 1.0e9
    impedance = data[:, 1] + 1j * data[:, 2]

    positive_frequency = frequencies > 0.0

    return (
        frequencies[positive_frequency],
        impedance[positive_frequency],
    )


def weighted_complex_error(
    parameters,
    fitFunction,
    x,
    y,
):
    """Use the same weighting as the pole-residue fits."""
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
    """Convert SmartBounds from (Q, fres) to pole bounds.

    For Q > 1/2, the upper-half-plane resonator pole is

        p = -alpha + j*beta

    with

        alpha = omega_r / (2 Q)

        beta = omega_r * sqrt(
            1 - 1 / (4 Q**2)
        )

    SmartBounds also supplies Rs bounds. These are not needed
    because the pole-residue method determines residues by
    linear least squares.
    """
    decay_bounds = []
    frequency_bounds = []

    for index in range(
        0,
        len(parameter_bounds),
        3,
    ):
        q_min, q_max = parameter_bounds[index + 1]
        f_min, f_max = parameter_bounds[index + 2]

        # A complex-conjugate pole pair requires Q > 1/2.
        # The clipping protects against numerical round-off
        # at the critical-damping boundary.
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

    # decode_log_poles expects the parameter order
    #
    # [
    #   all complex decay rates,
    #   all complex angular frequencies,
    # ]
    #
    # when no real poles are present. Real-pole bounds
    # are prepended later.
    return decay_bounds + frequency_bounds


def real_pole_bounds(
    number_real_poles,
    frequencies,
):
    """Distribute real-pole bounds over the measured time scales."""
    if number_real_poles == 0:
        return []

    minimum_rate = 2.0 * np.pi * np.min(frequencies) / 100.0

    maximum_rate = 2.0 * np.pi * np.max(frequencies) * 100.0

    logarithmic_edges = np.linspace(
        np.log10(minimum_rate),
        np.log10(maximum_rate),
        number_real_poles + 1,
    )

    return list(
        zip(
            logarithmic_edges[:-1],
            logarithmic_edges[1:],
        )
    )


def pointwise_relative_error(
    fitted,
    target,
):
    """Calculate a safely normalized pointwise complex error."""
    magnitude_floor = np.max(np.abs(target)) * 1.0e-12

    return np.abs(fitted - target) / np.maximum(
        np.abs(target),
        magnitude_floor,
    )


def print_metrics(
    name,
    frequencies,
    fitted,
    target,
    runtime,
):
    """Print common metrics for one fit."""
    weights = build_fit_weights(
        frequencies=frequencies,
        impedance=target,
        amplitude_weighting="sqrt_relative",
        frequency_weighting="samples",
    )

    relative_error = pointwise_relative_error(
        fitted,
        target,
    )

    normalized_l2_error = np.linalg.norm(fitted - target) / np.linalg.norm(target)

    weighted_normalized_l2_error = np.linalg.norm(
        weights * (fitted - target)
    ) / np.linalg.norm(weights * target)

    print(name)
    print(f"  runtime: {runtime:.2f} s")

    print(f"  normalized L2 error: {normalized_l2_error:.6e}")

    print(f"  weighted normalized L2 error: {weighted_normalized_l2_error:.6e}")

    print(f"  RMS relative error: {np.sqrt(np.mean(relative_error**2)):.6e}")

    print(f"  maximum relative error: {np.max(relative_error):.6e}")

    print()


def plot_comparison(
    frequencies,
    impedance,
    fits,
    logarithmic_frequency,
):
    """Plot data, fits, and errors."""
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(11, 12),
        sharex=True,
    )

    if logarithmic_frequency:

        def plot_curve(
            axis,
            x,
            y,
            **kwargs,
        ):
            axis.semilogx(
                x,
                y,
                **kwargs,
            )

        def plot_error(
            axis,
            x,
            y,
            **kwargs,
        ):
            axis.loglog(
                x,
                y,
                **kwargs,
            )

        scale_name = "logarithmic"

    else:

        def plot_curve(
            axis,
            x,
            y,
            **kwargs,
        ):
            axis.plot(
                x,
                y,
                **kwargs,
            )

        def plot_error(
            axis,
            x,
            y,
            **kwargs,
        ):
            axis.semilogy(
                x,
                y,
                **kwargs,
            )

        scale_name = "linear"

    plot_curve(
        axes[0],
        frequencies,
        impedance.real,
        color="black",
        linewidth=2.5,
        label="BWS data",
    )

    plot_curve(
        axes[1],
        frequencies,
        impedance.imag,
        color="black",
        linewidth=2.5,
        label="BWS data",
    )

    for label, fitted in fits.items():
        plot_curve(
            axes[0],
            frequencies,
            fitted.real,
            label=label,
        )

        plot_curve(
            axes[1],
            frequencies,
            fitted.imag,
            label=label,
        )

        plot_error(
            axes[2],
            frequencies,
            pointwise_relative_error(
                fitted,
                impedance,
            ),
            label=label,
        )

    axes[0].set_ylabel("Real(Z) [Ohm]")

    axes[1].set_ylabel("Imag(Z) [Ohm]")

    axes[2].set_ylabel("Pointwise relative error")

    axes[2].set_xlabel("Frequency [Hz]")

    for axis in axes:
        axis.grid(
            True,
            which="both",
        )
        axis.legend()

    figure.suptitle(f"SmartBounds BWS comparison: {scale_name} frequency axis")

    figure.tight_layout()

    return figure


def select_fit_indices(
    frequencies,
    smart_bounds,
    number_background_samples=100,
    number_samples_per_resonance=30,
    resonance_window_factor=2.0,
):
    """Select background points and additional resonance points.

    The background samples cover the complete frequency range.
    Around every resonance detected by SmartBounds, additional
    frequency samples are retained.
    """
    number_background_samples = min(
        number_background_samples,
        frequencies.size,
    )

    background_indices = np.linspace(
        0,
        frequencies.size - 1,
        number_background_samples,
        dtype=int,
    )

    selected_indices = [background_indices]

    for resonance_index, peak_index in enumerate(smart_bounds.peaks):
        peak_frequency = frequencies[peak_index]

        resonance_width = smart_bounds.upper_lower_bounds[resonance_index]

        lower_frequency = peak_frequency - resonance_window_factor * resonance_width

        upper_frequency = peak_frequency + resonance_window_factor * resonance_width

        indices_inside_window = np.where(
            (frequencies >= lower_frequency) & (frequencies <= upper_frequency)
        )[0]

        if indices_inside_window.size == 0:
            indices_inside_window = np.asarray(
                [peak_index],
                dtype=int,
            )

        if indices_inside_window.size > number_samples_per_resonance:
            local_selection = np.linspace(
                0,
                indices_inside_window.size - 1,
                number_samples_per_resonance,
                dtype=int,
            )

            indices_inside_window = indices_inside_window[local_selection]

        selected_indices.append(indices_inside_window)

        # Ensure that the detected peak itself is included.
        selected_indices.append(
            np.asarray(
                [peak_index],
                dtype=int,
            )
        )

    return np.unique(np.concatenate(selected_indices))


def main():
    data_path = Path(__file__).resolve().parent / "data" / "003_beam_wire_scanner.txt"

    frequencies, impedance = load_impedance(data_path)

    # -------------------------------------------------
    # Determine the same SmartBounds for all fits
    # -------------------------------------------------

    smart_bounds = iddefix.SmartBoundDetermination(
        frequency_data=frequencies,
        impedance_data=impedance.real,
        minimum_peak_height=MINIMUM_PEAK_HEIGHT,
    )

    resonator_parameter_bounds = smart_bounds.parameterBounds

    number_complex_pairs = int(smart_bounds.N_resonators)

    print(f"SmartBounds detected {number_complex_pairs} resonances.")

    print("Detected peak frequencies [Hz]:")

    print(frequencies[smart_bounds.peaks])

    smart_bounds.to_table()

    if USE_REDUCED_FREQUENCY_GRID:
        fit_indices = select_fit_indices(
            frequencies=frequencies,
            smart_bounds=smart_bounds,
            number_background_samples=(NUMBER_BACKGROUND_SAMPLES),
            number_samples_per_resonance=(NUMBER_SAMPLES_PER_RESONANCE),
            resonance_window_factor=(RESONANCE_WINDOW_FACTOR),
        )
    else:
        fit_indices = np.arange(frequencies.size)

    fit_frequencies = frequencies[fit_indices]

    fit_impedance = impedance[fit_indices]

    print(
        "Frequency samples used for fitting: "
        f"{fit_frequencies.size} "
        f"of {frequencies.size}"
    )

    print()
    fits = {}
    runtimes = {}

    # -------------------------------------------------
    # 1. Legacy resonator fit of complex Z
    # -------------------------------------------------

    np.random.seed(RANDOM_SEED)

    complex_resonator_model = iddefix.EvolutionaryAlgorithm(
        x_data=fit_frequencies,
        y_data=fit_impedance,
        N_resonators=number_complex_pairs,
        parameterBounds=(resonator_parameter_bounds),
        plane="longitudinal",
        objectiveFunction=(weighted_complex_error),
    )

    start_time = perf_counter()

    complex_resonator_model.run_differential_evolution(
        maxiter=LEGACY_MAXITER,
        popsize=LEGACY_POPSIZE,
        tol=TOLERANCE,
        mutation=MUTATION,
        crossover_rate=CROSSOVER_RATE,
        solver="scipy",
    )

    if LEGACY_LOCAL_MINIMIZATION:
        complex_resonator_model.run_minimization_algorithm(margin=0.5)

    runtimes["Legacy resonators"] = perf_counter() - start_time

    fits["Legacy resonators"] = complex_resonator_model.get_impedance(
        frequency_data=frequencies,
        use_minimization=(LEGACY_LOCAL_MINIMIZATION),
    )

    # -------------------------------------------------
    # Convert SmartBounds to complex-pole bounds
    # -------------------------------------------------

    complex_pole_bounds = smart_bounds_to_complex_pole_bounds(
        resonator_parameter_bounds
    )

    pole_results = {}

    # -------------------------------------------------
    # 2. Pole-residue fits of complex Z
    # -------------------------------------------------

    for number_real_poles in REAL_POLE_COUNTS:
        label = f"Pole-residue: +{number_real_poles} real poles"

        print(
            f"Fitting {number_complex_pairs} "
            "complex pairs + "
            f"{number_real_poles} real poles..."
        )

        parameter_bounds = (
            real_pole_bounds(
                number_real_poles,
                frequencies,
            )
            + complex_pole_bounds
        )

        legacy_dimension = 3 * number_complex_pairs
        pole_residue_dimension = number_real_poles + 2 * number_complex_pairs

        print(
            "  approximate SciPy populations: "
            f"legacy={LEGACY_POPSIZE * legacy_dimension}, "
            "pole-residue="
            f"{POLE_RESIDUE_POPSIZE * pole_residue_dimension}"
        )

        start_time = perf_counter()

        result = fit_poles_evolutionary(
            frequencies=fit_frequencies,
            impedance=fit_impedance,
            number_real_poles=(number_real_poles),
            number_complex_pairs=(number_complex_pairs),
            parameter_bounds=(parameter_bounds),
            fit_direct_term=(number_real_poles > 0),
            enforce_zero_dc=True,
            amplitude_weighting=("sqrt_relative"),
            frequency_weighting=("linear"),
            residue_solver="least_squares",
            maxiter=POLE_RESIDUE_MAXITER,
            popsize=POLE_RESIDUE_POPSIZE,
            mutation=MUTATION,
            crossover_rate=CROSSOVER_RATE,
            tol=TOLERANCE,
            polish=POLE_RESIDUE_POLISH,
            seed=(RANDOM_SEED + number_real_poles),
            workers=1,
        )

        runtimes[label] = perf_counter() - start_time

        pole_results[number_real_poles] = result

        fits[label] = PoleResidue.impedance(
            frequencies=frequencies,
            poles=result.residue_fit.poles,
            residues=(result.residue_fit.residues),
            direct_term=(result.residue_fit.direct_term),
        )

    # -------------------------------------------------
    # Print metrics
    # -------------------------------------------------

    print()

    for label, fitted in fits.items():
        print_metrics(
            name=label,
            frequencies=frequencies,
            fitted=fitted,
            target=impedance,
            runtime=runtimes[label],
        )

        if label.startswith("Pole-residue"):
            number_real_poles = int(label.split("+")[1].split()[0])

            result = pole_results[number_real_poles]

            print(f"  real poles: {result.real_poles}")

            print(f"  complex poles: {result.complex_poles}")

            print(f"  direct term: {result.residue_fit.direct_term:.6e}")

            print()

    # -------------------------------------------------
    # Plot logarithmic and linear frequency axes
    # -------------------------------------------------

    plot_comparison(
        frequencies=frequencies,
        impedance=impedance,
        fits=fits,
        logarithmic_frequency=True,
    )

    plot_comparison(
        frequencies=frequencies,
        impedance=impedance,
        fits=fits,
        logarithmic_frequency=False,
    )

    plt.show()


if __name__ == "__main__":
    main()

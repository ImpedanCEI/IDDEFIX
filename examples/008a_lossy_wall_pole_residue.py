"""Compare evolutionary models for a longitudinal lossy-wall impedance.

The benchmark compares the original resonator model, a pole-residue model
fitted entirely by Differential Evolution, and a pole-residue model whose
residues are eliminated by linear least squares.
"""

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

import iddefix
from iddefix.poleResidueFitting import (
    build_fit_weights,
    fit_poles_evolutionary,
)

# User settings
PIPE_RADIUS = 20.0e-3
PIPE_LENGTH = 1.0
CONDUCTIVITY = 5.8e7

MINIMUM_FREQUENCY = 1.0e4
MAXIMUM_FREQUENCY = 1.0e9
NUMBER_FREQUENCY_POINTS = 300

# N real poles are compared with N/2 resonators.
NUMBERS_OF_DYNAMIC_POLES = [2, 4, 6]

MAXIMUM_ITERATIONS = 300
POPULATION_SIZE = 12
TOLERANCE = 1.0e-7
MUTATION = (0.1, 0.5)
CROSSOVER_RATE = 0.8

# The original IDDEFIX SciPy solver currently uses workers=-1 internally.
# Use the same setting for both pole-residue variants.
WORKERS = -1

AMPLITUDE_WEIGHTING = "relative"
FREQUENCY_WEIGHTING = "log"


def longitudinal_lossy_wall_impedance(
    frequencies,
    pipe_radius,
    pipe_length,
    conductivity,
):
    """Return the thick-wall impedance of a circular beam pipe."""
    mu_0 = 4.0e-7 * np.pi
    s = 2j * np.pi * frequencies
    coefficient = (
        pipe_length / (2.0 * np.pi * pipe_radius) * np.sqrt(mu_0 / conductivity)
    )
    return coefficient * np.sqrt(s)


def build_real_pole_bounds(
    number_poles,
    minimum_frequency,
    maximum_frequency,
):
    """Build non-overlapping logarithmic bounds for real poles."""
    minimum_rate = 2.0 * np.pi * minimum_frequency / 100.0
    maximum_rate = 2.0 * np.pi * maximum_frequency * 100.0
    edges = np.linspace(
        np.log10(minimum_rate),
        np.log10(maximum_rate),
        number_poles + 1,
    )
    return [(edges[index], edges[index + 1]) for index in range(number_poles)]


def build_real_residue_bounds(
    pole_bounds,
    maximum_impedance,
):
    """Construct broad signed residue bounds from pole-rate bounds."""
    residue_bounds = []
    for _, upper_log_rate in pole_bounds:
        upper_rate = 10.0**upper_log_rate
        residue_scale = 20.0 * maximum_impedance * upper_rate
        residue_bounds.append((-residue_scale, residue_scale))
    return residue_bounds


def build_resonator_bounds(
    number_resonators,
    minimum_frequency,
    maximum_frequency,
    maximum_impedance,
):
    """Build R, Q and resonance-frequency bounds."""
    frequency_edges = np.logspace(
        np.log10(minimum_frequency / 100.0),
        np.log10(maximum_frequency * 100.0),
        number_resonators + 1,
    )
    bounds = []
    for index in range(number_resonators):
        bounds.extend(
            [
                (0.0, 10.0 * maximum_impedance),
                (1.0e-3, 0.5),
                (frequency_edges[index], frequency_edges[index + 1]),
            ]
        )
    return bounds


def weighted_resonator_error(parameters, fitFunction, x, y):
    """Use the same weighted complex squared error as pole-residue."""
    predicted = fitFunction(x, parameters)
    weights = build_fit_weights(
        frequencies=x,
        impedance=y,
        amplitude_weighting=AMPLITUDE_WEIGHTING,
        frequency_weighting=FREQUENCY_WEIGHTING,
    )
    normalization = np.sum(np.abs(weights * y) ** 2)
    return float(np.sum(np.abs(weights * (predicted - y)) ** 2) / normalization)


def calculate_metrics(target_impedance, fitted_impedance, weights):
    """Return identical error measures for all methods."""
    error = fitted_impedance - target_impedance
    relative_error = np.abs(error) / np.maximum(
        np.abs(target_impedance),
        np.finfo(float).tiny,
    )
    return {
        "relative_error": relative_error,
        "normalized_l2_error": float(
            np.linalg.norm(error) / np.linalg.norm(target_impedance)
        ),
        "weighted_normalized_l2_error": float(
            np.sqrt(
                np.sum(np.abs(weights * error) ** 2)
                / np.sum(np.abs(weights * target_impedance) ** 2)
            )
        ),
        "rms_relative_error": float(np.sqrt(np.mean(relative_error**2))),
        "maximum_relative_error": float(np.max(relative_error)),
    }


def print_result(label, runtime, metrics):
    print(label)
    print(f"  runtime: {runtime:.2f} s")
    print(f"  normalized L2 error: {metrics['normalized_l2_error']:.6e}")
    print(
        f"  weighted normalized L2 error: {metrics['weighted_normalized_l2_error']:.6e}"
    )
    print(f"  RMS relative error: {metrics['rms_relative_error']:.6e}")
    print(f"  maximum relative error: {metrics['maximum_relative_error']:.6e}")


def fit_original_resonators(
    frequencies,
    target_impedance,
    number_resonators,
    seed,
):
    """Fit the original resonator model entirely with DE."""
    np.random.seed(seed)
    model = iddefix.EvolutionaryAlgorithm(
        x_data=frequencies,
        y_data=target_impedance,
        N_resonators=number_resonators,
        parameterBounds=build_resonator_bounds(
            number_resonators=number_resonators,
            minimum_frequency=MINIMUM_FREQUENCY,
            maximum_frequency=MAXIMUM_FREQUENCY,
            maximum_impedance=np.max(np.abs(target_impedance)),
        ),
        plane="longitudinal",
        objectiveFunction=weighted_resonator_error,
    )
    start = perf_counter()
    model.run_differential_evolution(
        maxiter=MAXIMUM_ITERATIONS,
        popsize=POPULATION_SIZE,
        mutation=MUTATION,
        crossover_rate=CROSSOVER_RATE,
        tol=TOLERANCE,
        solver="scipy",
    )
    runtime = perf_counter() - start
    fitted_impedance = model.get_impedance(
        frequency_data=frequencies,
        use_minimization=False,
    )
    return fitted_impedance, runtime, model.evolutionParameters


def fit_pole_residue(
    frequencies,
    target_impedance,
    number_poles,
    residue_solver,
    seed,
):
    """Fit real poles using either DE or LS for the residues."""
    pole_bounds = build_real_pole_bounds(
        number_poles=number_poles,
        minimum_frequency=MINIMUM_FREQUENCY,
        maximum_frequency=MAXIMUM_FREQUENCY,
    )
    extra_arguments = {}
    if residue_solver == "differential_evolution":
        extra_arguments["residue_bounds"] = build_real_residue_bounds(
            pole_bounds=pole_bounds,
            maximum_impedance=np.max(np.abs(target_impedance)),
        )
        # With enforce_zero_dc=True, d is derived rather than optimized.
        extra_arguments["direct_term_bounds"] = (
            -10.0 * np.max(np.abs(target_impedance)),
            10.0 * np.max(np.abs(target_impedance)),
        )

    start = perf_counter()
    result = fit_poles_evolutionary(
        frequencies=frequencies,
        impedance=target_impedance,
        number_real_poles=number_poles,
        number_complex_pairs=0,
        parameter_bounds=pole_bounds,
        residue_solver=residue_solver,
        fit_direct_term=True,
        fit_proportional_term=False,
        enforce_zero_dc=True,
        amplitude_weighting=AMPLITUDE_WEIGHTING,
        frequency_weighting=FREQUENCY_WEIGHTING,
        maxiter=MAXIMUM_ITERATIONS,
        popsize=POPULATION_SIZE,
        mutation=MUTATION,
        crossover_rate=CROSSOVER_RATE,
        tol=TOLERANCE,
        polish=False,
        seed=seed,
        workers=WORKERS,
        **extra_arguments,
    )
    runtime = perf_counter() - start
    return result.residue_fit.fitted_impedance, runtime, result


def main():
    frequencies = np.logspace(
        np.log10(MINIMUM_FREQUENCY),
        np.log10(MAXIMUM_FREQUENCY),
        NUMBER_FREQUENCY_POINTS,
    )
    target_impedance = longitudinal_lossy_wall_impedance(
        frequencies=frequencies,
        pipe_radius=PIPE_RADIUS,
        pipe_length=PIPE_LENGTH,
        conductivity=CONDUCTIVITY,
    )
    weights = build_fit_weights(
        frequencies=frequencies,
        impedance=target_impedance,
        amplitude_weighting=AMPLITUDE_WEIGHTING,
        frequency_weighting=FREQUENCY_WEIGHTING,
    )

    methods = {
        "resonators_de": "Original IDDEFIX: resonators, full DE",
        "pole_residue_de": "Pole-residue: poles and residues by DE",
        "pole_residue_ls": "Pole-residue: poles by DE, residues by LS",
    }
    results = {method: {} for method in methods}

    for number_poles in NUMBERS_OF_DYNAMIC_POLES:
        if number_poles % 2:
            raise ValueError("NUMBERS_OF_DYNAMIC_POLES must contain even values")

        number_resonators = number_poles // 2
        print("=" * 78)
        print(f"Dynamic order: {number_poles} poles or {number_resonators} resonators")
        print("=" * 78)

        resonator_fit, runtime, parameters = fit_original_resonators(
            frequencies,
            target_impedance,
            number_resonators,
            seed=1000 + number_poles,
        )
        metrics = calculate_metrics(target_impedance, resonator_fit, weights)
        results["resonators_de"][number_poles] = {
            "fit": resonator_fit,
            "runtime": runtime,
            **metrics,
        }
        print_result(methods["resonators_de"], runtime, metrics)
        print(f"  parameters: {parameters}\n")

        for residue_solver, method_key in [
            ("differential_evolution", "pole_residue_de"),
            ("least_squares", "pole_residue_ls"),
        ]:
            fitted_impedance, runtime, result = fit_pole_residue(
                frequencies,
                target_impedance,
                number_poles,
                residue_solver,
                seed=2000 + number_poles,
            )
            metrics = calculate_metrics(target_impedance, fitted_impedance, weights)
            results[method_key][number_poles] = {
                "fit": fitted_impedance,
                "runtime": runtime,
                **metrics,
            }
            print_result(methods[method_key], runtime, metrics)
            print(f"  real poles: {result.real_poles}")
            print(f"  residues: {result.residue_fit.residues}")
            print(f"  direct term: {result.residue_fit.direct_term:.6e}\n")

    colors = {
        "resonators_de": "tab:orange",
        "pole_residue_de": "tab:green",
        "pole_residue_ls": "tab:blue",
    }
    linestyles = {
        "resonators_de": ":",
        "pole_residue_de": "-.",
        "pole_residue_ls": "--",
    }

    figure, axes = plt.subplots(
        len(NUMBERS_OF_DYNAMIC_POLES),
        2,
        figsize=(14, 4 * len(NUMBERS_OF_DYNAMIC_POLES)),
        sharex=True,
        constrained_layout=True,
    )
    if len(NUMBERS_OF_DYNAMIC_POLES) == 1:
        axes = axes[None, :]

    for row, number_poles in enumerate(NUMBERS_OF_DYNAMIC_POLES):
        impedance_axis, error_axis = axes[row]
        impedance_axis.loglog(
            frequencies,
            np.abs(target_impedance),
            color="black",
            linewidth=2.5,
            label="Lossy-wall target",
        )

        for method_key, method_label in methods.items():
            result = results[method_key][number_poles]
            impedance_axis.loglog(
                frequencies,
                np.abs(result["fit"]),
                color=colors[method_key],
                linestyle=linestyles[method_key],
                label=method_label,
            )
            error_axis.loglog(
                frequencies,
                np.maximum(
                    result["relative_error"],
                    np.finfo(float).eps,
                ),
                color=colors[method_key],
                linestyle=linestyles[method_key],
                label=f"{method_label} ({result['runtime']:.1f} s)",
            )

        impedance_axis.set_title(f"{number_poles} dynamic poles")
        impedance_axis.set_ylabel(r"$|Z_\parallel|$ [$\Omega$]")
        error_axis.set_title(f"Pointwise error, {number_poles} dynamic poles")
        error_axis.set_ylabel("Relative error")
        for axis in (impedance_axis, error_axis):
            axis.grid(True, which="both")
            axis.legend(fontsize=8)

    axes[-1, 0].set_xlabel("Frequency [Hz]")
    axes[-1, 1].set_xlabel("Frequency [Hz]")

    summary_figure, summary_axes = plt.subplots(
        1, 2, figsize=(12, 4.5), constrained_layout=True
    )
    for method_key, method_label in methods.items():
        orders = np.asarray(NUMBERS_OF_DYNAMIC_POLES)
        errors = [
            results[method_key][order]["weighted_normalized_l2_error"]
            for order in orders
        ]
        runtimes = [results[method_key][order]["runtime"] for order in orders]
        summary_axes[0].semilogy(
            orders,
            errors,
            marker="o",
            color=colors[method_key],
            label=method_label,
        )
        summary_axes[1].semilogy(
            orders,
            runtimes,
            marker="o",
            color=colors[method_key],
            label=method_label,
        )

    summary_axes[0].set_xlabel("Number of dynamic poles")
    summary_axes[0].set_ylabel("Weighted normalized L2 error")
    summary_axes[1].set_xlabel("Number of dynamic poles")
    summary_axes[1].set_ylabel("Runtime [s]")
    for axis in summary_axes:
        axis.grid(True, which="both")
        axis.legend(fontsize=8)

    plt.show()


if __name__ == "__main__":
    main()

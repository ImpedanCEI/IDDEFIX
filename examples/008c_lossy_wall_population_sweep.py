"""Population-size sweep for lossy-wall impedance fitting.

The model order and maximum number of generations are fixed. The actual DE
population size is varied for three approaches:

* three resonators, all parameters optimized by DE;
* six real poles and six residues optimized jointly by DE;
* six real poles optimized by DE, with residues fitted by least squares.

Several seeds are evaluated because Differential Evolution is stochastic.
"""

from functools import partial
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

from iddefix.poleResidueFitting import (
    build_fit_weights,
    pole_objective,
    pole_residue_objective,
)
from iddefix.resonatorFormulas import Impedances

# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

PIPE_RADIUS = 20.0e-3
PIPE_LENGTH = 1.0
CONDUCTIVITY = 5.8e7

MINIMUM_FREQUENCY = 1.0e4
MAXIMUM_FREQUENCY = 1.0e9
NUMBER_FREQUENCY_POINTS = 300

NUMBER_REAL_POLES = 6
NUMBER_RESONATORS = NUMBER_REAL_POLES // 2

# All values must be divisible by 6, 9 and 12. Their least common
# multiple is 36.
ACTUAL_POPULATION_SIZES = [36, 72, 144, 288]
MAXIMUM_GENERATIONS = 100
SEEDS = [1234, 2345, 3456]

MUTATION = (0.1, 0.5)
CROSSOVER_RATE = 0.8

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


def build_real_residue_bounds(pole_bounds, maximum_impedance):
    """Construct broad signed residue bounds from pole-rate bounds."""
    bounds = []
    for _, upper_log_rate in pole_bounds:
        upper_rate = 10.0**upper_log_rate
        scale = 20.0 * maximum_impedance * upper_rate
        bounds.append((-scale, scale))
    return bounds


def build_resonator_bounds(
    number_resonators,
    minimum_frequency,
    maximum_frequency,
    maximum_impedance,
):
    """Build Rs, Q and resonance-frequency bounds."""
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


def resonator_objective(parameters, frequencies, impedance, weights):
    """Return the normalized weighted complex squared error."""
    fitted = Impedances.n_Resonator_longitudinal_imp(
        frequencies,
        parameters,
    )
    error = weights * (fitted - impedance)
    normalization = np.sum(np.abs(weights * impedance) ** 2)
    value = np.sum(np.abs(error) ** 2) / normalization
    return float(value) if np.isfinite(value) else np.inf


def run_de(objective, bounds, actual_population_size, seed):
    """Run one fixed-generation DE fit."""
    dimension = len(bounds)
    if actual_population_size % dimension != 0:
        raise ValueError(
            f"Population {actual_population_size} is not divisible "
            f"by optimization dimension {dimension}."
        )

    scipy_popsize = actual_population_size // dimension
    start_time = perf_counter()
    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy="rand1bin",
        maxiter=MAXIMUM_GENERATIONS,
        popsize=scipy_popsize,
        mutation=MUTATION,
        recombination=CROSSOVER_RATE,
        tol=0.0,
        atol=0.0,
        polish=False,
        seed=seed,
        init="latinhypercube",
        workers=1,
        updating="immediate",
    )
    runtime = perf_counter() - start_time

    # The optimizer minimizes the normalized squared L2 error. Report its
    # square root so that the plotted quantity is a normalized L2 error.
    normalized_l2_error = float(np.sqrt(max(result.fun, 0.0)))

    return {
        "error": normalized_l2_error,
        "squared_error": float(result.fun),
        "runtime": runtime,
        "nfev": int(result.nfev),
        "nit": int(result.nit),
        "dimension": dimension,
        "scipy_popsize": scipy_popsize,
    }


def percentile_summary(values):
    """Return median and interquartile range."""
    values = np.asarray(values, dtype=float)
    return (
        float(np.median(values)),
        float(np.percentile(values, 25.0)),
        float(np.percentile(values, 75.0)),
    )


def plot_sweep(axis, results, methods, quantity, ylabel):
    """Plot individual runs, median and interquartile range."""
    population_sizes = np.asarray(ACTUAL_POPULATION_SIZES)

    for method_key, method in methods.items():
        samples = np.asarray(
            [
                [run[quantity] for run in results[method_key][population]]
                for population in population_sizes
            ]
        )
        median = np.median(samples, axis=1)
        lower = np.percentile(samples, 25.0, axis=1)
        upper = np.percentile(samples, 75.0, axis=1)

        for seed_index in range(samples.shape[1]):
            axis.loglog(
                population_sizes,
                samples[:, seed_index],
                color=method["color"],
                marker=".",
                linewidth=0.8,
                alpha=0.25,
            )

        axis.loglog(
            population_sizes,
            median,
            color=method["color"],
            marker="o",
            linewidth=2.0,
            label=method["label"],
        )
        axis.fill_between(
            population_sizes,
            lower,
            upper,
            color=method["color"],
            alpha=0.2,
        )

    axis.set_xlabel("Actual population size")
    axis.set_ylabel(ylabel)
    axis.grid(True, which="both")
    axis.legend(fontsize=8)


def main():
    frequencies = np.logspace(
        np.log10(MINIMUM_FREQUENCY),
        np.log10(MAXIMUM_FREQUENCY),
        NUMBER_FREQUENCY_POINTS,
    )
    impedance = longitudinal_lossy_wall_impedance(
        frequencies,
        PIPE_RADIUS,
        PIPE_LENGTH,
        CONDUCTIVITY,
    )
    weights = build_fit_weights(
        frequencies=frequencies,
        impedance=impedance,
        amplitude_weighting=AMPLITUDE_WEIGHTING,
        frequency_weighting=FREQUENCY_WEIGHTING,
    )
    maximum_impedance = np.max(np.abs(impedance))

    pole_bounds = build_real_pole_bounds(
        NUMBER_REAL_POLES,
        MINIMUM_FREQUENCY,
        MAXIMUM_FREQUENCY,
    )
    residue_bounds = build_real_residue_bounds(
        pole_bounds,
        maximum_impedance,
    )
    direct_term_bounds = (
        -10.0 * maximum_impedance,
        10.0 * maximum_impedance,
    )
    resonator_bounds = build_resonator_bounds(
        NUMBER_RESONATORS,
        MINIMUM_FREQUENCY,
        MAXIMUM_FREQUENCY,
        maximum_impedance,
    )

    methods = {
        "resonators_de": {
            "label": "3 resonators: full DE",
            "color": "tab:orange",
            "bounds": resonator_bounds,
            "objective": partial(
                resonator_objective,
                frequencies=frequencies,
                impedance=impedance,
                weights=weights,
            ),
        },
        "pole_residue_de": {
            "label": "6 real poles: poles and residues by DE",
            "color": "tab:green",
            "bounds": [*pole_bounds, *residue_bounds],
            "objective": partial(
                pole_residue_objective,
                frequencies=frequencies,
                impedance=impedance,
                number_real_poles=NUMBER_REAL_POLES,
                number_complex_pairs=0,
                weights=weights,
                fit_direct_term=True,
                enforce_zero_dc=True,
                direct_term_bounds=direct_term_bounds,
                plane="longitudinal",
            ),
        },
        "pole_residue_ls": {
            "label": "6 real poles: poles by DE, residues by LS",
            "color": "tab:blue",
            "bounds": pole_bounds,
            "objective": partial(
                pole_objective,
                frequencies=frequencies,
                impedance=impedance,
                number_real_poles=NUMBER_REAL_POLES,
                number_complex_pairs=0,
                weights=weights,
                fit_direct_term=True,
                enforce_zero_dc=True,
                plane="longitudinal",
            ),
        },
    }

    results = {
        method_key: {population: [] for population in ACTUAL_POPULATION_SIZES}
        for method_key in methods
    }

    print("Lossy-wall population-size sweep")
    print("=" * 78)
    print(f"Maximum generations: {MAXIMUM_GENERATIONS}")
    print(f"Population sizes: {ACTUAL_POPULATION_SIZES}")
    print(f"Seeds: {SEEDS}")

    for method_key, method in methods.items():
        print("\n" + method["label"])
        print(f"  optimization dimension: {len(method['bounds'])}")

        for population_size in ACTUAL_POPULATION_SIZES:
            print(f"  population: {population_size}")

            for seed in SEEDS:
                run = run_de(
                    objective=method["objective"],
                    bounds=method["bounds"],
                    actual_population_size=population_size,
                    seed=seed,
                )
                results[method_key][population_size].append(run)
                print(
                    f"    seed {seed}: error={run['error']:.6e}, "
                    f"runtime={run['runtime']:.2f} s, "
                    f"nfev={run['nfev']}"
                )

            errors = [run["error"] for run in results[method_key][population_size]]
            runtimes = [run["runtime"] for run in results[method_key][population_size]]
            error_median, error_q25, error_q75 = percentile_summary(errors)
            runtime_median, runtime_q25, runtime_q75 = percentile_summary(runtimes)
            print(
                f"    error median [Q25, Q75]: {error_median:.6e} "
                f"[{error_q25:.6e}, {error_q75:.6e}]"
            )
            print(
                f"    runtime median [Q25, Q75]: {runtime_median:.2f} "
                f"[{runtime_q25:.2f}, {runtime_q75:.2f}] s"
            )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        constrained_layout=True,
    )
    plot_sweep(
        axes[0],
        results,
        methods,
        quantity="error",
        ylabel="Final weighted normalized L2 error",
    )
    plot_sweep(
        axes[1],
        results,
        methods,
        quantity="runtime",
        ylabel="Runtime [s]",
    )
    axes[0].set_title("Fit accuracy versus population size")
    axes[1].set_title("Runtime versus population size")

    plt.show()


if __name__ == "__main__":
    main()

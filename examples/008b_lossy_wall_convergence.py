"""Convergence comparison for three lossy-wall fitting approaches.

The model order is fixed to six dynamic poles:

* three resonators, all parameters optimized by DE;
* six real poles and six residues optimized jointly by DE;
* six real poles optimized by DE, with residues fitted by least squares.

All methods use the same actual DE population size. Several random seeds are
used because Differential Evolution is stochastic.
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

# This is the intended total population, not SciPy's popsize multiplier.
# 144 is divisible by 6, 9 and 12, the dimensions of the three problems.
ACTUAL_POPULATION_SIZE = 144
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


def make_history_callback(start_time):
    """Record the best population member after every generation."""
    history = {
        "objective": [],
        "time": [],
    }

    # The parameter name ``intermediate_result`` makes recent SciPy versions
    # pass an OptimizeResult containing both x and fun.
    def callback(intermediate_result):
        history["objective"].append(float(intermediate_result.fun))
        history["time"].append(perf_counter() - start_time)
        return False

    return history, callback


def run_de(objective, bounds, seed):
    """Run DE with a prescribed total population and record convergence."""
    dimension = len(bounds)

    if ACTUAL_POPULATION_SIZE % dimension != 0:
        raise ValueError(
            "ACTUAL_POPULATION_SIZE must be divisible by every optimization dimension"
        )

    scipy_popsize = ACTUAL_POPULATION_SIZE // dimension
    start_time = perf_counter()
    history, callback = make_history_callback(start_time)

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
        callback=callback,
    )

    history["objective"] = np.asarray(history["objective"])
    history["time"] = np.asarray(history["time"])
    history["runtime"] = perf_counter() - start_time
    history["nfev"] = int(result.nfev)
    history["nit"] = int(result.nit)
    history["final_objective"] = float(result.fun)
    history["dimension"] = dimension
    history["scipy_popsize"] = scipy_popsize

    return result, history


def pad_history(values):
    """Pad an early-terminated history with its final value."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.full(MAXIMUM_GENERATIONS, np.nan)
    if values.size < MAXIMUM_GENERATIONS:
        values = np.pad(
            values,
            (0, MAXIMUM_GENERATIONS - values.size),
            mode="edge",
        )
    return values[:MAXIMUM_GENERATIONS]


def plot_median_band(axis, x, histories, color, label):
    """Plot median and interquartile range over the random seeds."""
    matrix = np.vstack([pad_history(history["objective"]) for history in histories])
    median = np.nanmedian(matrix, axis=0)
    lower = np.nanpercentile(matrix, 25.0, axis=0)
    upper = np.nanpercentile(matrix, 75.0, axis=0)
    axis.semilogy(x, median, color=color, linewidth=2.0, label=label)
    axis.fill_between(x, lower, upper, color=color, alpha=0.2)


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

    all_histories = {key: [] for key in methods}

    print("Lossy-wall convergence benchmark")
    print("=" * 78)
    print(f"Maximum generations: {MAXIMUM_GENERATIONS}")
    print(f"Actual population per method: {ACTUAL_POPULATION_SIZE}")
    print(f"Seeds: {SEEDS}")

    for method_key, method in methods.items():
        dimension = len(method["bounds"])
        print("\n" + method["label"])
        print(f"  optimization dimension: {dimension}")
        print(f"  scipy popsize multiplier: {ACTUAL_POPULATION_SIZE // dimension}")

        for seed in SEEDS:
            _, history = run_de(
                objective=method["objective"],
                bounds=method["bounds"],
                seed=seed,
            )
            all_histories[method_key].append(history)
            print(
                f"  seed {seed}: error={history['final_objective']:.6e}, "
                f"runtime={history['runtime']:.2f} s, "
                f"nfev={history['nfev']}"
            )

        final_errors = np.array(
            [h["final_objective"] for h in all_histories[method_key]]
        )
        runtimes = np.array([h["runtime"] for h in all_histories[method_key]])
        print(f"  median error: {np.median(final_errors):.6e}")
        print(f"  median runtime: {np.median(runtimes):.2f} s")

    generations = np.arange(1, MAXIMUM_GENERATIONS + 1)
    approximate_evaluations = (generations + 1) * ACTUAL_POPULATION_SIZE

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5),
        constrained_layout=True,
    )

    for method_key, method in methods.items():
        histories = all_histories[method_key]
        plot_median_band(
            axes[0],
            generations,
            histories,
            method["color"],
            method["label"],
        )
        plot_median_band(
            axes[1],
            approximate_evaluations,
            histories,
            method["color"],
            method["label"],
        )

        for index, history in enumerate(histories):
            axes[2].semilogy(
                history["time"],
                history["objective"],
                color=method["color"],
                alpha=0.35,
                label=method["label"] if index == 0 else None,
            )

    axes[0].set_xlabel("Generation")
    axes[1].set_xlabel("Approximate objective evaluations")
    axes[2].set_xlabel("Runtime [s]")
    axes[0].set_ylabel("Best weighted normalized squared error")
    axes[1].set_ylabel("Best weighted normalized squared error")
    axes[2].set_ylabel("Best weighted normalized squared error")
    axes[0].set_title("Convergence versus generation")
    axes[1].set_title("Convergence versus evaluation budget")
    axes[2].set_title("Convergence versus wall time")

    for axis in axes:
        axis.grid(True, which="both")
        axis.legend(fontsize=8)

    plt.show()


if __name__ == "__main__":
    main()

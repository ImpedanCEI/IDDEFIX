"""Measure the effort required to reach prescribed lossy-wall fit errors.

This example reuses the model definitions from
``008b_lossy_wall_convergence.py``. It compares generations, objective
evaluations and wall time required to reach several weighted normalized
L2-error targets.
"""

import runpy
from functools import partial
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

from iddefix.poleResidueFitting import (
    build_fit_weights,
    pole_objective,
    pole_residue_objective,
)

# Load shared analytical model and bound-building helpers without executing
# the main program of example 008b.
shared = runpy.run_path(Path(__file__).with_name("008b_lossy_wall_convergence.py"))

longitudinal_lossy_wall_impedance = shared["longitudinal_lossy_wall_impedance"]
build_real_pole_bounds = shared["build_real_pole_bounds"]
build_real_residue_bounds = shared["build_real_residue_bounds"]
build_resonator_bounds = shared["build_resonator_bounds"]
resonator_objective = shared["resonator_objective"]


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

ACTUAL_POPULATION_SIZE = 144
MAXIMUM_GENERATIONS = 1000
SEEDS = [1234, 2345, 3456]

# These are weighted normalized L2 errors, not squared errors.
TARGET_ERRORS = [1.0, 0.5, 0.1, 0.05]

MUTATION = (0.1, 0.5)
CROSSOVER_RATE = 0.8
AMPLITUDE_WEIGHTING = "relative"
FREQUENCY_WEIGHTING = "log"


def run_until_target(objective, bounds, seed):
    """Run DE until the strictest target or the maximum budget is reached."""
    dimension = len(bounds)
    if ACTUAL_POPULATION_SIZE % dimension != 0:
        raise ValueError(
            "ACTUAL_POPULATION_SIZE must be divisible by every optimization dimension"
        )

    scipy_popsize = ACTUAL_POPULATION_SIZE // dimension
    strictest_target = min(TARGET_ERRORS)
    start_time = perf_counter()
    history = {
        "error": [],
        "time": [],
        "generation": [],
        "evaluations": [],
    }

    def callback(intermediate_result):
        generation = len(history["generation"]) + 1
        error = float(np.sqrt(max(intermediate_result.fun, 0.0)))

        history["error"].append(error)
        history["time"].append(perf_counter() - start_time)
        history["generation"].append(generation)

        # Initial population plus one new population per generation.
        history["evaluations"].append((generation + 1) * ACTUAL_POPULATION_SIZE)

        return error <= strictest_target

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

    for key in ("error", "time", "generation", "evaluations"):
        history[key] = np.asarray(history[key])

    history["runtime"] = perf_counter() - start_time
    history["nfev"] = int(result.nfev)
    history["final_error"] = float(np.sqrt(max(result.fun, 0.0)))
    history["dimension"] = dimension

    return history


def first_target_crossing(history, target):
    """Return effort at the first generation satisfying an error target."""
    indices = np.flatnonzero(history["error"] <= target)
    if indices.size == 0:
        return None

    index = int(indices[0])
    return {
        "generation": float(history["generation"][index]),
        "evaluations": float(history["evaluations"][index]),
        "time": float(history["time"][index]),
    }


def collect_target_statistics(histories, target, quantity):
    """Return successful values and the success fraction."""
    values = []
    for history in histories:
        crossing = first_target_crossing(history, target)
        if crossing is not None:
            values.append(crossing[quantity])

    return np.asarray(values), len(values) / len(histories)


def plot_effort(axis, all_histories, methods, quantity, ylabel):
    """Plot median target-reaching effort and its interquartile range."""
    targets = np.asarray(TARGET_ERRORS)

    for method_key, method in methods.items():
        medians = []
        lower = []
        upper = []

        for target in targets:
            values, _ = collect_target_statistics(
                all_histories[method_key], target, quantity
            )
            if values.size:
                medians.append(np.median(values))
                lower.append(np.percentile(values, 25.0))
                upper.append(np.percentile(values, 75.0))
            else:
                medians.append(np.nan)
                lower.append(np.nan)
                upper.append(np.nan)

        medians = np.asarray(medians)
        lower = np.asarray(lower)
        upper = np.asarray(upper)

        axis.loglog(
            targets,
            medians,
            marker="o",
            linewidth=2.0,
            color=method["color"],
            label=method["label"],
        )
        axis.fill_between(
            targets,
            lower,
            upper,
            color=method["color"],
            alpha=0.2,
        )

    axis.set_xlabel("Target weighted normalized L2 error")
    axis.set_ylabel(ylabel)
    axis.invert_xaxis()
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

    all_histories = {key: [] for key in methods}

    print("Lossy-wall time-to-target benchmark")
    print("=" * 78)
    print(f"Population: {ACTUAL_POPULATION_SIZE}")
    print(f"Maximum generations: {MAXIMUM_GENERATIONS}")
    print(f"Target errors: {TARGET_ERRORS}")
    print(f"Seeds: {SEEDS}")

    for method_key, method in methods.items():
        print("\n" + method["label"])

        for seed in SEEDS:
            history = run_until_target(
                method["objective"],
                method["bounds"],
                seed,
            )
            all_histories[method_key].append(history)
            print(
                f"  seed {seed}: final error={history['final_error']:.6e}, "
                f"runtime={history['runtime']:.2f} s, "
                f"nfev={history['nfev']}"
            )

        for target in TARGET_ERRORS:
            parts = []
            for quantity in ("generation", "evaluations", "time"):
                values, success = collect_target_statistics(
                    all_histories[method_key], target, quantity
                )
                if values.size:
                    parts.append(f"{quantity}={np.median(values):.3g}")
                else:
                    parts.append(f"{quantity}=not reached")

            _, success = collect_target_statistics(
                all_histories[method_key], target, "generation"
            )
            print(f"  target {target:.3g}: success={success:.0%}, " + ", ".join(parts))

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5),
        constrained_layout=True,
    )
    plot_effort(
        axes[0],
        all_histories,
        methods,
        quantity="generation",
        ylabel="Generations to target",
    )
    plot_effort(
        axes[1],
        all_histories,
        methods,
        quantity="evaluations",
        ylabel="Approximate evaluations to target",
    )
    plot_effort(
        axes[2],
        all_histories,
        methods,
        quantity="time",
        ylabel="Wall time to target [s]",
    )
    axes[0].set_title("Generation effort")
    axes[1].set_title("Evaluation effort")
    axes[2].set_title("Runtime effort")

    plt.show()


if __name__ == "__main__":
    main()

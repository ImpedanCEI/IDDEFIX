"""Iterative resonator fitting with weighted residuals and scaled parameters."""

from __future__ import annotations

from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from scipy.optimize import minimize

from .objectiveFunctions import ObjectiveFunctions as obj
from .resonatorFormulas import Impedances as imp

if TYPE_CHECKING:
    from .framework import EvolutionaryAlgorithm

ParameterBounds = list[tuple[float, float]]


def run_iterative_minimization(
    self: EvolutionaryAlgorithm,
    N_resonators: int | None = None,
    max_resonators: int = 20,
    new_resonator_bounds: ParameterBounds | None = None,
    initial_q: float = 1.0,
    rms_target: float = 0.01,
    rms_floor: float = 1e-6,
    method: str = "Nelder-Mead",
    maxiter: int | None = None,
    tol: float | None = None,
    normalize_parameters: bool = True,
    verbose: bool = False,
    options: dict[str, Any] | None = None,
    margin: float | Sequence[float] | None = None,
) -> dict[int, "EvolutionaryAlgorithm"]:
    """Grow a fit by adding modes at the largest weighted residual.

    - Preserve estimates/fits or start with log-spaced frequencies and
      ``initial_q``. Added modes use ``new_resonator_bounds`` or defaults.
    - Optimize and report the worst component's relative RMS. Complex
      residuals share max(abs(data), rms_floor * peak) as their denominator;
      real/magnitude fits use the corresponding amplitude.
    - ``normalize_parameters`` uses asinh coordinates around each seed,
      supporting signed/zero Rs and wide frequency ranges. ``margin``
      optionally limits existing modes; None keeps full bounds.
    - Nelder-Mead defaults: xatol=1e-3, fatol=1e-8 (squared RMS units).
      ``tol`` overrides these; advanced ``options`` take precedence.
      ``maxiter`` defaults to 5000 per mode, without a separate evaluation
      cap for Nelder-Mead/L-BFGS-B. ``rms_target`` is the fitting target.
    - ``best_trials`` stores the best fit per count; ``trial_history``
      retains every attempt. Reruns refresh scores and grow from the best
      accepted fit. Stop at rms_target or max_resonators.

    The configured objective selects components (custom objectives use
    the input type). Its raw loss remains ``objective_value``; the solver
    minimizes relative RMS squared. Standalone minimization is unchanged.
    This is a local, growth-only search, not a global model-order optimum.
    """
    # Select components and use the same residual scaling throughout.
    x, y = self._fit_data
    magnitude = self.objectiveFunction in (
        obj.sumOfSquaredErrorAbs,
        obj.logsumOfSquaredErrorAbs,
    )
    real = self.objectiveFunction in (
        obj.sumOfSquaredErrorReal,
        obj.logsumOfSquaredErrorReal,
    )
    components = {"Magnitude": np.abs} if magnitude else {"Real": np.real}
    complex_objective = self.objectiveFunction in (
        obj.sumOfSquaredError,
        obj.logsumOfSquaredError,
    )
    if not magnitude and not real and (complex_objective or np.iscomplexobj(y)):
        components["Imaginary"] = np.imag

    def residuals(predicted, data):
        amplitude = (
            np.abs(data)
            if len(components) > 1
            else np.abs(next(iter(components.values()))(data))
        )
        scale = np.maximum(amplitude, rms_floor * (np.max(amplitude) or 1.0))
        return {
            name: (part(predicted) - part(data)) / scale
            for name, part in components.items()
        }

    def loss(parameters):
        errors = residuals(trial.fitFunction(x, parameters), y)
        return (
            max(float(np.mean(r**2)) for r in errors.values())
            if all(np.isfinite(r).all() for r in errors.values())
            else np.inf
        )

    def score(model):
        errors = residuals(model.fitFunction(x, model.minimizationParameters), y)
        model.rms_components = {
            name: float(np.sqrt(np.mean(r**2))) for name, r in errors.items()
        }
        model.rms_error = max(model.rms_components.values())
        model.real_rms_error = model.rms_components.get("Real", np.nan)
        model.imag_rms_error = model.rms_components.get("Imaginary", np.nan)
        return model.rms_error

    # Prepare the starting count, bounds, and log-spaced estimates.
    frequency, impedance = self._impedance_for_bounds()
    order = np.argsort(frequency)
    frequency, impedance = frequency[order], impedance[order]
    trial = copy(self)
    count = N_resonators if N_resonators is not None else self.N_resonators or 1
    changed_count = count != self.N_resonators
    trial.N_resonators = count
    trial.parameterBounds = (
        trial._default_parameterBounds()
        if self.parameterBounds is None
        else list(self.parameterBounds)
    )
    if changed_count:
        trial.parameterBounds = trial.parameterBounds[:3] * count
    estimates = None
    if not changed_count:
        for estimates in (
            self.parameterEstimates,
            self.minimizationParameters,
            self.evolutionParameters,
        ):
            if estimates is not None:
                break
    if estimates is None:
        lower, upper = np.asarray(trial.parameterBounds[:3]).T
        frequencies = np.geomspace(lower[2], upper[2], count + 2)[1:-1]
        estimates = np.column_stack(
            (
                np.interp(frequencies, frequency, np.abs(impedance)),
                np.full(count, initial_q),
                frequencies,
            )
        ).ravel()
    parameters = np.asarray(estimates).copy()
    added_bounds = (
        self._default_parameterBounds(N_resonators=1)
        if new_resonator_bounds is None
        else new_resonator_bounds
    )
    trial.best_trials, trial.trial_history, trial._smart_bounds = {}, {}, None
    trial.compute_uncertainties = False
    trials, history = self.best_trials, self.trial_history
    for saved in (*history.values(), *trials.values()):
        score(saved)
    best = min(trials.values(), key=lambda t: t.rms_error) if trials else None
    fitted = (
        self.minimizationParameters is not None or self.evolutionParameters is not None
    )
    existing_count = count if fitted and not changed_count else 0

    # Fit all parameters, optionally limiting movement of existing modes.
    for count in range(count, max_resonators + 1):
        bounds = np.asarray(trial.parameterBounds, dtype=float).copy()
        if margin is not None and existing_count:
            delta = np.tile(np.broadcast_to(margin, (3,)), existing_count) * np.abs(
                parameters[: 3 * existing_count]
            )
            bounds[: 3 * existing_count, 0] = np.maximum(
                bounds[: 3 * existing_count, 0],
                parameters[: 3 * existing_count] - delta,
            )
            bounds[: 3 * existing_count, 1] = np.minimum(
                bounds[: 3 * existing_count, 1],
                parameters[: 3 * existing_count] + delta,
            )
        parameters = np.clip(parameters, *bounds.T)

        # Scale around the seed, not the bound span; asinh also handles signed/zero Rs.
        reference = np.tile(
            [np.max(np.abs(impedance)) or 1.0, 1.0, np.max(frequency)], count
        )
        scale = (
            np.where(parameters != 0, np.abs(parameters), reference)
            if normalize_parameters
            else np.ones_like(parameters)
        )
        transform = np.arcsinh if normalize_parameters else np.asarray
        inverse = np.sinh if normalize_parameters else np.asarray
        encode = lambda p: transform(p / scale)
        decode = lambda p: scale * inverse(p)
        lower, upper = encode(bounds.T)
        x0 = encode(parameters)
        settings = {"maxiter": maxiter if maxiter is not None else 5000 * count}
        if method.lower() == "nelder-mead":
            settings.update(
                maxfev=np.inf,
                adaptive=True,
                xatol=tol if tol is not None else 1e-3,
                fatol=tol if tol is not None else 1e-8,
            )
            simplex = np.tile(x0, (len(x0) + 1, 1))
            step = (
                np.full_like(x0, 0.1)
                if normalize_parameters
                else np.maximum(np.abs(x0) * 0.05, 0.00025)
            )
            simplex[1:] += np.diag(
                np.where(x0 + step <= upper, step, -np.minimum(step, x0 - lower))
            )
            settings["initial_simplex"] = simplex
        elif method.lower() == "l-bfgs-b":
            settings.update(
                maxfun=np.inf,
                ftol=tol if tol is not None else 1e-9,
                gtol=tol if tol is not None else 1e-6,
            )
        settings.update(options or {})
        print(f"\nTrial: {count} resonators")
        result = minimize(
            lambda p: loss(decode(p)),
            x0=x0,
            bounds=list(zip(lower, upper)),
            method=method,
            tol=tol,
            options=settings,
        )
        result.x = decode(result.x)
        if "final_simplex" in result:
            vertices, values = result.final_simplex
            result.final_simplex = (decode(vertices), values)

        # Sort physical parameters and their bounds together; keep solver results usable.
        indices = (
            3 * np.argsort(result.x.reshape(-1, 3)[:, 2])[:, None] + np.arange(3)
        ).ravel()
        result.x = result.x[indices]
        if "final_simplex" in result:
            vertices, values = result.final_simplex
            result.final_simplex = (vertices[:, indices], values)
        trial.parameterBounds = [trial.parameterBounds[i] for i in indices]
        trial.minimizationBounds = bounds[indices].tolist()
        trial.minimizationParameters = trial.parameterEstimates = result.x.copy()
        trial.minimization_result = result
        trial.evolutionParameters = trial.evolutionParametersUncertainties = None
        trial.minimizationParametersUncertainties = None
        trial.objective_value = float(
            trial.objectiveFunction(result.x, fitFunction=trial.fitFunction, x=x, y=y)
        )
        score(trial)

        # Save every attempt and accept only improvements under the current RMS metric.
        snapshot = deepcopy(trial)
        history[max(history, default=0) + 1] = snapshot
        if count not in trials or snapshot.rms_error < trials[count].rms_error:
            trials[count] = snapshot
        if best is None or snapshot.rms_error < best.rms_error:
            best = snapshot
        print(
            " | ".join(
                f"{name} RMS: {value:.2%}"
                for name, value in trial.rms_components.items()
            )
        )
        print(
            f"Solver: {result.message} ({result.nit} iterations, {result.nfev} evaluations)"
        )
        if verbose:
            trial.display_resonator_parameters()
        if best.rms_error <= rms_target or count == max_resonators:
            print(
                "Stopping: RMS target reached."
                if best.rms_error <= rms_target
                else "Stopping: maximum resonator count reached."
            )
            break

        # Grow from the best accepted fit at the largest weighted spectral residual.
        trial = deepcopy(best)
        existing_count = trial.N_resonators
        while trial.N_resonators <= count:
            formula = getattr(imp, f"n_Resonator_{self.plane}_imp")
            predicted = formula(
                frequency, trial.minimizationParameters, self.wake_length
            )
            errors = residuals(predicted, impedance)
            weights = np.max([np.abs(r) for r in errors.values()], axis=0)
            eligible = (
                (frequency > 0)
                & (frequency >= added_bounds[2][0])
                & (frequency <= added_bounds[2][1])
            )
            index = np.argmax(np.where(eligible, weights, -np.inf))
            fr = frequency[index]
            rs = np.abs(impedance[index] - predicted[index])
            if added_bounds[0][0] < 0:
                rs *= np.sign((impedance[index] - predicted[index]).real)
            seed = np.clip([rs, initial_q, fr], *np.asarray(added_bounds).T)
            trial.add_resonator(*seed, parameter_bounds=added_bounds)
        parameters = trial.minimizationParameters.copy()

    # Restore the selected physical model and preserve persistent histories.
    for name in (
        "N_resonators",
        "parameterBounds",
        "parameterEstimates",
        "minimizationParameters",
        "minimizationBounds",
        "minimization_result",
        "objective_value",
        "rms_error",
        "rms_components",
        "real_rms_error",
        "imag_rms_error",
    ):
        setattr(self, name, deepcopy(getattr(best, name)))
    self.evolutionParameters = self.evolutionParametersUncertainties = None
    self.minimizationParametersUncertainties = self._smart_bounds = None
    self.best_trials, self.trial_history = trials, history
    self._update_parameter_flags()
    if self.compute_uncertainties:
        self.get_uncertainties(parameterBounds=self.minimizationBounds)
    print(f"Selected: {self.N_resonators} resonators, RMS: {self.rms_error:.2%}.")
    return trials

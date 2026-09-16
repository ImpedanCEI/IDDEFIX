"""Evolutionary pole fitting with optional linear residue elimination."""

from dataclasses import dataclass
from functools import partial

import numpy as np
import numpy.typing as npt
from scipy.linalg import lstsq
from scipy.optimize import differential_evolution

from .poleResidueFormulas import (
    SPEED_OF_LIGHT,
    impedance_plane_factor,
)

ArrayLike = npt.ArrayLike


def _scaled_lstsq(
    matrix: np.ndarray,
    right_hand_side: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Solve least squares after normalizing the matrix columns."""
    column_scales = np.linalg.norm(matrix, axis=0)
    column_scales = np.maximum(
        column_scales,
        np.finfo(float).tiny,
    )

    scaled_matrix = matrix / column_scales[None, :]

    scaled_coefficients, _, rank, _ = lstsq(
        scaled_matrix,
        right_hand_side,
        lapack_driver="gelsy",
        check_finite=False,
    )

    return scaled_coefficients / column_scales, int(rank)


@dataclass
class ResidueFitResult:
    """Residues and fitted response for a set of poles."""

    poles: np.ndarray
    residues: np.ndarray
    direct_term: float
    fitted_impedance: np.ndarray
    squared_error: float
    weighted_squared_error: float
    rank: int
    proportional_term: float = 0.0


@dataclass
class PoleOptimizationResult:
    """Result of the evolutionary pole optimization."""

    pole_parameters: np.ndarray
    real_poles: np.ndarray
    complex_poles: np.ndarray
    residue_fit: ResidueFitResult
    objective_value: float
    success: bool
    message: str


def _number_independent_residue_parameters(
    number_real_poles: int,
    number_complex_pairs: int,
) -> int:
    """Return the number of real parameters describing all residues."""
    return number_real_poles + 2 * number_complex_pairs


def decode_residue_parameters(
    parameters: ArrayLike,
    number_real_poles: int,
    number_complex_pairs: int,
) -> np.ndarray:
    """Restore real/conjugate residues from independent real parameters.

    The parameter order is

    ``[real residues, Re(complex residues), Im(complex residues)]``,

    with real and imaginary parts interleaved for each complex pair.
    """
    parameters = np.asarray(parameters, dtype=float)

    expected_size = _number_independent_residue_parameters(
        number_real_poles,
        number_complex_pairs,
    )

    if parameters.size != expected_size:
        raise ValueError(
            f"expected {expected_size} residue parameters, received {parameters.size}"
        )

    real_residues = parameters[:number_real_poles]
    complex_parameters = parameters[number_real_poles:]

    complex_residues = complex_parameters[0::2] + 1j * complex_parameters[1::2]

    return np.concatenate(
        [
            real_residues.astype(complex),
            complex_residues,
            np.conj(complex_residues),
        ]
    )


def decode_log_poles(
    parameters: ArrayLike,
    number_real_poles: int,
    number_complex_pairs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert logarithmic parameters into stable poles.

    The parameter order is

    [log10(a_real),
     log10(alpha_complex),
     log10(beta_complex)],

    where

    p_real = -a

    and

    p_complex = -alpha + 1j*beta.

    All rates are expressed in rad/s.
    """
    parameters = np.asarray(parameters, dtype=float)

    expected_size = number_real_poles + 2 * number_complex_pairs

    if parameters.size != expected_size:
        raise ValueError(
            f"expected {expected_size} pole parameters, received {parameters.size}"
        )

    real_stop = number_real_poles
    decay_stop = real_stop + number_complex_pairs

    real_rates = 10.0 ** parameters[:real_stop]

    complex_decay_rates = 10.0 ** parameters[real_stop:decay_stop]

    complex_frequencies = 10.0 ** parameters[decay_stop:]

    real_poles = -np.sort(real_rates).astype(complex)

    complex_poles = -complex_decay_rates + 1j * complex_frequencies

    complex_poles = complex_poles[np.argsort(complex_poles.imag)]

    return real_poles, complex_poles


def build_fit_weights(
    frequencies: ArrayLike,
    impedance: ArrayLike,
    amplitude_weighting: str = "uniform",
    frequency_weighting: str = "samples",
    magnitude_floor: float | None = None,
) -> np.ndarray:
    """Construct amplitude and frequency-grid fit weights."""
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))

    impedance = np.atleast_1d(np.asarray(impedance, dtype=complex))

    if frequencies.size != impedance.size:
        raise ValueError("frequencies and impedance must have the same length")

    if frequencies.size == 0:
        raise ValueError("input data must not be empty")

    if not np.all(np.isfinite(frequencies)):
        raise ValueError("frequencies must be finite")

    if not np.all(np.isfinite(impedance)):
        raise ValueError("impedance must be finite")

    # Amplitude weighting
    magnitude = np.abs(impedance)

    if magnitude_floor is None:
        maximum_magnitude = np.max(magnitude)
        magnitude_floor = max(
            maximum_magnitude * 1.0e-12,
            np.finfo(float).tiny,
        )

    if magnitude_floor <= 0.0:
        raise ValueError("magnitude_floor must be positive")

    safe_magnitude = np.maximum(
        magnitude,
        magnitude_floor,
    )

    if amplitude_weighting == "uniform":
        amplitude_weights = np.ones_like(frequencies)

    elif amplitude_weighting == "relative":
        amplitude_weights = 1.0 / safe_magnitude

    elif amplitude_weighting == "sqrt_relative":
        amplitude_weights = 1.0 / np.sqrt(safe_magnitude)

    else:
        raise ValueError(
            "amplitude_weighting must be 'uniform', 'relative', or 'sqrt_relative'"
        )

    # Frequency-grid weighting
    if frequency_weighting == "samples":
        frequency_weights = np.ones_like(frequencies)

    else:
        if frequencies.size < 2:
            raise ValueError("frequency-grid weighting requires at least two points")

        if frequency_weighting == "linear":
            coordinate = frequencies

        elif frequency_weighting == "log":
            if np.any(frequencies <= 0.0):
                raise ValueError(
                    "logarithmic frequency weighting requires "
                    "strictly positive frequencies"
                )

            coordinate = np.log(frequencies)

        else:
            raise ValueError(
                "frequency_weighting must be 'samples', 'linear', or 'log'"
            )

        order = np.argsort(coordinate)
        sorted_coordinate = coordinate[order]

        differences = np.diff(sorted_coordinate)

        if np.any(differences <= 0.0):
            raise ValueError("frequencies must not contain duplicates")

        quadrature_weights = np.empty_like(sorted_coordinate)

        quadrature_weights[0] = differences[0] / 2.0
        quadrature_weights[-1] = differences[-1] / 2.0

        if frequencies.size > 2:
            quadrature_weights[1:-1] = (
                sorted_coordinate[2:] - sorted_coordinate[:-2]
            ) / 2.0

        # Remove irrelevant overall scaling.
        quadrature_weights /= np.mean(quadrature_weights)

        sorted_frequency_weights = np.sqrt(quadrature_weights)

        frequency_weights = np.empty_like(sorted_frequency_weights)

        frequency_weights[order] = sorted_frequency_weights

    weights = amplitude_weights * frequency_weights

    # Improve numerical scaling without changing relative weights.
    weights /= np.sqrt(np.mean(weights**2))

    return weights


def _pole_basis(
    frequencies: np.ndarray,
    poles: np.ndarray,
    wake_length: float | None,
    plane: str = "longitudinal",
) -> np.ndarray:
    """Construct the impedance basis for individual poles."""
    plane_factor = impedance_plane_factor(plane)

    s = 2j * np.pi * frequencies

    denominator = s[:, None] - poles[None, :]

    if wake_length is None:
        return plane_factor / denominator

    duration = wake_length / SPEED_OF_LIGHT

    return plane_factor * (-np.expm1(-denominator * duration) / denominator)


def fit_residues(
    frequencies: ArrayLike,
    impedance: ArrayLike,
    real_poles: ArrayLike,
    complex_poles: ArrayLike,
    wake_length: float | None = None,
    weights: ArrayLike | None = None,
    fit_direct_term: bool = False,
    fit_proportional_term: bool = False,
    enforce_zero_dc: bool = False,
    plane: str = "longitudinal",
) -> ResidueFitResult:
    """Determine optimal residues for fixed poles.

    ``complex_poles`` contains only the poles in the upper half-plane.
    Their complex conjugates are added automatically.
    """
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
    impedance = np.atleast_1d(np.asarray(impedance, dtype=complex))
    real_poles = np.atleast_1d(np.asarray(real_poles, dtype=complex))
    complex_poles = np.atleast_1d(np.asarray(complex_poles, dtype=complex))

    if frequencies.size != impedance.size:
        raise ValueError("frequencies and impedance must have the same length")

    if weights is None:
        weights = np.ones_like(frequencies)
    else:
        weights = np.atleast_1d(np.asarray(weights, dtype=float))

        if weights.size != frequencies.size:
            raise ValueError("weights and frequencies must have the same length")

        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must be finite")

        if np.any(weights <= 0.0):
            raise ValueError("weights must be positive")

    if np.any(np.abs(real_poles.imag) > 1.0e-12):
        raise ValueError("real_poles must be real")

    if np.any(complex_poles.imag <= 0.0):
        raise ValueError("complex_poles must lie in the upper half-plane")

    independent_poles = np.concatenate([real_poles.real, complex_poles])

    if independent_poles.size == 0:
        raise ValueError("at least one pole must be provided")

    if np.any(independent_poles.real >= 0.0):
        raise ValueError("all poles must be stable")

    columns = []

    if real_poles.size:
        real_basis = _pole_basis(
            frequencies,
            real_poles,
            wake_length,
            plane,
        )

        columns.extend(real_basis[:, index] for index in range(real_poles.size))

    if complex_poles.size:
        positive_basis = _pole_basis(
            frequencies,
            complex_poles,
            wake_length,
            plane,
        )

        negative_basis = _pole_basis(
            frequencies,
            np.conj(complex_poles),
            wake_length,
            plane,
        )

        for index in range(complex_poles.size):
            phi_positive = positive_basis[:, index]
            phi_negative = negative_basis[:, index]

            # Coefficient multiplying Re(residue)
            columns.append(phi_positive + phi_negative)

            # Coefficient multiplying Im(residue)
            columns.append(1j * (phi_positive - phi_negative))

    if fit_direct_term:
        columns.append(
            impedance_plane_factor(plane)
            * np.ones(
                frequencies.size,
                dtype=complex,
            )
        )

    if fit_proportional_term:
        s = 2j * np.pi * frequencies
        columns.append(impedance_plane_factor(plane) * s)

    design_matrix = np.column_stack(columns)

    real_system_matrix = np.vstack([design_matrix.real, design_matrix.imag])

    real_right_hand_side = np.concatenate([impedance.real, impedance.imag])

    stacked_weights = np.concatenate([weights, weights])

    weighted_system_matrix = stacked_weights[:, None] * real_system_matrix

    weighted_right_hand_side = stacked_weights * real_right_hand_side

    if enforce_zero_dc:
        dc_columns = []

        if real_poles.size:
            dc_columns.extend(-1.0 / real_poles.real)

        for pole in complex_poles:
            phi_positive = -1.0 / pole
            phi_negative = -1.0 / np.conj(pole)

            dc_columns.append((phi_positive + phi_negative).real)

            dc_columns.append((1j * (phi_positive - phi_negative)).real)

        if fit_direct_term:
            dc_columns.append(1.0)

        if fit_proportional_term:
            # The proportional term h*s vanishes at s=0.
            dc_columns.append(0.0)

        dc_constraint = np.asarray(
            dc_columns,
            dtype=float,
        )

        # Eliminate the coefficient with the largest constraint
        # coefficient. This avoids division by a small value.
        pivot_index = int(np.argmax(np.abs(dc_constraint)))

        pivot_value = dc_constraint[pivot_index]

        free_indices = np.arange(dc_constraint.size) != pivot_index

        free_constraint = dc_constraint[free_indices]

        pivot_column = weighted_system_matrix[
            :,
            pivot_index,
        ]

        free_matrix = weighted_system_matrix[
            :,
            free_indices,
        ]

        # Substitute
        #
        # x_pivot = -(c_free @ x_free) / c_pivot
        #
        # into A @ x.
        reduced_system_matrix = free_matrix - np.outer(
            pivot_column,
            free_constraint / pivot_value,
        )

        reduced_coefficients, rank = _scaled_lstsq(
            reduced_system_matrix,
            weighted_right_hand_side,
        )

        coefficients = np.empty(
            dc_constraint.size,
            dtype=float,
        )

        coefficients[free_indices] = reduced_coefficients

        coefficients[pivot_index] = (
            -np.dot(
                free_constraint,
                reduced_coefficients,
            )
            / pivot_value
        )

    else:
        coefficients, rank = _scaled_lstsq(
            weighted_system_matrix,
            weighted_right_hand_side,
        )

    number_real = real_poles.size

    fitted_real_residues = coefficients[:number_real]

    fitted_complex_residues = []

    offset = number_real

    for index in range(complex_poles.size):
        real_part = coefficients[offset + 2 * index]
        imaginary_part = coefficients[offset + 2 * index + 1]

        fitted_complex_residues.append(real_part + 1j * imaginary_part)

    fitted_complex_residues = np.asarray(
        fitted_complex_residues,
        dtype=complex,
    )

    coefficient_index = number_real + 2 * complex_poles.size

    if fit_direct_term:
        direct_term = float(coefficients[coefficient_index])
        coefficient_index += 1
    else:
        direct_term = 0.0

    if fit_proportional_term:
        proportional_term = float(coefficients[coefficient_index])
    else:
        proportional_term = 0.0

    full_poles = np.concatenate(
        [
            real_poles,
            complex_poles,
            np.conj(complex_poles),
        ]
    )

    full_residues = np.concatenate(
        [
            fitted_real_residues,
            fitted_complex_residues,
            np.conj(fitted_complex_residues),
        ]
    )

    full_basis = _pole_basis(
        frequencies,
        full_poles,
        wake_length,
        plane,
    )
    fitted_impedance = (
        impedance_plane_factor(plane)
        * (direct_term + proportional_term * (2j * np.pi * frequencies))
        + full_basis @ full_residues
    )

    squared_error = float(np.sum(np.abs(impedance - fitted_impedance) ** 2))

    weighted_squared_error = float(
        np.sum(np.abs(weights * (impedance - fitted_impedance)) ** 2)
    )

    return ResidueFitResult(
        poles=full_poles,
        residues=full_residues,
        direct_term=direct_term,
        fitted_impedance=fitted_impedance,
        squared_error=squared_error,
        weighted_squared_error=weighted_squared_error,
        rank=int(rank),
        proportional_term=proportional_term,
    )


def evaluate_residue_parameters(
    frequencies: ArrayLike,
    impedance: ArrayLike,
    real_poles: ArrayLike,
    complex_poles: ArrayLike,
    residue_parameters: ArrayLike,
    wake_length: float | None = None,
    weights: ArrayLike | None = None,
    direct_term: float = 0.0,
    proportional_term: float = 0.0,
    enforce_zero_dc: bool = False,
    plane: str = "longitudinal",
) -> ResidueFitResult:
    """Evaluate explicitly supplied residue parameters.

    This is the counterpart of :func:`fit_residues` for the fully
    evolutionary formulation. No least-squares problem is solved.
    """
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
    impedance = np.atleast_1d(np.asarray(impedance, dtype=complex))
    real_poles = np.atleast_1d(np.asarray(real_poles, dtype=complex))
    complex_poles = np.atleast_1d(np.asarray(complex_poles, dtype=complex))

    if frequencies.size != impedance.size:
        raise ValueError("frequencies and impedance must have the same length")

    if weights is None:
        weights = np.ones(frequencies.size, dtype=float)
    else:
        weights = np.atleast_1d(np.asarray(weights, dtype=float))

        if weights.size != frequencies.size:
            raise ValueError("weights and frequencies must have the same length")

    full_poles = np.concatenate(
        [
            real_poles,
            complex_poles,
            np.conj(complex_poles),
        ]
    )

    full_residues = decode_residue_parameters(
        residue_parameters,
        number_real_poles=real_poles.size,
        number_complex_pairs=complex_poles.size,
    )

    if enforce_zero_dc:
        # Z(0) = factor * (d + sum(r_k / -p_k)) = 0.
        # The direct term is therefore not an independent DE parameter.
        direct_term = float(np.real(np.sum(full_residues / full_poles)))

    full_basis = _pole_basis(
        frequencies,
        full_poles,
        wake_length,
        plane,
    )

    fitted_impedance = (
        impedance_plane_factor(plane)
        * (direct_term + proportional_term * (2j * np.pi * frequencies))
        + full_basis @ full_residues
    )

    error = impedance - fitted_impedance

    squared_error = float(np.sum(np.abs(error) ** 2))

    weighted_squared_error = float(np.sum(np.abs(weights * error) ** 2))

    return ResidueFitResult(
        poles=full_poles,
        residues=full_residues,
        direct_term=float(direct_term),
        fitted_impedance=fitted_impedance,
        squared_error=squared_error,
        weighted_squared_error=weighted_squared_error,
        rank=-1,
        proportional_term=float(proportional_term),
    )


def pole_residue_objective(
    parameters: ArrayLike,
    frequencies: ArrayLike,
    impedance: ArrayLike,
    number_real_poles: int,
    number_complex_pairs: int,
    wake_length: float | None = None,
    weights: ArrayLike | None = None,
    fit_direct_term: bool = False,
    fit_proportional_term: bool = False,
    enforce_zero_dc: bool = False,
    direct_term_bounds: tuple[float, float] | None = None,
    plane: str = "longitudinal",
) -> float:
    """Evaluate a candidate containing poles and residues."""
    parameters = np.asarray(parameters, dtype=float)
    impedance = np.atleast_1d(np.asarray(impedance, dtype=complex))

    number_pole_parameters = number_real_poles + 2 * number_complex_pairs
    number_residue_parameters = _number_independent_residue_parameters(
        number_real_poles,
        number_complex_pairs,
    )

    pole_parameters = parameters[:number_pole_parameters]
    residue_stop = number_pole_parameters + number_residue_parameters
    residue_parameters = parameters[number_pole_parameters:residue_stop]

    if fit_direct_term and not enforce_zero_dc:
        direct_term = float(parameters[residue_stop])
        coefficient_index = residue_stop + 1
    else:
        direct_term = 0.0
        coefficient_index = residue_stop

    if fit_proportional_term:
        proportional_term = float(parameters[coefficient_index])
    else:
        proportional_term = 0.0

    real_poles, complex_poles = decode_log_poles(
        pole_parameters,
        number_real_poles,
        number_complex_pairs,
    )

    try:
        result = evaluate_residue_parameters(
            frequencies=frequencies,
            impedance=impedance,
            real_poles=real_poles,
            complex_poles=complex_poles,
            residue_parameters=residue_parameters,
            wake_length=wake_length,
            weights=weights,
            direct_term=direct_term,
            proportional_term=proportional_term,
            enforce_zero_dc=enforce_zero_dc,
            plane=plane,
        )
    except (ValueError, np.linalg.LinAlgError):
        return np.inf

    if (
        enforce_zero_dc
        and direct_term_bounds is not None
        and not (direct_term_bounds[0] <= result.direct_term <= direct_term_bounds[1])
    ):
        return np.inf

    if weights is None:
        weights_array = np.ones_like(
            impedance,
            dtype=float,
        )
    else:
        weights_array = np.asarray(weights, dtype=float)

    normalization = np.sum(np.abs(weights_array * impedance) ** 2)

    if normalization == 0.0:
        normalization = 1.0

    normalized_error = result.weighted_squared_error / normalization

    if not np.isfinite(normalized_error):
        return np.inf

    return float(normalized_error)


def pole_objective(
    parameters: ArrayLike,
    frequencies: ArrayLike,
    impedance: ArrayLike,
    number_real_poles: int,
    number_complex_pairs: int,
    wake_length: float | None = None,
    weights: ArrayLike | None = None,
    fit_direct_term: bool = False,
    fit_proportional_term: bool = False,
    enforce_zero_dc: bool = False,
    plane: str = "longitudinal",
) -> float:
    """Evaluate the normalized fitting error for candidate poles."""
    impedance = np.atleast_1d(np.asarray(impedance, dtype=complex))

    real_poles, complex_poles = decode_log_poles(
        parameters,
        number_real_poles,
        number_complex_pairs,
    )

    try:
        result = fit_residues(
            frequencies=frequencies,
            impedance=impedance,
            real_poles=real_poles,
            complex_poles=complex_poles,
            wake_length=wake_length,
            weights=weights,
            fit_direct_term=fit_direct_term,
            fit_proportional_term=fit_proportional_term,
            enforce_zero_dc=enforce_zero_dc,
            plane=plane,
        )
    except (ValueError, np.linalg.LinAlgError):
        return np.inf

    if weights is None:
        weights_array = np.ones_like(
            impedance,
            dtype=float,
        )
    else:
        weights_array = np.asarray(
            weights,
            dtype=float,
        )

    normalization = np.sum(np.abs(weights_array * impedance) ** 2)

    if normalization == 0.0:
        normalization = 1.0

    normalized_error = result.weighted_squared_error / normalization

    if not np.isfinite(normalized_error):
        return np.inf

    return float(normalized_error)


def fit_poles_evolutionary(
    frequencies: ArrayLike,
    impedance: ArrayLike,
    number_real_poles: int,
    number_complex_pairs: int,
    parameter_bounds: list[tuple[float, float]],
    wake_length: float | None = None,
    maxiter: int = 1000,
    popsize: int = 15,
    mutation: float | tuple[float, float] = (0.1, 0.5),
    crossover_rate: float = 0.8,
    tol: float = 0.01,
    polish: bool = False,
    seed: int | None = None,
    workers: int = 1,
    amplitude_weighting: str = "uniform",
    frequency_weighting: str = "samples",
    magnitude_floor: float | None = None,
    fit_direct_term: bool = False,
    fit_proportional_term: bool = False,
    enforce_zero_dc: bool = False,
    plane: str = "longitudinal",
    residue_solver: str = "least_squares",
    residue_bounds: list[tuple[float, float]] | None = None,
    direct_term_bounds: tuple[float, float] | None = None,
    proportional_term_bounds: tuple[float, float] | None = None,
) -> PoleOptimizationResult:
    """Fit a pole-residue model using Differential Evolution.

    With ``residue_solver='least_squares'``, Differential Evolution
    optimizes logarithmic pole parameters and the residues are eliminated
    by linear least squares for every pole candidate.

    With ``residue_solver='differential_evolution'``, poles and independent
    residue components are optimized jointly. The independent residue order
    is ``[real residues, Re(r_complex), Im(r_complex)]``, with the real and
    imaginary components interleaved for each complex-conjugate pair.

    Parameters
    ----------
    frequencies
        Frequencies in Hz.
    impedance
        Complex impedance data.
    number_real_poles
        Number of independent negative real poles.
    number_complex_pairs
        Number of complex-conjugate pole pairs.
    parameter_bounds
        Bounds for the logarithmic pole parameters. The order is

        [log10(real decay rates),
         log10(complex decay rates),
         log10(complex angular frequencies)].

        All rates are expressed in rad/s.
    wake_length
        Simulated wake length in metres. If None, fully decayed
        impedance data are assumed.
    residue_solver
        ``'least_squares'`` (default) or ``'differential_evolution'``.
    residue_bounds
        Bounds for the independent real residue parameters. Required for
        ``residue_solver='differential_evolution'``.
    direct_term_bounds
        Bounds for the direct term. Required when the direct term is fitted
        by Differential Evolution. With ``enforce_zero_dc=True``, the direct
        term is instead computed from the residues and these bounds are only
        used as an admissibility constraint when supplied.
    proportional_term_bounds
        Bounds for the proportional term. Required when it is fitted
        by Differential Evolution.
    """
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))

    impedance = np.atleast_1d(np.asarray(impedance, dtype=complex))

    weights = build_fit_weights(
        frequencies=frequencies,
        impedance=impedance,
        amplitude_weighting=amplitude_weighting,
        frequency_weighting=frequency_weighting,
        magnitude_floor=magnitude_floor,
    )
    expected_number_pole_bounds = number_real_poles + 2 * number_complex_pairs

    if len(parameter_bounds) != expected_number_pole_bounds:
        raise ValueError(
            f"expected {expected_number_pole_bounds} parameter bounds, "
            f"received {len(parameter_bounds)}"
        )

    normalized_residue_solver = residue_solver.lower()

    if normalized_residue_solver == "least_squares":
        if residue_bounds is not None:
            raise ValueError(
                "residue_bounds are only used with "
                "residue_solver='differential_evolution'"
            )

        objective_function = partial(
            pole_objective,
            frequencies=frequencies,
            impedance=impedance,
            number_real_poles=number_real_poles,
            number_complex_pairs=number_complex_pairs,
            wake_length=wake_length,
            weights=weights,
            fit_direct_term=fit_direct_term,
            fit_proportional_term=fit_proportional_term,
            enforce_zero_dc=enforce_zero_dc,
            plane=plane,
        )

        optimization_bounds = parameter_bounds

    elif normalized_residue_solver == "differential_evolution":
        expected_number_residue_bounds = _number_independent_residue_parameters(
            number_real_poles,
            number_complex_pairs,
        )

        if residue_bounds is None:
            raise ValueError(
                "residue_bounds are required with "
                "residue_solver='differential_evolution'"
            )

        if len(residue_bounds) != expected_number_residue_bounds:
            raise ValueError(
                f"expected {expected_number_residue_bounds} residue "
                f"bounds, received {len(residue_bounds)}"
            )

        if enforce_zero_dc and not fit_direct_term:
            raise ValueError(
                "The fully evolutionary solver currently requires "
                "fit_direct_term=True when enforce_zero_dc=True. "
                "The direct term is then determined by the DC constraint."
            )

        optimization_bounds = [
            *parameter_bounds,
            *residue_bounds,
        ]

        if fit_direct_term and not enforce_zero_dc:
            if direct_term_bounds is None:
                raise ValueError(
                    "direct_term_bounds are required when the direct "
                    "term is optimized by Differential Evolution"
                )

            optimization_bounds.append(direct_term_bounds)

        if fit_proportional_term:
            if proportional_term_bounds is None:
                raise ValueError(
                    "proportional_term_bounds are required when the "
                    "proportional term is optimized by Differential Evolution"
                )

            optimization_bounds.append(proportional_term_bounds)

        objective_function = partial(
            pole_residue_objective,
            frequencies=frequencies,
            impedance=impedance,
            number_real_poles=number_real_poles,
            number_complex_pairs=number_complex_pairs,
            wake_length=wake_length,
            weights=weights,
            fit_direct_term=fit_direct_term,
            fit_proportional_term=fit_proportional_term,
            enforce_zero_dc=enforce_zero_dc,
            direct_term_bounds=direct_term_bounds,
            plane=plane,
        )

    else:
        raise ValueError(
            "residue_solver must be 'least_squares' or 'differential_evolution'"
        )

    updating = "immediate" if workers == 1 else "deferred"

    optimization = differential_evolution(
        objective_function,
        bounds=optimization_bounds,
        strategy="rand1bin",
        maxiter=maxiter,
        popsize=popsize,
        mutation=mutation,
        recombination=crossover_rate,
        tol=tol,
        polish=polish,
        seed=seed,
        init="latinhypercube",
        workers=workers,
        updating=updating,
    )

    pole_parameters = optimization.x[:expected_number_pole_bounds]

    real_poles, complex_poles = decode_log_poles(
        pole_parameters,
        number_real_poles,
        number_complex_pairs,
    )

    if normalized_residue_solver == "least_squares":
        residue_fit = fit_residues(
            frequencies=frequencies,
            impedance=impedance,
            real_poles=real_poles,
            complex_poles=complex_poles,
            wake_length=wake_length,
            weights=weights,
            fit_direct_term=fit_direct_term,
            fit_proportional_term=fit_proportional_term,
            enforce_zero_dc=enforce_zero_dc,
            plane=plane,
        )
    else:
        residue_start = expected_number_pole_bounds
        residue_stop = residue_start + _number_independent_residue_parameters(
            number_real_poles,
            number_complex_pairs,
        )

        if fit_direct_term and not enforce_zero_dc:
            direct_term = float(optimization.x[residue_stop])
            coefficient_index = residue_stop + 1
        else:
            direct_term = 0.0
            coefficient_index = residue_stop

        if fit_proportional_term:
            proportional_term = float(optimization.x[coefficient_index])
        else:
            proportional_term = 0.0

        residue_fit = evaluate_residue_parameters(
            frequencies=frequencies,
            impedance=impedance,
            real_poles=real_poles,
            complex_poles=complex_poles,
            residue_parameters=optimization.x[residue_start:residue_stop],
            wake_length=wake_length,
            weights=weights,
            direct_term=direct_term,
            proportional_term=proportional_term,
            enforce_zero_dc=enforce_zero_dc,
            plane=plane,
        )

    return PoleOptimizationResult(
        pole_parameters=pole_parameters,
        real_poles=real_poles,
        complex_poles=complex_poles,
        residue_fit=residue_fit,
        objective_value=float(optimization.fun),
        success=bool(optimization.success),
        message=str(optimization.message),
    )

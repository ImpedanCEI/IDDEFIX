# Joint resonator fitting

`EvolutionaryAlgorithm` accepts any nonempty subset of these named data pairs:

- `impedance_x_data`, `impedance_y_data`: frequency in Hz and impedance in Ohm (longitudinal) or Ohm/m (transverse).
- `wake_potential_x_data`, `wake_potential_y_data`: time in seconds and Gaussian wake potential in V/C or V/C/m.
- `wake_x_data`, `wake_y_data`: time in seconds and point wake in V/C or V/C/m.

Every supplied pair must be complete, finite, nonempty, and have matching one-dimensional arrays. Sources may have different grids and lengths. All sources share the same flattened resonator vector `[Rs, Q, fr, ...]`, bounds, and initial estimates. Wake-potential data require an explicit positive `sigma` in seconds. The model uses the existing corrected wake-potential formulas for underdamped, critically damped, and overdamped resonators; see [the theory](theory.md).

```python
model = iddefix.EvolutionaryAlgorithm(
    impedance_x_data=frequency,
    impedance_y_data=impedance,
    wake_potential_x_data=time,
    wake_potential_y_data=wake_potential,
    sigma=50e-12,  # Example assumption; use the measured bunch length.
    plane="transverse",
    parameterBounds=bounds,
    parameterEstimates=initial_parameters,
)
model.run_differential_evolution()
model.run_minimization_algorithm()
```

Omit unavailable pairs. The constructor arguments start with impedance, wake potential, then wake data pairs. `x_data` and `y_data` have been removed from the constructor; replace them with the appropriate source pair. This is a breaking API change: the first six positional arguments now refer to the three data pairs. Use keywords for fitting options such as `N_resonators` and `parameterBounds`.

The default objective normalizes every active source, including a single supplied pair. With exactly one source, you may explicitly select an existing `objectiveFunction` (`"Real"`, `"Complex"`, `"Abs"`, or a callable); these retain their original unnormalized behavior. Multiple sources require the default normalized joint objective. Real-only impedance measurements constrain only the real model component; complex measurements constrain both components. Wake and wake-potential measurements must be real. The primary model domain is inferred from the supplied data pairs rather than `fitFunction`.

## Normalization and uncertainty

For each active source $s$, the measured-data scale is

$$
a_s=\sqrt{\frac{1}{n_s}\sum_i |y_{s,i}|^2}.
$$

An identically zero measured trace uses $a_s=1$ in its supplied units. Such a trace has no nonzero data-derived scale; this fallback avoids division by zero but cannot provide unit invariance for that source.

The residual components are $(y_s-\hat y_s)/(a_s\sqrt{n_s})$. Complex impedance stacks the real and imaginary components. The loss is

$$
L=\sum_s\frac{1}{n_s a_s^2}\sum_i |y_{s,i}-\hat y_{s,i}|^2.
$$

Every source has fixed weight 1. Thus units, measured magnitude, and sample count alone do not increase a source's influence. There is currently no configurable weighting. Normalization balances relative errors; it does not model measurement noise or correlations.

All DE backends, CMA-ES, and local minimization use the squared norm of this same residual vector. Uncertainty estimation differentiates that vector and uses the existing SVD covariance calculation, scaled by $L/(M-N)$, where $M$ counts real residual components and $N$ counts parameters. These are local approximate uncertainties for the normalized loss, not calibrated measurement-error confidence intervals. More real residuals than parameters are required. Joint uncertainty recomputation uses the stored named pairs; alternate `x_data`/`y_data` overrides are rejected.

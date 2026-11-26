# Fitting with EvolutionaryAlgorithm

Use `EvolutionaryAlgorithm` (EA) to keep the data, bounds, solver results,
and plots together. The usual workflow is:

1. Create the model from impedance, wake, or wake-potential data
2. Remove a baseline if needed, then call `model.estimate_bounds()`
3. Fit with `model.run_cmaes()` or `model.run_differential_evolution()`
4. Inspect the result with `model.plot()` or `model.plot_resonator_components()`

EA uses SmartBounds internally for peak detection. You do not need to
construct a `SmartBoundDetermination` object for this workflow.

## Recommended workflow

This runnable example fits a synthetic longitudinal impedance. Replace the
first two arrays with your measured or simulated data when using the model:

```python
import numpy as np
import matplotlib.pyplot as plt
import iddefix

# Generate the impedance or load it
frequency = np.linspace(0.1e9, 3e9, 1001)  # Hz
impedance = iddefix.Impedances.Resonator_longitudinal_imp(
    frequency, Rs=100, Q=8, fr=1e9,
)

# Prepare the model
model = iddefix.EvolutionaryAlgorithm(
    frequency, impedance, plane="longitudinal",
)
# For measured finite-wake data, set wake_length before optional baseline removal
# model.remove_baseline()
bounds = model.estimate_bounds(prominence=0.01, show_table=False, show_inspect=False)
bounds.inspect()
bounds.to_table()

# Fit the model, then refine it locally
model.downsample_data(samples=100, spacing="linear")
model.run_cmaes(maxiter=200, popsize=20)
model.run_minimization_algorithm()

# Inspect the fit in both domains
fig, axes = model.plot_resonator_components(function="impedance", type="abs")
time = np.linspace(-0.1e-9, 20e-9, 1001)  # Seconds from the bunch center
fig, axes = model.plot(time, function="wake potential", sigma=10e-12)

```

`estimate_bounds()` sets the resonator count, bound pairs, and starting
estimates on the model. By default, it also displays the estimated components
and a table; the example disables these displays. The estimator is available
as `model.bounds` (`model.smart_bounds` is an alias).

Local refinement with `model.run_minimization_algorithm()` is optional.
Both global solvers store their results in `evolutionParameters`; local
refinement stores its result in `minimizationParameters`. Evaluation and
plotting prefer local parameters when available. Pass `use_minimization=False`
to inspect the evolutionary fit instead.

`x_data` and `y_data` retain the original inputs. Baseline correction and
downsampling update the domain arrays used for fitting, such as
`impedance_data` or `wake_potential_data`.

## Input domains and units

The constructor's `fitFunction` selects what data you are fitting:

| `fitFunction` | x values | Longitudinal y values | Transverse y values |
| --- | --- | --- | --- |
| `"impedance"` | Frequency [Hz] | Ohm | Ohm/m |
| `"wake"` | Time [s] | V/C | V/C/m |
| `"wake potential"` | Time [s] | V/C | V/C/m |

Use a space in `"wake potential"`. Transverse wakes must already have their
required displacement normalization. Convert exports in V/pC to V/C before
fitting. For a dipolar wake, divide by the driving offset in metres if the
export has not already applied this normalization.

For a Gaussian wake potential, also pass `sigma`, the RMS bunch duration in
seconds. If the bunch length is given in metres, use `sigma = sigma_z / c`.
The time axis is measured from the bunch center and must cover its Gaussian
profile sufficiently on both sides. FFT/deconvolution requires uniform time
sampling.

```python
model = iddefix.EvolutionaryAlgorithm(
    time,
    wake_potential,
    plane="transverse",
    fitFunction="wake potential",
    sigma=sigma,
    wake_length=L,  # Recorded wake length in metres, if finite
)
bounds = model.estimate_bounds()
model.run_cmaes()
```

For time-domain inputs, `estimate_bounds()` transforms the data internally
for peak detection. Fitting still uses the original domain; there is no need
to calculate an FFT or deconvolution separately to construct bounds.

## Peak detection and sampling

`prominence` is an absolute threshold in the detection trace's units, not a
percentage. Pass `prominence=None` to disable that filter. The default trace
is the real impedance, except for the built-in absolute objective, which uses
magnitude. An explicit `impedance_type` overrides this choice.

`fmax` and `fft_samples` control the temporary frequency grid for time-domain
inputs. For direct impedance, `samples=None` retains the existing grid; an
explicit count linearly interpolates the peak-analysis trace. These options
do not change the fitting data or recover features absent from the input.

Use `model.downsample_data(factor=2, show=True)` to reduce the actual fitting
samples, or select an approximate count with `samples=1000`. This selects
original samples without interpolation. Check that narrow resonances remain
resolved and that time grids remain uniform before using FFT/deconvolution.

If no peaks are detected, inspect the data units, time origin, frequency band,
and prominence. A monotonic trace or an endpoint maximum may not identify a
resonance. If you already know the model order, you can supply a resonator
count and use the default bounds or your own bounds instead.

## Choosing bounds

Peak-based estimates are the recommended starting point. Their settings can
be passed directly to `model.estimate_bounds()`, including `Rs_bounds`,
`Q_bounds`, `fres_bounds`, and `q_side`. See the
[estimation formulas](theory.md#peak-based-parameter-estimation) for how peaks
and widths become bound pairs. The [API reference](iddefix.rst) lists defaults.

If peaks are unsuitable but the resonator count is known, supply
`N_resonators` when constructing EA. Missing bounds are generated from the
input spectrum when fitting begins. These broad defaults do not provide
peak-derived starting estimates.

For explicit bounds, supply three pairs per resonator in `(Rs, Q, fr)` order.
The count is inferred from the number of triples:

```python
model = iddefix.EvolutionaryAlgorithm(
    frequency,
    impedance,
    parameterBounds=[(0, 200), (0.1, 100), (0.5e9, 1.5e9)],
    parameterEstimates=[100, 8, 1e9],
)
model.display_parameter_bounds()
```

## Baseline correction

For a linear time-domain baseline, call `model.remove_baseline()` before
`estimate_bounds()`. The slope is fitted to the last 20% of nonnegative time
samples; `fit_interval=(start, stop)` selects another interval in seconds.
Alternatively, use `model.estimate_bounds(remove_baseline=True)` to perform
both steps. A known slope can be passed to `remove_baseline(slope=a)`.

Impedance baseline correction requires `wake_length` in metres. Supply
`sigma` only when that impedance was Gaussian-deconvolved from a wake
potential; omit it for direct impedance or an FFT of a wake function.
Magnitude-only data lack the phase needed for this correction.

Correction clears existing fits and estimates. The removed slope is stored
as `model.baseline_slope`. See the
[linear-baseline model](theory.md#linear-wake-baseline) for its assumptions
and transform.

## Finite-wake correction

Pass `wake_length=L` [m] to EA when the input impedance comes from a truncated
wake. Bounds estimation applies an approximate finite-decay correction to
$R_s$ and enables Hann peak detection by default. Use
`model.estimate_bounds(use_Hann=False)` to disable that detection window.
The fitting data remain unwindowed.

See the [finite-wake correction](theory.md#finite-wake-correction) and
[Hann detection formula](theory.md#hann-peak-detection) for the underlying
approximations. A finite wake can still bias the estimated frequency and Q;
inspect the result before treating peak-derived values as a fit.

## Parameter estimation

The [theory guide](theory.md#peak-based-parameter-estimation) contains the
longitudinal and transverse peak formulas, crossing rules, and bound scaling.
It is a reference for how `estimate_bounds()` works; ordinary fitting starts
with the EA calls above.

## Editing peaks and fitted resonators

Numbers are 1-based, matching the displayed tables. Lists refer to the
numbering before removal; duplicates count once and at least one mode must
remain.

Before fitting, edit peaks through EA:

```python
model.add_peak(2e9, Q=10)
model.remove_peak(2)
```

These methods update the bounds and estimates and clear previous fits.
After fitting, `model.remove_resonator([1, 3])` instead preserves the fitted
values, uncertainties, and bounds of the remaining resonators. To append a
mode manually, provide its parameters and finite bounds for later fitting:

```python
model.add_resonator(
    Rs=100, Q=5, fres=2e9,
    parameter_bounds=[(0, 1000), (0.1, 100), (1.5e9, 2.5e9)],
)
```

A manually added mode detaches the peak estimator because its peak list no
longer describes the model. Existing fitted parameters remain available.

## Plotting fitted functions

Use `plot()` for the reference data and total model, or
`plot_resonator_components()` to include individual resonators. Both accept
`function="impedance"`, `"wake"`, or `"wake potential"`, independently of the
original fitting domain. Omit `function` to use that domain.

```python
fig, axes = model.plot()  # Use the fitting grid and reference data
fig, axes = model.plot_resonator_components(type="abs")
```

`type` selects `"real"`, `"imag"`, `"abs"`, or `"both"`. Impedance defaults to
magnitude for an absolute objective, including complex reference data;
otherwise it shows real and imaginary panels. Wake plots default to real
values. Changing the displayed part does not change the objective or fit.

Passing `x_data` evaluates the fitted model directly on that grid. It does
not resample or modify the reference data. Data are overlaid automatically
only when the function and grid match the fitting data. Pass `y_data`
explicitly to compare other data on the requested grid.

```python
frequency_plot = np.linspace(0.1e9, 3e9, 2001)
fig, axes = model.plot(frequency_plot, function="impedance", type="abs")
time_plot = np.linspace(-1e-9, 30e-9, 2001)
fig, axes = model.plot(time_plot, function="wake")
fig, axes = model.plot(time_plot, function="wake potential", sigma=1e-10)
```

Frequency is in Hz; time and Gaussian RMS duration `sigma` are in seconds.
Omitting `x_data` requires a stored grid in the selected domain. Impedance
uses the model's `wake_length` by default; an explicit value overrides it
for that plot. Gaussian potentials use the stored `sigma`, or 1e-10 s if
unset. Plotting leaves both stored settings unchanged.

Both methods return `(figure, axes)`, with a one-dimensional array of axes.
Use `show=False` to customize the figure before displaying it in a notebook:

```python
fig, axes = model.plot(type="abs", scale="loglog", show=False)
axes[0].set_title("Fitted impedance")
fig.tight_layout()
plt.show()
```

The existing `figsize`, `dpi`, `cmap`, line-style dictionaries, and additional
Matplotlib subplot arguments remain available. Individual magnitudes do not
necessarily sum to the magnitude of the total complex impedance.

## Uncertainties

Uncertainty estimates are off by default. Enable them with
`compute_uncertainties=True`, or request them after fitting with
`model.get_uncertainties()`. See
[parameter uncertainty estimates](theory.md#parameter-uncertainty-estimates)
for the assumptions; these are local least-squares estimates, not a model
selection criterion.

## Direct SmartBounds API

Use the standalone estimator for interactive peak selection or separate
peak-analysis tools. Automatic bound estimation for an EA fit should normally
use `model.estimate_bounds()`.

In a notebook with the ipympl backend, create the interactive estimator, pick
peaks, finish with **Done**, and apply it to EA:

```python
%matplotlib ipympl
bounds = iddefix.SmartBoundDetermination(
    frequency, impedance, plane=model.plane, interactive=True,
)
await bounds.wait_for_selection()
model.apply_bounds(bounds)
```

Applying bounds updates the model count and starting estimates and clears
previous fits. If you edit a standalone estimator with `add_peak()` or
`remove_peak()`, apply it again afterward. For its complete settings, see
`SmartBoundDetermination` in the [API reference](iddefix.rst).

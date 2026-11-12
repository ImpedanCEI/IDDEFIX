# Estimating resonator bounds and fitting

The recommended workflow uses `EvolutionaryAlgorithm` (EA) to estimate bounds, fit resonators with CMA-ES, and inspect the result. SmartBounds supplies the peak estimates and remains available as a direct API.

## Recommended workflow

`x_data` and `y_data` retain the original input. Baseline correction and downsampling update the domain data (`impedance_data`, `wake_data`, or `wake_potential_data`) used for fitting and plotting.

Create the model from impedance, wake, or wake-potential data. If a linear baseline is present, call `model.remove_baseline()` before estimating bounds. For impedance data, baseline removal requires `wake_length` in meters; for a deconvolved wake-potential spectrum, also supply the Gaussian bunch length `sigma` in seconds.

```python
model = iddefix.EvolutionaryAlgorithm(
    frequency,
    impedance,
    plane="longitudinal",
    wake_length=L,
)

# If needed: model.remove_baseline()
bounds = model.estimate_bounds()
model.run_cmaes()
# Optional local refinement: model.run_minimization_algorithm()
model.plot_resonator_components()
```

`estimate_bounds()` finds the resonator count, fitting bounds, and starting estimates, then shows an inspection plot followed by a table. It enables Hann peak detection when `wake_length` is supplied. If a peak is missed, call `model.add_peak(frequency, Q=known_q)` before fitting; the latest estimator is available as `model.smart_bounds`.

For wake and wake-potential inputs, EA transforms the data to impedance for peak detection and fits the original time-domain data. Use `fmax` and `fft_samples` to control the temporary frequency grid.

CMA-ES is the recommended global optimizer here; `run_differential_evolution()` remains available as an alternative. Local refinement with `run_minimization_algorithm()` is optional. `plot_resonator_components()` shows the fitted total and individual resonators.

Uncertainty estimates are off by default. Set `compute_uncertainties=True` at construction or call `model.get_uncertainties()` after fitting; see [Parameter uncertainty estimates](theory.md#parameter-uncertainty-estimates).

## Parameter estimation

SmartBounds estimates the shunt impedance $R_s$, quality factor $Q$, and resonant frequency $f_r$ from peaks in an impedance trace. The peak-and-crossing formulas below apply to one isolated, fully decayed resonator. For overlapping resonators or a varying background, use them as starting estimates with sufficiently wide bounds. A known finite wake length also permits an $R_s$ correction.

### Longitudinal impedance

The fully decayed longitudinal impedance is

$$
Z_\parallel(f)=
\frac{R_s}{
1+iQ\left(
\dfrac{f}{f_r}-\dfrac{f_r}{f}
\right)}.
$$

SmartBounds can use either $\Re(Z_\parallel)$ or $|Z_\parallel|$.

#### Using $\Re(Z_\parallel)$

Let the selected peak be $(f_{\mathrm{peak}},A_{\mathrm{peak}})$, where

$$
A_{\mathrm{peak}}=
\Re\left(Z_\parallel(f_{\mathrm{peak}})\right).
$$

Let $f_{\mathrm{crossing}}$ be the frequency where $\Re(Z_\parallel)$
reaches half of the peak amplitude:

$$
\Re\left(Z_\parallel(f_{\mathrm{crossing}})\right)
=\frac{A_{\mathrm{peak}}}{2}.
$$

The resonator estimates are

$$
R_s=A_{\mathrm{peak}},
$$

$$
f_r=f_{\mathrm{peak}},
$$

$$
Q=
\frac{f_{\mathrm{crossing}}f_{\mathrm{peak}}}
{|f_{\mathrm{crossing}}^2-f_{\mathrm{peak}}^2|}.
$$

#### Using $|Z_\parallel|$

Let the selected peak be $(f_{\mathrm{peak}},A_{\mathrm{peak}})$, where

$$
A_{\mathrm{peak}}=|Z_\parallel(f_{\mathrm{peak}})|.
$$

Let $f_{\mathrm{crossing}}$ be the frequency where $|Z_\parallel|$ reaches
the half-power level:

$$
|Z_\parallel(f_{\mathrm{crossing}})|
=\frac{A_{\mathrm{peak}}}{\sqrt{2}}.
$$

The resonator estimates are

$$
|R_s|=A_{\mathrm{peak}},
$$

$$
f_r=f_{\mathrm{peak}},
$$

$$
Q=
\frac{f_{\mathrm{crossing}}f_{\mathrm{peak}}}
{|f_{\mathrm{crossing}}^2-f_{\mathrm{peak}}^2|}.
$$

The sign of $R_s$ cannot be determined from $|Z_\parallel|$ alone.

### Transverse impedance

The fully decayed transverse impedance is

$$
Z_\perp(f)=
\frac{f_r}{f}
\frac{R_s}{
1+iQ\left(
\dfrac{f}{f_r}-\dfrac{f_r}{f}
\right)}.
$$

The factor $f_r/f$ shifts the maximum of $\Re(Z_\perp)$ away from $f_r$ and
changes its amplitude. Therefore, the longitudinal relations
$f_r=f_{\mathrm{peak}}$ and $R_s=A_{\mathrm{peak}}$ do not apply.

#### Using $\Re(Z_\perp)$

Let the selected peak be $(f_{\mathrm{peak}},A_{\mathrm{peak}})$, where

$$
A_{\mathrm{peak}}=
\Re\left(Z_\perp(f_{\mathrm{peak}})\right).
$$

Let $f_{\mathrm{crossing}}$ be the frequency where $\Re(Z_\perp)$ reaches
half of the peak amplitude:

$$
\Re\left(Z_\perp(f_{\mathrm{crossing}})\right)
=\frac{A_{\mathrm{peak}}}{2}.
$$

The dimensionless peak-location factor is defined as

$$
\eta_{\mathrm{peak}}=
\left(\frac{f_{\mathrm{peak}}}{f_r}\right)^2.
$$

It describes where the observed peak lies relative to the resonant frequency;
it is not the frequency shift itself. Its square root is the frequency ratio

$$
\sqrt{\eta_{\mathrm{peak}}}=
\frac{f_{\mathrm{peak}}}{f_r},
$$

so the fractional downward shift of the transverse peak is

$$
\frac{f_r-f_{\mathrm{peak}}}{f_r}
=1-\sqrt{\eta_{\mathrm{peak}}}.
$$

Thus, $\eta_{\mathrm{peak}}=1$ means that the peak is not shifted, while a
smaller value means that it lies further below $f_r$. For the transverse real
part, $0<\eta_{\mathrm{peak}}<1$, and it approaches $1$ in the high-$Q$ limit.

Because $f_r$ is initially unknown, SmartBounds determines
$\eta_{\mathrm{peak}}$ from the measured peak and crossing frequencies:

$$
\eta_{\mathrm{peak}}=
\sqrt{
\frac{
4\dfrac{f_{\mathrm{crossing}}}{f_{\mathrm{peak}}}
-\left(\dfrac{f_{\mathrm{crossing}}}{f_{\mathrm{peak}}}\right)^2-1
}{
\left(\dfrac{f_{\mathrm{crossing}}}{f_{\mathrm{peak}}}\right)^4
-3\left(\dfrac{f_{\mathrm{crossing}}}{f_{\mathrm{peak}}}\right)^2
+4\dfrac{f_{\mathrm{crossing}}}{f_{\mathrm{peak}}}
}
}.
$$

The transverse resonator estimates are then

$$
R_s=
A_{\mathrm{peak}}
\frac{
2\sqrt{\eta_{\mathrm{peak}}}(1+\eta_{\mathrm{peak}})
}{
1+3\eta_{\mathrm{peak}}
},
$$

$$
Q=
\sqrt{
\frac{
\eta_{\mathrm{peak}}
}{
(1-\eta_{\mathrm{peak}})(1+3\eta_{\mathrm{peak}})
}
},
$$

$$
f_r=
\frac{f_{\mathrm{peak}}}{\sqrt{\eta_{\mathrm{peak}}}}.
$$

If a $Q$ estimate is supplied when adding a peak, SmartBounds skips the
crossing calculation. For transverse real impedance, it instead obtains the
peak-location factor directly from that $Q$:

$$
\eta_{\mathrm{peak}}=
\frac{2Q^2-1+\sqrt{16Q^4-4Q^2+1}}{6Q^2}.
$$

It then uses the same $R_s$ and $f_r$ relations above. A peak with neither a
crossing nor a supplied $Q$ uses the fallback $Q=1$ without a transverse
peak-shift correction.

#### Why $|Z_\perp|$ is not recommended

The absolute transverse impedance does not provide a reliable general-purpose
peak estimate. For

$$
Q\leq\frac{1}{\sqrt{2}},
$$

$|Z_\perp|$ is largest at zero frequency and decreases with frequency. It has
no nonzero peak from which $R_s$, $Q$, and $f_r$ can all be determined.

Use $\Re(Z_\perp)$ for transverse SmartBounds estimation; bounds from $|Z_\perp|$ may be inconclusive.

## Finite-wake correction

When the input impedance comes from a wake truncated after length $L$,
SmartBounds first estimates $R_s$, $Q$, and $f_r$ with the fully decayed rules
above. It then corrects $R_s$ using the fraction of the wake envelope
accumulated over $T=L/c$, where $c$ is the speed of light and
$\omega_r=2\pi f_r$.
The effective decay rate is

$$
B_{\mathrm{eff}}=
\begin{cases}
\dfrac{\omega_r}{2Q}, & Q\geq 0.5,\\[8pt]
\dfrac{\omega_r}{2Q}-\omega_r\sqrt{\dfrac{1}{4Q^2}-1},
& 0<Q<0.5.
\end{cases}
$$

For $Q<0.5$, $B_{\mathrm{eff}}$ is the slower of the two overdamped decay
rates. With $R_{s,\infty}$ denoting the initial fully decayed estimate, the
finite-wake estimate is

$$
R_{s,T}=\frac{R_{s,\infty}}{1-\exp(-B_{\mathrm{eff}}T)}.
$$

This envelope-based factor is an approximation to the peak-height correction:
oscillatory boundary terms can also affect the finite-wake impedance peak.
The correction applies to longitudinal $\Re(Z)$ or $|Z|$ and transverse
$\Re(Z)$; it is not applied to transverse $|Z|$. The estimates of $Q$ and
$f_r$ are not corrected for the finite wake, even though a short wake may
change the peak position and width.

### Hann peak detection

For a wake of length $L$, Hann windowing suppresses truncation ripples in peak detection. Pass `use_Hann=True` to SmartBounds; `model.estimate_bounds()` enables it by default when `wake_length` is supplied. With $T=L/c$ and $f_H=c/L$, the windowed impedance is

$$
Z_H(f)=\frac{1}{2}Z(f)-\frac{1}{4}Z(f-f_H)-\frac{1}{4}Z(f+f_H).
$$

Negative frequencies follow the symmetry of a real wake: $Z_\parallel(-f)=Z_\parallel(f)^*$ and $Z_\perp(-f)=-Z_\perp(f)^*$. This extends the Hann trace to DC when the supplied grid starts at zero.

With Hann enabled, `impedance_type="real"` finds peaks in $\Re(Z_H)$; `"absolute"` uses $|Z_H|$ when the complex impedance is available. Magnitude-only input lacks the phase needed for Hann windowing, so peak detection uses the original magnitude. Peak heights and widths always come from the unwindowed trace.

### Linear wake baseline

A linear wake baseline $w_{\mathrm{base}}(t)=at$ over $0\leq t\leq T$ contributes a low-frequency background. Its longitudinal impedance per unit slope is

$$
K_\parallel(f)=c\int_0^T t\,e^{-2\pi i f t}\,dt
=\begin{cases}
\dfrac{c\left[e^{-i\omega T}(1+i\omega T)-1\right]}{\omega^2},
& f\ne 0,\\[8pt]
\dfrac{cT^2}{2}, & f=0,
\end{cases}
\qquad \omega=2\pi f.
$$

For transverse impedance, $K_\perp(f)=iK_\parallel(f)$. If the ramp is in a Gaussian wake potential that was deconvolved, pass the RMS bunch length `sigma` in seconds; the kernel becomes $K_\sigma(f)=K(f)/\exp[-(2\pi f\sigma)^2/2]$. Omit `sigma` when the ramp belongs to the underlying wake before bunch convolution.

Baseline correction subtracts $aK(f)$ from $Z(f)$, where $K$ is the kernel selected by `sigma`. With `remove_baseline=True`, the real slope $a$ is fitted by least squares; for complex impedance,

$$
\hat a=\frac{\Re\left[\sum_f K(f)^* Z(f)\right]}{\sum_f |K(f)|^2}.
$$

Use `model.remove_baseline()` as shown above, or pass a known slope with `model.remove_baseline(slope=a)`. Impedance correction requires `wake_length`; magnitude-only data lack the phase needed for a reliable correction. The removed slope is available as `model.baseline_slope`.

For wake or wake-potential data, the slope is fitted on the last 20% of nonnegative time samples and $at$ is subtracted for $t\geq0$. Use `model.remove_baseline(fit_interval=(start, stop))` to choose another interval. To combine correction and bound estimation, pass `remove_baseline=True` to `estimate_bounds()`:

```python
model = iddefix.EvolutionaryAlgorithm(
    time,
    wake_potential,
    fitFunction="wake potential",
    sigma=sigma,       # seconds
    wake_length=L,     # meters
)
model.estimate_bounds(remove_baseline=True)
```

This method assumes a finite linear ramp that can be distinguished from the physical impedance. EA corrects its fitting data; direct SmartBounds correction affects only peak analysis.

## Constructing fitting bounds

After estimating $R_s$, $Q$, and $f_r$, SmartBounds constructs the parameter
bounds in the order

$$
[(R_{s,\min},R_{s,\max}),
 (Q_{\min},Q_{\max}),
 (f_{r,\min},f_{r,\max})].
$$

The `Rs_bounds` and `Q_bounds` settings provide multiplicative factors,
both defaulting to `[0.1, 10]`. The frequency bounds
instead use a characteristic half-width calculated from the central estimates:

$$
\Delta f_{\mathrm{estimate}}
= \frac{f_{r,\mathrm{estimate}}}{2Q_{\mathrm{estimate}}}.
$$

For a fully decayed longitudinal resonator, $f_r/Q$ is the full width at half
maximum, so $\Delta f_{\mathrm{estimate}}$ is half of that full width. For a
transverse resonator, SmartBounds first corrects the peak shift to estimate
$f_r$ and $Q$; $\Delta f_{\mathrm{estimate}}$ is then used as a characteristic
frequency scale rather than as an exact transverse half-width.

The two entries in `fres_bounds` are dimensionless factors that multiply this
scale independently:

$$
f_{r,\min} = \max\left(\varepsilon, f_{r,\mathrm{estimate}} +
\mathrm{fres\_bounds}_{\min}\Delta f_{\mathrm{estimate}}\right),
$$

$$
f_{r,\max} = f_{r,\mathrm{estimate}} +
\mathrm{fres\_bounds}_{\max}\Delta f_{\mathrm{estimate}}.
$$

Thus, the bounds are centered on the estimated resonant frequency, which is
not necessarily the detected peak frequency for a transverse resonator. The
default `fres_bounds=[-1, 1]` spans one characteristic half-width on either
side. For example, `fres_bounds=[-2, 3]` spans two characteristic half-widths
below and three above.

For $f_{r,\mathrm{estimate}}=1\,\mathrm{GHz}$, the default bounds are
$[0.75,1.25]\,\mathrm{GHz}$ when $Q_{\mathrm{estimate}}=2$, but only
$[0.95,1.05]\,\mathrm{GHz}$ when $Q_{\mathrm{estimate}}=10$. Broad, low-$Q$
resonators therefore receive wider frequency intervals automatically. The
lower bound is kept positive with machine epsilon $\varepsilon$, but the
interval is not clipped to the frequency range of the input data.

## Direct SmartBounds API

Use `SmartBoundDetermination` when you need direct control over peak selection or maintain code that constructs bounds separately from EA. The EA workflow above is the usual entry point.

```python
bounds = iddefix.SmartBoundDetermination(
    frequency,
    impedance,
    plane="transverse",
    impedance_type="real",
    wake_length=L,
    use_Hann=True,
    fres_bounds=[-2, 2],
)
bounds.inspect(show_components=True)
```

Direct SmartBounds uses $\Re(Z)$ by default; choose `impedance_type="absolute"` for $|Z|$. Hann detection is opt-in here. The `Rs_bounds` and `Q_bounds` settings scale the parameter estimates, while `fres_bounds` sets the number of estimated half-widths around $f_r$.

If detection misses a peak, call `bounds.add_peak(peak_frequency)`. Pass `Q=Q_estimate` when the quality factor is known, or `q_side="left"` or `"right"` to select a crossing. Use `bounds.inspect(show_bounds=True)` to review the frequency intervals and `bounds.get_impedance_components()` to access individual resonance curves.

To fit these bounds with an existing EA model, call `model.apply_bounds(bounds)`; the impedance plane must match. For compatibility with older code, `bounds.N_resonators`, `bounds.parameterBounds`, and `bounds.parameterEstimates` remain available for manual model construction.

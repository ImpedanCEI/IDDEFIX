# SmartBounds parameter estimation

SmartBounds estimates the shunt impedance $R_s$, quality factor $Q$, and
resonant frequency $f_r$ from the peak of an impedance trace.

The formulas below apply to one isolated, fully decayed resonator. For
overlapping resonators, a varying background, or a finite wake, they should be
used as initial estimates with sufficiently wide fitting bounds.

## Longitudinal impedance

The fully decayed longitudinal impedance is

$$
Z_\parallel(f)=
\frac{R_s}{
1+iQ\left(
\dfrac{f}{f_r}-\dfrac{f_r}{f}
\right)}.
$$

SmartBounds can use either $\Re(Z_\parallel)$ or $|Z_\parallel|$.

### Using $\Re(Z_\parallel)$

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

### Using $|Z_\parallel|$

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

## Transverse impedance

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

### Using $\Re(Z_\perp)$

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

### Why $|Z_\perp|$ is not recommended

The absolute transverse impedance does not provide a reliable general-purpose
peak estimate. For

$$
Q\leq\frac{1}{\sqrt{2}},
$$

$|Z_\perp|$ is largest at zero frequency and decreases with frequency. It has
no nonzero peak from which $R_s$, $Q$, and $f_r$ can all be determined.

:::{warning}
Use $\Re(Z_\perp)$ for transverse SmartBounds estimation. If
`plane="transverse"` is combined with `impedance_type="absolute"`,
SmartBounds warns that the estimates may be inconclusive and recommends
`impedance_type="real"`. The absolute trace retains the legacy approximate
bounds for backward compatibility.
:::

## Constructing fitting bounds

After estimating $R_s$, $Q$, and $f_r$, SmartBounds constructs the parameter
bounds in the order

$$
[(R_{s,\min},R_{s,\max}),
 (Q_{\min},Q_{\max}),
 (f_{r,\min},f_{r,\max})].
$$

The `Rs_bounds` and `Q_bounds` settings provide multiplicative factors,
defaulting to `[0.1, 10]` and `[0.5, 5]`, respectively. The frequency bounds
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

For transverse real impedance, a typical setup is:

```python
bounds = iddefix.SmartBoundDetermination(
    frequency,
    impedance.real,
    impedance_type="real",
    plane="transverse",
)
```

The standard `inspect()` plot shows the detected peaks and crossings. Calling
`inspect(show_bounds=True)` also shows each estimated $f_r$ and its
frequency-bound interval. In this view, each resonator uses one color for all
its markers, with the numbered peaks providing the legend. Use the plot to
confirm that these features represent the intended resonator.

`inspect(show_components=True)` also plots each estimated resonator and their
total model. For numerical access, `bounds.get_impedance_components()` returns
one complex impedance array per resonator on the analysis frequency grid; pass
a frequency array to evaluate the components on another grid.

If automatic detection misses a peak, add it by frequency and let SmartBounds
recompute all peak metadata, estimates, and bounds:

```python
bounds.add_peak(peak_frequency)
```

The supplied frequency is mapped to the nearest point on the analysis grid. An
optional `q_side="left"` or `q_side="right"` selects the crossing side for the
new peak; otherwise the configured single `q_side` value is reused, or `"auto"`
is used when the original configuration contains one side per peak.

If a $Q$ estimate is already known, it can be supplied directly:

```python
bounds.add_peak(peak_frequency, Q=Q_estimate)
```

For this peak, SmartBounds skips the crossing calculation and uses the supplied
$Q$ to determine the $Q$ and frequency bounds. For a transverse real
impedance, it also uses this $Q$ to correct the shift between the selected peak
frequency and the estimated resonant frequency. It obtains the peak-location
factor directly from the supplied value:

$$
\eta_{\mathrm{peak}}=
\frac{2Q^2-1+\sqrt{16Q^4-4Q^2+1}}{6Q^2}.
$$

It then applies the same $R_s$ and $f_r$ relations given above. A peak with
no crossing and no supplied $Q$ instead uses the fallback $Q=1$ without a
transverse peak-shift correction.

The corresponding central estimates are available independently of the search
bounds:

```python
bounds.parameterEstimates
```

They use the flattened order
`[Rs_1, Q_1, fr_1, Rs_2, Q_2, fr_2, ...]`. Pass both results when constructing
the fitting model:

```python
model = iddefix.EvolutionaryAlgorithm(
    frequency,
    impedance,
    N_resonators=bounds.N_resonators,
    parameterBounds=bounds.parameterBounds,
    parameterEstimates=bounds.parameterEstimates,
)
```

`run_cmaes()` uses `parameterEstimates` as its initial point. The SciPy backend
of `run_differential_evolution()` also accepts it as `x0`; the PyFDE backends
do not support an explicit initial point. A minimization-only run starts from
`parameterEstimates` when available. Individual CMA-ES and SciPy DE calls can
override the stored estimate with their `x0` argument.

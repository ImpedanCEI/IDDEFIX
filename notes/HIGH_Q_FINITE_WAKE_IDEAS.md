# High-$Q$ extrapolation from a finite wake

## Problem

SmartBounds estimates $Q$ from a peak crossing without using `wake_length`;
the length currently corrects only the initial $R_s$. For a high-$Q$ mode, the
wake duration $T=L/c$ may be much shorter than its amplitude decay time
$\tau=Q/(\pi f_r)$. Then the measured peak width is dominated by truncation.
For an isolated longitudinal mode, the real-part width approaches roughly
$0.60c/L$, so a width-based $Q$ estimate can saturate far below the true value.
The default `Q_bounds` may consequently exclude the true $Q$ before fitting.

## Suppress truncation lobes for peak detection

The existing finite-wake impedance is the transform of a wake cut off at
$T=L/c$:

$$
Z_T(f)=\kappa\int_0^T w(t)e^{-2\pi i f t}\,dt.
$$

For peak detection, apply a Hann window
$h(t)=\tfrac12[1-\cos(2\pi t/T)]$ to this same interval. Euler's identity gives

$$
\cos(2\pi t/T)e^{-2\pi i f t}
=\tfrac12 e^{-2\pi i(f-1/T)t}
+\tfrac12 e^{-2\pi i(f+1/T)t}.
$$

By linearity, the Hann-windowed impedance can therefore be formed directly
from the **complex** measured finite-wake impedance:

$$
Z_{\mathrm{Hann}}(f)=\tfrac12 Z_T(f)
-\tfrac14 Z_T(f-c/L)
-\tfrac14 Z_T(f+c/L).
$$

The shifts are $1/T=c/L$. They implement a time-domain taper using frequency
data alone; they do not deconvolve the sinc or recover the missing wake tail.
The taper makes both interval endpoints zero, reducing the sidelobes caused by
the hard cutoff. It also broadens the main peak and changes its height, so
SmartBounds should use this trace only to locate candidates, then estimate
parameters and fit against the original complex impedance. Form the shifted
combination before taking a real part or magnitude. Measured frequency grids
will need interpolation for the shifts and enough coverage at both edges.
Show both traces in `inspect()` so weak or overlapping modes can be reviewed.
In one isolated synthetic case ($f_r=1$ GHz, $Q=5000$, $L=300$ m), the same
peak-height cutoff found 11 peaks on the original real trace and one on the
Hann trace; this still needs validation on mixed modes and measured data.
See [SciPy's windowing guide](https://docs.scipy.org/doc/scipy/tutorial/signal.html)
for the sidelobe-versus-width tradeoff.

## Development direction

1. Detect candidate modes on the Hann-windowed analysis trace above while
   retaining the full-resolution input for fitting.
2. Use the existing finite-wake resonator formula and known `wake_length` to
   initialize modes from the original complex impedance. Scan candidate
   $(f_r,Q)$ values and solve for $R_s$ by linear least squares at each point;
   fit overlapping modes jointly. This uses the truncation pattern rather
   than interpreting its width as the intrinsic linewidth.
3. Profile the fit error over $Q$, refitting the other parameters. When many
   $Q$ values fit the observed wake, report that range and show the resulting
   spread in extrapolated wakes. A good fit within $T$ alone does not establish
   a unique long-range tail.
4. If raw wake samples are available, consider a damped-mode time-domain
   initializer before the joint finite-wake fit. The two representations of
   the same wake should not be counted as independent evidence.

Validate with isolated and overlapping high-$Q$ modes, realistic noise and
model mismatch, and the accuracy of extrapolations beyond $T$. Until then,
users can widen `Q_bounds` when calling `estimate_bounds()`.

Theory
===

```{contents}
:depth: 3
```

## Longitudinal Resonator formalism


The longitudinal impedance of a purely resonant structure can be modeled by an equivalent
parallel RLC (Resistor, Inductor, Capacitor) resonator circuit.

### Impedance resonator formula

The longitudinal shunt impedance $R_s$ and impedance $Z_\parallel$ are measured
in ohms. Both the fully decayed and partuially decayed formulas support $Q>0$. For a fully
decayed wake, the single-resonator impedance is

$$
Z_{\parallel}(\omega)=
\frac{R_s}{1+iQ\left(\frac{\omega}{\omega_r}-\frac{\omega_r}{\omega}\right)}.
$$

For a wake truncated after the length $L$, define

$$
B=\frac{\omega_r}{2Q}, \qquad
C=\omega_r\sqrt{1-\frac{1}{4Q^2}}, \qquad
T=\frac{L}{c}, \qquad
p=B+i\omega.
$$

#### Underdamped $Q\geq0.5$

For $Q\geq0.5$, its finite Fourier transform is

$$
Z_{\parallel,T}(\omega)=
\frac{R_s\omega_r}{Q}
\frac{i\omega+e^{-pT}\left[-i\omega\cos(CT)
+\left(C^2+Bp\right)\frac{\sin(CT)}{C}\right]}
{C^2+p^2}.
$$

#### Critically damped $Q=0.5$

At $Q=0.5$, $C=0$ and the finite ratio is evaluated using

$$
\lim_{C\to0}\frac{\sin(CT)}{C}=T.
$$

#### Overdamped $Q<0.5$

For the overdamped branch, $0<Q<0.5$, define

$$
D=\omega_r\sqrt{\frac{1}{4Q^2}-1}.
$$

The finite transform is then

$$
Z_{\parallel,T}(\omega)=
\frac{R_s\omega_r}{Q}
\frac{i\omega+e^{-pT}\left[-i\omega\cosh(DT)
+\left(-D^2+Bp\right)\frac{\sinh(DT)}{D}\right]}
{p^2-D^2}.
$$

This follows from the substitutions
$$
C^2=-D^2, \quad \cos(CT)=\cosh(DT), \quad \sin(CT)/C=\sinh(DT)/D
$$

The implementation combines the hyperbolic terms with $e^{-BT}$ through the
positive decay rates $B-D$ and $B+D$ to avoid numerical overflow.

For multiple resonators, the contributions add linearly:

$$
\bar Z_{\parallel}(\omega)=
\sum^{N}_{n=1}Z_{\parallel}(\omega,R_{s,n},Q_n,\omega_{r,n}).
$$

Both found in the `Impedances` class as the functions `Resonator_longitudinal_imp` and `n_Resonator_longitudinal_imp`.

The three parameters are the shunt impedance $R_s$, quality factor $Q$, and
resonant frequency $f_r=\omega_r/(2\pi)$.

Equivalently, the wake function can be described by these three parameters.

### Wake function resonator formula

The longitudinal point wake is causal: $w_\parallel(t)=0$ for $t<0$. Using
$B$, $C$, and $D$ defined above, its branches for $t>0$ are:

#### Underdamped $Q>0.5$

$$
w_{\parallel}(t)=R_s\frac{\omega_r}{Q}e^{-Bt}
\left[\cos(Ct)-\frac{B}{C}\sin(Ct)\right].
$$

#### Critically damped $Q=0.5$

Taking $\sin(Ct)/C\to t$ as $C\to0$ gives

$$
w_{\parallel}(t)=2R_s\omega_r e^{-\omega_rt}(1-\omega_rt).
$$

#### Overdamped $0<Q<0.5$

Replacing $C$ by $iD$ gives

$$
w_{\parallel}(t)=R_s\frac{\omega_r}{Q}e^{-Bt}
\left[\cosh(Dt)-\frac{B}{D}\sinh(Dt)\right].
$$

For multiple resonators, the contributions add linearly:

$$
\bar w_{\parallel}(t)=
\sum^{N}_{n=1}w_{\parallel}(t,R_{s,n},Q_n,\omega_{r,n}).
$$

Both found in the `Wakes` class as the functions `Resonator_longitudinal_wake` and `n_Resonator_longitudinal_wake`.

### Wake potential resonator formula

The wake potential is the point wake convolved with a Gaussian bunch of RMS
duration $\sigma$. Unlike the causal point wake, it can be nonzero for $t<0$.

#### Underdamped $Q>0.5$

The point-wake oscillation can be written as $e^{-Bt}e^{iCt}=e^{-at}$
with $a=B-iC$. Using the normalized Gaussian bunch profile
$\lambda_\sigma$ defined below, completing the square gives

$$
K_\sigma(a,t)=\int_0^\infty e^{-as}\lambda_\sigma(t-s)\,ds
=\frac{1}{2}e^{a^2\sigma^2/2-at}
\operatorname{erfc}\left(\frac{a\sigma^2-t}{\sqrt{2}\sigma}\right).
$$

For $a=B-iC$, factor out the real decay and Gaussian envelope. The remaining
complex factor is

$$
\mathcal{A}_\sigma(t)=e^{i(Ct-BC\sigma^2)}
\operatorname{erfc}\left(
-\frac{t-B\sigma^2+iC\sigma^2}{\sqrt{2}\sigma}
\right).
$$

Hence

$$
2K_\sigma(B-iC,t)=e^{(B^2-C^2)\sigma^2/2-Bt}
\mathcal{A}_\sigma(t).
$$

The transverse sine is the imaginary part of $e^{iCt}$. Longitudinally, the
real part of $(1+iB/C)e^{iCt}$ gives $\cos(Ct)-(B/C)\sin(Ct)$. Thus the same
auxiliary factor supplies both wake-potential formulas.

The longitudinal wake potential is

$$
W_{\parallel}(t)=
R_sB\,e^{(B^2-C^2)\sigma^2/2-Bt}
\left[
\operatorname{Re}\mathcal{A}_\sigma(t)
-\frac{B}{C}\operatorname{Im}\mathcal{A}_\sigma(t)
\right].
$$

#### Critically damped $Q=0.5$

At $Q=0.5$, the same decay parameter becomes real:
$a=B-iC=\omega_r$. In the overdamped branch, the two real decay rates are
$a=B-D$ and $a=B+D$. For a real rate $a>0$, define the Gaussian
convolutions of $e^{-as}$ and its first-moment $s e^{-as}$ as:

$$
I_\sigma(a,t)=\int_0^\infty e^{-as}\lambda_\sigma(t-s)\,ds
=\frac{1}{2}
e^{a^2\sigma^2/2-at}
\operatorname{erfc}\left(\frac{a\sigma^2-t}{\sqrt{2}\sigma}\right),
$$

$$
J_\sigma(a,t)=\int_0^\infty s e^{-as}\lambda_\sigma(t-s)\,ds
=-\frac{\partial I_\sigma(a,t)}{\partial a}
=(t-a\sigma^2)I_\sigma(a,t)
+\frac{\sigma}{\sqrt{2\pi}}e^{-t^2/(2\sigma^2)}.
$$

At $Q=0.5$, $B=\omega_r$ and the longitudinal potential is

$$
W_\parallel(t)=2R_s\omega_r
\left[I_\sigma(\omega_r,t)-\omega_rJ_\sigma(\omega_r,t)\right].
$$

#### Overdamped $0<Q<0.5$

The two decay rates $B-D$ and $B+D$ give

$$
W_\parallel(t)=\frac{R_s\omega_r}{2Q}
\left[
\left(1-\frac{B}{D}\right)I_\sigma(B-D,t)
+\left(1+\frac{B}{D}\right)I_\sigma(B+D,t)
\right].
$$

For multiple resonators, the contributions add linearly:

$$
\bar W_{\parallel}(t)=
\sum^{N}_{n=1}W_{\parallel}(t,R_{s,n},Q_n,\omega_{r,n},\sigma).
$$

All three branches are implemented by `Resonator_longitudinal_wake_potential`
and `n_Resonator_longitudinal_wake_potential`.

## Transverse Resonator formalism

The transverse shunt impedance $R_s$ and impedance $Z_\perp$ are measured in
ohms per metre. Both the fully decayed and finite formulas support $Q>0$. For
a fully decayed wake, the single-resonator impedance is

$$
Z_\perp(\omega)=
\frac{\omega_r}{\omega}
\frac{R_s}{1+iQ\left(\frac{\omega}{\omega_r}-\frac{\omega_r}{\omega}\right)}.
$$

Using the definitions of $B$, $C$, $T$, and $p$ above, the finite transverse
impedance for $Q\geq0.5$ is

$$
Z_{\perp,T}(\omega)=
\frac{iR_s\omega_r^2}{Q\left(C^2+p^2\right)}
\left[
1-e^{-pT}\left(\cos(CT)+p\frac{\sin(CT)}{C}\right)
\right].
$$

At $Q=0.5$, the same limit $\sin(CT)/C\to T$ keeps this expression finite.
For $0<Q<0.5$, the transverse expression becomes

$$
Z_{\perp,T}(\omega)=
\frac{iR_s\omega_r^2}{Q\left(p^2-D^2\right)}
\left[
1-e^{-pT}\left(\cosh(DT)+p\frac{\sinh(DT)}{D}\right)
\right].
$$

It is also evaluated using the stable decay-rate form. For multiple resonators,

$$
\bar Z_\perp(\omega)=
\sum^{N}_{n=1}Z_\perp(\omega,R_{s,n},Q_n,\omega_{r,n}).
$$

These formulas are implemented by `Resonator_transverse_imp` and
`n_Resonator_transverse_imp`.

### Behavior around zero frequency

The apparent singularities in the fully decayed formulas are removable. With
$x=\omega/\omega_r$ and the Fourier convention used here, their low-frequency
behavior is

$$
Z_\parallel(\omega)=\frac{iR_s}{Q}x+R_s\mathcal{O}(x^2),
\qquad
Z_\perp(\omega)=\frac{iR_s}{Q}+\frac{R_s}{Q^2}x+R_s\mathcal{O}(x^2).
$$

Consequently, their analytical DC values are

$$
Z_\parallel(0)=0,
\qquad
Z_\perp(0)=\frac{iR_s}{Q}.
$$

Truncating a wake at time $T$ changes its area. The finite longitudinal DC
value is the truncation boundary term

$$
Z_{\parallel,T}(0)=\frac{R_s\omega_r}{Q}
\begin{cases}
e^{-BT}\dfrac{\sin(CT)}{C}, & Q>0.5, \\
Te^{-BT}, & Q=0.5, \\
e^{-BT}\dfrac{\sinh(DT)}{D}, & 0<Q<0.5.
\end{cases}
$$

It is generally nonzero and real, but converges to zero as $T\to\infty$. The
finite transverse DC value is generally nonzero and imaginary,

$$
Z_{\perp,T}(0)=i\int_0^T W_\perp(t)\,dt,
$$

and converges to $iR_s/Q$ as $T\to\infty$. At $T=0$, both finite transforms
vanish. The nonzero finite longitudinal value and the associated impedance
ripples are both consequences of the abrupt wake truncation.

### Wake function resonator formula

The transverse point wake is also zero for $t<0$. For $t>0$:

#### Underdamped $Q>0.5$

$$
w_\perp(t)=
R_s\frac{\omega_r^2}{QC}e^{-Bt}\sin(Ct).
$$

#### Critically damped $Q=0.5$

Using $\sin(Ct)/C\to t$ gives

$$
w_\perp(t)=2R_s\omega_r^2t e^{-\omega_rt}.
$$

#### Overdamped $0<Q<0.5$

With $C=iD$, the sine becomes a hyperbolic sine:

$$
w_\perp(t)=R_s\frac{\omega_r^2}{QD}e^{-Bt}\sinh(Dt).
$$

For multiple resonators,

$$
\bar w_\perp(t)=
\sum^{N}_{n=1}w_\perp(t,R_{s,n},Q_n,\omega_{r,n}).
$$

These formulas are implemented by `Resonator_transverse_wake` and
`n_Resonator_transverse_wake`.

### Wake potential resonator formula

The transverse wake potential uses the same Gaussian bunch profile and the
same $I_\sigma$ and $J_\sigma$ defined for the longitudinal potential.

#### Underdamped $Q>0.5$

Using $\mathcal{A}_\sigma(t)$ defined above,

$$
W_\perp(t)=
R_s\frac{\omega_rB}{C}
e^{(B^2-C^2)\sigma^2/2-Bt}
\operatorname{Im}\mathcal{A}_\sigma(t).
$$

#### Critically damped $Q=0.5$

$$
W_\perp(t)=2R_s\omega_r^2J_\sigma(\omega_r,t).
$$

#### Overdamped $0<Q<0.5$

$$
W_\perp(t)=\frac{R_s\omega_r^2}{2QD}
\left[I_\sigma(B-D,t)-I_\sigma(B+D,t)\right].
$$

For multiple resonators,

$$
\bar W_\perp(t)=
\sum^{N}_{n=1}W_\perp(t,R_{s,n},Q_n,\omega_{r,n},\sigma).
$$

All three branches are implemented by `Resonator_transverse_wake_potential`
and `n_Resonator_transverse_wake_potential`.

## Relations between wake, wake potential, and impedance

Let $u$ denote either the longitudinal plane $\parallel$ or the transverse
plane $\perp$, and define

$$
\kappa_\parallel=1, \qquad \kappa_\perp=i.
$$

### Wake and impedance

With $\omega=2\pi f$, the IDDEFIX convention relates impedance and wake through

$$
Z_u(\omega)=
\kappa_u\mathcal{F}\!\left\{w_u\right\}(\omega)=
\kappa_u\int_{-\infty}^{\infty}w_u(t)e^{-i\omega t}\,dt.
$$

#### Numerical FFT

The Fourier transform is evaluated numerically by `compute_fft`. For evenly
spaced samples $t_n=t_0+n\Delta t$, it computes

$$
Z^{\mathrm{FFT}}(f_k)=
\Delta s\,e^{-2\pi i f_k t_0}
\sum_{n=0}^{N-1}x_n e^{-2\pi i f_k n\Delta t},
\qquad \Delta s=c\Delta t.
$$

The factor $\Delta s$ normalizes NumPy's unnormalized FFT as an integral over
distance. Therefore, when $x_n$ contains a wake sampled as a function of time,
pass $x_n=w_u(t_n)/c$ to approximate the time integral above. The phase factor
$e^{-2\pi i f_k t_0}$ restores the physical origin of the samples because
`numpy.fft.fft` treats the first array element as if it occurred at $t=0$.

#### Finite wakes

The transverse convention requires the additional factor $i$. The transverse
relation applies at nonzero frequency because IDDEFIX defines $Z_\perp(0)=0$.
Since the wakes are causal, a wake truncated at $T$ gives

$$
Z_{u,T}(\omega)=
\kappa_u\int_0^T w_u(t)e^{-i\omega t}\,dt.
$$

### Wake potential and convolution

For either plane, the wake potential is the convolution of the wake function
with the normalized Gaussian bunch profile

$$
\lambda_\sigma(t)=
\frac{1}{\sqrt{2\pi}\sigma}
e^{-t^2/(2\sigma^2)},
$$

so that

$$
W_u(t)=
\left(w_u*\lambda_\sigma\right)(t)=
\int_{-\infty}^{\infty}
w_u(t-\tau)\lambda_\sigma(\tau)\,d\tau.
$$

This convolution is implemented by `compute_convolution`. The convolution
theorem gives

$$
\mathcal{F}\!\left\{W_u\right\}(\omega)=
\frac{Z_u(\omega)}{\kappa_u}
\mathcal{F}\!\left\{\lambda_\sigma\right\}(\omega).
$$

### Impedance from wake-potential deconvolution

Consequently, deconvolution of a wake potential recovers the impedance through

$$
Z_u(\omega)=
\kappa_u
\frac{\mathcal{F}\!\left\{W_u\right\}(\omega)}
{\mathcal{F}\!\left\{\lambda_\sigma\right\}(\omega)}.
$$

Because `compute_deconvolution` builds a Gaussian profile containing a `1/c`
factor, pass `W_u / c` to recover the normalization above; multiply the
transverse result by $i$.

### Sign and coordinate conventions

```{important}
**Conventions when exchanging wake data**

IDDEFIX uses $t>0$ behind the source, lowercase $w$ for the point-charge
wake function, and uppercase $W$ for the bunch wake potential. Its
longitudinal wake is positive for energy loss, while a positive transverse
wake gives a positive kick for a positive source offset. Check all three
conventions before importing data:

- **Xsuite/Xwakes:** the resonator signs and the time-domain convention agree
  with IDDEFIX. Xwakes also provides the beam coordinate $\zeta$, for which
  $t=-\zeta/(\beta c)$; trailing particles therefore have $\zeta<0$.
- **PyHEADTAIL:** trailing particles have $\Delta t<0$. With
  $t=-\Delta t$, the conversion is
  $w_\parallel^{\mathrm{IDDEFIX}}(t)=w_\parallel^{\mathrm{PyHEADTAIL}}(-t)$
  and
  $w_\perp^{\mathrm{IDDEFIX}}(t)=-w_\perp^{\mathrm{PyHEADTAIL}}(-t)$.
- **CST:** exported wake data are bunch wake potentials $W$, so they must be
  deconvolved to recover a point-charge impedance. Check the sign and the
  transverse offset normalization recorded by the selected CST result before
  importing it.
- **Wakis:** wake data are also bunch wake potentials. Wakis uses $s>0$
  behind the source and
  $Z_\parallel=-\widetilde W_\parallel/(v\widetilde\lambda)$,
  whereas IDDEFIX uses the positive longitudinal transform. Therefore,
  negate a Wakis longitudinal wake potential before using the IDDEFIX
  deconvolution convention. Both use the factor $i$ for the transverse
  impedance. Wakis `Zx` and `Zy` describe the simulated source offset in
  ohms; divide a dipolar result by that offset before fitting it in IDDEFIX
  as an impedance in ohms per metre. The same normalization is required for
  an offset-dependent CST transverse result.

See the [Xwakes wakefield definitions](https://xsuite.readthedocs.io/en/doc-theme/xwakes.html#wakefield-definitions),
[PyHEADTAIL wake implementations](https://github.com/PyCOMPLETE/PyHEADTAIL/blob/539408fceaf250cea5f4c838bcce5035ba19ed80/PyHEADTAIL/impedances/wakes.py),
[CST Wakefield Solver example](https://indico.cern.ch/event/1466612/contributions/6449147/attachments/3090656/5474622/CST%20tutorial%20SPS%20wire%20scanners.pdf),
and the [Wakis physics guide](https://wakis.readthedocs.io/physicsguide.html#from-wake-to-impedance)
for the native definitions.
```

## Fitting Resonators with Differential Evolution

Differential Evolution (DE) is a metaheuristic optimization method in the field of evolutionary algorithms inspired by the principles of natural selection and evolution.

DE algorithms operate by evolving a population of candidate solutions over several generations. The key steps and workings of the algorithm are:


**Population Initialization**
The algorithm begins by initializing a population of $N$ individuals (candidate solutions) randomly within the bounds of the search space. Each individual is represented as a vector of decision variables:

$$
        \textbf{x}_i = [x_{i,1},x_{i,2},...,x_{i,D}], \qquad i=1,2,...,N
$$

where $D$ is the dimensionality of the problem. Each variable $x_{i,j}$ is to be initialized by some stochastic process like a uniform distribution or by some sampling that tries to maximize coverage of the available parameter space. `SciPy` uses Latin Hypercube sampling to maximize coverage and avoid clustering. In either case, boundary constraints of the variable are needed:

$$
        x_{j}^{min} \leq x_{j}\leq x_{j}^{max}
$$

**Mutation**
Mutation generates a **mutant vector** $\mathbf{v}_i$ for each individual $\mathbf{x}_i$ in the population. Many mutation strategies are available, this project will stick to the $DE/rand/1$ strategy. This strategy combines randomly selected individuals:

$$
        \mathbf{v}_i=\mathbf{x}_{r1}+F(\mathbf{x}_{r2}-\mathbf{x}_{r3}),
$$

where $\mathbf{F}$ is the **mutation constant**, also known as the differential weight and integers $r1, r2, r3$, are chosen randomly from the interval $[1, N]$. The parameter $F$ is typically specified as a range (min,max), enabling the use of dithering. Dithering introduces random variation to the mutation constant on a generation-by-generation basis, following a uniform distribution. This technique can significantly accelerate convergence by balancing exploration and exploitation. While increasing the mutation constant expands the search radius, it may also slow down the convergence process.

**Crossover**

Crossover combines the mutant vector $\mathbf{v}_i$ with the current solution $\mathbf{x}_i$ to create a **trial vector** $\mathbf{u}_i$. The crossover is introduced so as to increase the diversity of the perturbed parameter vectors. For each dimension $j$:

$$
        \mathbf{u}_{i,j} = \begin{cases}
            v_{i,j} & \text{if } r_j \leq \text{CR} \text{ or } j = j_{\text{rand}}, \\
            x_{i,j} & \text{otherwise},
            \end{cases}
$$

where $r_j$ is a random variable between 0 and 1. $\text{CR}$ is the Crossover Probability, also bounded to be between 0 and 1. Finally the $j_{r}$ assigns a random dimension, to ensure at least one dimension comes from the mutant vector, preventing $\mathbf{u}_i = \mathbf{x}_i$

**Selection**

Finally, the trial vector $\mathbf{u}_i$ competes with the current individual $\mathbf{x}_i$ for survival into the next generation. The one with the better fitness value is selected:

$$
        \mathbf{x}_i^{new} = \begin{cases}
            \mathbf{u}_{i} & \text{if } f(\mathbf{u}_{i}) \leq f(\mathbf{x}_{i})  \\
            \mathbf{x}_{i} & \text{otherwise}.
            \end{cases}
$$

$\mathbf{f(\cdot)}$ is the **objective function** to be minimized.

**Stopping criteria**
    The algorithm iterates through mutation, crossover, and selection until a predefined stopping criterion is met, such as:

* A maximum number of generations or function evaluations
* Convergence tolerance (e.g., minimal change in the best solution across generations)
* Achieving a target fitness value

## Fitting Resonators with CMA-ES

The Covariance Matrix Adaptation Evolution Strategy (CMA-ES) is a stochastic,
derivative-free optimizer for continuous, nonlinear problems. Instead of
mutating individual parameter vectors independently, it learns a multivariate
normal search distribution from the best candidates in each generation. This
allows the search to adapt to correlations between resonator parameters such
as $R_s$, $Q$, and $f_r$.

For generation $g$, candidate parameter vectors are sampled as

$$
\mathbf{x}^{(g)}_k=
\mathbf{m}^{(g)}+\sigma^{(g)}
\mathcal{N}\!\left(\mathbf{0},\mathbf{C}^{(g)}\right),
\qquad k=1,\ldots,\lambda,
$$

where $\mathbf{m}^{(g)}$ is the distribution mean, $\sigma^{(g)}$ is the
global step size, $\mathbf{C}^{(g)}$ is the covariance matrix, and $\lambda$
is the population size. After evaluating the objective function, CMA-ES uses a
weighted combination of the best $\mu$ candidates to update the mean:

$$
\mathbf{m}^{(g+1)}=
\sum_{k=1}^{\mu}a_k\mathbf{x}^{(g)}_{k:\lambda},
\qquad
\sum_{k=1}^{\mu}a_k=1,
$$

where $\mathbf{x}^{(g)}_{k:\lambda}$ denotes the candidate with rank $k$ and
$a_k>0$ are recombination weights. The covariance matrix expands along
successful search directions and contracts along unsuccessful ones, while the
step-size update controls the overall search radius.

### IDDEFIX implementation

IDDEFIX runs the [pymoo CMA-ES implementation](https://pymoo.org/algorithms/soo/cmaes.html)
through `EvolutionaryAlgorithm.run_cmaes`. The fitted vector contains the
triplet $(R_s,Q,f_r)$ for each resonator. The initial mean is the midpoint of
each parameter bound. Pymoo normalizes bounded variables by default, so
`sigma` normally describes the initial search width in the normalized search
space rather than in ohms, hertz, or physical $Q$ units.

The main arguments are:

- `sigma`: initial global standard deviation.
- `popsize`: number $\lambda$ of candidates sampled per generation.
- `maxiter`: internal CMA-ES iteration limit.
- `verbose`: select pymoo output or the IDDEFIX progress bar.

Additional keyword arguments are passed to pymoo's `CMAES` constructor. Useful
options include `tolfun`, `tolx`, `maxfevals`, `restarts`,
`restart_from_best`, `incpopsize`, `bipop`, and `seed`. IDDEFIX defaults to
three restarts from the best solution and seed 42; supplied keywords override
these defaults.

```{important}
`maxiter` is not a strict limit on pymoo's reported generation count. CMA-ES
has its own convergence criteria, and enabled restarts create additional runs
whose generations are included in `res.algorithm.n_gen`. Use `maxfevals` for a
direct limit on objective-function evaluations, or set `restarts=0` when a
single CMA-ES run is required.
```

The optimized parameters are stored in `evolutionParameters`. As with a DE
fit, `run_minimization_algorithm` can subsequently refine this solution with a
local minimizer.

## Parameter uncertainty estimates

After fitting, IDDEFIX can estimate the uncertainty of each resonator
parameter. This is a local covariance estimate around the fitted parameter
vector $\hat{\theta}=(R_{s,1},Q_1,f_{r,1},\ldots)$. It describes how much
the parameters could vary while remaining consistent with the residuals and
the model's local response. It does not repeat the optimization or explore the
full range of possible fits.

### Residual Jacobian

Let $r_i(\theta)$ be the difference between a measured value and the model
prediction. IDDEFIX estimates the Jacobian $J$ numerically using three-point
finite differences:

$$
J_{ij}=\frac{\partial r_i}{\partial \theta_j}.
$$

For real measurements, the residual vector contains the real residuals. When
the input contains nonzero imaginary parts, the real and imaginary residuals
are stacked into one real-valued vector. Derivative probes are restricted to
the bounds supplied to the uncertainty calculation. By default,
`get_uncertainties()` uses physical bounds for the derivatives: $R_s$ is
unbounded, while $Q>0$ and $f_r>0$.

### Covariance and reported uncertainties

Under a local linear approximation, the unscaled covariance is the
pseudoinverse of $J^\mathsf{T}J$. IDDEFIX scales it by the residual variance
estimated from the fit:

$$
\widehat{\mathrm{Cov}}(\hat{\theta}) \approx
\frac{\sum_{i=1}^{M} r_i^2}{M-N} (J^\mathsf{T}J)^{+},
$$

where $M$ is the number of real residual values and $N$ is the number of
parameters. The degrees of freedom $M-N$ must be positive for this estimate to
be meaningful. The residual variance is inferred from the fit; the calculation
does not accept per-measurement uncertainties or weights. Consequently, model
mismatch and systematic errors also affect the estimated scale.

The pseudoinverse is computed using an SVD. Singular directions below the
numerical precision threshold are discarded. If the Jacobian is rank
deficient, this does not make the corresponding parameter combinations
identifiable; the reported marginal errors may fail to capture uncertainty in
those directions. The reported uncertainty for parameter $\theta_j$ is

$$
\sigma_{\theta_j}=\sqrt{\widehat{\mathrm{Cov}}(\hat{\theta})_{jj}}.
$$

It is an approximate one-standard-deviation error in the same units as that
parameter. In the table it is shown as `value ± uncertainty`. The table shows
only the diagonal errors; correlations between fitted parameters are not
displayed. Under a well-behaved, approximately Gaussian local fit, a value plus
or minus one such error is often treated as an approximate 68% interval. This
interpretation is unreliable for strongly nonlinear fits, poorly identifiable
parameters, or parameters near bounds, where the local symmetric covariance
approximation does not describe the constrained parameter region well.

By default, fitting methods calculate these uncertainties automatically. Set
`compute_uncertainties=False` to skip that calculation, then call
`model.get_uncertainties()` to calculate them later. The table's large-error
warning uses the relative error $|\sigma_{\theta}/\theta|$ and defaults to
a threshold of 0.20. This ratio is not useful for parameters at or near zero.

`get_impedance_uncertainty()` uses the parameter errors to produce a separate
pointwise sensitivity envelope for the real impedance. It varies selected
parameters over their uncertainty ranges, subject to configured multiplicative
limits. The envelope is a sensitivity scenario; it is not a confidence
interval for the impedance.

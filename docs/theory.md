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
in ohms. Both the fully decayed and finite formulas support $Q>0$. For a fully
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

The implementation
combines the hyperbolic terms with $e^{-BT}$ through the positive decay rates
$B-D$ and $B+D$ to avoid numerical overflow.

Unlike the fully decayed impedance, the finite transform can be nonzero at
$\omega=0$ due to the truncation ripples from the windowing.

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

The longitudinal wake function supports all $Q>0$ through underdamped,
critically damped, and overdamped branches. The expression below shows the
underdamped branch, $Q>0.5$.

#### Single:

$$
w_{\parallel}(t)=R_s\frac{\omega_r}{Q}e^{-Bt}
\left[\cos(Ct)-\frac{B}{C}\sin(Ct)\right].
$$

#### Multiple:

$$
\bar w_{\parallel}(t)=
\sum^{N}_{n=1}w_{\parallel}(t,R_{s,n},Q_n,\omega_{r,n}).
$$

Both found in the `Wakes` class as the functions `Resonator_longitudinal_wake` and `n_Resonator_longitudinal_wake`.

### Wake potential resonator formula

The analytical longitudinal wake potential for a Gaussian bunch currently
supports the underdamped range $Q>0.5$. Let $\sigma$ be the RMS bunch length in
seconds and define the auxiliary factor

$$
\mathcal{A}_\sigma(t)=e^{i(Ct-BC\sigma^2)}
\operatorname{erfc}\left(
-\frac{t-B\sigma^2+iC\sigma^2}{\sqrt{2}\sigma}
\right).
$$

The single-resonator wake potential is

$$
W_{\parallel}(t)=
R_sB\,e^{(B^2-C^2)\sigma^2/2-Bt}
\left[
\operatorname{Re}\mathcal{A}_\sigma(t)
-\frac{B}{C}\operatorname{Im}\mathcal{A}_\sigma(t)
\right].
$$

For multiple resonators, the contributions add linearly:

$$
\bar W_{\parallel}(t)=
\sum^{N}_{n=1}W_{\parallel}(t,R_{s,n},Q_n,\omega_{r,n},\sigma).
$$

These formulas are implemented by `Resonator_longitudinal_wake_potential` and
`n_Resonator_longitudinal_wake_potential`.

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

It is also evaluated using the stable decay-rate form.

IDDEFIX defines the transverse impedance as zero at $\omega=0$. For multiple
resonators,

$$
\bar Z_\perp(\omega)=
\sum^{N}_{n=1}Z_\perp(\omega,R_{s,n},Q_n,\omega_{r,n}).
$$

These formulas are implemented by `Resonator_transverse_imp` and
`n_Resonator_transverse_imp`.

### Wake function resonator formula

The transverse wake function supports all $Q>0$ through underdamped,
critically damped, and overdamped branches. For the underdamped range $Q>0.5$,
the single-resonator expression is

$$
w_\perp(t)=
R_s\frac{\omega_r^2}{QC}e^{-Bt}\sin(Ct).
$$

For multiple resonators,

$$
\bar w_\perp(t)=
\sum^{N}_{n=1}w_\perp(t,R_{s,n},Q_n,\omega_{r,n}).
$$

These formulas are implemented by `Resonator_transverse_wake` and
`n_Resonator_transverse_wake`.

### Wake potential resonator formula

The analytical transverse wake potential for a Gaussian bunch currently
supports the underdamped range $Q>0.5$. Using the same auxiliary factor
$\mathcal{A}_\sigma(t)$ defined above, the single-resonator wake potential is

$$
W_\perp(t)=
R_s\frac{\omega_rB}{C}
e^{(B^2-C^2)\sigma^2/2-Bt}
\operatorname{Im}\mathcal{A}_\sigma(t).
$$

For multiple resonators,

$$
\bar W_\perp(t)=
\sum^{N}_{n=1}W_\perp(t,R_{s,n},Q_n,\omega_{r,n},\sigma).
$$

These formulas are implemented by `Resonator_transverse_wake_potential` and
`n_Resonator_transverse_wake_potential`.

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

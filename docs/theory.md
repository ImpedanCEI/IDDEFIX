Theory
===

## Resonator formalism


The longitudinal impedance of a purely resonant structure can be modeled by an equivalent
parallel RLC (Resistor, Inductor, Capacitor) resonator circuit.

### Impedance resonator formula

#### Single:
\begin{equation}
        Z_{\parallel}(\omega, R_s, Q, \omega_r) = \frac{R_S}{1+iQ(\frac{\omega_r}{\omega}-\frac{\omega}{\omega_r})}
\end{equation}



#### Multiple:

\begin{equation}
 \bar Z_{\parallel}(\omega) = \sum^{N}_{n=1}Z_{\parallel}(\omega, R_{s,n}, Q_n, \omega_{r,n})
\end{equation}

Both found in the `Impedances` class as the functions `Resonator_longitudinal_imp` and `n_Resonator_longitudinal_imp`.

The three key parameters to characterize a resonator’s impedance: shunt resistance ($R_s$), quality factor ($Q$), and resonant frequency ($f_r = ω_r/2π$ ).

Equivialently, the wake function can be described by these three parameters.

### Wake function resonator formula

#### Single:

\begin{equation}
W_{||}(t)=R_s \frac{\omega_r}{Q} e^{-\omega_r \frac{t}{2Q}} [\text{cos}(\bar \omega_r t)- \frac{\omega_r}{2Q \bar \omega_r}\text{sin}(\bar \omega_r t)],\qquad t=\mathrm{s/c}
\end{equation}

#### Multiple:

\begin{equation}
\bar W_{\parallel}(s) = \sum^{N}_{n=1}W_{\parallel}(s, R_{s,n}, Q_n, \omega_{r,n})
\end{equation}

Both found in the `Wakes` class as the functions `Resonator_longitudinal_wake` and `n_Resonator_longitudinal_wake`.

Transverse functions are available as `Resonator_transverse_wake`, `n_Resonator_transverse_wake`, `Resonator_transverse_imp` and `n_Resonator_transverse_imp`.

## Fitting Resonators with Differential Evolution

Differential Evolution (DE) is a metaheuristic optimization method in the field of evolutionary algorithms inspired by the principles of natural selection and evolution.

DE algorithms operate by evolving a population of candidate solutions over several generations. The key steps and workings of the algorithm are:


**Population Initialization**
The algorithm begins by initializing a population of $N$ individuals (candidate solutions) randomly within the bounds of the search space. Each individual is represented as a vector of decision variables:

$$
        \textbf{x}_i = [x_{i,1},x_{i,2},...,x_{i,D}], \qquad i=1,2,...,N
$$

where $D$ is the dimensionality of the roblem. Each variable $x_{i,j}$ is to be initialized by some stochastic process like a uniform distribution or by some sampling that tries to maximize coverage of the available parameter space. `SciPy` uses Latin Hypercube sampling to maximize coverage and avoid clustering. In either case, boundary constraints of the variable are needed:

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

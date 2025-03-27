# IDDEFIX v0.1.0 
*Coming soon!*

## 🚀 New Features
* Smart Bound Determination (SBD):
    * Method `to_table()` to display the estimated parameter bounds.
    * Custom scale factors for Rs, Q, and fres: `Rs_bounds`, `Q_bounds`, `fres_bounds` for init. 

* Utils:
    * Integration of [`neffint`](https://github.com/ImpedanCEI/neffint) for non-equidistant Fourier transforms inside functions:
        - `compute_neffint()`: alternative to compute FFT.
        - `compute_ineffint()`: allows going from impedance to Wake potential, alternative to iFFT.
        - `compute_deconvolution()`: Allows going from wake potential to impedance using `FFT(wake)/FFT(charge_distribution)`. Assumes charge distribution is a Gaussian with `sigmaz` specified by the user in [s].
  * File I/O:
    * `read_txt()`: reads ASCII `.txt` files into arrays or dicts.
    * `save_txt()`: exports x/y data to formatted `.txt` files.

* Framework:
    * In `run_minimization_algorithm()`, the argument `margin` now supports a list of independent values for `[Rs, Q, fres]`, allowing finer control over parameter variations during optimization.
    * Updated docstrings for `EvolutionaryAlgorithm` methods for better documentation and clarity.
    * Check if beam sigma parameter is not passed when using wake potential functions and print a warning

* Solvers:
    * Implemented `pymoo`'s CMA-ES algorithm (`run_cmaes()`) as an alternative global optimization solver.
    * Added `run_pymoo_cmaes_solver()` for CMA-ES optimization using `pymoo`, with support for custom population size, sigma, and stopping criteria.
    * Included an example integrating CMA-ES with previous optimization workflows.

## 💗 Other Tag Highlights
* 🔁 Nightly tests with GitHub Actions: 
    - 001 -> Compare `neffint`, FFT, and analytical methods.
* 📁 Examples: 
    - 004a: Fit directly the wake potential data with wake potential resonator formalism
    - 004b: Fit impedance from wake potential data using `compute_deconvolution()`
    - 005: Examples using `pymoo`'s CMA-ES algorithm

## 🐛 Bugfixes 
* SBD: Method `find()` now updates `parameterBounds` in `self`.
* Trimmed trailing whitespaces for improved code readability.


## 👋👩‍💻New Contributors
* @babreufig - Implemented the CMA-ES algorithm using `pymoo`

## 📝Full changelog
`git log v0.0.2... --date=short --pretty=format:"* %ad %d %s (%aN)" | copy`
* 2025-03-26  Little timing comparison between CMAES and DE (Bernardo Abreu Figueiredo)
* 2025-03-26  Added two more examples to notebook (Bernardo Abreu Figueiredo)
* 2025-03-17  fix/feat: Check if sigma is not passed, and use the sigma value in self or default and print warning (elenafuengar)
* 2025-03-17  feature: add functions to read and write to .txt files (elenafuengar)
* 2025-03-16  build: include pymoo (optional) (elenafuengar)
* 2025-03-16  docs: Update readme (elenafuengar)
* 2025-03-16  fix: add `OptimizationProblem` class inside pymoo routine for import handling and clarity (elenafuengar)
* 2025-03-16  docs: update release.md (elenafuengar)
* 2025-03-16  docs: add docstring and error handling for pymoo imports (elenafuengar)
* 2025-03-16  Merge pull request #3 from babreufig/main (Elena de la Fuente García)
* 2025-03-12  Implemented CMA-ES and added one mixed example of previous notebooks (Bernardo Abreu Figueiredo)
* 2025-03-12  Trimmed trailing whitespaces (Bernardo Abreu Figueiredo)
* 2025-03-11  docs: update docstrings inside EvolutionaryAlgorithm methods (elenafuengar)
* 2025-03-11  docs: update release.md (elenafuengar)
* 2025-03-11  feature: allow different margins for Rs, Q and fres when running the minimization algorithm (elenafuengar)
* 2025-03-10  docs: prepare for 0.1.0 release (elenafuengar)
* 2025-03-10  docs: include call to new method `SBD.to_table()` (elenafuengar)
* 2025-03-10  docs: add docstring (elenafuengar)
* 2025-03-10  feature: add custom scaling factors for Rs, Q, fres in init (elenafuengar)
* 2025-03-10  fix: `find()` method was not updating paramBounds in self (elenafuengar)
* 2025-03-10  feature: add method to display parameter bounds as a table (elenafuengar)
* 2025-02-28  style: adding wake with lines plot (elenafuengar)
* 2025-02-24  fix: change/remove units (elenafuengar)
* 2025-02-11  test: use iddefix new `compute_neffint` and `compute_ineffint` functions (elenafuengar)
* 2025-02-11  feature: functions to compute fft and ifft using `neffint` (elenafuengar)
* 2025-02-11  test: add adaptative frequency refining (elenafuengar)
* 2025-02-11  docs: fix typo, update notebook list, new RTD version (elenafuengar)
# IDDEFIX v0.1.0 
*Comming soon!*

## 🚀 New Features
* Smart Bound Determination (SBD):
    * Method `to_table()` to display the estimated parameter bounds
    * Custom scale factors for Rs, Q and fres: `Rs_bounds`, `Q_bounds`, `fres_bounds` for init 

* Utils:
    * Integration of [`neffint`](https://github.com/ImpedanCEI/neffint) for non-equidistant Fourier transforms inside functions:
        - `compute_neffint()`: alternative to compute FFT
        - `compute_ineffint()`: allows to go from impedance to Wake potential, alternative to iFFT
        - `compute_deconvolution()`: Allows to go from wake potential to impedance using FFT(wake)/FFT(charge_distribution). Assumes charge distribution is a gaussian with `sigmaz` specified by the user in [s].
  
* Framework:
    * In `run_minimization_algorithm()`, the argument `margin` now supports a list independent values for [Rs, Q, and fres] margins, allowing finer control over parameter variations during optimization.

## 💗 Other Tag highlights
* 🔁 Nightly tests with GitHub actions: 
    - 001 -> compare neffint, FFT, and analytical 
* 📁 Examples: examples for longitudinal and transverse real-case impedances and wakes

## 🐛 Bugfixes 
* SBD: method `find()` now updates `parameterBounds` in `self`


## 👋👩‍💻New Contributors


## 📝Full changelog
`git log v0.0.2... --date=short --pretty=format:"* %ad %d %s (%aN)" | copy`
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
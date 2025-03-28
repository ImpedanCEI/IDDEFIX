# Changelog since v0.0.1 tag
`git log v0.0.1... --date=short --pretty=format:"* %ad %d %s (%aN)" | copy `

* 2025-02-06  (HEAD -> main, origin/main, origin/HEAD) fix: remove tests outputs from remote (elenafuengar)
* 2025-02-06  new: add text 001 to compare analytical, fft and neffint fft methods (elenafuengar)
* 2025-02-05  docs: add publications (elenafuengar)
* 2025-02-04  update: add partially and fully decayed impedance to result comparison (elenafuengar)
* 2025-02-04  update: use fitFunction argument in notebook (elenafuengar)
* 2025-02-04  fix: initialize attributes to None (elenafuengar)
* 2025-02-04  new: generalize parameters and docstring for impedance or wake or wake potential fit function modes (elenafuengar)
* 2025-02-03  add changelog (elenafuengar)
* 2025-02-03  add example using directly the wake resonator formalism as fit function for the EA (elenafuengar)
* 2025-02-03  add deconvolution routine (elenafuengar)
* 2025-02-03  use multithreaded sum for 10x speedup! (elenafuengar)
* 2025-02-03  feature: make sigma an optional argument for wake potential so it is suitable for using it as a fitFunction in the evolutionary algorithm (elenafuengar)
* 2025-02-01  use ML jargon (elenafuengar)
* 2025-01-31  use new framework methods, add real and imaginary check, update units, cosmetic changes (elenafuengar)
* 2025-01-31  more error handling and bugfix in get_impedance_from_fitFunction (elenafuengar)
* 2025-01-31  change units of time data to [s] (elenafuengar)
* 2025-01-30  add docstring to class (elenafuengar)
* 2025-01-30  add new methods for impedance and wakes to the example (elenafuengar)
* 2025-01-30  remove warning (elenafuengar)
* 2025-01-30  cosmetic changes and adding the new methods to get impedance and wake from the DE class (elenafuengar)
* 2025-01-30  change Nres to N_resonators for readability (elenafuengar)
* 2025-01-30  add new methods for impedance and wakes to the example (elenafuengar)
* 2025-01-30  add markdown format to print, change flag to `use minimization` and minor bugfixes (elenafuengar)
* 2025-01-30  add n_wake_potential for longitudinal and transverse (elenafuengar)
* 2025-01-30  feature: Add methods to get wakes, impedance and wake potential to the Evolutionary Algorithm class using the computed evolutionary or minimization algorithms (elenafuengar)
* 2025-01-29  change from wake potential to wake function (elenafuengar)
* 2025-01-29  update docstring, allow for ndarray in `minimum_peak_height`, add more parameters to tweak scipys find_peaks, change numbering of peaks to start in 1 (elenafuengar)
* 2025-01-29  Update README.md (Elena de la Fuente García)
* 2025-01-29  Update 004_sps_transition.ipynb (Elena de la Fuente García)
* 2025-01-29  Update 003_beam_wire_scanner.ipynb (Elena de la Fuente García)
* 2025-01-29  Update 002_extrapolation_sim_data.ipynb (Elena de la Fuente García)
* 2025-01-29  Update 001_analytical_resonator.ipynb (Elena de la Fuente García)
* 2025-01-29  Update README.md (Elena de la Fuente García)
* 2025-01-28  Update 004_sps_transition.ipynb (Elena de la Fuente García)
* 2025-01-28  small refactor: moving helper functions and adding the compute fft function to utils.py (elenafuengar)
* 2025-01-28  bump python version to 3.11 (elenafuengar)
* 2025-01-28  Update 004_sps_transition.ipynb (Elena de la Fuente García)
* 2025-01-28  Update 003_beam_wire_scanner.ipynb (Elena de la Fuente García)
* 2025-01-28  Update 002_extrapolation_sim_data.ipynb (Elena de la Fuente García)
* 2025-01-28  Update 001_analytical_resonator.ipynb (Elena de la Fuente García)
* 2025-01-28  reorganize badges (elenafuengar)
* 2025-01-28  solve security vulnerability (elenafuengar)
* 2025-01-28  add fork upstream to readme (elenafuengar)
* 2025-01-27  First version of the documentation (elenafuengar)
* 2025-01-27  update gitignore (elenafuengar)
* 2025-01-27  put logo inside docs/ (elenafuengar)
* 2025-01-27  Update README.md (Elena de la Fuente García)
* 2025-01-27  (tag: v0.0.2) Update 003_beam_wire_scanner.ipynb (Elena de la Fuente García)
* 2025-01-27  prepare 0.0.2 release (elenafuengar)
* 2025-01-27  add Nres as attribute (elenafuengar)
* 2025-01-24  Move out from dict to ndarray for passing params (Elena de la Fuente García)
* 2025-01-24  Move out from dict to ndarray to pass params (Elena de la Fuente García)
* 2025-01-24  small fix (Elena de la Fuente García)
* 2025-01-24  move out from dict to ndarray to pass params (Elena de la Fuente García)
* 2025-01-24  move out from dict to ndarray to pass FitFunction params (Elena de la Fuente García)
* 2025-01-24  add stopping criteria if convergence>1 (elenafuengar)
* 2025-01-24  add ndarray option for resonator parameters (elenafuengar)
* 2025-01-24  Update README.md (MaltheRaschke)
* 2025-01-24  Update theory.md (MaltheRaschke)
* 2025-01-24  Progress bar update to convergence based (Malthe Raschke Nielsen)
* 2025-01-24  Update README.md (MaltheRaschke)
* 2025-01-23  minor updates and fixes (Malthe Raschke Nielsen)
* 2025-01-23  Added 004 example and data (Malthe Raschke Nielsen)
* 2025-01-22  Minor changes and fixes (Malthe Raschke Nielsen)
* 2025-01-21  updating examples + new 003 example for beam wire scanner (Malthe Raschke Nielsen)
* 2025-01-21  Edit of init to include SBD (Malthe Raschke Nielsen)
* 2025-01-21  Smart bound determination code added to IDDEFIX (Malthe Raschke Nielsen)
* 2025-01-21  New example. CST cavity simulation (Malthe Raschke Nielsen)
* 2025-01-21  time_data and wake_data to be none (Malthe Raschke Nielsen)
* 2025-01-20  Rename files (Malthe Raschke Nielsen)
* 2025-01-20  add tqdm progress bar to pyfde solvers (elenafuengar)
* 2025-01-20  bugfix: rework progress bar for scipy.diferential_evolution() (elenafuengar)
* 2025-01-20  impedance data added for accelerator cavity sim. by CST (Malthe Raschke Nielsen)
* 2025-01-20  Updated - draft for finished notebook (Malthe Raschke Nielsen)
* 2025-01-20  change 'hyperparameters' to 'evolutionParameters' as hyperparameters was used in wrong context. New variable name is more fitting (Malthe Raschke Nielsen)
* 2025-01-20  updating 001 example notebook - cleared outputs. (Malthe Raschke Nielsen)
* 2025-01-20  updating 001 example notebook + code notes in framework. No need for WP data and time data. (Malthe Raschke Nielsen)
* 2025-01-20  correcting from genetic to evolutionary (Malthe Raschke Nielsen)
* 2025-01-17  added analytical res. parameters (Malthe Raschke Nielsen)
* 2025-01-17  Changed genetic algorihm to evolutionary and differential evolution (Malthe Raschke Nielsen)
* 2025-01-17  Remove all old examples (Malthe Raschke Nielsen)
* 2025-01-17  Starting the first example (Malthe Raschke Nielsen)
* 2025-01-17  fix import issues (elenafuengar)
* 2025-01-17  docs placeholders (elenafuengar)
* 2025-01-17  update to Sebastiens new logo (elenafuengar)
* 2024-12-13  Update README.md (MaltheRaschke)
* 2024-11-27  change homepage for pipy (elenafuengar)
* 2024-11-27  Name changes for pypi (Malthe Raschke Nielsen)
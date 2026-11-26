---
sd_hide_title: true
---
# IDDEFIX

**IDDEFIX** fits resonator models to longitudinal or transverse impedance,
wake functions, and Gaussian wake potentials. It estimates shunt impedance
$R_s$, quality factor $Q$, and resonant frequency $f_r$, then evaluates the
fitted model on frequency or time grids of your choice.

The package provides:

- Analytical resonator formulas for fully and partially decayed wakes
- Peak-based starting estimates and parameter bounds through SmartBounds
- CMA-ES, differential evolution, and local minimization
- Model plots, individual resonator contributions, and parameter uncertainties

## Getting started

Start with the [EA fitting workflow](smartbounds.md#recommended-workflow).
It contains a runnable example using `EvolutionaryAlgorithm`,
`estimate_bounds()`, a solver, and `plot()`. Separate sections cover input
units, baseline correction, manual bounds, and model editing.

The [public API](iddefix.rst) describes arguments and defaults. The
[theory guide](theory.md) collects the resonator and peak-estimation formulas.
Direct `SmartBoundDetermination` use is an advanced option for interactive
selection or standalone analysis.

```{toctree}
:caption: User guide
:maxdepth: 2
smartbounds.md
theory.md
iddefix.rst
```

## Examples

The notebooks below show complete fitting and extrapolation workflows.
Their saved results are included in the documentation; the documentation
build does not rerun the simulations or fits. Some notebooks demonstrate
manual bounds or separate SmartBounds construction. For the recommended
automatic workflow, follow the EA guide above.

```{toctree}
:caption: Examples
:maxdepth: 2
examples/001_analytical_resonator
examples/002_extrapolation_sim_data
examples/003_beam_wire_scanner
examples/004a_sps_transition_wake
examples/004b_sps_transition_impedance
```

Source code and additional examples are available in the
[GitHub repository](https://github.com/ImpedanCEI/IDDEFIX).

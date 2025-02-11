---
sd_hide_title: true
---
# 🔎 Overview

## Welcome to `IDDEFIX` documentation

> **IDDEFIX**: **I**mpedance **D**etermination through **D**ifferential **E**volution **FI**tting and e**X**trapolation

`IDDEFIX` is a package for **fitting resonators by a Differential Evolution algorithm to Impedance data** developed by Sébastian Joly. It computes the shunt impedance, Q-factor and resonant frequecny of the resonators present in the impedance data for both partially and fully decayed wakes. Extrapolation to a desired wakelength is then possible to quickly reconstruct the fully decayed wake.


🚀 `IDDEFIX` features:

* Resonators formulas
    * Longitudinal and transverse impedance (Fully/ partially decayed)
    * Longitudinal and transverse wake
    * Longitudinal and transverse wake potentials

* Differential Evolution algorihm for fitting resonsators to impedance
    * SciPy
    * pyfde ClassicDE
    * pyfde JADE


The source code is available in the `IDDEFIX` [GitHub repository](https://github.com/ImpedanCEI/IDDEFIX).

📚 For information on `IDDEFIX`'s differential evolution theory and code implementation, see:

```{toctree} 
:caption: IDDEFIX
:maxdepth: 2
theory.md
iddefix.rst
```

📁 For examples on how to use, check out [notebook examples](https://github.com/ImpedanCEI/IDDEFIX/tree/main/examples). They have been embedded in the documentation using `myst_nb`:

```{toctree} 
:caption: Examples
:maxdepth: 2
examples/001_analytical_resonator
examples/002_extrapolation_sim_data
examples/003_beam_wire_scanner
examples/004a_sps_transition_wake
examples/004b_sps_transition_impedance
```
<img src="https://raw.githubusercontent.com/SebastienJoly/GARFIELD/main/logo.png"  width="100px"/>

# IDDEFIX
> Originally forked from https://github.com/SebastienJoly/IDDEFIX

**I**mpedance **D**etermination through **D**ifferential **E**volution **FI**tting and e**X**trapolation.

![PyPI - Version](https://img.shields.io/pypi/v/IDDEFIX?style=flat-square&color=green)
![PyPI - License](https://img.shields.io/pypi/l/IDDEFIX?style=flat-square&color=pink)
[![Documentation Status](https://readthedocs.org/projects/iddefix/badge/?version=latest)](https://iddefix.readthedocs.io/en/latest/?badge=latest)

IDDEFIX is a physics-informed evolutionary optimization framework that fits a resonator-based model (parameterized by R, f, Q) to wakefield simulation data. It leverages Differential Evolution to optimize these parameters, enabling efficient classification and extrapolation of electromagnetic wakefield behavior. This allows for reduced simulation time while maintaining long-term accuracy, akin to time-series forecasting in machine learning


## About

🚀 `IDDEFIX` features:

* Resonators formulas
    * Longitudinal and transverse impedance (Fully/ partially decayed)
    * Longitudinal and transverse wake
    * Longitudinal and transverse wake potentials

* Differential Evolution algorithm for fitting resonsators to impedance
    * **SciPy**'s [Differential Evolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)
    * pyfde ClassicDE
    * pyfde JADE
    * **pyMOO**'s [CMA-ES](https://pymoo.org/algorithms/soo/cmaes.html) "Covariance Matrix Adaptation Evolution Strategy"

* Smart Bound Determination for precise and easy boundary setting

## How to install
IDDEFIX is deployed to the [Python Package Index (pyPI)](https://pypi.org/project/iddefix/). To install it in a conda environment do:
```
pip install iddefix
```
It can also be installed directly from the Github source to get the latest changes:
```
pip install git+https://github.com/ImpedanCEI/IDDEFIX
```

## How to use / Examples

IDDEFIX is documented using `Sphinx` and `ReadTheDocs`. Documentation is available at: http://iddefix.readthedocs.io/ 

Check :file_folder: `examples/` for different DE resonator fitting cases
* Analytical resonator initialization and fitting
* Resonator fitting on accelerator cavity simulation and extrapolation
* Resonator fitting on beam wire scanner simulation
* Resonator fitting on SPS transistion device and extrapolation

<img src="https://mattermost.web.cern.ch/files/4si7ipbezfyjdmd1zzr567hswh/public?h=2dcugjRruq3p9yEYea-9f1mXPfUbuujKRNh8dTA77a4"/>

## Contributors :woman_technologist: :man_technologist:
* Author : Sébastien Joly (sebastien.joly@helmholtz-berlin.de)
* Collaborator : 
  * Malthe Raschke (malthe@raschke.dk)
    - Refactored code and PYPI deployment
    - Smart Bound Determination
    - Example notebooks for extrapolation of analytical and simulated devices
  * Bernardo Abreu Figueiredo (bernardo.abreu.figueiredo@cern.ch):
    - CMA-ES kernel integration from `pymoo` 
* Maintainer: Elena de la Fuente (elena.de.la.fuente.garcia@cern.ch)

## Publications about `iddefix`
- S. Joly, *Resonator impedance extrapolation of a partially decayed wake* presentation @ CERN ABP-CEI section meeting [link](https://indico.cern.ch/event/1265710/contributions/5315305/attachments/2621462/4532440/Partially_decayed_wake.pdf)
- S. Joly, PhD Thesis *Recent advances in the CERN PS impedance model and instability simulations following the LHC Injectors Upgrade project* Sapienza Universita di Roma [link](https://hdl.handle.net/11573/1718791)
- M. Raschke, *Evolutionary Algorithms for Wakefields* @ CERN ABP-CEI section meeting [link](https://indico.cern.ch/event/1496532/contributions/6303923/attachments/2992287/5283277/Evolutionary%20algorithms%20for%20Wakefields%20-%20CERN.pdf)
- B. Figueiredo, *Using Xsuite and (CMA-ES) genetic methods to optimize FCC GHS momentum acceptance* @ CERN ABP-CAP section meeting [link](https://indico.cern.ch/event/1510103/#2-using-xsuite-and-genetic-met)

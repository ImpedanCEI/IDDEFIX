Public API
==========

The main entry points are available from ``iddefix``. Array inputs use SI
units: frequencies in Hz and times in seconds. See each method for impedance,
wake, and transverse-normalization conventions.

Fitting model
-------------

.. autoclass:: iddefix.framework.EvolutionaryAlgorithm
   :members:

Peak detection and bounds
-------------------------

.. autoclass:: iddefix.smartBoundDetermination.SmartBoundDetermination
   :members:

Resonator formulas
------------------

.. autoclass:: iddefix.resonatorFormulas.Impedances
   :members:

.. autoclass:: iddefix.resonatorFormulas.Wakes
   :members:

Objective functions
-------------------

.. autoclass:: iddefix.objectiveFunctions.ObjectiveFunctions
   :members:

Transforms and bunch profiles
-----------------------------

These functions are also exported directly from ``iddefix``.

.. autofunction:: iddefix.utils.compute_fft

.. autofunction:: iddefix.utils.compute_convolution

.. autofunction:: iddefix.utils.compute_deconvolution

.. autofunction:: iddefix.utils.compute_ineffint

.. autofunction:: iddefix.utils.compute_neffint

.. autofunction:: iddefix.utils.gaussian_bunch

.. autofunction:: iddefix.utils.gaussian_bunch_spectrum

Solver backends
---------------

Ordinary fitting uses the methods on ``EvolutionaryAlgorithm``. These
wrappers provide direct access to the numerical backends.

.. autoclass:: iddefix.solvers.Solvers
   :members:

Parameter uncertainties
-----------------------

For a fitted model, prefer ``model.get_uncertainties()``. The lower-level
function accepts parameters, a fit function, and data explicitly.

.. autofunction:: iddefix.uncertainties.get_uncertainties

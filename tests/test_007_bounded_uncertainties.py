import numpy as np

import iddefix
from iddefix.solvers import Solvers


def test_cmaes_uncertainties_stay_inside_partial_decay_q_bounds(monkeypatch):
    frequency = np.linspace(0.7e9, 1.3e9, 80)
    parameters = np.array([100.0, 0.5, 1e9])
    impedance = iddefix.Impedances.n_Resonator_longitudinal_imp(
        frequency, {0: parameters}, wake_length=30.0
    )
    bounds = [(10.0, 200.0), (0.5, 5.0), (0.8e9, 1.2e9)]
    model = iddefix.EvolutionaryAlgorithm(
        x_data=frequency,
        y_data=impedance,
        N_resonators=1,
        parameterBounds=bounds,
        wake_length=30.0,
        objectiveFunction="Complex",
    )
    result = object()
    monkeypatch.setattr(
        Solvers,
        "run_pymoo_cmaes_solver",
        lambda *args, **kwargs: (parameters, "Convergence achieved", result),
    )

    assert model.run_cmaes() is result
    assert np.isfinite(model.evolutionParametersUncertainties).all()

"""Regression tests for loading and reusing resonator parameter tables."""

import numpy as np
import pytest

import iddefix


def _model(n_resonators=2):
    frequency = np.linspace(0.5e9, 1.5e9, 61)
    return iddefix.EvolutionaryAlgorithm(
        x_data=frequency,
        y_data=np.zeros_like(frequency),
        N_resonators=n_resonators,
        parameterBounds=[(0.0, 1000.0), (0.5, 1e5), (0.1e9, 2e9)] * n_resonators,
    )


def test_load_table_accepts_values_with_or_without_uncertainties():
    model = _model()
    table = """Resonator | Rs | Q | fres
    1 |␛[31m 100 ␛[0m| 3 | 1e9
    2 | 200 ± 20 | 5 | 1.5e9"""

    model.load_resonator_parameters(table)

    np.testing.assert_allclose(
        model.minimizationParameters, [100.0, 3.0, 1e9, 200.0, 5.0, 1.5e9]
    )
    np.testing.assert_allclose(
        model.minimizationParametersUncertainties,
        [0.0, 0.0, 0.0, 20.0, 0.0, 0.0],
    )
    np.testing.assert_array_equal(
        model.evolutionParameters, model.minimizationParameters
    )
    assert np.isfinite(model.get_impedance(model.x_data)).all()


@pytest.mark.parametrize("to_markdown", [False, True])
def test_displayed_table_round_trips(capsys, to_markdown):
    model = _model()
    parameters = np.array([100.0, 3.0, 1e9, 200.0, 5.0, 1.5e9])
    uncertainties = np.array([30.0, 0.5, 1e7, 20.0, 0.5, 2e7])
    model.display_resonator_parameters(
        params=parameters,
        uncertainties=uncertainties,
        to_markdown=to_markdown,
    )

    model.load_resonator_parameters(capsys.readouterr().out, use_minimization=False)

    np.testing.assert_allclose(model.evolutionParameters, parameters)
    np.testing.assert_allclose(model.evolutionParametersUncertainties, uncertainties)


def test_recompute_uncertainties_after_loading_outside_fit_bounds():
    frequency = np.linspace(0.7e9, 1.3e9, 60)
    parameters = [1000.0, 100.0, 1e9]
    impedance = iddefix.Impedances.Resonator_longitudinal_imp(
        frequency, *parameters, wake_length=30.0
    ).real + 0.1 * np.sin(np.linspace(0.0, 2 * np.pi, frequency.size))
    bounds = [(10.0, 500.0), (0.5, 50.0), (0.8e9, 1.2e9)]
    model = iddefix.EvolutionaryAlgorithm(
        x_data=frequency,
        y_data=impedance,
        N_resonators=1,
        parameterBounds=bounds,
        wake_length=30.0,
        objectiveFunction="real",
    )
    model.load_resonator_parameters("1 | 1000 | 100 | 1e9")

    uncertainties = model.get_uncertainties()

    assert np.isfinite(uncertainties).all()
    assert np.all(uncertainties > 0)
    np.testing.assert_array_equal(
        model.minimizationParametersUncertainties, uncertainties
    )
    assert model.parameterBounds == bounds

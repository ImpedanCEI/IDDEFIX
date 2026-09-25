"""Regression test for editing a loaded resonator model."""

import numpy as np

import iddefix


def test_add_then_remove_resonator_updates_impedance_and_uncertainties():
    frequency = np.linspace(0.5e9, 1.5e9, 21)
    model = iddefix.EvolutionaryAlgorithm(
        x_data=frequency,
        y_data=np.zeros_like(frequency),
        N_resonators=2,
        parameterBounds=[(0.0, 1000.0), (0.5, 10.0), (0.5e9, 2e9)] * 2,
    )
    model.load_resonator_parameters(
        "1 | 100 ± 10 | 3 ± 0.3 | 1e9 ± 1e7\n2 | 200 ± 20 | 5 ± 0.5 | 1.5e9 ± 2e7"
    )

    model.add_resonator(300.0, 7.0, 1.2e9, uncertainty=[30.0, 0.7, 3e7])
    model.remove_resonator(1)

    assert model.N_resonators == 2
    assert len(model.parameterBounds) == 6
    np.testing.assert_allclose(
        model.minimizationParameters,
        [200.0, 5.0, 1.5e9, 300.0, 7.0, 1.2e9],
    )
    np.testing.assert_allclose(
        model.minimizationParametersUncertainties,
        [20.0, 0.5, 2e7, 30.0, 0.7, 3e7],
    )
    np.testing.assert_array_equal(
        model.evolutionParameters, model.minimizationParameters
    )
    expected = iddefix.Impedances.n_Resonator_longitudinal_imp(
        frequency, model.minimizationParameters
    )
    np.testing.assert_allclose(model.get_impedance(frequency), expected)

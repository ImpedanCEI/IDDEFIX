"""Regression test for uncertainty estimation with real input data."""

import numpy as np

import iddefix


def test_real_data_uncertainties_ignore_model_imaginary_part():
    frequency = np.linspace(0.7e9, 1.3e9, 31)
    parameters = [100.0, 3.0, 1e9]
    impedance = iddefix.Impedances.Resonator_longitudinal_imp(frequency, *parameters)
    real_data = impedance.real + 0.1 * np.sin(np.linspace(0, 2 * np.pi, 31))
    model = iddefix.EvolutionaryAlgorithm(
        x_data=frequency,
        y_data=real_data,
        N_resonators=1,
        parameterBounds=[(50.0, 150.0), (0.5, 10.0), (0.8e9, 1.2e9)],
        objectiveFunction="real",
    )
    model.load_resonator_parameters("1 | 100 | 3 | 1e9")

    original_fit = model.fitFunction
    real_uncertainties = model.get_uncertainties()

    def shifted_fit(x, pars):
        return original_fit(x, pars) + 1000j

    model.fitFunction = shifted_fit
    shifted_uncertainties = model.get_uncertainties()
    np.testing.assert_allclose(shifted_uncertainties, real_uncertainties)

    model.y_data = real_data + 1j * impedance.imag
    complex_uncertainties = model.get_uncertainties()
    assert np.all(complex_uncertainties > real_uncertainties)

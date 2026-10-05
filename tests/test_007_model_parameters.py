"""Resonator model lifecycle: load, display, edit, and evaluate parameters.

The tests check public model behavior after parameter changes, including the
partial-wake Q boundary and the transverse impedance at critical damping.
"""

import numpy as np
import pytest

import iddefix


def _model(n_resonators=2, **kwargs):
    frequency = np.linspace(0.5e9, 1.5e9, 61)
    return iddefix.EvolutionaryAlgorithm(
        x_data=frequency,
        y_data=np.zeros_like(frequency),
        N_resonators=n_resonators,
        parameterBounds=[(0.0, 1000.0), (0.5, 1e5), (0.1e9, 2e9)] * n_resonators,
        **kwargs,
    )


def test_loaded_table_evaluates_resonators_with_optional_uncertainties():
    model = _model()
    model.load_resonator_parameters(
        "Resonator | Rs | Q | fres\n1 | 100 | 3 | 1e9\n2 | 200 ± 20 | 5 | 1.5e9"
    )

    expected = [100.0, 3.0, 1e9, 200.0, 5.0, 1.5e9]
    np.testing.assert_allclose(model.minimizationParameters, expected)
    np.testing.assert_allclose(
        model.minimizationParametersUncertainties,
        [0.0, 0.0, 0.0, 20.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        model.get_impedance(model.x_data),
        iddefix.Impedances.n_Resonator_longitudinal_imp(model.x_data, expected),
    )


@pytest.mark.parametrize("to_markdown", [False, True])
def test_displayed_parameters_can_be_loaded(capsys, to_markdown):
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


def test_edit_add_and_remove_update_the_evaluated_model():
    model = _model()
    model.load_resonator_parameters(
        "1 | 100 ± 10 | 3 ± 0.3 | 1e9 ± 1e7\n2 | 200 ± 20 | 5 ± 0.5 | 1.5e9 ± 2e7"
    )
    model.evolutionParameters[0] = 90.0  # The two fit stages can differ.

    model.modify_resonator(2, Q=1.2, fr=1.4e9)
    np.testing.assert_allclose(model.evolutionParameters, [90, 3, 1e9, 200, 1.2, 1.4e9])
    np.testing.assert_allclose(
        model.minimizationParameters, [100, 3, 1e9, 200, 1.2, 1.4e9]
    )
    assert model.evolutionParametersUncertainties is None
    assert model.minimizationParametersUncertainties is None

    model.add_resonator(300.0, 7.0, 1.2e9, uncertainty=[30.0, 0.7, 3e7])
    model.remove_resonator(1)

    expected = [200.0, 1.2, 1.4e9, 300.0, 7.0, 1.2e9]
    assert model.N_resonators == 2
    np.testing.assert_allclose(model.minimizationParameters, expected)
    np.testing.assert_allclose(
        model.minimizationParametersUncertainties,
        [0.0, 0.0, 0.0, 30.0, 0.7, 3e7],
    )
    np.testing.assert_allclose(
        model.get_impedance(model.x_data),
        iddefix.Impedances.n_Resonator_longitudinal_imp(model.x_data, expected),
    )


def test_invalid_edits_leave_parameters_unchanged():
    model = _model(n_resonators=1)
    with pytest.raises(ValueError, match="Load or fit"):
        model.modify_resonator(1, Q=1)
    model.load_resonator_parameters("1 | 100 | 3 | 1e9")
    original = model.minimizationParameters.copy()

    for number, changes in (
        (0, {"Q": 1}),
        (2, {"Q": 1}),
        (1, {"Q": 0}),
        (1, {"fr": -1}),
        (1, {"Rs": np.nan}),
    ):
        with pytest.raises(ValueError):
            model.modify_resonator(number, **changes)
        np.testing.assert_array_equal(model.minimizationParameters, original)

    # Fully decayed resonators allow positive Q below critical damping.
    model.modify_resonator(1, Rs=0, Q=0.3)
    np.testing.assert_allclose(model.minimizationParameters, [0, 0.3, 1e9])


def test_partial_transverse_model_accepts_q_half_and_remains_finite():
    model = _model(n_resonators=1, plane="transverse", wake_length=5.0)
    model.load_resonator_parameters("1 | 0.1 | 2 | 1e9")

    with pytest.raises(ValueError, match="at least 0.5"):
        model.modify_resonator(1, Q=0.3)
    model.modify_resonator(1, Q=0.5)

    critical = model.get_impedance_from_fitFunction()
    nearby = iddefix.Impedances.Resonator_transverse_imp(
        model.x_data,
        Rs=0.1,
        Q=0.5000001,
        resonant_frequency=1e9,
        wake_length=5.0,
    )
    assert np.isfinite(critical).all()
    np.testing.assert_allclose(critical, nearby, rtol=2e-6, atol=1e-9)

"""Invalid resonator predictions must not score as a perfect fit."""

import numpy as np
import pytest

import iddefix


@pytest.mark.parametrize(
    "objective",
    [
        iddefix.ObjectiveFunctions.sumOfSquaredError,
        iddefix.ObjectiveFunctions.sumOfSquaredErrorReal,
        iddefix.ObjectiveFunctions.sumOfSquaredErrorAbs,
        iddefix.ObjectiveFunctions.logsumOfSquaredError,
        iddefix.ObjectiveFunctions.logsumOfSquaredErrorReal,
        iddefix.ObjectiveFunctions.logsumOfSquaredErrorAbs,
    ],
)
@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_objective_rejects_nonfinite_model(objective, invalid_value):
    frequency = np.linspace(0.1e9, 2e9, 20)

    def invalid_impedance(x, parameters):
        return np.full(x.shape, invalid_value, dtype=complex)

    with np.errstate(invalid="ignore"):
        loss = objective(
            [1.0, 0.5, 1e9], invalid_impedance, frequency, np.ones_like(frequency)
        )

    assert loss == float("inf")

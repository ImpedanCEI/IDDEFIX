import numpy as np
import pytest

import iddefix


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Complex", iddefix.ObjectiveFunctions.sumOfSquaredError),
        ("Real", iddefix.ObjectiveFunctions.sumOfSquaredErrorReal),
        ("Abs", iddefix.ObjectiveFunctions.sumOfSquaredErrorAbs),
    ],
)
def test_evolutionary_algorithm_selects_named_objective(name, expected):
    model = iddefix.EvolutionaryAlgorithm(
        x_data=np.array([1e9, 2e9]),
        y_data=np.array([1 + 1j, 2 + 2j]),
        N_resonators=1,
        parameterBounds=[(1, 10), (0.5, 5), (1e9, 2e9)],
        objectiveFunction=name,
    )

    assert model.objectiveFunction is expected

import numpy as np

from src.risk_metrics import (
    calculate_var,
    calculate_expected_shortfall,
    probability_of_loss
)


def test_var_and_es():

    returns = np.random.normal(0.05, 0.10, 1000)

    var = calculate_var(returns, 0.95)

    es = calculate_expected_shortfall(returns, 0.95)

    assert es <= var


def test_probability_of_loss():

    returns = np.array([-0.1, 0.2, -0.05, 0.1])

    prob = probability_of_loss(returns)

    assert prob == 0.5
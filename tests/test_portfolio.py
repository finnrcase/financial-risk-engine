import numpy as np

from src.validation import validate_inputs
from src.portfolio import (
    build_covariance_matrix,
    calculate_portfolio_return,
    calculate_portfolio_volatility,
)


def test_validate_inputs_passes_for_valid_data():
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    mean_returns = np.array([0.08, 0.04, 0.10, 0.18])
    volatilities = np.array([0.16, 0.06, 0.22, 0.55])
    correlation_matrix = np.array([
        [1.00, 0.20, 0.75, 0.35],
        [0.20, 1.00, 0.15, 0.05],
        [0.75, 0.15, 1.00, 0.45],
        [0.35, 0.05, 0.45, 1.00],
    ])

    validate_inputs(weights, mean_returns, volatilities, correlation_matrix)


def test_portfolio_math():
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    mean_returns = np.array([0.08, 0.04, 0.10, 0.18])
    volatilities = np.array([0.16, 0.06, 0.22, 0.55])
    correlation_matrix = np.array([
        [1.00, 0.20, 0.75, 0.35],
        [0.20, 1.00, 0.15, 0.05],
        [0.75, 0.15, 1.00, 0.45],
        [0.35, 0.05, 0.45, 1.00],
    ])

    covariance = build_covariance_matrix(volatilities, correlation_matrix)

    portfolio_return = calculate_portfolio_return(weights, mean_returns)

    portfolio_vol = calculate_portfolio_volatility(weights, covariance)

    assert portfolio_return > 0
    assert portfolio_vol > 0
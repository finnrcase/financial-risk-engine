import numpy as np

from src.portfolio import build_covariance_matrix
from src.simulation import run_monte_carlo_engine


def test_monte_carlo_runs():

    weights = np.array([0.4, 0.3, 0.2, 0.1])

    mean_returns = np.array([0.08, 0.04, 0.10, 0.18]) / 252

    volatilities = np.array([0.16, 0.06, 0.22, 0.55]) / np.sqrt(252)

    correlation_matrix = np.array([
        [1.00, 0.20, 0.75, 0.35],
        [0.20, 1.00, 0.15, 0.05],
        [0.75, 0.15, 1.00, 0.45],
        [0.35, 0.05, 0.45, 1.00],
    ])

    covariance = build_covariance_matrix(volatilities, correlation_matrix)

    results = run_monte_carlo_engine(
        weights,
        mean_returns,
        covariance,
        num_simulations=100,
        time_horizon_days=50,
    )

    assert "portfolio_paths" in results
    assert "terminal_returns" in results

    assert results["portfolio_paths"].shape[0] == 100
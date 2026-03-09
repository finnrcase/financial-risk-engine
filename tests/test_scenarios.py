import numpy as np

from src.scenarios import apply_scenario


def test_market_crash_scenario_changes_inputs():
    mean_returns = np.array([0.08, 0.04, 0.10, 0.18]) / 252
    volatilities = np.array([0.16, 0.06, 0.22, 0.55]) / np.sqrt(252)
    correlation_matrix = np.array([
        [1.00, 0.20, 0.75, 0.35],
        [0.20, 1.00, 0.15, 0.05],
        [0.75, 0.15, 1.00, 0.45],
        [0.35, 0.05, 0.45, 1.00],
    ])

    stressed = apply_scenario(
        mean_returns,
        volatilities,
        correlation_matrix,
        "market_crash",
    )

    assert stressed["mean_returns"].mean() < mean_returns.mean()
    assert stressed["volatilities"].mean() > volatilities.mean()
    assert stressed["correlation_matrix"][0, 1] > correlation_matrix[0, 1]
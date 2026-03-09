import numpy as np

from src.portfolio import build_covariance_matrix
from src.optimization import (
    calculate_portfolio_statistics,
    optimize_min_variance,
    optimize_max_sharpe,
    generate_efficient_frontier,
)


def test_calculate_portfolio_statistics_returns_expected_keys():
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    mean_returns = np.array([0.08, 0.04, 0.10, 0.18])
    volatilities = np.array([0.16, 0.06, 0.22, 0.55])
    correlation_matrix = np.array([
        [1.00, 0.20, 0.75, 0.35],
        [0.20, 1.00, 0.15, 0.05],
        [0.75, 0.15, 1.00, 0.45],
        [0.35, 0.05, 0.45, 1.00],
    ])

    covariance_matrix = build_covariance_matrix(volatilities, correlation_matrix)

    stats = calculate_portfolio_statistics(
        weights=weights,
        mean_returns=mean_returns,
        covariance_matrix=covariance_matrix,
    )

    assert "expected_return" in stats
    assert "volatility" in stats
    assert "sharpe_ratio" in stats


def test_optimize_min_variance_weights_sum_to_one():
    mean_returns = np.array([0.08, 0.04, 0.10, 0.18])
    volatilities = np.array([0.16, 0.06, 0.22, 0.55])
    correlation_matrix = np.array([
        [1.00, 0.20, 0.75, 0.35],
        [0.20, 1.00, 0.15, 0.05],
        [0.75, 0.15, 1.00, 0.45],
        [0.35, 0.05, 0.45, 1.00],
    ])

    covariance_matrix = build_covariance_matrix(volatilities, correlation_matrix)

    result = optimize_min_variance(
        mean_returns=mean_returns,
        covariance_matrix=covariance_matrix,
    )

    assert np.isclose(result["weights"].sum(), 1.0)
    assert np.all(result["weights"] >= 0)
    assert result["volatility"] >= 0


def test_optimize_max_sharpe_weights_sum_to_one():
    mean_returns = np.array([0.08, 0.04, 0.10, 0.18])
    volatilities = np.array([0.16, 0.06, 0.22, 0.55])
    correlation_matrix = np.array([
        [1.00, 0.20, 0.75, 0.35],
        [0.20, 1.00, 0.15, 0.05],
        [0.75, 0.15, 1.00, 0.45],
        [0.35, 0.05, 0.45, 1.00],
    ])

    covariance_matrix = build_covariance_matrix(volatilities, correlation_matrix)

    result = optimize_max_sharpe(
        mean_returns=mean_returns,
        covariance_matrix=covariance_matrix,
        risk_free_rate=0.02,
    )

    assert np.isclose(result["weights"].sum(), 1.0)
    assert np.all(result["weights"] >= 0)


def test_generate_efficient_frontier_returns_nonempty_dataframe():
    mean_returns = np.array([0.08, 0.04, 0.10, 0.18])
    volatilities = np.array([0.16, 0.06, 0.22, 0.55])
    correlation_matrix = np.array([
        [1.00, 0.20, 0.75, 0.35],
        [0.20, 1.00, 0.15, 0.05],
        [0.75, 0.15, 1.00, 0.45],
        [0.35, 0.05, 0.45, 1.00],
    ])

    covariance_matrix = build_covariance_matrix(volatilities, correlation_matrix)

    frontier_df = generate_efficient_frontier(
        mean_returns=mean_returns,
        covariance_matrix=covariance_matrix,
        n_points=10,
        risk_free_rate=0.02,
    )

    assert len(frontier_df) > 0
    assert "expected_return" in frontier_df.columns
    assert "volatility" in frontier_df.columns
    assert "sharpe_ratio" in frontier_df.columns
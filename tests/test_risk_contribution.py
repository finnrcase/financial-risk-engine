import numpy as np

from src.portfolio import build_covariance_matrix, calculate_portfolio_volatility
from src.risk_contribution import (
    calculate_marginal_risk_contribution,
    calculate_component_risk_contribution,
    calculate_percentage_risk_contribution,
    build_risk_contribution_summary,
)


def test_risk_contributions_sum_to_portfolio_volatility():
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    volatilities = np.array([0.16, 0.06, 0.22, 0.55])
    correlation_matrix = np.array([
        [1.00, 0.20, 0.75, 0.35],
        [0.20, 1.00, 0.15, 0.05],
        [0.75, 0.15, 1.00, 0.45],
        [0.35, 0.05, 0.45, 1.00],
    ])

    covariance_matrix = build_covariance_matrix(volatilities, correlation_matrix)

    component_risk_contribution = calculate_component_risk_contribution(
        weights,
        covariance_matrix
    )

    portfolio_volatility = calculate_portfolio_volatility(weights, covariance_matrix)

    assert np.isclose(component_risk_contribution.sum(), portfolio_volatility)


def test_percentage_risk_contributions_sum_to_one():
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    volatilities = np.array([0.16, 0.06, 0.22, 0.55])
    correlation_matrix = np.array([
        [1.00, 0.20, 0.75, 0.35],
        [0.20, 1.00, 0.15, 0.05],
        [0.75, 0.15, 1.00, 0.45],
        [0.35, 0.05, 0.45, 1.00],
    ])

    covariance_matrix = build_covariance_matrix(volatilities, correlation_matrix)

    percentage_risk_contribution = calculate_percentage_risk_contribution(
        weights,
        covariance_matrix
    )

    assert np.isclose(percentage_risk_contribution.sum(), 1.0)


def test_marginal_risk_contribution_shape_matches_weights():
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    volatilities = np.array([0.16, 0.06, 0.22, 0.55])
    correlation_matrix = np.array([
        [1.00, 0.20, 0.75, 0.35],
        [0.20, 1.00, 0.15, 0.05],
        [0.75, 0.15, 1.00, 0.45],
        [0.35, 0.05, 0.45, 1.00],
    ])

    covariance_matrix = build_covariance_matrix(volatilities, correlation_matrix)

    marginal_risk_contribution = calculate_marginal_risk_contribution(
        weights,
        covariance_matrix
    )

    assert marginal_risk_contribution.shape == weights.shape


def test_build_risk_contribution_summary_returns_expected_columns():
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    volatilities = np.array([0.16, 0.06, 0.22, 0.55])
    correlation_matrix = np.array([
        [1.00, 0.20, 0.75, 0.35],
        [0.20, 1.00, 0.15, 0.05],
        [0.75, 0.15, 1.00, 0.45],
        [0.35, 0.05, 0.45, 1.00],
    ])
    asset_names = ["Equities", "Bonds", "Tech", "Crypto"]

    covariance_matrix = build_covariance_matrix(volatilities, correlation_matrix)

    summary = build_risk_contribution_summary(
        weights,
        covariance_matrix,
        asset_names=asset_names
    )

    expected_columns = [
        "asset",
        "weight",
        "marginal_risk_contribution",
        "component_risk_contribution",
        "percentage_risk_contribution",
    ]

    assert list(summary.columns) == expected_columns
    assert len(summary) == len(weights)
    assert summary["asset"].tolist() == asset_names
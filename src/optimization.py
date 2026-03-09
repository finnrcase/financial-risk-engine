"""
Portfolio optimization module for the Financial Risk Engine.

This module implements classic mean-variance portfolio optimization,
including minimum variance portfolios, maximum Sharpe portfolios,
and efficient frontier generation.

Responsibilities:
- compute portfolio statistics for optimization
- solve minimum variance portfolio
- solve maximum Sharpe portfolio
- generate efficient frontier points
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.portfolio import (
    calculate_portfolio_return,
    calculate_portfolio_volatility,
)


def calculate_portfolio_statistics(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_free_rate: float = 0.02,
) -> dict:
    """
    Calculate portfolio return, volatility, and Sharpe ratio.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.
    mean_returns : np.ndarray
        Expected asset returns.
    covariance_matrix : np.ndarray
        Covariance matrix.
    risk_free_rate : float
        Risk-free rate.

    Returns
    -------
    dict
        {
            "expected_return": ...,
            "volatility": ...,
            "sharpe_ratio": ...
        }
    """

    weights = np.asarray(weights, dtype=float)
    mean_returns = np.asarray(mean_returns, dtype=float)
    covariance_matrix = np.asarray(covariance_matrix, dtype=float)

    expected_return = calculate_portfolio_return(weights, mean_returns)
    volatility = calculate_portfolio_volatility(weights, covariance_matrix)

    if volatility == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (expected_return - risk_free_rate) / volatility

    return {
        "expected_return": float(expected_return),
        "volatility": float(volatility),
        "sharpe_ratio": float(sharpe_ratio),
    }


def _weight_sum_constraint(weights: np.ndarray) -> float:
    """
    Constraint ensuring weights sum to 1.
    """
    return np.sum(weights) - 1.0


def _portfolio_volatility_objective(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    covariance_matrix: np.ndarray,
) -> float:
    """
    Objective function for minimum variance optimization.
    """
    return calculate_portfolio_volatility(weights, covariance_matrix)


def _negative_sharpe_objective(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_free_rate: float,
) -> float:
    """
    Objective function for maximum Sharpe optimization.
    """
    stats = calculate_portfolio_statistics(
        weights=weights,
        mean_returns=mean_returns,
        covariance_matrix=covariance_matrix,
        risk_free_rate=risk_free_rate,
    )

    return -stats["sharpe_ratio"]


def optimize_min_variance(
    mean_returns: np.ndarray,
    covariance_matrix: np.ndarray,
) -> dict:
    """
    Solve for the minimum variance portfolio.

    Parameters
    ----------
    mean_returns : np.ndarray
        Expected asset returns.
    covariance_matrix : np.ndarray
        Covariance matrix.

    Returns
    -------
    dict
        {
            "weights": ...,
            "expected_return": ...,
            "volatility": ...,
            "sharpe_ratio": ...
        }
    """

    mean_returns = np.asarray(mean_returns, dtype=float)
    covariance_matrix = np.asarray(covariance_matrix, dtype=float)

    num_assets = len(mean_returns)
    initial_weights = np.ones(num_assets) / num_assets
    bounds = [(0.0, 1.0) for _ in range(num_assets)]
    constraints = [{"type": "eq", "fun": _weight_sum_constraint}]

    result = minimize(
        _portfolio_volatility_objective,
        x0=initial_weights,
        args=(mean_returns, covariance_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    optimal_weights = result.x

    stats = calculate_portfolio_statistics(
        weights=optimal_weights,
        mean_returns=mean_returns,
        covariance_matrix=covariance_matrix,
    )

    return {
        "weights": optimal_weights,
        "expected_return": stats["expected_return"],
        "volatility": stats["volatility"],
        "sharpe_ratio": stats["sharpe_ratio"],
    }


def optimize_max_sharpe(
    mean_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_free_rate: float = 0.02,
) -> dict:
    """
    Solve for the maximum Sharpe ratio portfolio.

    Parameters
    ----------
    mean_returns : np.ndarray
        Expected asset returns.
    covariance_matrix : np.ndarray
        Covariance matrix.
    risk_free_rate : float
        Risk-free rate.

    Returns
    -------
    dict
        {
            "weights": ...,
            "expected_return": ...,
            "volatility": ...,
            "sharpe_ratio": ...
        }
    """

    mean_returns = np.asarray(mean_returns, dtype=float)
    covariance_matrix = np.asarray(covariance_matrix, dtype=float)

    num_assets = len(mean_returns)
    initial_weights = np.ones(num_assets) / num_assets
    bounds = [(0.0, 1.0) for _ in range(num_assets)]
    constraints = [{"type": "eq", "fun": _weight_sum_constraint}]

    result = minimize(
        _negative_sharpe_objective,
        x0=initial_weights,
        args=(mean_returns, covariance_matrix, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    optimal_weights = result.x

    stats = calculate_portfolio_statistics(
        weights=optimal_weights,
        mean_returns=mean_returns,
        covariance_matrix=covariance_matrix,
        risk_free_rate=risk_free_rate,
    )

    return {
        "weights": optimal_weights,
        "expected_return": stats["expected_return"],
        "volatility": stats["volatility"],
        "sharpe_ratio": stats["sharpe_ratio"],
    }


def generate_efficient_frontier(
    mean_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    n_points: int = 25,
    risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    """
    Generate efficient frontier points by solving minimum-volatility
    portfolios for a range of target returns.

    Parameters
    ----------
    mean_returns : np.ndarray
        Expected asset returns.
    covariance_matrix : np.ndarray
        Covariance matrix.
    n_points : int
        Number of efficient frontier points.
    risk_free_rate : float
        Risk-free rate.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - target_return
        - expected_return
        - volatility
        - sharpe_ratio
    """

    mean_returns = np.asarray(mean_returns, dtype=float)
    covariance_matrix = np.asarray(covariance_matrix, dtype=float)

    num_assets = len(mean_returns)
    bounds = [(0.0, 1.0) for _ in range(num_assets)]
    initial_weights = np.ones(num_assets) / num_assets

    min_return = float(np.min(mean_returns))
    max_return = float(np.max(mean_returns))
    target_returns = np.linspace(min_return, max_return, n_points)

    frontier_rows = []

    for target_return in target_returns:
        constraints = [
            {"type": "eq", "fun": _weight_sum_constraint},
            {
                "type": "eq",
                "fun": lambda w, tr=target_return: (
                    calculate_portfolio_return(w, mean_returns) - tr
                ),
            },
        ]

        result = minimize(
            _portfolio_volatility_objective,
            x0=initial_weights,
            args=(mean_returns, covariance_matrix),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not result.success:
            continue

        optimal_weights = result.x

        stats = calculate_portfolio_statistics(
            weights=optimal_weights,
            mean_returns=mean_returns,
            covariance_matrix=covariance_matrix,
            risk_free_rate=risk_free_rate,
        )

        frontier_rows.append({
            "target_return": float(target_return),
            "expected_return": stats["expected_return"],
            "volatility": stats["volatility"],
            "sharpe_ratio": stats["sharpe_ratio"],
        })

    frontier_df = pd.DataFrame(frontier_rows)

    return frontier_df
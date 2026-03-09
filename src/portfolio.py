"""
Portfolio mathematics module for the Financial Risk Engine.

This module implements the core financial calculations needed
to construct portfolio statistics.

Responsibilities:
- build covariance matrix
- compute portfolio expected return
- compute portfolio variance
- compute portfolio volatility
"""

from __future__ import annotations

import numpy as np


def build_covariance_matrix(
    volatilities: np.ndarray,
    correlation_matrix: np.ndarray
) -> np.ndarray:
    """
    Build covariance matrix from volatilities and correlations.

    Σ = D * ρ * D

    where:
    D = diagonal matrix of volatilities
    ρ = correlation matrix

    Parameters
    ----------
    volatilities : np.ndarray
        Asset volatilities.
    correlation_matrix : np.ndarray
        Correlation matrix.

    Returns
    -------
    np.ndarray
        Covariance matrix.
    """

    volatilities = np.asarray(volatilities, dtype=float)
    correlation_matrix = np.asarray(correlation_matrix, dtype=float)

    diagonal_vol = np.diag(volatilities)

    covariance_matrix = diagonal_vol @ correlation_matrix @ diagonal_vol

    return covariance_matrix


def calculate_portfolio_return(
    weights: np.ndarray,
    mean_returns: np.ndarray
) -> float:
    """
    Calculate expected portfolio return.

    E[R_p] = wᵀ μ

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.
    mean_returns : np.ndarray
        Expected asset returns.

    Returns
    -------
    float
        Expected portfolio return.
    """

    weights = np.asarray(weights, dtype=float)
    mean_returns = np.asarray(mean_returns, dtype=float)

    portfolio_return = weights @ mean_returns

    return float(portfolio_return)


def calculate_portfolio_variance(
    weights: np.ndarray,
    covariance_matrix: np.ndarray
) -> float:
    """
    Calculate portfolio variance.

    Var(R_p) = wᵀ Σ w

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.
    covariance_matrix : np.ndarray
        Asset covariance matrix.

    Returns
    -------
    float
        Portfolio variance.
    """

    weights = np.asarray(weights, dtype=float)
    covariance_matrix = np.asarray(covariance_matrix, dtype=float)

    variance = weights.T @ covariance_matrix @ weights

    return float(variance)


def calculate_portfolio_volatility(
    weights: np.ndarray,
    covariance_matrix: np.ndarray
) -> float:
    """
    Calculate portfolio volatility.

    σ_p = sqrt(wᵀ Σ w)

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.
    covariance_matrix : np.ndarray
        Asset covariance matrix.

    Returns
    -------
    float
        Portfolio volatility.
    """

    variance = calculate_portfolio_variance(weights, covariance_matrix)

    volatility = np.sqrt(variance)

    return float(volatility)


def portfolio_summary(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    volatilities: np.ndarray,
    correlation_matrix: np.ndarray
) -> dict:
    """
    Compute a full portfolio summary.

    Returns key statistics used by the risk engine.

    Returns
    -------
    dict
        {
            "expected_return": ...,
            "variance": ...,
            "volatility": ...,
            "covariance_matrix": ...
        }
    """

    covariance_matrix = build_covariance_matrix(volatilities, correlation_matrix)

    expected_return = calculate_portfolio_return(weights, mean_returns)

    variance = calculate_portfolio_variance(weights, covariance_matrix)

    volatility = calculate_portfolio_volatility(weights, covariance_matrix)

    return {
        "expected_return": expected_return,
        "variance": variance,
        "volatility": volatility,
        "covariance_matrix": covariance_matrix,
    }
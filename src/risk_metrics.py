"""
Risk metrics module for the Financial Risk Engine.

This module calculates risk statistics from Monte Carlo simulation results.
"""

from __future__ import annotations

import numpy as np


def calculate_var(
    returns: np.ndarray,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Value at Risk (VaR).

    Parameters
    ----------
    returns : np.ndarray
        Portfolio return distribution.
    confidence_level : float
        Confidence level (e.g., 0.95 or 0.99).

    Returns
    -------
    float
        VaR estimate.
    """

    percentile = (1 - confidence_level) * 100

    var = np.percentile(returns, percentile)

    return float(var)


def calculate_expected_shortfall(
    returns: np.ndarray,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Expected Shortfall (ES).

    ES = average loss beyond VaR.

    Parameters
    ----------
    returns : np.ndarray
        Portfolio return distribution.
    confidence_level : float

    Returns
    -------
    float
        Expected Shortfall.
    """

    var = calculate_var(returns, confidence_level)

    tail_losses = returns[returns <= var]

    es = tail_losses.mean()

    return float(es)


def probability_of_loss(returns: np.ndarray) -> float:
    """
    Probability that portfolio return is negative.
    """

    return float(np.mean(returns < 0))


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02
) -> float:
    """
    Calculate Sharpe Ratio.

    Parameters
    ----------
    returns : np.ndarray
    risk_free_rate : float
        Annual risk-free rate.

    Returns
    -------
    float
    """

    mean_return = np.mean(returns)
    std_return = np.std(returns)

    if std_return == 0:
        return 0.0

    sharpe = (mean_return - risk_free_rate) / std_return

    return float(sharpe)


def calculate_max_drawdown(portfolio_paths: np.ndarray) -> float:
    """
    Calculate maximum drawdown across simulations.

    Parameters
    ----------
    portfolio_paths : np.ndarray

    Returns
    -------
    float
    """

    drawdowns = []

    for path in portfolio_paths:

        cumulative_max = np.maximum.accumulate(path)

        drawdown = (path - cumulative_max) / cumulative_max

        drawdowns.append(drawdown.min())

    return float(np.min(drawdowns))


def build_risk_summary(
    terminal_returns: np.ndarray,
    portfolio_paths: np.ndarray
) -> dict:
    """
    Build summary risk statistics.

    Returns
    -------
    dict
    """

    summary = {
        "mean_return": float(np.mean(terminal_returns)),
        "volatility": float(np.std(terminal_returns)),
        "var_95": calculate_var(terminal_returns, 0.95),
        "var_99": calculate_var(terminal_returns, 0.99),
        "es_95": calculate_expected_shortfall(terminal_returns, 0.95),
        "probability_of_loss": probability_of_loss(terminal_returns),
        "max_drawdown": calculate_max_drawdown(portfolio_paths),
        "sharpe_ratio": calculate_sharpe_ratio(terminal_returns),
    }

    return summary
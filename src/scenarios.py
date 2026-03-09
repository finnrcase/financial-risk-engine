"""
Scenario stress testing module for the Financial Risk Engine.

This module defines preset macro scenarios that modify expected returns,
volatilities, and correlations before simulation.
"""

from __future__ import annotations

import numpy as np


AVAILABLE_SCENARIOS = {
    "baseline": "No stress adjustments.",
    "market_crash": "Broad market selloff, higher volatility, higher correlations.",
    "rate_spike": "Interest rates rise sharply, bonds weaken, volatility rises.",
    "inflation_shock": "Inflation surge hurts bonds and growth assets, commodities improve.",
    "tech_correction": "Technology assets sell off disproportionately.",
    "energy_shock": "Energy prices spike, commodities benefit, broad volatility rises.",
}


def get_available_scenarios() -> dict:
    """
    Return dictionary of available scenarios.
    """
    return AVAILABLE_SCENARIOS.copy()


def _increase_correlations(
    correlation_matrix: np.ndarray,
    increment: float,
    max_correlation: float = 0.95,
) -> np.ndarray:
    """
    Increase off-diagonal correlations by a fixed increment.
    """
    correlation_matrix = np.asarray(correlation_matrix, dtype=float).copy()

    n = correlation_matrix.shape[0]

    for i in range(n):
        for j in range(n):
            if i != j:
                correlation_matrix[i, j] = min(
                    correlation_matrix[i, j] + increment,
                    max_correlation,
                )

    np.fill_diagonal(correlation_matrix, 1.0)

    return correlation_matrix


def apply_scenario(
    mean_returns: np.ndarray,
    volatilities: np.ndarray,
    correlation_matrix: np.ndarray,
    scenario_name: str,
) -> dict:
    """
    Apply a named scenario to baseline portfolio assumptions.

    Parameters
    ----------
    mean_returns : np.ndarray
        Baseline expected returns.
    volatilities : np.ndarray
        Baseline asset volatilities.
    correlation_matrix : np.ndarray
        Baseline correlation matrix.
    scenario_name : str
        Name of scenario to apply.

    Returns
    -------
    dict
        {
            "mean_returns": adjusted_mean_returns,
            "volatilities": adjusted_volatilities,
            "correlation_matrix": adjusted_correlation_matrix
        }
    """
    mean_returns = np.asarray(mean_returns, dtype=float).copy()
    volatilities = np.asarray(volatilities, dtype=float).copy()
    correlation_matrix = np.asarray(correlation_matrix, dtype=float).copy()

    if scenario_name not in AVAILABLE_SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario_name}'. "
            f"Available scenarios: {list(AVAILABLE_SCENARIOS.keys())}"
        )

    if scenario_name == "baseline":
        pass

    elif scenario_name == "market_crash":
        mean_returns = mean_returns - 0.30 / 252
        volatilities = volatilities * 1.50
        correlation_matrix = _increase_correlations(correlation_matrix, 0.20)

    elif scenario_name == "rate_spike":
        # Assumes asset order roughly like:
        # equities, bonds, tech, crypto
        if len(mean_returns) >= 2:
            mean_returns[1] -= 0.12 / 252
        if len(mean_returns) >= 1:
            mean_returns[0] -= 0.05 / 252
        if len(mean_returns) >= 3:
            mean_returns[2] -= 0.06 / 252

        volatilities = volatilities * 1.20
        correlation_matrix = _increase_correlations(correlation_matrix, 0.10)

    elif scenario_name == "inflation_shock":
        if len(mean_returns) >= 2:
            mean_returns[1] -= 0.10 / 252
        if len(mean_returns) >= 1:
            mean_returns[0] -= 0.04 / 252
        if len(mean_returns) >= 3:
            mean_returns[2] -= 0.05 / 252
        if len(mean_returns) >= 4:
            mean_returns[3] += 0.03 / 252

        volatilities = volatilities * 1.25
        correlation_matrix = _increase_correlations(correlation_matrix, 0.08)

    elif scenario_name == "tech_correction":
        if len(mean_returns) >= 3:
            mean_returns[2] -= 0.20 / 252
        if len(mean_returns) >= 1:
            mean_returns[0] -= 0.06 / 252
        if len(mean_returns) >= 4:
            mean_returns[3] -= 0.08 / 252

        volatilities = volatilities.copy()
        if len(volatilities) >= 3:
            volatilities[2] *= 1.50
        if len(volatilities) >= 4:
            volatilities[3] *= 1.25

        correlation_matrix = _increase_correlations(correlation_matrix, 0.12)

    elif scenario_name == "energy_shock":
        if len(mean_returns) >= 1:
            mean_returns[0] -= 0.03 / 252
        if len(mean_returns) >= 2:
            mean_returns[1] -= 0.04 / 252
        if len(mean_returns) >= 4:
            mean_returns[3] += 0.06 / 252

        volatilities = volatilities * 1.15
        correlation_matrix = _increase_correlations(correlation_matrix, 0.07)

    return {
        "mean_returns": mean_returns,
        "volatilities": volatilities,
        "correlation_matrix": correlation_matrix,
    }
"""
Monte Carlo simulation engine for the Financial Risk Engine.

This module simulates portfolio performance under uncertainty by generating
correlated asset return paths using a multivariate normal distribution.

Outputs include:
- simulated asset returns
- simulated portfolio value paths
- terminal portfolio returns
"""

from __future__ import annotations

import numpy as np


def generate_correlated_returns(
    mean_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    num_simulations: int,
    time_horizon_days: int,
) -> np.ndarray:
    """
    Generate correlated asset returns using a multivariate normal distribution.

    Parameters
    ----------
    mean_returns : np.ndarray
        Expected asset returns.
    covariance_matrix : np.ndarray
        Covariance matrix of asset returns.
    num_simulations : int
        Number of Monte Carlo simulations.
    time_horizon_days : int
        Number of time steps (e.g., trading days).

    Returns
    -------
    np.ndarray
        Simulated asset returns with shape:
        (num_simulations, time_horizon_days, num_assets)
    """

    num_assets = len(mean_returns)

    simulated_returns = np.random.multivariate_normal(
        mean=mean_returns,
        cov=covariance_matrix,
        size=(num_simulations, time_horizon_days),
    )

    return simulated_returns


def simulate_portfolio_paths(
    weights: np.ndarray,
    simulated_asset_returns: np.ndarray,
    initial_portfolio_value: float = 1.0,
) -> np.ndarray:
    """
    Convert simulated asset returns into portfolio value paths.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.
    simulated_asset_returns : np.ndarray
        Simulated asset returns.
    initial_portfolio_value : float
        Starting portfolio value.

    Returns
    -------
    np.ndarray
        Portfolio value paths:
        shape (num_simulations, time_horizon_days)
    """

    num_simulations, time_horizon_days, _ = simulated_asset_returns.shape

    portfolio_paths = np.zeros((num_simulations, time_horizon_days))

    portfolio_returns = simulated_asset_returns @ weights

    portfolio_paths[:, 0] = initial_portfolio_value * (1 + portfolio_returns[:, 0])

    for t in range(1, time_horizon_days):
        portfolio_paths[:, t] = portfolio_paths[:, t - 1] * (1 + portfolio_returns[:, t])

    return portfolio_paths


def compute_terminal_returns(
    portfolio_paths: np.ndarray,
    initial_portfolio_value: float = 1.0,
) -> np.ndarray:
    """
    Compute terminal portfolio returns from simulated paths.

    Parameters
    ----------
    portfolio_paths : np.ndarray
        Simulated portfolio paths.
    initial_portfolio_value : float
        Initial portfolio value.

    Returns
    -------
    np.ndarray
        Terminal returns for each simulation.
    """

    terminal_values = portfolio_paths[:, -1]

    terminal_returns = (terminal_values - initial_portfolio_value) / initial_portfolio_value

    return terminal_returns


def run_monte_carlo_engine(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    num_simulations: int,
    time_horizon_days: int,
    initial_portfolio_value: float = 1.0,
) -> dict:
    """
    Run the full Monte Carlo portfolio simulation.

    Returns
    -------
    dict
        {
            "asset_returns": simulated_asset_returns,
            "portfolio_paths": portfolio_paths,
            "terminal_returns": terminal_returns
        }
    """

    simulated_asset_returns = generate_correlated_returns(
        mean_returns,
        covariance_matrix,
        num_simulations,
        time_horizon_days,
    )

    portfolio_paths = simulate_portfolio_paths(
        weights,
        simulated_asset_returns,
        initial_portfolio_value,
    )

    terminal_returns = compute_terminal_returns(
        portfolio_paths,
        initial_portfolio_value,
    )

    return {
        "asset_returns": simulated_asset_returns,
        "portfolio_paths": portfolio_paths,
        "terminal_returns": terminal_returns,
    }
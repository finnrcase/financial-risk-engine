"""
Scenario analysis module for the Financial Risk Engine.

This module runs Monte Carlo simulations across multiple macro scenarios
and compares resulting portfolio risk metrics.

Responsibilities:
- run analysis for a single scenario
- run comparison across multiple scenarios
- build scenario comparison summary tables
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio import build_covariance_matrix
from src.risk_metrics import build_risk_summary
from src.scenarios import apply_scenario
from src.simulation import run_monte_carlo_engine


def run_single_scenario_analysis(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    volatilities: np.ndarray,
    correlation_matrix: np.ndarray,
    scenario_name: str,
    num_simulations: int,
    time_horizon_days: int,
    initial_portfolio_value: float = 1.0,
) -> dict:
    """
    Run full risk analysis for a single named scenario.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.
    mean_returns : np.ndarray
        Baseline expected asset returns.
    volatilities : np.ndarray
        Baseline asset volatilities.
    correlation_matrix : np.ndarray
        Baseline correlation matrix.
    scenario_name : str
        Name of scenario to apply.
    num_simulations : int
        Number of Monte Carlo simulations.
    time_horizon_days : int
        Number of time steps in the simulation.
    initial_portfolio_value : float
        Starting portfolio value.

    Returns
    -------
    dict
        {
            "scenario": scenario_name,
            "risk_summary": ...,
            "simulation_results": ...,
            "adjusted_inputs": ...
        }
    """

    adjusted = apply_scenario(
        mean_returns=mean_returns,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        scenario_name=scenario_name,
    )

    adjusted_covariance_matrix = build_covariance_matrix(
        adjusted["volatilities"],
        adjusted["correlation_matrix"],
    )

    simulation_results = run_monte_carlo_engine(
        weights=weights,
        mean_returns=adjusted["mean_returns"],
        covariance_matrix=adjusted_covariance_matrix,
        num_simulations=num_simulations,
        time_horizon_days=time_horizon_days,
        initial_portfolio_value=initial_portfolio_value,
    )

    risk_summary = build_risk_summary(
        terminal_returns=simulation_results["terminal_returns"],
        portfolio_paths=simulation_results["portfolio_paths"],
    )

    return {
        "scenario": scenario_name,
        "risk_summary": risk_summary,
        "simulation_results": simulation_results,
        "adjusted_inputs": {
            "mean_returns": adjusted["mean_returns"],
            "volatilities": adjusted["volatilities"],
            "correlation_matrix": adjusted["correlation_matrix"],
            "covariance_matrix": adjusted_covariance_matrix,
        },
    }


def run_multi_scenario_comparison(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    volatilities: np.ndarray,
    correlation_matrix: np.ndarray,
    scenario_names: list[str],
    num_simulations: int,
    time_horizon_days: int,
    initial_portfolio_value: float = 1.0,
) -> list[dict]:
    """
    Run full risk analysis across multiple scenarios.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.
    mean_returns : np.ndarray
        Baseline expected asset returns.
    volatilities : np.ndarray
        Baseline asset volatilities.
    correlation_matrix : np.ndarray
        Baseline correlation matrix.
    scenario_names : list[str]
        Scenario names to evaluate.
    num_simulations : int
        Number of Monte Carlo simulations.
    time_horizon_days : int
        Number of time steps in the simulation.
    initial_portfolio_value : float
        Starting portfolio value.

    Returns
    -------
    list[dict]
        List of scenario analysis result dictionaries.
    """

    results = []

    for scenario_name in scenario_names:
        scenario_result = run_single_scenario_analysis(
            weights=weights,
            mean_returns=mean_returns,
            volatilities=volatilities,
            correlation_matrix=correlation_matrix,
            scenario_name=scenario_name,
            num_simulations=num_simulations,
            time_horizon_days=time_horizon_days,
            initial_portfolio_value=initial_portfolio_value,
        )
        results.append(scenario_result)

    return results


def build_scenario_comparison_table(
    scenario_results: list[dict]
) -> pd.DataFrame:
    """
    Build a comparison table of risk metrics across scenarios.

    Parameters
    ----------
    scenario_results : list[dict]
        Output from run_multi_scenario_comparison().

    Returns
    -------
    pd.DataFrame
        Comparison table with one row per scenario.
    """

    rows = []

    for result in scenario_results:
        risk_summary = result["risk_summary"]

        rows.append({
            "scenario": result["scenario"],
            "mean_return": risk_summary["mean_return"],
            "volatility": risk_summary["volatility"],
            "var_95": risk_summary["var_95"],
            "var_99": risk_summary["var_99"],
            "es_95": risk_summary["es_95"],
            "probability_of_loss": risk_summary["probability_of_loss"],
            "max_drawdown": risk_summary["max_drawdown"],
            "sharpe_ratio": risk_summary["sharpe_ratio"],
        })

    comparison_table = pd.DataFrame(rows)

    return comparison_table
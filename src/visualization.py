"""
Visualization module for the Financial Risk Engine.

This module generates charts for portfolio simulations and risk analysis.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_return_distribution(
    returns: np.ndarray,
    var_95: float | None = None,
    es_95: float | None = None
):
    """
    Plot distribution of simulated portfolio returns.
    """

    plt.figure(figsize=(10, 6))

    plt.hist(returns, bins=50, alpha=0.7)

    if var_95 is not None:
        plt.axvline(var_95, linestyle="--", label="VaR 95%")

    if es_95 is not None:
        plt.axvline(es_95, linestyle=":", label="Expected Shortfall")

    plt.title("Portfolio Return Distribution")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()

    return plt


def plot_portfolio_paths(
    portfolio_paths: np.ndarray,
    num_paths_to_show: int = 100
):
    """
    Plot Monte Carlo portfolio paths.
    """

    plt.figure(figsize=(10, 6))

    num_simulations = portfolio_paths.shape[0]

    num_paths_to_show = min(num_paths_to_show, num_simulations)

    for i in range(num_paths_to_show):
        plt.plot(portfolio_paths[i], alpha=0.3)

    plt.title("Monte Carlo Portfolio Paths")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")

    plt.tight_layout()

    return plt


def compute_drawdowns(portfolio_paths: np.ndarray) -> np.ndarray:
    """
    Compute drawdowns for each simulation.
    """

    drawdowns = []

    for path in portfolio_paths:
        running_max = np.maximum.accumulate(path)
        dd = (path - running_max) / running_max
        drawdowns.append(dd.min())

    return np.array(drawdowns)


def plot_drawdown_distribution(
    portfolio_paths: np.ndarray
):
    """
    Plot distribution of drawdowns.
    """

    drawdowns = compute_drawdowns(portfolio_paths)

    plt.figure(figsize=(10, 6))

    plt.hist(drawdowns, bins=50, alpha=0.7)

    plt.title("Maximum Drawdown Distribution")
    plt.xlabel("Drawdown")
    plt.ylabel("Frequency")

    plt.tight_layout()

    return plt


def plot_risk_contributions(risk_contribution_df):
    """
    Plot percentage risk contribution by asset.
    """

    plt.figure(figsize=(10, 6))

    plt.bar(
        risk_contribution_df["asset"],
        risk_contribution_df["percentage_risk_contribution"]
    )

    plt.title("Asset Risk Contributions")
    plt.xlabel("Asset")
    plt.ylabel("Percentage Risk Contribution")

    plt.xticks(rotation=45)
    plt.tight_layout()

    return plt


def plot_scenario_metric_comparison(
    comparison_df,
    metric: str
):
    """
    Plot a selected risk metric across scenarios.
    """

    plt.figure(figsize=(10, 6))

    plt.bar(comparison_df["scenario"], comparison_df[metric])

    plt.title(f"Scenario Comparison: {metric}")
    plt.xlabel("Scenario")
    plt.ylabel(metric)

    plt.xticks(rotation=45)
    plt.tight_layout()

    return plt


def plot_efficient_frontier(
    frontier_df,
    current_portfolio: dict | None = None,
    min_variance_portfolio: dict | None = None,
    max_sharpe_portfolio: dict | None = None,
):
    """
    Plot efficient frontier with optional portfolio overlays.
    """

    plt.figure(figsize=(10, 6))

    plt.plot(
        frontier_df["volatility"],
        frontier_df["expected_return"],
        marker="o",
        linestyle="-",
        label="Efficient Frontier",
    )

    if current_portfolio is not None:
        plt.scatter(
            current_portfolio["volatility"],
            current_portfolio["expected_return"],
            label="Current Portfolio",
        )

    if min_variance_portfolio is not None:
        plt.scatter(
            min_variance_portfolio["volatility"],
            min_variance_portfolio["expected_return"],
            label="Min Variance",
        )

    if max_sharpe_portfolio is not None:
        plt.scatter(
            max_sharpe_portfolio["volatility"],
            max_sharpe_portfolio["expected_return"],
            label="Max Sharpe",
        )

    plt.title("Efficient Frontier")
    plt.xlabel("Volatility")
    plt.ylabel("Expected Return")
    plt.legend()
    plt.tight_layout()

    return plt


def plot_portfolio_allocation(asset_names, weights):
    """
    Plot portfolio allocation pie chart.
    """

    plt.figure(figsize=(6, 6))

    plt.pie(
        weights,
        labels=asset_names,
        autopct="%1.1f%%",
        startangle=90,
    )

    plt.title("Portfolio Allocation")

    plt.tight_layout()

    return plt
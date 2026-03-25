"""
Visualization module for the Financial Risk Engine.

This module generates charts for portfolio simulations and risk analysis.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


FIG_BG = "#f8fafc"
AX_BG = "#ffffff"
GRID = "#d9e1ec"
TEXT = "#142033"
MUTED = "#667085"
ACCENT = "#6f93bf"
ACCENT_DARK = "#456a98"
ACCENT_SOFT = "#c9d8ea"
POSITIVE = "#3f7c68"
NEGATIVE = "#9a5f6b"


def _apply_axis_style(ax, title: str, xlabel: str, ylabel: str) -> None:
    """
    Apply a consistent chart language across the application.
    """
    ax.set_facecolor(AX_BG)
    ax.set_title(title, fontsize=14, fontweight="bold", color=TEXT, pad=16)
    ax.set_xlabel(xlabel, color=MUTED, labelpad=10)
    ax.set_ylabel(ylabel, color=MUTED, labelpad=10)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.grid(axis="x", visible=False)

    for spine in ax.spines.values():
        spine.set_visible(False)


def _build_figure(figsize: tuple[float, float] = (10, 6)):
    """
    Create a figure with consistent product styling.
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(FIG_BG)
    return fig, ax


def plot_return_distribution(
    returns: np.ndarray,
    var_95: float | None = None,
    es_95: float | None = None,
):
    """
    Plot distribution of simulated portfolio returns.
    """
    fig, ax = _build_figure()
    ax.hist(returns, bins=50, alpha=0.9, color=ACCENT, edgecolor=AX_BG, linewidth=0.4)

    if var_95 is not None:
        ax.axvline(var_95, linestyle="--", linewidth=2.0, color=NEGATIVE, label="VaR 95%")

    if es_95 is not None:
        ax.axvline(es_95, linestyle=":", linewidth=2.3, color=ACCENT_DARK, label="Expected Shortfall")

    _apply_axis_style(ax, "Portfolio Return Distribution", "Return", "Frequency")
    if var_95 is not None or es_95 is not None:
        ax.legend(frameon=False, fontsize=10, labelcolor=MUTED)
    fig.tight_layout()
    return plt


def plot_portfolio_paths(
    portfolio_paths: np.ndarray,
    num_paths_to_show: int = 100,
):
    """
    Plot Monte Carlo portfolio paths.
    """
    fig, ax = _build_figure()
    num_simulations = portfolio_paths.shape[0]
    num_paths_to_show = min(num_paths_to_show, num_simulations)

    for i in range(num_paths_to_show):
        ax.plot(portfolio_paths[i], alpha=0.18, linewidth=1.0, color=ACCENT)

    ax.plot(portfolio_paths[:num_paths_to_show].mean(axis=0), color=ACCENT_DARK, linewidth=2.4)
    _apply_axis_style(ax, "Monte Carlo Portfolio Paths", "Time", "Portfolio Value")
    fig.tight_layout()
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


def plot_drawdown_distribution(portfolio_paths: np.ndarray):
    """
    Plot distribution of drawdowns.
    """
    drawdowns = compute_drawdowns(portfolio_paths)
    fig, ax = _build_figure()
    ax.hist(drawdowns, bins=50, alpha=0.92, color=NEGATIVE, edgecolor=AX_BG, linewidth=0.4)
    _apply_axis_style(ax, "Maximum Drawdown Distribution", "Drawdown", "Frequency")
    fig.tight_layout()
    return plt


def plot_risk_contributions(risk_contribution_df):
    """
    Plot percentage risk contribution by asset.
    """
    fig, ax = _build_figure()
    values = risk_contribution_df["percentage_risk_contribution"]
    colors = [ACCENT_DARK if value >= values.mean() else ACCENT_SOFT for value in values]
    ax.bar(
        risk_contribution_df["asset"],
        values,
        color=colors,
        edgecolor=AX_BG,
        linewidth=0.5,
        width=0.66,
    )
    _apply_axis_style(ax, "Asset Risk Contributions", "Asset", "Percentage Risk Contribution")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return plt


def plot_scenario_metric_comparison(comparison_df, metric: str):
    """
    Plot a selected risk metric across scenarios.
    """
    fig, ax = _build_figure()
    ax.bar(
        comparison_df["scenario"],
        comparison_df[metric],
        color=ACCENT,
        edgecolor=AX_BG,
        linewidth=0.5,
        width=0.62,
    )
    _apply_axis_style(ax, f"Scenario Comparison: {metric}", "Scenario", metric)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
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
    fig, ax = _build_figure()
    ax.plot(
        frontier_df["volatility"],
        frontier_df["expected_return"],
        marker="o",
        markersize=4.5,
        linestyle="-",
        linewidth=2.0,
        color=ACCENT,
        label="Efficient Frontier",
    )

    if current_portfolio is not None:
        ax.scatter(
            current_portfolio["volatility"],
            current_portfolio["expected_return"],
            color=TEXT,
            s=85,
            label="Current Portfolio",
            zorder=3,
        )

    if min_variance_portfolio is not None:
        ax.scatter(
            min_variance_portfolio["volatility"],
            min_variance_portfolio["expected_return"],
            color=POSITIVE,
            s=90,
            label="Min Variance",
            zorder=3,
        )

    if max_sharpe_portfolio is not None:
        ax.scatter(
            max_sharpe_portfolio["volatility"],
            max_sharpe_portfolio["expected_return"],
            color=ACCENT_DARK,
            s=90,
            label="Max Sharpe",
            zorder=3,
        )

    _apply_axis_style(ax, "Efficient Frontier", "Volatility", "Expected Return")
    ax.legend(frameon=False, fontsize=10, labelcolor=MUTED, loc="best")
    fig.tight_layout()
    return plt


def plot_portfolio_allocation(asset_names, weights):
    """
    Plot portfolio allocation as a rounded donut chart.
    """
    fig, ax = _build_figure((6, 6))
    colors = [ACCENT_DARK, ACCENT, ACCENT_SOFT, "#d8dee8"]
    ax.pie(
        weights,
        labels=asset_names,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors[: len(weights)],
        wedgeprops={"width": 0.46, "edgecolor": FIG_BG, "linewidth": 2},
        textprops={"color": MUTED, "fontsize": 10},
    )
    ax.set_title("Portfolio Allocation", fontsize=14, fontweight="bold", color=TEXT, pad=14)
    ax.set_aspect("equal")
    fig.tight_layout()
    return plt

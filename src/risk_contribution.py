"""
Risk contribution module for the Financial Risk Engine.

This module decomposes portfolio volatility into asset-level
risk contributions.

Responsibilities:
- compute marginal risk contributions
- compute component risk contributions
- compute percentage risk contributions
- build a summary table
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio import calculate_portfolio_volatility


def calculate_marginal_risk_contribution(
    weights: np.ndarray,
    covariance_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculate marginal contribution to portfolio volatility.

    MRC_i = (Σw)_i / σ_p

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.
    covariance_matrix : np.ndarray
        Asset covariance matrix.

    Returns
    -------
    np.ndarray
        Marginal risk contribution for each asset.
    """

    weights = np.asarray(weights, dtype=float)
    covariance_matrix = np.asarray(covariance_matrix, dtype=float)

    portfolio_volatility = calculate_portfolio_volatility(weights, covariance_matrix)

    if portfolio_volatility == 0:
        return np.zeros_like(weights, dtype=float)

    marginal_risk_contribution = (covariance_matrix @ weights) / portfolio_volatility

    return marginal_risk_contribution


def calculate_component_risk_contribution(
    weights: np.ndarray,
    covariance_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculate component contribution to portfolio volatility.

    CRC_i = w_i * MRC_i

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.
    covariance_matrix : np.ndarray
        Asset covariance matrix.

    Returns
    -------
    np.ndarray
        Component risk contribution for each asset.
    """

    weights = np.asarray(weights, dtype=float)

    marginal_risk_contribution = calculate_marginal_risk_contribution(
        weights,
        covariance_matrix
    )

    component_risk_contribution = weights * marginal_risk_contribution

    return component_risk_contribution


def calculate_percentage_risk_contribution(
    weights: np.ndarray,
    covariance_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculate percentage contribution to portfolio volatility.

    PCR_i = CRC_i / σ_p

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.
    covariance_matrix : np.ndarray
        Asset covariance matrix.

    Returns
    -------
    np.ndarray
        Percentage risk contribution for each asset.
    """

    component_risk_contribution = calculate_component_risk_contribution(
        weights,
        covariance_matrix
    )

    total_volatility = component_risk_contribution.sum()

    if total_volatility == 0:
        return np.zeros_like(component_risk_contribution, dtype=float)

    percentage_risk_contribution = component_risk_contribution / total_volatility

    return percentage_risk_contribution


def build_risk_contribution_summary(
    weights: np.ndarray,
    covariance_matrix: np.ndarray,
    asset_names: list[str] | None = None
) -> pd.DataFrame:
    """
    Build a summary table of asset-level risk contributions.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.
    covariance_matrix : np.ndarray
        Asset covariance matrix.
    asset_names : list[str] | None, optional
        Asset names to label rows. If None, generic names are used.

    Returns
    -------
    pd.DataFrame
        DataFrame containing:
        - asset
        - weight
        - marginal_risk_contribution
        - component_risk_contribution
        - percentage_risk_contribution
    """

    weights = np.asarray(weights, dtype=float)

    num_assets = len(weights)

    if asset_names is None:
        asset_names = [f"Asset {i + 1}" for i in range(num_assets)]

    marginal_risk_contribution = calculate_marginal_risk_contribution(
        weights,
        covariance_matrix
    )

    component_risk_contribution = calculate_component_risk_contribution(
        weights,
        covariance_matrix
    )

    percentage_risk_contribution = calculate_percentage_risk_contribution(
        weights,
        covariance_matrix
    )

    summary = pd.DataFrame({
        "asset": asset_names,
        "weight": weights,
        "marginal_risk_contribution": marginal_risk_contribution,
        "component_risk_contribution": component_risk_contribution,
        "percentage_risk_contribution": percentage_risk_contribution,
    })

    summary = summary.sort_values(
        by="percentage_risk_contribution",
        ascending=False
    ).reset_index(drop=True)

    return summary
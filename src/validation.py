"""
Validation functions for the Financial Risk Engine.

This module checks that portfolio inputs are mathematically valid before
they are used in simulation or risk calculations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_weights(weights: np.ndarray, tolerance: float = 1e-6) -> None:
    """
    Validate portfolio weights.

    Rules:
    - weights must be numeric
    - weights must be one-dimensional
    - weights must sum to 1
    - weights should not contain NaN values

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.
    tolerance : float
        Numerical tolerance for sum-to-one check.

    Raises
    ------
    ValueError
        If weights are invalid.
    """
    weights = np.asarray(weights, dtype=float)

    if weights.ndim != 1:
        raise ValueError("Weights must be a one-dimensional array.")

    if len(weights) == 0:
        raise ValueError("Weights array cannot be empty.")

    if np.isnan(weights).any():
        raise ValueError("Weights contain NaN values.")

    if not np.isclose(weights.sum(), 1.0, atol=tolerance):
        raise ValueError(
            f"Weights must sum to 1.0. Current sum is {weights.sum():.6f}."
        )


def validate_mean_returns(mean_returns: np.ndarray) -> None:
    """
    Validate expected returns.

    Parameters
    ----------
    mean_returns : np.ndarray
        Expected return assumptions.

    Raises
    ------
    ValueError
        If expected returns are invalid.
    """
    mean_returns = np.asarray(mean_returns, dtype=float)

    if mean_returns.ndim != 1:
        raise ValueError("Mean returns must be a one-dimensional array.")

    if len(mean_returns) == 0:
        raise ValueError("Mean returns array cannot be empty.")

    if np.isnan(mean_returns).any():
        raise ValueError("Mean returns contain NaN values.")


def validate_volatilities(volatilities: np.ndarray) -> None:
    """
    Validate asset volatilities.

    Parameters
    ----------
    volatilities : np.ndarray
        Asset volatility assumptions.

    Raises
    ------
    ValueError
        If volatilities are invalid.
    """
    volatilities = np.asarray(volatilities, dtype=float)

    if volatilities.ndim != 1:
        raise ValueError("Volatilities must be a one-dimensional array.")

    if len(volatilities) == 0:
        raise ValueError("Volatilities array cannot be empty.")

    if np.isnan(volatilities).any():
        raise ValueError("Volatilities contain NaN values.")

    if (volatilities < 0).any():
        raise ValueError("Volatilities cannot be negative.")


def validate_correlation_matrix(correlation_matrix: np.ndarray) -> None:
    """
    Validate the correlation matrix.

    Rules:
    - matrix must be square
    - matrix must not contain NaN values
    - diagonal must be 1
    - values should be between -1 and 1
    - matrix should be symmetric

    Parameters
    ----------
    correlation_matrix : np.ndarray
        Correlation matrix.

    Raises
    ------
    ValueError
        If correlation matrix is invalid.
    """
    correlation_matrix = np.asarray(correlation_matrix, dtype=float)

    if correlation_matrix.ndim != 2:
        raise ValueError("Correlation matrix must be two-dimensional.")

    rows, cols = correlation_matrix.shape
    if rows != cols:
        raise ValueError(
            f"Correlation matrix must be square. Got shape {correlation_matrix.shape}."
        )

    if rows == 0:
        raise ValueError("Correlation matrix cannot be empty.")

    if np.isnan(correlation_matrix).any():
        raise ValueError("Correlation matrix contains NaN values.")

    if not np.allclose(correlation_matrix, correlation_matrix.T, atol=1e-8):
        raise ValueError("Correlation matrix must be symmetric.")

    diagonal = np.diag(correlation_matrix)
    if not np.allclose(diagonal, 1.0, atol=1e-8):
        raise ValueError("Correlation matrix diagonal elements must all equal 1.")

    if ((correlation_matrix < -1) | (correlation_matrix > 1)).any():
        raise ValueError("Correlation matrix values must be between -1 and 1.")


def validate_dimensions(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    volatilities: np.ndarray,
    correlation_matrix: np.ndarray,
) -> None:
    """
    Validate that all portfolio inputs have matching dimensions.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.
    mean_returns : np.ndarray
        Expected returns.
    volatilities : np.ndarray
        Asset volatilities.
    correlation_matrix : np.ndarray
        Asset correlation matrix.

    Raises
    ------
    ValueError
        If input dimensions do not align.
    """
    weights = np.asarray(weights, dtype=float)
    mean_returns = np.asarray(mean_returns, dtype=float)
    volatilities = np.asarray(volatilities, dtype=float)
    correlation_matrix = np.asarray(correlation_matrix, dtype=float)

    n_assets = len(weights)

    if len(mean_returns) != n_assets:
        raise ValueError(
            f"Mean returns length ({len(mean_returns)}) must match weights length ({n_assets})."
        )

    if len(volatilities) != n_assets:
        raise ValueError(
            f"Volatilities length ({len(volatilities)}) must match weights length ({n_assets})."
        )

    if correlation_matrix.shape != (n_assets, n_assets):
        raise ValueError(
            "Correlation matrix shape must match number of assets. "
            f"Expected ({n_assets}, {n_assets}), got {correlation_matrix.shape}."
        )


def validate_inputs(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    volatilities: np.ndarray,
    correlation_matrix: np.ndarray,
) -> None:
    """
    Run the full validation suite for core portfolio inputs.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.
    mean_returns : np.ndarray
        Expected returns.
    volatilities : np.ndarray
        Asset volatilities.
    correlation_matrix : np.ndarray
        Asset correlation matrix.

    Raises
    ------
    ValueError
        If any input fails validation.
    """
    validate_weights(weights)
    validate_mean_returns(mean_returns)
    validate_volatilities(volatilities)
    validate_correlation_matrix(correlation_matrix)
    validate_dimensions(weights, mean_returns, volatilities, correlation_matrix)


def validate_asset_data_frame(asset_data: pd.DataFrame) -> None:
    """
    Validate a portfolio assumptions DataFrame.

    Expected columns:
    - asset
    - expected_return
    - volatility
    - weight

    Parameters
    ----------
    asset_data : pd.DataFrame
        Asset assumptions table.

    Raises
    ------
    ValueError
        If DataFrame structure or contents are invalid.
    """
    required_columns = {"asset", "expected_return", "volatility", "weight"}

    if not isinstance(asset_data, pd.DataFrame):
        raise ValueError("asset_data must be a pandas DataFrame.")

    missing_columns = required_columns - set(asset_data.columns)
    if missing_columns:
        raise ValueError(
            f"Asset data is missing required columns: {sorted(missing_columns)}"
        )

    if asset_data.empty:
        raise ValueError("Asset data DataFrame cannot be empty.")

    if asset_data["asset"].isna().any():
        raise ValueError("Asset names cannot contain missing values.")

    if asset_data["asset"].duplicated().any():
        raise ValueError("Asset names must be unique.")

    validate_mean_returns(asset_data["expected_return"].to_numpy())
    validate_volatilities(asset_data["volatility"].to_numpy())
    validate_weights(asset_data["weight"].to_numpy())
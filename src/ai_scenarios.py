"""
AI scenario design module for the Financial Risk Engine.

This module provides a rule-based natural-language scenario interpreter that
maps macro narratives into structured portfolio assumption adjustments.

The implementation is intentionally transparent and deterministic so it can be
used without an external LLM. A future LLM-backed parser can replace the
rule-based parsing layer while keeping the downstream validation and
application functions unchanged.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from src.validation import (
    validate_correlation_matrix,
    validate_mean_returns,
    validate_volatilities,
)


RETURN_SHOCK_BOUNDS = (-0.25 / 252, 0.25 / 252)
VOL_MULTIPLIER_BOUNDS = (0.50, 2.50)
CORRELATION_BOUNDS = (-0.95, 0.95)


def _asset_category(asset_name: str) -> str:
    """
    Infer a broad asset category from the asset label.
    """
    normalized_name = asset_name.strip().lower()

    category_keywords = {
        "tech": ["tech", "technology", "software", "semiconductor", "ai", "growth"],
        "bonds": ["bond", "treasury", "fixed income", "rates", "duration", "credit"],
        "energy": ["energy", "oil", "gas", "commodity", "commodities"],
        "crypto": ["crypto", "bitcoin", "ethereum", "digital asset"],
        "equities": ["equity", "equities", "stock", "stocks", "market", "index"],
    }

    for category, keywords in category_keywords.items():
        if any(keyword in normalized_name for keyword in keywords):
            return category

    return "generic"


def _build_pairwise_adjustments(
    asset_names: list[str],
    shift: float,
    focus_categories: set[str] | None = None,
) -> list[dict]:
    """
    Build pairwise correlation adjustments for selected assets.
    """
    pairwise_adjustments = []
    focus_categories = focus_categories or set()

    asset_categories = {
        asset_name: _asset_category(asset_name)
        for asset_name in asset_names
    }

    for asset_1, asset_2 in combinations(asset_names, 2):
        if focus_categories:
            asset_1_in_focus = asset_categories[asset_1] in focus_categories
            asset_2_in_focus = asset_categories[asset_2] in focus_categories
            if not (asset_1_in_focus or asset_2_in_focus):
                continue

        pairwise_adjustments.append({
            "asset_1": asset_1,
            "asset_2": asset_2,
            "shift": shift,
        })

    return pairwise_adjustments


def parse_scenario_prompt(prompt: str, asset_names: list[str]) -> dict:
    """
    Parse a natural-language scenario prompt into structured scenario intent.

    Notes
    -----
    A future LLM-backed parser could replace this function and emit the same
    normalized dictionary schema used below.
    """
    normalized_prompt = (prompt or "").strip().lower()

    parsed = {
        "prompt": prompt,
        "regime_labels": [],
        "market_bias": "neutral",
        "return_shocks": {asset_name: 0.0 for asset_name in asset_names},
        "volatility_multipliers": {asset_name: 1.0 for asset_name in asset_names},
        "correlation_shift": 0.0,
        "pairwise_correlation_adjustments": [],
        "confidence": "low",
        "parser": "rule_based_v1",
    }

    if not normalized_prompt:
        parsed["regime_labels"] = ["baseline"]
        return parsed

    asset_categories = {
        asset_name: _asset_category(asset_name)
        for asset_name in asset_names
    }

    def add_return_shock(target_categories: set[str], shock: float) -> None:
        for asset_name, category in asset_categories.items():
            if category in target_categories:
                parsed["return_shocks"][asset_name] += shock

    def add_vol_multiplier(target_categories: set[str], multiplier: float) -> None:
        for asset_name, category in asset_categories.items():
            if category in target_categories:
                parsed["volatility_multipliers"][asset_name] *= multiplier

    def add_regime(label: str) -> None:
        if label not in parsed["regime_labels"]:
            parsed["regime_labels"].append(label)

    if any(term in normalized_prompt for term in ["recession", "hard landing", "contraction"]):
        add_regime("recession")
        parsed["market_bias"] = "risk_off"
        add_return_shock({"equities", "tech", "crypto", "generic"}, -0.08 / 252)
        add_return_shock({"bonds"}, 0.02 / 252)
        parsed["correlation_shift"] += 0.10
        add_vol_multiplier({"equities", "tech", "crypto", "generic"}, 1.20)

    if any(term in normalized_prompt for term in ["inflation", "sticky inflation", "rates remain elevated", "rate spike", "higher for longer"]):
        add_regime("inflation shock")
        add_return_shock({"bonds"}, -0.10 / 252)
        add_return_shock({"tech"}, -0.04 / 252)
        add_return_shock({"equities", "generic"}, -0.03 / 252)
        add_vol_multiplier({"bonds", "tech", "equities", "generic"}, 1.10)
        parsed["correlation_shift"] += 0.06

    if any(term in normalized_prompt for term in ["stagflation", "inflation shock combined with recession"]):
        add_regime("stagflation")
        parsed["market_bias"] = "risk_off"
        add_return_shock({"equities", "tech", "crypto", "generic"}, -0.05 / 252)
        add_return_shock({"bonds"}, -0.05 / 252)
        add_vol_multiplier({"equities", "tech", "crypto", "bonds", "generic"}, 1.15)
        parsed["correlation_shift"] += 0.08

    if any(term in normalized_prompt for term in ["tech selloff", "tech correction", "tech crash", "ai bubble bursts", "bubble bursts"]):
        add_regime("tech crash")
        parsed["market_bias"] = "risk_off"
        add_return_shock({"tech"}, -0.18 / 252)
        add_return_shock({"crypto"}, -0.10 / 252)
        add_return_shock({"equities"}, -0.05 / 252)
        add_vol_multiplier({"tech"}, 1.45)
        add_vol_multiplier({"crypto"}, 1.25)
        parsed["correlation_shift"] += 0.08
        parsed["pairwise_correlation_adjustments"].extend(
            _build_pairwise_adjustments(asset_names, 0.12, {"tech", "crypto", "equities"})
        )

    if any(term in normalized_prompt for term in ["energy shock", "oil spike", "commodity spike"]):
        add_regime("energy shock")
        add_return_shock({"energy"}, 0.12 / 252)
        add_return_shock({"equities", "generic"}, -0.03 / 252)
        add_return_shock({"bonds"}, -0.04 / 252)
        add_vol_multiplier({"energy"}, 1.20)
        add_vol_multiplier({"equities", "bonds", "generic"}, 1.10)
        parsed["correlation_shift"] += 0.07

    if any(term in normalized_prompt for term in ["risk-off", "flight to quality", "selloff", "severe", "weak bond performance"]):
        add_regime("risk-off")
        parsed["market_bias"] = "risk_off"
        add_return_shock({"equities", "tech", "crypto", "generic"}, -0.04 / 252)
        add_vol_multiplier({"equities", "tech", "crypto", "generic"}, 1.15)
        parsed["correlation_shift"] += 0.08

    if any(term in normalized_prompt for term in ["soft landing", "risk-on", "falling inflation", "disinflation", "strong tech performance", "lower volatility"]):
        add_regime("soft landing")
        parsed["market_bias"] = "risk_on"
        add_return_shock({"equities", "tech", "crypto", "generic"}, 0.04 / 252)
        add_return_shock({"tech"}, 0.06 / 252)
        add_return_shock({"bonds"}, 0.03 / 252)
        add_vol_multiplier({"equities", "tech", "crypto", "generic", "bonds"}, 0.90)
        parsed["correlation_shift"] -= 0.05

    if "strong tech" in normalized_prompt or "tech rally" in normalized_prompt:
        add_regime("risk-on")
        parsed["market_bias"] = "risk_on"
        add_return_shock({"tech"}, 0.08 / 252)
        add_vol_multiplier({"tech"}, 0.95)

    if "high cross-asset correlation" in normalized_prompt or "correlation spike" in normalized_prompt:
        parsed["correlation_shift"] += 0.15
        parsed["pairwise_correlation_adjustments"].extend(
            _build_pairwise_adjustments(asset_names, 0.15)
        )

    if not parsed["regime_labels"]:
        parsed["regime_labels"] = ["baseline"]
        parsed["confidence"] = "low"
    else:
        parsed["confidence"] = "medium" if len(parsed["regime_labels"]) == 1 else "high"

    return parsed


def validate_ai_scenario(adjustments: dict, asset_names: list[str]) -> dict:
    """
    Validate and clamp AI scenario adjustments to safe ranges.
    """
    if len(asset_names) == 0:
        raise ValueError("asset_names cannot be empty.")

    return_shocks = {
        asset_name: 0.0
        for asset_name in asset_names
    }
    volatility_multipliers = {
        asset_name: 1.0
        for asset_name in asset_names
    }

    for asset_name, value in adjustments.get("return_shocks", {}).items():
        if asset_name in return_shocks and np.isfinite(value):
            return_shocks[asset_name] = float(np.clip(value, *RETURN_SHOCK_BOUNDS))

    for asset_name, value in adjustments.get("volatility_multipliers", {}).items():
        if asset_name in volatility_multipliers and np.isfinite(value):
            volatility_multipliers[asset_name] = float(np.clip(value, *VOL_MULTIPLIER_BOUNDS))

    pairwise_correlation_adjustments = []
    seen_pairs = set()
    for pair in adjustments.get("pairwise_correlation_adjustments", []):
        asset_1 = pair.get("asset_1")
        asset_2 = pair.get("asset_2")
        shift = pair.get("shift", 0.0)

        if asset_1 not in asset_names or asset_2 not in asset_names or asset_1 == asset_2:
            continue

        if not np.isfinite(shift):
            continue

        pair_key = tuple(sorted((asset_1, asset_2)))
        if pair_key in seen_pairs:
            continue

        seen_pairs.add(pair_key)
        pairwise_correlation_adjustments.append({
            "asset_1": pair_key[0],
            "asset_2": pair_key[1],
            "shift": float(np.clip(shift, -0.50, 0.50)),
        })

    correlation_shift = adjustments.get("correlation_shift", 0.0)
    if not np.isfinite(correlation_shift):
        correlation_shift = 0.0

    validated = {
        "prompt": adjustments.get("prompt", ""),
        "regime_labels": list(dict.fromkeys(adjustments.get("regime_labels", ["baseline"]))),
        "market_bias": adjustments.get("market_bias", "neutral"),
        "return_shocks": return_shocks,
        "volatility_multipliers": volatility_multipliers,
        "correlation_shift": float(np.clip(correlation_shift, -0.50, 0.50)),
        "pairwise_correlation_adjustments": pairwise_correlation_adjustments,
        "confidence": adjustments.get("confidence", "low"),
        "parser": adjustments.get("parser", "rule_based_v1"),
    }

    if not validated["regime_labels"]:
        validated["regime_labels"] = ["baseline"]

    return validated


def _regularize_correlation_matrix(correlation_matrix: np.ndarray) -> np.ndarray:
    """
    Enforce a symmetric, bounded, positive-semidefinite correlation matrix.
    """
    correlation_matrix = np.asarray(correlation_matrix, dtype=float)
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2.0
    correlation_matrix = np.clip(correlation_matrix, *CORRELATION_BOUNDS)
    np.fill_diagonal(correlation_matrix, 1.0)

    eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
    eigenvalues = np.clip(eigenvalues, 1e-8, None)
    regularized = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    diagonal = np.sqrt(np.diag(regularized))
    diagonal[diagonal == 0.0] = 1.0
    regularized = regularized / np.outer(diagonal, diagonal)
    regularized = np.clip(regularized, *CORRELATION_BOUNDS)
    regularized = (regularized + regularized.T) / 2.0
    np.fill_diagonal(regularized, 1.0)

    return regularized


def apply_ai_scenario(
    base_returns: np.ndarray,
    base_vols: np.ndarray,
    base_corr: np.ndarray,
    adjustments: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply validated AI scenario adjustments to baseline assumptions.
    """
    base_returns = np.asarray(base_returns, dtype=float)
    base_vols = np.asarray(base_vols, dtype=float)
    base_corr = np.asarray(base_corr, dtype=float)

    adjusted_returns = base_returns.copy()
    adjusted_vols = base_vols.copy()
    adjusted_corr = base_corr.copy()

    asset_names = list(adjustments["return_shocks"].keys())
    asset_index = {
        asset_name: index
        for index, asset_name in enumerate(asset_names)
    }

    for asset_name, shock in adjustments["return_shocks"].items():
        adjusted_returns[asset_index[asset_name]] += shock

    for asset_name, multiplier in adjustments["volatility_multipliers"].items():
        adjusted_vols[asset_index[asset_name]] *= multiplier

    adjusted_vols = np.clip(adjusted_vols, 1e-8, None)

    if adjustments["correlation_shift"] != 0.0:
        for i in range(adjusted_corr.shape[0]):
            for j in range(i + 1, adjusted_corr.shape[1]):
                shifted_value = adjusted_corr[i, j] + adjustments["correlation_shift"]
                bounded_value = float(np.clip(shifted_value, *CORRELATION_BOUNDS))
                adjusted_corr[i, j] = bounded_value
                adjusted_corr[j, i] = bounded_value

    for pair in adjustments["pairwise_correlation_adjustments"]:
        i = asset_index[pair["asset_1"]]
        j = asset_index[pair["asset_2"]]
        shifted_value = adjusted_corr[i, j] + pair["shift"]
        bounded_value = float(np.clip(shifted_value, *CORRELATION_BOUNDS))
        adjusted_corr[i, j] = bounded_value
        adjusted_corr[j, i] = bounded_value

    adjusted_corr = _regularize_correlation_matrix(adjusted_corr)

    validate_mean_returns(adjusted_returns)
    validate_volatilities(adjusted_vols)
    validate_correlation_matrix(adjusted_corr)

    return adjusted_returns, adjusted_vols, adjusted_corr


def build_ai_scenario_adjustments(
    prompt: str,
    asset_names: list[str],
    base_returns: np.ndarray,
    base_vols: np.ndarray,
    base_corr: np.ndarray,
) -> dict:
    """
    Build validated AI scenario adjustments and adjusted assumptions.
    """
    validate_mean_returns(np.asarray(base_returns, dtype=float))
    validate_volatilities(np.asarray(base_vols, dtype=float))
    validate_correlation_matrix(np.asarray(base_corr, dtype=float))

    if len(asset_names) != len(base_returns) or len(asset_names) != len(base_vols):
        raise ValueError("asset_names length must match base return and volatility arrays.")

    if np.asarray(base_corr).shape != (len(asset_names), len(asset_names)):
        raise ValueError("base_corr shape must align with asset_names.")

    parsed = parse_scenario_prompt(prompt, asset_names)
    validated = validate_ai_scenario(parsed, asset_names)

    adjusted_returns, adjusted_vols, adjusted_corr = apply_ai_scenario(
        base_returns=base_returns,
        base_vols=base_vols,
        base_corr=base_corr,
        adjustments=validated,
    )

    return {
        **validated,
        "adjusted_mean_returns": adjusted_returns,
        "adjusted_volatilities": adjusted_vols,
        "adjusted_correlation_matrix": adjusted_corr,
    }


def summarize_ai_scenario(adjustments: dict) -> str:
    """
    Create a concise scenario summary for dashboard display.
    """
    regime_labels = adjustments.get("regime_labels", ["baseline"])
    correlation_shift = adjustments.get("correlation_shift", 0.0)
    non_neutral_assets = [
        asset_name
        for asset_name, shock in adjustments.get("return_shocks", {}).items()
        if abs(shock) > 1e-12
    ]

    summary_parts = [
        f"Regime: {', '.join(regime_labels)}.",
        f"Correlation shift: {correlation_shift:+.2f}.",
    ]

    if non_neutral_assets:
        summary_parts.append(
            "Affected assets: " + ", ".join(non_neutral_assets) + "."
        )
    else:
        summary_parts.append("No material asset-level shocks were detected.")

    return " ".join(summary_parts)

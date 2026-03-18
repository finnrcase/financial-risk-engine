"""
JrAnalyst.AI commentary engine for the Financial Risk Engine.

This module generates deterministic, grounded portfolio commentary using
existing analytics outputs from the risk engine. The goal is to provide a
professional interpretation layer without relying on unsupported freeform text.

The interface is intentionally modular so a future LLM-based generator can
augment or replace the rule-based narrative functions while keeping the input
schema stable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def classify_portfolio_profile(
    weights: np.ndarray,
    risk_summary: dict,
    current_portfolio_stats: dict,
) -> dict:
    """
    Classify the portfolio's overall risk posture.
    """
    weights = np.asarray(weights, dtype=float)

    leverage_proxy = float(np.max(weights)) if len(weights) > 0 else 0.0
    volatility = float(risk_summary["volatility"])
    probability_of_loss = float(risk_summary["probability_of_loss"])
    sharpe_ratio = float(current_portfolio_stats["sharpe_ratio"])

    if volatility >= 0.30 or probability_of_loss >= 0.55:
        posture = "aggressive"
    elif volatility >= 0.18 or probability_of_loss >= 0.45:
        posture = "moderate"
    else:
        posture = "defensive"

    if sharpe_ratio >= 0.80:
        efficiency = "strong"
    elif sharpe_ratio >= 0.35:
        efficiency = "balanced"
    else:
        efficiency = "weak"

    if leverage_proxy >= 0.45:
        concentration = "concentrated"
    elif leverage_proxy >= 0.30:
        concentration = "moderately concentrated"
    else:
        concentration = "well distributed"

    commentary = (
        f"The portfolio currently screens as a {posture} risk posture with "
        f"{efficiency} risk-adjusted efficiency. Position sizing is "
        f"{concentration}, with the largest allocation at {leverage_proxy:.1%}."
    )

    return {
        "posture": posture,
        "efficiency": efficiency,
        "concentration": concentration,
        "commentary": commentary,
    }


def analyze_tail_risk(risk_summary: dict) -> dict:
    """
    Interpret tail risk metrics from the simulation output.
    """
    var_95 = abs(float(risk_summary["var_95"]))
    es_95 = abs(float(risk_summary["es_95"]))
    var_99 = abs(float(risk_summary["var_99"]))
    probability_of_loss = float(risk_summary["probability_of_loss"])
    max_drawdown = abs(float(risk_summary["max_drawdown"]))

    severity_score = 0
    if var_95 >= 0.12:
        severity_score += 1
    if es_95 >= 0.15:
        severity_score += 1
    if probability_of_loss >= 0.50:
        severity_score += 1
    if max_drawdown >= 0.30:
        severity_score += 1

    if severity_score >= 3:
        level = "elevated"
        commentary = (
            f"Tail-risk metrics are elevated. The portfolio shows 95% VaR of "
            f"{var_95:.1%}, expected shortfall of {es_95:.1%}, and maximum "
            f"drawdown of {max_drawdown:.1%}, indicating a meaningful adverse-path "
            "risk profile."
        )
    elif severity_score >= 1:
        level = "moderate"
        commentary = (
            f"Tail risk appears manageable but not benign. Downside measures show "
            f"95% VaR of {var_95:.1%}, expected shortfall of {es_95:.1%}, and "
            f"probability of loss of {probability_of_loss:.1%}."
        )
    else:
        level = "contained"
        commentary = (
            f"Tail risk is relatively contained. Loss metrics remain moderate, with "
            f"95% VaR of {var_95:.1%} and maximum drawdown of {max_drawdown:.1%}."
        )

    return {
        "level": level,
        "commentary": commentary,
    }


def analyze_diversification(risk_contribution_summary: pd.DataFrame) -> dict:
    """
    Assess diversification quality from asset-level risk contributions.
    """
    summary = risk_contribution_summary.copy()
    top_contribution = float(summary["percentage_risk_contribution"].iloc[0])
    top_two_contribution = float(summary["percentage_risk_contribution"].head(2).sum())

    if top_contribution >= 0.45 or top_two_contribution >= 0.75:
        diversification = "weak"
        commentary = (
            f"Diversification appears weak. The largest risk contributor accounts "
            f"for {top_contribution:.1%} of portfolio risk, and the top two assets "
            f"account for {top_two_contribution:.1%}."
        )
    elif top_contribution >= 0.30 or top_two_contribution >= 0.60:
        diversification = "mixed"
        commentary = (
            f"Diversification is present but uneven. The top risk contributor "
            f"accounts for {top_contribution:.1%} of total risk, suggesting some "
            "dependence on a limited set of exposures."
        )
    else:
        diversification = "strong"
        commentary = (
            f"Risk is reasonably diversified. No single asset dominates the risk "
            f"budget, and the top two contributors account for {top_two_contribution:.1%} "
            "of total portfolio risk."
        )

    return {
        "diversification": diversification,
        "top_contribution": top_contribution,
        "top_two_contribution": top_two_contribution,
        "commentary": commentary,
    }


def analyze_risk_concentrations(risk_contribution_summary: pd.DataFrame) -> dict:
    """
    Identify the largest downside drivers in the current portfolio.
    """
    summary = risk_contribution_summary.sort_values(
        by="percentage_risk_contribution",
        ascending=False,
    ).reset_index(drop=True)

    lead_asset = str(summary.loc[0, "asset"])
    lead_risk = float(summary.loc[0, "percentage_risk_contribution"])
    secondary_asset = str(summary.loc[1, "asset"]) if len(summary) > 1 else lead_asset
    secondary_risk = float(summary.loc[1, "percentage_risk_contribution"]) if len(summary) > 1 else lead_risk

    commentary = (
        f"The main downside driver is {lead_asset}, which contributes {lead_risk:.1%} "
        f"of total portfolio risk. {secondary_asset} is the next-largest contributor "
        f"at {secondary_risk:.1%}, indicating where stress would likely concentrate first."
    )

    return {
        "lead_asset": lead_asset,
        "lead_risk": lead_risk,
        "secondary_asset": secondary_asset,
        "secondary_risk": secondary_risk,
        "commentary": commentary,
    }


def analyze_scenario_vulnerability(
    scenario_comparison_df: pd.DataFrame,
) -> dict:
    """
    Evaluate the portfolio's most material scenario vulnerability.
    """
    if scenario_comparison_df.empty or "baseline" not in scenario_comparison_df["scenario"].values:
        return {
            "worst_scenario": "insufficient_data",
            "commentary": (
                "Scenario vulnerability commentary is limited because a baseline "
                "comparison was not available."
            ),
        }

    comparison = scenario_comparison_df.set_index("scenario").copy()
    baseline = comparison.loc["baseline"]
    stressed = comparison.drop(index="baseline", errors="ignore")

    if stressed.empty:
        return {
            "worst_scenario": "baseline_only",
            "commentary": (
                "Scenario vulnerability assessment is limited because only the "
                "baseline scenario is currently selected."
            ),
        }

    stressed["drawdown_delta"] = stressed["max_drawdown"] - baseline["max_drawdown"]
    stressed["var_delta"] = stressed["var_95"] - baseline["var_95"]
    stressed["sharpe_delta"] = stressed["sharpe_ratio"] - baseline["sharpe_ratio"]
    stressed["stress_score"] = (
        stressed["drawdown_delta"].abs()
        + stressed["var_delta"].abs()
        + stressed["sharpe_delta"].abs() * 0.10
    )

    worst_scenario = str(stressed["stress_score"].idxmax())
    worst_row = stressed.loc[worst_scenario]

    commentary = (
        f"The most material scenario vulnerability appears under {worst_scenario}. "
        f"Relative to baseline, 95% VaR moves by {worst_row['var_delta']:+.1%}, "
        f"maximum drawdown shifts by {worst_row['drawdown_delta']:+.1%}, and the "
        f"Sharpe ratio changes by {worst_row['sharpe_delta']:+.2f}."
    )

    return {
        "worst_scenario": worst_scenario,
        "commentary": commentary,
    }


def compare_to_optimized_portfolios(
    current_portfolio_stats: dict,
    min_variance_portfolio: dict,
    max_sharpe_portfolio: dict,
) -> dict:
    """
    Compare the current portfolio to optimization outputs.
    """
    current_vol = float(current_portfolio_stats["volatility"])
    current_sharpe = float(current_portfolio_stats["sharpe_ratio"])
    min_variance_vol = float(min_variance_portfolio["volatility"])
    max_sharpe = float(max_sharpe_portfolio["sharpe_ratio"])

    sharpe_gap = max_sharpe - current_sharpe
    volatility_gap = current_vol - min_variance_vol

    if sharpe_gap >= 0.30:
        efficiency_view = "materially less efficient"
    elif sharpe_gap >= 0.10:
        efficiency_view = "somewhat less efficient"
    else:
        efficiency_view = "broadly aligned"

    commentary = (
        f"Relative to the optimized set, the current allocation looks "
        f"{efficiency_view} on a risk-adjusted basis. The Sharpe ratio gap versus "
        f"the max-Sharpe portfolio is {sharpe_gap:+.2f}, while current volatility "
        f"is {volatility_gap:+.1%} above the minimum-variance portfolio."
    )

    return {
        "sharpe_gap": sharpe_gap,
        "volatility_gap": volatility_gap,
        "efficiency_view": efficiency_view,
        "commentary": commentary,
    }


def generate_jr_analyst_commentary(
    weights: np.ndarray,
    asset_names: list[str],
    risk_summary: dict,
    risk_contribution_summary: pd.DataFrame,
    scenario_comparison_df: pd.DataFrame,
    current_portfolio_stats: dict,
    min_variance_portfolio: dict,
    max_sharpe_portfolio: dict,
) -> dict:
    """
    Generate full JrAnalyst.AI commentary from the engine's existing outputs.
    """
    _ = asset_names

    portfolio_profile = classify_portfolio_profile(
        weights=weights,
        risk_summary=risk_summary,
        current_portfolio_stats=current_portfolio_stats,
    )
    tail_risk = analyze_tail_risk(risk_summary)
    diversification = analyze_diversification(risk_contribution_summary)
    concentrations = analyze_risk_concentrations(risk_contribution_summary)
    scenario_vulnerability = analyze_scenario_vulnerability(scenario_comparison_df)
    optimization_insight = compare_to_optimized_portfolios(
        current_portfolio_stats=current_portfolio_stats,
        min_variance_portfolio=min_variance_portfolio,
        max_sharpe_portfolio=max_sharpe_portfolio,
    )

    bottom_line = (
        f"Overall, the portfolio reflects a {portfolio_profile['posture']} risk stance "
        f"with {tail_risk['level']} tail-risk characteristics. Diversification is "
        f"{diversification['diversification']}, and current efficiency is "
        f"{optimization_insight['efficiency_view']} relative to the optimized set."
    )

    key_takeaways = [
        concentrations["commentary"],
        scenario_vulnerability["commentary"],
        optimization_insight["commentary"],
    ]

    return {
        "product_name": "JrAnalyst.AI",
        "sections": {
            "Portfolio Profile": portfolio_profile["commentary"],
            "Tail Risk Assessment": tail_risk["commentary"],
            "Diversification / Concentration": (
                diversification["commentary"] + " " + concentrations["commentary"]
            ),
            "Scenario Vulnerability": scenario_vulnerability["commentary"],
            "Optimization Insight": optimization_insight["commentary"],
            "Bottom-Line Analyst View": bottom_line,
        },
        "key_takeaways": key_takeaways,
        "signals": {
            "portfolio_profile": portfolio_profile,
            "tail_risk": tail_risk,
            "diversification": diversification,
            "risk_concentrations": concentrations,
            "scenario_vulnerability": scenario_vulnerability,
            "optimization_insight": optimization_insight,
        },
    }

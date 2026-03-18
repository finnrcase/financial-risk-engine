import numpy as np
import pandas as pd

from src.jr_analyst import (
    analyze_diversification,
    compare_to_optimized_portfolios,
    generate_jr_analyst_commentary,
)


def _sample_inputs():
    weights = np.array([0.40, 0.30, 0.20, 0.10])
    asset_names = ["Equities", "Bonds", "Tech", "Crypto"]
    risk_summary = {
        "mean_return": 0.07,
        "volatility": 0.24,
        "var_95": -0.14,
        "var_99": -0.22,
        "es_95": -0.18,
        "probability_of_loss": 0.49,
        "max_drawdown": -0.31,
        "sharpe_ratio": 0.38,
    }
    risk_contribution_summary = pd.DataFrame({
        "asset": asset_names,
        "weight": weights,
        "marginal_risk_contribution": [0.08, 0.02, 0.10, 0.12],
        "component_risk_contribution": [0.032, 0.006, 0.020, 0.012],
        "percentage_risk_contribution": [0.46, 0.09, 0.28, 0.17],
    }).sort_values("percentage_risk_contribution", ascending=False).reset_index(drop=True)
    scenario_comparison_df = pd.DataFrame([
        {
            "scenario": "baseline",
            "mean_return": 0.07,
            "volatility": 0.24,
            "var_95": -0.14,
            "var_99": -0.22,
            "es_95": -0.18,
            "probability_of_loss": 0.49,
            "max_drawdown": -0.31,
            "sharpe_ratio": 0.38,
        },
        {
            "scenario": "market_crash",
            "mean_return": -0.12,
            "volatility": 0.39,
            "var_95": -0.24,
            "var_99": -0.33,
            "es_95": -0.29,
            "probability_of_loss": 0.68,
            "max_drawdown": -0.47,
            "sharpe_ratio": -0.18,
        },
        {
            "scenario": "inflation_shock",
            "mean_return": -0.03,
            "volatility": 0.30,
            "var_95": -0.19,
            "var_99": -0.27,
            "es_95": -0.23,
            "probability_of_loss": 0.58,
            "max_drawdown": -0.40,
            "sharpe_ratio": 0.02,
        },
    ])
    current_portfolio_stats = {
        "expected_return": 0.07,
        "volatility": 0.24,
        "sharpe_ratio": 0.38,
    }
    min_variance_portfolio = {
        "weights": np.array([0.20, 0.55, 0.15, 0.10]),
        "expected_return": 0.05,
        "volatility": 0.16,
        "sharpe_ratio": 0.31,
    }
    max_sharpe_portfolio = {
        "weights": np.array([0.35, 0.25, 0.25, 0.15]),
        "expected_return": 0.09,
        "volatility": 0.21,
        "sharpe_ratio": 0.74,
    }
    return {
        "weights": weights,
        "asset_names": asset_names,
        "risk_summary": risk_summary,
        "risk_contribution_summary": risk_contribution_summary,
        "scenario_comparison_df": scenario_comparison_df,
        "current_portfolio_stats": current_portfolio_stats,
        "min_variance_portfolio": min_variance_portfolio,
        "max_sharpe_portfolio": max_sharpe_portfolio,
    }


def test_generate_jr_analyst_commentary_returns_expected_sections():
    inputs = _sample_inputs()

    commentary = generate_jr_analyst_commentary(**inputs)

    expected_sections = [
        "Portfolio Profile",
        "Tail Risk Assessment",
        "Diversification / Concentration",
        "Scenario Vulnerability",
        "Optimization Insight",
        "Bottom-Line Analyst View",
    ]

    assert commentary["product_name"] == "JrAnalyst.AI"
    assert list(commentary["sections"].keys()) == expected_sections
    assert len(commentary["key_takeaways"]) == 3


def test_commentary_changes_when_metrics_change_materially():
    inputs = _sample_inputs()
    conservative_inputs = _sample_inputs()

    conservative_inputs["risk_summary"] = {
        **conservative_inputs["risk_summary"],
        "volatility": 0.11,
        "var_95": -0.05,
        "es_95": -0.07,
        "probability_of_loss": 0.32,
        "max_drawdown": -0.12,
    }
    conservative_inputs["current_portfolio_stats"] = {
        **conservative_inputs["current_portfolio_stats"],
        "volatility": 0.11,
        "sharpe_ratio": 0.72,
    }

    high_risk_commentary = generate_jr_analyst_commentary(**inputs)
    conservative_commentary = generate_jr_analyst_commentary(**conservative_inputs)

    assert (
        high_risk_commentary["sections"]["Tail Risk Assessment"]
        != conservative_commentary["sections"]["Tail Risk Assessment"]
    )


def test_concentration_logic_detects_weak_diversification():
    inputs = _sample_inputs()

    diversification = analyze_diversification(inputs["risk_contribution_summary"])

    assert diversification["diversification"] == "weak"
    assert "46.0%" in diversification["commentary"]


def test_optimization_comparison_highlights_efficiency_gap():
    inputs = _sample_inputs()

    comparison = compare_to_optimized_portfolios(
        current_portfolio_stats=inputs["current_portfolio_stats"],
        min_variance_portfolio=inputs["min_variance_portfolio"],
        max_sharpe_portfolio=inputs["max_sharpe_portfolio"],
    )

    assert comparison["efficiency_view"] == "materially less efficient"
    assert comparison["sharpe_gap"] > 0.30


def test_outputs_remain_grounded_and_reference_real_scenarios():
    inputs = _sample_inputs()

    commentary = generate_jr_analyst_commentary(**inputs)

    scenario_text = commentary["sections"]["Scenario Vulnerability"]
    optimization_text = commentary["sections"]["Optimization Insight"]

    assert "market_crash" in scenario_text
    assert "Sharpe ratio gap" in optimization_text

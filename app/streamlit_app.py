"""
Streamlit dashboard for the Financial Risk Engine.

This app provides an interactive interface for:
- portfolio configuration
- Monte Carlo risk simulation
- stress scenario analysis
- risk contribution analysis
- portfolio optimization / efficient frontier
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.validation import validate_inputs
from src.portfolio import build_covariance_matrix
from src.simulation import run_monte_carlo_engine
from src.risk_metrics import build_risk_summary
from src.ai_scenarios import (
    build_ai_scenario_adjustments,
    summarize_ai_scenario,
)
from src.scenarios import get_available_scenarios, apply_scenario
from src.risk_contribution import build_risk_contribution_summary
from src.scenario_analysis import (
    run_multi_scenario_comparison,
    build_scenario_comparison_table,
)
from src.optimization import (
    calculate_portfolio_statistics,
    optimize_min_variance,
    optimize_max_sharpe,
    generate_efficient_frontier,
)
from src.visualization import (
    plot_return_distribution,
    plot_portfolio_paths,
    plot_drawdown_distribution,
    plot_risk_contributions,
    plot_scenario_metric_comparison,
    plot_efficient_frontier,
    plot_portfolio_allocation,
)


TRADING_DAYS_PER_YEAR = 252


def format_pct(value: float) -> str:
    """
    Format decimal as percentage string.
    """
    return f"{value:.2%}"


def format_num(value: float) -> str:
    """
    Format numeric value for display.
    """
    return f"{value:.4f}"


def annual_return_to_daily(annual_return: np.ndarray) -> np.ndarray:
    """
    Convert annual expected returns to daily expected returns.
    """
    return np.asarray(annual_return, dtype=float) / TRADING_DAYS_PER_YEAR


def annual_vol_to_daily(annual_vol: np.ndarray) -> np.ndarray:
    """
    Convert annual volatility to daily volatility.
    """
    return np.asarray(annual_vol, dtype=float) / np.sqrt(TRADING_DAYS_PER_YEAR)


def daily_return_to_annual(daily_return: np.ndarray) -> np.ndarray:
    """
    Convert daily expected returns to annual expected returns.
    """
    return np.asarray(daily_return, dtype=float) * TRADING_DAYS_PER_YEAR


def daily_vol_to_annual(daily_vol: np.ndarray) -> np.ndarray:
    """
    Convert daily volatility to annual volatility.
    """
    return np.asarray(daily_vol, dtype=float) * np.sqrt(TRADING_DAYS_PER_YEAR)


def is_positive_semidefinite(matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
    """
    Check whether a matrix is positive semidefinite.
    """
    eigenvalues = np.linalg.eigvalsh(matrix)
    return bool(np.all(eigenvalues >= -tolerance))


def render_matplotlib_chart(plot_function, *args, **kwargs) -> None:
    """
    Render matplotlib chart returned from a visualization helper.
    """
    plt.close("all")
    plot_module = plot_function(*args, **kwargs)
    figure = plot_module.gcf()
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)


def build_correlation_matrix(
    corr_eq_bonds: float,
    corr_eq_tech: float,
    corr_eq_crypto: float,
    corr_bonds_tech: float,
    corr_bonds_crypto: float,
    corr_tech_crypto: float,
) -> np.ndarray:
    """
    Build a 4x4 symmetric correlation matrix from pairwise correlations.
    """
    correlation_matrix = np.array([
        [1.0,            corr_eq_bonds,    corr_eq_tech,     corr_eq_crypto],
        [corr_eq_bonds,  1.0,              corr_bonds_tech,  corr_bonds_crypto],
        [corr_eq_tech,   corr_bonds_tech,  1.0,              corr_tech_crypto],
        [corr_eq_crypto, corr_bonds_crypto, corr_tech_crypto, 1.0],
    ])

    return correlation_matrix


st.set_page_config(
    page_title="Financial Risk Engine",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main > div {
        padding-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Financial Risk Engine")
st.caption(
    "Institutional Portfolio Risk Simulator — Monte Carlo simulation, "
    "stress testing, risk attribution, and portfolio optimization."
)

st.markdown(
    """
    This dashboard simulates portfolio risk under uncertainty using correlated
    Monte Carlo return paths, scenario stress testing, asset-level risk
    attribution, and mean-variance optimization.
    """
)

# -----------------------------------------------------------------------------
# Sidebar inputs
# -----------------------------------------------------------------------------

st.sidebar.header("Portfolio Configuration")

default_asset_names = ["Equities", "Bonds", "Tech", "Crypto"]
default_weights = [0.40, 0.30, 0.20, 0.10]
default_returns = [0.08, 0.04, 0.10, 0.18]
default_vols = [0.16, 0.06, 0.22, 0.55]

with st.sidebar.expander("Asset Inputs", expanded=True):
    asset_1 = st.text_input("Asset 1 Name", value=default_asset_names[0])
    weight_1 = st.number_input("Asset 1 Weight", min_value=0.0, max_value=1.0, value=default_weights[0], step=0.01)
    ret_1 = st.number_input("Asset 1 Expected Return (annual)", value=default_returns[0], step=0.01, format="%.4f")
    vol_1 = st.number_input("Asset 1 Volatility (annual)", min_value=0.0, value=default_vols[0], step=0.01, format="%.4f")

    st.markdown("---")

    asset_2 = st.text_input("Asset 2 Name", value=default_asset_names[1])
    weight_2 = st.number_input("Asset 2 Weight", min_value=0.0, max_value=1.0, value=default_weights[1], step=0.01)
    ret_2 = st.number_input("Asset 2 Expected Return (annual)", value=default_returns[1], step=0.01, format="%.4f")
    vol_2 = st.number_input("Asset 2 Volatility (annual)", min_value=0.0, value=default_vols[1], step=0.01, format="%.4f")

    st.markdown("---")

    asset_3 = st.text_input("Asset 3 Name", value=default_asset_names[2])
    weight_3 = st.number_input("Asset 3 Weight", min_value=0.0, max_value=1.0, value=default_weights[2], step=0.01)
    ret_3 = st.number_input("Asset 3 Expected Return (annual)", value=default_returns[2], step=0.01, format="%.4f")
    vol_3 = st.number_input("Asset 3 Volatility (annual)", min_value=0.0, value=default_vols[2], step=0.01, format="%.4f")

    st.markdown("---")

    asset_4 = st.text_input("Asset 4 Name", value=default_asset_names[3])
    weight_4 = st.number_input("Asset 4 Weight", min_value=0.0, max_value=1.0, value=default_weights[3], step=0.01)
    ret_4 = st.number_input("Asset 4 Expected Return (annual)", value=default_returns[3], step=0.01, format="%.4f")
    vol_4 = st.number_input("Asset 4 Volatility (annual)", min_value=0.0, value=default_vols[3], step=0.01, format="%.4f")

with st.sidebar.expander("Correlation Assumptions", expanded=True):
    corr_eq_bonds = st.slider(f"{asset_1} / {asset_2}", min_value=-0.95, max_value=0.95, value=0.20, step=0.05)
    corr_eq_tech = st.slider(f"{asset_1} / {asset_3}", min_value=-0.95, max_value=0.95, value=0.75, step=0.05)
    corr_eq_crypto = st.slider(f"{asset_1} / {asset_4}", min_value=-0.95, max_value=0.95, value=0.35, step=0.05)
    corr_bonds_tech = st.slider(f"{asset_2} / {asset_3}", min_value=-0.95, max_value=0.95, value=0.15, step=0.05)
    corr_bonds_crypto = st.slider(f"{asset_2} / {asset_4}", min_value=-0.95, max_value=0.95, value=0.05, step=0.05)
    corr_tech_crypto = st.slider(f"{asset_3} / {asset_4}", min_value=-0.95, max_value=0.95, value=0.45, step=0.05)

st.sidebar.header("Simulation Settings")

num_simulations = st.sidebar.slider(
    "Number of Simulations",
    min_value=500,
    max_value=10000,
    value=3000,
    step=500,
)

time_horizon_days = st.sidebar.slider(
    "Investment Horizon (trading days)",
    min_value=21,
    max_value=756,
    value=252,
    step=21,
)

initial_portfolio_value = st.sidebar.number_input(
    "Initial Portfolio Value",
    min_value=0.01,
    value=1.0,
    step=0.10,
    format="%.2f",
)

risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate (annual)",
    min_value=0.0,
    max_value=0.20,
    value=0.02,
    step=0.005,
    format="%.4f",
)

st.sidebar.header("Scenario Controls")

available_scenarios = get_available_scenarios()
selected_scenario = st.sidebar.selectbox(
    "Primary Scenario",
    options=list(available_scenarios.keys()),
    index=0,
)

selected_comparison_scenarios = st.sidebar.multiselect(
    "Scenarios for Comparison",
    options=list(available_scenarios.keys()),
    default=["baseline", "market_crash", "inflation_shock"],
)

comparison_metric = st.sidebar.selectbox(
    "Scenario Comparison Metric",
    options=[
        "mean_return",
        "volatility",
        "var_95",
        "var_99",
        "es_95",
        "probability_of_loss",
        "max_drawdown",
        "sharpe_ratio",
    ],
    index=2,
)

st.sidebar.header("Optimization Settings")

frontier_points = st.sidebar.slider(
    "Efficient Frontier Points",
    min_value=10,
    max_value=60,
    value=25,
    step=5,
)

# -----------------------------------------------------------------------------
# Input preparation
# -----------------------------------------------------------------------------

asset_names = [asset_1, asset_2, asset_3, asset_4]
weights = np.array([weight_1, weight_2, weight_3, weight_4], dtype=float)
annual_mean_returns = np.array([ret_1, ret_2, ret_3, ret_4], dtype=float)
annual_volatilities = np.array([vol_1, vol_2, vol_3, vol_4], dtype=float)

correlation_matrix = build_correlation_matrix(
    corr_eq_bonds=corr_eq_bonds,
    corr_eq_tech=corr_eq_tech,
    corr_eq_crypto=corr_eq_crypto,
    corr_bonds_tech=corr_bonds_tech,
    corr_bonds_crypto=corr_bonds_crypto,
    corr_tech_crypto=corr_tech_crypto,
)

if np.isclose(weights.sum(), 0.0):
    st.error("Portfolio weights sum to zero. Please enter at least one positive weight.")
    st.stop()

if not np.isclose(weights.sum(), 1.0):
    st.warning(
        f"Portfolio weights currently sum to {weights.sum():.2f}. "
        "Weights have been normalized automatically for analysis."
    )
    weights = weights / weights.sum()

daily_mean_returns = annual_return_to_daily(annual_mean_returns)
daily_volatilities = annual_vol_to_daily(annual_volatilities)

baseline_covariance_daily = build_covariance_matrix(daily_volatilities, correlation_matrix)

if not is_positive_semidefinite(correlation_matrix):
    st.error(
        "The correlation matrix is not positive semidefinite. "
        "Please adjust the correlation assumptions."
    )
    st.stop()

if not is_positive_semidefinite(baseline_covariance_daily):
    st.error(
        "The implied covariance matrix is not positive semidefinite. "
        "Please adjust the correlation or volatility assumptions."
    )
    st.stop()

try:
    validate_inputs(
        weights=weights,
        mean_returns=daily_mean_returns,
        volatilities=daily_volatilities,
        correlation_matrix=correlation_matrix,
    )
except Exception as exc:
    st.error(f"Input validation error: {exc}")
    st.stop()

# -----------------------------------------------------------------------------
# Main scenario simulation
# -----------------------------------------------------------------------------

try:
    adjusted_inputs = apply_scenario(
        mean_returns=daily_mean_returns,
        volatilities=daily_volatilities,
        correlation_matrix=correlation_matrix,
        scenario_name=selected_scenario,
    )

    adjusted_covariance_daily = build_covariance_matrix(
        adjusted_inputs["volatilities"],
        adjusted_inputs["correlation_matrix"],
    )

    simulation_results = run_monte_carlo_engine(
        weights=weights,
        mean_returns=adjusted_inputs["mean_returns"],
        covariance_matrix=adjusted_covariance_daily,
        num_simulations=num_simulations,
        time_horizon_days=time_horizon_days,
        initial_portfolio_value=initial_portfolio_value,
    )

    risk_summary = build_risk_summary(
        terminal_returns=simulation_results["terminal_returns"],
        portfolio_paths=simulation_results["portfolio_paths"],
    )

except Exception as exc:
    st.error(f"Simulation error: {exc}")
    st.stop()

# -----------------------------------------------------------------------------
# Risk contribution analysis
# -----------------------------------------------------------------------------

risk_contribution_summary = build_risk_contribution_summary(
    weights=weights,
    covariance_matrix=adjusted_covariance_daily,
    asset_names=asset_names,
)

risk_contribution_display = risk_contribution_summary.copy()
risk_contribution_display["weight"] = risk_contribution_display["weight"].map(format_pct)
risk_contribution_display["marginal_risk_contribution"] = risk_contribution_display["marginal_risk_contribution"].map(format_num)
risk_contribution_display["component_risk_contribution"] = risk_contribution_display["component_risk_contribution"].map(format_num)
risk_contribution_display["percentage_risk_contribution"] = risk_contribution_display["percentage_risk_contribution"].map(format_pct)

# -----------------------------------------------------------------------------
# Scenario comparison
# -----------------------------------------------------------------------------

scenario_comparison_df = pd.DataFrame()

if selected_comparison_scenarios:
    try:
        comparison_results = run_multi_scenario_comparison(
            weights=weights,
            mean_returns=daily_mean_returns,
            volatilities=daily_volatilities,
            correlation_matrix=correlation_matrix,
            scenario_names=selected_comparison_scenarios,
            num_simulations=num_simulations,
            time_horizon_days=time_horizon_days,
            initial_portfolio_value=initial_portfolio_value,
        )
        scenario_comparison_df = build_scenario_comparison_table(comparison_results)
    except Exception as exc:
        st.error(f"Scenario comparison error: {exc}")

scenario_comparison_display = pd.DataFrame()
if not scenario_comparison_df.empty:
    scenario_comparison_display = scenario_comparison_df.copy()
    for column in [
        "mean_return",
        "volatility",
        "var_95",
        "var_99",
        "es_95",
        "probability_of_loss",
        "max_drawdown",
    ]:
        scenario_comparison_display[column] = scenario_comparison_display[column].map(format_pct)

    scenario_comparison_display["sharpe_ratio"] = scenario_comparison_display["sharpe_ratio"].map(format_num)

# -----------------------------------------------------------------------------
# Optimization analysis
# -----------------------------------------------------------------------------

adjusted_annual_mean_returns = daily_return_to_annual(adjusted_inputs["mean_returns"])
adjusted_annual_volatilities = daily_vol_to_annual(adjusted_inputs["volatilities"])
adjusted_annual_covariance = build_covariance_matrix(
    adjusted_annual_volatilities,
    adjusted_inputs["correlation_matrix"],
)

current_portfolio_stats = calculate_portfolio_statistics(
    weights=weights,
    mean_returns=adjusted_annual_mean_returns,
    covariance_matrix=adjusted_annual_covariance,
    risk_free_rate=risk_free_rate,
)

min_variance_portfolio = optimize_min_variance(
    mean_returns=adjusted_annual_mean_returns,
    covariance_matrix=adjusted_annual_covariance,
)

max_sharpe_portfolio = optimize_max_sharpe(
    mean_returns=adjusted_annual_mean_returns,
    covariance_matrix=adjusted_annual_covariance,
    risk_free_rate=risk_free_rate,
)

frontier_df = generate_efficient_frontier(
    mean_returns=adjusted_annual_mean_returns,
    covariance_matrix=adjusted_annual_covariance,
    n_points=frontier_points,
    risk_free_rate=risk_free_rate,
)

optimization_weights_df = pd.DataFrame({
    "asset": asset_names,
    "current_weight": weights,
    "min_variance_weight": min_variance_portfolio["weights"],
    "max_sharpe_weight": max_sharpe_portfolio["weights"],
})

optimization_weights_display = optimization_weights_df.copy()
for column in ["current_weight", "min_variance_weight", "max_sharpe_weight"]:
    optimization_weights_display[column] = optimization_weights_display[column].map(format_pct)

# -----------------------------------------------------------------------------
# AI scenario state
# -----------------------------------------------------------------------------

if "ai_scenario_adjustments" not in st.session_state:
    st.session_state["ai_scenario_adjustments"] = None

if "ai_scenario_risk_summary" not in st.session_state:
    st.session_state["ai_scenario_risk_summary"] = None

if "ai_scenario_simulation_results" not in st.session_state:
    st.session_state["ai_scenario_simulation_results"] = None

# -----------------------------------------------------------------------------
# Display
# -----------------------------------------------------------------------------

st.header("Portfolio Setup")
st.write(
    f"**Primary scenario:** `{selected_scenario}` — "
    f"{available_scenarios[selected_scenario]}"
)
st.divider()

overview_df = pd.DataFrame({
    "asset": asset_names,
    "weight": weights,
    "expected_return_annual": annual_mean_returns,
    "volatility_annual": annual_volatilities,
})

overview_display = overview_df.copy()
overview_display["weight"] = overview_display["weight"].map(format_pct)
overview_display["expected_return_annual"] = overview_display["expected_return_annual"].map(format_pct)
overview_display["volatility_annual"] = overview_display["volatility_annual"].map(format_pct)

col1, col2 = st.columns([1.2, 1])

with col1:
    st.dataframe(overview_display, width="stretch", hide_index=True)

with col2:
    st.markdown("**Portfolio Allocation**")
    render_matplotlib_chart(
        plot_portfolio_allocation,
        asset_names,
        weights
    )

metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)

metric_col_1.metric("Mean Return", format_pct(risk_summary["mean_return"]))
metric_col_2.metric("Volatility", format_pct(risk_summary["volatility"]))
metric_col_3.metric("VaR 95", format_pct(risk_summary["var_95"]))
metric_col_4.metric("Expected Shortfall", format_pct(risk_summary["es_95"]))

metric_col_5, metric_col_6, metric_col_7, metric_col_8 = st.columns(4)

metric_col_5.metric("VaR 99", format_pct(risk_summary["var_99"]))
metric_col_6.metric("Probability of Loss", format_pct(risk_summary["probability_of_loss"]))
metric_col_7.metric("Max Drawdown", format_pct(risk_summary["max_drawdown"]))
metric_col_8.metric("Sharpe Ratio", format_num(risk_summary["sharpe_ratio"]))

st.divider()
tab_1, tab_2, tab_3, tab_4, tab_5 = st.tabs([
    "Core Risk Dashboard",
    "Risk Attribution",
    "Scenario Comparison",
    "Optimization",
    "AI Scenario Designer",
])

with tab_1:
    st.markdown("### Monte Carlo Risk Dashboard")

    chart_col_1, chart_col_2 = st.columns(2)

    with chart_col_1:
        st.markdown("**Return Distribution**")
        render_matplotlib_chart(
            plot_return_distribution,
            simulation_results["terminal_returns"],
            risk_summary["var_95"],
            risk_summary["es_95"],
        )

    with chart_col_2:
        st.markdown("**Drawdown Distribution**")
        render_matplotlib_chart(
            plot_drawdown_distribution,
            simulation_results["portfolio_paths"],
        )

    st.markdown("**Portfolio Paths**")
    render_matplotlib_chart(
        plot_portfolio_paths,
        simulation_results["portfolio_paths"],
        100,
    )

with tab_2:
    st.markdown("### Risk Contribution Analysis")
    st.write(
        "This section shows which assets contribute most to total portfolio risk "
        "under the selected scenario."
    )

    attr_col_1, attr_col_2 = st.columns([1, 1])

    with attr_col_1:
        st.markdown("**Risk Contribution Table**")
        st.dataframe(risk_contribution_display, width="stretch", hide_index=True)

    with attr_col_2:
        st.markdown("**Risk Contribution Chart**")
        render_matplotlib_chart(
            plot_risk_contributions,
            risk_contribution_summary,
        )

with tab_3:
    st.markdown("### Scenario Comparison")
    st.write(
        "Compare portfolio behavior across multiple macro stress scenarios."
    )

    if scenario_comparison_df.empty:
        st.info("Select at least one comparison scenario in the sidebar.")
    else:
        st.markdown("**Scenario Comparison Table**")
        st.dataframe(scenario_comparison_display, width="stretch", hide_index=True)

        st.markdown(f"**Scenario Metric Chart: {comparison_metric}**")
        render_matplotlib_chart(
            plot_scenario_metric_comparison,
            scenario_comparison_df,
            comparison_metric,
        )

with tab_4:
    st.markdown("### Portfolio Optimization")
    st.write(
        "Optimization is based on scenario-adjusted annualized assumptions."
    )

    opt_col_1, opt_col_2, opt_col_3 = st.columns(3)

    with opt_col_1:
        st.markdown("**Current Portfolio**")
        st.metric("Expected Return", format_pct(current_portfolio_stats["expected_return"]))
        st.metric("Volatility", format_pct(current_portfolio_stats["volatility"]))
        st.metric("Sharpe Ratio", format_num(current_portfolio_stats["sharpe_ratio"]))

    with opt_col_2:
        st.markdown("**Minimum Variance Portfolio**")
        st.metric("Expected Return", format_pct(min_variance_portfolio["expected_return"]))
        st.metric("Volatility", format_pct(min_variance_portfolio["volatility"]))
        st.metric("Sharpe Ratio", format_num(min_variance_portfolio["sharpe_ratio"]))

    with opt_col_3:
        st.markdown("**Maximum Sharpe Portfolio**")
        st.metric("Expected Return", format_pct(max_sharpe_portfolio["expected_return"]))
        st.metric("Volatility", format_pct(max_sharpe_portfolio["volatility"]))
        st.metric("Sharpe Ratio", format_num(max_sharpe_portfolio["sharpe_ratio"]))

    st.markdown("**Optimization Weights**")
    st.dataframe(optimization_weights_display, width="stretch", hide_index=True)

    st.markdown("**Efficient Frontier**")
    render_matplotlib_chart(
        plot_efficient_frontier,
        frontier_df,
        current_portfolio_stats,
        min_variance_portfolio,
        max_sharpe_portfolio,
    )

with tab_5:
    st.markdown("### AI Scenario Designer")
    st.write(
        "Translate a macro narrative into structured return, volatility, and "
        "correlation adjustments before running a separate simulation."
    )

    ai_prompt = st.text_area(
        "Scenario Prompt",
        value=st.session_state.get(
            "ai_scenario_prompt",
            "Severe tech selloff with sticky inflation and weak bond performance",
        ),
        height=120,
        placeholder="Describe a macro or market regime in plain language.",
        key="ai_scenario_prompt",
    )

    ai_action_col_1, ai_action_col_2 = st.columns([1, 1])

    with ai_action_col_1:
        generate_ai_scenario = st.button(
            "Generate AI Scenario",
            use_container_width=True,
        )

    with ai_action_col_2:
        run_ai_scenario = st.button(
            "Run AI Scenario Simulation",
            use_container_width=True,
            disabled=st.session_state["ai_scenario_adjustments"] is None,
        )

    if generate_ai_scenario:
        if not ai_prompt.strip():
            st.warning("Enter a scenario prompt to generate structured assumptions.")
            st.session_state["ai_scenario_adjustments"] = None
            st.session_state["ai_scenario_risk_summary"] = None
            st.session_state["ai_scenario_simulation_results"] = None
        else:
            try:
                st.session_state["ai_scenario_adjustments"] = build_ai_scenario_adjustments(
                    prompt=ai_prompt,
                    asset_names=asset_names,
                    base_returns=daily_mean_returns,
                    base_vols=daily_volatilities,
                    base_corr=correlation_matrix,
                )
                st.session_state["ai_scenario_risk_summary"] = None
                st.session_state["ai_scenario_simulation_results"] = None
            except Exception as exc:
                st.error(f"AI scenario generation error: {exc}")
                st.session_state["ai_scenario_adjustments"] = None

    ai_scenario_adjustments = st.session_state["ai_scenario_adjustments"]

    if ai_scenario_adjustments is not None:
        ai_summary = summarize_ai_scenario(ai_scenario_adjustments)
        ai_covariance_daily = build_covariance_matrix(
            ai_scenario_adjustments["adjusted_volatilities"],
            ai_scenario_adjustments["adjusted_correlation_matrix"],
        )

        preview_df = pd.DataFrame({
            "asset": asset_names,
            "base_return_annual": annual_mean_returns,
            "return_shock_annual": daily_return_to_annual(
                np.array([
                    ai_scenario_adjustments["return_shocks"][asset_name]
                    for asset_name in asset_names
                ])
            ),
            "adjusted_return_annual": daily_return_to_annual(
                ai_scenario_adjustments["adjusted_mean_returns"]
            ),
            "base_volatility_annual": annual_volatilities,
            "volatility_multiplier": [
                ai_scenario_adjustments["volatility_multipliers"][asset_name]
                for asset_name in asset_names
            ],
            "adjusted_volatility_annual": daily_vol_to_annual(
                ai_scenario_adjustments["adjusted_volatilities"]
            ),
        })

        preview_display = preview_df.copy()
        for column in [
            "base_return_annual",
            "return_shock_annual",
            "adjusted_return_annual",
            "base_volatility_annual",
            "adjusted_volatility_annual",
        ]:
            preview_display[column] = preview_display[column].map(format_pct)

        preview_display["volatility_multiplier"] = preview_display["volatility_multiplier"].map(format_num)

        pairwise_df = pd.DataFrame(
            ai_scenario_adjustments["pairwise_correlation_adjustments"]
        )
        if not pairwise_df.empty:
            pairwise_df["pair"] = pairwise_df["asset_1"] + " / " + pairwise_df["asset_2"]
            pairwise_display = pairwise_df[["pair", "shift"]].copy()
            pairwise_display["shift"] = pairwise_display["shift"].map(format_num)
        else:
            pairwise_display = pd.DataFrame(columns=["pair", "shift"])

        summary_col_1, summary_col_2, summary_col_3 = st.columns([1, 1, 1])

        with summary_col_1:
            st.markdown("**Interpreted Regime**")
            st.write(", ".join(ai_scenario_adjustments["regime_labels"]))

        with summary_col_2:
            st.markdown("**Parser Confidence**")
            st.write(ai_scenario_adjustments["confidence"].title())

        with summary_col_3:
            st.markdown("**Broad Correlation Shift**")
            st.write(format_num(ai_scenario_adjustments["correlation_shift"]))

        st.markdown("**Scenario Summary**")
        st.write(ai_summary)

        preview_col_1, preview_col_2 = st.columns([1.5, 1])

        with preview_col_1:
            st.markdown("**Generated Asset Assumptions**")
            st.dataframe(preview_display, width="stretch", hide_index=True)

        with preview_col_2:
            st.markdown("**Pairwise Correlation Adjustments**")
            if pairwise_display.empty:
                st.info("No pair-specific correlation adjustments were generated.")
            else:
                st.dataframe(pairwise_display, width="stretch", hide_index=True)

        st.markdown("**Adjusted Correlation Matrix**")
        st.dataframe(
            pd.DataFrame(
                ai_scenario_adjustments["adjusted_correlation_matrix"],
                index=asset_names,
                columns=asset_names,
            ).style.format("{:.2f}"),
            width="stretch",
        )

        if run_ai_scenario:
            try:
                ai_simulation_results = run_monte_carlo_engine(
                    weights=weights,
                    mean_returns=ai_scenario_adjustments["adjusted_mean_returns"],
                    covariance_matrix=ai_covariance_daily,
                    num_simulations=num_simulations,
                    time_horizon_days=time_horizon_days,
                    initial_portfolio_value=initial_portfolio_value,
                )
                st.session_state["ai_scenario_simulation_results"] = ai_simulation_results
                st.session_state["ai_scenario_risk_summary"] = build_risk_summary(
                    terminal_returns=ai_simulation_results["terminal_returns"],
                    portfolio_paths=ai_simulation_results["portfolio_paths"],
                )
            except Exception as exc:
                st.error(f"AI scenario simulation error: {exc}")
                st.session_state["ai_scenario_simulation_results"] = None
                st.session_state["ai_scenario_risk_summary"] = None

    ai_risk_summary = st.session_state["ai_scenario_risk_summary"]
    ai_simulation_results = st.session_state["ai_scenario_simulation_results"]

    if ai_risk_summary is not None and ai_simulation_results is not None:
        st.markdown("**AI Scenario Risk Summary**")
        ai_metric_col_1, ai_metric_col_2, ai_metric_col_3, ai_metric_col_4 = st.columns(4)
        ai_metric_col_1.metric("Mean Return", format_pct(ai_risk_summary["mean_return"]))
        ai_metric_col_2.metric("Volatility", format_pct(ai_risk_summary["volatility"]))
        ai_metric_col_3.metric("VaR 95", format_pct(ai_risk_summary["var_95"]))
        ai_metric_col_4.metric("Expected Shortfall", format_pct(ai_risk_summary["es_95"]))

        ai_metric_col_5, ai_metric_col_6, ai_metric_col_7, ai_metric_col_8 = st.columns(4)
        ai_metric_col_5.metric("VaR 99", format_pct(ai_risk_summary["var_99"]))
        ai_metric_col_6.metric("Probability of Loss", format_pct(ai_risk_summary["probability_of_loss"]))
        ai_metric_col_7.metric("Max Drawdown", format_pct(ai_risk_summary["max_drawdown"]))
        ai_metric_col_8.metric("Sharpe Ratio", format_num(ai_risk_summary["sharpe_ratio"]))

        ai_chart_col_1, ai_chart_col_2 = st.columns(2)

        with ai_chart_col_1:
            st.markdown("**Terminal Return Distribution**")
            render_matplotlib_chart(
                plot_return_distribution,
                ai_simulation_results["terminal_returns"],
                ai_risk_summary["var_95"],
                ai_risk_summary["es_95"],
            )

        with ai_chart_col_2:
            st.markdown("**Portfolio Paths**")
            render_matplotlib_chart(
                plot_portfolio_paths,
                ai_simulation_results["portfolio_paths"],
                100,
            )

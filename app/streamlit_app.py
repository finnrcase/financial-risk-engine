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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.ai_scenarios import (
    build_ai_scenario_adjustments,
    summarize_ai_scenario,
)
from src.jr_analyst import generate_jr_analyst_commentary
from src.optimization import (
    calculate_portfolio_statistics,
    generate_efficient_frontier,
    optimize_max_sharpe,
    optimize_min_variance,
)
from src.portfolio import build_covariance_matrix
from src.risk_contribution import build_risk_contribution_summary
from src.risk_metrics import build_risk_summary
from src.scenario_analysis import (
    build_scenario_comparison_table,
    run_multi_scenario_comparison,
)
from src.scenarios import apply_scenario, get_available_scenarios
from src.simulation import run_monte_carlo_engine
from src.validation import validate_inputs
from src.visualization import (
    plot_drawdown_distribution,
    plot_efficient_frontier,
    plot_portfolio_allocation,
    plot_portfolio_paths,
    plot_return_distribution,
    plot_risk_contributions,
    plot_scenario_metric_comparison,
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


def inject_design_system() -> None:
    """
    Inject a cohesive presentation layer for the Streamlit interface.
    """
    st.markdown(
        """
        <style>
        :root {
            --bg: #f3f5f8;
            --surface: rgba(255, 255, 255, 0.92);
            --surface-strong: #ffffff;
            --surface-muted: #eef2f7;
            --text: #0f172a;
            --muted: #5b6474;
            --line: rgba(15, 23, 42, 0.08);
            --line-strong: rgba(73, 99, 138, 0.22);
            --accent: #6f93bf;
            --accent-strong: #456a98;
            --accent-soft: rgba(111, 147, 191, 0.16);
            --shadow-soft: 0 22px 50px rgba(15, 23, 42, 0.08);
            --shadow-card: 0 12px 28px rgba(15, 23, 42, 0.06);
            --radius-xl: 28px;
            --radius-md: 16px;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(111, 147, 191, 0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(15, 23, 42, 0.05), transparent 24%),
                linear-gradient(180deg, #f8fafc 0%, var(--bg) 100%);
            color: var(--text);
        }

        .main > div {
            padding-top: 1.15rem;
            padding-bottom: 2.5rem;
            max-width: 1440px;
        }

        [data-testid="stHeader"] {
            background: rgba(243, 245, 248, 0.64);
            backdrop-filter: blur(10px);
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.95) 0%, rgba(246, 248, 251, 0.98) 100%);
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.2rem;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: var(--radius-md);
            background: rgba(255, 255, 255, 0.76);
            box-shadow: var(--shadow-card);
            overflow: hidden;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] details summary {
            padding: 0.9rem 1rem;
            font-weight: 600;
        }

        .hero-card,
        .section-card,
        .metric-card,
        .content-card,
        .insight-card {
            background: var(--surface);
            border: 1px solid rgba(255, 255, 255, 0.76);
            border-radius: var(--radius-xl);
            box-shadow: var(--shadow-soft);
            backdrop-filter: blur(14px);
        }

        .hero-card {
            padding: 1.55rem 1.6rem 1.35rem;
            margin-bottom: 1.15rem;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.75fr) minmax(300px, 0.95fr);
            gap: 1rem;
            align-items: start;
        }

        .eyebrow {
            display: inline-flex;
            padding: 0.38rem 0.72rem;
            border-radius: 999px;
            background: var(--surface-muted);
            border: 1px solid var(--line);
            color: var(--accent-strong);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }

        .hero-title {
            margin: 0.95rem 0 0.55rem;
            font-size: clamp(2rem, 3.2vw, 3.25rem);
            line-height: 1.02;
            letter-spacing: -0.05em;
        }

        .hero-copy,
        .section-copy,
        .panel-copy {
            margin: 0;
            color: var(--muted);
            line-height: 1.65;
            font-size: 0.98rem;
        }

        .hero-meta {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: 0.65rem;
        }

        .meta-pill {
            min-width: 150px;
            padding: 0.72rem 0.92rem;
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(244,247,251,0.94));
            border: 1px solid var(--line);
            box-shadow: var(--shadow-card);
        }

        .meta-label,
        .metric-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .meta-value,
        .metric-value {
            margin-top: 0.38rem;
            color: var(--text);
            font-weight: 700;
            letter-spacing: -0.03em;
        }

        .section-card {
            padding: 1.25rem;
            margin-bottom: 1rem;
        }

        .section-head {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: flex-start;
            margin-bottom: 1rem;
        }

        .section-title {
            margin: 0;
            font-size: 1.18rem;
            letter-spacing: -0.03em;
        }

        .section-chip {
            padding: 0.4rem 0.72rem;
            border-radius: 999px;
            background: var(--accent-soft);
            border: 1px solid rgba(111, 147, 191, 0.2);
            color: var(--accent-strong);
            font-size: 0.78rem;
            font-weight: 700;
            white-space: nowrap;
        }

        .metric-card {
            padding: 0.98rem 1rem 0.9rem;
            min-height: 120px;
            margin-bottom: 0.8rem;
            background: linear-gradient(180deg, rgba(255,255,255,0.97), rgba(246,248,251,0.93));
        }

        .metric-value {
            font-size: clamp(1.28rem, 2vw, 1.85rem);
            margin-top: 0.62rem;
        }

        .metric-note {
            margin-top: 0.52rem;
            color: var(--accent-strong);
            font-size: 0.82rem;
            font-weight: 600;
        }

        .content-card,
        .insight-card {
            padding: 1rem 1rem 0.82rem;
            margin-bottom: 1rem;
        }

        .panel-title {
            margin: 0 0 0.3rem;
            color: var(--text);
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .takeaway-card {
            min-height: 150px;
            padding: 1rem;
            border-radius: var(--radius-md);
            background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(242,246,251,0.92));
            border: 1px solid var(--line);
        }

        div[data-testid="stTabs"] button {
            border-radius: 999px;
            border: 1px solid transparent;
            padding: 0.62rem 0.95rem;
            color: var(--muted);
            background: rgba(255, 255, 255, 0.52);
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            background: var(--surface-strong);
            color: var(--text);
            border-color: var(--line-strong);
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 16px;
            border: 1px solid rgba(69, 106, 152, 0.14);
            background: linear-gradient(180deg, #fdfefe 0%, #eff4f9 100%);
            color: var(--text);
            font-weight: 600;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
            padding: 0.7rem 1rem;
        }

        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input,
        div[data-baseweb="select"] > div,
        .stMultiSelect [data-baseweb="select"] > div {
            border-radius: 16px !important;
            border-color: rgba(15, 23, 42, 0.1) !important;
            background: rgba(255, 255, 255, 0.92) !important;
            box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.03);
        }

        div[data-baseweb="tag"] {
            border-radius: 999px !important;
            background: var(--accent-soft) !important;
            color: var(--accent-strong) !important;
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid var(--line);
            box-shadow: var(--shadow-card);
            background: rgba(255, 255, 255, 0.92);
        }

        [data-testid="stAlert"] {
            border-radius: 18px;
            border: 1px solid var(--line);
        }

        @media (max-width: 1100px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }

            .hero-meta {
                justify-content: flex-start;
                margin-top: 1rem;
            }

            .section-head {
                flex-direction: column;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, note: str) -> None:
    """
    Render a custom metric card with a softer analytics treatment.
    """
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, copy: str, chip: str | None = None) -> None:
    """
    Render a framed section header.
    """
    chip_markup = f'<div class="section-chip">{chip}</div>' if chip else ""
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-head">
                <div>
                    <div class="section-title">{title}</div>
                    <p class="section-copy">{copy}</p>
                </div>
                {chip_markup}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel_intro(title: str, copy: str) -> None:
    """
    Render a compact framed intro for a panel.
    """
    st.markdown(
        f"""
        <div class="content-card">
            <div class="panel-title">{title}</div>
            <p class="panel-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        [1.0, corr_eq_bonds, corr_eq_tech, corr_eq_crypto],
        [corr_eq_bonds, 1.0, corr_bonds_tech, corr_bonds_crypto],
        [corr_eq_tech, corr_bonds_tech, 1.0, corr_tech_crypto],
        [corr_eq_crypto, corr_bonds_crypto, corr_tech_crypto, 1.0],
    ])

    return correlation_matrix


st.set_page_config(
    page_title="Financial Risk Engine",
    page_icon="FRE",
    layout="wide",
)

inject_design_system()

# -----------------------------------------------------------------------------
# Sidebar inputs
# -----------------------------------------------------------------------------

st.sidebar.markdown("### Control Center")
st.sidebar.caption("Portfolio assumptions, simulation settings, scenario controls, and optimization parameters.")

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

st.sidebar.markdown("### Simulation Settings")

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

st.sidebar.markdown("### Scenario Controls")

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

st.sidebar.markdown("### Optimization Settings")

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

weights_sum_before_normalization = weights.sum()
weights_were_normalized = not np.isclose(weights_sum_before_normalization, 1.0)

if weights_were_normalized:
    st.warning(
        f"Portfolio weights currently sum to {weights_sum_before_normalization:.2f}. "
        "Weights have been normalized automatically for analysis."
    )
    weights = weights / weights_sum_before_normalization

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

jr_analyst_commentary = generate_jr_analyst_commentary(
    weights=weights,
    asset_names=asset_names,
    risk_summary=risk_summary,
    risk_contribution_summary=risk_contribution_summary,
    scenario_comparison_df=scenario_comparison_df,
    current_portfolio_stats=current_portfolio_stats,
    min_variance_portfolio=min_variance_portfolio,
    max_sharpe_portfolio=max_sharpe_portfolio,
)

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

st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-grid">
            <div>
                <div class="eyebrow">Financial Risk Engine</div>
                <div class="hero-title">Institutional portfolio risk analytics for simulation, stress testing, and optimization.</div>
                <p class="hero-copy">
                    Explore Monte Carlo risk, scenario stress, attribution, optimization, and AI-assisted
                    scenario design from a single operating surface built for investment and strategy teams.
                </p>
            </div>
            <div class="hero-meta">
                <div class="meta-pill">
                    <div class="meta-label">Primary Scenario</div>
                    <div class="meta-value">{selected_scenario}</div>
                </div>
                <div class="meta-pill">
                    <div class="meta-label">Simulation Count</div>
                    <div class="meta-value">{num_simulations:,}</div>
                </div>
                <div class="meta-pill">
                    <div class="meta-label">Horizon</div>
                    <div class="meta-value">{time_horizon_days} trading days</div>
                </div>
                <div class="meta-pill">
                    <div class="meta-label">Weighting</div>
                    <div class="meta-value">{'Normalized' if weights_were_normalized else 'Direct input'}</div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_section_header(
    "Portfolio Setup",
    "Review portfolio composition, annual assumptions, and core risk signals in a single overview.",
    chip=available_scenarios[selected_scenario],
)

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

overview_col_1, overview_col_2 = st.columns([1.25, 0.95])

with overview_col_1:
    render_panel_intro(
        "Portfolio assumptions",
        "Asset mix, expected annual return, and annual volatility for the current portfolio.",
    )
    st.dataframe(overview_display, width="stretch", hide_index=True)

with overview_col_2:
    render_panel_intro(
        "Allocation snapshot",
        "Current exposure by asset class.",
    )
    render_matplotlib_chart(plot_portfolio_allocation, asset_names, weights)

risk_metric_rows = [
    ("Mean Return", format_pct(risk_summary["mean_return"]), "Central expectation"),
    ("Volatility", format_pct(risk_summary["volatility"]), "Dispersion across outcomes"),
    ("VaR 95", format_pct(risk_summary["var_95"]), "Tail threshold"),
    ("Expected Shortfall", format_pct(risk_summary["es_95"]), "Average beyond VaR"),
    ("VaR 99", format_pct(risk_summary["var_99"]), "More severe tail cut"),
    ("Probability of Loss", format_pct(risk_summary["probability_of_loss"]), "Chance of finishing negative"),
    ("Max Drawdown", format_pct(risk_summary["max_drawdown"]), "Peak-to-trough stress"),
    ("Sharpe Ratio", format_num(risk_summary["sharpe_ratio"]), "Risk-adjusted efficiency"),
]

metric_columns = st.columns(4)
for index, (label, value, note) in enumerate(risk_metric_rows):
    with metric_columns[index % 4]:
        render_metric_card(label, value, note)

tab_1, tab_2, tab_3, tab_4, tab_5, tab_6 = st.tabs([
    "Core Risk Dashboard",
    "Risk Attribution",
    "Scenario Comparison",
    "Optimization",
    "JrAnalyst.AI",
    "AI Scenario Designer",
])

with tab_1:
    render_section_header(
        "Core Risk Dashboard",
        "Monte Carlo outcome views for returns, drawdowns, and sampled portfolio paths.",
    )

    chart_col_1, chart_col_2 = st.columns(2)

    with chart_col_1:
        render_panel_intro(
            "Return distribution",
            "Distribution of simulated terminal returns with VaR and expected shortfall markers.",
        )
        render_matplotlib_chart(
            plot_return_distribution,
            simulation_results["terminal_returns"],
            risk_summary["var_95"],
            risk_summary["es_95"],
        )

    with chart_col_2:
        render_panel_intro(
            "Drawdown distribution",
            "Stress profile across path-level maximum drawdowns.",
        )
        render_matplotlib_chart(
            plot_drawdown_distribution,
            simulation_results["portfolio_paths"],
        )

    render_panel_intro(
        "Portfolio paths",
        "Sampled Monte Carlo trajectories over the selected horizon.",
    )
    render_matplotlib_chart(
        plot_portfolio_paths,
        simulation_results["portfolio_paths"],
        100,
    )

with tab_2:
    render_section_header(
        "Risk Attribution",
        "Asset-level contribution detail is preserved, but the table and chart now feel like one coordinated workspace instead of two disconnected blocks.",
    )

    attr_col_1, attr_col_2 = st.columns([1, 1])

    with attr_col_1:
        render_panel_intro(
            "Risk contribution table",
            "Marginal, component, and percentage contribution under the active scenario.",
        )
        st.dataframe(risk_contribution_display, width="stretch", hide_index=True)

    with attr_col_2:
        render_panel_intro(
            "Risk contribution chart",
            "Which sleeves dominate portfolio risk under the active scenario.",
        )
        render_matplotlib_chart(
            plot_risk_contributions,
            risk_contribution_summary,
        )

with tab_3:
    render_section_header(
        "Scenario Comparison",
        "Stress-testing views remain intact, with improved hierarchy for selecting, reading, and comparing macro regimes.",
    )

    if scenario_comparison_df.empty:
        st.info("Select at least one comparison scenario in the sidebar.")
    else:
        render_panel_intro(
            "Scenario comparison table",
            "All configured comparison scenarios with the same metrics and no workflow changes.",
        )
        st.dataframe(scenario_comparison_display, width="stretch", hide_index=True)

        render_panel_intro(
            f"Scenario metric chart: {comparison_metric}",
            "Focused view of the selected metric across comparison regimes.",
        )
        render_matplotlib_chart(
            plot_scenario_metric_comparison,
            scenario_comparison_df,
            comparison_metric,
        )

with tab_4:
    render_section_header(
        "Optimization",
        "Compare current positioning with minimum-variance and maximum-Sharpe portfolio outputs.",
    )

    opt_col_1, opt_col_2, opt_col_3 = st.columns(3)

    with opt_col_1:
        render_metric_card("Current Expected Return", format_pct(current_portfolio_stats["expected_return"]), "Current portfolio")
        render_metric_card("Current Volatility", format_pct(current_portfolio_stats["volatility"]), "Current portfolio")
        render_metric_card("Current Sharpe", format_num(current_portfolio_stats["sharpe_ratio"]), "Current portfolio")

    with opt_col_2:
        render_metric_card("Min Variance Return", format_pct(min_variance_portfolio["expected_return"]), "Optimized for lower risk")
        render_metric_card("Min Variance Volatility", format_pct(min_variance_portfolio["volatility"]), "Optimized for lower risk")
        render_metric_card("Min Variance Sharpe", format_num(min_variance_portfolio["sharpe_ratio"]), "Optimized for lower risk")

    with opt_col_3:
        render_metric_card("Max Sharpe Return", format_pct(max_sharpe_portfolio["expected_return"]), "Optimized for efficiency")
        render_metric_card("Max Sharpe Volatility", format_pct(max_sharpe_portfolio["volatility"]), "Optimized for efficiency")
        render_metric_card("Max Sharpe Sharpe", format_num(max_sharpe_portfolio["sharpe_ratio"]), "Optimized for efficiency")

    weights_col, frontier_col = st.columns([1, 1.1])

    with weights_col:
        render_panel_intro(
            "Optimization weights",
            "Current and optimized allocations with the same underlying outputs.",
        )
        st.dataframe(optimization_weights_display, width="stretch", hide_index=True)

    with frontier_col:
        render_panel_intro(
            "Efficient frontier",
            "Scenario-adjusted opportunity set with current, min-variance, and max-Sharpe overlays.",
        )
        render_matplotlib_chart(
            plot_efficient_frontier,
            frontier_df,
            current_portfolio_stats,
            min_variance_portfolio,
            max_sharpe_portfolio,
        )

with tab_5:
    render_section_header(
        "JrAnalyst.AI",
        "Deterministic portfolio commentary generated from the current analytics state.",
    )

    st.markdown('<div class="insight-card">', unsafe_allow_html=True)
    for section_title, section_text in jr_analyst_commentary["sections"].items():
        st.markdown(f"**{section_title}**")
        st.write(section_text)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### Key Takeaways")
    takeaway_col_1, takeaway_col_2, takeaway_col_3 = st.columns(3)
    takeaway_columns = [takeaway_col_1, takeaway_col_2, takeaway_col_3]

    for column, takeaway in zip(takeaway_columns, jr_analyst_commentary["key_takeaways"]):
        with column:
            st.markdown(
                f"""
                <div class="takeaway-card">
                    <div class="panel-title">Takeaway</div>
                    <p class="panel-copy">{takeaway}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

with tab_6:
    render_section_header(
        "AI Scenario Designer",
        "Translate a market narrative into structured assumptions and run a dedicated simulation.",
    )

    render_panel_intro(
        "Scenario prompt",
        "Describe a market regime in plain language, generate structured shocks, then run a separate simulation using those adjusted assumptions.",
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

        summary_col_1, summary_col_2, summary_col_3 = st.columns(3)

        with summary_col_1:
            render_metric_card(
                "Interpreted Regime",
                ", ".join(ai_scenario_adjustments["regime_labels"]),
                "Narrative parser labels",
            )

        with summary_col_2:
            render_metric_card(
                "Parser Confidence",
                ai_scenario_adjustments["confidence"].title(),
                "Structured scenario extraction",
            )

        with summary_col_3:
            render_metric_card(
                "Broad Correlation Shift",
                format_num(ai_scenario_adjustments["correlation_shift"]),
                "Average cross-asset move",
            )

        render_panel_intro(
            "Scenario summary",
            ai_summary,
        )

        preview_col_1, preview_col_2 = st.columns([1.45, 1])

        with preview_col_1:
            render_panel_intro(
                "Generated asset assumptions",
                "Adjusted returns and volatilities derived from the narrative prompt.",
            )
            st.dataframe(preview_display, width="stretch", hide_index=True)

        with preview_col_2:
            render_panel_intro(
                "Pairwise correlation adjustments",
                "Any pair-specific correlation shifts surfaced by the parser.",
            )
            if pairwise_display.empty:
                st.info("No pair-specific correlation adjustments were generated.")
            else:
                st.dataframe(pairwise_display, width="stretch", hide_index=True)

        render_panel_intro(
            "Adjusted correlation matrix",
            "Generated matrix used for the AI scenario simulation path.",
        )
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
        render_panel_intro(
            "AI scenario risk summary",
            "Separate risk readout using the generated AI scenario assumptions.",
        )

        ai_metric_rows = [
            ("Mean Return", format_pct(ai_risk_summary["mean_return"]), "AI scenario"),
            ("Volatility", format_pct(ai_risk_summary["volatility"]), "AI scenario"),
            ("VaR 95", format_pct(ai_risk_summary["var_95"]), "AI scenario"),
            ("Expected Shortfall", format_pct(ai_risk_summary["es_95"]), "AI scenario"),
            ("VaR 99", format_pct(ai_risk_summary["var_99"]), "AI scenario"),
            ("Probability of Loss", format_pct(ai_risk_summary["probability_of_loss"]), "AI scenario"),
            ("Max Drawdown", format_pct(ai_risk_summary["max_drawdown"]), "AI scenario"),
            ("Sharpe Ratio", format_num(ai_risk_summary["sharpe_ratio"]), "AI scenario"),
        ]

        ai_metric_columns = st.columns(4)
        for index, (label, value, note) in enumerate(ai_metric_rows):
            with ai_metric_columns[index % 4]:
                render_metric_card(label, value, note)

        ai_chart_col_1, ai_chart_col_2 = st.columns(2)

        with ai_chart_col_1:
            render_panel_intro(
                "Terminal return distribution",
                "Terminal outcomes under the generated AI scenario.",
            )
            render_matplotlib_chart(
                plot_return_distribution,
                ai_simulation_results["terminal_returns"],
                ai_risk_summary["var_95"],
                ai_risk_summary["es_95"],
            )

        with ai_chart_col_2:
            render_panel_intro(
                "Portfolio paths",
                "Sampled trajectory set for the AI scenario run.",
            )
            render_matplotlib_chart(
                plot_portfolio_paths,
                ai_simulation_results["portfolio_paths"],
                100,
            )

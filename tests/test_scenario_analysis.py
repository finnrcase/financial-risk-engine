import numpy as np

from src.scenario_analysis import (
    run_single_scenario_analysis,
    run_multi_scenario_comparison,
    build_scenario_comparison_table,
)


def test_run_single_scenario_analysis_returns_expected_keys():
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    mean_returns = np.array([0.08 / 252, 0.04 / 252, 0.10 / 252, 0.18 / 252])
    volatilities = np.array([0.16, 0.06, 0.22, 0.55])
    correlation_matrix = np.array([
        [1.00, 0.20, 0.75, 0.35],
        [0.20, 1.00, 0.15, 0.05],
        [0.75, 0.15, 1.00, 0.45],
        [0.35, 0.05, 0.45, 1.00],
    ])

    result = run_single_scenario_analysis(
        weights=weights,
        mean_returns=mean_returns,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        scenario_name="baseline",
        num_simulations=500,
        time_horizon_days=30,
    )

    assert "scenario" in result
    assert "risk_summary" in result
    assert "simulation_results" in result
    assert "adjusted_inputs" in result
    assert result["scenario"] == "baseline"


def test_run_multi_scenario_comparison_returns_correct_number_of_results():
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    mean_returns = np.array([0.08 / 252, 0.04 / 252, 0.10 / 252, 0.18 / 252])
    volatilities = np.array([0.16, 0.06, 0.22, 0.55])
    correlation_matrix = np.array([
        [1.00, 0.20, 0.75, 0.35],
        [0.20, 1.00, 0.15, 0.05],
        [0.75, 0.15, 1.00, 0.45],
        [0.35, 0.05, 0.45, 1.00],
    ])
    scenario_names = ["baseline", "market_crash", "tech_correction"]

    results = run_multi_scenario_comparison(
        weights=weights,
        mean_returns=mean_returns,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        scenario_names=scenario_names,
        num_simulations=250,
        time_horizon_days=20,
    )

    assert len(results) == len(scenario_names)
    assert [result["scenario"] for result in results] == scenario_names


def test_build_scenario_comparison_table_returns_expected_columns():
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    mean_returns = np.array([0.08 / 252, 0.04 / 252, 0.10 / 252, 0.18 / 252])
    volatilities = np.array([0.16, 0.06, 0.22, 0.55])
    correlation_matrix = np.array([
        [1.00, 0.20, 0.75, 0.35],
        [0.20, 1.00, 0.15, 0.05],
        [0.75, 0.15, 1.00, 0.45],
        [0.35, 0.05, 0.45, 1.00],
    ])
    scenario_names = ["baseline", "market_crash"]

    results = run_multi_scenario_comparison(
        weights=weights,
        mean_returns=mean_returns,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        scenario_names=scenario_names,
        num_simulations=250,
        time_horizon_days=20,
    )

    comparison_table = build_scenario_comparison_table(results)

    expected_columns = [
        "scenario",
        "mean_return",
        "volatility",
        "var_95",
        "var_99",
        "es_95",
        "probability_of_loss",
        "max_drawdown",
        "sharpe_ratio",
    ]

    assert list(comparison_table.columns) == expected_columns
    assert len(comparison_table) == len(scenario_names)
    assert comparison_table["scenario"].tolist() == scenario_names
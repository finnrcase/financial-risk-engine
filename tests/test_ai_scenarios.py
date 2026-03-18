import numpy as np

from src.ai_scenarios import (
    RETURN_SHOCK_BOUNDS,
    VOL_MULTIPLIER_BOUNDS,
    apply_ai_scenario,
    build_ai_scenario_adjustments,
    parse_scenario_prompt,
    validate_ai_scenario,
)


def _sample_inputs():
    asset_names = ["Equities", "Bonds", "Tech", "Crypto"]
    mean_returns = np.array([0.08, 0.04, 0.10, 0.18]) / 252
    volatilities = np.array([0.16, 0.06, 0.22, 0.55]) / np.sqrt(252)
    correlation_matrix = np.array([
        [1.00, 0.20, 0.75, 0.35],
        [0.20, 1.00, 0.15, 0.05],
        [0.75, 0.15, 1.00, 0.45],
        [0.35, 0.05, 0.45, 1.00],
    ])
    return asset_names, mean_returns, volatilities, correlation_matrix


def test_parse_scenario_prompt_detects_common_regimes():
    asset_names, _, _, _ = _sample_inputs()

    parsed = parse_scenario_prompt(
        "Severe tech selloff with sticky inflation and weak bond performance",
        asset_names,
    )

    assert "tech crash" in parsed["regime_labels"]
    assert "inflation shock" in parsed["regime_labels"]
    assert "risk-off" in parsed["regime_labels"]
    assert parsed["return_shocks"]["Tech"] < 0.0
    assert parsed["volatility_multipliers"]["Tech"] > 1.0
    assert parsed["correlation_shift"] > 0.0


def test_apply_ai_scenario_preserves_valid_correlation_structure():
    asset_names, mean_returns, volatilities, correlation_matrix = _sample_inputs()

    adjustments = build_ai_scenario_adjustments(
        prompt="Energy shock combined with recession and high cross-asset correlation",
        asset_names=asset_names,
        base_returns=mean_returns,
        base_vols=volatilities,
        base_corr=correlation_matrix,
    )

    adjusted_corr = adjustments["adjusted_correlation_matrix"]

    assert adjusted_corr.shape == correlation_matrix.shape
    assert np.allclose(adjusted_corr, adjusted_corr.T)
    assert np.allclose(np.diag(adjusted_corr), 1.0)
    assert ((adjusted_corr >= -0.95) & (adjusted_corr <= 1.0)).all()


def test_ai_scenario_adjustments_change_returns_vol_and_correlations():
    asset_names, mean_returns, volatilities, correlation_matrix = _sample_inputs()

    adjustments = build_ai_scenario_adjustments(
        prompt="AI bubble bursts while rates remain elevated",
        asset_names=asset_names,
        base_returns=mean_returns,
        base_vols=volatilities,
        base_corr=correlation_matrix,
    )

    assert not np.allclose(adjustments["adjusted_mean_returns"], mean_returns)
    assert not np.allclose(adjustments["adjusted_volatilities"], volatilities)
    assert not np.allclose(adjustments["adjusted_correlation_matrix"], correlation_matrix)


def test_vague_prompt_produces_safe_defaults():
    asset_names, mean_returns, volatilities, correlation_matrix = _sample_inputs()

    adjustments = build_ai_scenario_adjustments(
        prompt="monitor evolving conditions",
        asset_names=asset_names,
        base_returns=mean_returns,
        base_vols=volatilities,
        base_corr=correlation_matrix,
    )

    assert adjustments["regime_labels"] == ["baseline"]
    assert np.allclose(adjustments["adjusted_mean_returns"], mean_returns)
    assert np.allclose(adjustments["adjusted_volatilities"], volatilities)
    assert np.allclose(adjustments["adjusted_correlation_matrix"], correlation_matrix)


def test_validate_ai_scenario_clamps_extreme_inputs():
    asset_names, mean_returns, volatilities, correlation_matrix = _sample_inputs()

    raw_adjustments = {
        "regime_labels": ["stress"],
        "return_shocks": {
            "Equities": 10.0,
            "Bonds": -10.0,
        },
        "volatility_multipliers": {
            "Equities": 10.0,
            "Bonds": -2.0,
        },
        "correlation_shift": np.nan,
        "pairwise_correlation_adjustments": [
            {"asset_1": "Equities", "asset_2": "Bonds", "shift": 10.0},
            {"asset_1": "Equities", "asset_2": "Equities", "shift": 0.2},
        ],
    }

    validated = validate_ai_scenario(raw_adjustments, asset_names)
    adjusted_returns, adjusted_vols, adjusted_corr = apply_ai_scenario(
        mean_returns,
        volatilities,
        correlation_matrix,
        validated,
    )

    assert validated["return_shocks"]["Equities"] == RETURN_SHOCK_BOUNDS[1]
    assert validated["return_shocks"]["Bonds"] == RETURN_SHOCK_BOUNDS[0]
    assert validated["volatility_multipliers"]["Equities"] == VOL_MULTIPLIER_BOUNDS[1]
    assert validated["volatility_multipliers"]["Bonds"] == VOL_MULTIPLIER_BOUNDS[0]
    assert validated["correlation_shift"] == 0.0
    assert len(validated["pairwise_correlation_adjustments"]) == 1
    assert adjusted_returns.shape == mean_returns.shape
    assert adjusted_vols.shape == volatilities.shape
    assert adjusted_corr.shape == correlation_matrix.shape

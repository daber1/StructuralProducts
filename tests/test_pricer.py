import pytest

from structpy.core.pricer import Pricer
from structpy.core.scenario import ScenarioGenerator


def _config() -> dict:
    return {
        "meta": {
            "isin": "RU000A108HM7",
            "name": "Sberbank CIB Phoenix Autocall",
            "type": "PhoenixAutocall",
        },
        "params": {
            "nominal": 1000.0,
            "currency": "RUB",
            "underlyings": ["TATN"],
            "feature_type": "standard",
            "memory_effect": True,
            "coupon_rate": 0.05,
            "barriers": {
                "autocall": 1.0,
                "coupon": 1.0,
                "capital_protection": 0.90,
            },
        },
    }


def test_pricer_returns_expected_result_shape(make_product):
    product = make_product(_config(), {"TATN": 700.0})
    generator = ScenarioGenerator(
        tickers=["TATN"],
        initial_prices={"TATN": 700.0},
        volatilities={"TATN": 0.25},
        risk_free_rate=0.15,
        dividend_yields={"TATN": 0.10},
    )

    result = Pricer(product, generator).price(n_scenarios=250, n_steps=3, T=3.0, seed=42)

    assert result["fair_value"] > 0
    assert result["std_error"] >= 0
    assert len(result["confidence_interval_95"]) == 2
    assert result["confidence_interval_95"][0] <= result["fair_value"]
    assert result["confidence_interval_95"][1] >= result["fair_value"]
    assert result["mean_payout_undiscounted"] > 0
    assert len(result["payouts"]) == 250


def test_pricer_rejects_invalid_inputs(make_product):
    product = make_product(_config(), {"TATN": 700.0})
    generator = ScenarioGenerator(
        tickers=["TATN"],
        initial_prices={"TATN": 700.0},
        volatilities={"TATN": 0.25},
        risk_free_rate=0.15,
    )
    pricer = Pricer(product, generator)

    with pytest.raises(ValueError):
        pricer.price(n_scenarios=0)

    with pytest.raises(ValueError):
        pricer.price(n_steps=0)

    with pytest.raises(ValueError):
        pricer.price(T=0.0)

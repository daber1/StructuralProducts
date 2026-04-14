import numpy as np
import pytest

from structpy.core.pricer import Pricer
from structpy.core.risk import RiskManager
from structpy.core.scenario import ScenarioGenerator


def _config() -> dict:
    return {
        "meta": {
            "isin": "RU000A108HM7",
            "name": "Multivariate Autocall",
            "type": "PhoenixAutocall",
        },
        "params": {
            "nominal": 1000.0,
            "currency": "RUB",
            "underlyings": ["SBER", "GAZP"],
            "feature_type": "standard",
            "memory_effect": True,
            "coupon_rate": 0.15,
            "barriers": {
                "autocall": 1.0,
                "coupon": 0.85,
                "capital_protection": 0.75,
            },
        },
    }


def _risk_manager(make_product):
    initial_prices = {"SBER": 300.0, "GAZP": 160.0}
    product = make_product(_config(), initial_prices)
    generator = ScenarioGenerator(
        tickers=["SBER", "GAZP"],
        initial_prices=initial_prices,
        volatilities={"SBER": 0.30, "GAZP": 0.35},
        risk_free_rate=0.15,
        correlation_matrix=np.array([[1.0, 0.6], [0.6, 1.0]]),
        dividend_yields={"SBER": 0.12, "GAZP": 0.10},
    )
    return RiskManager(Pricer(product, generator))


def test_risk_metrics_have_expected_bounds(make_product):
    risk_manager = _risk_manager(make_product)

    metrics = risk_manager.calculate_metrics(
        confidence_level=0.95,
        n_scenarios=500,
        n_steps=3,
        T=3.0,
        seed=42,
    )

    assert metrics["Confidence_Level"] == pytest.approx(0.95)
    assert metrics["Nominal_Investment"] == pytest.approx(1000.0)
    assert metrics["VaR"] >= 0
    assert metrics["Expected_Shortfall"] >= metrics["VaR"]
    assert metrics["Max_Loss"] >= metrics["Expected_Shortfall"]


def test_risk_manager_calculates_deltas_for_all_underlyings(make_product):
    risk_manager = _risk_manager(make_product)

    deltas = risk_manager.calculate_all_deltas(bump=0.01, n_scenarios=300)

    assert set(deltas) == {"SBER", "GAZP"}
    assert all(np.isfinite(value) for value in deltas.values())


def test_risk_manager_rejects_invalid_confidence_level(make_product):
    risk_manager = _risk_manager(make_product)

    with pytest.raises(ValueError):
        risk_manager.calculate_metrics(confidence_level=1.0)

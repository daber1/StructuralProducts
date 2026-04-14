from __future__ import annotations

import copy
from datetime import date

import numpy as np
import pytest

from structpy.core.pricer import Pricer
from structpy.core.scenario import ScenarioGenerator
from structpy.parser import parse_dict_config


def _build_product(config: dict, initial_prices: dict):
    product = parse_dict_config(copy.deepcopy(config))
    product.set_initial_prices(initial_prices)
    return product


def _year_fraction(start_date: str, observation_dates: list[str]) -> float:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(observation_dates[-1])
    return max((end - start).days / 365.0, 1.0 / 365.0)


def _build_generator(
    initial_prices: dict[str, float],
    volatilities: dict[str, float],
    correlation_matrix: np.ndarray,
):
    return ScenarioGenerator(
        tickers=list(initial_prices.keys()),
        initial_prices=initial_prices,
        volatilities=volatilities,
        risk_free_rate=0.10,
        correlation_matrix=correlation_matrix,
        dividend_yields={ticker: 0.0 for ticker in initial_prices},
    )


SUPPORTED_CASES = [
    pytest.param(
        {
            "label": "simple_barrier_note",
            "start_date": "2026-01-15",
            "observation_dates": ["2027-01-15"],
            "config": {
                "meta": {
                    "isin": "MANUAL-BARRIER",
                    "name": "Simple Barrier Note",
                    "type": "PhoenixAutocall",
                },
                "params": {
                    "nominal": 1000.0,
                    "currency": "RUB",
                    "underlyings": ["SBER"],
                    "feature_type": "standard",
                    "coupon_rate": 0.0,
                    "memory_effect": False,
                    "coupon_frequency": 1,
                    "barriers": {
                        "autocall": 10.0,
                        "coupon": 10.0,
                        "capital_protection": 0.70,
                    },
                },
            },
            "initial_prices": {"SBER": 100.0},
            "evaluation_path": [{"SBER": 75.0}],
            "volatilities": {"SBER": 0.20},
            "correlation_matrix": np.array([[1.0]]),
            "expected_action": "maturity",
            "expected_cashflows": [(1, 1000.0)],
            "expected_payout": 1000.0,
        },
        id="simple-barrier-note",
    ),
    pytest.param(
        {
            "label": "quarterly_autocall",
            "start_date": "2026-01-15",
            "observation_dates": [
                "2026-04-15",
                "2026-07-15",
                "2026-10-15",
                "2027-01-15",
            ],
            "config": {
                "meta": {
                    "isin": "MANUAL-AUTOCALL",
                    "name": "Quarterly Phoenix",
                    "type": "PhoenixAutocall",
                },
                "params": {
                    "nominal": 1000.0,
                    "currency": "RUB",
                    "underlyings": ["GAZP"],
                    "feature_type": "standard",
                    "coupon_rate": 0.12,
                    "memory_effect": True,
                    "coupon_frequency": 4,
                    "barriers": {
                        "autocall": 1.0,
                        "coupon": 0.80,
                        "capital_protection": 0.70,
                    },
                },
            },
            "initial_prices": {"GAZP": 100.0},
            "evaluation_path": [
                {"GAZP": 70.0},
                {"GAZP": 90.0},
                {"GAZP": 105.0},
                {"GAZP": 105.0},
            ],
            "volatilities": {"GAZP": 0.25},
            "correlation_matrix": np.array([[1.0]]),
            "expected_action": "autocall",
            "expected_cashflows": [(2, 60.0), (3, 30.0), (3, 1000.0)],
            "expected_payout": 1090.0,
        },
        id="quarterly-autocall",
    ),
    pytest.param(
        {
            "label": "basket_worst_of",
            "start_date": "2026-01-15",
            "observation_dates": ["2027-01-15"],
            "config": {
                "meta": {
                    "isin": "MANUAL-BASKET",
                    "name": "Worst-Of Basket Note",
                    "type": "PhoenixAutocall",
                },
                "params": {
                    "nominal": 1000.0,
                    "currency": "RUB",
                    "underlyings": ["SBER", "LKOH", "GAZP"],
                    "feature_type": "worst_of",
                    "coupon_rate": 0.0,
                    "memory_effect": False,
                    "coupon_frequency": 1,
                    "barriers": {
                        "autocall": 10.0,
                        "coupon": 10.0,
                        "capital_protection": 0.70,
                    },
                },
            },
            "initial_prices": {"SBER": 100.0, "LKOH": 100.0, "GAZP": 100.0},
            "evaluation_path": [{"SBER": 110.0, "LKOH": 95.0, "GAZP": 60.0}],
            "volatilities": {"SBER": 0.22, "LKOH": 0.18, "GAZP": 0.30},
            "correlation_matrix": np.array(
                [
                    [1.0, 0.45, 0.60],
                    [0.45, 1.0, 0.35],
                    [0.60, 0.35, 1.0],
                ]
            ),
            "expected_action": "maturity",
            "expected_cashflows": [(1, 600.0)],
            "expected_payout": 600.0,
        },
        id="basket-worst-of",
    ),
    pytest.param(
        {
            "label": "participation_note",
            "start_date": "2026-01-15",
            "observation_dates": ["2027-01-15"],
            "config": {
                "meta": {
                    "isin": "MANUAL-PARTICIPATION",
                    "name": "Participation Note 1.5x",
                    "type": "PhoenixAutocall",
                },
                "params": {
                    "nominal": 1000.0,
                    "currency": "RUB",
                    "underlyings": ["MOEX"],
                    "feature_type": "standard",
                    "coupon_rate": 0.0,
                    "memory_effect": False,
                    "coupon_frequency": 1,
                    "participation_rate": 1.5,
                    "strike": 1.0,
                    "barriers": {
                        "autocall": 10.0,
                        "coupon": 10.0,
                        "capital_protection": 1.0,
                    },
                },
            },
            "initial_prices": {"MOEX": 100.0},
            "evaluation_path": [{"MOEX": 120.0}],
            "volatilities": {"MOEX": 0.24},
            "correlation_matrix": np.array([[1.0]]),
            "expected_action": "maturity",
            "expected_cashflows": [(1, 1300.0)],
            "expected_payout": 1300.0,
        },
        id="participation-note",
    ),
]


@pytest.mark.parametrize("case", SUPPORTED_CASES)
def test_manual_supported_product_scenarios_parse_and_price(case):
    product = _build_product(case["config"], case["initial_prices"])

    evaluation_result = product.evaluate(case["evaluation_path"])
    assert evaluation_result["action"] == case["expected_action"]
    assert evaluation_result["cashflows"] == case["expected_cashflows"]
    assert evaluation_result["payout"] == pytest.approx(case["expected_payout"])

    generator = _build_generator(
        initial_prices=case["initial_prices"],
        volatilities=case["volatilities"],
        correlation_matrix=case["correlation_matrix"],
    )
    result = Pricer(product, generator).price(
        n_scenarios=64,
        n_steps=len(case["observation_dates"]),
        T=_year_fraction(case["start_date"], case["observation_dates"]),
        seed=42,
    )

    assert result["fair_value"] > 0.0
    assert result["std_error"] >= 0.0
    assert result["mean_payout_undiscounted"] > 0.0
    assert len(result["payouts"]) == 64

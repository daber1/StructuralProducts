import math

import numpy as np
import pytest
from scipy.stats import norm

from structpy.core.pricer import Pricer
from structpy.core.scenario import ScenarioGenerator


def _bs_put(s0: float, k: float, r: float, sigma: float, t: float) -> float:
    d1 = (math.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return k * math.exp(-r * t) * norm.cdf(-d2) - s0 * norm.cdf(-d1)


def _bond_like_config() -> dict:
    return {
        "meta": {"name": "Bond-like Note", "type": "PhoenixAutocall"},
        "params": {
            "nominal": 1000.0,
            "currency": "RUB",
            "underlyings": ["ASSET1"],
            "coupon_rate": 0.0,
            "barriers": {
                "autocall": 10.0,
                "coupon": 10.0,
                "capital_protection": 0.0,
            },
        },
    }


def _put_like_config() -> dict:
    return {
        "meta": {"name": "Put-like Note", "type": "PhoenixAutocall"},
        "params": {
            "nominal": 1000.0,
            "currency": "RUB",
            "underlyings": ["ASSET1"],
            "coupon_rate": 0.0,
            "barriers": {
                "autocall": 10.0,
                "coupon": 10.0,
                "capital_protection": 1.0,
            },
        },
    }


def _participation_config(participation_rate: float) -> dict:
    return {
        "meta": {"name": "Participation Note", "type": "PhoenixAutocall"},
        "params": {
            "nominal": 1000.0,
            "currency": "RUB",
            "underlyings": ["ASSET1"],
            "coupon_rate": 0.0,
            "participation_rate": participation_rate,
            "strike": 1.0,
            "barriers": {
                "autocall": 10.0,
                "coupon": 10.0,
                "capital_protection": 1.0,
            },
        },
    }


def _basket_participation_config(
    participation_rate: float,
    capital_protection: float = 0.0,
) -> dict:
    return {
        "meta": {"name": "Worst-Of Basket Participation Note", "type": "PhoenixAutocall"},
        "params": {
            "nominal": 1000.0,
            "currency": "RUB",
            "underlyings": ["ASSET1", "ASSET2", "ASSET3"],
            "feature_type": "worst_of",
            "coupon_rate": 0.0,
            "participation_rate": participation_rate,
            "strike": 1.0,
            "barriers": {
                "autocall": 10.0,
                "coupon": 10.0,
                "capital_protection": capital_protection,
            },
        },
    }


def _price_product(make_product, config: dict, s0: float, sigma: float, r: float, t: float):
    product = make_product(config, {"ASSET1": s0})
    generator = ScenarioGenerator(
        tickers=["ASSET1"],
        initial_prices={"ASSET1": s0},
        volatilities={"ASSET1": sigma},
        risk_free_rate=r,
    )
    pricer = Pricer(product, generator)
    return pricer.price(n_scenarios=12000, n_steps=24, T=t, seed=42)


def test_zero_coupon_limit_matches_discounted_nominal(make_product):
    r = 0.10
    t = 2.0

    result = _price_product(make_product, _bond_like_config(), 100.0, 0.2, r, t)
    expected = 1000.0 * math.exp(-r * t)

    assert result["fair_value"] == pytest.approx(expected, abs=2.0)


def test_put_like_limit_matches_bond_minus_put(make_product):
    s0 = 100.0
    r = 0.02
    sigma = 0.30
    t = 1.0

    result = _price_product(make_product, _put_like_config(), s0, sigma, r, t)
    put_price = _bs_put(s0, s0, r, sigma, t)
    expected = 1000.0 * math.exp(-r * t) - 1000.0 / s0 * put_price

    assert result["fair_value"] == pytest.approx(expected, abs=7.0)


def test_put_like_structure_value_decreases_with_higher_volatility(make_product):
    low_vol = _price_product(make_product, _put_like_config(), 100.0, 0.20, 0.0, 1.0)
    high_vol = _price_product(make_product, _put_like_config(), 100.0, 0.50, 0.0, 1.0)

    assert high_vol["fair_value"] < low_vol["fair_value"]


def test_participation_structure_value_increases_with_higher_participation(make_product):
    no_participation = _price_product(
        make_product, _participation_config(0.0), 100.0, 0.25, 0.02, 1.0
    )
    leveraged_participation = _price_product(
        make_product, _participation_config(1.5), 100.0, 0.25, 0.02, 1.0
    )

    assert leveraged_participation["fair_value"] > no_participation["fair_value"]


def _price_basket_product(
    make_product,
    config: dict,
    initial_prices: dict[str, float],
    volatilities: dict[str, float],
    correlation: float,
    r: float,
    t: float,
):
    tickers = list(initial_prices.keys())
    product = make_product(config, initial_prices)
    correlation_matrix = np.full((len(tickers), len(tickers)), correlation, dtype=float)
    np.fill_diagonal(correlation_matrix, 1.0)
    generator = ScenarioGenerator(
        tickers=tickers,
        initial_prices=initial_prices,
        volatilities=volatilities,
        risk_free_rate=r,
        correlation_matrix=correlation_matrix,
    )
    pricer = Pricer(product, generator)
    return pricer.price(n_scenarios=20000, n_steps=24, T=t, seed=42)


def test_worst_of_basket_participation_is_sensitive_to_correlation(make_product):
    initial_prices = {"ASSET1": 100.0, "ASSET2": 100.0, "ASSET3": 100.0}
    volatilities = {"ASSET1": 0.25, "ASSET2": 0.30, "ASSET3": 0.28}
    config = _basket_participation_config(participation_rate=1.0)

    low_corr = _price_basket_product(
        make_product, config, initial_prices, volatilities, correlation=0.2, r=0.02, t=1.0
    )
    high_corr = _price_basket_product(
        make_product, config, initial_prices, volatilities, correlation=0.8, r=0.02, t=1.0
    )
    assert high_corr["fair_value"] > low_corr["fair_value"]


def test_worst_of_basket_with_downside_is_sensitive_to_correlation(make_product):
    initial_prices = {"ASSET1": 100.0, "ASSET2": 100.0, "ASSET3": 100.0}
    volatilities = {"ASSET1": 0.25, "ASSET2": 0.30, "ASSET3": 0.28}
    config = _basket_participation_config(
        participation_rate=1.0,
        capital_protection=0.7,
    )

    low_corr = _price_basket_product(
        make_product, config, initial_prices, volatilities, correlation=0.2, r=0.02, t=1.0
    )
    high_corr = _price_basket_product(
        make_product, config, initial_prices, volatilities, correlation=0.8, r=0.02, t=1.0
    )
    assert high_corr["fair_value"] > low_corr["fair_value"]

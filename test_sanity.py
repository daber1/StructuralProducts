import math

import numpy as np
from scipy.stats import norm

from structpy.core.pricer import Pricer
from structpy.core.scenario import ScenarioGenerator
from structpy.parser.json_parser import load_product


def bs_put(s0, k, r, sigma, t):
    d1 = (math.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return k * math.exp(-r * t) * norm.cdf(-d2) - s0 * norm.cdf(-d1)


def test_phoenix_vs_black_scholes():

    json_config = """
    {
      "meta": {"name": "Black Scholes Put Test", "type": "PhoenixAutocall"},
      "params": {
        "nominal": 1000.0,
        "underlyings": ["ASSET1"],
        "coupon_rate": 0.0,
        "barriers": {
          "autocall": 10.0,
          "coupon": 0.0,
          "capital_protection": 1.0
        }
      }
    }
    """

    product = load_product(json_config)

    s0 = 100.0
    k = 100.0
    nominal = 1000.0

    product.set_initial_prices({"ASSET1": s0})

    r = 0.02
    sigma = 0.30
    t = 1.0

    generator = ScenarioGenerator(
        tickers=["ASSET1"],
        initial_prices={"ASSET1": s0},
        volatilities={"ASSET1": sigma},
        risk_free_rate=r,
    )

    pricer = Pricer(product, generator)
    mc = pricer.price(n_scenarios=200000, n_steps=50, T=t, seed=123)
    mc_price = mc["fair_value"]

    put = bs_put(s0, k, r, sigma, t)
    analytic_price = nominal * math.exp(-r * t) - nominal / s0 * put

    print("Monte Carlo:", mc_price)
    print("Black-Scholes:", analytic_price)

    error = abs(mc_price - analytic_price)
    print("Error:", error)

    assert error < 5.0


def run_test_disappearing_risk():
    # Дальние барьеры 
    json_config = """
    {
      "meta": {"name": "Test Note", "type": "PhoenixAutocall"},
      "params": {
        "nominal": 1000.0,
        "underlyings": ["ASSET1"],
        "coupon_rate": 0.3,
        "barriers": {
          "autocall": 10.0,
          "coupon": 10.0,
          "capital_protection": 0.0
        }
      }
    }
    """

    product = load_product(json_config)
    initial_prices = {"ASSET1": 100.0}
    product.set_initial_prices(initial_prices)

    r = 0.10
    t = 2.0
    generator = ScenarioGenerator(
        tickers=["ASSET1"],
        initial_prices=initial_prices,
        volatilities={"ASSET1": 0.2},
        risk_free_rate=r,
    )

    pricer = Pricer(product, generator)
    res = pricer.price(n_scenarios=2000, n_steps=12, T=t, seed=42)

    expected_fv = 1000.0 * np.exp(-r * t)
    actual_fv = res["fair_value"]

    print(f"Expected FV: {expected_fv:.2f}")
    print(f"Actual FV:   {actual_fv:.2f}")
    print(f"Difference:  {abs(actual_fv - expected_fv):.4f}")

    assert abs(actual_fv - expected_fv) < 1.0, "Тест 1 провален"

def run_test_vega_sensitivity():
    # Чувствительность к веге

    json_config = """
    {
      "meta": {"name": "Test Note", "type": "PhoenixAutocall"},
      "params": {
        "nominal": 1000.0,
        "underlyings": ["ASSET1"],
        "coupon_rate": 0.5,
        "barriers": {
          "autocall": 10.0,
          "coupon": 1.5,
          "capital_protection": 0.0
        }
      }
    }
    """

    product = load_product(json_config)
    initial_prices = {"ASSET1": 100.0}
    product.set_initial_prices(initial_prices)

    r = 0.05
    t = 1.0

    vols = [0.1, 0.2, 0.3, 0.4, 0.5]
    fvs = []

    for vol in vols:
        generator = ScenarioGenerator(
            tickers=["ASSET1"],
            initial_prices=initial_prices,
            volatilities={"ASSET1": vol},
            risk_free_rate=r,
        )
        pricer = Pricer(product, generator)
        res = pricer.price(n_scenarios=2000, n_steps=12, T=t, seed=42)
        fvs.append(res["fair_value"])
        print(f"Vol: {vol:.1f} -> FV: {res['fair_value']:.2f}")

    is_increasing = all(left < right for left, right in zip(fvs, fvs[1:]))
    print(f"Value increases with volatility: {is_increasing}")
    if is_increasing:
        print("Test passed")


if __name__ == "__main__":
    run_test_disappearing_risk()
    run_test_vega_sensitivity()
    test_phoenix_vs_black_scholes()

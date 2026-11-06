"""Run an offline calculation using only the installed structpy package."""

import json

from structpy.core.pricer import Pricer
from structpy.core.risk import RiskManager
from structpy.core.scenario import ScenarioGenerator
from structpy.parser import parse_dict_config


def run_pricing() -> dict:
    config = {
        "meta": {
            "isin": "DEMO-PHOENIX",
            "name": "One-year educational Phoenix note",
            "type": "PhoenixAutocall",
        },
        "params": {
            "nominal": 1000.0,
            "currency": "RUB",
            "underlyings": ["ASSET"],
            "feature_type": "worst_of",
            "coupon_rate": 0.12,
            "coupon_frequency": 4,
            "memory_effect": True,
            "barriers": {
                "autocall": 1.0,
                "coupon": 0.8,
                "capital_protection": 0.7,
            },
        },
    }
    initial_prices = {"ASSET": 100.0}
    product = parse_dict_config(config)
    product.set_initial_prices(initial_prices)

    generator = ScenarioGenerator(
        tickers=["ASSET"],
        initial_prices=initial_prices,
        volatilities={"ASSET": 0.25},
        risk_free_rate=0.05,
        dividend_yields={"ASSET": 0.0},
    )
    pricer = Pricer(product, generator)
    simulation = {"n_scenarios": 1000, "n_steps": 4, "T": 1.0, "seed": 42}

    pricing = pricer.price(**simulation)
    pricing.pop("payouts")
    risk = RiskManager(pricer).calculate_metrics(**simulation)
    return {"isin": product.isin, "pricing": pricing, "risk": risk}


if __name__ == "__main__":
    print(json.dumps(run_pricing(), indent=2))

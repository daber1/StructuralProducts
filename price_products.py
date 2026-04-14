import json
from typing import Any, Dict

import numpy as np

from structpy.core.pricer import Pricer
from structpy.core.risk import RiskManager
from structpy.core.scenario import ScenarioGenerator
from structpy.parser.json_parser import load_product


def print_banner() -> None:
    print("=" * 60)
    print("      STRUCTURAL PRODUCTS PRICING ENGINE v2.0")
    print("=" * 60)


def price_product_workflow(
    config_data: Dict[str, Any], market_params: Dict[str, Any]
) -> Dict[str, Any]:
    product = load_product(json.dumps(config_data))

    initial_prices = market_params.get("initial_prices", {})
    if not initial_prices:
        initial_prices = {asset.ticker: 1000.0 for asset in product.underlyings}

    product.set_initial_prices(initial_prices)

    correlation_matrix = market_params.get("correlation_matrix")
    if correlation_matrix is not None:
        correlation_matrix = np.array(correlation_matrix, dtype=float)

    generator = ScenarioGenerator(
        tickers=[asset.ticker for asset in product.underlyings],
        initial_prices=dict(initial_prices),
        volatilities=market_params.get("volatilities", {}),
        risk_free_rate=market_params.get("risk_free_rate", 0.15),
        correlation_matrix=correlation_matrix,
        dividend_yields=market_params.get("dividend_yields", {}),
    )

    pricer = Pricer(product, generator)
    n_scenarios = int(market_params.get("n_scenarios", 10000))
    T = float(market_params.get("T", 3.0))
    n_steps = int(market_params.get("n_steps", int(T * 4)))

    print(f"[*] Running Monte Carlo simulation with {n_scenarios} scenarios...")
    pricing_result = pricer.price(n_scenarios=n_scenarios, n_steps=n_steps, T=T, seed=42)

    risk_manager = RiskManager(pricer)
    risk_metrics = risk_manager.calculate_metrics(
        n_scenarios=n_scenarios,
        n_steps=n_steps,
        T=T,
        seed=42,
    )
    deltas = risk_manager.calculate_all_deltas(n_scenarios=2000)

    return {
        "product_name": product.name,
        "isin": product.isin,
        "pricing": pricing_result,
        "risk": risk_metrics,
        "deltas": deltas,
    }


def main() -> None:
    print_banner()

    example_config = {
        "meta": {
            "isin": "RU000A108HM7",
            "name": "Phoenix Autocall High Yield",
            "type": "PhoenixAutocall",
        },
        "params": {
            "nominal": 1000.0,
            "currency": "RUB",
            "underlyings": ["SBER", "YDEX", "TCSG"],
            "coupon_rate": 0.25,
            "memory_effect": True,
            "coupon_frequency": 4,
            "barriers": {
                "autocall": 1.0,
                "coupon": 0.8,
                "capital_protection": 0.7,
            },
        },
    }

    market_params = {
        "initial_prices": {"SBER": 280.0, "YDEX": 4000.0, "TCSG": 3000.0},
        "volatilities": {"SBER": 0.35, "YDEX": 0.45, "TCSG": 0.40},
        "risk_free_rate": 0.16,
        "dividend_yields": {"SBER": 0.10, "YDEX": 0.0, "TCSG": 0.05},
        "correlation_matrix": [[1.0, 0.4, 0.5], [0.4, 1.0, 0.3], [0.5, 0.3, 1.0]],
        "n_scenarios": 10000,
        "T": 3.0,
    }

    try:
        report = price_product_workflow(example_config, market_params)

        print("\n" + "-" * 60)
        print(f"PRODUCT REPORT: {report['product_name']} ({report['isin']})")
        print("-" * 60)

        pricing = report["pricing"]
        risk = report["risk"]
        deltas = report["deltas"]

        print(
            f"Fair Value:           {pricing['fair_value']:.2f} RUB "
            f"({pricing['fair_value'] / 1000:.2%})"
        )
        print(f"Std Error:            {pricing['std_error']:.4f}")
        print(f"Mean Payout (Gross):  {pricing['mean_payout_undiscounted']:.2f} RUB")
        print("-" * 30)
        print(f"VaR (95%):            {risk['VaR']:.2f} RUB")
        print(f"Expected Shortfall:   {risk['Expected_Shortfall']:.2f} RUB")
        print(f"Maximum Loss:         {risk['Max_Loss']:.2f} RUB")
        print("-" * 30)
        print("SENSITIVITIES (Delta):")
        for ticker, value in deltas.items():
            print(f"  d(V)/d({ticker}):   {value:+.4f}")
        print("-" * 60)

        report_to_save = report.copy()
        report_to_save["pricing"] = report["pricing"].copy()
        report_to_save["pricing"].pop("payouts", None)

        with open("pricing_report.json", "w", encoding="utf-8") as handle:
            json.dump(report_to_save, handle, indent=4)

        print("[!] Detailed report saved to 'pricing_report.json'")
    except Exception as exc:
        print(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()

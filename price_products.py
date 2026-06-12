import json
from typing import Any, Dict, Iterable

import numpy as np

from structpy.core.pricer import Pricer
from structpy.core.risk import RiskManager
from structpy.core.scenario import ScenarioGenerator
from structpy.parser.json_parser import load_product


def print_banner() -> None:
    print("Structural Products Pricing Engine")


def run_convergence_analysis(
    pricer: Pricer,
    scenario_counts: list[int],
    *,
    n_steps: int,
    T: float,
    seed: int = 42,
) -> list[Dict[str, Any]]:
    """Функция для запуска нескольких прайсеров с разным количеством генераций."""
    results = []
    for n_scenarios in scenario_counts:
        result = pricer.price(
            n_scenarios=int(n_scenarios),
            n_steps=n_steps,
            T=T,
            seed=seed,
        )
        ci_low, ci_high = result["confidence_interval_95"]
        results.append(
            {
                "n_scenarios": int(n_scenarios),
                "fair_value": result["fair_value"],
                "std_error": result["std_error"],
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
            }
        )
    return results


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

    print(f"Running Monte Carlo simulation with {n_scenarios} scenarios...")
    pricing_result = pricer.price(n_scenarios=n_scenarios, n_steps=n_steps, T=T, seed=42)

    risk_manager = RiskManager(pricer)
    risk_metrics = risk_manager.calculate_metrics(
        n_scenarios=n_scenarios,
        n_steps=n_steps,
        T=T,
        seed=42,
    )
    deltas = risk_manager.calculate_all_deltas(n_scenarios=2000)
    convergence = run_convergence_analysis(
        pricer,
        market_params.get("convergence_scenarios", [1000, 5000, 10000, 25000, 50000]),
        n_steps=n_steps,
        T=T,
        seed=42,
    )

    return {
        "product_name": product.name,
        "isin": product.isin,
        "pricing": pricing_result,
        "risk": risk_metrics,
        "deltas": deltas,
        "convergence": convergence,
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
        "convergence_scenarios": [1000, 5000, 10000, 25000, 50000],
        "T": 3.0,
    }

    try:
        report = price_product_workflow(example_config, market_params)

        print(f"\nProduct: {report['product_name']} ({report['isin']})")

        pricing = report["pricing"]
        risk = report["risk"]
        deltas = report["deltas"]
        ci_low, ci_high = pricing["confidence_interval_95"]

        print(f"Fair value: {pricing['fair_value']:.2f} RUB ({pricing['fair_value'] / 1000:.2%})")
        print(f"95% CI: [{ci_low:.2f}, {ci_high:.2f}] RUB")
        print(f"Standard error: {pricing['std_error']:.4f}")
        print(f"Mean payout gross: {pricing['mean_payout_undiscounted']:.2f} RUB")
        print(
            "Risk metrics: "
            f"VaR 95% = {risk['VaR']:.2f}, "
            f"ES = {risk['Expected_Shortfall']:.2f}, "
            f"Max loss = {risk['Max_Loss']:.2f}"
        )
        print(
            "Deltas: "
            + ", ".join(f"{ticker}={value:+.4f}" for ticker, value in deltas.items())
        )
        print("\nMonte Carlo convergence:")
        for row in report["convergence"]:
            print(
                f"  {row['n_scenarios']:>5} scenarios: "
                f"FV={row['fair_value']:.2f}, "
                f"SE={row['std_error']:.4f}, "
                f"95% CI=[{row['ci_95_low']:.2f}, {row['ci_95_high']:.2f}]"
            )

        report_to_save = report.copy()
        report_to_save["pricing"] = report["pricing"].copy()
        report_to_save["pricing"].pop("payouts", None)

        with open("pricing_report.json", "w", encoding="utf-8") as handle:
            json.dump(report_to_save, handle, indent=4)

        print("Detailed report saved to pricing_report.json")
    except Exception as exc:
        print(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()

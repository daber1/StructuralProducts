from structpy.parser.json_parser import load_product
from structpy.core.scenario import ScenarioGenerator
from structpy.core.pricer import Pricer
import numpy as np

json_config = """
{
  "meta": {
    "isin": "RU000A108HJ3",
    "type": "PhoenixAutocall",
    "name": "SBER-YDEX Phoenix"
  },
  "params": {
    "nominal": 1000.0,
    "currency": "RUB",
    "underlyings": ["SBER", "YDEX"],
    "feature_type": "standard",
    "coupon_rate": 0.28,
    "memory_effect": true,
    "coupon_frequency": 4,
    "barriers": {
      "autocall": 1.00,
      "coupon": 1.00,
      "capital_protection": 1.00
    }
  }
}
"""

def main():
    product = load_product(json_config)
    
    initial_prices = {"SBER": 280.0, "YDEX": 4000.0}
    product.set_initial_prices(initial_prices)
    
    volatilities = {"SBER": 0.30, "YDEX": 0.40}
    risk_free_rate = 0.16 
    dividend_yields = {"SBER": 0.10, "YDEX": 0.0}
    
    correlation_matrix = np.array([
        [1.0, 0.5],
        [0.5, 1.0]
    ])
    
    generator = ScenarioGenerator(
        tickers=["SBER", "YDEX"],
        initial_prices=initial_prices,
        volatilities=volatilities,
        risk_free_rate=risk_free_rate,
        correlation_matrix=correlation_matrix,
        dividend_yields=dividend_yields
    )
    
    pricer = Pricer(product, generator)
    
    n_scenarios = 10000
    T = 3.0
    n_steps = int(T * 4) # 4 quarters per year
    
    print(f"--- Pricing Product: {product.isin} ---")
    print(f"Underlyings: {', '.join(initial_prices.keys())}")
    print(f"Simulation: {n_scenarios} scenarios, {T} years, {n_steps} steps")
    
    result = pricer.price(n_scenarios=n_scenarios, n_steps=n_steps, T=T, seed=42)
    
    print("\nResults:")
    print(f"Fair Value: {result['fair_value']:.2f} RUB")
    print(f"Mean Payout (Undiscounted): {result['mean_payout_undiscounted']:.2f} RUB")
    print(f"Relative Fair Value: {result['fair_value'] / 1000.0:.2%}")

if __name__ == "__main__":
    main()

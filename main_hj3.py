from structpy.parser.json_parser import load_product


json_config = """
{
  "meta": {
    "isin": "RU000A108HJ3",
    "type": "PhoenixAutocall"
  },
  "params": {
    "nominal": 1000,
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


def main() -> None:
    product = load_product(json_config)
    product.set_initial_prices({"SBER": 1000.0, "YDEX": 1000.0})

    series_of_prices = [
        {"SBER": 900.0, "YDEX": 950.0},
        {"SBER": 950.0, "YDEX": 900.0},
        {"SBER": 800.0, "YDEX": 850.0},
        {"SBER": 990.0, "YDEX": 990.0},
        {"SBER": 1050.0, "YDEX": 1100.0},
        {"SBER": 1200.0, "YDEX": 1200.0},
    ]

    result = product.evaluate(series_of_prices)

    print(f"Product: {product.isin}")
    print(f"Performance (Average): {result['perf_avg']:.2%}")
    print(f"Performance (Worst): {result['perf_worst']:.2%}")
    print(f"Action: {result['action']}")
    print(f"Payout: {result['payout']} RUB")


if __name__ == "__main__":
    main()

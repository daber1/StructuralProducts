from structpy.parser.json_parser import load_product


json_config = """
{
  "meta": {
    "isin": "RU000A108HM7",
    "type": "PhoenixAutocall"
  },
  "params": {
    "nominal": 1000,
    "underlyings": ["NVTK", "SBER", "TCSG", "ALRS", "YDEX"],
    "feature_type": "hybrid_avg_worst",
    "coupon_rate": 0.3,
    "memory_effect": true,
    "coupon_frequency": 4,
    "barriers": {
      "autocall": 1.10,
      "coupon": 1.10,
      "capital_protection": 0.90
    }
  }
}
"""


def main() -> None:
    product = load_product(json_config)

    initial_prices = {
        "NVTK": 1000.0,
        "SBER": 1000.0,
        "TCSG": 1000.0,
        "ALRS": 1000.0,
        "YDEX": 1000.0,
    }
    product.set_initial_prices(initial_prices)

    series_of_prices = [
        {"NVTK": 900.0, "SBER": 900.0, "TCSG": 900.0, "ALRS": 900.0, "YDEX": 900.0},
        {"NVTK": 900.0, "SBER": 900.0, "TCSG": 900.0, "ALRS": 900.0, "YDEX": 900.0},
        {"NVTK": 1100.0, "SBER": 1100.0, "TCSG": 1100.0, "ALRS": 1100.0, "YDEX": 1100.0},
        {"NVTK": 1100.0, "SBER": 1100.0, "TCSG": 1100.0, "ALRS": 1100.0, "YDEX": 1100.0},
        {"NVTK": 1100.0, "SBER": 1100.0, "TCSG": 1100.0, "ALRS": 1100.0, "YDEX": 1100.0},
    ]

    result = product.evaluate(series_of_prices)

    print(f"Product: {product.isin}")
    print(f"Performance (Average): {result['perf_avg']:.2%}")
    print(f"Performance (Worst): {result['perf_worst']:.2%}")
    print(f"Action: {result['action']}")
    print(f"Payout: {result['payout']} RUB")


if __name__ == "__main__":
    main()

import json
from structpy.parser.json_parser import load_product
from structpy.data.mock_provider import MockDataProvider
import structpy.products 

# json_config = """
# {
#   "meta": { 
#     "isin": "RU000A108HM7", 
#     "type": "PhoenixAutocall" 
#   },
#   "params": {
#     "nominal": 1000,
#     "underlyings": ["NVTK", "SBER", "TCSG", "ALRS", "YDEX"],
#     "feature_type": "hybrid_avg_worst", 
#     "coupon_rate": 0.30, 
#     "barriers": {
#       "autocall": 1.10,
#       "coupon": 1.10,
#       "capital_protection": 0.90
#     }
#   }
# }
# """
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
    "barriers": {
      "autocall": 1.00,
      "coupon": 1.00,
      "capital_protection": 1.00
    }
  }
}
"""
# 2. Загрузка продукта
product = load_product(json_config)

initial_prices = {
    "NVTK": 1000, "SBER": 1000, "TCSG": 1000, "ALRS": 1000, "YDEX": 1000
}
product.set_initial_prices(initial_prices)

current_prices = {
    "NVTK": 1000,   # 0.9
    "SBER": 1000,   # 1.4
    "TCSG": 1000,  # 1.1
    "ALRS": 1000,    # 1.1
    "YDEX": 1000   # 1.1
}

data_provider = MockDataProvider(current_prices)
prices = data_provider.get_prices([a.ticker for a in product.underlyings])

# 5. Оценка
result = product.evaluate(prices, is_final_date=True)

print(f"Product: {product.isin}")
print(f"Performance (Average): {result['perf_avg']:.2%}")
print(f"Performance (Worst): {result['perf_worst']:.2%}")
print(f"Action: {result['action']}")
print(f"Payout: {result['payout']} RUB")
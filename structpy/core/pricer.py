from typing import Any, Dict

import numpy as np

from .product import StructuralProduct
from .scenario import ScenarioGenerator


class Pricer:
    def __init__(self, product: StructuralProduct, generator: ScenarioGenerator):
        self.product = product
        self.generator = generator

    def price(
        self,
        n_scenarios: int = 10000,
        n_steps: int = 12,
        T: float = 3.0,
        seed: int | None = None,
        initial_missed_coupons: int = 0,
    ) -> Dict[str, Any]:
        if n_scenarios <= 0:
            raise ValueError("n_scenarios > 0")
        if n_steps <= 0:
            raise ValueError("n_steps > 0")
        if T <= 0:
            raise ValueError("T > 0")

        paths = self.generator.generate_paths(n_scenarios, n_steps, T, seed)

        total_payouts = np.zeros(n_scenarios)
        discounted_payouts = np.zeros(n_scenarios)
        discount_rate = self.generator.risk_free_rate

        for i in range(n_scenarios):
            scenario_data = []
            for step in range(1, n_steps + 1):
                step_prices = {
                    asset.ticker: paths[asset.ticker][i, step]
                    for asset in self.product.underlyings
                }
                scenario_data.append(step_prices)

            result = self.product.evaluate(
                scenario_data,
                initial_missed_coupons=initial_missed_coupons,
            )

            payout_sum = 0.0
            discounted_sum = 0.0

            for cf_step, cf_amount in result.get("cashflows", []):
                time_to_payout = (cf_step / n_steps) * T
                discount_factor = np.exp(-discount_rate * time_to_payout)

                payout_sum += cf_amount
                discounted_sum += cf_amount * discount_factor

            total_payouts[i] = payout_sum
            discounted_payouts[i] = discounted_sum

        fair_value = float(np.mean(discounted_payouts))
        std_error = float(np.std(discounted_payouts) / np.sqrt(n_scenarios))
        ci_half_width = 1.96 * std_error

        return {
            "fair_value": fair_value,
            "std_error": std_error,
            "confidence_interval_95": (
                float(fair_value - ci_half_width),
                float(fair_value + ci_half_width),
            ),
            "mean_payout_undiscounted": float(np.mean(total_payouts)),
            "payouts": total_payouts.tolist() if n_scenarios <= 1000 else [],
        }

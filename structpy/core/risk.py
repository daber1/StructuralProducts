from typing import Dict, Optional

import numpy as np

from structpy.core.pricer import Pricer


class RiskManager:
    def __init__(self, pricer: Pricer):
        self.pricer = pricer

    def calculate_metrics(
        self,
        confidence_level: float = 0.95,
        n_scenarios: int = 10000,
        n_steps: int = 12,
        T: float = 3.0,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if n_scenarios <= 0:
            raise ValueError("n_scenarios must be positive")

        paths = self.pricer.generator.generate_paths(n_scenarios, n_steps, T, seed)

        total_payouts = np.zeros(n_scenarios)
        nominal_investment = self.pricer.product.nominal

        for i in range(n_scenarios):
            scenario_data = []
            for step in range(1, n_steps + 1):
                step_prices = {
                    asset.ticker: paths[asset.ticker][i, step]
                    for asset in self.pricer.product.underlyings
                }
                scenario_data.append(step_prices)

            result = self.pricer.product.evaluate(scenario_data)
            total_payouts[i] = float(result.get("payout", 0.0))

        pnls = total_payouts - nominal_investment
        losses = -pnls
        sorted_losses = np.sort(losses)

        var_index = max(0, int(np.ceil(confidence_level * n_scenarios)) - 1)
        var_index = min(var_index, n_scenarios - 1)

        var_value = float(sorted_losses[var_index])
        tail_losses = sorted_losses[var_index:]
        es_value = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_value

        return {
            "VaR": max(0.0, var_value),
            "Expected_Shortfall": max(0.0, es_value),
            "Confidence_Level": confidence_level,
            "Max_Loss": float(np.max(losses)),
            "Nominal_Investment": float(nominal_investment),
        }

    def calculate_delta(
        self,
        ticker: str,
        bump: float = 0.01,
        n_scenarios: int = 5000,
        n_steps: int = 12,
        T: float = 3.0,
        seed: int = 42,
    ) -> float:
        if bump <= 0:
            raise ValueError("bump must be positive")

        base_price = self.pricer.price(n_scenarios, n_steps, T, seed=seed)["fair_value"]

        original_initial_price = self.pricer.generator.initial_prices[ticker]
        self.pricer.generator.initial_prices[ticker] = original_initial_price * (1 + bump)
        try:
            up_price = self.pricer.price(n_scenarios, n_steps, T, seed=seed)["fair_value"]
        finally:
            self.pricer.generator.initial_prices[ticker] = original_initial_price

        delta = (up_price - base_price) / (original_initial_price * bump)
        return float(delta)

    def calculate_all_deltas(
        self,
        bump: float = 0.01,
        n_scenarios: int = 5000,
    ) -> Dict[str, float]:
        deltas = {}
        for asset in self.pricer.product.underlyings:
            deltas[asset.ticker] = self.calculate_delta(
                asset.ticker,
                bump,
                n_scenarios=n_scenarios,
            )
        return deltas

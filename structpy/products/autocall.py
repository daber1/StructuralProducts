from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from structpy.core.asset import Asset
from structpy.core.product import StructuralProduct
from structpy.parser.json_parser import ProductFactory


class PhoenixAutocallProduct(StructuralProduct):
    def __init__(
        self,
        isin: str,
        name: str,
        nominal: float,
        currency: str,
        coupon_rate: float,
        barriers: Dict[str, float],
        feature_type: str = "standard",
        memory_effect: bool = False,
        coupon_frequency: int = 1,
        participation_rate: float = 0.0,
        strike: float = 1.0,
    ):
        super().__init__(isin, name, nominal, currency)
        self.coupon_rate = coupon_rate
        self.barriers = barriers
        self.feature_type = feature_type
        self.memory_effect = memory_effect
        self.coupon_frequency = coupon_frequency
        self.participation_rate = participation_rate
        self.strike = strike

    @classmethod
    def from_config(cls, config: dict):
        params = config["params"]
        meta = config["meta"]
        instance = cls(
            isin=meta.get("isin", "Unknown"),
            name=meta.get("name", "Unknown"),
            nominal=float(params.get("nominal", 1000.0)),
            currency=params.get("currency", "RUB"),
            coupon_rate=float(params.get("coupon_rate", 0.0)),
            barriers=params.get("barriers", {}),
            feature_type=params.get("feature_type", "standard"),
            memory_effect=bool(params.get("memory_effect", False)),
            coupon_frequency=int(params.get("coupon_frequency", 1)),
            participation_rate=float(params.get("participation_rate", 0.0)),
            strike=float(params.get("strike", 1.0)),
        )
        for ticker in params.get("underlyings", []):
            instance.underlyings.append(Asset(ticker))
        return instance

    def _redemption_amount(self, trigger_metric: float, risk_metric: float) -> float:
        capital_protection = float(self.barriers.get("capital_protection", 0.9))
        # Перформанс ниже уровня защиты капитала - инвестор получает пропорционально этому уровню
        if risk_metric < capital_protection:
            return self.nominal * risk_metric

        redemption_amount = self.nominal
        # Если есть участие в росте и перформанс выше страйка, добавляем дополнительную выплату
        if self.participation_rate > 0.0 and trigger_metric > self.strike:
            upside = self.participation_rate * (trigger_metric - self.strike)
            redemption_amount += self.nominal * upside

        return redemption_amount

    def _performance_metrics(
        self, current_prices: Dict[str, float]
    ) -> Tuple[float, float, float, float]:
        """Считает показатели для текущих цен активов."""
        if not self.underlyings:
            raise ValueError("У продукта нет базовых активов.")

        performances: List[float] = []
        for asset in self.underlyings:
            if asset.ticker not in self.initial_prices:
                raise ValueError(
                    f"Нет изначальной цены для актива с тикером '{asset.ticker}'."
                )

            initial_price = float(self.initial_prices[asset.ticker])
            if initial_price <= 0:
                raise ValueError(
                    f"Изначальная цена для актива с тикером '{asset.ticker}' должна быть положительной."
                )

            current_price = float(current_prices.get(asset.ticker, initial_price))
            performances.append(current_price / initial_price)

        perf_worst = min(performances)
        perf_average = float(np.mean(performances))

        if self.feature_type == "hybrid_avg_worst":
            trigger_metric = perf_average
            risk_metric = perf_worst
        elif self.feature_type in {"standard", "worst_of"}:
            trigger_metric = perf_worst
            risk_metric = perf_worst
        elif self.feature_type == "average":
            trigger_metric = perf_average
            risk_metric = perf_average
        else:
            raise ValueError(f"Неизвестный feature_type: {self.feature_type}")

        return perf_worst, perf_average, trigger_metric, risk_metric

    def _evaluate_path(
        self, market_data_series: Sequence[Dict[str, float]], initial_missed_coupons: int
    ) -> Dict[str, Any]:
        cashflows = []
        missed_coupons = initial_missed_coupons
        total_payout = 0.0

        for step_idx, current_prices in enumerate(market_data_series):
            step = step_idx + 1
            is_final_date = step == len(market_data_series)

            perf_worst, perf_average, trigger_metric, risk_metric = self._performance_metrics(
                current_prices
            )

            coupon_barrier = float(self.barriers.get("coupon", 1.0))
            autocall_level = float(self.barriers.get("autocall", 1.0))
            # Если достигнут барьер для купона, начисляем купон с учетом эффекта памяти
            if trigger_metric >= coupon_barrier:
                multiplier = (missed_coupons + 1) if self.memory_effect else 1
                periodic_coupon_rate = self.coupon_rate / float(self.coupon_frequency)
                coupon_amount = self.nominal * periodic_coupon_rate * float(multiplier)
                cashflows.append((step, coupon_amount))
                total_payout += coupon_amount
                missed_coupons = 0
            else:
                missed_coupons += 1
            # Проверка автокола
            if not is_final_date and trigger_metric >= autocall_level:
                cashflows.append((step, self.nominal))
                total_payout += self.nominal
                return {
                    "action": "autocall",
                    "step": step,
                    "payout": float(total_payout),
                    "cashflows": cashflows,
                    "perf_worst": perf_worst,
                    "perf_avg": perf_average,
                }
            # Экспирация
            if is_final_date:
                redemption_amount = self._redemption_amount(
                    trigger_metric=trigger_metric,
                    risk_metric=risk_metric,
                )

                cashflows.append((step, redemption_amount))
                total_payout += redemption_amount

                return {
                    "action": "maturity",
                    "step": step,
                    "payout": float(total_payout),
                    "cashflows": cashflows,
                    "perf_worst": perf_worst,
                    "perf_avg": perf_average,
                }

        return {
            "action": "hold",
            "payout": float(total_payout),
            "cashflows": cashflows,
            "perf_worst": 1.0,
            "perf_avg": 1.0,
        }

    def evaluate(
        self,
        market_data_series: Sequence[Dict[str, float]],
        initial_missed_coupons: int = 0,
    ) -> Dict[str, Any]:
        if market_data_series is None:
            raise ValueError("Необходимо указать market_data_series")
        if isinstance(market_data_series, dict):
            raise TypeError(
                "Метод evaluate ожидает последовательность словарей с ценами"
            )

        market_path = list(market_data_series)
        if not market_path:
            raise ValueError("market_data_series должно содержать хотя бы одно наблюдение")

        return self._evaluate_path(
            market_path,
            initial_missed_coupons=initial_missed_coupons,
        )


ProductFactory.register("PhoenixAutocall", PhoenixAutocallProduct)

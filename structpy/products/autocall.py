from typing import Dict, List
import numpy as np
from ..core.product import StructuralProduct
from ..core.asset import Asset
from ..parser.json_parser import ProductFactory

class PhoenixAutocallProduct(StructuralProduct):
    
    def __init__(self, isin, name, nominal, currency, 
                 coupon_rate, barriers, feature_type="standard"):
        super().__init__(isin, name, nominal, currency)
        self.coupon_rate = coupon_rate
        self.barriers = barriers # dict: {'autocall': 1.0, 'coupon': 0.8, ...}
        self.feature_type = feature_type

    @classmethod
    def from_config(cls, config: dict):
        p = config['params']
        meta = config['meta']
        instance = cls(
            isin=meta['isin'],
            name=meta.get('name', 'Unknown'),
            nominal=p['nominal'],
            currency=p.get('currency', 'RUB'),
            coupon_rate=p['coupon_rate'],
            barriers=p['barriers'],
            feature_type=p.get('feature_type', 'standard')
        )
        for ticker in p['underlyings']:
            instance.underlyings.append(Asset(ticker))
        return instance

    def evaluate(self, current_prices: Dict[str, float], is_final_date=False,  years_held=3.0):
        """
        Оценка состояния продукта на дату наблюдения.
        
        :param current_prices: Словарь текущих цен {ticker: price}
        :param is_final_date: Флаг даты погашения (Maturity)
        :param years_held: Сколько лет прошло с начала (для расчета годового купона)
        """
        # 1. Считаем перформанс каждой бумаги: P_current / P_initial
        perfs = []
        for asset in self.underlyings:
            # Если актив упал на 30%, perf = 0.70
            p0 = self.initial_prices[asset.ticker]
            pt = current_prices.get(asset.ticker, p0) # safe get
            perfs.append(pt / p0)
        
        # 2. Вычисляем метрики
        val_worst = min(perfs)
        val_average = np.mean(perfs)

        result = {
            "perf_worst": val_worst,
            "perf_avg": val_average,
            "payout": 0.0,
            "action": "hold"
        }

        # --- ЛОГИКА RU000A108HM7 (Hybrid) ---
        if self.feature_type == 'hybrid_avg_worst':
            # Основная метрика для Купона/Автоколла - СРЕДНЕЕ
            trigger_metric = val_average
            # Метрика для защиты капитала - ХУДШЕЕ
            risk_metric = val_worst
        else:
            # Стандартная (обычно Worst of)
            # Здесь можно добавить параметр barrier_metric в конфиг, 
            # но по умолчанию берем worst для надежности
            trigger_metric = val_worst
            risk_metric = val_worst

        # 1. Проверка Автоколла (досрочное погашение)
        # Если сегодня не финал и метрика выше барьера автоколла
        autocall_level = self.barriers.get('autocall', 1.0)
        if not is_final_date and trigger_metric >= autocall_level:
            result['action'] = 'autocall'
            # Номинал + Купон
            result['payout'] = self.nominal * (1 + self.coupon_rate * years_held)
            return result

        # 2. Финальное погашение
        if is_final_date:
            result['action'] = 'maturity'
            
            # Сценарий А: Выполнилось условие доходности (Сценарий > 110%)
            # Для HM7: Если Среднее >= 110%
            coupon_barrier = self.barriers.get('coupon', 1.0)
            
            if trigger_metric >= coupon_barrier:
                # Номинал + Купон
                result['payout'] = self.nominal * (1 + self.coupon_rate * years_held)
            
            # Сценарий Б: Дохода нет, проверяем защиту капитала
            # Для HM7: Если Среднее < 110%, но Худший >= 90% -> Вернуть Номинал
            elif risk_metric >= self.barriers.get('capital_protection', 0.9):
                result['payout'] = self.nominal
            
            # Сценарий В: Защита пробита (Худший < 90%)
            # Выплата по Худшему
            else:
                result['payout'] = self.nominal * risk_metric 
                
        return result

# Регистрация класса в фабрике
ProductFactory.register("PhoenixAutocall", PhoenixAutocallProduct)
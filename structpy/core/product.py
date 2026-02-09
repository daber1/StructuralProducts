from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datetime import date
from .asset import Asset

class StructuralProduct(ABC):
    def __init__(self, isin: str, name: str, nominal: float, currency: str):
        self.isin = isin
        self.name = name
        self.nominal = nominal
        self.currency = currency
        self.underlyings: List[Asset] = []
        self.initial_prices: Dict[str, float] = {}

    def set_initial_prices(self, prices: Dict[str, float]):
        """Фиксация начальных цен (Strike levels)"""
        self.initial_prices = prices

    @abstractmethod
    def evaluate(self, market_data: 'MarketDataSnapshot') -> Dict[str, float]:
        """Возвращает денежные потоки (cashflows) на основе данных"""
        pass
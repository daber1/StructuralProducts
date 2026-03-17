from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .asset import Asset


class StructuralProduct(ABC):
    def __init__(self, isin: str, name: str, nominal: float, currency: str):
        self.isin = isin
        self.name = name
        self.nominal = nominal
        self.currency = currency
        self.underlyings: List[Asset] = []
        self.initial_prices: Dict[str, float] = {}

    def set_initial_prices(self, prices: Dict[str, float]) -> None:
        self.initial_prices = dict(prices)

    @abstractmethod
    def evaluate(self, market_data: Any) -> Dict[str, Any]:
        pass


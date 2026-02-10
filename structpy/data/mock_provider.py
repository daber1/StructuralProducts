from .base import MarketDataProvider
from typing import Dict, List, Optional
from datetime import date

class MockDataProvider(MarketDataProvider):
    def __init__(self, prices: Dict[str, float]):
        self._prices = prices

    def get_current_prices(self, tickers: List[str], date_obs: Optional[date] = None) -> Dict[str, float]:
        return {t: self._prices.get(t, 100.0) for t in tickers}
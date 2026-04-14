from datetime import date
from typing import Dict, List

import pandas as pd

from .base import MarketDataProvider

class MockDataProvider(MarketDataProvider):
    def __init__(self, prices: Dict[str, float]):
        self._prices = dict(prices)

    def get_history(
        self, tickers: List[str], start_date: date, end_date: date
    ) -> Dict[str, pd.DataFrame]:
        index = pd.to_datetime([start_date, end_date])
        history = {}
        for ticker in tickers:
            price = float(self._prices.get(ticker, 100.0))
            history[ticker] = pd.DataFrame({"CLOSE": [price, price]}, index=index)
        return history

    def get_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        return {t: self._prices.get(t, 100.0) for t in tickers}

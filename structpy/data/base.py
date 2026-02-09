from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import date

class MarketDataProvider(ABC):
    @abstractmethod
    def get_prices(self, tickers: List[str], date_obs: Optional[date] = None) -> Dict[str, float]:
        pass
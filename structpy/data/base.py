from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import date
import pandas as pd


class MarketDataProvider(ABC):
    @abstractmethod
    def get_history(self, 
                    tickers: List[str], 
                    start_date: date, 
                    end_date: date) -> Dict[str, pd.DataFrame]:
        """
        Возвращает словарь {ticker: DataFrame}, где DataFrame содержит
        колонки ['CLOSE'] (и другие при необходимости) с индексом-датой.
        """
        pass
    
    @abstractmethod
    def get_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        Возвращает словарь {ticker: last_price}
        """
        pass
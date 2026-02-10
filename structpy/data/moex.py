import requests
import pandas as pd
import time
from typing import List, Dict
from datetime import date, datetime
from .base import MarketDataProvider

class MoexDataProvider(MarketDataProvider):
    BASE_URL = "https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities"

    def _fetch_single_ticker(self, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Внутренний метод для выгрузки одного тикера с пагинацией.
        """
        all_data = []
        start = 0
        fmt_start = start_date.strftime('%Y-%m-%d')
        fmt_end = end_date.strftime('%Y-%m-%d')
        
        url = f"{self.BASE_URL}/{ticker}.json"
        
        while True:
            params = {
                'from': fmt_start,
                'till': fmt_end,
                'interval': '24', # дневные
                'start': start,
                'iss.meta': 'off'
            }
            
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if 'history' not in data or 'data' not in data['history']:
                    break
                    
                chunk = data['history']['data']
                columns = data['history']['columns']
                
                if not chunk:
                    break
                    
                all_data.extend(chunk)
                
                start += 100
                
                time.sleep(0.1)
                
            except requests.RequestException as e:
                print(f"Ошибка {ticker}: {e}")
                break

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data, columns=columns)
        
        if 'TRADEDATE' in df.columns and 'CLOSE' in df.columns:
            df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
            df['CLOSE'] = pd.to_numeric(df['CLOSE'], errors='coerce')
            
            df.set_index('TRADEDATE', inplace=True)
            df.sort_index(inplace=True)
            
            df = df[['CLOSE']].dropna()
        else:
            return pd.DataFrame()
            
        return df

    def get_history(self, tickers: List[str], start_date: date, end_date: date) -> Dict[str, pd.DataFrame]:
        """
        Возвращает словарь {ticker: DataFrame}, где DataFrame содержит
        колонки ['CLOSE'] с индексом-датой.
        """
        result = {}
        print(f"Загружаем данные для {len(tickers)} тикеров...")
        
        for ticker in tickers:
            print(f" -> {ticker}...", end=" ")
            df = self._fetch_single_ticker(ticker, start_date, end_date)
            if not df.empty:
                result[ticker] = df
                print(f"Ок ({len(df)} строк)")
            else:
                print("Данные не найдены!")
                
        return result

    def get_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        Возвращает словарь {ticker: last_price}
        """
        end = date.today()
        start = date.fromtimestamp(time.time() - 7 * 24 * 3600)
        
        history = self.get_history(tickers, start, end)
        prices = {}
        for t, df in history.items():
            if not df.empty:
                prices[t] = df['CLOSE'].iloc[-1]
        return prices

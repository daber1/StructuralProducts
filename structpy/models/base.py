import numpy as np
from abc import ABC, abstractmethod

class MarketModel(ABC):
    @abstractmethod
    def simulate_paths(self, 
                       S0: float,          # Начальная цена
                       T_years: float,     # Срок симуляции в годах
                       n_steps: int,       # Количество шагов
                       n_sims: int) -> np.ndarray: # Количество симуляций
        """
        Возвращает матрицу цен shape=(n_sims, n_steps + 1)
        """
        pass
    
    @abstractmethod
    def calibrate(self, historical_returns: np.ndarray):
        """
        Калибровка параметров модели на исторических данных
        """
        pass

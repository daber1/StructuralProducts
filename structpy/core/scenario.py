from typing import Dict, List

import numpy as np


class ScenarioGenerator:
    def __init__(
        self,
        tickers: List[str],
        initial_prices: Dict[str, float],
        volatilities: Dict[str, float],
        risk_free_rate: float,
        correlation_matrix: np.ndarray | None = None,
        dividend_yields: Dict[str, float] | None = None,
        jump_params: Dict[str, Dict[str, float]] | None = None,
    ):
        self.tickers = tickers
        self.initial_prices = initial_prices
        self.volatilities = volatilities
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(tickers)

        self.correlation_matrix = (
            np.eye(self.n_assets)
            if correlation_matrix is None
            else np.asarray(correlation_matrix, dtype=float)
        )
        if self.correlation_matrix.shape != (self.n_assets, self.n_assets):
            raise ValueError(
                "Переданная матрица корреляций не соотвествует количеству активов "
                f"({self.n_assets}x{self.n_assets})"
            )

        self.dividend_yields = dividend_yields or {ticker: 0.0 for ticker in tickers}
        self.jump_params = jump_params

    def generate_paths(
        self,
        n_scenarios: int,
        n_steps: int,
        T: float,
        seed: int | None = None,
    ) -> Dict[str, np.ndarray]:
        """Метод для генерации путей.
        
        args:
            n_scenarios - количество сценариев
            n_step - количество шагов в сценарии
            T - временной период, в годах
            seed - сид для воспроизводимости
        returns:
            Словарь, содержащий пути для каждого тикера.
        """
        if n_scenarios <= 0:
            raise ValueError("n_scenarios > 0")
        if n_steps <= 0:
            raise ValueError("n_steps > 0 ")
        if T <= 0:
            raise ValueError("T > 0")

        if seed is not None:
            np.random.seed(seed)

        dt = T / n_steps
        chol = np.linalg.cholesky(self.correlation_matrix)

        # Генерация независимых случайных шоков
        z = np.random.standard_normal((n_scenarios, n_steps, self.n_assets))
        # Преобразование этих шоков в коррелированные
        w = np.einsum("sna,ab->snb", z, chol.T)

        paths = {
            ticker: np.zeros((n_scenarios, n_steps + 1), dtype=float)
            for ticker in self.tickers
        }

        for ticker in self.tickers:
            if ticker not in self.initial_prices:
                raise ValueError(f"Отсутствует начальная цена для '{ticker}'")
            if ticker not in self.volatilities:
                raise ValueError(f"Отсутсвует значение волатильности для '{ticker}'")
            paths[ticker][:, 0] = self.initial_prices[ticker]

        for step in range(1, n_steps + 1):
            for i, ticker in enumerate(self.tickers):
                s_prev = paths[ticker][:, step - 1] # предущыщие цены
                vol = self.volatilities[ticker]
                div = self.dividend_yields.get(ticker, 0.0)

                drift = (self.risk_free_rate - div - 0.5 * vol**2) * dt # риск нейтральный GBM
                jump_effect = np.ones(n_scenarios)

                if self.jump_params and ticker in self.jump_params:
                    jump_cfg = self.jump_params[ticker]
                    lam = float(jump_cfg.get("lambda", 0.0)) # количество джампов в году
                    mu_j = float(jump_cfg.get("mu", 0.0)) # средний лог джампа
                    sig_j = float(jump_cfg.get("sigma", 0.0)) # волатильность джампа

                    compensator = lam * (np.exp(mu_j + 0.5 * sig_j**2) - 1) # коррекция для дрифта
                    drift -= compensator * dt

                    n_jumps = np.random.poisson(lam * dt, n_scenarios)
                    for idx in range(n_scenarios):
                        if n_jumps[idx] > 0:
                            jumps = np.random.normal(mu_j, sig_j, n_jumps[idx])
                            jump_effect[idx] = np.exp(np.sum(jumps))

                diffusion = vol * np.sqrt(dt) * w[:, step - 1, i]
                paths[ticker][:, step] = s_prev * np.exp(drift + diffusion) * jump_effect

        return paths

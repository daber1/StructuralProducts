import numpy as np
from .base import MarketModel


class MertonJumpDiffusion(MarketModel):
    def __init__(self, r: float = 0.0, div_yield: float = 0.0):
        """
        :param r: Безрисковая ставка (risk-free rate), непрерывная
        :param div_yield: Дивидендная доходность (q), непрерывная
        """
        self.r = r
        self.q = div_yield

        # Параметры модели (будут заполнены после calibrate)
        self.sigma = 0.2    # Волатильность диффузии
        self.lam = 0.0      # Интенсивность прыжков (lambda)
        self.mu_j = 0.0     # Средний размер прыжка (log)
        self.sigma_j = 0.0  # Волатильность прыжка (log)

    def calibrate(self, historical_returns: np.ndarray, trading_days=252, jump_threshold=3.0):
        """
        Калибровка параметров по историческим доходностям (log returns).
        Использует логику из ноутбука: 
        1. Считаем обычную волатильность.
        2. Детектируем прыжки (выбросы > 3 сигм).
        3. Оцениваем параметры прыжков отдельно.
        """
        # 1. Оценка волатильности
        daily_vol = np.std(historical_returns)
        self.sigma = daily_vol * np.sqrt(trading_days)

        # 2. Детекция прыжков
        mean_ret = np.mean(historical_returns)
        jump_mask = np.abs(historical_returns -
                           mean_ret) > jump_threshold * daily_vol

        jumps = historical_returns[jump_mask]

        # 3. Параметры прыжков
        n_years = len(historical_returns) / trading_days
        n_jumps = len(jumps)

        if n_jumps > 0:
            self.lam = n_jumps / n_years
            self.mu_j = np.mean(jumps)
            self.sigma_j = np.std(jumps)
        else:
            self.lam = 0.0
            self.mu_j = 0.0
            self.sigma_j = 0.0

    def simulate_paths(self, S0: float, T_years: float, n_steps: int, n_sims: int) -> np.ndarray:
        """
        Генерация путей методом Эйлера-Маруямы для процесса Мертона.
        """
        dt = T_years / n_steps

        # Матрица результатов: (N_sims, Steps + 1)
        paths = np.zeros((n_sims, n_steps + 1))
        paths[:, 0] = S0

        Z = np.random.normal(0, 1, size=(n_sims, n_steps))

        if self.lam > 0:
            N_jumps = np.random.poisson(self.lam * dt, size=(n_sims, n_steps))
        else:
            N_jumps = np.zeros((n_sims, n_steps), dtype=int)

        k = np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1
        drift = (self.r - self.q - self.lam * k) * dt

        diffusion = self.sigma * np.sqrt(dt) * Z

        log_S = np.full(n_sims, np.log(S0))

        for t in range(n_steps):

            jump_part = np.zeros(n_sims)

            has_jump = N_jumps[:, t] > 0
            count_jumps = np.sum(has_jump)

            if count_jumps > 0:
                M = np.random.normal(self.mu_j, self.sigma_j, size=count_jumps)
                jump_part[has_jump] = M

            d_log_S = drift + diffusion[:, t] + jump_part

            log_S += d_log_S
            paths[:, t+1] = np.exp(log_S)

        return paths

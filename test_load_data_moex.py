from datetime import date, timedelta
import matplotlib.pyplot as plt
import numpy as np

from structpy.data.moex import MoexDataProvider
from structpy.models.merton import MertonJumpDiffusion

# 1. Загрузка данных
provider = MoexDataProvider()
tickers = ['SBER', 'GAZP']
end_date = date.today()
start_date = end_date - timedelta(days=365*2) # 2 года истории

data = provider.get_history(tickers, start_date, end_date)

# 2. Подготовка данных для Сбера
sber_df = data['SBER']
# Считаем лог-доходности: ln(Pt / Pt-1)
sber_returns = np.log(sber_df['CLOSE'] / sber_df['CLOSE'].shift(1)).dropna().values

# 3. Калибровка модели
# Ставка 18% (ключевая ЦБ), Дивиденды ~10% (грубо)
model = MertonJumpDiffusion(r=0.18, div_yield=0.10)
model.calibrate(sber_returns)

print("\n--- SBER Model Parameters ---")
print(f"Sigma (diffusion): {model.sigma:.2%}")
print(f"Lambda (jumps/year): {model.lam:.2f}")
print(f"Mean Jump Size: {model.mu_j:.2%}")
print(f"Jump Volatility: {model.sigma_j:.2%}")

# 4. Симуляция будущего (на 1 год)
S0 = sber_df['CLOSE'].iloc[-1]
paths = model.simulate_paths(S0=S0, T_years=1.0, n_steps=252, n_sims=50)

# 5. График
plt.figure(figsize=(12, 6))
# Историческая часть
plt.plot(sber_df.index, sber_df['CLOSE'], label='History', color='black', linewidth=2)

# Будущая часть (симуляции)
# Генерируем даты для оси X
future_dates = [sber_df.index[-1] + timedelta(days=i) for i in range(1, 253)]
for i in range(10): # Рисуем первые 10 путей
    plt.plot(future_dates, paths[i, 1:], alpha=0.5) # paths[:, 0] это S0

plt.title(f"SBER: History + Merton Simulation (Lambda={model.lam:.2f})")
plt.grid(True)
plt.legend()
plt.show()

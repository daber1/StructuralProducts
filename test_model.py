import numpy as np
import matplotlib.pyplot as plt
from structpy.models.merton import MertonJumpDiffusion

# 1. Создаем фейковые исторические данные (с парой обвалов)
np.random.seed(42)
returns = np.random.normal(0.001, 0.02, 1000) # Обычные дни
# Добавляем "черных лебедей"
returns[100] = -0.15 # Обвал на 15%
returns[500] = -0.10 # Обвал на 10%
returns[800] = 0.08  # Ракета на 8%

# 2. Инициализация и калибровка модели
model = MertonJumpDiffusion(r=0.18, div_yield=0.05) # r=18% (как в РФ), q=5%
model.calibrate(returns)

print(f"Calibrated Params:")
print(f"Sigma (diffusion): {model.sigma:.2%}")
print(f"Lambda (jumps/year): {model.lam:.2f}")
print(f"Mean Jump Size: {model.mu_j:.2%}")

# 3. Симуляция на 1 год вперед
S0 = 100.0
T = 1.0
n_steps = 252
n_sims = 100

paths = model.simulate_paths(S0, T, n_steps, n_sims)

# 4. Визуализация
plt.figure(figsize=(10, 6))
plt.plot(paths.T, alpha=0.4, color='blue')
plt.title("Merton Jump Diffusion Simulations")
plt.xlabel("Days")
plt.ylabel("Price")
plt.grid(True)
plt.show()

import numpy as np

from structpy.models.merton import MertonJumpDiffusion


def test_merton_calibration_detects_jump_component():
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.02, 1000)
    returns[100] = -0.15
    returns[500] = -0.10
    returns[800] = 0.08

    model = MertonJumpDiffusion(r=0.18, div_yield=0.05)
    model.calibrate(returns)

    assert model.sigma > 0
    assert model.lam > 0
    assert model.sigma_j >= 0


def test_merton_simulate_paths_returns_positive_matrix():
    model = MertonJumpDiffusion(r=0.18, div_yield=0.05)
    model.sigma = 0.25
    model.lam = 1.0
    model.mu_j = -0.05
    model.sigma_j = 0.10

    paths = model.simulate_paths(S0=100.0, T_years=1.0, n_steps=12, n_sims=20)

    assert paths.shape == (20, 13)
    assert np.all(paths > 0)

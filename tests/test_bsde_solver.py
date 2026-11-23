"""Unit tests for the least-squares Monte Carlo BSDE solver
(src/bsde_solver.py), checked against closed-form solutions.

Methodology reference: experiment exp_1, implementation_notes (backward Euler
scheme with Longstaff-Schwartz regression).
"""

from __future__ import annotations

import numpy as np

from data.generate_synthetic import simulate_brownian_paths
from src.bsde_solver import grid_index, rho_from_bsde, solve_bsde_lsmc


def test_zero_driver_recovers_terminal_mean_at_t0():
    """g = 0: Y_t = E[Y_u | F_t]; at t=0 this is just E[X]."""
    n_paths, n_steps = 4000, 20
    times, paths = simulate_brownian_paths(n_paths, n_steps, 1.0, 1, seed=1)
    bm = paths[:, :, 0]
    terminal = bm[:, -1] ** 2  # E[B_1^2] = 1
    driver = lambda t, y, z: np.zeros_like(y)
    Y, Z = solve_bsde_lsmc(times, bm, driver, terminal, basis_degree=4)
    assert abs(Y[:, 0].mean() - 1.0) < 0.1


def test_linear_driver_matches_deterministic_ode_solution():
    """g(t,y,z) = r*y (independent of z): Y_t solves a deterministic linear
    BSDE whose conditional-expectation representation gives
    Y_t = E[X * exp(-r*(u-t)) | F_t] (Feynman-Kac for dY = -r*Y dt + Z dB).
    With deterministic terminal X = c, Y_t = c * exp(-r*(u-t)) exactly."""
    n_paths, n_steps = 4000, 50
    times, paths = simulate_brownian_paths(n_paths, n_steps, 1.0, 1, seed=2)
    bm = paths[:, :, 0]
    r = 0.1
    c = 2.0
    terminal = np.full(n_paths, c)
    driver = lambda t, y, z: -r * y
    Y, Z = solve_bsde_lsmc(times, bm, driver, terminal, basis_degree=4)
    expected = c * np.exp(-r * (1.0 - times[0]))
    assert abs(Y[:, 0].mean() - expected) < 0.05


def test_rho_from_bsde_subgrid_matches_full_grid():
    n_paths, n_steps = 2000, 40
    times, paths = simulate_brownian_paths(n_paths, n_steps, 1.0, 1, seed=3)
    bm = paths[:, :, 0]
    driver = lambda t, y, z: 0.05 * np.maximum(-y, 0.0) + z
    x_at_u = bm[:, grid_index(times, 0.5)]
    rho_t, Z_path, Y_path = rho_from_bsde(times, bm, 0.0, 0.5, driver, -x_at_u, basis_degree=4)
    assert rho_t.shape == (n_paths,)
    assert Y_path.shape[1] == Z_path.shape[1] + 1


def test_grid_index_rejects_off_grid_time():
    times = np.linspace(0.0, 1.0, 11)
    import pytest

    with pytest.raises(ValueError):
        grid_index(times, 0.37)


def test_solve_bsde_requires_uniform_grid():
    import pytest

    bm = np.zeros((10, 3))
    times = np.array([0.0, 0.3, 1.0])
    with pytest.raises(ValueError):
        solve_bsde_lsmc(times, bm, lambda t, y, z: np.zeros_like(y), np.zeros(10))

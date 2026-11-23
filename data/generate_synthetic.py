"""Synthetic data generation for the BSDE / generalized-shortfall experiments.

The paper works entirely with a probability space supporting a d-dimensional
Brownian motion and with terminal conditions that are measurable functions of
that Brownian motion (Section 2). No market data is used anywhere in this
replication: every experiment consumes paths and terminal conditions produced
here.
"""

from __future__ import annotations

import numpy as np


def simulate_brownian_paths(
    n_paths: int,
    n_steps: int,
    horizon: float,
    d: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate ``n_paths`` trajectories of a ``d``-dimensional Brownian motion.

    Parameters
    ----------
    n_paths: number of Monte Carlo paths.
    n_steps: number of equally spaced time steps on ``[0, horizon]``.
    horizon: terminal time ``T`` of the simulation grid.
    d: dimension of the Brownian motion (the paper does not prescribe a
        value for its numerical examples; set to ``d = 1`` throughout, see
        ``src/config.py``).
    seed: seed of the random generator, for reproducibility.

    Returns
    -------
    times: array of shape ``(n_steps + 1,)`` with the grid ``0 = t_0 < ... < t_n = T``.
    paths: array of shape ``(n_paths, n_steps + 1, d)`` with ``paths[:, 0, :] == 0``.
    """
    if n_paths <= 0 or n_steps <= 0 or horizon <= 0:
        raise ValueError("n_paths, n_steps and horizon must be strictly positive")
    rng = np.random.default_rng(seed)
    dt = horizon / n_steps
    times = np.linspace(0.0, horizon, n_steps + 1)
    increments = rng.normal(loc=0.0, scale=np.sqrt(dt), size=(n_paths, n_steps, d))
    paths = np.concatenate(
        [np.zeros((n_paths, 1, d)), np.cumsum(increments, axis=1)], axis=1
    )
    return times, paths


def terminal_constant(n_paths: int, value: float) -> np.ndarray:
    """F_u-measurable constant terminal condition X = value."""
    return np.full(n_paths, float(value))


def terminal_normal(n_paths: int, mean: float, std: float, seed: int | None = None) -> np.ndarray:
    """F_u-measurable terminal condition X ~ N(mean, std^2), independent of the
    simulated Brownian motion (used as a generic random position)."""
    if std < 0:
        raise ValueError("std must be non-negative")
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=std, size=n_paths)


def terminal_from_bm(paths: np.ndarray, times: np.ndarray, u: float, func=None) -> np.ndarray:
    """F_u-measurable terminal condition X = func(B_u), defaulting to X = B_u^{(1)}.

    ``paths`` has shape ``(n_paths, n_steps + 1, d)`` as returned by
    :func:`simulate_brownian_paths`; ``u`` must be one of the grid points in
    ``times`` (up to floating point tolerance).
    """
    idx = _grid_index(times, u)
    b_u = paths[:, idx, 0]
    if func is None:
        return b_u
    return func(b_u)


def terminal_put_payoff(paths: np.ndarray, times: np.ndarray, u: float, strike: float, spot: float = 1.0) -> np.ndarray:
    """F_u-measurable put-option payoff on S_u = spot * exp(B_u - u/2) (a toy
    exponential-Brownian underlying), X = (strike - S_u)^+."""
    idx = _grid_index(times, u)
    b_u = paths[:, idx, 0]
    s_u = spot * np.exp(b_u - 0.5 * u)
    return np.maximum(strike - s_u, 0.0)


def _grid_index(times: np.ndarray, t: float, atol: float = 1e-9) -> int:
    idx = int(np.argmin(np.abs(times - t)))
    if abs(times[idx] - t) > atol:
        raise ValueError(f"time {t} is not on the simulation grid")
    return idx

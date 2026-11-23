"""Generic least-squares Monte Carlo (Longstaff-Schwartz style) backward
solver for BSDEs with Lipschitz or quadratic drivers.

Methodology (experiment exp_1, implementation_notes): the backward Euler
scheme

    Y_{t_i} = E[Y_{t_{i+1}} + g(t_{i+1}, Y_{t_{i+1}}, Z_{t_{i+1}}) * dt | F_{t_i}]

is implemented with the conditional expectation approximated by polynomial
regression of the continuation value on the current state of the (Markov)
driving Brownian motion, in the spirit of Longstaff & Schwartz (2001).

The process (Z_t) is estimated by the finite-difference-of-regression-
function method (Bouchard & Touzi (2004); Gobet, Lemor & Warin (2005)):
since the driving state at t_{i+1} is state_i + dB_i, a first-order Taylor
expansion of the fitted continuation function f_{i+1} gives

    Y_{i+1} = f_{i+1}(state_i + dB_i) ~= f_{i+1}(state_i) + f_{i+1}'(state_i) dB_i,

so that E[Y_{i+1} dB_i / dt | F_i] ~= f_{i+1}'(state_i), i.e. Z_i is the
spatial derivative of the regression function fitted to Y_{i+1}, evaluated
at the current state. This avoids regressing the O(1/dt)-variance target
Y_{i+1} dB_i / dt directly on the state, which is unstable for the fine time
grids used here (n_steps of order 100) unless the number of paths is
enormous.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

Driver = Callable[[float, np.ndarray, np.ndarray], np.ndarray]


def _fit_coeffs(state: np.ndarray, target: np.ndarray, degree: int) -> np.ndarray:
    """Least-squares polynomial regression coefficients of ``target`` on
    ``state`` (highest power first, as returned by :func:`numpy.polyfit`).

    When ``state`` is (numerically) constant across paths -- as happens at
    the very first grid point t=0, where every simulated path starts at
    B_0 = 0 -- the Vandermonde matrix is rank-deficient and
    :func:`numpy.polyfit` raises; the conditional expectation given a
    constant (trivial) state is simply the unconditional mean, so the
    regression collapses to a constant polynomial in that case.
    """
    if np.std(state) < 1e-12:
        coeffs = np.zeros(degree + 1)
        coeffs[-1] = np.mean(target)
        return coeffs
    return np.polyfit(state, target, deg=degree)


def _fit_predict(state: np.ndarray, target: np.ndarray, degree: int) -> np.ndarray:
    """Least-squares regression of ``target`` on a polynomial basis of
    ``state``, returning the in-sample fitted values (the Longstaff-Schwartz
    approximation of the conditional expectation given the state)."""
    coeffs = _fit_coeffs(state, target, degree)
    return np.polyval(coeffs, state)


def solve_bsde_lsmc(
    times: np.ndarray,
    bm_path: np.ndarray,
    driver: Driver,
    terminal: np.ndarray,
    basis_degree: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve, by backward induction, the BSDE

        Y_s = terminal - int_s^{times[-1]} Z_r dB_r + int_s^{times[-1]} g(r, Y_r, Z_r) dr

    on the sub-grid ``times`` (assumed uniform), driven by the 1-dimensional
    Brownian motion path ``bm_path`` (shape ``(n_paths, len(times))``).

    Parameters
    ----------
    times: strictly increasing grid ``[t_0, ..., t_n]`` (uniform spacing).
    bm_path: simulated Brownian motion restricted to this grid, one column
        per grid point, ``bm_path[:, 0]`` being the state at ``t_0``.
    driver: vectorized callable ``g(t, y, z) -> array`` with ``t`` the
        (scalar) time at which the driver is evaluated.
    terminal: terminal condition ``Y_{t_n}`` (shape ``(n_paths,)``).
    basis_degree: degree of the polynomial regression basis.

    Returns
    -------
    Y: array of shape ``(n_paths, n_steps + 1)`` with the fitted solution.
    Z: array of shape ``(n_paths, n_steps)`` with the fitted control process,
       ``Z[:, i]`` being the estimate of ``Z_{t_i}``.
    """
    n_steps = bm_path.shape[1] - 1
    if n_steps < 1:
        raise ValueError("at least one time step is required")
    dt = times[1] - times[0]
    if not np.allclose(np.diff(times), dt):
        raise ValueError("the time grid must be uniform")

    n_paths = bm_path.shape[0]
    Y = np.empty((n_paths, n_steps + 1))
    Z = np.empty((n_paths, n_steps))
    Y[:, -1] = terminal

    for i in range(n_steps - 1, -1, -1):
        t_next = times[i + 1]
        state = bm_path[:, i]
        state_next = bm_path[:, i + 1]

        # Z_i: derivative, at state_i, of the regression function fitted to
        # Y_{i+1} on state_{i+1} (finite-difference-of-regression-function
        # estimator, see module docstring).
        y_coeffs = _fit_coeffs(state_next, Y[:, i + 1], basis_degree)
        z_coeffs = np.polyder(y_coeffs)
        Z[:, i] = np.polyval(z_coeffs, state)

        g_val = driver(t_next, Y[:, i + 1], Z[:, i])
        continuation = Y[:, i + 1] + g_val * dt
        Y[:, i] = _fit_predict(state, continuation, basis_degree)

    return Y, Z


def grid_index(times: np.ndarray, t: float, atol: float = 1e-9) -> int:
    """Index of time ``t`` on ``times`` (within tolerance)."""
    idx = int(np.argmin(np.abs(times - t)))
    if abs(times[idx] - t) > atol:
        raise ValueError(f"time {t} is not on the simulation grid")
    return idx


def rho_from_bsde(
    times: np.ndarray,
    bm_path: np.ndarray,
    t: float,
    u: float,
    driver: Driver,
    minus_x_at_u: np.ndarray,
    basis_degree: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fully-dynamic risk measure rho_{t,u}(X) = Y_t solving the Lipschitz (or
    quadratic) BSDE on ``[t, u]`` with terminal condition ``Y_u = -X``.

    Returns
    -------
    rho_t: array of shape ``(n_paths,)``, the F_t-measurable solution Y_t
        (one value per simulated path, since Y_t is a regression function of
        the state B_t evaluated along each path).
    Z_path: array of shape ``(n_paths, n_steps)`` with the fitted Z process.
    Y_path: array of shape ``(n_paths, n_steps + 1)`` with the full fitted Y.
    """
    it = grid_index(times, t)
    iu = grid_index(times, u)
    if it >= iu:
        raise ValueError("t must be strictly less than u")
    sub_times = times[it : iu + 1]
    sub_bm = bm_path[:, it : iu + 1]
    Y, Z = solve_bsde_lsmc(sub_times, sub_bm, driver, minus_x_at_u, basis_degree)
    return Y[:, 0], Z, Y

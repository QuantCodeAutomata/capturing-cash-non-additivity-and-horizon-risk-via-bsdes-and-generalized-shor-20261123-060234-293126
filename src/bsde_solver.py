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


def solve_bsde_affine_in_z(
    times: np.ndarray,
    bm_path: np.ndarray,
    h: Callable[[float, np.ndarray], np.ndarray],
    c_func: Callable[[float], float],
    terminal: np.ndarray,
    basis_degree: int = 5,
) -> np.ndarray:
    """Solve, by backward induction, the BSDE with driver affine in z,

        g(t, y, z) = h(t, y) + c(t) * z,

    e.g. g(t,y,z) = r_t*y^- + z (Example 6, c(t) = 1, h(t,y) = r_t*y^-).

    Rather than plugging a finite-difference estimate of Z back into the
    driver (which is numerically unstable here because Y has a kink at
    y = 0 from the y^- term, and the resulting bias violates h-longevity
    in simulation), the linear-in-z part is removed by a Girsanov change of
    measure, exactly as in Theorem 5's proof: under Q with density
    dQ/dP|_{F_s} = exp(c(s) dB_s - 1/2 c(s)^2 ds), B_s - int c(r) dr is a
    Q-Brownian motion, so

        Y_s = E^Q[ Y_{s+ds} + h(s+ds, Y_{s+ds}) ds | F_s ]
            = E^P[ weight_s * (Y_{s+ds} + h(s+ds, Y_{s+ds}) ds) | F_s ],

    with weight_s = exp(c(s+ds)(B_{s+ds}-B_s) - 1/2 c(s+ds)^2 ds) the
    incremental Girsanov density, computed on the original simulated paths
    and folded into the regression target -- no Z estimate is required.

    Returns
    -------
    Y: array of shape ``(n_paths, n_steps + 1)``.
    """
    n_steps = bm_path.shape[1] - 1
    if n_steps < 1:
        raise ValueError("at least one time step is required")
    dt = times[1] - times[0]
    if not np.allclose(np.diff(times), dt):
        raise ValueError("the time grid must be uniform")

    n_paths = bm_path.shape[0]
    Y = np.empty((n_paths, n_steps + 1))
    Y[:, -1] = terminal

    for i in range(n_steps - 1, -1, -1):
        t_next = times[i + 1]
        state = bm_path[:, i]
        d_b = bm_path[:, i + 1] - bm_path[:, i]
        c_val = c_func(t_next)
        weight = np.exp(c_val * d_b - 0.5 * c_val**2 * dt)
        continuation = weight * (Y[:, i + 1] + h(t_next, Y[:, i + 1]) * dt)
        Y[:, i] = _fit_predict(state, continuation, basis_degree)

    return Y


def rho_from_bsde_affine_in_z(
    times: np.ndarray,
    bm_path: np.ndarray,
    t: float,
    u: float,
    h: Callable[[float, np.ndarray], np.ndarray],
    c_func: Callable[[float], float],
    minus_x_at_u: np.ndarray,
    basis_degree: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fully-dynamic risk measure rho_{t,u}(X) = Y_t for a driver affine in z,
    g(t,y,z) = h(t,y) + c(t)*z, solved via :func:`solve_bsde_affine_in_z`.

    Returns
    -------
    rho_t: array of shape ``(n_paths,)``.
    Y_path: array of shape ``(n_paths, n_steps + 1)`` with the full solution.
    """
    it = grid_index(times, t)
    iu = grid_index(times, u)
    if it >= iu:
        raise ValueError("t must be strictly less than u")
    sub_times = times[it : iu + 1]
    sub_bm = bm_path[:, it : iu + 1]
    Y = solve_bsde_affine_in_z(sub_times, sub_bm, h, c_func, minus_x_at_u, basis_degree)
    return Y[:, 0], Y


def solve_bsde_affine_in_z_extended(
    times: np.ndarray,
    bm_path: np.ndarray,
    t: float,
    u: float,
    v: float,
    h: Callable[[float, np.ndarray], np.ndarray],
    c_func: Callable[[float], float],
    minus_x_at_u: np.ndarray,
    basis_degree: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve rho_{t,v}(X) = Y_t for X that is F_u-measurable with t <= u <= v,
    driver g(s,y,z) = h(s,y) + c(s) z (Example 6).

    Because the terminal condition Y_v = -X is already F_u-measurable, the
    BSDE on (u, v] carries no new randomness to hedge: by uniqueness of the
    Lipschitz BSDE, the solution there has Z_s = 0 and Y solves the backward
    ODE dY_s/ds = -h(s, Y_s), Y_v = -X (Theorem 5's extension (17): Z_s = 0
    and Y_s = bar_Y_s for s in (u, v]). Representing this segment through a
    single-state regression on B_s (as in :func:`solve_bsde_affine_in_z`)
    is invalid here, because -X depends on B_u, which is not a deterministic
    function of B_s for s > u (a path's B_s does not determine its B_u):
    regressing on B_s alone shrinks -X towards its conditional mean given
    B_s and biases Y_t away from the exact restriction/h-longevity values.
    The ODE step below instead propagates each path's own -X value exactly,
    with no cross-path regression, before handing the result to the genuine
    (stochastic) regression-based solve on [t, u].

    Returns
    -------
    rho_t: array of shape ``(n_paths,)``.
    Y_uv: array of shape ``(n_paths, n_steps_uv + 1)``, the (deterministic
        per path) solution on the ``[u, v]`` leg, ``Y_uv[:, 0]`` being the
        induced terminal condition ``Y_u`` used for the ``[t, u]`` leg.
    Y_tv_full: array of shape ``(n_paths, n_steps_tv + 1)``, the full
        solution on ``[t, v]`` (the genuine ``Y_s^v`` process, not to be
        confused with the comparison process ``bar_Y`` of Eq. (17), which
        freezes the terminal condition at ``u`` instead of ODE-propagating
        it to ``v``).
    """
    it, iu, iv = grid_index(times, t), grid_index(times, u), grid_index(times, v)
    if not (it <= iu <= iv):
        raise ValueError("require t <= u <= v")
    dt = times[1] - times[0]

    if iu < iv:
        n_paths = bm_path.shape[0]
        Y_uv = np.empty((n_paths, iv - iu + 1))
        Y_uv[:, -1] = minus_x_at_u
        for k in range(iv - iu - 1, -1, -1):
            s_next = times[iu + k + 1]
            Y_uv[:, k] = Y_uv[:, k + 1] + h(s_next, Y_uv[:, k + 1]) * dt
        y_at_u = Y_uv[:, 0]
    else:
        Y_uv = minus_x_at_u.reshape(-1, 1)
        y_at_u = minus_x_at_u

    if it < iu:
        sub_times = times[it : iu + 1]
        sub_bm = bm_path[:, it : iu + 1]
        Y_tu = solve_bsde_affine_in_z(sub_times, sub_bm, h, c_func, y_at_u, basis_degree)
        rho_t = Y_tu[:, 0]
        Y_tv_full = np.concatenate([Y_tu[:, :-1], Y_uv], axis=1)
    else:
        rho_t = y_at_u
        Y_tv_full = Y_uv

    return rho_t, Y_uv, Y_tv_full


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

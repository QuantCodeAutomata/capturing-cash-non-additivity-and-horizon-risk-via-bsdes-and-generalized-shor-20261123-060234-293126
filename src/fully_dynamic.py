"""Fully-dynamic risk measures: h-longevity, cash (sub)additivity, the
explicit h-longevity index of Theorem 5, the hq-entropic risk measure on
losses (Eq. (6)/(26)), fully-dynamic certainty equivalents (Definition 28)
and the h-Value at Risk (Example 15).

All conditional expectations at t = 0 reduce to plain Monte Carlo averages;
for t > 0 they are approximated by polynomial regression of the relevant
payoff on the Brownian state B_t, exactly as in :mod:`src.bsde_solver`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import norm

from src.bsde_solver import _fit_predict, grid_index
from src.qdeformed import exp_q, ln_q


# --------------------------------------------------------------------------
# Conditional expectation helper shared by the hq-entropic and certainty
# equivalent computations (Section 3 and Section 5).
# --------------------------------------------------------------------------
def conditional_expectation(payoff: np.ndarray, times: np.ndarray, bm_path: np.ndarray, t: float, basis_degree: int = 5) -> np.ndarray:
    """E[payoff | F_t]: the sample mean when t = 0, a polynomial regression
    of ``payoff`` on B_t otherwise. Returns an array of shape (n_paths,)."""
    if np.isclose(t, 0.0):
        return np.full(payoff.shape[0], np.mean(payoff))
    it = grid_index(times, t)
    state = bm_path[:, it]
    return _fit_predict(state, payoff, basis_degree)


def integral_a(a_func: Callable[[float], float], lo: float, hi: float, n_quad: int = 1000) -> float:
    """A(lo, hi) = int_lo^hi a(s) ds, by Simpson quadrature (a is a smooth
    deterministic non-negative function, Section 3)."""
    if hi < lo:
        raise ValueError("hi must be >= lo")
    if np.isclose(hi, lo):
        return 0.0
    from scipy.integrate import simpson

    grid = np.linspace(lo, hi, n_quad + 1)
    values = np.array([a_func(s) for s in grid])
    return float(simpson(values, x=grid))


# --------------------------------------------------------------------------
# hq-entropic risk measure on losses, Eq. (6)/(26)
# --------------------------------------------------------------------------
def hq_entropic_rho(
    x_u: np.ndarray,
    times: np.ndarray,
    bm_path: np.ndarray,
    t: float,
    u: float,
    q: float,
    beta: float,
    alpha_q: float,
    a_func: Callable[[float], float],
    basis_degree: int = 5,
) -> np.ndarray:
    """rho_tu^{hq}(X) = ln_q E[exp_q((X+beta)^- + alpha_q + A(0,u)) | F_t] - A(0,t).

    ``x_u`` is the (F_u-measurable) terminal condition sampled along every
    path. Returns an array of shape (n_paths,), one value per path (constant
    across paths when t = 0).
    """
    a0u = integral_a(a_func, 0.0, u)
    a0t = integral_a(a_func, 0.0, t)
    loss_part = np.maximum(-(x_u + beta), 0.0) + alpha_q + a0u
    transformed = exp_q(loss_part, q)
    cond_exp = conditional_expectation(transformed, times, bm_path, t, basis_degree)
    return ln_q(cond_exp, q) - a0t


# --------------------------------------------------------------------------
# h-longevity index, Theorem 5 (Lipschitz BSDE case) and direct difference
# --------------------------------------------------------------------------
def gamma_direct(rho_tv: np.ndarray, rho_tu: np.ndarray) -> np.ndarray:
    """gamma(t, u, v, X) = rho_tv(X) - rho_tu(X), Eq. (2)."""
    return rho_tv - rho_tu


def gamma_theorem5(
    times: np.ndarray,
    bm_path: np.ndarray,
    t: float,
    u: float,
    v: float,
    delta_y_vals: np.ndarray,
    delta_z_const: float,
    g_minus_x_zero_vals: np.ndarray,
    basis_degree: int = 5,
) -> np.ndarray:
    """Explicit h-longevity index of Theorem 5,

        gamma(t,u,v,X) = E_{Qtilde_X}[ int_u^v exp(int_t^s Delta_y g(r) dr)
                           g(s,-X,0) ds | F_t ]
                        = E_P[ W_tv * int_u^v exp(int_t^s Delta_y g(r) dr)
                           g(s,-X,0) ds | F_t ],

    with Girsanov density (equation preceding Theorem 5)

        W_tv = exp{ -1/2 int_t^v |Delta_z g(s)|^2 ds + int_t^v Delta_z g(s) dB_s }.

    The running (s-dependent, not fixed at v) exponential discount factor is
    the solution of the linear BSDE with source term that underlies the
    proof of Theorem 5 (the comparison tilde_Y_s = Y_s^v - bar_Y_s^u solves
    d(tilde_Y_s) = -Delta_y g(s) tilde_Y_s ds - g(s,-X,0) 1_{(u,v]}(s) ds +
    Z_s^v dB_s^{Qtilde}, tilde_Y_v = 0, whose standard linear-BSDE-with-
    running-cost representation integrates the discount factor exp(int_t^s
    Delta_y g(r)dr) against the source term g(s,-X,0) under the integral
    sign). This was checked against a closed-form deterministic case
    (constant X, driver g=r y^- + z with r constant, so Z = 0 and Qtilde=P):
    the running-exponent form reproduces rho_tv(X) - rho_tu(X) exactly,
    while discounting by the fixed factor exp(int_t^v Delta_y g(r)dr) does
    not.

    This implementation assumes Delta_z g(s) is a known deterministic
    constant over ``[t, v]`` (true for the Example 6 driver g = r_t y^- + z,
    whose z-slope equals 1 regardless of y, z), so the stochastic integral
    reduces to ``delta_z_const * (B_v - B_t)``.

    Parameters
    ----------
    delta_y_vals: array (n_paths, n_steps_tv + 1), Delta_y g(s) evaluated at
        every grid point of ``[t, v]`` (Eq. (18), from the bar-Y extension
        (17) of the BSDE solutions on [t,u] and [t,v]).
    delta_z_const: constant value of Delta_z g(s) on [t, v].
    g_minus_x_zero_vals: array (n_paths, n_steps_tv + 1), g(s, -X, 0) at
        every grid point of ``[t, v]`` (only ``[u, v]`` is integrated).
    """
    from scipy.integrate import cumulative_trapezoid

    it = grid_index(times, t)
    iv = grid_index(times, v)
    sub_times = times[it : iv + 1]

    # Running exponent int_t^s Delta_y g(r) dr at every grid point s in [t, v].
    cum_integral = cumulative_trapezoid(delta_y_vals, sub_times, axis=1, initial=0.0)
    running_discount = np.exp(cum_integral)

    iu_local = grid_index(sub_times, u)
    integrand = running_discount[:, iu_local:] * g_minus_x_zero_vals[:, iu_local:]
    inner_integral = np.trapz(integrand, sub_times[iu_local:], axis=1)

    b_t = bm_path[:, it]
    b_v = bm_path[:, iv]
    girsanov_weight = np.exp(
        -0.5 * delta_z_const**2 * (v - t) + delta_z_const * (b_v - b_t)
    )

    payoff = girsanov_weight * inner_integral
    return conditional_expectation(payoff, times, bm_path, t, basis_degree)


# --------------------------------------------------------------------------
# Fully-dynamic certainty equivalent risk measures, Definition 28
# --------------------------------------------------------------------------
def certainty_equivalent_rho(
    x_u: np.ndarray,
    times: np.ndarray,
    bm_path: np.ndarray,
    t: float,
    util: Callable[[np.ndarray], np.ndarray],
    util_inv: Callable[[np.ndarray], np.ndarray],
    basis_degree: int = 5,
) -> np.ndarray:
    """rho_tu^{Util}(X) = -Util^{-1}(E[Util(X) | F_t]), Eq. (52)."""
    cond_exp = conditional_expectation(util(x_u), times, bm_path, t, basis_degree)
    return -util_inv(cond_exp)


# --------------------------------------------------------------------------
# h-Value at Risk, Example 15
# --------------------------------------------------------------------------
def h_var_normal(mu: float, sigma: float, t: float, u: float, alpha_u: float, b_t: np.ndarray) -> np.ndarray:
    """h-VaR for X = mu*u + sigma*B_u, conditional on F_t = sigma(B_t):

        X | F_t ~ N(mu*u + sigma*B_t, sigma^2 (u - t)),
        rho_tu(X) = -quantile_{alpha_u}(X | F_t)
                  = -(mu*u + sigma*b_t) - sigma*sqrt(u-t)*Phi^{-1}(alpha_u).

    ``b_t`` is an array of realized B_t values (one per path); returns the
    corresponding array of rho_tu(X) values.
    """
    if u < t:
        raise ValueError("u must be >= t")
    cond_mean = mu * u + sigma * b_t
    cond_std = sigma * np.sqrt(max(u - t, 0.0))
    return -cond_mean - cond_std * norm.ppf(alpha_u)


def h_var_empirical(x_samples: np.ndarray, alpha_u: float) -> float:
    """Unconditional (t=0) h-VaR via the empirical alpha_u-quantile,
    rho_{0u}(X) = -quantile_{alpha_u}(X)."""
    return float(-np.quantile(x_samples, alpha_u))

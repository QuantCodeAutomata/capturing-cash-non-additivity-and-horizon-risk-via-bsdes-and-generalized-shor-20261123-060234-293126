"""Generalized shortfall risk measures (Section 4, Definition 16) and the
classical dynamic shortfall (Eq. (27)), together with utility and aggregator
functions used throughout exp_3, exp_4, exp_5 and exp_6.

    rho^{U,f,B}(X) = inf{m in R : E[U(f(X,m))] >= B}                 (31)

is solved by root-finding on the monotone map m -> E[U(f(X,m))] - B,
assuming f is non-decreasing in m (Proposition 17, item 1a).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import brentq

Vec = np.ndarray


# --------------------------------------------------------------------------
# Utility functions U (concave, non-decreasing)
# --------------------------------------------------------------------------
def utility_identity(x: Vec) -> Vec:
    return np.asarray(x, dtype=float)


def utility_exponential(x: Vec, gamma: float = 1.0) -> Vec:
    """U(x) = 1 - exp(-x/gamma), risk-aversion gamma > 0.

    The exponent is clipped to the largest value representable in double
    precision before exponentiating: for a very negative loss x the exact
    value is U(x) -> -inf, so the clip only avoids a floating point overflow
    warning and does not change any finite result.
    """
    exponent = np.clip(-np.asarray(x, dtype=float) / gamma, a_min=None, a_max=700.0)
    return 1.0 - np.exp(exponent)


def utility_exponential_inverse(y: Vec, gamma: float = 1.0) -> Vec:
    """Inverse of :func:`utility_exponential` on y < 1."""
    y = np.asarray(y, dtype=float)
    return -gamma * np.log(np.clip(1.0 - y, a_min=1e-300, a_max=None))


def utility_crra(x: Vec, gamma: float) -> Vec:
    """CRRA utility U(x) = x^{1-gamma} / (1-gamma), defined for x > 0 (and,
    by convention, -inf for x <= 0 since the domain of CRRA utility excludes
    non-positive wealth)."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, -np.inf)
    mask = x > 0
    out[mask] = np.power(x[mask], 1.0 - gamma) / (1.0 - gamma)
    return out


def utility_crra_inverse(y: Vec, gamma: float) -> Vec:
    """Inverse of :func:`utility_crra` for y such that (1-gamma)*y > 0."""
    y = np.asarray(y, dtype=float)
    return np.power(np.clip((1.0 - gamma) * y, a_min=1e-300, a_max=None), 1.0 / (1.0 - gamma))


def utility_var(x: Vec, alpha: float) -> Vec:
    """U^alpha(x) = (alpha-1) 1_{x<0} + alpha 1_{x>=0}, Example 15, Eq. (30)."""
    x = np.asarray(x, dtype=float)
    return np.where(x < 0.0, alpha - 1.0, alpha)


# --------------------------------------------------------------------------
# Aggregator functions f(x, m)
# --------------------------------------------------------------------------
def aggregator_classical(x: Vec, m: float) -> Vec:
    """f(x,m) = x + m, the classical shortfall aggregator, Eq. (27)."""
    return np.asarray(x, dtype=float) + m


def aggregator_linear(x: Vec, m: float, beta: float) -> Vec:
    """f(x,m) = beta*x + m, beta in (0,1], Example 19 i)."""
    return beta * np.asarray(x, dtype=float) + m


def aggregator_exponential(x: Vec, m: float, nu: float) -> Vec:
    """f(x,m) = 1 - exp(-nu*x - m), nu in (0,1), Example 19 ii).

    The exponent is clipped at 700 (exp(700) already exceeds float64 range)
    before exponentiating: this only affects values driven to -inf, which the
    root finder never selects as the shortfall level, so it does not change
    any reported quantity but avoids a spurious overflow warning.
    """
    exponent = np.clip(-nu * np.asarray(x, dtype=float) - m, -np.inf, 700.0)
    return 1.0 - np.exp(exponent)


# --------------------------------------------------------------------------
# Generalized shortfall risk measure, Eq. (31)/(34)
# --------------------------------------------------------------------------
def generalized_shortfall(
    x_samples: Vec,
    utility: Callable[[Vec], Vec],
    aggregator: Callable[[Vec, float], Vec],
    target_b: float,
    bracket: tuple[float, float] = (-100.0, 100.0),
) -> float:
    """rho^{U,f,B}(X) = inf{m : E[U(f(X,m))] >= B} via bisection (Definition 16).

    ``aggregator`` must be a function of (x, m) only (curry any extra shape
    parameter, e.g. ``lambda x, m: aggregator_linear(x, m, beta)``), and
    non-decreasing in m so that ``h(m) = E[U(f(X,m))] - B`` is monotone.
    """

    def h(m: float) -> float:
        return float(np.mean(utility(aggregator(x_samples, m)))) - target_b

    lo, hi = bracket
    h_lo, h_hi = h(lo), h(hi)
    expand = 0
    while h_lo > 0.0 and expand < 20:
        lo -= (hi - lo)
        h_lo = h(lo)
        expand += 1
    while h_hi < 0.0 and expand < 40:
        hi += (hi - lo)
        h_hi = h(hi)
        expand += 1
    if h_lo > 0.0 or h_hi < 0.0:
        raise RuntimeError("could not bracket the root of E[U(f(X,m))] - B")
    return brentq(h, lo, hi, xtol=1e-6)


def acceptance_indicator(
    y_samples: Vec,
    utility: Callable[[Vec], Vec],
    aggregator: Callable[[Vec, float], Vec],
    target_b: float,
    m: float,
) -> bool:
    """Membership Y in A^{U,f,B;m}, i.e. E[U(f(Y,m))] >= B, Eq. (35)."""
    return float(np.mean(utility(aggregator(y_samples, m)))) >= target_b


# --------------------------------------------------------------------------
# Dual representation: classical shortfall + exponential utility is exactly
# an entropic risk measure, Foellmer & Schied Example 4.115, which admits
# the closed-form convex-dual representation
#     rho(X) = sup_Q ( E_Q[-X] - gamma * H(Q|P) )
# with H(Q|P) the relative entropy of Q with respect to P. This is the
# tractable instance of Proposition 20 used to numerically validate the
# dual representation (31)/(37)-(38) for the generalized shortfall.
# --------------------------------------------------------------------------
def entropic_shortfall_closed_form(x_samples: Vec, gamma: float, target_b: float) -> float:
    """Closed form of rho^{U,f,B}(X) for U(x)=1-exp(-x/gamma), f(x,m)=x+m:

        E[1 - exp(-(X+m)/gamma)] = B
        <=>  exp(-m/gamma) E[exp(-X/gamma)] = 1 - B
        <=>  m = gamma * ( ln E[exp(-X/gamma)] - ln(1-B) ).
    """
    if not (0.0 <= target_b < 1.0):
        raise ValueError("target_b must lie in [0, 1) for the exponential utility shortfall")
    log_mgf = float(np.log(np.mean(np.exp(-x_samples / gamma))))
    return gamma * (log_mgf - np.log(1.0 - target_b))


def entropic_dual_sup(
    x_samples: Vec,
    gamma: float,
    target_b: float,
    theta_grid: Vec,
) -> tuple[float, Vec]:
    """Numerical sup_Q (E_Q[-X] - gamma*H(Q|P) ) + gamma*ln(1/(1-B)) over a
    grid of exponentially tilted measures dQ_theta/dP = exp(theta X) / E[exp(theta X)].

    For these measures the relative entropy is H(Q_theta|P) = theta E_{Q_theta}[X]
    - ln E[exp(theta X)]. The grid search approximates the dual
    representation of Proposition 20 for the entropic/classical-shortfall
    instance; the supremum over theta is attained, in theory, at
    theta = -1/gamma, which is included for comparison.
    """
    values = np.empty_like(theta_grid, dtype=float)
    for i, theta in enumerate(theta_grid):
        # Center before exponentiating (log-sum-exp trick) to avoid float
        # overflow for large |theta|; the tilted weights are unaffected since
        # they are renormalized by weights.sum() below.
        exponent = theta * x_samples
        shift = float(np.max(exponent))
        shifted = np.exp(exponent - shift)  # exponent - shift <= 0: never overflows
        log_mgf_theta = shift + float(np.log(np.mean(shifted)))
        weights = shifted / shifted.sum()  # same normalized weights, overflow-free
        e_q_minus_x = -float(np.sum(weights * x_samples))
        e_q_x = float(np.sum(weights * x_samples))
        relative_entropy = theta * e_q_x - log_mgf_theta
        values[i] = e_q_minus_x - gamma * relative_entropy
    offset = gamma * np.log(1.0 / (1.0 - target_b))
    dual_values = values + offset
    return float(np.max(dual_values)), dual_values

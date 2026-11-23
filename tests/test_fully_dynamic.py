"""Unit tests for fully-dynamic risk measures (src/fully_dynamic.py):
hq-entropic (Eq. (6)/(26)), certainty equivalents (Definition 28), and
h-Value at Risk (Example 15)."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from src.fully_dynamic import (
    certainty_equivalent_rho,
    conditional_expectation,
    gamma_direct,
    h_var_empirical,
    h_var_normal,
    hq_entropic_rho,
    integral_a,
)
from src.qdeformed import exp_q, ln_q


def test_integral_a_constant_function():
    a = integral_a(lambda s: 0.05, 0.2, 0.7)
    assert abs(a - 0.05 * 0.5) < 1e-6


def test_integral_a_rejects_reversed_bounds():
    import pytest

    with pytest.raises(ValueError):
        integral_a(lambda s: 1.0, 0.7, 0.2)


def test_conditional_expectation_at_t0_is_sample_mean():
    payoff = np.array([1.0, 2.0, 3.0, 4.0])
    times = np.linspace(0, 1, 5)
    bm = np.zeros((4, 5))
    ce = conditional_expectation(payoff, times, bm, 0.0)
    assert np.allclose(ce, payoff.mean())


def test_hq_entropic_converges_to_classical_entropic_as_q_to_one():
    """As q -> 1, rho_tu^{hq}(X) -> ln E[exp((X+beta)^- + alpha)] (classical
    entropic risk measure on losses), a(t)=0 (page 18)."""
    rng = np.random.default_rng(0)
    n = 200_000
    times = np.linspace(0, 1, 3)
    bm = np.zeros((n, 3))
    x_u = rng.normal(0.0, 1.0, size=n)
    beta, alpha = 0.0, 0.5
    rho_q = hq_entropic_rho(x_u, times, bm, 0.0, 1.0, 0.999, beta, alpha, lambda s: 0.0)
    classical = np.log(np.mean(np.exp(np.maximum(-(x_u + beta), 0.0) + alpha)))
    assert abs(rho_q.mean() - classical) < 0.05


def test_hq_entropic_normalization_holds_iff_alpha_zero():
    """With a(t)=0, rho_tu^q(0) = alpha_q exactly; normalization (=0) holds
    iff alpha_q = 0 (page 18)."""
    n = 50_000
    times = np.linspace(0, 1, 3)
    bm = np.zeros((n, 3))
    zero = np.zeros(n)
    for alpha in (0.0, 1.0, 2.0):
        rho = hq_entropic_rho(zero, times, bm, 0.0, 1.0, 0.5, 0.0, alpha, lambda s: 0.0)
        assert abs(rho.mean() - alpha) < 1e-6
    assert True  # alpha=0 is the only normalized case, checked above numerically


def test_hq_entropic_increasing_in_alpha_q():
    rng = np.random.default_rng(1)
    n = 100_000
    times = np.linspace(0, 1, 3)
    bm = np.zeros((n, 3))
    x_u = rng.normal(0.0, 1.0, size=n)
    alphas = [0.0, 0.5, 1.0, 2.0, 5.0]
    values = [hq_entropic_rho(x_u, times, bm, 0.0, 1.0, 0.5, 0.0, a, lambda s: 0.0).mean() for a in alphas]
    assert all(values[i] <= values[i + 1] + 1e-9 for i in range(len(values) - 1))


def test_hq_entropic_h_longevity_with_nonnegative_a():
    """gamma(t,u,v,X) = A(u,v) = integral of a(s) from u to v >= 0 when
    a(t)=0.05*t is non-negative (Proposition / Example in Section 3, page 18)."""
    rng = np.random.default_rng(2)
    n = 50_000
    times = np.linspace(0, 1, 5)
    bm = np.zeros((n, 5))
    x_u = rng.normal(0.0, 1.0, size=n)
    a_func = lambda s: 0.05 * s
    rho_tu = hq_entropic_rho(x_u, times, bm, 0.0, 0.5, 0.5, 0.0, 0.0, a_func)
    rho_tv = hq_entropic_rho(x_u, times, bm, 0.0, 1.0, 0.5, 0.0, 0.0, a_func)
    assert rho_tv.mean() >= rho_tu.mean() - 1e-6


def test_certainty_equivalent_exponential_utility_matches_entropic():
    rng = np.random.default_rng(3)
    n = 100_000
    times = np.linspace(0, 1, 3)
    bm = np.zeros((n, 3))
    x_u = rng.normal(0.0, 1.0, size=n)
    util = lambda x: 1.0 - np.exp(-x)
    util_inv = lambda y: -np.log(np.clip(1.0 - y, 1e-300, None))
    rho = certainty_equivalent_rho(x_u, times, bm, 0.0, util, util_inv)
    expected = np.log(np.mean(np.exp(-x_u)))
    assert abs(rho.mean() - expected) < 0.05


def test_h_var_normal_matches_scipy_quantile_at_t0():
    mu, sigma, u, alpha = 0.02, 0.2, 1.0, 0.05
    b_t = np.zeros(5)
    rho = h_var_normal(mu, sigma, 0.0, u, alpha, b_t)
    expected = -(mu * u) - sigma * np.sqrt(u) * norm.ppf(alpha)
    assert np.allclose(rho, expected)


def test_h_var_empirical_matches_quantile():
    rng = np.random.default_rng(4)
    x = rng.normal(0.0, 1.0, size=200_000)
    rho = h_var_empirical(x, 0.05)
    assert abs(rho - (-norm.ppf(0.05))) < 0.05


def test_gamma_direct_definition():
    rho_tv = np.array([1.0, 2.0])
    rho_tu = np.array([0.5, 1.5])
    assert np.allclose(gamma_direct(rho_tv, rho_tu), [0.5, 0.5])

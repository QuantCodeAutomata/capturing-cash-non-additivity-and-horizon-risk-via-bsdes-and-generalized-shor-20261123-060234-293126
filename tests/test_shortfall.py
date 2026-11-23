"""Unit tests for generalized shortfall risk measures (src/shortfall.py),
Definition 16 and Proposition 17 of the paper."""

from __future__ import annotations

import numpy as np

from src.shortfall import (
    acceptance_indicator,
    aggregator_classical,
    aggregator_exponential,
    aggregator_linear,
    entropic_shortfall_closed_form,
    generalized_shortfall,
    utility_exponential,
    utility_identity,
)


def test_classical_shortfall_exponential_utility_matches_closed_form():
    """Classical aggregator f(x,m)=x+m with U(x)=1-exp(-x/gamma) reproduces
    the entropic risk measure in closed form (Foellmer & Schied Ex. 4.115)."""
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=200_000)
    gamma, target_b = 1.5, 0.3
    numeric = generalized_shortfall(
        x, lambda z: utility_exponential(z, gamma),
        lambda z, m: aggregator_classical(z, m), target_b,
    )
    closed_form = entropic_shortfall_closed_form(x, gamma, target_b)
    assert abs(numeric - closed_form) < 0.01


def test_linear_aggregator_beta_one_is_cash_additive():
    """beta=1 reduces to the classical (cash additive) shortfall:
    rho(X+m) = rho(X) - m exactly."""
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, size=100_000)
    agg = lambda z, m: aggregator_linear(z, m, beta=1.0)
    rho_x = generalized_shortfall(x, utility_identity, agg, target_b=0.0)
    m = 0.5
    rho_shift = generalized_shortfall(x + m, utility_identity, agg, target_b=0.0)
    assert abs(rho_shift - (rho_x - m)) < 1e-6


def test_linear_aggregator_beta_below_one_is_cash_subadditive_strictly():
    """beta<1: rho(X+m) > rho(X) - m for m>0 (Example 19 i))."""
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, size=200_000)
    beta = 0.7
    agg = lambda z, m: aggregator_linear(z, m, beta=beta)
    rho_x = generalized_shortfall(x, lambda z: utility_exponential(z, 1.0), agg, target_b=0.3)
    m = 0.5
    rho_shift = generalized_shortfall(x + m, lambda z: utility_exponential(z, 1.0), agg, target_b=0.3)
    assert rho_shift > (rho_x - m) - 1e-6
    assert rho_shift - (rho_x - m) > 1e-3  # strictly, not just numerically equal


def test_exponential_aggregator_monotone_condition_f_y_le_f_k():
    """Proposition 17 item 2c: d f/d y <= d f/d k for f(y,k)=1-exp(-nu*y-k)
    with nu in (0,1), which is the condition underlying cash subadditivity."""
    nu = 0.4
    y, k, h = 0.3, 0.1, 1e-5
    df_dy = (aggregator_exponential(y + h, k, nu) - aggregator_exponential(y - h, k, nu)) / (2 * h)
    df_dk = (aggregator_exponential(y, k + h, nu) - aggregator_exponential(y, k - h, nu)) / (2 * h)
    assert df_dy <= df_dk + 1e-6


def test_quasi_convexity_of_generalized_shortfall():
    """rho(lambda*X + (1-lambda)*Y) <= max(rho(X), rho(Y)) for all lambda."""
    rng = np.random.default_rng(3)
    n = 100_000
    x = rng.normal(0.0, 1.0, size=n)
    y = rng.normal(0.5, 2.0, size=n)
    agg = lambda z, m: aggregator_classical(z, m)
    util = lambda z: utility_exponential(z, 1.0)
    rho_x = generalized_shortfall(x, util, agg, target_b=0.3)
    rho_y = generalized_shortfall(y, util, agg, target_b=0.3)
    bound = max(rho_x, rho_y)
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        mixed = lam * x + (1 - lam) * y
        rho_mixed = generalized_shortfall(mixed, util, agg, target_b=0.3)
        assert rho_mixed <= bound + 0.05


def test_acceptance_indicator_consistent_with_shortfall_root():
    rng = np.random.default_rng(4)
    x = rng.normal(0.0, 1.0, size=50_000)
    agg = lambda z, m: aggregator_classical(z, m)
    util = lambda z: utility_exponential(z, 1.0)
    m_star = generalized_shortfall(x, util, agg, target_b=0.3)
    assert acceptance_indicator(x, util, agg, 0.3, m_star + 1e-2)
    assert not acceptance_indicator(x, util, agg, 0.3, m_star - 1e-2)

"""Unit tests for the Tsallis q-deformed exponential and logarithm
(src/qdeformed.py), paper Section 3, page 4."""

from __future__ import annotations

import numpy as np

from src.qdeformed import exp_q, ln_q, q_domain_floor


def test_exp_q_converges_to_exp_as_q_to_one():
    x = np.linspace(-2.0, 2.0, 21)
    errors = [np.max(np.abs(exp_q(x, q) - np.exp(x))) for q in (0.9, 0.99, 0.999)]
    assert errors[0] > errors[1] > errors[2], "error should shrink as q -> 1"
    assert errors[2] < 0.05, f"exp_q(q=0.999) should be close to exp, got {errors[2]}"
    assert np.allclose(exp_q(x, 1.0), np.exp(x))


def test_ln_q_converges_to_ln_as_q_to_one():
    x = np.linspace(0.1, 5.0, 21)
    errors = [np.max(np.abs(ln_q(x, q) - np.log(x))) for q in (0.9, 0.99, 0.999)]
    assert errors[0] > errors[1] > errors[2], "error should shrink as q -> 1"
    assert errors[2] < 0.05, f"ln_q(q=0.999) should be close to ln, got {errors[2]}"
    assert np.allclose(ln_q(x, 1.0), np.log(x))


def test_exp_q_ln_q_are_mutual_inverses_on_domain():
    q = 0.6
    x = np.linspace(0.0, 2.0, 10)  # exp_q(x) >= 0, ln_q defined there
    y = exp_q(x, q)
    assert np.allclose(ln_q(y, q), x, atol=1e-8)


def test_exp_q_zero_below_domain_floor():
    q = 0.4
    floor = q_domain_floor(q)
    assert floor < 0.0  # 1/(q-1) < 0 for q in (0,1)
    below = floor - 1.0
    assert exp_q(np.array([below]), q)[0] == 0.0


def test_ln_q_nonnegative_domain_and_monotone():
    q = 0.7
    x = np.linspace(0.01, 10.0, 50)
    vals = ln_q(x, q)
    assert np.all(np.diff(vals) > 0.0)  # ln_q strictly increasing


def test_invalid_q_raises():
    import pytest

    with pytest.raises(ValueError):
        exp_q(np.array([0.0]), 1.5)
    with pytest.raises(ValueError):
        ln_q(np.array([1.0]), -0.2)

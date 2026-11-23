"""Tsallis q-deformed exponential and logarithm (paper Section 3, page 4).

The hq-entropic risk measure on losses is defined through the generalized
exponential and logarithm functions

    exp_q(x) = [1 + (1 - q) x]^{1/(1-q)},
    ln_q(x)  = (x^{1-q} - 1) / (1 - q),

for q in (0, 1); see [33, 34] as cited in the paper. Both functions converge
to the classical exp and ln as q -> 1.
"""

from __future__ import annotations

import numpy as np


def exp_q(x: np.ndarray, q: float) -> np.ndarray:
    """Tsallis q-exponential, exp_q(x) = [1 + (1-q)x]_+^{1/(1-q)}.

    The domain condition x >= 1/(q-1) (paper, page 4) keeps the base
    non-negative; outside the domain the standard convention exp_q(x) = 0
    is used (positive-part convention of the Tsallis q-exponential).
    """
    x = np.asarray(x, dtype=float)
    if not (0.0 < q < 1.0 or np.isclose(q, 1.0)):
        raise ValueError("q must lie in (0, 1] for this implementation")
    if np.isclose(q, 1.0):
        return np.exp(x)
    base = 1.0 + (1.0 - q) * x
    result = np.where(base > 0.0, np.power(np.clip(base, a_min=1e-300, a_max=None), 1.0 / (1.0 - q)), 0.0)
    return result


def ln_q(x: np.ndarray, q: float) -> np.ndarray:
    """Tsallis q-logarithm, ln_q(x) = (x^{1-q} - 1) / (1 - q), x >= 0."""
    x = np.asarray(x, dtype=float)
    if not (0.0 < q < 1.0 or np.isclose(q, 1.0)):
        raise ValueError("q must lie in (0, 1] for this implementation")
    if np.isclose(q, 1.0):
        return np.log(np.clip(x, a_min=1e-300, a_max=None))
    x_clipped = np.clip(x, a_min=0.0, a_max=None)
    return (np.power(x_clipped, 1.0 - q) - 1.0) / (1.0 - q)


def q_domain_floor(q: float) -> float:
    """Lower bound 1/(q-1) of the domain of exp_q / ln_q for q in (0,1)."""
    if np.isclose(q, 1.0):
        return -np.inf
    return 1.0 / (q - 1.0)

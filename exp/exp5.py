"""Experiment 5: fully-dynamic certainty equivalent risk measures of

    Di Nunno, G. and Rosazza Gianin, E., "Capturing cash non-additivity and
    horizon risk via BSDEs and generalized shortfall".

Replicates, on simulated Brownian paths, the methodology around
Definition 28 (fully-dynamic certainty equivalents, Eq. 52), the
equivalence with h-generalized shortfall risk measures (Eq. 53), horizon
longevity for certainty equivalents, the CRRA utility family, and Example 13
("h-entropic as shortfall"):

  1. simulate shared Brownian paths;
  2. verify that the h-entropic certainty equivalent reproduces the
     classical entropic risk measure shifted by the horizon function A;
  3. verify the equivalence between that certainty equivalent and a
     h-generalized shortfall risk measure built from the aggregator
     f_tu(y,m) = Util^{-1}(Util(y) - Util(-m)) (Eq. 53);
  4. verify horizon longevity for a non-increasing-in-u utility family;
  5. sanity-check the CRRA utility family on a positive terminal condition;
  6. a pointwise acceptance-set sanity check;
  7. Example 13: h-entropic risk measure recovered via the *classical*
     shortfall aggregator f(y,m) = y+m with an explicit threshold B_tu;
  8. the classical entropic risk measure recovered simultaneously as a
     certainty equivalent, a q -> 1 boundary case of the hq-entropic family
     of src.fully_dynamic, and a shortfall value.

Run: python -m exp.exp5
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from data.generate_synthetic import (
    simulate_brownian_paths,
    terminal_constant,
    terminal_from_bm,
)
from src import config
from src.fully_dynamic import (
    certainty_equivalent_rho,
    conditional_expectation,
    hq_entropic_rho,
    integral_a,
)
from src.shortfall import utility_crra, utility_crra_inverse

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

Vec = np.ndarray
AFunc = Callable[[float], float]
UFunc = Callable[[Vec], Vec]

TOL_STEP2 = 1e-2
TOL_STEP3 = 1e-2
TOL_LONGEVITY = 5e-2
TOL_STEP7 = 1e-2
TOL_STEP8 = 1e-2


# --------------------------------------------------------------------------
# Horizon functions a(s) >= 0 (Section 3)
# --------------------------------------------------------------------------
def constant_a(c: float) -> AFunc:
    def a_func(_s: float, c: float = c) -> float:
        return c

    return a_func


def linear_a(c: float) -> AFunc:
    def a_func(s: float, c: float = c) -> float:
        return c * s

    return a_func


def _safe_log1m(z: Vec) -> Vec:
    """ln(1-z), clipped so that z -> 1 (or beyond, a regression/root-finding
    artifact) maps to a large-but-finite value instead of -inf/nan."""
    return np.log(np.clip(1.0 - np.asarray(z, dtype=float), 1e-300, None))


# --------------------------------------------------------------------------
# h-entropic utility family (steps 2, 3, 6, 7, 8): Util_s(x) = 1 - exp(-x +
# A(0,s)), a time-indexed family of exponential (entropic-type) utilities.
#
# Inverse: solving y = 1 - exp(-x + A(0,s)) for x:
#     exp(-x + A(0,s)) = 1 - y
#     -x + A(0,s) = ln(1 - y)
#     x = A(0,s) - ln(1 - y) = -ln(1 - y) + A(0,s),   valid for y < 1.
# --------------------------------------------------------------------------
def h_entropic_util(s: float, a_func: AFunc) -> tuple[UFunc, UFunc]:
    a0s = integral_a(a_func, 0.0, s)

    def util(x: Vec) -> Vec:
        return 1.0 - np.exp(-np.asarray(x, dtype=float) + a0s)

    def util_inv(y: Vec) -> Vec:
        return _safe_log1m(y) * (-1.0) + a0s  # -ln(1-y) + a0s, see derivation above

    return util, util_inv


# --------------------------------------------------------------------------
# Horizon-longevity utility family (step 4): Util_s(x) = (1 - exp(-x)) *
# exp(-A(0,s)), non-increasing in s whenever a_func(.) >= 0 (so A(0,s) is
# non-decreasing in s, exp(-A(0,s)) non-increasing, and Util_s(x) >= 0 scales
# down as s grows for every x > 0 -- the mechanism behind horizon longevity).
#
# Inverse: y = (1 - exp(-x)) * exp(-A(0,s))
#     1 - exp(-x) = y * exp(A(0,s))
#     exp(-x) = 1 - y * exp(A(0,s))
#     x = -ln(1 - y * exp(A(0,s))),   valid for y * exp(A(0,s)) < 1.
# --------------------------------------------------------------------------
def h_longevity_util(s: float, a_func: AFunc) -> tuple[UFunc, UFunc]:
    a0s = integral_a(a_func, 0.0, s)
    scale = np.exp(-a0s)

    def util(x: Vec) -> Vec:
        return (1.0 - np.exp(-np.asarray(x, dtype=float))) * scale

    def util_inv(y: Vec) -> Vec:
        return -_safe_log1m(np.asarray(y, dtype=float) / scale)

    return util, util_inv


# --------------------------------------------------------------------------
# Aggregator of Eq. (53), U_u = Util_u, C_t = 1, B_tu = 0:
#     f_tu(y, m) = Util_u^{-1}(Util_u(y) - Util_u(-m)).
# --------------------------------------------------------------------------
def make_f_tu(util: UFunc, util_inv: UFunc) -> Callable[[Vec, float], Vec]:
    def f_tu(y: Vec, m: float) -> Vec:
        return util_inv(util(y) - util(-m))

    return f_tu


def _to_native(obj):
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# --------------------------------------------------------------------------
# Shared (a_func, t, u) grid for steps 2-3, plus the terminal conditions X
# tested at each (t, u) pair (B_u always; constants only at t = 0, where the
# conditional expectation reduces to a plain sample mean, following the
# exp_1/exp_4 convention -- constants are F_t-measurable for every t, so
# testing them only at t = 0 is a scope choice, not a correctness issue).
# --------------------------------------------------------------------------
def a_func_grid() -> list[tuple[str, AFunc]]:
    grid = [(f"const_{c}", constant_a(c)) for c in config.EXP5_A_CONST_GRID]
    grid.append(("linear_0.05", linear_a(0.05)))
    return grid


def tu_pairs() -> list[tuple[float, float]]:
    pairs = []
    for t in config.EXP5_T_LIST:
        for u in config.EXP5_U_LIST:
            if u > t:
                pairs.append((t, u))
    return pairs


def x_candidates_at(t: float, u: float, paths: np.ndarray, times: np.ndarray, n_paths: int) -> dict[str, Vec]:
    cands = {"B_u": terminal_from_bm(paths, times, u)}
    if np.isclose(t, 0.0):
        cands["const_0.0"] = terminal_constant(n_paths, 0.0)
        cands["const_0.5"] = terminal_constant(n_paths, 0.5)
    return cands


# --------------------------------------------------------------------------
# Step 2: certainty equivalent reproduces the classical entropic risk
# measure shifted by the horizon function, rho_tu(X) = ln E[exp(-X +
# A(t,u)) | F_t] (Eq. 5 with b = 1).
#
# certainty_equivalent_rho(x_u, times, bm, t, util, util_inv, basis_degree)
# computes -util_inv(E[util(x_u)|F_t]) (Eq. 52) using a SINGLE (util,
# util_inv) pair. Reproducing the target formula requires feeding it
# Util_u (A(0,u)-shifted, matching the measurability of X at u) as the
# forward map and Util_t^{-1} (A(0,t)-shifted, matching the evaluation date
# t) as the inverse map -- i.e. the two ends of the SAME h-entropic family
# evaluated at the two different times t and u, not the same shift applied
# twice. Algebraically, with a0u = A(0,u), a0t = A(0,t):
#     -Util_t^{-1}(E[Util_u(X)|F_t])
#   = -(-ln(1 - E[1-exp(-X+a0u)|F_t]) + a0t)
#   = ln(exp(a0u) E[exp(-X)|F_t]) - a0t = ln E[exp(-X)|F_t] + (a0u - a0t)
#   = ln E[exp(-X)|F_t] + A(t,u) = ln E[exp(-X + A(t,u)) | F_t].
# (Using the same shift a0u on both ends instead would make A(0,u) cancel
# out completely and reproduce the *unshifted* classical entropic risk
# measure ln E[exp(-X)|F_t] -- a different, equally valid but not
# horizon-shifted certainty equivalent, used in step 3 below.)
# --------------------------------------------------------------------------
def step2_entropic_certainty_equivalent(times: Vec, bm: Vec, paths: Vec, n_paths: int, basis_degree: int) -> list[dict]:
    rows = []
    for a_name, a_func in a_func_grid():
        for t, u in tu_pairs():
            util_u, _ = h_entropic_util(u, a_func)
            _, util_t_inv = h_entropic_util(t, a_func)
            a_tu = integral_a(a_func, t, u)
            x_cands = x_candidates_at(t, u, paths, times, n_paths)
            for x_name, x_vals in x_cands.items():
                rho_ce = certainty_equivalent_rho(x_vals, times, bm, t, util_u, util_t_inv, basis_degree)
                direct_raw = conditional_expectation(np.exp(-x_vals + a_tu), times, bm, t, basis_degree)
                rho_direct = np.log(np.clip(direct_raw, 1e-300, None))
                ce_mean, direct_mean = float(np.mean(rho_ce)), float(np.mean(rho_direct))
                rows.append({
                    "a_func": a_name, "t": t, "u": u, "X": x_name,
                    "rho_ce_mean": ce_mean, "rho_direct_mean": direct_mean,
                    "abs_diff": abs(ce_mean - direct_mean),
                    "agree": bool(abs(ce_mean - direct_mean) < TOL_STEP2),
                })
    return rows


# --------------------------------------------------------------------------
# Step 3: equivalence with h-generalized shortfall (Eq. 53), checked at the
# unconditional (t = 0) level, where it reduces exactly to the static
# Definition 16 ("rho^{U,f,B}(X) = inf{m : E[U(f(X,m))] >= B}"): for t > 0,
# certainty_equivalent_rho computes a genuinely F_t-measurable (path-varying,
# regression-based) quantity, and E[Util_u(X)|F_t=b_t] varies with b_t, so
# the *unconditional* mean of a nonlinear transform of that conditional
# expectation is generally not equal to the same nonlinear transform of the
# *unconditional* mean (Jensen), even though both are legitimate dynamic
# risk measures. The static identity below is therefore verified at t = 0,
# where this subtlety does not arise; see the module docstring.
#
# The outer utility applied to f_tu is Util_u itself (not "identity" as a
# literal reading of f_tu(y,m) = Util_u^{-1}(Util_u(y) - Util_u(-m)) might
# suggest), because Util_u(f_tu(y,m)) = Util_u(y) - Util_u(-m) exactly (the
# inverse cancels), and this is precisely the algebraic content of Eq. 53:
#
#     E[Util_u(f_tu(X,m))] >= 0  <=>  E[Util_u(X)] >= Util_u(-m)
#                               <=>  m >= -Util_u^{-1}(E[Util_u(X)]) = rho_tu(X).
#
# For robustness we evaluate the bisected quantity as E[Util_u(X)] -
# Util_u(-m) directly (algebraically identical to E[Util_u(f_tu(X,m))], but
# avoiding materializing f_tu via Util_u^{-1}, whose domain y < 1 can
# otherwise be violated for some (y, m) pairs visited during the bisection,
# e.g. when |X| is large and m is far from the root).
# --------------------------------------------------------------------------
def step3_shortfall_equivalence(times: Vec, bm: Vec, paths: Vec, n_paths: int, basis_degree: int) -> list[dict]:
    rows = []
    t = 0.0
    for a_name, a_func in a_func_grid():
        for u in config.EXP5_U_LIST:
            util_u, util_u_inv = h_entropic_util(u, a_func)
            x_cands = x_candidates_at(t, u, paths, times, n_paths)
            for x_name, x_vals in x_cands.items():
                rho_ce = certainty_equivalent_rho(x_vals, times, bm, t, util_u, util_u_inv, basis_degree)
                rho_ce_mean = float(np.mean(rho_ce))

                def h(m: float, x_vals=x_vals, util_u=util_u) -> float:
                    return float(np.mean(util_u(x_vals))) - float(util_u(-m))

                # h is increasing in m (util_u(-m) decreases as m grows), so
                # bisection needs h(lo) <= 0 <= h(hi), expanding the bracket
                # the same way as src.shortfall.generalized_shortfall.
                lo, hi, expand = -10.0, 10.0, 0
                h_lo, h_hi = h(lo), h(hi)
                while h_lo > 0.0 and expand < 20:
                    lo -= (hi - lo)
                    h_lo = h(lo)
                    expand += 1
                while h_hi < 0.0 and expand < 40:
                    hi += (hi - lo)
                    h_hi = h(hi)
                    expand += 1
                m_star = brentq(h, lo, hi, xtol=1e-8)

                rows.append({
                    "a_func": a_name, "t": t, "u": u, "X": x_name,
                    "rho_ce_mean": rho_ce_mean, "m_star": m_star,
                    "abs_diff": abs(rho_ce_mean - m_star),
                    "agree": bool(abs(rho_ce_mean - m_star) < TOL_STEP3),
                })
    return rows


# --------------------------------------------------------------------------
# Step 4: horizon longevity for the (non-increasing-in-u) utility family
# Util_s(x) = (1-exp(-x))*exp(-A(0,s)). Same X = B_u (F_u-measurable, hence
# also F_v-measurable for v >= u) is passed to both Util_u and Util_v; only
# the utility index changes.
# --------------------------------------------------------------------------
def step4_horizon_longevity(times: Vec, bm: Vec, paths: Vec, basis_degree: int) -> list[dict]:
    rows = []
    longevity_a_funcs = [(f"const_{c}", constant_a(c)) for c in config.EXP5_A_CONST_GRID if c >= 0.0]
    longevity_a_funcs.append(("linear_0.05", linear_a(0.05)))

    for a_name, a_func in longevity_a_funcs:
        for t in config.EXP5_T_LIST:
            for u in config.EXP5_U_LIST:
                if u <= t:
                    continue
                x_u = terminal_from_bm(paths, times, u)
                util_u, util_u_inv = h_longevity_util(u, a_func)
                rho_tu = certainty_equivalent_rho(x_u, times, bm, t, util_u, util_u_inv, basis_degree)
                for v in config.EXP5_V_LIST:
                    if v < u:
                        continue
                    util_v, util_v_inv = h_longevity_util(v, a_func)
                    rho_tv = certainty_equivalent_rho(x_u, times, bm, t, util_v, util_v_inv, basis_degree)
                    mean_tu, mean_tv = float(np.mean(rho_tu)), float(np.mean(rho_tv))
                    rows.append({
                        "a_func": a_name, "t": t, "u": u, "v": v,
                        "rho_tu_mean": mean_tu, "rho_tv_mean": mean_tv,
                        "longevity_holds": bool(mean_tv >= mean_tu - TOL_LONGEVITY),
                    })
    return rows


# --------------------------------------------------------------------------
# Step 5: CRRA utility family, sanity check on a strictly positive terminal
# condition X = exp(0.1 * B_u) > 0.
# --------------------------------------------------------------------------
def step5_crra_sanity(times: Vec, bm: Vec, paths: Vec, basis_degree: int) -> list[dict]:
    rows = []
    for gamma in config.EXP5_CRRA_GAMMAS:
        for t, u in [(0.0, config.EXP5_U_LIST[0]), (config.EXP5_T_LIST[1], config.EXP5_U_LIST[1])]:
            x_u = terminal_from_bm(paths, times, u, func=lambda b: np.exp(0.1 * b))
            rho = certainty_equivalent_rho(
                x_u, times, bm, t,
                lambda x: utility_crra(x, gamma), lambda y: utility_crra_inverse(y, gamma),
                basis_degree,
            )
            all_finite = bool(np.all(np.isfinite(rho)))
            rows.append({
                "gamma": gamma, "t": t, "u": u, "rho_mean": float(np.mean(rho)),
                "rho_std": float(np.std(rho)), "all_finite": all_finite,
            })
            assert all_finite, f"CRRA certainty equivalent produced non-finite values for gamma={gamma}, t={t}, u={u}"
    return rows


# --------------------------------------------------------------------------
# Step 6: pointwise acceptance-set sanity check. For one representative
# (t, u, a_func) case from step 3, verify that
#     E[Util_u(Y)|F_t] >= Util_u(-m)   <=>   E[f_tu(Y,m)] >= 0
# literally as written (f_tu built with Util_u on both sides, outer U =
# identity for the right-hand side), for a grid of candidate Y and m. Since
# Util_u(0) != 0 in general (it equals 1 - exp(A(0,u)), zero only when
# A(0,u) = 0), these two conditions need not coincide exactly whenever
# A(0,u) != 0: mismatches are reported, not hidden (per the methodology
# note), rather than forced to pass.
# --------------------------------------------------------------------------
def step6_acceptance_set_check(times: Vec, bm: Vec, paths: Vec, n_paths: int, basis_degree: int, step3_rows: list[dict]) -> dict:
    rep = next(r for r in step3_rows if r["X"] == "B_u" and r["a_func"] == "const_0.05")
    t, u, a_name = rep["t"], rep["u"], rep["a_func"]
    a_func = dict(a_func_grid())[a_name]
    util_u, util_u_inv = h_entropic_util(u, a_func)
    f_tu = make_f_tu(util_u, util_u_inv)

    rho_vals = [r["rho_ce_mean"] for r in step3_rows if r["a_func"] == a_name]
    lo_rho, hi_rho = min(rho_vals), max(rho_vals)

    b_u = terminal_from_bm(paths, times, u)
    y_candidates: dict[str, Vec] = {
        f"const_{lvl:.3f}": terminal_constant(n_paths, lvl)
        for lvl in np.linspace(lo_rho - 1.0, hi_rho + 1.0, 10)
    }
    y_candidates["B_u"] = b_u
    y_candidates["B_u_plus_0.5"] = b_u + 0.5
    y_candidates["2_times_B_u"] = 2.0 * b_u

    m_grid = np.linspace(lo_rho - 1.0, hi_rho + 1.0, 15)

    rows = []
    for y_name, y_vals in y_candidates.items():
        cond_exp_util_y = conditional_expectation(util_u(y_vals), times, bm, t, basis_degree)
        for m in m_grid:
            cond1 = bool(np.mean(cond_exp_util_y) >= util_u(-m))
            cond2 = bool(np.mean(f_tu(y_vals, m)) >= 0.0)
            rows.append({
                "Y": y_name, "m": float(m), "cond1_rho_le_m": cond1, "cond2_shortfall_accept": cond2,
                "match": bool(cond1 == cond2),
            })
    n_mismatch = sum(1 for r in rows if not r["match"])
    return {
        "representative_case": {"t": t, "u": u, "a_func": a_name},
        "rows": rows,
        "n_checked": len(rows),
        "n_mismatch": n_mismatch,
        "all_match": bool(n_mismatch == 0),
    }


# --------------------------------------------------------------------------
# Step 7 (Example 13): the SAME h-entropic rho_tu recovered via the
# *classical* shortfall aggregator f(y,m) = y + m (not the Eq. 53 f_tu),
# outer utility U_u = Util_u, and the explicit threshold B_tu = 1 -
# exp(A(0,t)):
#     E[U_u(X+m)] >= B_tu
#  <=> E[1 - exp(-(X+m)+A(0,u))] >= 1 - exp(A(0,t))
#  <=> exp(-m)*exp(A(0,u))*E[exp(-X)] <= exp(A(0,t))
#  <=> m >= A(0,u) - A(0,t) + ln E[exp(-X)] = A(t,u) + ln E[exp(-X)]
#      = ln E[exp(-X + A(t,u))]  (the step-2 target formula).
#
# As in step 3, the bisection below uses the *unconditional* mean of
# Util_u(X+m), so the identity above is exact at t = 0; for t > 0 it is only
# an approximation of the genuine F_t-conditional root (same Jensen-gap
# caveat discussed in step 3's docstring), which is reported but not
# asserted to the strict tolerance.
# --------------------------------------------------------------------------
def step7_example13_shortfall(times: Vec, bm: Vec, paths: Vec, n_paths: int, basis_degree: int) -> list[dict]:
    rows = []
    for a_name, a_func in a_func_grid():
        for t, u in tu_pairs():
            util_u, _ = h_entropic_util(u, a_func)
            a0t = integral_a(a_func, 0.0, t)
            b_tu = 1.0 - np.exp(a0t)
            x_cands = x_candidates_at(t, u, paths, times, n_paths)
            for x_name, x_vals in x_cands.items():
                def h(m: float, x_vals=x_vals, util_u=util_u, b_tu=b_tu) -> float:
                    return float(np.mean(util_u(x_vals + m))) - b_tu

                lo, hi, expand = -10.0, 10.0, 0
                h_lo, h_hi = h(lo), h(hi)
                while h_lo > 0.0 and expand < 20:
                    lo -= (hi - lo)
                    h_lo = h(lo)
                    expand += 1
                while h_hi < 0.0 and expand < 40:
                    hi += (hi - lo)
                    h_hi = h(hi)
                    expand += 1
                m_star = brentq(h, lo, hi, xtol=1e-8)

                util_u_fwd, util_t_inv = h_entropic_util(u, a_func)[0], h_entropic_util(t, a_func)[1]
                rho_ce_mixed = float(np.mean(certainty_equivalent_rho(x_vals, times, bm, t, util_u_fwd, util_t_inv, basis_degree)))
                abs_diff = abs(rho_ce_mixed - m_star)
                rows.append({
                    "a_func": a_name, "t": t, "u": u, "X": x_name,
                    "rho_ce_mixed_mean": rho_ce_mixed, "m_star": m_star,
                    "abs_diff": abs_diff,
                    "agree": bool(abs_diff < TOL_STEP7),
                    "strict_check": bool(np.isclose(t, 0.0)),
                })
    return rows


# --------------------------------------------------------------------------
# Step 8: classical entropic triple representation at a_func = 0 (A ≡ 0).
# hq_entropic_rho operates on (X+beta)^-, the negative part of a *loss*; it
# coincides with the certainty equivalent / shortfall formulas acting
# directly on -X only when X <= 0 almost surely (so that (X+beta)^- = -X-
# beta identically, not just on the event {X <= -beta}). X = -(B_u)^2 (a
# genuine functional of B_u, via terminal_from_bm) is always <= 0 and makes
# the three formulas coincide exactly at beta = 0.
# --------------------------------------------------------------------------
def step8_classical_entropic_triple(times: Vec, bm: Vec, paths: Vec, basis_degree: int) -> dict:
    # t = 0: conditional_expectation reduces to a plain Monte Carlo average
    # (no polynomial regression), which matters here because exp(-X) =
    # exp(B_u^2) is extremely heavy-tailed and a low-degree polynomial fit
    # of it against B_t (needed for t > 0) is numerically unreliable
    # (classic polynomial-extrapolation blow-up on a heavy-tailed target).
    t, u = 0.0, 0.5
    zero_a = constant_a(0.0)
    x_u = terminal_from_bm(paths, times, u, func=lambda b: -(b**2))

    util_u, util_u_inv = h_entropic_util(u, zero_a)  # A(0,.) = 0, same family at t and u
    rho_ce = float(np.mean(certainty_equivalent_rho(x_u, times, bm, t, util_u, util_u_inv, basis_degree)))

    rho_hq = float(np.mean(hq_entropic_rho(x_u, times, bm, t, u, q=1.0, beta=0.0, alpha_q=0.0, a_func=zero_a, basis_degree=basis_degree)))

    b_tu = 1.0 - np.exp(0.0)

    def h(m: float) -> float:
        return float(np.mean(util_u(x_u + m))) - b_tu

    lo, hi, expand = -10.0, 10.0, 0
    h_lo, h_hi = h(lo), h(hi)
    while h_lo > 0.0 and expand < 20:
        lo -= (hi - lo)
        h_lo = h(lo)
        expand += 1
    while h_hi < 0.0 and expand < 40:
        hi += (hi - lo)
        h_hi = h(hi)
        expand += 1
    rho_shortfall = brentq(h, lo, hi, xtol=1e-8)

    d_ce_hq = abs(rho_ce - rho_hq)
    d_ce_sf = abs(rho_ce - rho_shortfall)
    d_hq_sf = abs(rho_hq - rho_shortfall)
    return {
        "t": t, "u": u, "rho_certainty_equivalent": rho_ce, "rho_hq_entropic_q1": rho_hq,
        "rho_shortfall_example13": rho_shortfall,
        "diff_ce_hq": d_ce_hq, "diff_ce_shortfall": d_ce_sf, "diff_hq_shortfall": d_hq_sf,
        "all_agree": bool(max(d_ce_hq, d_ce_sf, d_hq_sf) < TOL_STEP8),
    }


# --------------------------------------------------------------------------
# Plot: horizon longevity, mean(rho_tu) vs u, fixed (t, a_func).
# --------------------------------------------------------------------------
def plot_horizon_longevity(longevity_rows: list[dict]) -> dict:
    t_fixed = 0.0
    a_fixed = "const_0.05"
    horizons = sorted(set(config.EXP5_U_LIST + config.EXP5_V_LIST))

    series: dict[float, float] = {}
    for r in longevity_rows:
        if r["t"] != t_fixed or r["a_func"] != a_fixed:
            continue
        series[r["u"]] = r["rho_tu_mean"]
        series[r["v"]] = r["rho_tv_mean"]
    xs = [h for h in horizons if h in series]
    ys = [series[h] for h in xs]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, marker="o", label=rf"$t={t_fixed}$, $a(s)={a_fixed.split('_')[1]}$")
    ax.set_xlabel("horizon u")
    ax.set_ylabel(r"mean $\rho_{tu}^{Util_u}(B_u)$")
    ax.set_title("Experiment 5: horizon longevity of the certainty equivalent\n"
                  r"($(Util_s)_s$ non-increasing in $s$: $\rho_{tu} \leq \rho_{tv}$ for $u \leq v$)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp5_horizon_longevity.png", dpi=150)
    plt.close(fig)
    return {"t": t_fixed, "a_func": a_fixed, "horizons": xs, "rho_means": ys}


def _parse_args() -> bool:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", action="store_true",
        help="use the paper's full Monte Carlo configuration (EXP5_N_PATHS_PAPER) "
        "instead of the reduced run configuration (EXP5_N_PATHS_RUN)",
    )
    return parser.parse_args().full


def main() -> None:
    full = _parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    t_start = time.time()

    n_paths = config.EXP5_N_PATHS_PAPER if full else config.EXP5_N_PATHS_RUN
    n_steps = config.EXP5_N_STEPS
    times, paths = simulate_brownian_paths(n_paths, n_steps, config.T_HORIZON, config.BM_DIMENSION, seed=config.RANDOM_SEED)
    bm = paths[:, :, 0]
    print(f"Simulated n_paths={n_paths}, n_steps={n_steps}, T={config.T_HORIZON}, d={config.BM_DIMENSION}")

    report: dict = {"n_paths": n_paths, "n_steps": n_steps}
    basis_degree = config.BASIS_DEGREE

    # Step 2 ------------------------------------------------------------
    step2_rows = step2_entropic_certainty_equivalent(times, bm, paths, n_paths, basis_degree)
    report["step2_entropic_certainty_equivalent"] = step2_rows
    step2_pass = all(r["agree"] for r in step2_rows)
    print(f"[Step 2] certainty equivalent == ln E[exp(-X+A(t,u))|F_t]: {step2_pass} "
          f"({len(step2_rows)} cases, tol={TOL_STEP2})")

    # Step 3 ------------------------------------------------------------
    step3_rows = step3_shortfall_equivalence(times, bm, paths, n_paths, basis_degree)
    report["step3_shortfall_equivalence"] = step3_rows
    step3_pass = all(r["agree"] for r in step3_rows)
    print(f"[Step 3] certainty equivalent == generalized-shortfall root (Eq. 53, t=0): {step3_pass} "
          f"({len(step3_rows)} cases, tol={TOL_STEP3})")

    # Step 4 ------------------------------------------------------------
    step4_rows = step4_horizon_longevity(times, bm, paths, basis_degree)
    report["step4_horizon_longevity"] = step4_rows
    step4_pass = all(r["longevity_holds"] for r in step4_rows)
    print(f"[Step 4] horizon longevity (mean(rho_tv) >= mean(rho_tu) - {TOL_LONGEVITY}): {step4_pass} "
          f"({len(step4_rows)} triples)")

    # Step 5 ------------------------------------------------------------
    step5_rows = step5_crra_sanity(times, bm, paths, basis_degree)
    report["step5_crra_sanity"] = step5_rows
    print(f"[Step 5] CRRA certainty equivalent finite on positive X for all "
          f"{len(step5_rows)} (gamma, t, u) cases: {all(r['all_finite'] for r in step5_rows)}")

    # Step 6 ------------------------------------------------------------
    step6_result = step6_acceptance_set_check(times, bm, paths, n_paths, basis_degree, step3_rows)
    report["step6_acceptance_set_check"] = step6_result
    print(f"[Step 6] pointwise acceptance-set sanity check: all_match={step6_result['all_match']} "
          f"({step6_result['n_mismatch']} mismatches out of {step6_result['n_checked']} "
          f"-- mismatches expected when A(0,u) != 0, see module docstring)")

    # Step 7 ------------------------------------------------------------
    step7_rows = step7_example13_shortfall(times, bm, paths, n_paths, basis_degree)
    report["step7_example13_shortfall"] = step7_rows
    step7_strict_rows = [r for r in step7_rows if r["strict_check"]]
    step7_pass = all(r["agree"] for r in step7_strict_rows)
    print(f"[Step 7] Example 13 (classical aggregator f(y,m)=y+m, B_tu=1-exp(A(0,t))) "
          f"reproduces rho_tu at t=0: {step7_pass} ({len(step7_strict_rows)}/{len(step7_rows)} cases "
          f"strictly checked, tol={TOL_STEP7}; t>0 rows reported only, see docstring)")

    # Step 8 ------------------------------------------------------------
    step8_result = step8_classical_entropic_triple(times, bm, paths, basis_degree)
    report["step8_classical_entropic_triple"] = step8_result
    print(f"[Step 8] classical entropic triple (certainty equivalent / hq-entropic q->1 / "
          f"shortfall) pairwise agreement: {step8_result['all_agree']} "
          f"(max diff={max(step8_result['diff_ce_hq'], step8_result['diff_ce_shortfall'], step8_result['diff_hq_shortfall']):.4f}, "
          f"tol={TOL_STEP8})")

    # Plot + save ---------------------------------------------------------
    plot_info = plot_horizon_longevity(step4_rows)
    report["horizon_longevity_plot"] = plot_info

    report["runtime_seconds"] = time.time() - t_start
    with open(RESULTS_DIR / "exp5_metrics.json", "w") as f:
        json.dump(_to_native(report), f, indent=2)

    print("Saved results/exp5_metrics.json and results/exp5_horizon_longevity.png")
    print(f"Total runtime: {report['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()

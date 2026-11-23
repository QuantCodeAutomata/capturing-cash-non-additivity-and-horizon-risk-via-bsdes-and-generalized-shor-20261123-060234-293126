"""Experiment 4: hq-entropic risk measure on losses as a h-generalized
shortfall risk measure (Theorem 31), and its failure to be a fully-dynamic
certainty equivalent (Remark 32), of

    Di Nunno, G. and Rosazza Gianin, E., "Capturing cash non-additivity and
    horizon risk via BSDEs and generalized shortfall".

Theorem 31 states that the hq-entropic risk measure on losses

    rho_tu^{hq}(X) = ln_q E[exp_q((X+beta)^- + alpha_q + A(0,u)) | F_t] - A(0,t)

coincides with the h-generalized shortfall risk measure rho^{U,f,B} built
from U_u(x) = x (identity), target level B_tu = 0, and

    f_tu(y,m) = C_t*( exp_q(m + A(0,t), q) - exp_q((y+beta)^- + alpha_q + A(0,u), q) )
                                                     for m >= 1/(q-1) - A(0,t),
    f_tu(y,m) = -epsilon                             for m <  1/(q-1) - A(0,t),

since C_t > 0 and exp_q is increasing, so that (acceptance-set equivalence)

    rho^{U,f,B}(X) <= m  <=>  E[f_tu(X,m)] >= 0  <=>  m >= rho_tu^{hq}(X).

Step 3 verifies this acceptance-set equivalence numerically on a grid of
(q, beta, alpha_q, a_const, t, u) and payoffs Y, together with the
"empty acceptance set" claim below the domain-floor threshold
1/(q-1) - A(0,t) (where f_tu is pinned to -epsilon < 0, so E[f_tu] < 0
always and the acceptance set is empty there).

Step 4 numerically illustrates Remark 32: with beta=alpha_q=0 and A≡0, no
constants (C_t, B_tu) make a fixed candidate utility shape Util reproduce
the already-derived f_tu through the certainty-equivalent identity (53)
U_u(f_tu(y,m)) - B_tu = C_t*(Util(y) - Util(-m)); this is numerical evidence
(not a proof) that rho^{hq} is not a certainty equivalent.

Run: python -m exp.exp4
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

from data.generate_synthetic import (
    simulate_brownian_paths,
    terminal_constant,
    terminal_from_bm,
    terminal_normal,
)
from src import config
from src.fully_dynamic import hq_entropic_rho, integral_a
from src.qdeformed import exp_q, q_domain_floor

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

Vec = np.ndarray
AFunc = Callable[[float], float]


def constant_a(c: float) -> AFunc:
    """a(t) = c for every t (constant horizon-risk profile)."""

    def a_func(_s: float, c: float = c) -> float:
        return c

    return a_func


# --------------------------------------------------------------------------
# f_tu of Theorem 31 (U_u = identity, B_tu = 0) and the induced acceptance set
# --------------------------------------------------------------------------
def f_tu(
    y: Vec,
    m: float,
    t: float,
    u: float,
    q: float,
    beta: float,
    alpha_q: float,
    a_func: AFunc,
    times: np.ndarray,
    bm: np.ndarray,
    basis_degree: int = 5,
) -> Vec:
    """f_tu(y,m) of Theorem 31 (two branches, split at the domain floor
    1/(q-1) - A(0,t) of exp_q(m + A(0,t), q)); ``times``/``bm`` are accepted
    for interface consistency with the other path-based helpers of this
    experiment but are not needed here since f_tu is a pointwise (pathwise)
    function of y and the deterministic scalar m -- the expectation E[f_tu]
    is taken afterwards, outside this function, by the caller.
    """
    y = np.asarray(y, dtype=float)
    a0t = integral_a(a_func, 0.0, t)
    a0u = integral_a(a_func, 0.0, u)
    floor = q_domain_floor(q) - a0t
    loss_part = np.maximum(-(y + beta), 0.0) + alpha_q + a0u
    above_branch = config.EXP4_C_T * (exp_q(m + a0t, q) - exp_q(loss_part, q))
    below_branch = np.full_like(y, -config.EXP4_EPSILON)
    return np.where(m >= floor, above_branch, below_branch)


def acceptance_set_generalized_shortfall(
    y_samples: Vec,
    m: float,
    t: float,
    u: float,
    q: float,
    beta: float,
    alpha_q: float,
    a_func: AFunc,
    times: np.ndarray,
    bm: np.ndarray,
    basis_degree: int = 5,
) -> bool:
    """True iff Y in A^{U,f,B;m}, i.e. the Monte Carlo estimate of
    E[f_tu(Y,m)] is >= B_tu = 0."""
    values = f_tu(y_samples, m, t, u, q, beta, alpha_q, a_func, times, bm, basis_degree)
    return bool(np.mean(values) >= 0.0)


# --------------------------------------------------------------------------
# Parameter grid for step 3 (acceptance-set equivalence + empty-set check).
# Restricted to a manageable subset (per the methodology note): a q-sweep at
# the "normalized" beta=alpha_q=0, a=0 defaults, plus a separate sweep of
# (beta, alpha_q, a_const) at a fixed q=0.5, each combined with a small set
# of (t, u) pairs with t < u (not the full T_LIST x U_LIST Cartesian
# product, to keep the total number of combinations on the order of 20-40
# as required, well under 2 minutes of runtime).
# --------------------------------------------------------------------------
TU_PAIRS_SWEEP_A = [(0.0, 0.5), (0.25, 0.75), (0.5, 1.0)]
TU_PAIRS_SWEEP_B = [(0.0, 0.5), (0.5, 1.0)]


def build_param_combos() -> list[dict]:
    combos = []
    for q in config.EXP4_Q_GRID:
        for t, u in TU_PAIRS_SWEEP_A:
            combos.append({"q": q, "beta": 0.0, "alpha_q": 0.0, "a_const": 0.0, "t": t, "u": u})
    for beta in config.EXP4_BETA_GRID:
        for alpha_q in config.EXP4_ALPHA_Q_GRID:
            for a_const in config.EXP4_A_GRID:
                for t, u in TU_PAIRS_SWEEP_B:
                    combos.append({"q": 0.5, "beta": beta, "alpha_q": alpha_q, "a_const": a_const, "t": t, "u": u})
    return combos


def y_candidates_for(combo: dict, times: np.ndarray, paths: np.ndarray, n_paths: int) -> dict[str, Vec]:
    """X = B_u (a genuine F_u-measurable path functional) is always included.
    The other, path-independent candidates are only added at t = 0, where
    the conditional expectation reduces to a plain sample mean and a
    polynomial regression on B_t is not needed to keep it meaningful (see
    module docstring / methodology note)."""
    cands = {"B_u": terminal_from_bm(paths, times, combo["u"])}
    if np.isclose(combo["t"], 0.0):
        cands["const_0.0"] = terminal_constant(n_paths, 0.0)
        cands["const_1.0"] = terminal_constant(n_paths, 1.0)
        cands["normal_0_1"] = terminal_normal(n_paths, 0.0, 1.0, seed=config.RANDOM_SEED + 7)
    return cands


# --------------------------------------------------------------------------
# Step 3: acceptance-set equivalence rho_tu^{hq}(X) <= m  <=>  Y in A^{U,f,B;m}
# --------------------------------------------------------------------------
def step3_acceptance_sets(times: np.ndarray, bm: np.ndarray, paths: np.ndarray, n_paths: int, basis_degree: int):
    combos = build_param_combos()
    match_rows: list[dict] = []
    empty_set_rows: list[dict] = []
    boundary_band = 0.05

    for combo in combos:
        q, beta, alpha_q, a_const = combo["q"], combo["beta"], combo["alpha_q"], combo["a_const"]
        t, u = combo["t"], combo["u"]
        a_func = constant_a(a_const)
        y_cands = y_candidates_for(combo, times, paths, n_paths)

        for y_name, y_vals in y_cands.items():
            rho_hq_arr = hq_entropic_rho(y_vals, times, bm, t, u, q, beta, alpha_q, a_func, basis_degree)
            rho_hq_mean = float(np.mean(rho_hq_arr))

            for m in config.EXP4_M_GRID:
                mean_f = float(np.mean(f_tu(y_vals, m, t, u, q, beta, alpha_q, a_func, times, bm, basis_degree)))
                gs_accept = bool(mean_f >= 0.0)
                hq_fraction = float(np.mean(rho_hq_arr <= m))
                hq_accept = bool(hq_fraction >= 0.5)
                near_boundary = bool(abs(m - rho_hq_mean) < boundary_band)
                match_rows.append({
                    "q": q, "beta": beta, "alpha_q": alpha_q, "a_const": a_const, "t": t, "u": u,
                    "Y": y_name, "m": m, "rho_hq_mean": rho_hq_mean, "hq_fraction_accept": hq_fraction,
                    "hq_accept": hq_accept, "mean_E_f_tu": mean_f, "gs_accept": gs_accept,
                    "agree": bool(hq_accept == gs_accept), "near_boundary": near_boundary,
                })

            # Empty-set check: m < 1/(q-1) - A(0,t) => f_tu pinned to -eps < 0
            # => E[f_tu] < 0 always => acceptance set must be empty (False).
            floor = q_domain_floor(q) - integral_a(a_func, 0.0, t)
            below = [m for m in config.EXP4_M_GRID if m < floor]
            extended = False
            if not below:
                below = [floor - 5.0]
                extended = True
            for m in below:
                gs_accept = acceptance_set_generalized_shortfall(
                    y_vals, m, t, u, q, beta, alpha_q, a_func, times, bm, basis_degree
                )
                empty_set_rows.append({
                    "q": q, "beta": beta, "alpha_q": alpha_q, "a_const": a_const, "t": t, "u": u,
                    "Y": y_name, "m": m, "floor": floor, "extended_m": extended,
                    "gs_accept": gs_accept, "empty_set_holds": bool(not gs_accept),
                })

    return match_rows, empty_set_rows


# --------------------------------------------------------------------------
# Step 4: certainty-equivalent impossibility check (Remark 32).
#
# With beta = alpha_q = 0 and A ≡ 0, f_tu(y,m) reduces (for m >= 1/(q-1)) to
# C_t*(exp_q(m,q) - exp_q(max(-y,0),q)), independent of t and u. Condition
# (53) for rho^{hq} to be a certainty equivalent generated by some concave
# increasing utility Util would require constants (C_t, B_tu) with
# U_u(f_tu(y,m)) - B_tu = C_t*(Util(y) - Util(-m)) for ALL (y,m); here
# U_u = identity, so the left-hand side is simply f_tu(y,m) - B_tu. We fit
# (C_t, B_tu) by linear least squares against three candidate utility shapes
# and report that the fit is poor for all of them (large relative residual),
# which is numerical evidence -- not a formal proof -- against rho^{hq}
# being a certainty equivalent, matching Remark 32.
# --------------------------------------------------------------------------
def step4_certainty_equivalent_impossibility(times: np.ndarray, bm: np.ndarray, basis_degree: int) -> dict:
    q = 0.5  # not prescribed by the paper; matches the EXP2/EXP4 default used elsewhere
    t, u = 0.0, 1.0  # A ≡ 0 makes f_tu independent of t, u; values chosen arbitrarily
    zero_a = constant_a(0.0)

    y_grid = np.linspace(-2.0, 2.0, 15)
    m_grid = np.linspace(-2.0, 2.0, 15)
    y_mesh, m_mesh = np.meshgrid(y_grid, m_grid, indexing="ij")

    lhs = np.empty_like(y_mesh)
    for i, y in enumerate(y_grid):
        for j, m in enumerate(m_grid):
            lhs[i, j] = float(f_tu(np.array([y]), float(m), t, u, q, 0.0, 0.0, zero_a, times, bm, basis_degree)[0])

    y_flat = y_mesh.ravel()
    m_flat = m_mesh.ravel()
    lhs_flat = lhs.ravel()
    lhs_std = float(np.std(lhs_flat))

    candidates: dict[str, Callable[[Vec], Vec]] = {
        "exp_q(.,q)": lambda z: exp_q(z, q),
        "identity": lambda z: z,
        "cube": lambda z: z**3,
    }

    rows = []
    for name, util in candidates.items():
        feature = util(y_flat) - util(-m_flat)
        design = np.column_stack([feature, np.ones_like(feature)])
        coeffs, _, _, _ = np.linalg.lstsq(design, lhs_flat, rcond=None)
        c_t_fit, b_tu_fit = float(coeffs[0]), float(coeffs[1])
        residual = lhs_flat - design @ coeffs
        max_abs_residual = float(np.max(np.abs(residual)))
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum((lhs_flat - np.mean(lhs_flat))**2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
        relative_residual = max_abs_residual / lhs_std if lhs_std > 0.0 else float("inf")
        rows.append({
            "utility_candidate": name,
            "C_t_fit": c_t_fit,
            "B_tu_fit": b_tu_fit,
            "max_abs_residual": max_abs_residual,
            "lhs_std": lhs_std,
            "relative_residual": relative_residual,
            "r_squared": r_squared,
            "no_certainty_equivalent_match": bool(relative_residual > 0.1),
        })

    return {"q": q, "t": t, "u": u, "grid_size": len(y_grid) * len(m_grid), "fits": rows}


# --------------------------------------------------------------------------
# Representative plot: E[f_tu(Y,m)] vs m, with rho_hq and B_tu=0 overlaid.
# --------------------------------------------------------------------------
def plot_acceptance_curve(times: np.ndarray, bm: np.ndarray, paths: np.ndarray, basis_degree: int) -> dict:
    q, beta, alpha_q, a_const, t, u = 0.5, 0.0, 0.0, 0.0, 0.0, 0.5
    a_func = constant_a(a_const)
    y_vals = terminal_from_bm(paths, times, u)

    rho_hq_mean = float(np.mean(hq_entropic_rho(y_vals, times, bm, t, u, q, beta, alpha_q, a_func, basis_degree)))
    m_dense = np.linspace(-3.0, 3.0, 121)
    mean_f_vals = np.array([
        float(np.mean(f_tu(y_vals, m, t, u, q, beta, alpha_q, a_func, times, bm, basis_degree)))
        for m in m_dense
    ])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(m_dense, mean_f_vals, label=r"$E[f_{tu}(Y,m)]$")
    ax.axhline(0.0, color="black", linestyle="-", linewidth=1, label=r"$B_{tu}=0$")
    ax.axvline(rho_hq_mean, color="red", linestyle="--", label=rf"$m=\rho^{{hq}}_{{tu}}(Y)={rho_hq_mean:.3f}$")
    ax.set_xlabel("m")
    ax.set_ylabel(r"$E[f_{tu}(Y,m)]$")
    ax.set_title(f"Experiment 4: acceptance sets of rho^{{hq}} as generalized shortfall\n"
                 f"(q={q}, beta={beta}, alpha_q={alpha_q}, a={a_const}, t={t}, u={u}, Y=B_u)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp4_acceptance_sets.png", dpi=150)
    plt.close(fig)

    return {"q": q, "beta": beta, "alpha_q": alpha_q, "a_const": a_const, "t": t, "u": u, "rho_hq_mean": rho_hq_mean}


def _to_native(obj):
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _parse_args() -> bool:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", action="store_true",
        help="use the paper's full Monte Carlo configuration (EXP4_N_SAMPLES_PAPER) "
        "instead of the reduced run configuration (EXP4_N_SAMPLES_RUN)",
    )
    return parser.parse_args().full


def main() -> None:
    full = _parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    t_start = time.time()

    # n_steps=100: not specified in the paper, chosen for consistency with
    # the time discretization used in the other experiments (e.g. exp_1).
    n_paths = config.EXP4_N_SAMPLES_PAPER if full else config.EXP4_N_SAMPLES_RUN
    n_steps = 100
    times, paths = simulate_brownian_paths(n_paths, n_steps, config.T_HORIZON, config.BM_DIMENSION, seed=config.RANDOM_SEED)
    bm = paths[:, :, 0]
    print(f"Simulated n_paths={n_paths}, n_steps={n_steps}, T={config.T_HORIZON}, d={config.BM_DIMENSION}")

    report: dict = {"n_paths": n_paths, "n_steps": n_steps}

    # Step 3 -----------------------------------------------------------------
    match_rows, empty_set_rows = step3_acceptance_sets(times, bm, paths, n_paths, config.BASIS_DEGREE)
    report["step3_acceptance_set_match"] = match_rows
    report["step3_empty_set_check"] = empty_set_rows

    strict_rows = [r for r in match_rows if not r["near_boundary"]]
    n_boundary = len(match_rows) - len(strict_rows)
    step3_pass = all(r["agree"] for r in strict_rows)
    n_mismatch = sum(1 for r in strict_rows if not r["agree"])
    print(f"[Step 3a] acceptance-set equivalence rho_tu^hq(X)<=m <=> E[f_tu(X,m)]>=0: "
          f"{step3_pass} ({len(strict_rows)} strict rows checked, {n_mismatch} mismatches, "
          f"{n_boundary} boundary rows excluded out of {len(match_rows)} total)")

    n_extended = sum(1 for r in empty_set_rows if r["extended_m"])
    empty_set_pass = all(r["empty_set_holds"] for r in empty_set_rows)
    print(f"[Step 3b] empty acceptance set below domain floor 1/(q-1)-A(0,t): "
          f"{empty_set_pass} ({len(empty_set_rows)} rows checked, {n_extended} used an extended m "
          f"because the EXP4_M_GRID did not reach below the threshold)")

    # Representative acceptance-curve plot -----------------------------------
    plot_info = plot_acceptance_curve(times, bm, paths, config.BASIS_DEGREE)
    report["representative_plot"] = plot_info

    # Step 4 -------------------------------------------------------------------
    step4_result = step4_certainty_equivalent_impossibility(times, bm, config.BASIS_DEGREE)
    report["step4_certainty_equivalent_impossibility"] = step4_result
    step4_pass = all(row["no_certainty_equivalent_match"] for row in step4_result["fits"])
    print(f"[Step 4] no candidate utility reproduces f_tu via a certainty-equivalent fit "
          f"(Remark 32, numerical evidence only): {step4_pass}")
    for row in step4_result["fits"]:
        print(f"    candidate={row['utility_candidate']:>10s}  relative_residual={row['relative_residual']:.3f}  "
              f"R^2={row['r_squared']:.3f}  no_match={row['no_certainty_equivalent_match']}")

    report["runtime_seconds"] = time.time() - t_start
    with open(RESULTS_DIR / "exp4_metrics.json", "w") as f:
        json.dump(_to_native(report), f, indent=2)

    print("Saved results/exp4_metrics.json and results/exp4_acceptance_sets.png")
    print(f"Total runtime: {report['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()

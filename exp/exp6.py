"""Experiment 6: numerical study of h-Value at Risk and fully-dynamic
shortfall with horizon-dependent confidence level (Example 15, Proposition
14 of the paper).

For the normal P&L X = mu*u + sigma*B_u this experiment:

1. Simulates one shared set of Brownian paths used throughout.
2. Implements three *decreasing* confidence-level schedules alpha_u (as a
   function of the horizon u): linear, exponential and step, plus a flat
   (non-decreasing) constant schedule used as a negative control.
3. Computes the h-VaR rho_tu(X) = -quantile_{alpha_u}(X | F_t) both by an
   empirical quantile (t=0, Monte Carlo) and by the closed-form Gaussian
   formula (any t), and cross-checks the two.
4. Verifies h-longevity (rho_tv >= rho_tu for t <= u <= v) for the three
   decreasing schedules (Proposition 14): since alpha_u is decreasing in u,
   alpha_v <= alpha_u for v >= u, hence (as the standard-normal quantile
   function is increasing) -Phi^{-1}(alpha_v) >= -Phi^{-1}(alpha_u), so the
   conditional VaR is non-decreasing in the horizon.
5. Repeats the check with the constant schedule (negative control): the
   strict h-longevity inequality is generally not expected to hold in the
   "isolated" sense (same X, only alpha changing with the horizon of
   evaluation); the restriction property is recovered instead.
6. Verifies that the generalized-shortfall representation of Example 15,
   with utility U^alpha and the classical aggregator f(x,m) = x + m and
   target B = 0, reproduces the h-VaR.
7. Builds the cash-subadditive VaR variant obtained with the linear
   aggregator f(x,m) = beta*x + m (Example 19 i)) and verifies cash
   subadditivity rho(X + Delta) >= rho(X) - Delta.

Run: python -m exp.exp6
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
from scipy.stats import norm

from data.generate_synthetic import simulate_brownian_paths
from src import config
from src.fully_dynamic import h_var_empirical, h_var_normal
from src.shortfall import aggregator_classical, aggregator_linear, generalized_shortfall, utility_var

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# n_steps = 100 is not prescribed by the paper for this experiment; it is
# inferred from the "n = 100 time steps" discretization used elsewhere in
# the paper's numerical setup (exp_1, exp_5).
N_STEPS = 100

# Representative (mu, sigma) subset used to keep the full grid manageable:
# the first value of each config grid plus one additional value, i.e.
# mu in {EXP6_MU_GRID[0], EXP6_MU_GRID[1]} = {0.0, 0.01} and
# sigma in {EXP6_SIGMA_GRID[0], EXP6_SIGMA_GRID[1]} = {0.1, 0.2}.
MU_SUBSET = config.EXP6_MU_GRID[:2]
SIGMA_SUBSET = config.EXP6_SIGMA_GRID[:2]


def grid_index(times: np.ndarray, t: float, atol: float = 1e-9) -> int:
    """Exact index of time ``t`` on the simulation grid ``times``; raises
    ValueError if ``t`` is not an exact grid point (within ``atol``)."""
    idx = int(np.searchsorted(times, t))
    idx = min(max(idx, 0), len(times) - 1)
    if abs(times[idx] - t) > atol and idx > 0 and abs(times[idx - 1] - t) <= atol:
        idx -= 1
    if abs(times[idx] - t) > atol:
        raise ValueError(f"time {t} is not an exact grid point of the simulation (within {atol})")
    return idx


# --------------------------------------------------------------------------
# Confidence-level schedules alpha_u, all sharing the signature
# (u, alpha0, kappa) so that they can be driven uniformly (kappa is unused
# by the step and constant schedules, kept for a uniform call signature).
# --------------------------------------------------------------------------
def alpha_linear(u: float, alpha0: float, kappa: float) -> float:
    return max(alpha0 - kappa * u, 1e-6)


def alpha_exponential(u: float, alpha0: float, kappa: float) -> float:
    return alpha0 * np.exp(-kappa * u)


def alpha_step(u: float, alpha0: float, kappa: float, u0: float = 0.5) -> float:
    return alpha0 if u <= u0 else alpha0 * 0.5


def alpha_constant(u: float, alpha0: float, kappa: float) -> float:
    return alpha0


SCHEDULES: dict[str, Callable[[float, float, float], float]] = {
    "linear": alpha_linear,
    "exponential": alpha_exponential,
    "step": alpha_step,
}
DECREASING_SCHEDULE_NAMES = list(SCHEDULES.keys())
ALL_SCHEDULES: dict[str, Callable[[float, float, float], float]] = {**SCHEDULES, "constant": alpha_constant}

# Representative (alpha_0, kappa) used whenever a single schedule parameter
# pair is needed (h-longevity checks, shortfall checks).
ALPHA0_REP = config.EXP6_ALPHA0_GRID[0]
KAPPA_REP = config.EXP6_KAPPA_GRID[0]


# --------------------------------------------------------------------------
# Step 3: h-VaR at t = 0 (empirical vs. closed form) and at t > 0 (closed
# form, conditional on F_t).
# --------------------------------------------------------------------------
def t0_empirical_vs_closed_form(bm: np.ndarray, times: np.ndarray) -> list[dict]:
    rows = []
    for mu in MU_SUBSET:
        for sigma in SIGMA_SUBSET:
            for sched_name, sched in ALL_SCHEDULES.items():
                for u in config.EXP6_U_LIST:
                    iu = grid_index(times, u)
                    alpha_u = sched(u, ALPHA0_REP, KAPPA_REP)
                    x_samples = mu * u + sigma * bm[:, iu]
                    empirical = h_var_empirical(x_samples, alpha_u)
                    closed_form = float(h_var_normal(mu, sigma, 0.0, u, alpha_u, b_t=np.zeros(1))[0])
                    agree = bool(abs(empirical - closed_form) < 5e-2)
                    rows.append({
                        "mu": mu, "sigma": sigma, "schedule": sched_name, "u": u,
                        "alpha_u": alpha_u, "h_var_empirical": empirical,
                        "h_var_closed_form": closed_form, "agree_within_5e-2": agree,
                    })
    return rows


def t_positive_conditional(bm: np.ndarray, times: np.ndarray) -> list[dict]:
    rows = []
    for mu in MU_SUBSET:
        for sigma in SIGMA_SUBSET:
            for sched_name, sched in ALL_SCHEDULES.items():
                for u in config.EXP6_U_LIST:
                    for t in config.EXP6_T_LIST:
                        if t >= u:
                            continue
                        it = grid_index(times, t)
                        alpha_u = sched(u, ALPHA0_REP, KAPPA_REP)
                        rho = h_var_normal(mu, sigma, t, u, alpha_u, b_t=bm[:, it])
                        rows.append({
                            "mu": mu, "sigma": sigma, "schedule": sched_name,
                            "t": t, "u": u, "alpha_u": alpha_u,
                            "rho_tu_mean": float(np.mean(rho)), "rho_tu_std": float(np.std(rho)),
                        })
    return rows


# --------------------------------------------------------------------------
# Sensitivity of the closed-form h-VaR to the schedule parameters (alpha_0,
# kappa), at t = 0, for the representative (mu, sigma) = (MU_SUBSET[0],
# SIGMA_SUBSET[0]) ("Study the effect of different decreasing schedules").
# --------------------------------------------------------------------------
def schedule_sensitivity() -> list[dict]:
    mu, sigma = MU_SUBSET[0], SIGMA_SUBSET[0]
    rows = []
    for sched_name, sched in ALL_SCHEDULES.items():
        kappas = config.EXP6_KAPPA_GRID if sched_name in ("linear", "exponential") else [None]
        for alpha0 in config.EXP6_ALPHA0_GRID:
            for kappa in kappas:
                kappa_eff = kappa if kappa is not None else 0.0
                for u in config.EXP6_U_LIST:
                    alpha_u = sched(u, alpha0, kappa_eff)
                    rho = float(h_var_normal(mu, sigma, 0.0, u, alpha_u, b_t=np.zeros(1))[0])
                    rows.append({
                        "schedule": sched_name, "alpha_0": alpha0,
                        "kappa": kappa, "u": u, "alpha_u": alpha_u, "h_var": rho,
                    })
    return rows


# --------------------------------------------------------------------------
# Step 4/5: h-longevity for decreasing schedules, and the negative control
# with the constant schedule (Proposition 14, Example 15).
# --------------------------------------------------------------------------
def triples_t_u_v() -> list[tuple[float, float, float]]:
    triples = []
    for t in config.EXP6_T_LIST:
        for u in config.EXP6_U_LIST:
            if u < t:
                continue
            for v in config.EXP6_U_LIST:
                if v < u:
                    continue
                triples.append((t, u, v))
    return triples


def h_longevity_check(bm: np.ndarray, times: np.ndarray, sched_name: str, sched, mu: float, sigma: float) -> list[dict]:
    """h-longevity compares rho_tu(X) and rho_tv(X) for the SAME F_u-measurable
    X = mu*u + sigma*B_u (Example 15, Proposition 14): only the confidence
    level used to evaluate X changes between the two reporting horizons u and
    v, not X's own law. ``h_var_normal`` is therefore called with the
    horizon fixed at ``u`` in both cases (as in ``isolated_alpha_effect``
    above); only ``alpha_u`` vs. ``alpha_v`` differs."""
    rows = []
    for t, u, v in triples_t_u_v():
        it = grid_index(times, t)
        b_t = bm[:, it]
        alpha_u = sched(u, ALPHA0_REP, KAPPA_REP)
        alpha_v = sched(v, ALPHA0_REP, KAPPA_REP)
        rho_tu = float(np.mean(h_var_normal(mu, sigma, t, u, alpha_u, b_t)))
        rho_tv = float(np.mean(h_var_normal(mu, sigma, t, u, alpha_v, b_t)))
        rows.append({
            "schedule": sched_name, "t": t, "u": u, "v": v,
            "alpha_u": alpha_u, "alpha_v": alpha_v,
            "rho_tu": rho_tu, "rho_tv": rho_tv,
            "gap_rho_tv_minus_rho_tu": rho_tv - rho_tu,
            "h_longevity_holds": bool(rho_tv >= rho_tu - 1e-2),
        })
    return rows


def isolated_alpha_effect(sched_name: str, sched) -> list[dict]:
    """Fix X = mu*u_fixed + sigma*B_{u_fixed} (so X's law never changes) and
    only vary the confidence level used to evaluate it at two different
    reporting horizons u <= v, using the SAME conditional mean/std: this
    isolates the effect of the alpha_u schedule alone from the u-dependence
    of X's own law."""
    mu, sigma = config.EXP6_MU_GRID[0], config.EXP6_SIGMA_GRID[0]
    u_fixed = config.EXP6_U_LIST[0]
    t = 0.0
    cond_mean = mu * u_fixed  # b_t = 0 since t = 0
    cond_std = sigma * np.sqrt(u_fixed - t)
    rows = []
    for u in config.EXP6_U_LIST:
        for v in config.EXP6_U_LIST:
            if v < u:
                continue
            alpha_u = sched(u, ALPHA0_REP, KAPPA_REP)
            alpha_v = sched(v, ALPHA0_REP, KAPPA_REP)
            rho_1 = -cond_mean - cond_std * norm.ppf(alpha_u)
            rho_2 = -cond_mean - cond_std * norm.ppf(alpha_v)
            rows.append({
                "schedule": sched_name, "u_fixed": u_fixed, "u_report": u, "v_report": v,
                "alpha_u": alpha_u, "alpha_v": alpha_v,
                "rho_1_isolated": rho_1, "rho_2_isolated": rho_2,
                "gap_isolated": rho_2 - rho_1,
            })
    return rows


# --------------------------------------------------------------------------
# Step 6: shortfall representation of the h-VaR (Example 15): U^alpha and
# the classical aggregator f(x,m) = x + m reproduce rho_tu = h_var_empirical.
# --------------------------------------------------------------------------
def shortfall_reproduces_var(bm: np.ndarray, times: np.ndarray) -> list[dict]:
    mu, sigma = MU_SUBSET[0], SIGMA_SUBSET[0]
    rows = []
    for alpha_u in config.EXP6_ALPHA0_GRID:
        for u in config.EXP6_U_LIST:
            iu = grid_index(times, u)
            x_samples = mu * u + sigma * bm[:, iu]
            shortfall_m = generalized_shortfall(
                x_samples, lambda x: utility_var(x, alpha_u), aggregator_classical, target_b=0.0
            )
            empirical = h_var_empirical(x_samples, alpha_u)
            rows.append({
                "mu": mu, "sigma": sigma, "alpha_u": alpha_u, "u": u,
                "shortfall_m": shortfall_m, "h_var_empirical": empirical,
                "agree_within_5e-2": bool(abs(shortfall_m - empirical) < 5e-2),
            })
    return rows


# --------------------------------------------------------------------------
# Step 7: cash-subadditive VaR variant via the linear aggregator f(x,m) =
# beta*x + m (Example 19 i)).
# --------------------------------------------------------------------------
def cash_subadditive_var(bm: np.ndarray, times: np.ndarray) -> list[dict]:
    mu, sigma = MU_SUBSET[0], SIGMA_SUBSET[0]
    alpha_u = config.EXP6_ALPHA0_GRID[0]
    u = config.EXP6_U_LIST[-1]
    iu = grid_index(times, u)
    x_samples = mu * u + sigma * bm[:, iu]
    utility = lambda x: utility_var(x, alpha_u)
    rows = []
    for beta in config.EXP6_BETA_CASH_SUBADD:
        aggregator = lambda x, m, b=beta: aggregator_linear(x, m, b)
        rho_x = generalized_shortfall(x_samples, utility, aggregator, target_b=0.0)
        for delta in config.EXP6_DELTA_SUBADDITIVITY:
            rho_x_delta = generalized_shortfall(x_samples + delta, utility, aggregator, target_b=0.0)
            bound = rho_x - delta
            gap = bound - rho_x_delta
            rows.append({
                "beta": beta, "delta": delta, "alpha_u": alpha_u, "u": u,
                "rho_X": rho_x, "rho_X_plus_delta": rho_x_delta,
                "rho_X_minus_delta_bound": bound, "gap_bound_minus_shifted": gap,
                "cash_subadditivity_holds": bool(rho_x_delta >= bound - 5e-2),
            })
    return rows


def _parse_args() -> bool:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", action="store_true",
        help="use the paper's full Monte Carlo configuration (EXP6_N_PATHS_PAPER) "
        "instead of the reduced run configuration (EXP6_N_PATHS_RUN)",
    )
    return parser.parse_args().full


def main() -> None:
    full = _parse_args()
    start = time.time()
    RESULTS_DIR.mkdir(exist_ok=True)

    n_paths = config.EXP6_N_PATHS_PAPER if full else config.EXP6_N_PATHS_RUN
    n_steps = N_STEPS
    times, paths = simulate_brownian_paths(n_paths, n_steps, config.T_HORIZON, config.BM_DIMENSION, seed=config.RANDOM_SEED)
    bm = paths[:, :, 0]
    del paths
    print(f"Simulated {n_paths} paths, {n_steps} steps, T={config.T_HORIZON}, d={config.BM_DIMENSION}")

    report: dict = {
        "n_paths": n_paths, "n_steps": n_steps,
        "mu_subset": MU_SUBSET, "sigma_subset": SIGMA_SUBSET,
        "alpha0_representative": ALPHA0_REP, "kappa_representative": KAPPA_REP,
    }

    # Step 3: h-VaR at t=0 (empirical vs closed form) and t>0 (closed form).
    t0_rows = t0_empirical_vs_closed_form(bm, times)
    report["t0_empirical_vs_closed_form"] = t0_rows
    step3_ok = all(r["agree_within_5e-2"] for r in t0_rows)
    for r in t0_rows:
        assert r["agree_within_5e-2"], f"empirical vs closed-form h-VaR mismatch: {r}"
    report["schedule_sensitivity"] = schedule_sensitivity()
    tpos_rows = t_positive_conditional(bm, times)
    report["t_positive_conditional"] = tpos_rows
    print(f"[Step 3] h-VaR empirical vs closed form agree (t=0, tol=5e-2): {step3_ok}")

    # Step 4: h-longevity for the 3 decreasing schedules.
    mu0, sigma0 = config.EXP6_MU_GRID[0], config.EXP6_SIGMA_GRID[0]
    longevity_rows = []
    for sched_name in DECREASING_SCHEDULE_NAMES:
        longevity_rows.extend(h_longevity_check(bm, times, sched_name, SCHEDULES[sched_name], mu0, sigma0))
    report["h_longevity_decreasing_schedules"] = longevity_rows
    step4_ok = all(r["h_longevity_holds"] for r in longevity_rows)
    print(f"[Step 4] H-longevity holds for all tested (t,u,v) triples, decreasing schedules: {step4_ok}")

    # Step 5: negative control with the constant schedule.
    constant_longevity_rows = h_longevity_check(bm, times, "constant", alpha_constant, mu0, sigma0)
    report["h_longevity_constant_schedule"] = constant_longevity_rows
    mean_gap_constant = float(np.mean([r["gap_rho_tv_minus_rho_tu"] for r in constant_longevity_rows]))
    mean_gap_decreasing = float(np.mean([r["gap_rho_tv_minus_rho_tu"] for r in longevity_rows]))

    isolated_rows: list[dict] = []
    for sched_name, sched in ALL_SCHEDULES.items():
        isolated_rows.extend(isolated_alpha_effect(sched_name, sched))
    report["isolated_alpha_effect"] = isolated_rows
    isolated_constant = [r for r in isolated_rows if r["schedule"] == "constant"]
    isolated_constant_exact_equality = bool(
        max(abs(r["gap_isolated"]) for r in isolated_constant) < 1e-9
    )
    print(f"[Step 5] Negative control (constant schedule): mean gap rho_tv-rho_tu = {mean_gap_constant:.4e} "
          f"(vs. {mean_gap_decreasing:.4e} for decreasing schedules); "
          f"isolated same-X check gives exact equality for the constant schedule: {isolated_constant_exact_equality}")

    # Step 6: shortfall representation of the h-VaR.
    shortfall_rows = shortfall_reproduces_var(bm, times)
    report["shortfall_reproduces_var"] = shortfall_rows
    step6_ok = all(r["agree_within_5e-2"] for r in shortfall_rows)
    for r in shortfall_rows:
        assert r["agree_within_5e-2"], f"shortfall representation mismatch: {r}"
    print(f"[Step 6] Generalized-shortfall representation reproduces h-VaR (tol=5e-2): {step6_ok}")

    # Step 7: cash-subadditive VaR variant.
    subadd_rows = cash_subadditive_var(bm, times)
    report["cash_subadditive_var"] = subadd_rows
    step7_ok = all(r["cash_subadditivity_holds"] for r in subadd_rows)
    beta1_gaps = [r["gap_bound_minus_shifted"] for r in subadd_rows if np.isclose(r["beta"], 1.0) and r["delta"] > 0]
    beta_lt1_gaps = [r["gap_bound_minus_shifted"] for r in subadd_rows if r["beta"] < 1.0 and r["delta"] > 0]
    beta1_mean_gap = float(np.mean(beta1_gaps)) if beta1_gaps else 0.0
    beta_lt1_mean_gap = float(np.mean(beta_lt1_gaps)) if beta_lt1_gaps else 0.0
    more_slack_for_beta_lt1 = bool(beta_lt1_mean_gap < beta1_mean_gap - 1e-6)
    report["cash_subadditive_summary"] = {
        "beta1_mean_gap": beta1_mean_gap, "beta_lt1_mean_gap": beta_lt1_mean_gap,
        "more_slack_for_beta_lt1": more_slack_for_beta_lt1,
    }
    print(f"[Step 7] Cash subadditivity holds for all (beta, Delta): {step7_ok}; "
          f"beta=1 mean gap={beta1_mean_gap:.4e} (should be close to 0) vs. "
          f"beta<1 mean gap={beta_lt1_mean_gap:.4e} (should be more negative): {more_slack_for_beta_lt1}")

    # JSON-serializable: cast everything explicitly as plain Python types.
    with open(RESULTS_DIR / "exp6_metrics.json", "w") as f:
        json.dump(report, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)

    # Plot: h-VaR as a function of u, 3 decreasing schedules + constant.
    mu_plot, sigma_plot = config.EXP6_MU_GRID[0], config.EXP6_SIGMA_GRID[0]
    fig, ax = plt.subplots(figsize=(7, 5))
    for sched_name, sched in ALL_SCHEDULES.items():
        values = [
            float(h_var_normal(mu_plot, sigma_plot, 0.0, u, sched(u, ALPHA0_REP, KAPPA_REP), b_t=np.zeros(1))[0])
            for u in config.EXP6_U_LIST
        ]
        ax.plot(config.EXP6_U_LIST, values, marker="o", label=sched_name)
    ax.set_xlabel("horizon u")
    ax.set_ylabel(r"h-VaR $\rho_{0u}(X)$")
    ax.set_title(
        f"Experiment 6: h-VaR by horizon (mu={mu_plot}, sigma={sigma_plot}, "
        f"alpha_0={ALPHA0_REP}, kappa={KAPPA_REP})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp6_hvar_by_horizon.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    elapsed = time.time() - start
    print(f"Saved results/exp6_metrics.json and results/exp6_hvar_by_horizon.png (elapsed: {elapsed:.1f}s)")


if __name__ == "__main__":
    main()


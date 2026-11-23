"""Experiment 3: generalized shortfall risk measures with various aggregator
functions, quasi-convexity, and cash subadditivity.

Reproduces the numerical study around Definition 16, Proposition 17,
Example 19, and Proposition 20 of

    Di Nunno, G. and Rosazza Gianin, E., "Capturing cash non-additivity and
    horizon risk via BSDEs and generalized shortfall",

i.e. the generalized shortfall risk measure

    rho^{U,f,B}(X) = inf{m in R : E[U(f(X,m))] >= B}                 (31)

for: (2) the classical aggregator f(x,m)=x+m with exponential utility, which
reduces to the entropic risk measure in closed form; (3) the linear
cash-subadditive aggregator f(x,m)=beta*x+m, beta in (0,1] (Example 19 i));
(4) the exponential cash-subadditive aggregator f(y,k)=1-exp(-nu*y-k), nu in
(0,1) (Example 19 ii), Proposition 17 item 2c); (5) quasi-convexity of
rho^{U,f,B} in X; and (6) the convex-dual representation (Proposition 20) for
the tractable entropic/classical-shortfall instance. Step (7) illustrates the
acceptance-set view of cash subadditivity (nestedness of A^{m} in m).

Run: python -m exp.exp3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.shortfall import (
    acceptance_indicator,
    aggregator_classical,
    aggregator_exponential,
    aggregator_linear,
    entropic_dual_sup,
    entropic_shortfall_closed_form,
    generalized_shortfall,
    utility_exponential,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

Vec = np.ndarray


# --------------------------------------------------------------------------
# 1. Sample draws (independent seeded RNGs, config.RANDOM_SEED + 0..3)
# --------------------------------------------------------------------------
def draw_samples(n: int = config.EXP3_N_SAMPLES_RUN) -> dict[str, Vec]:
    rng_normal = np.random.default_rng(config.RANDOM_SEED + 0)
    rng_normal2 = np.random.default_rng(config.RANDOM_SEED + 1)
    rng_uniform = np.random.default_rng(config.RANDOM_SEED + 2)
    rng_exp = np.random.default_rng(config.RANDOM_SEED + 3)
    return {
        "X_normal": rng_normal.normal(0.0, 1.0, size=n),
        "X_normal2": rng_normal2.normal(0.3, 1.5, size=n),
        "X_uniform": rng_uniform.uniform(-2.0, 2.0, size=n),
        # Centered Exponential(mean=1): Exponential(1) - 1, so it has mean 0
        # like the other P&L-like variables above and can take negative
        # values (choice documented per the experiment's methodology note).
        "X_exponential": rng_exp.exponential(1.0, size=n) - 1.0,
    }


# --------------------------------------------------------------------------
# 2. Classical shortfall reproduces the entropic risk measure (Section 4,
# Foellmer & Schied Example 4.115)
# --------------------------------------------------------------------------
def step2_classical_entropic(samples: dict[str, Vec]) -> list[dict]:
    x = samples["X_normal"]
    gamma = config.EXP3_ENTROPIC_GAMMA
    utility = lambda z: utility_exponential(z, gamma)
    rows = []
    for b in config.EXP3_B_TARGET_GRID:
        if not (0.0 <= b < 1.0):
            continue
        numeric = generalized_shortfall(x, utility, aggregator_classical, b)
        closed_form = entropic_shortfall_closed_form(x, gamma, b)
        abs_diff = abs(numeric - closed_form)
        rows.append({
            "B": b,
            "numeric_root_finding": numeric,
            "closed_form": closed_form,
            "abs_diff": abs_diff,
            "matches": bool(abs_diff < 1e-3),
        })
    return rows


# --------------------------------------------------------------------------
# 3. Linear aggregator f(x,m)=beta*x+m: cash subadditivity (Example 19 i),
# Proposition 17)
# --------------------------------------------------------------------------
def step3_linear_subadditivity(samples: dict[str, Vec]) -> list[dict]:
    x = samples["X_normal"]
    utility = lambda z: utility_exponential(z, 1.0)
    target_b = 0.9
    tol = 1e-2
    eps = 1e-3
    rows = []
    for beta in config.EXP3_BETA_GRID:
        agg: Callable[[Vec, float], Vec] = lambda z, m, beta=beta: aggregator_linear(z, m, beta)
        rho_x = generalized_shortfall(x, utility, agg, target_b)
        for m in config.EXP3_M_SUBADDITIVITY:
            rho_shift = generalized_shortfall(x + m, utility, agg, target_b)
            bound = rho_x - m
            subadditivity_holds = bool(rho_shift >= bound - tol)
            if beta < 1.0:
                # Strict cash subadditivity for m > 0; trivial equality at m=0.
                beta_specific_check = bool(m == 0.0 or rho_shift > bound + eps)
            else:
                # beta = 1.0: classical, (numerically) cash additive, i.e. equality.
                beta_specific_check = bool(abs(rho_shift - bound) <= tol)
            rows.append({
                "beta": beta,
                "m": m,
                "rho_X": rho_x,
                "rho_X_plus_m": rho_shift,
                "bound_rho_X_minus_m": bound,
                "cash_subadditivity_holds": subadditivity_holds,
                "beta_specific_check_holds": beta_specific_check,
            })
    return rows


# --------------------------------------------------------------------------
# 4. Exponential aggregator f(y,k)=1-exp(-nu*y-k): monotone condition
# f(y,k) <= f(y-m,k+m) (Proposition 17 item 2c) and cash subadditivity
# (Example 19 ii))
# --------------------------------------------------------------------------
def step4_exponential_aggregator(samples: dict[str, Vec]) -> tuple[list[dict], list[dict]]:
    x = samples["X_normal"]
    utility = lambda z: utility_exponential(z, 1.0)
    target_b = 0.5
    y_grid = np.linspace(-2.0, 2.0, 20)
    k_grid = np.linspace(-1.0, 1.0, 5)
    m_grid = [0.1, 0.5, 1.0]

    monotone_rows = []
    subadditivity_rows = []
    for nu in config.EXP3_NU_GRID:
        max_violation = -np.inf
        for y in y_grid:
            for k in k_grid:
                for m in m_grid:
                    f_yk = float(aggregator_exponential(np.array([y]), k, nu)[0])
                    f_shifted = float(aggregator_exponential(np.array([y - m]), k + m, nu)[0])
                    violation = f_yk - f_shifted  # should be <= 0
                    max_violation = max(max_violation, violation)
        monotone_holds = bool(max_violation <= 1e-9)
        monotone_rows.append({
            "nu": nu,
            "max_violation_f_yk_minus_f_shifted": float(max_violation),
            "monotone_condition_holds": monotone_holds,
        })

        agg: Callable[[Vec, float], Vec] = lambda z, m, nu=nu: aggregator_exponential(z, m, nu)
        rho_x = generalized_shortfall(x, utility, agg, target_b)
        for m in config.EXP3_M_SUBADDITIVITY:
            rho_shift = generalized_shortfall(x + m, utility, agg, target_b)
            bound = rho_x - m
            subadditivity_rows.append({
                "nu": nu,
                "m": m,
                "rho_X": rho_x,
                "rho_X_plus_m": rho_shift,
                "bound_rho_X_minus_m": bound,
                "cash_subadditivity_holds": bool(rho_shift >= bound - 1e-2),
            })
    return monotone_rows, subadditivity_rows


# --------------------------------------------------------------------------
# 5. Quasi-convexity of rho^{U,f,B} in X (Proposition 17)
# --------------------------------------------------------------------------
def step5_quasi_convexity(samples: dict[str, Vec]) -> list[dict]:
    x = samples["X_normal"]
    y = samples["X_uniform"]
    utility = lambda z: utility_exponential(z, 1.0)
    # For the classical and linear aggregators f(x,m) ranges over all of R as
    # m varies, so E[U(f(X,m))] sweeps the whole range (-inf,1) of U and
    # B=0.9 is attainable. The exponential aggregator f(x,m)=1-exp(-nu*x-m)
    # satisfies f(x,m) < 1 for every finite (x,m) (Example 19 ii)), so
    # U(f(x,m)) = 1-exp(-f(x,m)) < 1-exp(-1) ~= 0.632 pointwise and hence
    # E[U(f(X,m))] can never reach B=0.9 for any finite m: B=0.9 is
    # mathematically infeasible for this aggregator/utility pair, so B=0.5
    # (the same feasible level used in step 4) is used for it instead.
    aggregators: dict[str, tuple[Callable[[Vec, float], Vec], float]] = {
        "classical": (lambda z, m: aggregator_classical(z, m), 0.9),
        "linear_beta0.7": (lambda z, m: aggregator_linear(z, m, 0.7), 0.9),
        "exponential_nu0.5": (lambda z, m: aggregator_exponential(z, m, 0.5), 0.5),
    }
    rows = []
    for name, (agg, target_b) in aggregators.items():
        rho_x = generalized_shortfall(x, utility, agg, target_b)
        rho_y = generalized_shortfall(y, utility, agg, target_b)
        bound = max(rho_x, rho_y)
        for lam in config.EXP3_LAMBDAS:
            # Z = lambda*X + (1-lambda)*Y: elementwise combination of the two
            # equal-length Monte Carlo sample arrays, which is a valid Monte
            # Carlo sample of lambda*X+(1-lambda)*Y only under the (standard)
            # assumption that X and Y are defined on a common probability
            # space indexed by the same sample draw.
            z = lam * x + (1.0 - lam) * y
            rho_z = generalized_shortfall(z, utility, agg, target_b)
            rows.append({
                "aggregator": name,
                "B": target_b,
                "lambda": lam,
                "rho_X": rho_x,
                "rho_Y": rho_y,
                "rho_Z": rho_z,
                "max_rho_X_rho_Y": bound,
                "quasi_convexity_holds": bool(rho_z <= bound + 1e-2),
            })
    return rows


# --------------------------------------------------------------------------
# 6. Dual representation (Proposition 20) for the tractable entropic
# instance
# --------------------------------------------------------------------------
def step6_dual_representation(samples: dict[str, Vec]) -> dict:
    x = samples["X_normal"]
    gamma = config.EXP3_ENTROPIC_GAMMA
    target_b = 0.9
    utility = lambda z: utility_exponential(z, gamma)
    theta_grid = np.linspace(
        -config.EXP3_THETA_GRID_HALF_WIDTH, config.EXP3_THETA_GRID_HALF_WIDTH, config.EXP3_THETA_GRID_POINTS
    )
    dual_sup, values = entropic_dual_sup(x, gamma, target_b, theta_grid)
    primal = generalized_shortfall(x, utility, aggregator_classical, target_b)
    abs_diff = abs(dual_sup - primal)
    matches = bool(abs_diff < 5e-2)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(theta_grid, values, label="dual objective over the theta grid")
    ax.axhline(primal, color="red", linestyle="--", label=f"primal rho(X) = {primal:.4f}")
    ax.axhline(dual_sup, color="green", linestyle=":", label=f"grid dual sup = {dual_sup:.4f}")
    ax.set_xlabel(r"tilting parameter $\theta$")
    ax.set_ylabel("dual objective value")
    ax.set_title("Experiment 3: dual representation of the entropic shortfall")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3_dual_representation.png", dpi=150)
    plt.close(fig)

    return {
        "gamma": gamma,
        "B": target_b,
        "dual_sup": dual_sup,
        "primal": primal,
        "abs_diff": abs_diff,
        "matches": matches,
        "theta_grid": theta_grid.tolist(),
        "values": values.tolist(),
    }


# --------------------------------------------------------------------------
# 7. Acceptance-set view of cash subadditivity (nestedness of A^{m} in m,
# Proposition 22-style check, simplified)
# --------------------------------------------------------------------------
def step7_acceptance_sets(samples: dict[str, Vec]) -> list[dict]:
    utility = lambda z: utility_exponential(z, 1.0)
    target_b = 0.9
    agg: Callable[[Vec, float], Vec] = lambda z, m: aggregator_linear(z, m, 0.7)

    candidates: dict[str, Vec] = {
        "X_normal2": samples["X_normal2"],
        "X_uniform": samples["X_uniform"],
        "X_exponential": samples["X_exponential"],
    }
    for c in np.linspace(-2.0, 2.0, 9):
        candidates[f"X_normal_shift_{c:.2f}"] = samples["X_normal"] + c

    rows = []
    for m in (0.0, 0.5):
        for name, y in candidates.items():
            in_m = acceptance_indicator(y, utility, agg, target_b, m)
            in_m_delta = acceptance_indicator(y, utility, agg, target_b, m + 0.3)
            rows.append({
                "m": m,
                "candidate": name,
                "in_A_m": bool(in_m),
                "in_A_m_plus_0.3": bool(in_m_delta),
                "nested_consistent": bool((not in_m) or in_m_delta),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 3: generalized shortfall risk measures (aggregator "
        "functions, quasi-convexity, cash subadditivity) of Di Nunno, G. and "
        "Rosazza Gianin, E., 'Capturing cash non-additivity and horizon risk via "
        "BSDEs and generalized shortfall'."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use the paper-scale Monte Carlo sample size "
        f"(N={config.EXP3_N_SAMPLES_PAPER}) instead of the reduced runtime budget "
        f"(N={config.EXP3_N_SAMPLES_RUN}).",
    )
    args = parser.parse_args()
    n_samples = config.EXP3_N_SAMPLES_PAPER if args.full else config.EXP3_N_SAMPLES_RUN

    RESULTS_DIR.mkdir(exist_ok=True)
    t_start = time.time()

    samples = draw_samples(n_samples)
    print(f"Drew {n_samples} samples for each of {len(samples)} random variables:")
    sample_moments = {}
    for name, arr in samples.items():
        mean, std = float(np.mean(arr)), float(np.std(arr))
        sample_moments[name] = {"n": len(arr), "mean": mean, "std": std}
        print(f"  {name}: n={len(arr)}, mean={mean:.4f}, std={std:.4f}")

    report: dict = {
        "n_samples": n_samples,
        "full_scale": args.full,
        "sample_moments": sample_moments,
    }

    step2_rows = step2_classical_entropic(samples)
    report["step2_classical_entropic"] = step2_rows
    for row in step2_rows:
        assert row["matches"], f"classical shortfall vs closed form mismatch at B={row['B']}: {row}"
    step2_pass = all(r["matches"] for r in step2_rows)
    print(f"[Step 2] classical shortfall == entropic closed form for all B: {step2_pass}")

    step3_rows = step3_linear_subadditivity(samples)
    report["step3_linear_aggregator"] = step3_rows
    step3_pass = all(r["cash_subadditivity_holds"] and r["beta_specific_check_holds"] for r in step3_rows)
    print(f"[Step 3] linear aggregator: cash subadditivity + beta<1 strict / beta=1 equality: {step3_pass}")

    monotone_rows, subadd_rows = step4_exponential_aggregator(samples)
    report["step4_exponential_monotone_condition"] = monotone_rows
    report["step4_exponential_subadditivity"] = subadd_rows
    step4_pass = all(r["monotone_condition_holds"] for r in monotone_rows) and all(
        r["cash_subadditivity_holds"] for r in subadd_rows
    )
    print(f"[Step 4] exponential aggregator: monotone condition + cash subadditivity for all nu: {step4_pass}")

    step5_rows = step5_quasi_convexity(samples)
    report["step5_quasi_convexity"] = step5_rows
    step5_pass = all(r["quasi_convexity_holds"] for r in step5_rows)
    print(f"[Step 5] quasi-convexity (3 aggregators x {len(config.EXP3_LAMBDAS)} lambdas): {step5_pass}")

    step6_result = step6_dual_representation(samples)
    report["step6_dual_representation"] = step6_result
    print(f"[Step 6] dual representation matches primal (tol 5e-2): {step6_result['matches']}")

    step7_rows = step7_acceptance_sets(samples)
    report["step7_acceptance_sets"] = step7_rows
    step7_consistent = all(r["nested_consistent"] for r in step7_rows)
    print(f"[Step 7] acceptance sets nested consistently (A^m subset A^{{m+delta}}): {step7_consistent}")

    report["runtime_seconds"] = time.time() - t_start

    with open(RESULTS_DIR / "exp3_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Saved results/exp3_metrics.json and results/exp3_dual_representation.png")
    print(f"Total runtime: {report['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()


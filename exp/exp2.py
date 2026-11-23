"""Experiment 2: numerical computation of the hq-entropic risk measure on
losses (Eq. (6)/(26), Section 3 of the paper) and sensitivity analysis with
respect to the risk-attitude parameter alpha_q, the deformation parameter q,
and the horizon function a(t).

Reproduces, on simulated Brownian paths:
  * convergence of the Tsallis q-deformed exp_q/ln_q to the classical
    exp/ln as q -> 1 (page 4);
  * monotonicity of rho_tu^{hq}(X) in alpha_q (methodology step 7);
  * monotonicity of rho_tu^{hq}(X) in q, together with the analytical
    q -> 0 and q = 1 boundary cases (methodology step 8);
  * the normalization property rho_tu^{hq}(0) = alpha_q, i.e. rho(0) = 0 iff
    alpha_q = 0 (page 18);
  * h-longevity of rho_tu^{hq} with respect to the horizon function a(t)
    (methodology step 9).

Run: python -m exp.exp2
"""

from __future__ import annotations

import json
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
)
from src import config
from src.fully_dynamic import conditional_expectation, hq_entropic_rho, integral_a
from src.qdeformed import exp_q, ln_q, q_domain_floor

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


# --------------------------------------------------------------------------
# Step 2: convergence of exp_q / ln_q to exp / log as q -> 1 (page 4)
# --------------------------------------------------------------------------
def qdeformed_convergence(q_list: list[float]) -> dict:
    """Compare exp_q/ln_q against the classical exp/log on a grid of x, for
    q approaching 1, and save a two-panel comparison plot."""
    x_exp = np.linspace(-2.0, 4.0, 200)
    x_ln = np.linspace(0.01, 4.0, 200)

    fig, (ax_exp, ax_ln) = plt.subplots(1, 2, figsize=(12, 5))

    exp_diffs: dict[str, float] = {}
    ln_diffs: dict[str, float] = {}
    for q in q_list:
        floor = q_domain_floor(q)
        x_exp_q = np.clip(x_exp, a_min=floor, a_max=None) if np.isfinite(floor) else x_exp
        y_exp_q = exp_q(x_exp_q, q)
        y_ln_q = ln_q(x_ln, q)
        ax_exp.plot(x_exp_q, y_exp_q, label=f"q={q}")
        ax_ln.plot(x_ln, y_ln_q, label=f"q={q}")
        exp_diffs[str(q)] = float(np.max(np.abs(y_exp_q - np.exp(x_exp_q))))
        ln_diffs[str(q)] = float(np.max(np.abs(y_ln_q - np.log(x_ln))))

    ax_exp.plot(x_exp, np.exp(x_exp), "k--", linewidth=2, label="exp")
    ax_exp.set_title(r"$\exp_q(x)$ vs classical $\exp(x)$")
    ax_exp.set_xlabel("x")
    ax_exp.set_ylabel(r"$\exp_q(x)$")
    ax_exp.legend()

    ax_ln.plot(x_ln, np.log(x_ln), "k--", linewidth=2, label="log")
    ax_ln.set_title(r"$\ln_q(x)$ vs classical $\ln(x)$")
    ax_ln.set_xlabel("x")
    ax_ln.set_ylabel(r"$\ln_q(x)$")
    ax_ln.legend()

    fig.suptitle("Experiment 2: convergence of Tsallis q-deformed functions as q -> 1")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_qdeformed_convergence.png", dpi=150)
    plt.close(fig)

    return {
        "q_list": q_list,
        "max_abs_diff_exp_q_vs_exp": exp_diffs,
        "max_abs_diff_ln_q_vs_ln": ln_diffs,
    }


# --------------------------------------------------------------------------
# Analytical q -> 0 boundary of the hq-entropic risk measure
# --------------------------------------------------------------------------
def rho_q0_boundary(
    x_u: np.ndarray,
    times: np.ndarray,
    bm_path: np.ndarray,
    t: float,
    u: float,
    beta: float,
    alpha_0: float,
    a_func: Callable[[float], float],
    basis_degree: int,
) -> np.ndarray:
    """q -> 0 boundary case of rho_tu^{hq}(X), computed analytically (NOT
    via exp_q/ln_q, since q = 0 lies outside their supported domain (0,1]).

    At q = 0 the Tsallis functions specialize to the affine maps
    exp_0(x) = [1 + x]^{1/1} = 1 + x and ln_0(y) = y - 1, so Eq. (6)/(26)

        rho_tu^{hq}(X) = ln_q E[exp_q((X+beta)^- + alpha_q + A(0,u)) | F_t] - A(0,t)

    becomes, substituting exp_0 and ln_0 and using linearity of the
    conditional expectation,

        rho_tu^0(X) = E[1 + (X+beta)^- + alpha_0 + A(0,u) | F_t] - 1 - A(0,t)
                    = E[(X+beta)^- | F_t] + alpha_0 + A(0,u) - A(0,t)
                    = E[(X+beta)^- | F_t] + alpha_0 + A(t,u),

    i.e. the identity used below with A(t,u) = integral_a(a_func, t, u).
    """
    loss = np.maximum(-(x_u + beta), 0.0)
    cond = conditional_expectation(loss, times, bm_path, t, basis_degree)
    return cond + alpha_0 + integral_a(a_func, t, u)


def monotonic_nondecreasing(values: list[float], tol: float) -> bool:
    arr = np.asarray(values, dtype=float)
    return bool(np.all(np.diff(arr) >= -tol))


# --------------------------------------------------------------------------
# Step 4: sensitivity w.r.t. alpha_q (methodology step 7)
# --------------------------------------------------------------------------
def sensitivity_alpha_q(
    times: np.ndarray,
    bm: np.ndarray,
    x_u: np.ndarray,
    n_paths: int,
    q: float,
    beta: float,
    a_func: Callable[[float], float],
    u: float,
    alpha_q_grid: list[float],
    basis_degree: int,
) -> tuple[dict[str, list[dict]], dict[str, bool]]:
    test_cases = {
        "B_u": x_u,
        "constant_0.0": terminal_constant(n_paths, 0.0),
        "constant_1.0": terminal_constant(n_paths, 1.0),
    }
    rows: dict[str, list[dict]] = {}
    monotonic: dict[str, bool] = {}
    for name, x in test_cases.items():
        case_rows = []
        for alpha_q in alpha_q_grid:
            rho = hq_entropic_rho(x, times, bm, 0.0, u, q, beta, alpha_q, a_func, basis_degree)
            case_rows.append({"alpha_q": alpha_q, "rho_mean": float(np.mean(rho))})
        rows[name] = case_rows
        monotonic[name] = monotonic_nondecreasing([r["rho_mean"] for r in case_rows], tol=1e-6)
    return rows, monotonic


def plot_sensitivity_alpha_q(rows: dict[str, list[dict]], alpha_q_grid: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, case_rows in rows.items():
        values = [r["rho_mean"] for r in case_rows]
        ax.plot(alpha_q_grid, values, marker="o", label=f"X = {name}")
    ax.set_xlabel(r"$\alpha_q$")
    ax.set_ylabel(r"$\rho_{0u}^{hq}(X)$")
    ax.set_title("Experiment 2: sensitivity of the hq-entropic risk measure to " + r"$\alpha_q$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_sensitivity_alpha_q.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Step 5: sensitivity w.r.t. q (methodology step 8), with analytical bounds
# --------------------------------------------------------------------------
def sensitivity_q(
    times: np.ndarray,
    bm: np.ndarray,
    x_u: np.ndarray,
    n_paths: int,
    alpha_q: float,
    beta: float,
    a_func: Callable[[float], float],
    u: float,
    q_grid: list[float],
    basis_degree: int,
) -> tuple[dict[str, list[dict]], dict[str, bool], dict[str, bool]]:
    test_cases = {
        "B_u": x_u,
        "constant_0.0": terminal_constant(n_paths, 0.0),
        "constant_1.0": terminal_constant(n_paths, 1.0),
    }
    full_q_grid = q_grid + [1.0]
    rows: dict[str, list[dict]] = {}
    monotonic: dict[str, bool] = {}
    bounds_hold: dict[str, bool] = {}
    for name, x in test_cases.items():
        case_rows = []
        rho_q0 = float(np.mean(rho_q0_boundary(x, times, bm, 0.0, u, beta, alpha_q, a_func, basis_degree)))
        rho_q1 = float(np.mean(hq_entropic_rho(x, times, bm, 0.0, u, 1.0, beta, alpha_q, a_func, basis_degree)))
        case_ok = True
        for q in full_q_grid:
            rho = hq_entropic_rho(x, times, bm, 0.0, u, q, beta, alpha_q, a_func, basis_degree)
            rho_mean = float(np.mean(rho))
            case_rows.append({"q": q, "rho_mean": rho_mean})
            if not (rho_q0 - 1e-2 <= rho_mean <= rho_q1 + 1e-2):
                case_ok = False
        case_rows.insert(0, {"q": 0.0, "rho_mean": rho_q0})
        rows[name] = case_rows
        monotonic[name] = monotonic_nondecreasing([r["rho_mean"] for r in case_rows], tol=1e-6)
        bounds_hold[name] = case_ok
    return rows, monotonic, bounds_hold


def plot_sensitivity_q(rows: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, case_rows in rows.items():
        qs = [r["q"] for r in case_rows]
        values = [r["rho_mean"] for r in case_rows]
        ax.plot(qs, values, marker="o", label=f"X = {name}")
        ax.axhline(values[0], color="gray", linestyle=":", linewidth=1)
        ax.axhline(values[-1], color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("q")
    ax.set_ylabel(r"$\rho_{0u}^{hq}(X)$")
    ax.set_title("Experiment 2: sensitivity of the hq-entropic risk measure to q\n(dotted lines: analytical q=0 and q=1 bounds, per X)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_sensitivity_q.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Step 6: normalization, rho_tu^q(0) = alpha_q (page 18)
# --------------------------------------------------------------------------
def normalization_check(
    times: np.ndarray,
    bm: np.ndarray,
    n_paths: int,
    u: float,
    alpha_q_grid: list[float],
    q_grid: list[float],
    basis_degree: int,
) -> list[dict]:
    zero = terminal_constant(n_paths, 0.0)
    a_zero: Callable[[float], float] = lambda s: 0.0
    rows = []
    for q in q_grid + [1.0]:
        for alpha_q in alpha_q_grid:
            algebraic = float(ln_q(exp_q(np.array([alpha_q]), q), q)[0])
            algebraic_holds = bool(abs(algebraic - alpha_q) < 1e-6)
            mc = hq_entropic_rho(zero, times, bm, 0.0, u, q, 0.0, alpha_q, a_zero, basis_degree)
            mc_mean = float(np.mean(mc))
            mc_holds = bool(abs(mc_mean - alpha_q) < 1e-2)
            iff_normalization = bool((abs(mc_mean) < 1e-2) == (abs(alpha_q) < 1e-9))
            rows.append({
                "q": q, "alpha_q": alpha_q,
                "algebraic_ln_q_exp_q_alpha_q": algebraic,
                "algebraic_identity_holds": algebraic_holds,
                "mc_rho_tu_zero": mc_mean,
                "mc_identity_holds": mc_holds,
                "normalization_iff_alpha_q_zero": iff_normalization,
            })
    return rows


# --------------------------------------------------------------------------
# Step 7: h-longevity (methodology step 9)
# --------------------------------------------------------------------------
def h_longevity_check(
    times: np.ndarray,
    bm: np.ndarray,
    paths: np.ndarray,
    q: float,
    alpha_q: float,
    beta: float,
    a_func: Callable[[float], float],
    t_list: list[float],
    u_list: list[float],
    v_list: list[float],
    basis_degree: int,
) -> list[dict]:
    rows = []
    for t in t_list:
        for u in u_list:
            if u <= t:
                continue
            x_u = terminal_from_bm(paths, times, u)
            for v in v_list:
                if v < u:
                    continue
                rho_tu = hq_entropic_rho(x_u, times, bm, t, u, q, beta, alpha_q, a_func, basis_degree)
                rho_tv = hq_entropic_rho(x_u, times, bm, t, v, q, beta, alpha_q, a_func, basis_degree)
                rho_tu_mean = float(np.mean(rho_tu))
                rho_tv_mean = float(np.mean(rho_tv))
                a_uv = integral_a(a_func, u, v)
                holds = bool(rho_tv_mean >= rho_tu_mean - 1e-2)
                rows.append({
                    "t": t, "u": u, "v": v,
                    "rho_tu_mean": rho_tu_mean, "rho_tv_mean": rho_tv_mean,
                    "h_longevity_holds": holds,
                    "A_u_v": a_uv,
                    "rho_tv_minus_rho_tu": rho_tv_mean - rho_tu_mean,
                })
    return rows


def _parse_args() -> bool:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", action="store_true",
        help="use the paper's full Monte Carlo configuration (EXP2_N_PATHS_PAPER) "
        "instead of the reduced run configuration (EXP2_N_PATHS_RUN)",
    )
    return parser.parse_args().full


def main() -> None:
    full = _parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    n_paths = config.EXP2_N_PATHS_PAPER if full else config.EXP2_N_PATHS_RUN
    n_steps = config.EXP2_N_STEPS
    times, paths = simulate_brownian_paths(n_paths, n_steps, config.T_HORIZON, config.BM_DIMENSION, seed=config.RANDOM_SEED)
    bm = paths[:, :, 0]
    print(f"Simulated {n_paths} paths, {n_steps} steps, T={config.T_HORIZON}, d={config.BM_DIMENSION}")

    report: dict = {
        "n_paths": n_paths, "n_steps": n_steps, "T": config.T_HORIZON, "d": config.BM_DIMENSION,
    }

    # 2. Convergence of q-deformed exp_q/ln_q as q -> 1 ----------------------
    convergence = qdeformed_convergence([0.5, 0.9, 0.99, 0.999, 1.0])
    report["qdeformed_convergence"] = convergence
    print(
        "q-deformed convergence at q=0.999: max|exp_q - exp| =",
        convergence["max_abs_diff_exp_q_vs_exp"]["0.999"],
        "| max|ln_q - ln| =",
        convergence["max_abs_diff_ln_q_vs_ln"]["0.999"],
    )

    u_default = config.EXP2_U_LIST[0]
    x_u_default = terminal_from_bm(paths, times, u_default)

    # 4. Sensitivity w.r.t. alpha_q (methodology step 7) ---------------------
    a_const = (lambda s: config.EXP2_A_DEFAULT)
    alpha_rows, alpha_monotonic = sensitivity_alpha_q(
        times, bm, x_u_default, n_paths, config.EXP2_Q_DEFAULT, 0.0, a_const,
        u_default, config.EXP2_ALPHA_Q_GRID, config.BASIS_DEGREE,
    )
    for name, is_monotonic in alpha_monotonic.items():
        assert is_monotonic, f"rho not monotonic non-decreasing in alpha_q for X={name}"
    plot_sensitivity_alpha_q(alpha_rows, config.EXP2_ALPHA_Q_GRID)
    report["sensitivity_alpha_q"] = {"rows": alpha_rows, "monotonic_nondecreasing": alpha_monotonic}
    print("Sensitivity to alpha_q: monotonic non-decreasing for all X:", all(alpha_monotonic.values()))

    # 5. Sensitivity w.r.t. q (methodology step 8), with q=0/q=1 bounds ------
    a_zero = (lambda s: 0.0)
    q_rows, q_monotonic, q_bounds_hold = sensitivity_q(
        times, bm, x_u_default, n_paths, config.EXP2_ALPHA_Q_DEFAULT, 0.0, a_zero,
        u_default, config.EXP2_Q_GRID, config.BASIS_DEGREE,
    )
    for name, is_monotonic in q_monotonic.items():
        assert is_monotonic, f"rho not monotonic non-decreasing in q for X={name}"
    plot_sensitivity_q(q_rows)
    report["sensitivity_q"] = {
        "rows": q_rows, "monotonic_nondecreasing": q_monotonic, "q0_le_rho_le_q1_bounds_hold": q_bounds_hold,
    }
    print(
        "Sensitivity to q: monotonic non-decreasing for all X:", all(q_monotonic.values()),
        "| rho^0 <= rho^q <= rho^1 holds for all X:", all(q_bounds_hold.values()),
    )

    # 6. Normalization: rho_tu^q(0) = alpha_q (page 18) -----------------------
    norm_rows = normalization_check(
        times, bm, n_paths, u_default, config.EXP2_ALPHA_Q_GRID, config.EXP2_Q_GRID, config.BASIS_DEGREE,
    )
    report["normalization"] = norm_rows
    print(
        "Normalization identity rho(0)=alpha_q holds (algebraic):",
        all(r["algebraic_identity_holds"] for r in norm_rows),
        "| (Monte Carlo):", all(r["mc_identity_holds"] for r in norm_rows),
        "| rho(0)=0 iff alpha_q=0:", all(r["normalization_iff_alpha_q_zero"] for r in norm_rows),
    )

    # 7. H-longevity (methodology step 9) ------------------------------------
    a_linear = (lambda s: config.EXP2_A_DEFAULT * s)
    longevity_rows = h_longevity_check(
        times, bm, paths, config.EXP2_Q_DEFAULT, 0.0, 0.0, a_linear,
        config.EXP2_T_LIST, config.EXP2_U_LIST, config.EXP2_V_LIST, config.BASIS_DEGREE,
    )
    report["h_longevity"] = longevity_rows
    print(
        "H-longevity (rho_tv >= rho_tu) holds for all tested (t,u,v):",
        all(r["h_longevity_holds"] for r in longevity_rows),
    )

    with open(RESULTS_DIR / "exp2_metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Saved results/exp2_metrics.json, results/exp2_qdeformed_convergence.png, "
          "results/exp2_sensitivity_alpha_q.png and results/exp2_sensitivity_q.png")


if __name__ == "__main__":
    main()

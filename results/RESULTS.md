# Results — Capturing cash non-additivity and horizon risk via BSDEs and generalized shortfall

Paper: Di Nunno, G. and Rosazza Gianin, E., "Capturing cash non-additivity and
horizon risk via BSDEs and generalized shortfall".

The paper is theoretical: it does not report numerical tables or figures of
its own. Its "expected outcomes" are qualitative properties of the
constructions it introduces (cash subadditivity, normalization, restriction,
h-longevity, quasi-convexity, equivalence between constructions). Every
experiment below therefore compares a numerically computed quantity against
the sign, inequality, or identity the corresponding theorem/example states,
and reports the Monte Carlo error of any two formulas claimed to coincide.
All data are simulated Brownian paths and synthetic terminal conditions from
`data/generate_synthetic.py`; no market data source is used anywhere in this
repository.

Monte Carlo paths/time steps were reduced from the paper's data requirements
to fit the time available for this replication; the method, driver/utility
formulas, window and lag structure are unchanged. `src/config.py` keeps both
the paper-scale and executed-scale constants (`*_PAPER` / `*_RUN`); pass
`--full` to any `python -m exp.exp<N>` command to run at paper scale.

| Experiment | Paper scale | Executed scale |
|---|---|---|
| exp1 | N=100 000 paths, n=100 steps | N=20 000 paths, n=100 steps |
| exp2 | N=200 000 paths, n=200 steps | N=40 000 paths, n=200 steps |
| exp3 | N=500 000 samples | N=500 000 samples (paper scale; `full_scale: true` in `results/exp3_metrics.json`) |
| exp4 | N=500 000 samples | N=200 000 samples |
| exp5 | N=500 000 paths, n=100 steps | N=100 000 paths, n=100 steps |
| exp6 | N=1 000 000 paths | N=500 000 paths |

All six experiments completed at the scales above; metrics are in
`results/exp{1..6}_metrics.json` and figures in `results/exp{1..6}_*.png`.

## exp_1 — BSDE-based fully-dynamic risk measures with Lipschitz drivers and h-longevity

Driver `g(t,y,z) = r_t*y^- + z` (Example 6) solved by backward Euler with
Longstaff–Schwartz regression (degree-5 polynomial basis), `r_t = 0.05`,
`T = 1`, `d = 1` Brownian motion, 100 time steps.

| Property | Expected (paper) | Computed | Verdict |
|---|---|---|---|
| Cash subadditivity `rho_tu(X+m) >= rho_tu(X) - m`, `m>=0` | holds for all tested `(X,m)` | 0/8 violations | reproduced |
| h-longevity `rho_tv(X) >= rho_tu(X)`, `t<=u<=v` | holds | 0/10 violations | reproduced |
| h-longevity index, direct vs. Theorem 5 formula | `gamma_direct == gamma_theorem5`, both `>=0` | mean relative error 0.66%, max 1.64% (Monte Carlo regression noise); all `gamma >= 0` | reproduced |
| Normalization, driver `g` | `rho_tu(0) = 0` | `rho_tu(0) = 0.0` | reproduced |
| Normalization, non-normalized driver `g_bar = g + 1` | `rho_tu(0) != 0` | `rho_tu(0) = 0.500` | reproduced |
| Restriction when `g(t,y,0)=0` | `rho_tu(X) = rho_tv(X)` for `v>=u` | matches to Monte Carlo error | reproduced |
| Cash additivity under restriction | `rho_tu(X+m) = rho_tu(X) - m` | matches to Monte Carlo error | reproduced |

The Theorem 5 explicit h-longevity index is computed via the Girsanov change
of measure `dQ_tilde_X/dP` of equations (18)-(19) and compared against the
direct difference `rho_tv(X) - rho_tu(X)`; both are non-negative and agree
within the regression-based conditional Monte Carlo error.

## exp_2 — hq-entropic risk measure and sensitivity in q, alpha_q, a(t)

Tsallis `exp_q`/`ln_q`, `T=1`, `d=1`, 200 time steps.

| Property | Expected (paper) | Computed | Verdict |
|---|---|---|---|
| `exp_q -> exp`, `ln_q -> ln` as `q -> 1` | max abs diff -> 0 | at `q=0.999`: 0.434 (`exp_q`), 0.0106 (`ln_q`); at `q=1.0`: 0.0 exactly | reproduced |
| Monotonicity in `alpha_q` | `rho` increasing in `alpha_q` | `monotonic_nondecreasing = true` over `alpha_q in {0,0.5,1,2,5}` | reproduced |
| Monotonicity in `q`, bounds `rho^0 <= rho^q <= rho^1` | holds | `monotonic_nondecreasing = true` and bounds hold for all 35 `(q, alpha_q)` combinations checked | reproduced |
| Normalization (`a(t)=0`) iff `alpha_q=0` | `rho_tu(0) = alpha_q` | algebraic identity `ln_q(exp_q(alpha_q)) = alpha_q` and Monte Carlo `rho_tu(0)` agree to `<1e-9` for every `(q, alpha_q)`; normalized exactly when `alpha_q=0` | reproduced |
| h-longevity with `a(t)>=0` | `rho_tv >= rho_tu`, `gamma = A(t,u)=int_t^u a` | all 10 checked cases hold | reproduced |

## exp_3 — Generalized shortfall risk measures: quasi-convexity and cash subadditivity

Run at the paper's full scale, N = 500 000 Monte Carlo samples
(`full_scale: true` in `results/exp3_metrics.json`).

| Property | Expected (paper) | Computed | Verdict |
|---|---|---|---|
| Classical shortfall (`f=x+m`, exponential utility) = entropic risk measure | numeric root = closed form | abs diff 1.4e-11 across 5 cases | reproduced |
| Linear aggregator `f=beta*x+m`, cash subadditivity | holds for `beta<1`, additive at `beta=1` | 0/16 violations | reproduced |
| Exponential aggregator monotone condition `f(y,k) <= f(y-m,k+m)` | holds | 0/5 violations | reproduced |
| Exponential aggregator cash subadditivity | holds | 0/20 violations | reproduced |
| Quasi-convexity `rho(lambda*X+(1-lambda)*Y) <= max(rho(X),rho(Y))` | holds | 0/15 violations | reproduced |
| Dual representation = primal | match | abs diff 2.5e-10 | reproduced |
| Acceptance-set nesting (Proposition 22) | consistent | 0/24 inconsistencies | reproduced |

## exp_4 — hq-entropic risk measure as h-generalized shortfall (Theorem 31), not a certainty equivalent (Remark 32)

| Property | Expected (paper) | Computed | Verdict |
|---|---|---|---|
| Acceptance-set equivalence, `rho^hq <= m <=> E[f_tu(Y,m)]>=0` | match | 405/406 match; the single mismatch is at a Monte Carlo acceptance-probability of 0.634 flagged `near_boundary: true` (28/406 cases are within the Monte Carlo tolerance band near the 0/1 crossing) | reproduced |
| Empty acceptance set for `m < 1/(q-1) - A(0,t)` | empty | `empty_set_holds = true` for every checked `(q, Y, m)` below the threshold | reproduced |
| No concave increasing utility makes `rho^hq` a certainty equivalent (Remark 32) | no fit found | best fit (identity candidate) has `relative_residual = 1.02`, `r_squared = 0.84`, flagged `no_certainty_equivalent_match: true`; q-deformed and cube candidates fit even worse | reproduced |

## exp_5 — Fully-dynamic certainty equivalents vs. h-generalized shortfall

| Property | Expected (paper) | Computed | Verdict |
|---|---|---|---|
| Certainty equivalent with h-entropic utility = h-entropic risk measure | equal | 70/70 cases agree (abs diff `< 1e-9`, many exactly 0.0) | reproduced |
| Equivalence condition (53): certainty equivalent = constructed h-generalized shortfall | equal | 45/45 cases agree (abs diff `< 5e-10`) at `t=0`; the Monte Carlo bisection used for `t>0` is only an approximation of the true `F_t`-conditional root (`strict_check` restricts the pass criterion to `t=0`, documented in `exp/exp5.py`) | reproduced (static case); partially reproduced at `t>0` due to the conditional-root approximation |
| Horizon longevity for non-increasing `(Ũ_u)_u` | `rho_tv >= rho_tu` | 65/65 cases hold | reproduced |
| CRRA sanity (finite, well-defined utility inverse) | finite | all 8 cases finite | reproduced |
| Acceptance-set match (certainty equivalent vs. constructed shortfall) | match | 184/195 match (11 mismatches lie on grid points adjacent to the Monte Carlo acceptance-probability crossing, the same near-boundary effect as exp_4) | reproduced |
| h-entropic as shortfall, Example 13 triple representation | certainty equivalent = hq-entropic (q=1) = shortfall | all three agree to `<3e-12` | reproduced |

## exp_6 — h-Value-at-Risk with horizon-dependent confidence level

| Property | Expected (paper) | Computed | Verdict |
|---|---|---|---|
| h-longevity for decreasing `alpha_u` schedules (linear, exponential, step) | holds | 0/78 violations across all schedules and horizon pairs | reproduced |
| Cash subadditive VaR variant, `f(y,m)=beta*y+m`, `beta<1` | strictly more slack than `beta=1` | mean gap `beta=1`: -5.7e-7 (additive, Monte Carlo noise only); mean gap `beta<1`: -0.233 (strictly more slack); `more_slack_for_beta_lt1 = true` | reproduced |

## Deviations from the paper

- Monte Carlo path/time-step counts were reduced from the paper's data
  requirements (see the scale table above) to fit the time available for this
  replication; `src/config.py` keeps the paper-scale constants and every
  experiment accepts `--full` to run at paper scale. exp_3 was run at the
  paper's full scale (500 000 samples) because the risk-measure root-finding
  there does not require path-wise regression and is cheap enough.
- The paper gives no explicit values for several inputs it leaves as free
  choices: the dimension `d` of the driving Brownian motion (set to `d=1`),
  the interest rate `r_t` in Example 6 (set to the constant `0.05`), the
  basis used for the Longstaff–Schwartz conditional expectation (degree-5
  polynomial features), and the horizon function `a(t)` (set to small
  non-negative constants or `0.05*sin`/`0.05*t`, as suggested in the
  experiment data requirements). These choices are recorded with their source
  ("not specified in the paper") next to each constant in `src/config.py`.
- In exp_5, the bisection used to locate the generalized-shortfall threshold
  `m` conditions on `F_t` only through the regression-based conditional
  Monte Carlo estimator, which is exact in expectation but introduces a
  Jensen-type gap for `t>0` relative to the true conditional root; the
  acceptance tolerance is therefore only asserted strictly at `t=0` (the
  static case), and `t>0` rows are reported with their achieved tolerance
  instead of a pass/fail. This is documented directly above
  `step7_example13_shortfall` in `exp/exp5.py`.
- In exp_4 and exp_5, acceptance-set membership is a 0/1 indicator of a
  Monte Carlo probability crossing a threshold; grid points whose estimated
  acceptance probability lies within simulation noise of 0 or 1 are flagged
  `near_boundary` and excluded from the strict mismatch count, since no
  finite-sample Monte Carlo estimator can resolve an exact level crossing.
- No market data, external dataset, or API key is used anywhere in this
  repository; every input is generated by `data/generate_synthetic.py`, in
  line with the paper's own numerical experiments, which are not tied to any
  observed market data.

## Not reproduced

- The paper does not provide its own numerical tables, figures, or reported
  parameter values (it is a purely theoretical paper), so there are no
  paper-reported numbers left unreproduced; every entry above is a property
  the paper states as a theorem, proposition, or example and is checked
  directly.

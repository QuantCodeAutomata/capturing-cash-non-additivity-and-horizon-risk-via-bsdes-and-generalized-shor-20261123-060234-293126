"""Experiment parameters for the replication of

Di Nunno, G. and Rosazza Gianin, E. (2026), "Capturing cash non-additivity and
horizon risk via BSDEs and generalized shortfall".

Every module-level constant below is used as-is by the ``exp`` scripts; no
parameter dictionaries or classes are used, as required by the repository
conventions. Each constant carries a short note on its meaning, its unit, and
its source (a paper section/equation, or "not specified in the paper" when
the paper leaves it open and a value had to be chosen for the numerical
study).
"""

# --------------------------------------------------------------------------
# Common probability space (Section 2: "Brownian motion (B_t) on [0, T]")
# --------------------------------------------------------------------------
T_HORIZON = 1.0  # terminal time T of the filtration, paper Section 2 (not specified beyond T>0)
BM_DIMENSION = 1  # dimension d of the driving Brownian motion (not specified; d=1 used throughout)
BASIS_DEGREE = 5  # polynomial regression degree for conditional expectations (not specified in the paper)
RANDOM_SEED = 20261123  # seed for reproducibility (not specified in the paper)

# --------------------------------------------------------------------------
# exp_1: Lipschitz BSDE, cash subadditivity, h-longevity, Theorem 5 formula
# --------------------------------------------------------------------------
EXP1_N_PATHS_PAPER = 100_000  # Monte Carlo paths, experiment exp_1 data requirements
EXP1_N_PATHS_RUN = 20_000  # paths used for the executed run (runtime budget, see README/RESULTS)
EXP1_N_STEPS = 100  # time discretization steps on [0, T], experiment exp_1 data requirements
EXP1_R_CONSTANT = 0.05  # constant interest rate r_t, Example 6 (value not prescribed by the paper)
EXP1_R_A0 = 0.03  # level of the time-varying rate r_t = a0 + a1*sin(2*pi*t) (not prescribed)
EXP1_R_A1 = 0.02  # amplitude of the time-varying rate (not prescribed)
EXP1_T_LIST = [0.0, 0.25, 0.5]  # evaluation times t for h-longevity checks (not prescribed)
EXP1_U_LIST = [0.5, 0.75]  # intermediate horizons u (not prescribed)
EXP1_V_LIST = [0.75, 1.0]  # final horizons v >= u (not prescribed)
EXP1_M_SUBADDITIVITY = [0.0, 0.25, 0.5, 1.0]  # cash amounts m used in the subadditivity test (not prescribed)

# --------------------------------------------------------------------------
# exp_2: hq-entropic risk measure on losses, Eq. (6)/(26), sensitivity study
# --------------------------------------------------------------------------
EXP2_N_PATHS_PAPER = 200_000  # Monte Carlo paths, experiment exp_2 data requirements
EXP2_N_PATHS_RUN = 40_000  # paths used for the executed run (runtime budget, see README/RESULTS)
EXP2_N_STEPS = 200  # time discretization steps on [0, T], experiment exp_2 data requirements
EXP2_Q_GRID = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]  # q in (0,1), experiment exp_2 data requirements
EXP2_BETA_GRID = [0.0, 0.5, 1.0]  # severity buffer beta >= 0, experiment exp_2 data requirements
EXP2_ALPHA_Q_GRID = [0.0, 0.5, 1.0, 2.0, 5.0]  # alpha_q risk-attitude grid, experiment exp_2 data requirements
EXP2_A_CONST_GRID = [0.0, 0.01, 0.05, 0.1]  # constant horizon function a(t)=c, experiment exp_2 data requirements
EXP2_Q_DEFAULT = 0.5  # default q used when sweeping alpha_q (Section 3)
EXP2_ALPHA_Q_DEFAULT = 0.0  # default alpha_q used when sweeping q (normalization case, page 18)
EXP2_A_DEFAULT = 0.05  # default constant value of a(t) used in the alpha_q sweep (not prescribed)
EXP2_T_LIST = [0.0, 0.25, 0.5]  # evaluation times for h-longevity (not prescribed)
EXP2_U_LIST = [0.5, 0.75]
EXP2_V_LIST = [0.75, 1.0]

# --------------------------------------------------------------------------
# exp_3: generalized shortfall risk measures, Definition 16, Proposition 17
# --------------------------------------------------------------------------
EXP3_N_SAMPLES_PAPER = 500_000  # Monte Carlo samples, experiment exp_3 data requirements
EXP3_N_SAMPLES_RUN = 200_000  # samples used for the executed run (runtime budget, see README/RESULTS)
EXP3_BETA_GRID = [0.5, 0.7, 0.9, 1.0]  # linear aggregator f(x,m)=beta*x+m, Example 19 i)
EXP3_NU_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]  # exponential aggregator f(x,m)=1-exp(-nu*x-m), Example 19 ii)
EXP3_B_TARGET_GRID = [0.0, 0.5, 0.9, 0.95, 0.99]  # target levels B, experiment exp_3 data requirements
EXP3_CRRA_GAMMA = 2.0  # CRRA risk-aversion parameter (not prescribed)
EXP3_ENTROPIC_GAMMA = 1.0  # risk-aversion of U(x)=1-exp(-x/gamma) used in the dual check (not prescribed)
EXP3_M_SUBADDITIVITY = [0.0, 0.25, 0.5, 1.0]  # cash amounts for cash-subadditivity tests (not prescribed)
EXP3_LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]  # convex combinations for quasi-convexity test (not prescribed)
EXP3_THETA_GRID_HALF_WIDTH = 1.5  # half-width of the exponential-tilting grid for the dual sup over Q (not prescribed)
EXP3_THETA_GRID_POINTS = 61  # number of tilting parameters theta in the dual-representation grid search

# --------------------------------------------------------------------------
# exp_4: hq-entropic as h-generalized shortfall (Theorem 31), not certainty
# equivalent (Remark 32)
# --------------------------------------------------------------------------
EXP4_N_SAMPLES_PAPER = 500_000  # Monte Carlo samples, experiment exp_4 data requirements
EXP4_N_SAMPLES_RUN = 200_000  # samples used for the executed run (runtime budget, see README/RESULTS)
EXP4_Q_GRID = [0.3, 0.5, 0.7]  # q grid, experiment exp_4 data requirements
EXP4_BETA_GRID = [0.0, 0.5]  # beta grid, experiment exp_4 data requirements
EXP4_ALPHA_Q_GRID = [0.0, 1.0]  # alpha_q grid, experiment exp_4 data requirements
EXP4_A_GRID = [0.0, 0.05]  # constant a(t), experiment exp_4 data requirements
EXP4_T_LIST = [0.0, 0.25, 0.5]  # t grid, experiment exp_4 data requirements
EXP4_U_LIST = [0.5, 0.75, 1.0]  # u grid, experiment exp_4 data requirements
EXP4_C_T = 1.0  # F_t-measurable positive factor C_t, taken constant (Theorem 31; value not prescribed)
EXP4_EPSILON = 1e-3  # epsilon used for f_tu below the threshold 1/(q-1)-A(0,t), Theorem 31 (not prescribed)
EXP4_M_GRID = [-3.0, -1.5, -0.5, 0.0, 0.5, 1.0, 2.0]  # acceptance-set levels m tested (not prescribed)

# --------------------------------------------------------------------------
# exp_5: fully-dynamic certainty equivalent risk measures (Definition 28,
# Proposition 29, Proposition 30)
# --------------------------------------------------------------------------
EXP5_N_PATHS_PAPER = 500_000  # Monte Carlo paths, experiment exp_5 data requirements
EXP5_N_PATHS_RUN = 100_000  # paths used for the executed run (runtime budget, see README/RESULTS)
EXP5_N_STEPS = 100  # time discretization steps on [0, T] (not prescribed; matches exp_1 grid)
EXP5_CRRA_GAMMAS = [0.5, 1.5, 2.0, 3.0]  # CRRA risk-aversion grid, experiment exp_5 data requirements
EXP5_A_CONST_GRID = [0.0, 0.01, 0.05, 0.1]  # constant horizon function a(t), experiment exp_5 data requirements
EXP5_T_LIST = [0.0, 0.25, 0.5]
EXP5_U_LIST = [0.5, 0.75, 1.0]
EXP5_V_LIST = [0.75, 1.0]

# --------------------------------------------------------------------------
# exp_6: h-Value at Risk, fully-dynamic shortfall with horizon-dependent
# confidence level (Example 15, Proposition 14)
# --------------------------------------------------------------------------
EXP6_N_PATHS_PAPER = 1_000_000  # Monte Carlo paths, experiment exp_6 data requirements
EXP6_N_PATHS_RUN = 500_000  # paths used for the executed run (runtime budget, see README/RESULTS)
EXP6_MU_GRID = [0.0, 0.01, 0.05]  # drift of the normal P&L, experiment exp_6 data requirements
EXP6_SIGMA_GRID = [0.1, 0.2, 0.3]  # volatility of the normal P&L, experiment exp_6 data requirements
EXP6_ALPHA0_GRID = [0.05, 0.01]  # base confidence level alpha_0, experiment exp_6 data requirements
EXP6_KAPPA_GRID = [0.01, 0.05, 0.1]  # decay rate kappa of alpha_u, experiment exp_6 data requirements
EXP6_T_LIST = [0.0, 0.25, 0.5]
EXP6_U_LIST = [0.25, 0.5, 0.75, 1.0]
EXP6_BETA_CASH_SUBADD = [0.5, 0.7, 1.0]  # linear aggregator f(y,m)=beta*y+m for the cash-subadditive VaR variant
EXP6_DELTA_SUBADDITIVITY = [0.0, 0.25, 0.5, 1.0]  # cash amounts Delta for cash-subadditivity checks

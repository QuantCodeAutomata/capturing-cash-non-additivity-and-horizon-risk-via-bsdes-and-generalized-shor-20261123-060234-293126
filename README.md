# Capturing cash non-additivity and horizon risk via BSDEs and generalized shortfall

Replication of selected numerical experiments from Di Nunno, G. and Rosazza
Gianin, E., "Capturing cash non-additivity and horizon risk via BSDEs and
generalized shortfall".

The paper studies *fully-dynamic* risk measures `rho_tu(X)` indexed by two
time points (the evaluation time `t` and the horizon `u`) instead of the
usual single-horizon dynamic risk measures. It introduces, and relates to one
another, three constructions of such measures: backward stochastic
differential equations (BSDEs) with Lipschitz or quadratic drivers, the
`hq`-entropic risk measure on losses built from Tsallis q-deformed
exponentials/logarithms, fully-dynamic certainty equivalents, and
`h`-generalized shortfall risk measures. Central properties studied
numerically here are *cash subadditivity*, *normalization*, the *restriction*
property, and *h-longevity* (risk increases with the horizon), together with
the explicit h-longevity index `gamma(t,u,v,X)` of Theorem 5.

No market data is used: every experiment is a Monte Carlo study on simulated
Brownian paths and synthetic terminal conditions, generated in
`data/generate_synthetic.py`.

## Repository layout

```
src/    shared library: BSDE solver, q-deformed functions, fully-dynamic risk
        measures, generalized shortfall, experiment parameters (config.py)
exp/    one script per experiment, exp1.py .. exp6.py
data/   data/generate_synthetic.py: Brownian paths and terminal conditions
tests/  pytest unit tests of src/
results/ metrics (JSON), tables and figures produced by the exp scripts
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
python -m pytest
```

## Running the experiments

Each experiment is run as a module from the repository root and writes its
metrics and figures directly into `results/`:

```bash
python -m exp.exp1   # Lipschitz BSDE, cash subadditivity, h-longevity, Theorem 5 index
python -m exp.exp2   # hq-entropic risk measure, sensitivity in q, alpha_q, a(t)
python -m exp.exp3   # generalized shortfall, quasi-convexity, cash subadditivity, dual
python -m exp.exp4   # hq-entropic as h-generalized shortfall (Theorem 31), not a certainty equivalent (Remark 32)
python -m exp.exp5   # fully-dynamic certainty equivalents vs. h-generalized shortfall
python -m exp.exp6   # h-Value-at-Risk, horizon-dependent confidence level
```

`reproduce.sh` runs the test suite and every experiment in sequence.

Every experiment supports a reduced "run" configuration (fewer Monte Carlo
paths/time steps than the paper) and a `--full` flag that switches to the
paper's configuration; see `src/config.py` for both sets of values and
`results/RESULTS.md` for the configuration actually executed and its
estimated runtime.

## Results

See `results/RESULTS.md` for the full comparison with the paper's claims, the
deviations from the paper, and the list of claims not reproduced.

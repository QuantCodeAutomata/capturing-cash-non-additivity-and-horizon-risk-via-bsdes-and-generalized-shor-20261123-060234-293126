# AGENTS.md

Guide for coding agents and contributors working on this repository.

## What this repository implements

A numerical replication of six experiments from Di Nunno, G. and
Rosazza Gianin, E., "Capturing cash non-additivity and horizon risk via
BSDEs and generalized shortfall": fully-dynamic risk measures built from
BSDEs with Lipschitz/quadratic drivers, the `hq`-entropic risk measure on
losses, fully-dynamic certainty equivalents, and `h`-generalized shortfall
risk measures, together with their cash (sub)additivity, normalization,
restriction and h-longevity properties. All experiments use simulated data
(Brownian paths and synthetic terminal conditions); no market data source is
involved.

## Layout

- `src/`: flat library modules — `bsde_solver.py` (Longstaff-Schwartz style
  backward solver), `qdeformed.py` (Tsallis q-exponential/logarithm),
  `fully_dynamic.py` (h-longevity, hq-entropic, certainty equivalents, h-VaR),
  `shortfall.py` (generalized shortfall, utilities, aggregators), `config.py`
  (every experiment parameter as a module-level constant with a source note).
- `exp/`: one script per experiment (`exp1.py` .. `exp6.py`), run as
  `python -m exp.exp<N>` from the repository root.
- `data/generate_synthetic.py`: Brownian path and terminal-condition
  generation; the only source of data for every experiment.
- `tests/`: pytest unit tests of `src/`.
- `results/`: JSON metrics, CSV tables and PNG figures written by the `exp`
  scripts, plus `results/RESULTS.md` with the comparison to the paper.

## Commands

```bash
pip install -r requirements.txt
python -m pytest                      # unit tests
python -m exp.exp1                    # .. python -m exp.exp6
bash reproduce.sh                     # tests + every experiment
```

Full paper-scale Monte Carlo configurations are available via `--full` on
each experiment script; the default (no flag) uses the reduced configuration
documented in `results/RESULTS.md`.

## Conventions

- Every experiment parameter lives in `src/config.py` as an upper-case
  module-level constant, never in a class, dataclass or dict, and never
  hard-coded inside `exp/*.py`.
- Data is produced only by `data/generate_synthetic.py`; no experiment script
  defines its own generator.
- Deviations from the paper (interpretations of unspecified details,
  reduced Monte Carlo sizes, substituted parameters) are recorded only in
  `results/RESULTS.md`, never in code comments or commit messages.
- After re-running an experiment, its command overwrites the corresponding
  files in `results/`; the numbers and verdicts in `results/RESULTS.md` are
  not updated automatically and must be checked against the new files and
  edited by hand.
- No API keys or market-data loaders are used anywhere in this repository.

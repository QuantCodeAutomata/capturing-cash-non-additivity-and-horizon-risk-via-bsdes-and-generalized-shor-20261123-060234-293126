#!/usr/bin/env bash
# Runs the test suite and every experiment of the replication in sequence.
set -uo pipefail

python -m pytest || exit 1

for i in 1 2 3 4 5 6; do
    python -m "exp.exp${i}" || exit 1
done

echo "Done. See results/RESULTS.md for the comparison with the paper."

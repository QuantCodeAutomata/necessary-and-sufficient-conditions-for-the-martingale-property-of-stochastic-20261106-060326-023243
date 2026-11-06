#!/usr/bin/env bash
# Runs the test suite followed by every experiment in its reduced (default)
# Monte Carlo configuration. Pass --full as the first argument to run every
# experiment at the paper-scale configuration from src/config.py instead.
set -uo pipefail

FULL_FLAG=""
if [[ "${1:-}" == "--full" ]]; then
    FULL_FLAG="--full"
fi

cd "$(dirname "${BASH_SOURCE[0]}")"

python -m pytest -m "not market_data" || exit 1

for i in 1 2 3 4 5 6 7 8 9 10; do
    python -m exp.exp"${i}" ${FULL_FLAG} || exit 1
done

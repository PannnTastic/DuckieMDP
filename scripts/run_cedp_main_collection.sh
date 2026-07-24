#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.
export PYTHONWARNINGS=ignore

exec .venv-sac/bin/python scripts/run_cedp_collect.py \
  --seeds 5 \
  --episodes-per-seed 1 \
  --max-decisions 200

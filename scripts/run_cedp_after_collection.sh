#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.
export PYTHONWARNINGS=ignore

while pgrep -f ".venv-sac/bin/python scripts/run_cedp_collect.py" >/dev/null; do
  sleep 30
done

test -f runs/explanations/cedp_v2/collection_summary.json

.venv-sac/bin/python scripts/run_cedp_segment.py
.venv-sac/bin/python scripts/run_cedp_discovery.py
.venv-sac/bin/python scripts/run_cedp_certify.py
.venv-sac/bin/python scripts/run_cedp_reconcile.py
.venv-sac/bin/python scripts/run_cedp_runtime.py
.venv-sac/bin/python scripts/run_cedp_ablation.py
.venv-sac/bin/python scripts/run_cedp_report.py

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv-sac/bin/python}"
CONFIG="${CONFIG:-configs/explainability/four_policy_reproducible.yaml}"
OUTPUT="${OUTPUT:-runs/explanations/four_policy_reproduction}"
MODE="${1:-verify}"

export PYTHONPATH="$ROOT"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"

verify() {
  "$PYTHON" scripts/verify_explanation_reproducibility.py
}

freeze_run() {
  "$PYTHON" scripts/run_cedp_freeze.py \
    --config "$CONFIG" \
    --output-dir "$OUTPUT"
}

postprocess() {

run_collect() {
  mkdir -p "$OUTPUT"
  "$PYTHON" -c 'import pyglet, runpy, sys
pyglet.options["headless"] = True
sys.argv = ["scripts/run_cedp_collect.py"] + sys.argv[1:]
runpy.run_path("scripts/run_cedp_collect.py", run_name="__main__")' \
    "$@" 2>>"$OUTPUT/simulator_warnings.log"
}
  "$PYTHON" scripts/run_cedp_segment.py --config "$CONFIG" --output-dir "$OUTPUT"
  "$PYTHON" scripts/run_cedp_discovery.py --config "$CONFIG" --output-dir "$OUTPUT"
  "$PYTHON" scripts/run_cedp_certify.py --config "$CONFIG" --output-dir "$OUTPUT"
  "$PYTHON" scripts/run_cedp_reconcile.py --config "$CONFIG" --output-dir "$OUTPUT"
  "$PYTHON" scripts/run_cedp_runtime.py --config "$CONFIG" --output-dir "$OUTPUT"
  "$PYTHON" scripts/run_cedp_ablation.py --config "$CONFIG" --output-dir "$OUTPUT"
  "$PYTHON" scripts/run_cedp_report.py --config "$CONFIG" --output-dir "$OUTPUT"
  "$PYTHON" scripts/generate_primitive_real_evidence.py \
    --input-dir "$OUTPUT" \
    --json-output "$OUTPUT/primitive_real_evidence.json" \
    --markdown-output "$OUTPUT/primitive_real_evidence.md"
}

case "$MODE" in
  verify)
    verify
    ;;
  test)
    verify
    "$PYTHON" -m pytest -q -p no:warnings \
      tests/test_explainability_m1.py \
      tests/test_explainability_primitives.py \
      tests/test_explainability_replay.py \
      tests/test_explainability_temporal_outcomes.py \
      tests/test_explainability_counterfactual.py \
      tests/test_explainability_metamorphic.py \
      tests/test_explainability_exact_q.py \
      tests/test_explainability_sac_internal.py \
      tests/test_explainability_m11_clustering.py \
      tests/test_explainability_sarsa.py \
      tests/test_explainability_eddp.py \
      tests/test_certified_primitives.py \
      tests/test_support_aware_verification.py
    ;;
  smoke)
    verify
    OUTPUT="${OUTPUT}_smoke"
    freeze_run
    # Five anchors per solver are collected. One explanation from each solver
    # is then branched, which exercises every adapter without a long run.
    for start in 0 5 10 15; do
      "$PYTHON" scripts/run_cedp_collect.py \
        --config "$CONFIG" \
        --output-dir "$OUTPUT" \
        --seeds 1 \
        --episodes-per-seed 1 \
        --max-decisions 5 \
        --start-index "$start" \
        --max-instances 1
    done
    "$PYTHON" - "$OUTPUT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
shards = sorted((root / "instance_shards").glob("*.json"))
solvers = sorted({
    json.loads(path.read_text(encoding="utf-8"))["solver"] for path in shards
})
expected = ["q_learning", "sac", "sarsa", "td3"]
if solvers != expected or len(shards) != 4:
    raise SystemExit("smoke mismatch: solvers=%r shards=%d" % (solvers, len(shards)))
print("four-policy explanation smoke PASS: %s" % ", ".join(solvers))
PY
    ;;
  postprocess)
    verify
    postprocess
    ;;
  full)
    verify
    freeze_run
    # Collection is resumable: completed instance shards are reused after an
    # interruption. Re-running this command is safe.
    "$PYTHON" scripts/run_cedp_collect.py \
      --config "$CONFIG" \
      --output-dir "$OUTPUT"
    postprocess
    ;;
  *)
    echo "usage: $0 {verify|test|smoke|postprocess|full}" >&2
    exit 2
    ;;
esac

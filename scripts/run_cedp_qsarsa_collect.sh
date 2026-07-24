#!/usr/bin/env bash
# Resumable batched collection for the tabular Q-learning + SARSA C-EDDP run
# (inherently gated via classify_duck). Fresh process per batch keeps WSL stable.
set -uo pipefail

cd "$(dirname "$0")/.."

CONFIG="configs/explainability/cedp_v2_qsarsa.yaml"
SHARD_DIR="runs/explanations/cedp_v2_qsarsa/instance_shards"
TARGET=2000
BATCH=40

export PYTHONPATH=.
export PYTHONWARNINGS=ignore
export LIBGL_ALWAYS_SOFTWARE=1

count() {
  find "${SHARD_DIR}" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l
}

while :; do
  start="$(count)"
  if [ "${start}" -ge "${TARGET}" ]; then
    break
  fi
  echo "CEDP qsarsa batch: ${start}/${TARGET}"
  .venv-sac/bin/python -c 'import pyglet, runpy, sys
pyglet.options["headless"] = True
sys.argv = ["run_cedp_collect", "--config", "configs/explainability/cedp_v2_qsarsa.yaml",
            "--start-index", sys.argv[1], "--max-instances", sys.argv[2]]
runpy.run_path("scripts/run_cedp_collect.py", run_name="__main__")' "${start}" "${BATCH}" \
    || sleep 5
done

echo "CEDP qsarsa collection complete: $(count)/${TARGET}"

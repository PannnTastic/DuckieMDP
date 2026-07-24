#!/usr/bin/env bash
# Resumable batched collection for the gated 2-policy (SAC-gated + TD3) C-EDDP
# run. Each batch is a fresh headless process so simulator/OpenGL resources are
# released between batches, keeping WSL stable.
set -uo pipefail

cd "$(dirname "$0")/.."

CONFIG="configs/explainability/cedp_v2_gated.yaml"
SHARD_DIR="runs/explanations/cedp_v2_gated/instance_shards"
TARGET=2000
BATCH=25

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
  echo "CEDP gated batch: ${start}/${TARGET}"
  .venv-sac/bin/python -c 'import pyglet, runpy, sys
pyglet.options["headless"] = True
sys.argv = ["run_cedp_collect", "--config", "configs/explainability/cedp_v2_gated.yaml",
            "--start-index", sys.argv[1], "--max-instances", sys.argv[2]]
runpy.run_path("scripts/run_cedp_collect.py", run_name="__main__")' "${start}" "${BATCH}" \
    || sleep 5
done

echo "CEDP gated collection complete: $(count)/${TARGET}"

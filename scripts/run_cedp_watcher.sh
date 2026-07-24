#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

OUTPUT_DIR="runs/explanations/cedp_v2"
SHARD_DIR="${OUTPUT_DIR}/instance_shards"
SUMMARY="${OUTPUT_DIR}/collection_summary.json"
WATCH_LOG="${OUTPUT_DIR}/watcher.log"
COLLECT_LOG="${OUTPUT_DIR}/collection.stdout.log"
POST_LOG="${OUTPUT_DIR}/postprocess.stdout.log"
DONE_MARKER="${OUTPUT_DIR}/pipeline_complete.marker"

mkdir -p "${OUTPUT_DIR}" "${SHARD_DIR}"

timestamp() {
  date --iso-8601=seconds
}

shard_count() {
  find "${SHARD_DIR}" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l
}

collection_complete() {
  test -f "${SUMMARY}" || return 1
  .venv-sac/bin/python -c \
    'import json, sys; p=json.load(open(sys.argv[1], encoding="utf-8")); sys.exit(0 if p.get("collection_complete") else 1)' \
    "${SUMMARY}"
}

log_status() {
  printf '%s %s\n' "$(timestamp)" "$1" | tee -a "${WATCH_LOG}"
}

attempt=0
while ! collection_complete; do
  attempt=$((attempt + 1))
  before="$(shard_count)"
  log_status "collection_attempt=${attempt} shards_before=${before}/3000"

  if bash scripts/run_cedp_main_collection.sh >>"${COLLECT_LOG}" 2>&1; then
    status=0
  else
    status=$?
  fi

  after="$(shard_count)"
  log_status "collection_exit=${status} shards_after=${after}/3000"

  if collection_complete; then
    break
  fi

  if test "${after}" -le "${before}"; then
    log_status "no_progress_detected retrying_in=30s"
    sleep 30
  else
    sleep 5
  fi
done

log_status "collection_complete=true starting_postprocess"
if bash scripts/run_cedp_after_collection.sh >>"${POST_LOG}" 2>&1; then
  printf '%s\n' "completed_at=$(timestamp)" >"${DONE_MARKER}"
  log_status "pipeline_complete=true"
else
  status=$?
  log_status "postprocess_exit=${status} pipeline_complete=false"
  exit "${status}"
fi

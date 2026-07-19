#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_DIR="runs/sac_full_50k_clockwise_wandb"
CONFIG="configs/sac_full_50k_clockwise_wandb.yaml"

until [[ -f "$RUN_DIR/sac_final.pt" ]]; do
  sleep 30
done

.venv-sac/bin/python -m src.select_sac_checkpoint \
  --config "$CONFIG" \
  --checkpoint-dir "$RUN_DIR/checkpoints" \
  --output "$RUN_DIR/sac_best.pt"

.venv-sac/bin/python -m src.evaluate_sac \
  --config "$CONFIG" \
  --checkpoint "$RUN_DIR/sac_best.pt" \
  --output "$RUN_DIR/evaluation_final.json"

.venv-sac/bin/python -m src.render_sac_multiview_video \
  --config "$CONFIG" \
  --checkpoint "$RUN_DIR/sac_best.pt" \
  --output "$RUN_DIR/sac_best_multiview_20fps.mp4" \
  --seed 10101 \
  --fps 20 \
  --max-steps 1500

echo "full_50k_clockwise_postprocess=complete"

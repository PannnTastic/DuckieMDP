#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Training writes the final checkpoint atomically. Waiting for this exact file
# prevents selection from racing with the active CUDA process.
until [[ -f runs/sac_lane/sac_final.pt ]]; do
  sleep 30
done

.venv-sac/bin/python -m src.select_sac_checkpoint \
  --config configs/sac_lane.yaml \
  --checkpoint-dir runs/sac_lane/checkpoints \
  --output runs/sac_lane/sac_best.pt

.venv-sac/bin/python -m src.evaluate_sac \
  --config configs/sac_lane.yaml \
  --checkpoint runs/sac_lane/sac_best.pt \
  --output runs/sac_lane/evaluation_final.json

.venv-sac/bin/python -m src.render_sac_video \
  --config configs/sac_lane.yaml \
  --checkpoint runs/sac_lane/sac_best.pt \
  --output runs/sac_lane/sac_best_camera_20fps.mp4 \
  --seed 10101 \
  --fps 20 \
  --view camera

.venv-sac/bin/python -m src.render_sac_multiview_video \
  --config configs/sac_lane.yaml \
  --checkpoint runs/sac_lane/sac_best.pt \
  --output runs/sac_lane/sac_best_multiview_20fps.mp4 \
  --seed 10101 \
  --fps 20 \
  --max-steps 1500

echo "lane_postprocess=complete"

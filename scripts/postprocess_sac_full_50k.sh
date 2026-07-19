#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

until [[ -f runs/sac_full_50k_wandb/sac_final.pt ]]; do
  sleep 30
done

.venv-sac/bin/python -m src.select_sac_checkpoint \
  --config configs/sac_full_50k_wandb.yaml \
  --checkpoint-dir runs/sac_full_50k_wandb/checkpoints \
  --output runs/sac_full_50k_wandb/sac_best.pt

.venv-sac/bin/python -m src.evaluate_sac \
  --config configs/sac_full_50k_wandb.yaml \
  --checkpoint runs/sac_full_50k_wandb/sac_best.pt \
  --output runs/sac_full_50k_wandb/evaluation_final.json

.venv-sac/bin/python -m src.render_sac_multiview_video \
  --config configs/sac_full_50k_wandb.yaml \
  --checkpoint runs/sac_full_50k_wandb/sac_best.pt \
  --output runs/sac_full_50k_wandb/sac_best_multiview_20fps.mp4 \
  --seed 10101 \
  --fps 20 \
  --max-steps 1500

echo "full_50k_postprocess=complete"

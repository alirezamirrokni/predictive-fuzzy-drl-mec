#!/usr/bin/env bash
set -euo pipefail

# Full two-scenario production run. Four PPO jobs receive at most five hours
# each, leaving time for predictor training, evaluation, and plots in ~24 h.
PYTHONPATH=. python -m experiments.run_full_pipeline \
  --scenario both \
  --train-predictors \
  --force-train \
  --total-timesteps 1048576 \
  --episodes 8 \
  --time-budget-hours-per-model 5

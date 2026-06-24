#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
SCENARIO_CONFIG=${1:-configs/scenario_a_600.yaml}
OUTPUT_DIR=${2:-data/results/scenario_a_600}
PREDICTOR_EPOCHS=${3:-10}
TOTAL_TIMESTEPS=${4:-2000}
MAX_EPISODE_TASKS=${5:-2000}
EPISODES=${6:-3}
MODULE=experiments.run_scenario_a
case "$SCENARIO_CONFIG" in
  *scenario_b*) MODULE=experiments.run_scenario_b ;;
  *) MODULE=experiments.run_scenario_a ;;
esac
python -m "$MODULE" \
  --config "$SCENARIO_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --train-predictors \
  --force-train \
  --predictor-epochs "$PREDICTOR_EPOCHS" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --episodes "$EPISODES" \
  --max-episode-tasks "$MAX_EPISODE_TASKS"
python -m plots.generate_all_plots --results-dir data/results --figures-dir reports/figures

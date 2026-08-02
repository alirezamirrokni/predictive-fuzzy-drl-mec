# Reproducible commands

Run all commands from the repository root with the virtual environment active.

## Scenario A

```bash
PYTHONPATH=. python -m experiments.run_scenario_a \
  --train-predictors \
  --force-train \
  --total-timesteps 1048576 \
  --episodes 8 \
  --time-budget-hours-per-model 5
```

## Scenario B

```bash
PYTHONPATH=. python -m experiments.run_scenario_b \
  --train-predictors \
  --force-train \
  --total-timesteps 1048576 \
  --episodes 8 \
  --time-budget-hours-per-model 5
```

## Predictor-only training/evaluation

```bash
PYTHONPATH=. python -m predictors.train_lstm \
  --scenario-config configs/scenario_a.yaml --restart \
  --checkpoint-path data/generated/checkpoints/lstm_scenario_a_last.pt \
  --best-checkpoint-path data/generated/checkpoints/lstm_scenario_a_best.pt
PYTHONPATH=. python -m predictors.train_gnn \
  --scenario-config configs/scenario_a.yaml --restart \
  --checkpoint-path data/generated/checkpoints/gnn_scenario_a_last.pt \
  --best-checkpoint-path data/generated/checkpoints/gnn_scenario_a_best.pt

PYTHONPATH=. python -m predictors.evaluate_predictors \
  --model lstm \
  --scenario-config configs/scenario_a.yaml \
  --checkpoint-path data/generated/checkpoints/lstm_scenario_a_best.pt \
  --include-scenario-name
```

The scenario pipeline uses scenario-specific checkpoint names automatically. Direct predictor commands must set matching output/checkpoint paths if both scenarios are trained.

## Resume an interrupted PPO job

```bash
PYTHONPATH=. python -m rl.train_ppo \
  --scenario-config configs/scenario_a.yaml \
  --predictor-type lstm \
  --predictor-checkpoint-path data/generated/checkpoints/lstm_scenario_a_best.pt \
  --predictor-model-config-path configs/model_lstm.yaml \
  --checkpoint-path data/generated/checkpoints/lstm_fuzzy_drl_scenario_a.pt \
  --best-checkpoint-path data/generated/checkpoints/lstm_fuzzy_drl_scenario_a_best.pt \
  --resume
```

## Regenerate plots

```bash
PYTHONPATH=. python -m plots.generate_all_plots \
  --results-dir data/results \
  --figures-dir reports/figures
```

## Tests

```bash
PYTHONPATH=. pytest -q
python -m compileall -q .
```

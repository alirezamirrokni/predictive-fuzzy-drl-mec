# Predictive-Fuzzy-DRL MEC

This repository implements a Mobile Edge Computing task-offloading framework with:

- MEC simulator with local, binary edge, and partial offloading
- heuristic baselines
- fuzzy controller and fuzzy reward weighting
- LSTM predictive layer
- GNN predictive baseline layer
- Gymnasium MEC environment
- pure PyTorch PPO training and evaluation
- scenario, ablation, sensitivity, and plotting scripts

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick tests

```bash
PYTHONPATH=. pytest -q
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="."
pytest -q
```

## Train predictors

```bash
PYTHONPATH=. python -m predictors.train_lstm --scenario-config configs/phase1_small.yaml --restart
PYTHONPATH=. python -m predictors.train_gnn --scenario-config configs/phase1_small.yaml --restart
```

## Evaluate predictors

```bash
PYTHONPATH=. python -m predictors.evaluate_predictors --model lstm --scenario-config configs/phase1_small.yaml
PYTHONPATH=. python -m predictors.evaluate_predictors --model gnn --scenario-config configs/phase1_small.yaml
```

## Train PPO without predictor

```bash
PYTHONPATH=. python -m rl.train_ppo \
  --scenario-config configs/phase1_small.yaml \
  --checkpoint-path data/generated/checkpoints/no_prediction_drl.pt \
  --no-prediction \
  --total-timesteps 10000 \
  --max-episode-tasks 400
```

## Train static-weight PPO baseline

```bash
PYTHONPATH=. python -m rl.train_ppo \
  --scenario-config configs/phase1_small.yaml \
  --checkpoint-path data/generated/checkpoints/static_weight_drl.pt \
  --static-weights \
  --no-prediction \
  --total-timesteps 10000 \
  --max-episode-tasks 400
```

## Train LSTM-Fuzzy-PPO

```bash
PYTHONPATH=. python -m rl.train_ppo \
  --scenario-config configs/phase1_small.yaml \
  --checkpoint-path data/generated/checkpoints/lstm_fuzzy_drl.pt \
  --predictor-type lstm \
  --predictor-checkpoint-path data/generated/checkpoints/lstm_best.pt \
  --predictor-model-config-path configs/model_lstm.yaml \
  --total-timesteps 10000 \
  --max-episode-tasks 400
```

## Train GNN-Fuzzy-PPO baseline

```bash
PYTHONPATH=. python -m rl.train_ppo \
  --scenario-config configs/phase1_small.yaml \
  --checkpoint-path data/generated/checkpoints/gnn_fuzzy_drl.pt \
  --predictor-type gnn \
  --predictor-checkpoint-path data/generated/checkpoints/gnn_best.pt \
  --predictor-model-config-path configs/model_gnn.yaml \
  --total-timesteps 10000 \
  --max-episode-tasks 400
```

## Evaluate a PPO checkpoint

```bash
PYTHONPATH=. python -m rl.evaluate_policy \
  --policy ppo \
  --scenario-config configs/phase1_small.yaml \
  --checkpoint-path data/generated/checkpoints/lstm_fuzzy_drl.pt \
  --output data/results/lstm_fuzzy_drl_eval.json \
  --metrics-csv data/results/lstm_fuzzy_drl_metrics.csv \
  --deterministic
```

## Run scenario scripts

```bash
PYTHONPATH=. python -m experiments.run_scenario_a --train-predictors --total-timesteps 10000 --max-episode-tasks 2000
PYTHONPATH=. python -m experiments.run_scenario_b --train-predictors --total-timesteps 10000 --max-episode-tasks 2000
```

## Run ablation and sensitivity

```bash
PYTHONPATH=. python -m experiments.run_ablation --total-timesteps 5000
PYTHONPATH=. python -m experiments.run_sensitivity
```

## Generate plots

```bash
PYTHONPATH=. python -m plots.generate_all_plots --results-dir data/results --figures-dir reports/figures
```

## Full pipeline

```bash
PYTHONPATH=. python -m experiments.run_full_pipeline --scenario both --train-predictors --total-timesteps 10000 --max-episode-tasks 2000
```

For quick debugging, reduce `--total-timesteps` and `--max-episode-tasks`.

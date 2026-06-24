# Colab commands for DRL-Fuzzy-Predictive MEC

The public ResearchGate/IEEE pages for RASOUL expose the paper identity and abstract, but not the full numerical experiment table. The project therefore centralizes all experiment variables in `configs/rasoul_reference.yaml` and uses the same field names in all scenario YAML files. If you obtain the IEEE PDF table, update those values and rerun the same commands.

## 1) Install

```bash
unzip predictive-fuzzy-drl-mec-complete.zip -d /content
cd /content/predictive-fuzzy-drl-mec-main
python -m pip install -U pip
python -m pip install -r requirements.txt
export PYTHONPATH=.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## 2) Sanity check

```bash
PYTHONPATH=. pytest -q
PYTHONPATH=. python main.py --config configs/phase1_small.yaml --policy greedy_latency --output-dir data/results/debug
PYTHONPATH=. python -m plots.generate_all_plots --results-dir data/results --figures-dir reports/figures
```

## 3) Scenario A: 600 IoT devices, 100 heterogeneous edge servers, 200 tasks/device

This is the runtime-friendly lower-bound configuration for scenario A: 120,000 tasks.

```bash
PYTHONPATH=. python -m experiments.run_scenario_a \
  --config configs/scenario_a_600.yaml \
  --output-dir data/results/scenario_a_600 \
  --train-predictors \
  --force-train \
  --predictor-epochs 10 \
  --total-timesteps 2000 \
  --episodes 3 \
  --max-episode-tasks 2000
```

## 4) Scenario A upper-bound variant: 1000 IoT devices, 100 edge servers, 200 tasks/device

This generates 200,000 tasks and is heavier.

```bash
PYTHONPATH=. python -m experiments.run_scenario_a \
  --config configs/scenario_a_1000.yaml \
  --output-dir data/results/scenario_a_1000 \
  --train-predictors \
  --force-train \
  --predictor-epochs 10 \
  --total-timesteps 2000 \
  --episodes 3 \
  --max-episode-tasks 2000
```

## 5) Scenario B: 300 IoT devices, 30 heterogeneous edge servers, 200 tasks/device

This is the runtime-friendly lower-bound configuration for scenario B: 60,000 tasks.

```bash
PYTHONPATH=. python -m experiments.run_scenario_b \
  --config configs/scenario_b_30.yaml \
  --output-dir data/results/scenario_b_30 \
  --train-predictors \
  --force-train \
  --predictor-epochs 10 \
  --total-timesteps 2000 \
  --episodes 3 \
  --max-episode-tasks 2000
```

## 6) Scenario B upper-bound variant: 300 IoT devices, 100 edge servers, 200 tasks/device

```bash
PYTHONPATH=. python -m experiments.run_scenario_b \
  --config configs/scenario_b_100.yaml \
  --output-dir data/results/scenario_b_100 \
  --train-predictors \
  --force-train \
  --predictor-epochs 10 \
  --total-timesteps 2000 \
  --episodes 3 \
  --max-episode-tasks 2000
```

## 7) Generate every required visualization

```bash
PYTHONPATH=. python -m plots.generate_all_plots \
  --results-dir data/results \
  --figures-dir reports/figures
```

Generated figures include separate plots for:

- system energy
- latency
- success ratio
- reliability
- task execution modes
- task input size, output size, CPU demand, deadline, and priority
- online overhead per task
- communication/queue overhead until task exit
- learning overhead in task-equivalent steps
- LSTM-Fuzzy-DRL vs GNN-Fuzzy-DRL baseline comparisons

## 8) Individual training commands

### LSTM predictor

```bash
PYTHONPATH=. python -m predictors.train_lstm \
  --scenario-config configs/scenario_a_600.yaml \
  --epochs 10 \
  --restart \
  --checkpoint-path data/generated/checkpoints/lstm_scenario_a_600_last.pt \
  --best-checkpoint-path data/generated/checkpoints/lstm_scenario_a_600_best.pt \
  --log-path data/results/scenario_a_600/lstm_training_log.csv \
  --result-path data/results/scenario_a_600/lstm_training_summary.json
```

### GNN predictor baseline

```bash
PYTHONPATH=. python -m predictors.train_gnn \
  --scenario-config configs/scenario_a_600.yaml \
  --epochs 10 \
  --restart \
  --checkpoint-path data/generated/checkpoints/gnn_scenario_a_600_last.pt \
  --best-checkpoint-path data/generated/checkpoints/gnn_scenario_a_600_best.pt \
  --log-path data/results/scenario_a_600/gnn_training_log.csv \
  --result-path data/results/scenario_a_600/gnn_training_summary.json
```

### LSTM-Fuzzy-DRL PPO

```bash
PYTHONPATH=. python -m rl.train_ppo \
  --scenario-config configs/scenario_a_600.yaml \
  --checkpoint-path data/generated/checkpoints/lstm_fuzzy_drl_scenario_a_600.pt \
  --predictor-type lstm \
  --predictor-checkpoint-path data/generated/checkpoints/lstm_scenario_a_600_best.pt \
  --predictor-model-config-path configs/model_lstm.yaml \
  --total-timesteps 2000 \
  --max-episode-tasks 2000 \
  --log-path data/results/scenario_a_600/lstm_fuzzy_drl_train_log.csv \
  --result-path data/results/scenario_a_600/lstm_fuzzy_drl_train_summary.json
```

### GNN-Fuzzy-DRL PPO baseline

```bash
PYTHONPATH=. python -m rl.train_ppo \
  --scenario-config configs/scenario_a_600.yaml \
  --checkpoint-path data/generated/checkpoints/gnn_fuzzy_drl_scenario_a_600.pt \
  --predictor-type gnn \
  --predictor-checkpoint-path data/generated/checkpoints/gnn_scenario_a_600_best.pt \
  --predictor-model-config-path configs/model_gnn.yaml \
  --total-timesteps 2000 \
  --max-episode-tasks 2000 \
  --log-path data/results/scenario_a_600/gnn_fuzzy_drl_train_log.csv \
  --result-path data/results/scenario_a_600/gnn_fuzzy_drl_train_summary.json
```

## 9) Heuristic baseline commands

```bash
for policy in local_only random greedy_latency greedy_energy; do
  PYTHONPATH=. python main.py \
    --config configs/scenario_a_600.yaml \
    --policy $policy \
    --output-dir data/results/scenario_a_600
 done
```

## 10) Fast debug run

Use this before full training.

```bash
PYTHONPATH=. python -m experiments.run_scenario_a \
  --config configs/phase1_small.yaml \
  --output-dir data/results/debug_small \
  --train-predictors \
  --force-train \
  --predictor-epochs 1 \
  --total-timesteps 200 \
  --episodes 1 \
  --max-episode-tasks 200

PYTHONPATH=. python -m plots.generate_all_plots --results-dir data/results --figures-dir reports/figures
```

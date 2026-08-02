# DRL–Fuzzy–Predictive MEC (final corrected implementation)

This repository implements the requested three-layer framework with two fair predictive variants:

- **LSTM + fixed fuzzy controller + feed-forward PPO** (proposed method)
- **GNN/GAT + the same fixed fuzzy controller + the same feed-forward PPO** (predictive baseline)
- **Local only**
- **Random mixed offloading**

Only these four methods appear in the production experiment and comparison plots. Greedy latency is used only to generate supervised predictor traces; it is not an evaluation baseline.

## What is fixed

- Scenario A samples one IoT population size uniformly from **600–1000** per replication and uses **100** heterogeneous edge servers.
- Scenario B uses **300** IoT devices and samples one edge-server population size uniformly from **30–100** per replication.
- Every device/server/task attribute is sampled independently from the full documented interval. A run does not create one separate experiment for every value.
- Every sampled IoT device receives exactly **200 tasks**.
- The production clock is EdgeSimPy v1.1.0. Production fails if EdgeSimPy is absent; only `configs/debug.yaml` permits the internal test clock.
- Offloading is always `mixed`: local, full edge, or partial. For a partial decision, the task's sampled `local_fraction` is local and `1-local_fraction` is offloaded.
- LSTM and GNN use identical tasks, seeds, top-16 candidates, fuzzy rules, PPO architecture/budget, and evaluation seeds.
- Predictor loading is strict. Production evaluation never replaces a missing/broken predictor with current-state features.
- Predictor validation is chronological, with a 16-window separation gap and early stopping on lowest validation MSE.
- PPO saves a recoverable checkpoint after every update and selects a separate best checkpoint by a fixed validation seed, never by test results.
- PPO state includes four forecast features and four fuzzy weights. PPO itself is feed-forward, not recurrent.
- GAE episode masking is corrected; reset seeds now actually change the sampled scenario.
- Outputs include energy, latency, success ratio, average reliability, reliability-target satisfaction, task modes/specifications, and mean/maximum overhead.

## Install

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

EdgeSimPy is pinned to its stable v1.1.0 Git tag in `requirements.txt`.

## Fast verification

The debug configuration keeps the same physical ranges but deliberately uses only 5 devices, 3 servers, and 4 tasks/device:

```bash
PYTHONPATH=. pytest -q
PYTHONPATH=. python main.py --config configs/debug.yaml --policy random --output-dir data/results/debug
```

Debug results are not valid project results.

## Full production run (both scenarios, approximately one day)

```bash
bash scripts/run_colab_project.sh
```

The script trains both predictors and all four scenario/model PPO combinations. Each PPO job is capped at five wall-clock hours and at 1,048,576 environment steps, so the complete workflow targets roughly 24 hours; exact duration depends on CPU/GPU and EdgeSimPy overhead. Predictor early stopping can finish before 200 epochs.

Equivalent command:

```bash
PYTHONPATH=. python -m experiments.run_full_pipeline \
  --scenario both \
  --train-predictors \
  --force-train \
  --total-timesteps 1048576 \
  --episodes 8 \
  --time-budget-hours-per-model 5
```

Do not set `--max-episode-tasks` for final results. That option exists only for smoke runs.

## Short smoke run of the learning pipeline

After installing PyTorch, Gymnasium, and EdgeSimPy:

```bash
PYTHONPATH=. python -m predictors.train_lstm --scenario-config configs/debug.yaml --epochs 2 --restart
PYTHONPATH=. python -m predictors.train_gnn --scenario-config configs/debug.yaml --epochs 2 --restart
```

For PPO smoke tests, point the commands in `COMMANDS.md` to `configs/debug.yaml`, use `--total-timesteps 2048`, and use the debug predictor checkpoints. Never mix debug checkpoints with production evaluation.

## Outputs

For each scenario, the pipeline writes:

- `data/results/scenario_*/{lstm_fuzzy_drl,gnn_fuzzy_drl,local_only,random}_eval.json`
- one per-task metrics CSV per method
- predictor/PPO training logs and summaries
- `reports/figures/scenario_*/comparison/` with separate requested comparison charts
- per-method task-mode and task-specification charts
- mean online overhead, maximum online overhead, per-task overhead, and learning-overhead charts


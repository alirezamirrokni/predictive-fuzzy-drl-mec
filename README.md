# Predictive-Fuzzy-DRL for Resource Management in Mobile Edge Computing

This repository is organized for the full project. The current implementation contains Phase 1 only: a Mobile Edge Computing simulator.

## Phase 1

Implemented components:

- IoT devices
- heterogeneous edge servers
- task generation
- wireless channel model
- device mobility
- local execution
- binary offloading
- partial offloading
- random baseline
- local-only baseline
- greedy-latency baseline
- greedy-energy baseline
- metric logging
- scenario A and B configs
- plotting scripts
- tests

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Test

```bash
pytest
```

## Run a small Phase 1 demo

```bash
python main.py --config configs/phase1_small.yaml --policy greedy_latency --output-dir data/results
```

## Run full scenarios

```bash
python experiments/run_scenario_a.py
python experiments/run_scenario_b.py
```

## Plot outputs

```bash
python plots/plot_energy.py --input data/results/phase1_small_greedy_latency_metrics.csv --output reports/figures/energy.png
python plots/plot_latency.py --input data/results/phase1_small_greedy_latency_metrics.csv --output reports/figures/latency.png
python plots/plot_success.py --input data/results/phase1_small_greedy_latency_metrics.csv --output reports/figures/success.png
python plots/plot_reliability.py --input data/results/phase1_small_greedy_latency_metrics.csv --output reports/figures/reliability.png
python plots/plot_task_stats.py --input data/results/phase1_small_greedy_latency_metrics.csv --output reports/figures/task_modes.png
```

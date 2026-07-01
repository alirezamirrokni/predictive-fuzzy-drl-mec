# DRL-Fuzzy-Predictive MEC — Kaggle / Colab Commands


## 1) Kaggle setup used in `iotlab.ipynb`

Use this when the project is attached as a Kaggle input dataset.

```python
import os, shutil

SRC = "/kaggle/input/datasets/asalmeskin/predictive-fuzzy-drl-mec-complete/predictive-fuzzy-drl-mec-main"
DST = "/kaggle/working/predictive-fuzzy-drl-mec-main"

assert os.path.exists(SRC), f"Source directory not found: {SRC}"

if os.path.exists(DST):
    shutil.rmtree(DST)

shutil.copytree(SRC, DST)

print("Copied to:", DST)
print(os.listdir(DST)[:20])
```

Then install requirements and set environment variables:

```python
%cd /kaggle/working/predictive-fuzzy-drl-mec-main

!python -m pip install -q -U pip
!python -m pip install -q -r requirements.txt

%env PYTHONPATH=/kaggle/working/predictive-fuzzy-drl-mec-main
%env OMP_NUM_THREADS=1
%env MKL_NUM_THREADS=1

!python -c "import torch, gymnasium, yaml, numpy; print('setup ok')"
```

---

## 2) Google Colab setup alternative

Use this if you upload the ZIP manually to Colab instead of running on Kaggle.

```python
from google.colab import files
files.upload()  # upload predictive-fuzzy-drl-mec-complete.zip or predictive-fuzzy-drl-mec-fixed.zip
```

```bash
unzip predictive-fuzzy-drl-mec-complete.zip -d /content
cd /content/predictive-fuzzy-drl-mec-main
python -m pip install -U pip
python -m pip install -r requirements.txt
export PYTHONPATH=/content/predictive-fuzzy-drl-mec-main
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -c "import torch, gymnasium, yaml, numpy; print('setup ok')"
```

If your ZIP name is `predictive-fuzzy-drl-mec-fixed.zip`, replace the unzip command accordingly.

---

## 3) Sanity check

This is the same validation sequence used in the notebook. It should run the unit tests and a small greedy-latency debug simulation.

```bash
PYTHONPATH=/kaggle/working/predictive-fuzzy-drl-mec-main pytest -q

PYTHONPATH=/kaggle/working/predictive-fuzzy-drl-mec-main python main.py \
  --config configs/phase1_small.yaml \
  --policy greedy_latency \
  --output-dir data/results/debug
```

Expected notebook result: `6 passed`, followed by output files similar to:

```text
data/results/debug/phase1_small_greedy_latency_metrics.csv
data/results/debug/phase1_small_greedy_latency_summary.json
```

---

## 4) Scenario A: 600 IoT devices, 100 heterogeneous edge servers, 200 tasks/device

This is the main Scenario A run from `iotlab.ipynb`. It generates 120,000 tasks and writes outputs to `data/results/scenario_a_600_final`.

```bash
PYTHONPATH=/kaggle/working/predictive-fuzzy-drl-mec-main python -m experiments.run_scenario_a \
  --config configs/scenario_a_600.yaml \
  --output-dir data/results/scenario_a_600_final \
  --train-predictors \
  --force-train \
  --predictor-epochs 50 \
  --total-timesteps 20000 \
  --episodes 5 \
  --max-episode-tasks 10000
```

Typical generated files include:

```text
data/results/scenario_a_600_final/scenario_a_600_local_only_metrics.csv
data/results/scenario_a_600_final/scenario_a_600_random_metrics.csv
data/results/scenario_a_600_final/scenario_a_600_greedy_latency_metrics.csv
data/results/scenario_a_600_final/scenario_a_600_greedy_energy_metrics.csv
data/results/scenario_a_600_final/lstm_fuzzy_drl_metrics.csv
data/results/scenario_a_600_final/gnn_fuzzy_drl_metrics.csv
data/results/scenario_a_600_final/no_prediction_drl_metrics.csv
data/results/scenario_a_600_final/static_weight_drl_metrics.csv
```

---

## 5) Scenario B: 300 IoT devices, 30 heterogeneous edge servers, 200 tasks/device

This is the main Scenario B run from `iotlab.ipynb`. It generates 60,000 tasks and writes outputs to `data/results/scenario_b_30`.

```bash
PYTHONPATH=/kaggle/working/predictive-fuzzy-drl-mec-main python -m experiments.run_scenario_b \
  --config configs/scenario_b_30.yaml \
  --output-dir data/results/scenario_b_30 \
  --train-predictors \
  --force-train \
  --predictor-epochs 50 \
  --total-timesteps 20000 \
  --episodes 5 \
  --max-episode-tasks 10000
```

Typical generated files include:

```text
data/results/scenario_b_30/scenario_b_30_local_only_metrics.csv
data/results/scenario_b_30/scenario_b_30_random_metrics.csv
data/results/scenario_b_30/scenario_b_30_greedy_latency_metrics.csv
data/results/scenario_b_30/scenario_b_30_greedy_energy_metrics.csv
data/results/scenario_b_30/lstm_fuzzy_drl_metrics.csv
data/results/scenario_b_30/gnn_fuzzy_drl_metrics.csv
data/results/scenario_b_30/no_prediction_drl_metrics.csv
data/results/scenario_b_30/static_weight_drl_metrics.csv
```

---

## 6) Generate every required visualization

Run this after Scenario A and/or Scenario B finishes.

```bash
PYTHONPATH=/kaggle/working/predictive-fuzzy-drl-mec-main python -m plots.generate_all_plots \
  --results-dir data/results \
  --figures-dir reports/figures
```

If task-specification histograms are slow or Matplotlib appears to hang while saving many PNG files, run the faster version:

```bash
PYTHONPATH=/kaggle/working/predictive-fuzzy-drl-mec-main python -m plots.generate_all_plots \
  --results-dir data/results \
  --figures-dir reports/figures \
  --skip-task-specs
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

The output folder should look like this:

```text
reports/figures/scenario_a_600_final/
reports/figures/scenario_b_30/
```

---

## 7) Zip results, figures, and checkpoints on Kaggle

Use this at the end of the notebook to create a downloadable archive.

```bash
zip -r /kaggle/working/mec_results_and_figures.zip \
  data/results \
  reports/figures \
  data/generated/checkpoints \
  -x "*.pyc" "__pycache__/*"
```

Kaggle output file:

```text
/kaggle/working/mec_results_and_figures.zip
```





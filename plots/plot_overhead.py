from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from plots._common import ensure_parent, read_metrics


ONLINE_COLUMNS = [
    "online_overhead_s",
    "policy_overhead_s",
    "drl_overhead_s",
    "prediction_overhead_s",
    "fuzzy_overhead_s",
    "simulator_apply_overhead_s",
]
EXECUTION_COLUMNS = ["execution_overhead_s", "tx_time_s", "rx_time_s", "queue_delay_s"]


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _values(rows: List[Dict[str, str]], column: str) -> List[float]:
    return [_to_float(row.get(column, 0.0)) for row in rows if row.get(column, "") != ""]


def _label_from_path(path: Path) -> str:
    return path.stem.replace("_metrics", "")


def plot_per_task(input_path: str | Path, output_dir: str | Path) -> list[Path]:
    path = Path(input_path)
    rows = read_metrics(path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    for column, ylabel in [
        ("online_overhead_s", "Online overhead per task (s)"),
        ("execution_overhead_s", "Communication/queue overhead per task (s)"),
    ]:
        values = _values(rows, column)
        if not values:
            continue
        out = output / f"{path.stem}_{column}.png"
        plt.figure(figsize=(7, 4))
        plt.plot(range(1, len(values) + 1), values)
        plt.xlabel("Task index")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(ensure_parent(out), dpi=200)
        plt.close()
        produced.append(out)
    return produced


def plot_overhead_breakdown(input_path: str | Path, output_path: str | Path) -> Path | None:
    rows = read_metrics(input_path)
    means: list[float] = []
    labels: list[str] = []
    for column in ONLINE_COLUMNS:
        vals = _values(rows, column)
        if vals and any(v != 0.0 for v in vals):
            labels.append(column.replace("_s", ""))
            means.append(float(np.mean(vals)))
    if not means:
        return None
    out = Path(output_path)
    plt.figure(figsize=(max(7, len(labels) * 0.8), 4))
    plt.bar(labels, means)
    plt.ylabel("Mean online overhead (s/task)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(ensure_parent(out), dpi=200)
    plt.close()
    return out


def _csv_metric(path: Path, column: str) -> tuple[float, float]:
    rows = read_metrics(path)
    vals = _values(rows, column)
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.sum(vals))


def plot_online_comparison(results_dir: str | Path, output_path: str | Path) -> Path | None:
    results = Path(results_dir)
    items: list[tuple[str, float]] = []
    for csv_path in sorted(results.rglob("*_metrics.csv")):
        mean_value, _ = _csv_metric(csv_path, "online_overhead_s")
        if mean_value > 0:
            items.append((_label_from_path(csv_path), mean_value))
    if not items:
        return None
    labels, values = zip(*items)
    out = Path(output_path)
    plt.figure(figsize=(max(8, len(labels) * 0.8), 4))
    plt.bar(labels, values)
    plt.ylabel("Mean online overhead (s/task)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(ensure_parent(out), dpi=200)
    plt.close()
    return out


def _learning_equivalent_from_json(path: Path) -> float:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0.0
    if isinstance(data, dict):
        for key in ["learning_task_equivalent", "global_step"]:
            if key in data:
                return _to_float(data[key])
        metadata = data.get("checkpoint_metadata") or data.get("metadata") or {}
        if isinstance(metadata, dict):
            for key in ["learning_task_equivalent", "global_step"]:
                if key in metadata:
                    return _to_float(metadata[key])
    return 0.0


def plot_learning_overhead(results_dir: str | Path, output_path: str | Path) -> Path | None:
    results = Path(results_dir)
    items: list[tuple[str, float]] = []
    for path in sorted(results.rglob("*.json")):
        value = _learning_equivalent_from_json(path)
        if value > 0:
            label = path.stem.replace("_train_summary", "").replace("_training_summary", "")
            items.append((label, value))
    if not items:
        return None
    labels, values = zip(*items)
    out = Path(output_path)
    plt.figure(figsize=(max(8, len(labels) * 0.8), 4))
    plt.bar(labels, values)
    plt.ylabel("Learning overhead (task-equivalent steps)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(ensure_parent(out), dpi=200)
    plt.close()
    return out


def generate(results_dir: str, figures_dir: str) -> list[Path]:
    results = Path(results_dir)
    figures = Path(figures_dir)
    produced: list[Path] = []
    for csv_path in sorted(results.rglob("*_metrics.csv")):
        rel_dir = csv_path.parent.relative_to(results)
        out_dir = figures / rel_dir / "overhead"
        produced.extend(plot_per_task(csv_path, out_dir))
        breakdown = plot_overhead_breakdown(csv_path, out_dir / f"{csv_path.stem}_online_breakdown.png")
        if breakdown is not None:
            produced.append(breakdown)
    item = plot_online_comparison(results, figures / "overhead_online_comparison.png")
    if item is not None:
        produced.append(item)
    item = plot_learning_overhead(results, figures / "overhead_learning_task_equivalent.png")
    if item is not None:
        produced.append(item)
    return produced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--figures-dir", default="reports/figures")
    args = parser.parse_args()
    for item in generate(args.results_dir, args.figures_dir):
        print(item)


if __name__ == "__main__":
    main()

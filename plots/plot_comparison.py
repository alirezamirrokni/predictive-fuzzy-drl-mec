from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


METHOD_LABELS = {
    "lstm_fuzzy_drl": "LSTM + Fuzzy PPO",
    "gnn_fuzzy_drl": "GNN + Fuzzy PPO",
    "local_only": "Local only",
    "random": "Random",
}
METHOD_ORDER = list(METHOD_LABELS)


def load_summary(path: str | Path) -> Dict[str, float]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if "summary" in data and isinstance(data["summary"], dict):
        return data["summary"]
    if "simulator_summary_last_episode" in data and isinstance(data["simulator_summary_last_episode"], dict):
        return data["simulator_summary_last_episode"]
    return data


def discover(input_dir: str | Path) -> List[Tuple[str, Dict[str, float]]]:
    rows = []
    for path in sorted(Path(input_dir).glob("*_eval.json")):
        if path.name.endswith("all_summaries.json"):
            continue
        try:
            summary = load_summary(path)
        except Exception:
            continue
        if not any(key in summary for key in ["success_ratio", "average_latency_s", "average_energy_j", "average_reliability"]):
            continue
        key = path.stem.replace("_eval", "")
        if key in METHOD_LABELS:
            rows.append((key, summary))
    return sorted(rows, key=lambda row: METHOD_ORDER.index(row[0]))


def plot_metric(rows: List[Tuple[str, Dict[str, float]]], metric: str, ylabel: str, output_path: str | Path) -> None:
    labels = [METHOD_LABELS[label] for label, _ in rows]
    values = [float(summary.get(metric, 0.0)) for _, summary in rows]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(7, len(labels) * 0.9), 4))
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def generate(input_dir: str, output_dir: str) -> None:
    rows = discover(input_dir)
    if not rows:
        raise ValueError(f"no summary JSON files found in {input_dir}")
    output = Path(output_dir)
    plot_metric(rows, "total_energy_j", "Total system energy (J)", output / "comparison_energy.png")
    plot_metric(rows, "average_latency_s", "Average latency (s)", output / "comparison_latency.png")
    plot_metric(rows, "success_ratio", "Success ratio", output / "comparison_success_ratio.png")
    plot_metric(rows, "average_reliability", "Average reliability", output / "comparison_reliability.png")
    plot_metric(rows, "reliability_satisfaction_ratio", "Reliability target satisfaction ratio", output / "comparison_reliability_target.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="reports/figures")
    args = parser.parse_args()
    generate(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()

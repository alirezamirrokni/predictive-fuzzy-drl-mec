from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


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
    for path in sorted(Path(input_dir).glob("*.json")):
        if path.name.endswith("all_summaries.json"):
            continue
        try:
            summary = load_summary(path)
        except Exception:
            continue
        if not any(key in summary for key in ["success_ratio", "average_latency_s", "average_energy_j", "average_reliability"]):
            continue
        label = path.stem.replace("_summary", "").replace("_eval", "")
        rows.append((label, summary))
    return rows


def plot_metric(rows: List[Tuple[str, Dict[str, float]]], metric: str, ylabel: str, output_path: str | Path) -> None:
    labels = [label for label, _ in rows]
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
    plot_metric(rows, "average_energy_j", "Average energy (J)", output / "comparison_energy.png")
    plot_metric(rows, "average_latency_s", "Average latency (s)", output / "comparison_latency.png")
    plot_metric(rows, "success_ratio", "Success ratio", output / "comparison_success_ratio.png")
    plot_metric(rows, "average_reliability", "Average reliability", output / "comparison_reliability.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="reports/figures")
    args = parser.parse_args()
    generate(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))

from plots._common import ensure_parent, read_metrics


TASK_COLUMNS = {
    "data_size_mb": ("Task input size (MB)", "task_input_size"),
    "output_size_mb": ("Task output size (MB)", "task_output_size"),
    "cpu_cycles_mi": ("Task CPU demand (MI)", "task_cpu_cycles"),
    "deadline_s": ("Task deadline (s)", "task_deadline"),
    "task_local_fraction": ("Fraction processed locally", "task_local_fraction"),
    "release_interval_s": ("Task release interval (s)", "task_release_interval"),
    "reliability_target": ("IoT reliability target", "device_reliability_target"),
}


def _numeric_values(rows: list[dict[str, str]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(column, "")
        if raw == "":
            continue
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def plot(input_path: str, output_dir: str) -> list[Path]:
    rows = read_metrics(input_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    stem = Path(input_path).stem
    for column, (xlabel, suffix) in TASK_COLUMNS.items():
        values = _numeric_values(rows, column)
        if not values:
            continue
        path = output / f"{stem}_{suffix}.png"
        plt.figure(figsize=(6, 4))
        bins = 40
        plt.hist(values, bins=bins)
        plt.xlabel(xlabel)
        plt.ylabel("Tasks")
        plt.tight_layout()
        plt.savefig(ensure_parent(path), dpi=200)
        plt.close()
        produced.append(path)
    return produced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="reports/figures/task_specs")
    args = parser.parse_args()
    for item in plot(args.input, args.output_dir):
        print(item)


if __name__ == "__main__":
    main()

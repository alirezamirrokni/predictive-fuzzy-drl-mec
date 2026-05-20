from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))

from plots._common import ensure_parent, read_metrics


def plot(input_path: str, output_path: str) -> None:
    rows = read_metrics(input_path)
    counter = Counter(row["mode"] for row in rows)
    labels = list(counter.keys())
    values = [counter[label] for label in labels]
    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Tasks")
    plt.tight_layout()
    plt.savefig(ensure_parent(output_path), dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="reports/figures/task_modes.png")
    args = parser.parse_args()
    plot(args.input, args.output)


if __name__ == "__main__":
    main()

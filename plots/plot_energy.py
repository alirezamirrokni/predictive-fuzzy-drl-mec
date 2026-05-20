from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))

from plots._common import ensure_parent, read_metrics


def plot(input_path: str, output_path: str) -> None:
    rows = read_metrics(input_path)
    values = [float(row["energy_j"]) for row in rows]
    plt.figure()
    plt.hist(values, bins=40)
    plt.xlabel("Energy per task (J)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(ensure_parent(output_path), dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="reports/figures/energy.png")
    args = parser.parse_args()
    plot(args.input, args.output)


if __name__ == "__main__":
    main()

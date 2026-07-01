from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import Counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))

from plots._common import ensure_parent, read_metrics


def _success_to_int(value: Any) -> int:
    """Accept success values stored as 1/0, 1.0/0.0, True/False, yes/no."""
    if isinstance(value, bool):
        return int(value)

    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "t", "yes", "y", "successful", "success"}:
        return 1
    if text in {"0", "0.0", "false", "f", "no", "n", "failed", "fail", ""}:
        return 0

    try:
        return 1 if float(text) != 0.0 else 0
    except ValueError as exc:
        raise ValueError(f"cannot parse success value {value!r}") from exc


def plot(input_path: str, output_path: str) -> None:
    rows = read_metrics(input_path)
    counter = Counter(_success_to_int(row.get("success", 0)) for row in rows)
    labels = ["failed", "successful"]
    values = [counter[0], counter[1]]
    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Tasks")
    plt.tight_layout()
    plt.savefig(ensure_parent(output_path), dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="reports/figures/success.png")
    args = parser.parse_args()
    plot(args.input, args.output)


if __name__ == "__main__":
    main()

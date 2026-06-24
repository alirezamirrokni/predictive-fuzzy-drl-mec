from __future__ import annotations

import argparse
from pathlib import Path

from plots.plot_comparison import generate as generate_comparison
from plots.plot_energy import plot as plot_energy
from plots.plot_latency import plot as plot_latency
from plots.plot_reliability import plot as plot_reliability
from plots.plot_success import plot as plot_success
from plots.plot_task_stats import plot as plot_task_stats
from plots.plot_task_specs import plot as plot_task_specs
from plots.plot_overhead import generate as generate_overhead


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--figures-dir", default="reports/figures")
    args = parser.parse_args()
    results = Path(args.results_dir)
    figures = Path(args.figures_dir)
    for csv_path in results.rglob("*_metrics.csv"):
        stem = csv_path.stem
        out_dir = figures / csv_path.parent.relative_to(results)
        try:
            plot_energy(str(csv_path), str(out_dir / f"{stem}_energy.png"))
            plot_latency(str(csv_path), str(out_dir / f"{stem}_latency.png"))
            plot_reliability(str(csv_path), str(out_dir / f"{stem}_reliability.png"))
            plot_success(str(csv_path), str(out_dir / f"{stem}_success.png"))
            plot_task_stats(str(csv_path), str(out_dir / f"{stem}_task_modes.png"))
            plot_task_specs(str(csv_path), str(out_dir / "task_specs"))
        except Exception as exc:
            print(f"skip {csv_path}: {exc}")
    generate_overhead(str(results), str(figures))
    for directory in [item for item in results.rglob("*") if item.is_dir()] + [results]:
        if list(directory.glob("*.json")):
            try:
                rel = directory.relative_to(results)
                generate_comparison(str(directory), str(figures / rel / "comparison"))
            except Exception as exc:
                print(f"skip comparison {directory}: {exc}")


if __name__ == "__main__":
    main()

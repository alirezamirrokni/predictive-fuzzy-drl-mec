from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib

# Use a non-interactive backend. This avoids Windows GUI/backend hangs while saving PNGs.
matplotlib.use("Agg")

from plots.plot_comparison import generate as generate_comparison
from plots.plot_energy import plot as plot_energy
from plots.plot_latency import plot as plot_latency
from plots.plot_reliability import plot as plot_reliability
from plots.plot_success import plot as plot_success
from plots.plot_task_stats import plot as plot_task_stats
from plots.plot_task_specs import plot as plot_task_specs
from plots.plot_overhead import generate as generate_overhead


def _safe_run(label: str, func: Callable[[], object]) -> None:
    try:
        func()
    except Exception as exc:
        print(f"skip {label}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument(
        "--skip-task-specs",
        action="store_true",
        help="Skip task specification histograms if you only need performance plots.",
    )
    args = parser.parse_args()

    results = Path(args.results_dir)
    figures = Path(args.figures_dir)
    figures.mkdir(parents=True, exist_ok=True)

    if not results.exists():
        raise FileNotFoundError(f"results directory does not exist: {results}")

    for csv_path in sorted(results.rglob("*_metrics.csv")):
        stem = csv_path.stem
        out_dir = figures / csv_path.parent.relative_to(results)
        print(f"plotting {csv_path}")

        _safe_run(f"energy {csv_path}", lambda: plot_energy(str(csv_path), str(out_dir / f"{stem}_energy.png")))
        _safe_run(f"latency {csv_path}", lambda: plot_latency(str(csv_path), str(out_dir / f"{stem}_latency.png")))
        _safe_run(
            f"reliability {csv_path}",
            lambda: plot_reliability(str(csv_path), str(out_dir / f"{stem}_reliability.png")),
        )
        _safe_run(f"success {csv_path}", lambda: plot_success(str(csv_path), str(out_dir / f"{stem}_success.png")))
        _safe_run(f"task modes {csv_path}", lambda: plot_task_stats(str(csv_path), str(out_dir / f"{stem}_task_modes.png")))
        if not args.skip_task_specs:
            _safe_run(f"task specs {csv_path}", lambda: plot_task_specs(str(csv_path), str(out_dir / "task_specs")))

    _safe_run("overhead plots", lambda: generate_overhead(str(results), str(figures)))

    for directory in [item for item in results.rglob("*") if item.is_dir()] + [results]:
        if list(directory.glob("*.json")):
            rel = directory.relative_to(results)
            _safe_run(f"comparison {directory}", lambda directory=directory, rel=rel: generate_comparison(str(directory), str(figures / rel / "comparison")))

    print(f"done. figures saved under: {figures}")


if __name__ == "__main__":
    main()

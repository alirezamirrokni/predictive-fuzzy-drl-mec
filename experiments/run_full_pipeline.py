from __future__ import annotations

import argparse
from pathlib import Path

from experiments._experiment_utils import call


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["a", "b", "both"], default="both")
    parser.add_argument("--train-predictors", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--total-timesteps", type=int, default=10000)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-episode-tasks", type=int, default=2000)
    args = parser.parse_args()
    targets = []
    if args.scenario in {"a", "both"}:
        targets.append("experiments.run_scenario_a")
    if args.scenario in {"b", "both"}:
        targets.append("experiments.run_scenario_b")
    import sys

    for module in targets:
        cmd = [
            sys.executable,
            "-m",
            module,
            "--total-timesteps",
            str(args.total_timesteps),
            "--episodes",
            str(args.episodes),
            "--max-episode-tasks",
            str(args.max_episode_tasks),
        ]
        if args.train_predictors:
            cmd.append("--train-predictors")
        if args.force_train:
            cmd.append("--force-train")
        call(cmd)
    call([sys.executable, "-m", "plots.generate_all_plots", "--results-dir", "data/results", "--figures-dir", "reports/figures"])


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from experiments._experiment_utils import collect_json_summaries, evaluate_drl_method, maybe_train_predictor, run_heuristics, train_drl_method


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scenario_b.yaml")
    parser.add_argument("--output-dir", default="data/results/scenario_b")
    parser.add_argument("--skip-heuristics", action="store_true")
    parser.add_argument("--skip-drl", action="store_true")
    parser.add_argument("--train-predictors", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--total-timesteps", type=int, default=10000)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-episode-tasks", type=int, default=2000)
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if not args.skip_heuristics:
        run_heuristics(args.config, args.output_dir)
    if not args.skip_drl:
        if args.train_predictors:
            maybe_train_predictor("lstm", args.config, args.force_train)
            maybe_train_predictor("gnn", args.config, args.force_train)
        for method in ["no_prediction_drl", "static_weight_drl", "lstm_fuzzy_drl", "gnn_fuzzy_drl"]:
            train_drl_method(method, args.config, args.output_dir, args.total_timesteps, args.max_episode_tasks, args.force_train)
            evaluate_drl_method(method, args.config, args.output_dir, args.episodes, args.max_episode_tasks)
    collect_json_summaries(args.output_dir, str(Path(args.output_dir) / "scenario_b_all_summaries.json"))


if __name__ == "__main__":
    main()

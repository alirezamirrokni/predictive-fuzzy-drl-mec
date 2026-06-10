from __future__ import annotations

import argparse
from pathlib import Path

from experiments._experiment_utils import collect_json_summaries, evaluate_drl_method, train_drl_method


ABLATION_METHODS = [
    "no_prediction_drl",
    "static_weight_drl",
    "lstm_fuzzy_drl",
    "gnn_fuzzy_drl",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1_small.yaml")
    parser.add_argument("--output-dir", default="data/results/ablation")
    parser.add_argument("--total-timesteps", type=int, default=5000)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-episode-tasks", type=int, default=400)
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for method in ABLATION_METHODS:
        train_drl_method(method, args.config, args.output_dir, args.total_timesteps, args.max_episode_tasks, args.force_train)
        evaluate_drl_method(method, args.config, args.output_dir, args.episodes, args.max_episode_tasks)
    collect_json_summaries(args.output_dir, str(Path(args.output_dir) / "ablation_all_summaries.json"))


if __name__ == "__main__":
    main()

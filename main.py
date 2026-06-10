from __future__ import annotations

import argparse
from pathlib import Path

from baselines.greedy_energy import GreedyEnergyPolicy
from baselines.greedy_latency import GreedyLatencyPolicy
from baselines.local_only import LocalOnlyPolicy
from baselines.random_policy import RandomPolicy
from baselines.no_prediction_drl import NoPredictionDRLPolicy
from baselines.static_weight_drl import StaticWeightDRLPolicy
from mec.simulator import build_simulator_from_config, load_yaml_config


POLICIES = {
    "random": RandomPolicy,
    "local_only": LocalOnlyPolicy,
    "greedy_latency": GreedyLatencyPolicy,
    "greedy_energy": GreedyEnergyPolicy,
    "no_prediction_drl": NoPredictionDRLPolicy,
    "static_weight_drl": StaticWeightDRLPolicy,
}


def build_policy(name: str):
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name}")
    return POLICIES[name]()


def run(config_path: str, policy_name: str, output_dir: str) -> None:
    config = load_yaml_config(config_path)
    simulator = build_simulator_from_config(config)
    policy = build_policy(policy_name)
    metrics = simulator.run(policy)
    scenario_name = str(config.get("scenario_name", Path(config_path).stem))
    output = Path(output_dir)
    metrics_path = output / f"{scenario_name}_{policy_name}_metrics.csv"
    summary_path = output / f"{scenario_name}_{policy_name}_summary.json"
    metrics.to_csv(metrics_path)
    metrics.summary_to_json(summary_path)
    print(metrics.summary())
    print(metrics_path)
    print(summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1_small.yaml")
    parser.add_argument("--policy", default="greedy_latency", choices=sorted(POLICIES.keys()))
    parser.add_argument("--output-dir", default="data/results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, args.policy, args.output_dir)


if __name__ == "__main__":
    main()

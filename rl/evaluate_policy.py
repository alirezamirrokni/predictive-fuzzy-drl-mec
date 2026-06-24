from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

import numpy as np
import torch

from .mec_env import MECEnvConfig, MECOffloadingEnv
from .ppo_agent import load_ppo_checkpoint
from .reward import RewardConfig
from .train_ppo import split_env_reward_config


def rows_from_records(records) -> List[Dict[str, Any]]:
    return [record.__dict__ if hasattr(record, "__dict__") else record for record in records]


def write_info_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8") as file:
            file.write("")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(step_rows: List[Dict[str, Any]], total_rewards: List[float]) -> Dict[str, float]:
    successes = [float(row.get("success", 0.0)) for row in step_rows]
    latencies = [float(row.get("latency_s", 0.0)) for row in step_rows]
    energies = [float(row.get("energy_j", 0.0)) for row in step_rows]
    reliabilities = [float(row.get("reliability", 0.0)) for row in step_rows]
    deadline_violations = [float(row.get("deadline_violation", 0.0)) for row in step_rows]
    execution_overheads = [float(row.get("execution_overhead_s", 0.0)) for row in step_rows]
    online_overheads = [float(row.get("online_overhead_s", 0.0)) for row in step_rows]
    return {
        "episodes": float(len(total_rewards)),
        "tasks": float(len(step_rows)),
        "average_episode_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
        "success_ratio": float(np.mean(successes)) if successes else 0.0,
        "average_latency_s": float(np.mean(latencies)) if latencies else 0.0,
        "average_energy_j": float(np.mean(energies)) if energies else 0.0,
        "average_reliability": float(np.mean(reliabilities)) if reliabilities else 0.0,
        "deadline_violation_rate": float(np.mean(deadline_violations)) if deadline_violations else 0.0,
        "average_execution_overhead_s": float(np.mean(execution_overheads)) if execution_overheads else 0.0,
        "average_online_overhead_s": float(np.mean(online_overheads)) if online_overheads else 0.0,
        "total_online_overhead_s": float(np.sum(online_overheads)) if online_overheads else 0.0,
    }


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    env_config, reward_config = split_env_reward_config(args.env_config, args)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = None
    checkpoint = {}
    if args.policy == "ppo":
        model, checkpoint = load_ppo_checkpoint(args.checkpoint_path, map_location=device)
        model = model.to(device)
        model.eval()
    total_rewards: List[float] = []
    step_rows: List[Dict[str, Any]] = []
    simulator_summary: Dict[str, Any] = {}
    for episode in range(int(args.episodes)):
        env = MECOffloadingEnv(args.scenario_config, env_config, reward_config)
        obs, _ = env.reset(seed=int(args.seed) + episode)
        done = False
        episode_reward = 0.0
        rng = np.random.default_rng(int(args.seed) + episode)
        while not done:
            action_start = perf_counter()
            if args.policy == "random":
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    action_tensor, _, _, _ = model.get_action_and_value(obs_tensor, deterministic=bool(args.deterministic))
                    action = action_tensor.squeeze(0).detach().cpu().numpy()
            policy_inference_overhead_s = perf_counter() - action_start
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            episode_reward += float(reward)
            if "task_id" in info:
                row = dict(info)
                row["episode"] = episode
                row["step_reward"] = float(reward)
                row["drl_overhead_s"] = policy_inference_overhead_s if args.policy == "ppo" else 0.0
                row["policy_overhead_s"] = policy_inference_overhead_s
                row["online_overhead_s"] = float(row.get("online_overhead_s", 0.0)) + policy_inference_overhead_s
                step_rows.append(row)
        total_rewards.append(episode_reward)
        if env.simulator is not None:
            simulator_summary = env.simulator.metrics.summary()
        env.close()
    result = {
        "policy": args.policy,
        "checkpoint": args.checkpoint_path if args.policy == "ppo" else "",
        "scenario_config": args.scenario_config,
        "summary": summarize(step_rows, total_rewards),
        "simulator_summary_last_episode": simulator_summary,
        "checkpoint_metadata": checkpoint.get("metadata", {}) if checkpoint else {},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    if args.metrics_csv:
        write_info_csv(step_rows, args.metrics_csv)
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["ppo", "random"], default="ppo")
    parser.add_argument("--scenario-config", default="configs/phase1_small.yaml")
    parser.add_argument("--env-config", default="configs/rl_env.yaml")
    parser.add_argument("--checkpoint-path", default="data/generated/checkpoints/ppo.pt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output", default="data/results/ppo_eval.json")
    parser.add_argument("--metrics-csv", default="data/results/ppo_eval_metrics.csv")
    parser.add_argument("--device", default="")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--max-episode-tasks", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--predictor-type", default="")
    parser.add_argument("--predictor-checkpoint-path", default="")
    parser.add_argument("--predictor-model-config-path", default="")
    parser.add_argument("--static-weights", action="store_true")
    parser.add_argument("--no-fuzzy-weights", action="store_true")
    parser.add_argument("--no-prediction", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())

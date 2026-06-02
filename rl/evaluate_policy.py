from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .mec_env import MECEnvConfig, MECOffloadingEnv


def evaluate_random(config_path: str, episodes: int, max_episode_tasks: int | None, seed: int) -> dict:
    rewards = []
    successes = []
    latencies = []
    energies = []
    for episode in range(episodes):
        env = MECOffloadingEnv(config_path, MECEnvConfig(max_episode_tasks=max_episode_tasks))
        observation, _ = env.reset(seed=seed + episode)
        done = False
        rng = np.random.default_rng(seed + episode)
        total_reward = 0.0
        while not done:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if "success" in info:
                successes.append(float(info["success"]))
                latencies.append(float(info["latency_s"]))
                energies.append(float(info["energy_j"]))
        rewards.append(total_reward)
        env.close()
    return {
        "episodes": episodes,
        "average_reward": float(np.mean(rewards)) if rewards else 0.0,
        "average_success": float(np.mean(successes)) if successes else 0.0,
        "average_latency_s": float(np.mean(latencies)) if latencies else 0.0,
        "average_energy_j": float(np.mean(energies)) if energies else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1_small.yaml")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-episode-tasks", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="data/results/random_env_eval.json")
    args = parser.parse_args()
    result = evaluate_random(args.config, args.episodes, args.max_episode_tasks, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

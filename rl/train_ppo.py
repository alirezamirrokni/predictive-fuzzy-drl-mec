from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml

from .mec_env import MECEnvConfig, MECOffloadingEnv
from .ppo_agent import ActorCritic, PPOModelConfig, RolloutBuffer, action_dims_from_env, load_ppo_checkpoint, save_ppo_checkpoint
from .reward import RewardConfig


@dataclass(slots=True)
class PPOTrainConfig:
    seed: int = 0
    total_timesteps: int = 10_000
    n_steps: int = 512
    update_epochs: int = 4
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_size: int = 128
    adam_epsilon: float = 1e-5
    normalize_advantage: bool = True
    checkpoint_every_updates: int = 1
    recent_success_window_tasks: int = 5000
    validation_every_updates: int = 10
    validation_tasks: int = 10000
    validation_seed: int = 9000

    @classmethod
    def from_dict(cls, data: Dict[str, object] | None) -> "PPOTrainConfig":
        if not data:
            return cls()
        aliases = {
            "batch_size": "minibatch_size",
            "n_epochs": "update_epochs",
        }
        fields = cls.__dataclass_fields__
        cleaned: Dict[str, object] = {}
        for key, value in data.items():
            key = aliases.get(key, key)
            if key in fields:
                cleaned[key] = value
        return cls(**cleaned)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"configuration {path} must be a mapping")
    return data


def split_env_reward_config(config_path: str | Path, overrides: argparse.Namespace | None = None) -> Tuple[MECEnvConfig, RewardConfig]:
    config = load_yaml(config_path) if config_path and Path(config_path).exists() else {}
    env_data = {key: value for key, value in config.items() if key != "reward"}
    reward_data = config.get("reward", {})
    if overrides is not None:
        if overrides.max_episode_tasks is not None:
            env_data["max_episode_tasks"] = overrides.max_episode_tasks
        if overrides.top_k is not None:
            env_data["top_k"] = overrides.top_k
        if overrides.predictor_type:
            env_data["predictor_type"] = overrides.predictor_type
        if overrides.predictor_checkpoint_path:
            env_data["predictor_checkpoint_path"] = overrides.predictor_checkpoint_path
        if overrides.predictor_model_config_path:
            env_data["predictor_model_config_path"] = overrides.predictor_model_config_path
        if overrides.no_fuzzy_weights:
            env_data["include_fuzzy_weights"] = False
        if overrides.no_prediction:
            env_data["include_prediction"] = False
        if overrides.static_weights:
            reward_data["use_static_weights"] = True
    return MECEnvConfig.from_dict(env_data), RewardConfig.from_dict(reward_data)


def make_env(args: argparse.Namespace) -> MECOffloadingEnv:
    env_config, reward_config = split_env_reward_config(args.env_config, args)
    return MECOffloadingEnv(args.scenario_config, env_config, reward_config)


def train(args: argparse.Namespace) -> Dict[str, Any]:
    ppo_config_data = load_yaml(args.ppo_config) if args.ppo_config and Path(args.ppo_config).exists() else {}
    cfg = PPOTrainConfig.from_dict(ppo_config_data)
    if args.total_timesteps is not None:
        cfg.total_timesteps = int(args.total_timesteps)
    if args.n_steps is not None:
        cfg.n_steps = int(args.n_steps)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.learning_rate is not None:
        cfg.learning_rate = float(args.learning_rate)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    env = make_env(args)
    obs, reset_info = env.reset(seed=cfg.seed)
    obs_dim = int(np.asarray(obs).shape[0])
    action_dims = action_dims_from_env(env)
    model = ActorCritic(PPOModelConfig(obs_dim, action_dims, cfg.hidden_size)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon)
    resume_metadata: Dict[str, Any] = {}
    if args.resume and Path(args.checkpoint_path).exists():
        restored, checkpoint = load_ppo_checkpoint(args.checkpoint_path, map_location=device)
        if restored.config.observation_dim != obs_dim or list(restored.config.action_dims) != action_dims:
            raise ValueError("PPO checkpoint observation/action schema does not match the current environment")
        model.load_state_dict(restored.state_dict())
        if checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        resume_metadata = dict(checkpoint.get("metadata", {}))

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_mode = "a" if args.resume and log_path.exists() else "w"
    with log_path.open(log_mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["update", "global_step", "mean_reward", "recent_success_ratio", "policy_loss", "value_loss", "entropy", "approx_kl", "validation_reward"])
        if log_mode == "w":
            writer.writeheader()

    global_step = int(resume_metadata.get("global_step", 0))
    update = int(resume_metadata.get("update", 0))
    episode_rewards = []
    current_episode_reward = 0.0
    recent_rewards = []
    recent_successes = []
    best_validation_reward = -float("inf")
    started_at = monotonic()
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
    n_steps = max(1, int(cfg.n_steps))

    while global_step < cfg.total_timesteps:
        rollout_size = min(n_steps, cfg.total_timesteps - global_step)
        buffer = RolloutBuffer(rollout_size, obs_dim, len(action_dims), device)
        for _ in range(rollout_size):
            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(obs_tensor.unsqueeze(0))
            action_np = action.squeeze(0).detach().cpu().numpy()
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = bool(terminated or truncated)
            buffer.add(obs_tensor, action.squeeze(0), logprob.squeeze(0), float(reward), done, value.squeeze(0))
            current_episode_reward += float(reward)
            recent_rewards.append(float(reward))
            recent_successes.append(float(info.get("success", 0.0)))
            window = int(cfg.recent_success_window_tasks)
            if len(recent_rewards) > window:
                recent_rewards = recent_rewards[-window:]
                recent_successes = recent_successes[-window:]
            global_step += 1
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                next_obs, _ = env.reset(seed=cfg.seed + global_step)
            obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            _, _, _, last_value = model.get_action_and_value(obs_tensor.unsqueeze(0), deterministic=True)
        buffer.compute_returns_and_advantages(last_value.squeeze(0), cfg.gamma, cfg.gae_lambda)
        advantages = buffer.advantages
        if cfg.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_losses = []
        value_losses = []
        entropies = []
        kls = []
        for _epoch in range(int(cfg.update_epochs)):
            for batch_idx in buffer.minibatches(int(cfg.minibatch_size)):
                _, new_logprob, entropy, new_value = model.get_action_and_value(buffer.observations[batch_idx], buffer.actions[batch_idx])
                logratio = new_logprob - buffer.logprobs[batch_idx]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                adv = advantages[batch_idx]
                policy_loss_1 = -adv * ratio
                policy_loss_2 = -adv * torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
                value_loss = 0.5 * ((new_value - buffer.returns[batch_idx]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss - cfg.entropy_coef * entropy_loss + cfg.value_coef * value_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                policy_losses.append(float(policy_loss.detach().cpu()))
                value_losses.append(float(value_loss.detach().cpu()))
                entropies.append(float(entropy_loss.detach().cpu()))
                kls.append(float(approx_kl.detach().cpu()))
        update += 1
        validation_reward = float("nan")
        if update % max(1, int(cfg.validation_every_updates)) == 0:
            validation_reward = _validation_reward(model, args, cfg, device)
            if validation_reward > best_validation_reward:
                best_validation_reward = validation_reward
                best_path = args.best_checkpoint_path or str(Path(args.checkpoint_path).with_name(Path(args.checkpoint_path).stem + "_best.pt"))
                save_ppo_checkpoint(best_path, model, {"global_step": global_step, "update": update, "validation_reward": validation_reward}, optimizer)
        row = {
            "update": update,
            "global_step": global_step,
            "mean_reward": float(np.mean(recent_rewards)) if recent_rewards else 0.0,
            "recent_success_ratio": float(np.mean(recent_successes)) if recent_successes else 0.0,
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl": float(np.mean(kls)) if kls else 0.0,
            "validation_reward": validation_reward,
        }
        with log_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row.keys()))
            writer.writerow(row)
        print(row)

        if update % max(1, int(cfg.checkpoint_every_updates)) == 0:
            save_ppo_checkpoint(
                args.checkpoint_path,
                model,
                {
                    "scenario_config": args.scenario_config,
                    "global_step": global_step,
                    "update": update,
                    "best_validation_reward": best_validation_reward,
                    "elapsed_training_s": monotonic() - started_at,
                    "predictor_type": env.env_config.predictor_type,
                },
                optimizer,
            )
        if args.time_budget_hours is not None and monotonic() - started_at >= float(args.time_budget_hours) * 3600.0:
            break

    metadata = {
        "scenario_config": args.scenario_config,
        "env_config": args.env_config,
        "ppo_config": args.ppo_config,
        "global_step": global_step,
        "episodes": len(episode_rewards),
        "mean_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else current_episode_reward,
        "predictor_type": env.env_config.predictor_type,
        "include_fuzzy_weights": env.env_config.include_fuzzy_weights,
        "include_prediction": env.env_config.include_prediction,
        "learning_task_equivalent": int(global_step),
        "max_episode_tasks": env.env_config.max_episode_tasks if env.env_config.max_episode_tasks is not None else 0,
        "best_validation_reward": best_validation_reward,
        "elapsed_training_s": monotonic() - started_at,
    }
    save_ppo_checkpoint(args.checkpoint_path, model, metadata, optimizer)
    best_path = args.best_checkpoint_path or str(Path(args.checkpoint_path).with_name(Path(args.checkpoint_path).stem + "_best.pt"))
    if not Path(best_path).exists():
        save_ppo_checkpoint(best_path, model, metadata, optimizer)
    env.close()
    result = {"checkpoint": args.checkpoint_path, "log_path": args.log_path, **metadata}
    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-config", default="configs/scenario_b.yaml")
    parser.add_argument("--env-config", default="configs/rl_env.yaml")
    parser.add_argument("--ppo-config", default="configs/ppo.yaml")
    parser.add_argument("--checkpoint-path", default="data/generated/checkpoints/ppo.pt")
    parser.add_argument("--log-path", default="data/results/ppo_training_log.csv")
    parser.add_argument("--result-path", default="data/results/ppo_training_summary.json")
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-episode-tasks", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--predictor-type", default="")
    parser.add_argument("--predictor-checkpoint-path", default="")
    parser.add_argument("--predictor-model-config-path", default="")
    parser.add_argument("--best-checkpoint-path", default="")
    parser.add_argument("--time-budget-hours", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--static-weights", action="store_true")
    parser.add_argument("--no-fuzzy-weights", action="store_true")
    parser.add_argument("--no-prediction", action="store_true")
    return parser.parse_args()


def _validation_reward(model: ActorCritic, args: argparse.Namespace, cfg: PPOTrainConfig, device: torch.device) -> float:
    env_config, reward_config = split_env_reward_config(args.env_config, args)
    env_config.max_episode_tasks = int(cfg.validation_tasks)
    env = MECOffloadingEnv(args.scenario_config, env_config, reward_config)
    obs, _ = env.reset(seed=int(cfg.validation_seed))
    rewards = []
    done = False
    while not done:
        with torch.no_grad():
            tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, _, _, _ = model.get_action_and_value(tensor, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
        rewards.append(float(reward))
        done = bool(terminated or truncated)
    env.close()
    return float(np.mean(rewards)) if rewards else -float("inf")


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))

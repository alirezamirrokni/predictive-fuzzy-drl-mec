from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass(slots=True)
class PPOModelConfig:
    observation_dim: int
    action_dims: Sequence[int]
    hidden_size: int = 128


class ActorCritic(nn.Module):
    def __init__(self, config: PPOModelConfig) -> None:
        super().__init__()
        self.config = config
        self.shared = nn.Sequential(
            nn.Linear(config.observation_dim, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
        )
        self.policy_heads = nn.ModuleList([nn.Linear(config.hidden_size, int(dim)) for dim in config.action_dims])
        self.value_head = nn.Linear(config.hidden_size, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        hidden = self.shared(obs)
        logits = [head(hidden) for head in self.policy_heads]
        value = self.value_head(hidden).squeeze(-1)
        return logits, value

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None, deterministic: bool = False):
        logits, value = self.forward(obs)
        distributions = [Categorical(logits=item) for item in logits]
        if action is None:
            if deterministic:
                action_parts = [torch.argmax(dist.logits, dim=-1) for dist in distributions]
            else:
                action_parts = [dist.sample() for dist in distributions]
            action = torch.stack(action_parts, dim=-1)
        logprob = torch.zeros(obs.shape[0], device=obs.device)
        entropy = torch.zeros(obs.shape[0], device=obs.device)
        for index, dist in enumerate(distributions):
            logprob = logprob + dist.log_prob(action[:, index])
            entropy = entropy + dist.entropy()
        return action, logprob, entropy, value


@dataclass(slots=True)
class PPOCheckpoint:
    model_config: Dict[str, object]
    model_state_dict: Dict[str, torch.Tensor]
    metadata: Dict[str, object]


def save_ppo_checkpoint(
    path: str | Path,
    model: ActorCritic,
    metadata: Dict[str, object],
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": {
                "observation_dim": int(model.config.observation_dim),
                "action_dims": [int(dim) for dim in model.config.action_dims],
                "hidden_size": int(model.config.hidden_size),
            },
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_ppo_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Tuple[ActorCritic, Dict[str, object]]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    config = PPOModelConfig(**checkpoint["model_config"])
    model = ActorCritic(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


class RolloutBuffer:
    def __init__(self, size: int, observation_dim: int, action_shape: int, device: torch.device) -> None:
        self.size = int(size)
        self.observations = torch.zeros((size, observation_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((size, action_shape), dtype=torch.long, device=device)
        self.logprobs = torch.zeros(size, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(size, dtype=torch.float32, device=device)
        self.values = torch.zeros(size, dtype=torch.float32, device=device)
        self.advantages = torch.zeros(size, dtype=torch.float32, device=device)
        self.returns = torch.zeros(size, dtype=torch.float32, device=device)
        self.index = 0

    def add(self, obs, action, logprob, reward, done, value) -> None:
        i = self.index
        self.observations[i] = obs
        self.actions[i] = action
        self.logprobs[i] = logprob
        self.rewards[i] = float(reward)
        self.dones[i] = float(done)
        self.values[i] = value
        self.index += 1

    def compute_returns_and_advantages(self, last_value: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        last_gae = 0.0
        for step in reversed(range(self.size)):
            if step == self.size - 1:
                next_nonterminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                # dones[t] describes the transition from state t to state t+1.
                next_nonterminal = 1.0 - self.dones[step]
                next_value = self.values[step + 1]
            delta = self.rewards[step] + gamma * next_value * next_nonterminal - self.values[step]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            self.advantages[step] = last_gae
        self.returns = self.advantages + self.values

    def minibatches(self, batch_size: int):
        indices = torch.randperm(self.size, device=self.observations.device)
        for start in range(0, self.size, batch_size):
            yield indices[start : start + batch_size]


def action_dims_from_env(env) -> List[int]:
    if not hasattr(env.action_space, "nvec"):
        raise ValueError("PPO implementation expects a MultiDiscrete action space")
    return [int(item) for item in np.asarray(env.action_space.nvec).tolist()]

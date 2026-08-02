from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from fuzzy.fuzzy_controller import FuzzyController, FuzzyWeights
from mec.device import IoTDevice
from mec.server import EdgeServer
from mec.simulator import SimulationOutcome
from mec.task import Task


@dataclass(slots=True)
class RewardConfig:
    success_alpha: float = 1.0
    energy_beta: float = 2.0
    deadline_penalty: float = 1.0
    failure_penalty: float = 1.0
    reliability_target_penalty: float = 0.5
    reward_clip: float = 10.0

    @classmethod
    def from_dict(cls, data: Dict[str, object] | None) -> "RewardConfig":
        if not data:
            return cls()
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in fields})


class MECRewardFunction:
    def __init__(self, fuzzy_controller: FuzzyController | None = None, config: RewardConfig | None = None) -> None:
        self.fuzzy_controller = FuzzyController() if fuzzy_controller is None else fuzzy_controller
        self.config = RewardConfig() if config is None else config

    def weights_for(self, task: Task, device: IoTDevice, candidate_servers: Iterable[EdgeServer], prediction_uncertainty: float = 0.0) -> FuzzyWeights:
        return self.fuzzy_controller.compute_from_task_context(task, device, candidate_servers, prediction_uncertainty)

    def compute(self, outcome: SimulationOutcome, task: Task, device: IoTDevice, candidate_servers: Iterable[EdgeServer], prediction_uncertainty: float = 0.0):
        weights = self.weights_for(task, device, candidate_servers, prediction_uncertainty).to_dict()
        latency_term = min(5.0, outcome.latency_s / max(task.deadline_s, 1e-9))
        local_reference = device.compute_power_w * task.cpu_cycles_mi / max(device.cpu_capacity_mips, 1e-9)
        energy_term = min(5.0, outcome.energy_j / max(local_reference, 1e-9))
        reward = (
            self.config.success_alpha * weights["success"] * float(outcome.success)
            + weights["reliability"] * float(outcome.reliability)
            - weights["latency"] * latency_term
            - self.config.energy_beta * weights["energy"] * energy_term
        )
        if outcome.deadline_violation:
            reward -= self.config.deadline_penalty
        if not outcome.success:
            reward -= self.config.failure_penalty
        if not outcome.reliability_satisfied:
            reward -= self.config.reliability_target_penalty
        clip = abs(float(self.config.reward_clip))
        reward = max(-clip, min(clip, float(reward)))
        return reward, {
            "reward": reward,
            "latency_term": latency_term,
            "energy_term": energy_term,
            "weight_energy": weights["energy"],
            "weight_latency": weights["latency"],
            "weight_success": weights["success"],
            "weight_reliability": weights["reliability"],
        }

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
    latency_normalizer: float = 10.0
    energy_normalizer: float = 10.0
    deadline_penalty: float = 1.0
    failure_penalty: float = 1.0
    battery_penalty: float = 0.5
    channel_penalty: float = 0.5
    server_penalty: float = 0.5
    reward_clip: float = 10.0
    use_static_weights: bool = False
    static_energy_weight: float = 0.25
    static_latency_weight: float = 0.25
    static_success_weight: float = 0.25
    static_reliability_weight: float = 0.25

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
        if self.config.use_static_weights:
            return FuzzyWeights(
                energy=float(self.config.static_energy_weight),
                latency=float(self.config.static_latency_weight),
                success=float(self.config.static_success_weight),
                reliability=float(self.config.static_reliability_weight),
            ).normalized()
        return self.fuzzy_controller.compute_from_task_context(task, device, candidate_servers, prediction_uncertainty)

    def compute(
        self,
        outcome: SimulationOutcome,
        task: Task,
        device: IoTDevice,
        candidate_servers: Iterable[EdgeServer],
        prediction_uncertainty: float = 0.0,
    ) -> tuple[float, Dict[str, float]]:
        weights = self.weights_for(task, device, candidate_servers, prediction_uncertainty)
        weight_dict = weights.to_dict()
        latency = min(5.0, outcome.latency_s / max(task.deadline_s, self.config.latency_normalizer, 1e-9))
        energy = min(5.0, outcome.energy_j / max(self.config.energy_normalizer, 1e-9))
        reward = (
            weight_dict["success"] * float(outcome.success)
            + weight_dict["reliability"] * float(outcome.reliability)
            - weight_dict["latency"] * latency
            - weight_dict["energy"] * energy
        )
        if outcome.deadline_violation:
            reward -= self.config.deadline_penalty
        if not outcome.success:
            reward -= self.config.failure_penalty
        if outcome.failed_due_to_battery:
            reward -= self.config.battery_penalty
        if outcome.failed_due_to_channel:
            reward -= self.config.channel_penalty
        if outcome.failed_due_to_server:
            reward -= self.config.server_penalty
        clip = abs(float(self.config.reward_clip))
        reward = max(-clip, min(clip, float(reward)))
        info = {
            "reward": reward,
            "latency_term": latency,
            "energy_term": energy,
            "weight_energy": weight_dict["energy"],
            "weight_latency": weight_dict["latency"],
            "weight_success": weight_dict["success"],
            "weight_reliability": weight_dict["reliability"],
        }
        return reward, info

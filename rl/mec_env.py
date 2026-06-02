from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError("Install gymnasium to use MECOffloadingEnv: pip install gymnasium") from exc

from fuzzy.fuzzy_controller import FuzzyController
from mec.device import IoTDevice
from mec.server import EdgeServer
from mec.simulator import MECSimulator, OffloadingDecision, build_simulator_from_config, load_yaml_config
from mec.task import Task
from .candidate_selector import CandidateSelector, CandidateSelectorConfig
from .reward import MECRewardFunction, RewardConfig


@dataclass(slots=True)
class MECEnvConfig:
    top_k: int = 5
    include_fuzzy_weights: bool = True
    prediction_uncertainty: float = 0.0
    max_episode_tasks: Optional[int] = None


class MECOffloadingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_config: str | Dict[str, Any] = "configs/phase1_small.yaml",
        env_config: MECEnvConfig | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        super().__init__()
        self.scenario_config_source = scenario_config
        self.env_config = MECEnvConfig() if env_config is None else env_config
        self.selector = CandidateSelector(CandidateSelectorConfig(top_k=self.env_config.top_k))
        self.fuzzy_controller = FuzzyController()
        self.reward_function = MECRewardFunction(self.fuzzy_controller, reward_config)
        self.partial_ratios = [0.25, 0.5, 0.75, 1.0]
        self.action_space = spaces.MultiDiscrete([3, self.env_config.top_k, len(self.partial_ratios)])
        self.observation_dim = 6 + 7 + self.env_config.top_k * 9 + (4 if self.env_config.include_fuzzy_weights else 0)
        self.observation_space = spaces.Box(low=0.0, high=5.0, shape=(self.observation_dim,), dtype=np.float32)
        self.simulator: MECSimulator | None = None
        self.tasks: List[Task] = []
        self.current_index = 0
        self.current_time = 0
        self.last_info: Dict[str, Any] = {}

    def _load_config(self) -> Dict[str, Any]:
        if isinstance(self.scenario_config_source, dict):
            return self.scenario_config_source
        return load_yaml_config(self.scenario_config_source)

    def _build(self) -> None:
        self.simulator = build_simulator_from_config(self._load_config())
        self.tasks = []
        for device in self.simulator.devices:
            self.tasks.extend(device.task_queue)
            device.task_queue.clear()
        self.tasks.sort(key=lambda task: (task.arrival_time, task.device_id, task.task_id))
        if self.env_config.max_episode_tasks is not None:
            self.tasks = self.tasks[: int(self.env_config.max_episode_tasks)]
        self.current_index = 0
        self.current_time = 0
        self.last_info = {}

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._build()
        return self._observation(), {"tasks": len(self.tasks)}

    def step(self, action: np.ndarray | List[int] | Tuple[int, ...]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.simulator is None:
            raise RuntimeError("call reset before step")
        if self.current_index >= len(self.tasks):
            return self._zero_observation(), 0.0, True, False, {"empty": True}
        task = self.tasks[self.current_index]
        device = self.simulator.device_index[task.device_id]
        self._advance_to(task.arrival_time)
        candidates = self._current_candidates(task, device)
        decision = self._decode_action(action, candidates)
        outcome = self.simulator.apply(self.current_time, task, device, decision)
        reward, reward_info = self.reward_function.compute(outcome, task, device, candidates, self.env_config.prediction_uncertainty)
        self.current_index += 1
        terminated = self.current_index >= len(self.tasks)
        observation = self._zero_observation() if terminated else self._observation()
        info = {
            "task_id": task.task_id,
            "device_id": task.device_id,
            "mode": decision.mode,
            "server_id": -1 if decision.server_id is None else decision.server_id,
            "partial_ratio": decision.partial_ratio,
            "latency_s": outcome.latency_s,
            "energy_j": outcome.energy_j,
            "reliability": outcome.reliability,
            "success": outcome.success,
            "deadline_violation": outcome.deadline_violation,
            **reward_info,
        }
        self.last_info = info
        return observation, float(reward), terminated, False, info

    def _advance_to(self, target_time: int) -> None:
        if self.simulator is None:
            return
        while self.current_time < target_time:
            for server in self.simulator.servers:
                server.process_slot(self.simulator.slot_duration_s)
            for device in self.simulator.devices:
                self.simulator.mobility.update(device, self.simulator.rng, self.simulator.slot_duration_s)
            self.current_time += 1

    def _decode_action(self, action: np.ndarray | List[int] | Tuple[int, ...], candidates: List[EdgeServer]) -> OffloadingDecision:
        values = np.asarray(action, dtype=np.int64).reshape(-1)
        mode_index = int(values[0]) if values.size > 0 else 0
        server_index = int(values[1]) if values.size > 1 else 0
        ratio_index = int(values[2]) if values.size > 2 else 0
        mode_index = int(np.clip(mode_index, 0, 2))
        server_index = int(np.clip(server_index, 0, len(candidates) - 1))
        ratio_index = int(np.clip(ratio_index, 0, len(self.partial_ratios) - 1))
        if mode_index == 0:
            return OffloadingDecision("local")
        server_id = candidates[server_index].server_id
        if mode_index == 1:
            return OffloadingDecision("edge", server_id, 1.0)
        return OffloadingDecision("partial", server_id, self.partial_ratios[ratio_index])

    def _current_task(self) -> Tuple[Task, IoTDevice]:
        if self.simulator is None:
            raise RuntimeError("simulator is not initialized")
        if self.current_index >= len(self.tasks):
            raise RuntimeError("no current task")
        task = self.tasks[self.current_index]
        return task, self.simulator.device_index[task.device_id]

    def _current_candidates(self, task: Task, device: IoTDevice) -> List[EdgeServer]:
        if self.simulator is None:
            raise RuntimeError("simulator is not initialized")
        return self.selector.pad(self.selector.select(self.simulator, task, device))

    def _observation(self) -> np.ndarray:
        if self.simulator is None or self.current_index >= len(self.tasks):
            return self._zero_observation()
        task, device = self._current_task()
        candidates = self._current_candidates(task, device)
        values: List[float] = []
        values.extend(self._task_features(task))
        values.extend(self._device_features(device))
        for server in candidates:
            values.extend(self._server_features(server, device))
        if self.env_config.include_fuzzy_weights:
            weights = self.fuzzy_controller.compute_from_task_context(task, device, candidates, self.env_config.prediction_uncertainty).to_dict()
            values.extend([weights["energy"], weights["latency"], weights["success"], weights["reliability"]])
        arr = np.asarray(values, dtype=np.float32)
        if arr.shape[0] != self.observation_dim:
            raise RuntimeError(f"invalid observation size {arr.shape[0]}, expected {self.observation_dim}")
        return np.nan_to_num(arr, nan=0.0, posinf=5.0, neginf=0.0).astype(np.float32)

    def _zero_observation(self) -> np.ndarray:
        return np.zeros((self.observation_dim,), dtype=np.float32)

    def _task_features(self, task: Task) -> List[float]:
        if self.simulator is None:
            max_time = 1.0
        else:
            max_time = max(1.0, float(self.simulator.time_slots))
        return [
            task.data_size_mb / 10.0,
            task.output_size_mb / 2.0,
            task.cpu_cycles_mi / 10000.0,
            task.deadline_s / 10.0,
            task.priority / 5.0,
            task.arrival_time / max_time,
        ]

    def _device_features(self, device: IoTDevice) -> List[float]:
        if self.simulator is None:
            width = height = 1.0
        else:
            width = max(1.0, self.simulator.mobility.area_width_m)
            height = max(1.0, self.simulator.mobility.area_height_m)
        return [
            device.cpu_capacity_mips / 2000.0,
            device.battery_j / 20000.0,
            device.tx_power_w / 2.0,
            device.local_power_w / 3.0,
            device.position[0] / width,
            device.position[1] / height,
            device.failure_probability,
        ]

    def _server_features(self, server: EdgeServer, device: IoTDevice) -> List[float]:
        if self.simulator is None:
            distance = 0.0
            diag = 1.0
            rate = 0.0
            width = height = 1.0
        else:
            distance = self.simulator.channel.distance_m(device.position, server.position)
            diag = max(1.0, (self.simulator.mobility.area_width_m ** 2 + self.simulator.mobility.area_height_m ** 2) ** 0.5)
            rate = self.simulator.channel.data_rate_mbps(server.bandwidth_mhz, device.tx_power_w, device.position, server.position)
            width = max(1.0, self.simulator.mobility.area_width_m)
            height = max(1.0, self.simulator.mobility.area_height_m)
        return [
            server.cpu_capacity_mips / 30000.0,
            server.bandwidth_mhz / 100.0,
            server.queue_delay() / 10.0,
            server.queue_workload_mi / max(server.cpu_capacity_mips, 1e-9),
            server.reliability,
            distance / diag,
            rate / 1000.0,
            server.position[0] / width,
            server.position[1] / height,
        ]

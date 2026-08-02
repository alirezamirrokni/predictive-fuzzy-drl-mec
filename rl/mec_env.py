from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError("Install requirements.txt to use MECOffloadingEnv") from exc

from fuzzy.fuzzy_controller import FuzzyController
from mec.device import IoTDevice
from mec.server import EdgeServer
from mec.simulator import MECSimulator, OffloadingDecision, build_simulator_from_config, load_yaml_config
from mec.task import Task
from predictors.runtime import PredictiveConfig, PredictiveFeatureProvider, PredictiveSnapshot
from .candidate_selector import CandidateSelector, CandidateSelectorConfig
from .reward import MECRewardFunction, RewardConfig


@dataclass(slots=True)
class MECEnvConfig:
    offloading_mode: str = "mixed"
    top_k: int = 16
    include_fuzzy_weights: bool = True
    max_episode_tasks: Optional[int] = None
    predictor_type: str = "none"
    predictor_checkpoint_path: str = ""
    predictor_model_config_path: str = ""
    include_prediction: bool = True
    predictive_feature_size: int = 4
    strict_predictor_loading: bool = True
    predictor_fallback_permitted: bool = False
    device: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, object] | None) -> "MECEnvConfig":
        if not data:
            return cls()
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in fields})


class MECOffloadingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_config: str | Dict[str, Any] = "configs/scenario_b.yaml",
        env_config: MECEnvConfig | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        super().__init__()
        self.scenario_config_source = scenario_config
        self.env_config = MECEnvConfig() if env_config is None else env_config
        if self.env_config.offloading_mode != "mixed":
            raise ValueError("The final project supports offloading_mode='mixed' only")
        self.selector = CandidateSelector(CandidateSelectorConfig(top_k=self.env_config.top_k))
        self.fuzzy_controller = FuzzyController()
        self.reward_function = MECRewardFunction(self.fuzzy_controller, reward_config)
        self.predictor = PredictiveFeatureProvider(
            PredictiveConfig(
                predictor_type=self.env_config.predictor_type,
                checkpoint_path=self.env_config.predictor_checkpoint_path,
                model_config_path=self.env_config.predictor_model_config_path,
                feature_size=self.env_config.predictive_feature_size,
                strict_loading=self.env_config.strict_predictor_loading,
                fallback_permitted=self.env_config.predictor_fallback_permitted,
                device=self.env_config.device,
            )
        )
        # Mixed action: local, full edge, or task-defined partial split; plus server.
        self.action_space = spaces.MultiDiscrete([3, self.env_config.top_k])
        self.observation_dim = (
            6 + 4 + self.env_config.top_k * 7
            + (self.env_config.predictive_feature_size if self.env_config.include_prediction else 0)
            + (4 if self.env_config.include_fuzzy_weights else 0)
        )
        self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(self.observation_dim,), dtype=np.float32)
        self.simulator: MECSimulator | None = None
        self.tasks: List[Task] = []
        self.current_index = 0
        self.current_time = 0
        self.last_info: Dict[str, Any] = {}
        self.last_prediction: PredictiveSnapshot | None = None
        self.last_prediction_overhead_s = 0.0
        self._ready_counts: Dict[int, int] = {}

    def _load_config(self) -> Dict[str, Any]:
        if isinstance(self.scenario_config_source, dict):
            return self.scenario_config_source
        return load_yaml_config(self.scenario_config_source)

    def _build(self, seed: Optional[int]) -> None:
        self.simulator = build_simulator_from_config(self._load_config(), seed_override=seed)
        all_tasks: List[Task] = []
        for device in self.simulator.devices:
            all_tasks.extend(device.task_queue)
            device.task_queue.clear()
        if self.env_config.max_episode_tasks is not None and len(all_tasks) > int(self.env_config.max_episode_tasks):
            indices = self.simulator.rng.choice(len(all_tasks), size=int(self.env_config.max_episode_tasks), replace=False)
            all_tasks = [all_tasks[int(i)] for i in indices]
        self.tasks = sorted(all_tasks, key=lambda task: (task.arrival_slot, task.device_id, task.task_id))
        self._ready_counts = {}
        for task in self.tasks:
            self._ready_counts[task.arrival_slot] = self._ready_counts.get(task.arrival_slot, 0) + 1
        self.current_index = 0
        self.current_time = 0
        self.last_info = {}
        self.predictor.reset()
        self.predictor.observe(self.simulator, ready_task_rate=0.0)
        self.last_prediction = None
        self.last_prediction_overhead_s = 0.0

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.action_space.seed(seed)
        self._build(seed)
        info = {"tasks": len(self.tasks), **(self.simulator.scenario_sample if self.simulator else {})}
        return self._observation(), info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.simulator is None:
            raise RuntimeError("call reset before step")
        if self.current_index >= len(self.tasks):
            return self._zero_observation(), 0.0, True, False, {"empty": True}
        task = self.tasks[self.current_index]
        device = self.simulator.device_index[task.device_id]
        self._advance_to(task.arrival_slot)
        online_start = perf_counter()
        candidates = self._current_candidates(task, device)
        prediction = self.last_prediction if self.last_prediction is not None else self.predictor.predict(self.simulator)
        prediction_overhead_s = self.last_prediction_overhead_s
        decision = self._decode_action(action, candidates, task)
        apply_start = perf_counter()
        outcome = self.simulator.apply(self.current_time, task, device, decision, prediction_overhead_s=prediction_overhead_s)
        simulator_apply_overhead_s = perf_counter() - apply_start
        fuzzy_start = perf_counter()
        reward, reward_info = self.reward_function.compute(outcome, task, device, candidates, prediction.uncertainty)
        fuzzy_overhead_s = perf_counter() - fuzzy_start
        online_overhead_s = perf_counter() - online_start + prediction_overhead_s
        record = self.simulator.metrics.records[-1]
        record.fuzzy_overhead_s = fuzzy_overhead_s
        record.online_overhead_s = online_overhead_s
        self.current_index += 1
        terminated = self.current_index >= len(self.tasks)
        self.last_prediction = None
        self.last_prediction_overhead_s = 0.0
        observation = self._zero_observation() if terminated else self._observation()
        info = {
            "task_id": task.task_id,
            "device_id": task.device_id,
            "mode": decision.mode,
            "server_id": -1 if decision.server_id is None else decision.server_id,
            "partial_ratio": decision.partial_ratio,
            "task_local_fraction": task.local_fraction,
            "latency_s": outcome.latency_s,
            "energy_j": outcome.energy_j,
            "reliability": outcome.reliability,
            "reliability_target": device.reliability_target,
            "reliability_satisfied": outcome.reliability_satisfied,
            "success": outcome.success,
            "deadline_violation": outcome.deadline_violation,
            "prediction_source": prediction.source,
            "prediction_available": prediction.available,
            "prediction_uncertainty": prediction.uncertainty,
            "data_size_mb": task.data_size_mb,
            "output_size_mb": task.output_size_mb,
            "cpu_cycles_mi": task.cpu_cycles_mi,
            "deadline_s": task.deadline_s,
            "release_time_s": task.release_time_s,
            "release_interval_s": task.release_interval_s,
            "tx_time_s": outcome.tx_time_s,
            "rx_time_s": outcome.rx_time_s,
            "queue_delay_s": outcome.queue_delay_s,
            "edge_compute_time_s": outcome.edge_compute_time_s,
            "local_compute_time_s": outcome.local_compute_time_s,
            "execution_overhead_s": outcome.execution_overhead_s,
            "prediction_overhead_s": prediction_overhead_s,
            "fuzzy_overhead_s": fuzzy_overhead_s,
            "simulator_apply_overhead_s": simulator_apply_overhead_s,
            "online_overhead_s": online_overhead_s,
            **reward_info,
        }
        self.last_info = info
        return observation, float(reward), terminated, False, info

    def _advance_to(self, target_time: int) -> None:
        if self.simulator is None:
            return
        while self.current_time < int(target_time):
            self.simulator.advance_one_slot()
            self.current_time += 1
            ready_rate = self._ready_counts.get(self.current_time, 0) / max(1, len(self.simulator.devices))
            self.predictor.observe(self.simulator, ready_task_rate=ready_rate)

    def _decode_action(self, action, candidates: List[EdgeServer], task: Task) -> OffloadingDecision:
        values = np.asarray(action, dtype=np.int64).reshape(-1)
        mode_index = int(np.clip(values[0] if values.size else 0, 0, 2))
        server_index = int(np.clip(values[1] if values.size > 1 else 0, 0, len(candidates) - 1))
        if mode_index == 0:
            return OffloadingDecision("local")
        server_id = candidates[server_index].server_id
        if mode_index == 1:
            return OffloadingDecision("edge", server_id, 1.0)
        return OffloadingDecision("partial", server_id, 1.0 - task.local_fraction)

    def _current_task(self) -> Tuple[Task, IoTDevice]:
        if self.simulator is None or self.current_index >= len(self.tasks):
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
        self._advance_to(task.arrival_slot)
        candidates = self._current_candidates(task, device)
        prediction_start = perf_counter()
        prediction = self.predictor.predict(self.simulator)
        self.last_prediction_overhead_s = perf_counter() - prediction_start
        self.last_prediction = prediction
        values: List[float] = self._task_features(task) + self._device_features(device)
        for server in candidates:
            values.extend(self._server_features(server, device))
        if self.env_config.include_prediction:
            values.extend(prediction.vector[: self.env_config.predictive_feature_size].tolist())
        if self.env_config.include_fuzzy_weights:
            weights = self.fuzzy_controller.compute_from_task_context(task, device, candidates, prediction.uncertainty).to_dict()
            values.extend([weights["energy"], weights["latency"], weights["success"], weights["reliability"]])
        arr = np.clip(np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=10.0, neginf=0.0), 0.0, 10.0)
        if arr.shape != (self.observation_dim,):
            raise RuntimeError(f"invalid observation size {arr.shape[0]}, expected {self.observation_dim}")
        return arr

    def _zero_observation(self) -> np.ndarray:
        return np.zeros((self.observation_dim,), dtype=np.float32)

    def _task_features(self, task: Task) -> List[float]:
        max_release = max(1.0, self.simulator.time_slots * self.simulator.slot_duration_s) if self.simulator else 1.0
        return [
            task.data_size_mb / 10.0,
            task.cpu_cycles_mi / 2500.0,
            task.deadline_s / 1.5,
            task.local_fraction / 0.5,
            task.release_interval_s / 0.5,
            task.release_time_s / max_release,
        ]

    def _device_features(self, device: IoTDevice) -> List[float]:
        return [
            device.cpu_frequency_ghz,
            device.tx_power_w / 0.1,
            device.compute_power_w / 0.01,
            device.reliability_target,
        ]

    def _server_features(self, server: EdgeServer, device: IoTDevice) -> List[float]:
        rate = self.simulator.channel.data_rate_mbps(server.bandwidth_mhz, device.tx_power_w, device.device_id, server.server_id)
        return [
            server.cores / 4.0,
            server.cpu_frequency_ghz / 10.0,
            server.bandwidth_mhz / 20.0,
            server.queue_delay() / 1.5,
            server.queue_workload_mi / max(server.cpu_capacity_mips, 1e-9),
            server.availability,
            rate / 500.0,
        ]

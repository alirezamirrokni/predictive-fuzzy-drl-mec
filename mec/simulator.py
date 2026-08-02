from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import yaml

from .channel import WirelessChannel
from .device import IoTDevice
from .edgesimpy_backend import EdgeSimPyBackend
from .metrics import MetricRecord, MetricsLogger
from .project_spec import validate_production_config
from .server import EdgeServer
from .task import Task


@dataclass(slots=True)
class OffloadingDecision:
    """partial_ratio is the remotely executed fraction."""

    mode: str
    server_id: Optional[int] = None
    partial_ratio: float = 0.0

    def normalized(self) -> "OffloadingDecision":
        if self.mode == "local":
            return OffloadingDecision("local", None, 0.0)
        if self.mode == "edge":
            if self.server_id is None:
                raise ValueError("edge decision requires server_id")
            return OffloadingDecision("edge", int(self.server_id), 1.0)
        if self.mode == "partial":
            if self.server_id is None:
                raise ValueError("partial decision requires server_id")
            ratio = min(1.0, max(0.0, float(self.partial_ratio)))
            if ratio <= 0.0:
                return OffloadingDecision("local", None, 0.0)
            if ratio >= 1.0:
                return OffloadingDecision("edge", int(self.server_id), 1.0)
            return OffloadingDecision("partial", int(self.server_id), ratio)
        raise ValueError("mode must be local, edge, or partial")


@dataclass(slots=True)
class SimulationOutcome:
    latency_s: float
    energy_j: float
    reliability: float
    reliability_satisfied: bool
    success: bool
    deadline_violation: bool
    failed_due_to_battery: bool = False
    failed_due_to_channel: bool = False
    failed_due_to_server: bool = False
    tx_time_s: float = 0.0
    rx_time_s: float = 0.0
    queue_delay_s: float = 0.0
    edge_compute_time_s: float = 0.0
    local_compute_time_s: float = 0.0

    @property
    def execution_overhead_s(self) -> float:
        return float(self.tx_time_s + self.rx_time_s + self.queue_delay_s)


class MECSimulator:
    def __init__(
        self,
        devices: List[IoTDevice],
        servers: List[EdgeServer],
        channel: WirelessChannel,
        slot_duration_s: float,
        time_slots: int,
        rng: np.random.Generator,
        require_edgesimpy: bool,
        scenario_sample: Dict[str, Any],
    ) -> None:
        self.devices = devices
        self.servers = servers
        self.channel = channel
        self.slot_duration_s = float(slot_duration_s)
        self.time_slots = int(time_slots)
        self.rng = rng
        self.metrics = MetricsLogger()
        self.server_index = {server.server_id: server for server in servers}
        self.device_index = {device.device_id: device for device in devices}
        self.current_slot = 0
        self.scenario_sample = dict(scenario_sample)
        for device in devices:
            device.validate()
        for server in servers:
            server.validate()
        self.backend = EdgeSimPyBackend(self.slot_duration_s, self._physics_tick, required=require_edgesimpy)
        self.scenario_sample["simulation_backend"] = self.backend.name

    def _physics_tick(self) -> None:
        for server in self.servers:
            server.process_slot(self.slot_duration_s)
        self.current_slot += 1

    def advance_one_slot(self) -> None:
        self.backend.step()

    def advance_to(self, target_slot: int) -> None:
        while self.current_slot < int(target_slot):
            self.advance_one_slot()

    def server_by_id(self, server_id: int) -> EdgeServer:
        if server_id not in self.server_index:
            raise KeyError(f"unknown server_id {server_id}")
        return self.server_index[server_id]

    def candidate_servers(self, device: IoTDevice, top_k: Optional[int] = None) -> List[EdgeServer]:
        servers = sorted(
            self.servers,
            key=lambda server: (
                server.queue_delay(),
                -server.availability,
                -server.cpu_capacity_mips,
                -self.channel.data_rate_mbps(server.bandwidth_mhz, device.tx_power_w, device.device_id, server.server_id),
            ),
        )
        if top_k is None or top_k <= 0:
            return servers
        return servers[: min(int(top_k), len(servers))]

    def all_candidate_decisions(
        self,
        include_partial: bool = True,
        device: Optional[IoTDevice] = None,
        task: Optional[Task] = None,
        top_k: Optional[int] = None,
    ) -> Iterable[OffloadingDecision]:
        yield OffloadingDecision("local")
        servers = self.servers if device is None else self.candidate_servers(device, top_k)
        for server in servers:
            yield OffloadingDecision("edge", server.server_id, 1.0)
            if include_partial:
                local_fraction = 0.3 if task is None else task.local_fraction
                yield OffloadingDecision("partial", server.server_id, 1.0 - local_fraction)

    def estimate(self, task: Task, device: IoTDevice, decision: OffloadingDecision) -> SimulationOutcome:
        decision = decision.normalized()
        if decision.mode == "local":
            return self._estimate_local(task, device)
        server = self.server_by_id(int(decision.server_id))
        if decision.mode == "edge":
            return self._estimate_remote(task, device, server, 1.0)
        return self._estimate_partial(task, device, server, float(decision.partial_ratio))

    def apply(
        self,
        time_slot: int,
        task: Task,
        device: IoTDevice,
        decision: OffloadingDecision,
        policy_overhead_s: float = 0.0,
        prediction_overhead_s: float = 0.0,
        fuzzy_overhead_s: float = 0.0,
        drl_overhead_s: float = 0.0,
        online_overhead_s: float = 0.0,
    ) -> SimulationOutcome:
        decision = decision.normalized()
        apply_start = perf_counter()
        outcome = self._sample_availability(device, decision, self.estimate(task, device, decision))
        if decision.mode in {"edge", "partial"} and decision.server_id is not None:
            self.server_by_id(decision.server_id).add_workload(task.cpu_cycles_mi * decision.partial_ratio)
        simulator_apply_overhead_s = perf_counter() - apply_start
        if online_overhead_s <= 0.0:
            online_overhead_s = policy_overhead_s + prediction_overhead_s + fuzzy_overhead_s + drl_overhead_s + simulator_apply_overhead_s
        self.metrics.log(
            MetricRecord(
                time_slot=int(time_slot),
                task_id=task.task_id,
                device_id=task.device_id,
                mode=decision.mode,
                server_id=-1 if decision.server_id is None else int(decision.server_id),
                partial_ratio=decision.partial_ratio,
                latency_s=outcome.latency_s,
                energy_j=outcome.energy_j,
                reliability=outcome.reliability,
                reliability_satisfied=int(outcome.reliability_satisfied),
                reliability_target=device.reliability_target,
                success=int(outcome.success),
                deadline_violation=int(outcome.deadline_violation),
                failed_due_to_battery=0,
                failed_due_to_channel=0,
                failed_due_to_server=int(outcome.failed_due_to_server),
                data_size_mb=task.data_size_mb,
                output_size_mb=task.output_size_mb,
                cpu_cycles_mi=task.cpu_cycles_mi,
                deadline_s=task.deadline_s,
                priority=1,
                arrival_time=task.arrival_slot,
                release_time_s=task.release_time_s,
                release_interval_s=task.release_interval_s,
                task_local_fraction=task.local_fraction,
                tx_time_s=outcome.tx_time_s,
                rx_time_s=outcome.rx_time_s,
                queue_delay_s=outcome.queue_delay_s,
                edge_compute_time_s=outcome.edge_compute_time_s,
                local_compute_time_s=outcome.local_compute_time_s,
                execution_overhead_s=outcome.execution_overhead_s,
                policy_overhead_s=policy_overhead_s,
                prediction_overhead_s=prediction_overhead_s,
                fuzzy_overhead_s=fuzzy_overhead_s,
                drl_overhead_s=drl_overhead_s,
                simulator_apply_overhead_s=simulator_apply_overhead_s,
                online_overhead_s=online_overhead_s,
            )
        )
        return outcome

    def run(self, policy: Any) -> MetricsLogger:
        for slot in range(self.time_slots):
            self.advance_to(slot)
            for device in self.devices:
                for task in device.pop_ready_tasks(slot):
                    start = perf_counter()
                    decision = policy.choose(self, task, device)
                    self.apply(slot, task, device, decision, policy_overhead_s=perf_counter() - start)
        return self.metrics

    def _estimate_local(self, task: Task, device: IoTDevice) -> SimulationOutcome:
        local_time = task.cpu_cycles_mi / device.cpu_capacity_mips
        energy = device.compute_power_w * local_time
        deadline_violation = local_time > task.deadline_s
        return SimulationOutcome(
            latency_s=local_time,
            energy_j=energy,
            reliability=1.0,
            reliability_satisfied=True,
            success=not deadline_violation,
            deadline_violation=deadline_violation,
            local_compute_time_s=local_time,
        )

    def _estimate_remote(self, task: Task, device: IoTDevice, server: EdgeServer, ratio: float) -> SimulationOutcome:
        rate = self.channel.data_rate_mbps(server.bandwidth_mhz, device.tx_power_w, device.device_id, server.server_id)
        tx_time = task.data_size_mb * 8.0 * ratio / rate
        rx_time = task.output_size_mb * 8.0 * ratio / rate
        queue_delay = server.queue_delay()
        compute_time = task.cpu_cycles_mi * ratio / server.cpu_capacity_mips
        latency = tx_time + queue_delay + compute_time + rx_time
        energy = device.tx_power_w * tx_time
        reliability = server.availability
        deadline_violation = latency > task.deadline_s
        return SimulationOutcome(
            latency_s=latency,
            energy_j=energy,
            reliability=reliability,
            reliability_satisfied=reliability >= device.reliability_target,
            success=not deadline_violation,
            deadline_violation=deadline_violation,
            tx_time_s=tx_time,
            rx_time_s=rx_time,
            queue_delay_s=queue_delay,
            edge_compute_time_s=compute_time,
        )

    def _estimate_partial(self, task: Task, device: IoTDevice, server: EdgeServer, offload_ratio: float) -> SimulationOutcome:
        remote = self._estimate_remote(task, device, server, offload_ratio)
        local_ratio = 1.0 - offload_ratio
        local_time = task.cpu_cycles_mi * local_ratio / device.cpu_capacity_mips
        local_energy = device.compute_power_w * local_time
        latency = max(local_time, remote.latency_s)
        energy = local_energy + remote.energy_j
        deadline_violation = latency > task.deadline_s
        return SimulationOutcome(
            latency_s=latency,
            energy_j=energy,
            reliability=remote.reliability,
            reliability_satisfied=remote.reliability_satisfied,
            success=not deadline_violation,
            deadline_violation=deadline_violation,
            tx_time_s=remote.tx_time_s,
            rx_time_s=remote.rx_time_s,
            queue_delay_s=remote.queue_delay_s,
            edge_compute_time_s=remote.edge_compute_time_s,
            local_compute_time_s=local_time,
        )

    def _sample_availability(self, device: IoTDevice, decision: OffloadingDecision, outcome: SimulationOutcome) -> SimulationOutcome:
        failed_server = False
        if decision.mode in {"edge", "partial"} and decision.server_id is not None:
            failed_server = bool(self.rng.random() > self.server_by_id(decision.server_id).availability)
        outcome.failed_due_to_server = failed_server
        outcome.success = bool(outcome.success and not failed_server)
        outcome.reliability_satisfied = bool(outcome.reliability >= device.reliability_target)
        return outcome


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    if not isinstance(loaded, dict):
        raise ValueError("configuration must be a mapping")
    return loaded


def _uniform(rng: np.random.Generator, bounds: Sequence[float]) -> float:
    if len(bounds) != 2 or float(bounds[0]) > float(bounds[1]):
        raise ValueError(f"invalid uniform bounds: {bounds}")
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def _integers(rng: np.random.Generator, bounds: Sequence[int]) -> int:
    if len(bounds) != 2 or int(bounds[0]) > int(bounds[1]):
        raise ValueError(f"invalid integer bounds: {bounds}")
    return int(rng.integers(int(bounds[0]), int(bounds[1]) + 1))


def _sample_count(rng: np.random.Generator, value: int | Sequence[int]) -> int:
    return int(value) if isinstance(value, (int, np.integer)) else _integers(rng, value)


def build_simulator_from_config(config: Dict[str, Any], seed_override: Optional[int] = None) -> MECSimulator:
    validate_production_config(config)
    seed = int(config.get("seed", 0) if seed_override is None else seed_override)
    rng = np.random.default_rng(seed)
    device_cfg = config["devices"]
    server_cfg = config["servers"]
    task_cfg = config["tasks"]
    sim_cfg = config["simulation"]
    channel_cfg = config["channel"]
    device_count = _sample_count(rng, device_cfg["count"])
    server_count = _sample_count(rng, server_cfg["count"])

    devices = [
        IoTDevice(
            device_id=i,
            cpu_frequency_ghz=_uniform(rng, device_cfg["processing_frequency_ghz"]),
            tx_power_w=_uniform(rng, device_cfg["transmission_power_mw"]) / 1000.0,
            compute_power_w=_uniform(rng, device_cfg["computation_power_mw"]) / 1000.0,
            reliability_target=_uniform(rng, device_cfg["reliability_target"]),
            position=tuple(rng.uniform(0.0, 1.0, size=2).tolist()),
        )
        for i in range(device_count)
    ]
    servers = [
        EdgeServer(
            server_id=i,
            cores=_integers(rng, server_cfg["cores"]),
            cpu_frequency_ghz=_uniform(rng, server_cfg["processing_frequency_ghz"]),
            bandwidth_mhz=_uniform(rng, server_cfg["bandwidth_mhz"]),
            availability=_uniform(rng, server_cfg["long_term_availability"]),
            position=tuple(rng.uniform(0.0, 1.0, size=2).tolist()),
        )
        for i in range(server_count)
    ]
    tick = float(sim_cfg["tick_duration_s"])
    tasks: List[Task] = []
    task_id = 0
    for device_id in range(device_count):
        release_time = 0.0
        for _ in range(int(task_cfg["tasks_per_device"])):
            interval = _uniform(rng, task_cfg["release_interval_s"])
            release_time += interval
            task = Task(
                task_id=task_id,
                device_id=device_id,
                data_size_mb=_uniform(rng, task_cfg["data_size_mb"]),
                cpu_cycles_mi=_uniform(rng, task_cfg["cpu_cycles"]) / 1e6,
                deadline_s=_uniform(rng, task_cfg["relative_deadline_s"]),
                local_fraction=_uniform(rng, task_cfg["local_fraction"]),
                release_interval_s=interval,
                release_time_s=release_time,
                arrival_slot=int(np.ceil(release_time / tick)),
            )
            task.validate()
            tasks.append(task)
            devices[device_id].add_task(task)
            task_id += 1
    time_slots = max((task.arrival_slot for task in tasks), default=0) + 1
    gain_low, gain_high = channel_cfg["channel_gain"]
    gain_matrix = rng.uniform(float(gain_low), float(gain_high), size=(device_count, server_count)).astype(np.float64)
    channel = WirelessChannel(noise_power_w=float(channel_cfg["noise_power_mw"]) / 1000.0, gain_matrix=gain_matrix)
    sample = {
        "seed": seed,
        "device_count": device_count,
        "server_count": server_count,
        "tasks_per_device": int(task_cfg["tasks_per_device"]),
        "total_tasks": len(tasks),
        "time_slots": time_slots,
        "tick_duration_s": tick,
    }
    return MECSimulator(
        devices=devices,
        servers=servers,
        channel=channel,
        slot_duration_s=tick,
        time_slots=time_slots,
        rng=rng,
        require_edgesimpy=bool(sim_cfg.get("require_edgesimpy", True)),
        scenario_sample=sample,
    )

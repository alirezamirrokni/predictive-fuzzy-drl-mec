from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import yaml

from .channel import WirelessChannel
from .device import IoTDevice
from .metrics import MetricRecord, MetricsLogger
from .mobility import MobilityModel
from .server import EdgeServer
from .task import Task


@dataclass(slots=True)
class OffloadingDecision:
    mode: str
    server_id: Optional[int] = None
    partial_ratio: float = 0.0

    def normalized(self) -> "OffloadingDecision":
        if self.mode == "local":
            return OffloadingDecision("local", None, 0.0)
        if self.mode == "edge":
            if self.server_id is None:
                raise ValueError("edge decision requires server_id")
            return OffloadingDecision("edge", self.server_id, 1.0)
        if self.mode == "partial":
            if self.server_id is None:
                raise ValueError("partial decision requires server_id")
            ratio = min(1.0, max(0.0, float(self.partial_ratio)))
            if ratio == 0.0:
                return OffloadingDecision("local", None, 0.0)
            if ratio == 1.0:
                return OffloadingDecision("edge", self.server_id, 1.0)
            return OffloadingDecision("partial", self.server_id, ratio)
        raise ValueError("mode must be local, edge, or partial")


@dataclass(slots=True)
class SimulationOutcome:
    latency_s: float
    energy_j: float
    reliability: float
    success: bool
    deadline_violation: bool
    failed_due_to_battery: bool
    failed_due_to_channel: bool
    failed_due_to_server: bool
    tx_time_s: float = 0.0
    rx_time_s: float = 0.0
    queue_delay_s: float = 0.0
    edge_compute_time_s: float = 0.0
    local_compute_time_s: float = 0.0

    @property
    def execution_overhead_s(self) -> float:
        """Non-compute communication/queueing overhead until the task leaves the system."""
        return float(self.tx_time_s + self.rx_time_s + self.queue_delay_s)


class MECSimulator:
    def __init__(
        self,
        devices: List[IoTDevice],
        servers: List[EdgeServer],
        channel: WirelessChannel,
        mobility: MobilityModel,
        slot_duration_s: float,
        time_slots: int,
        rng: np.random.Generator,
    ) -> None:
        self.devices = devices
        self.servers = servers
        self.channel = channel
        self.mobility = mobility
        self.slot_duration_s = slot_duration_s
        self.time_slots = time_slots
        self.rng = rng
        self.metrics = MetricsLogger()
        self.server_index = {server.server_id: server for server in servers}
        self.device_index = {device.device_id: device for device in devices}
        for device in devices:
            device.validate()
        for server in servers:
            server.validate()

    def server_by_id(self, server_id: int) -> EdgeServer:
        if server_id not in self.server_index:
            raise KeyError(f"unknown server_id {server_id}")
        return self.server_index[server_id]

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
        outcome = self.estimate(task, device, decision)
        outcome = self._sample_failures(task, device, decision, outcome)
        device.consume_energy(outcome.energy_j)
        if decision.mode in {"edge", "partial"} and decision.server_id is not None:
            server = self.server_by_id(decision.server_id)
            server.add_workload(task.cpu_cycles_mi * decision.partial_ratio)
        simulator_apply_overhead_s = perf_counter() - apply_start
        if online_overhead_s <= 0.0:
            online_overhead_s = policy_overhead_s + prediction_overhead_s + fuzzy_overhead_s + drl_overhead_s + simulator_apply_overhead_s
        self.metrics.log(
            MetricRecord(
                time_slot=time_slot,
                task_id=task.task_id,
                device_id=task.device_id,
                mode=decision.mode,
                server_id=-1 if decision.server_id is None else int(decision.server_id),
                partial_ratio=decision.partial_ratio,
                latency_s=outcome.latency_s,
                energy_j=outcome.energy_j,
                reliability=outcome.reliability,
                success=int(outcome.success),
                deadline_violation=int(outcome.deadline_violation),
                failed_due_to_battery=int(outcome.failed_due_to_battery),
                failed_due_to_channel=int(outcome.failed_due_to_channel),
                failed_due_to_server=int(outcome.failed_due_to_server),
                data_size_mb=task.data_size_mb,
                output_size_mb=task.output_size_mb,
                cpu_cycles_mi=task.cpu_cycles_mi,
                deadline_s=task.deadline_s,
                priority=task.priority,
                arrival_time=task.arrival_time,
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
        for time_slot in range(self.time_slots):
            for server in self.servers:
                server.process_slot(self.slot_duration_s)
            for device in self.devices:
                self.mobility.update(device, self.rng, self.slot_duration_s)
            for device in self.devices:
                ready_tasks = device.pop_ready_tasks(time_slot)
                for task in ready_tasks:
                    policy_start = perf_counter()
                    decision = policy.choose(self, task, device)
                    policy_overhead_s = perf_counter() - policy_start
                    self.apply(time_slot, task, device, decision, policy_overhead_s=policy_overhead_s)
        remaining = []
        for device in self.devices:
            remaining.extend(device.task_queue)
            device.task_queue.clear()
        if remaining:
            last_slot = max(0, self.time_slots - 1)
            for task in remaining:
                device = self.device_index[task.device_id]
                policy_start = perf_counter()
                decision = policy.choose(self, task, device)
                policy_overhead_s = perf_counter() - policy_start
                self.apply(last_slot, task, device, decision, policy_overhead_s=policy_overhead_s)
        return self.metrics

    def candidate_servers(self, device: IoTDevice, top_k: Optional[int] = None) -> List[EdgeServer]:
        servers = sorted(
            self.servers,
            key=lambda server: (
                server.queue_delay(),
                self.channel.distance_m(device.position, server.position),
                -server.cpu_capacity_mips,
            ),
        )
        if top_k is None or top_k <= 0:
            return servers
        return servers[: min(top_k, len(servers))]

    def all_candidate_decisions(self, include_partial: bool = True, device: Optional[IoTDevice] = None, top_k: Optional[int] = None) -> Iterable[OffloadingDecision]:
        yield OffloadingDecision("local")
        servers = self.servers if device is None else self.candidate_servers(device, top_k)
        for server in servers:
            yield OffloadingDecision("edge", server.server_id, 1.0)
            if include_partial:
                for ratio in (0.25, 0.5, 0.75):
                    yield OffloadingDecision("partial", server.server_id, ratio)

    def _estimate_local(self, task: Task, device: IoTDevice) -> SimulationOutcome:
        local_time = task.cpu_cycles_mi / device.cpu_capacity_mips
        energy = device.local_power_w * local_time
        reliability = max(0.0, 1.0 - device.failure_probability)
        failed_battery = device.battery_j < energy
        deadline_violation = local_time > task.deadline_s
        success = not failed_battery and not deadline_violation
        return SimulationOutcome(
            latency_s=local_time,
            energy_j=energy,
            reliability=reliability,
            success=success,
            deadline_violation=deadline_violation,
            failed_due_to_battery=failed_battery,
            failed_due_to_channel=False,
            failed_due_to_server=False,
            local_compute_time_s=local_time,
        )

    def _sample_failures(self, task: Task, device: IoTDevice, decision: OffloadingDecision, outcome: SimulationOutcome) -> SimulationOutcome:
        failed_channel = False
        failed_server = False
        failed_device = False
        if decision.mode == "local":
            failed_device = self.rng.random() < device.failure_probability
        if decision.mode in {"edge", "partial"} and decision.server_id is not None:
            server = self.server_by_id(decision.server_id)
            failed_server = self.rng.random() < server.failure_probability
            failed_channel = self.rng.random() < self.channel.packet_loss_probability(device.position, server.position)
        if decision.mode == "partial":
            failed_device = self.rng.random() < device.failure_probability
        success = outcome.success and not failed_channel and not failed_server and not failed_device
        return SimulationOutcome(
            latency_s=outcome.latency_s,
            energy_j=outcome.energy_j,
            reliability=outcome.reliability,
            success=success,
            deadline_violation=outcome.deadline_violation,
            failed_due_to_battery=outcome.failed_due_to_battery,
            failed_due_to_channel=failed_channel,
            failed_due_to_server=failed_server or failed_device,
            tx_time_s=outcome.tx_time_s,
            rx_time_s=outcome.rx_time_s,
            queue_delay_s=outcome.queue_delay_s,
            edge_compute_time_s=outcome.edge_compute_time_s,
            local_compute_time_s=outcome.local_compute_time_s,
        )

    def _estimate_remote(self, task: Task, device: IoTDevice, server: EdgeServer, ratio: float) -> SimulationOutcome:
        data_rate = self.channel.data_rate_mbps(server.bandwidth_mhz, device.tx_power_w, device.position, server.position)
        tx_time = task.data_size_mb * 8.0 * ratio / data_rate
        rx_time = task.output_size_mb * 8.0 * ratio / data_rate
        queue_delay = server.queue_delay()
        compute_time = task.cpu_cycles_mi * ratio / server.cpu_capacity_mips
        latency = tx_time + queue_delay + compute_time + rx_time
        energy = device.tx_power_w * tx_time
        packet_loss_probability = self.channel.packet_loss_probability(device.position, server.position)
        reliability = max(0.0, (1.0 - packet_loss_probability) * server.reliability)
        failed_battery = device.battery_j < energy
        deadline_violation = latency > task.deadline_s
        success = not failed_battery and not deadline_violation
        return SimulationOutcome(
            latency_s=latency,
            energy_j=energy,
            reliability=reliability,
            success=success,
            deadline_violation=deadline_violation,
            failed_due_to_battery=failed_battery,
            failed_due_to_channel=False,
            failed_due_to_server=False,
            tx_time_s=tx_time,
            rx_time_s=rx_time,
            queue_delay_s=queue_delay,
            edge_compute_time_s=compute_time,
        )

    def _estimate_partial(self, task: Task, device: IoTDevice, server: EdgeServer, ratio: float) -> SimulationOutcome:
        remote = self._estimate_remote(task, device, server, ratio)
        local_cycles = task.cpu_cycles_mi * (1.0 - ratio)
        local_time = local_cycles / device.cpu_capacity_mips
        local_energy = device.local_power_w * local_time
        latency = max(local_time, remote.latency_s)
        energy = local_energy + remote.energy_j
        local_reliability = max(0.0, 1.0 - device.failure_probability)
        reliability = local_reliability * remote.reliability
        failed_battery = device.battery_j < energy
        deadline_violation = latency > task.deadline_s
        success = remote.success and not failed_battery and not deadline_violation
        return SimulationOutcome(
            latency_s=latency,
            energy_j=energy,
            reliability=reliability,
            success=success,
            deadline_violation=deadline_violation,
            failed_due_to_battery=failed_battery,
            failed_due_to_channel=remote.failed_due_to_channel,
            failed_due_to_server=remote.failed_due_to_server,
            tx_time_s=remote.tx_time_s,
            rx_time_s=remote.rx_time_s,
            queue_delay_s=remote.queue_delay_s,
            edge_compute_time_s=remote.edge_compute_time_s,
            local_compute_time_s=local_time,
        )


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    if not isinstance(loaded, dict):
        raise ValueError("configuration must be a mapping")
    return loaded


def build_simulator_from_config(config: Dict[str, Any]) -> MECSimulator:
    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed)
    area = config["area"]
    mobility = MobilityModel(float(area["width_m"]), float(area["height_m"]))
    devices = _generate_devices(config, mobility, rng)
    servers = _generate_servers(config, mobility, rng)
    tasks = _generate_tasks(config, rng)
    for task in tasks:
        devices[task.device_id].add_task(task)
    channel_config = config["channel"]
    channel = WirelessChannel(
        noise_power_w=float(channel_config["noise_power_w"]),
        path_loss_exponent=float(channel_config["path_loss_exponent"]),
        reference_gain=float(channel_config["reference_gain"]),
        min_rate_mbps=float(channel_config["min_rate_mbps"]),
        packet_loss_base=float(channel_config["packet_loss_base"]),
    )
    return MECSimulator(
        devices=devices,
        servers=servers,
        channel=channel,
        mobility=mobility,
        slot_duration_s=float(config["simulation"]["slot_duration_s"]),
        time_slots=int(config["simulation"]["time_slots"]),
        rng=rng,
    )


def _uniform(rng: np.random.Generator, bounds: List[float]) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def _integers(rng: np.random.Generator, bounds: List[int]) -> int:
    return int(rng.integers(int(bounds[0]), int(bounds[1]) + 1))


def _generate_devices(config: Dict[str, Any], mobility: MobilityModel, rng: np.random.Generator) -> List[IoTDevice]:
    device_config = config["devices"]
    devices = []
    for device_id in range(int(device_config["count"])):
        devices.append(
            IoTDevice(
                device_id=device_id,
                cpu_capacity_mips=_uniform(rng, device_config["cpu_capacity_mips"]),
                battery_j=_uniform(rng, device_config["battery_j"]),
                tx_power_w=_uniform(rng, device_config["tx_power_w"]),
                local_power_w=_uniform(rng, device_config["local_power_w"]),
                position=mobility.random_position(rng),
                mobility_speed_mps=_uniform(rng, device_config["mobility_speed_mps"]),
                failure_probability=_uniform(rng, device_config["failure_probability"]),
            )
        )
    return devices


def _generate_servers(config: Dict[str, Any], mobility: MobilityModel, rng: np.random.Generator) -> List[EdgeServer]:
    server_config = config["servers"]
    servers = []
    for server_id in range(int(server_config["count"])):
        servers.append(
            EdgeServer(
                server_id=server_id,
                cpu_capacity_mips=_uniform(rng, server_config["cpu_capacity_mips"]),
                bandwidth_mhz=_uniform(rng, server_config["bandwidth_mhz"]),
                position=mobility.random_position(rng),
                failure_probability=_uniform(rng, server_config["failure_probability"]),
                static_power_w=_uniform(rng, server_config["static_power_w"]),
                dynamic_power_w=_uniform(rng, server_config["dynamic_power_w"]),
            )
        )
    return servers


def _generate_tasks(config: Dict[str, Any], rng: np.random.Generator) -> List[Task]:
    device_count = int(config["devices"]["count"])
    task_config = config["tasks"]
    tasks_per_device = int(task_config["tasks_per_device"])
    time_slots = int(config["simulation"]["time_slots"])
    tasks = []
    task_id = 0
    for device_id in range(device_count):
        for _ in range(tasks_per_device):
            task = Task(
                task_id=task_id,
                device_id=device_id,
                data_size_mb=_uniform(rng, task_config["data_size_mb"]),
                output_size_mb=_uniform(rng, task_config["output_size_mb"]),
                cpu_cycles_mi=_uniform(rng, task_config["cpu_cycles_mi"]),
                deadline_s=_uniform(rng, task_config["deadline_s"]),
                priority=_integers(rng, task_config["priority"]),
                arrival_time=int(rng.integers(0, max(1, time_slots))),
            )
            task.validate()
            tasks.append(task)
            task_id += 1
    tasks.sort(key=lambda item: (item.arrival_time, item.device_id, item.task_id))
    return tasks

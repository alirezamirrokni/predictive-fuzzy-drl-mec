from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from mec.device import IoTDevice
from mec.server import EdgeServer
from mec.simulator import MECSimulator
from mec.task import Task


@dataclass(slots=True)
class CandidateSelectorConfig:
    top_k: int = 16


class CandidateSelector:
    """Deterministic, information-only preselection shared by both DRL methods."""

    def __init__(self, config: CandidateSelectorConfig | None = None) -> None:
        self.config = CandidateSelectorConfig() if config is None else config

    def score(self, simulator: MECSimulator, device: IoTDevice, server: EdgeServer) -> float:
        rate = simulator.channel.data_rate_mbps(server.bandwidth_mhz, device.tx_power_w, device.device_id, server.server_id)
        load = server.queue_workload_mi / max(server.cpu_capacity_mips, 1e-9)
        # Lower is better. Terms are normalized only by documented maxima.
        return load + server.queue_delay() / 1.5 - server.availability - server.cpu_frequency_ghz / 10.0 - rate / 500.0

    def select(self, simulator: MECSimulator, task: Task, device: IoTDevice) -> List[EdgeServer]:
        _ = task
        servers = sorted(simulator.servers, key=lambda server: self.score(simulator, device, server))
        return servers[: max(1, min(int(self.config.top_k), len(servers)))]

    def pad(self, servers: Sequence[EdgeServer]) -> List[EdgeServer]:
        if not servers:
            raise ValueError("at least one server is required")
        result = list(servers)
        while len(result) < self.config.top_k:
            result.append(result[-1])
        return result[: self.config.top_k]

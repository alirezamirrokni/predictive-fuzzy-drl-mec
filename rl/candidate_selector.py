from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from mec.device import IoTDevice
from mec.server import EdgeServer
from mec.simulator import MECSimulator
from mec.task import Task


@dataclass(slots=True)
class CandidateSelectorConfig:
    top_k: int = 5
    queue_weight: float = 1.0
    distance_weight: float = 1.0
    cpu_weight: float = 0.5
    bandwidth_weight: float = 0.3
    reliability_weight: float = 0.5


class CandidateSelector:
    def __init__(self, config: CandidateSelectorConfig | None = None) -> None:
        self.config = CandidateSelectorConfig() if config is None else config

    def score(self, simulator: MECSimulator, device: IoTDevice, server: EdgeServer) -> float:
        distance = simulator.channel.distance_m(device.position, server.position)
        area_diag = max(1.0, (simulator.mobility.area_width_m ** 2 + simulator.mobility.area_height_m ** 2) ** 0.5)
        queue = server.queue_delay()
        distance_term = distance / area_diag
        cpu_term = server.cpu_capacity_mips / 30000.0
        bandwidth_term = server.bandwidth_mhz / 100.0
        reliability_term = server.reliability
        return (
            self.config.queue_weight * queue
            + self.config.distance_weight * distance_term
            - self.config.cpu_weight * cpu_term
            - self.config.bandwidth_weight * bandwidth_term
            - self.config.reliability_weight * reliability_term
        )

    def select(self, simulator: MECSimulator, task: Task, device: IoTDevice) -> List[EdgeServer]:
        servers = sorted(simulator.servers, key=lambda server: self.score(simulator, device, server))
        k = max(1, min(int(self.config.top_k), len(servers)))
        return servers[:k]

    def pad(self, servers: Sequence[EdgeServer]) -> List[EdgeServer]:
        if not servers:
            raise ValueError("at least one server is required")
        result = list(servers)
        while len(result) < self.config.top_k:
            result.append(result[-1])
        return result[: self.config.top_k]

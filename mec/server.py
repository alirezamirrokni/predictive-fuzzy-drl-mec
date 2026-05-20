from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(slots=True)
class EdgeServer:
    server_id: int
    cpu_capacity_mips: float
    bandwidth_mhz: float
    position: Tuple[float, float]
    failure_probability: float
    static_power_w: float
    dynamic_power_w: float
    queue_workload_mi: float = 0.0

    def validate(self) -> None:
        if self.cpu_capacity_mips <= 0:
            raise ValueError("cpu_capacity_mips must be positive")
        if self.bandwidth_mhz <= 0:
            raise ValueError("bandwidth_mhz must be positive")
        if not 0 <= self.failure_probability <= 1:
            raise ValueError("failure_probability must be in [0, 1]")
        if self.static_power_w < 0:
            raise ValueError("static_power_w must be non-negative")
        if self.dynamic_power_w < 0:
            raise ValueError("dynamic_power_w must be non-negative")
        if self.queue_workload_mi < 0:
            raise ValueError("queue_workload_mi must be non-negative")

    def queue_delay(self) -> float:
        return self.queue_workload_mi / self.cpu_capacity_mips

    def add_workload(self, workload_mi: float) -> None:
        self.queue_workload_mi += max(0.0, workload_mi)

    def process_slot(self, slot_duration_s: float) -> None:
        processed = self.cpu_capacity_mips * slot_duration_s
        self.queue_workload_mi = max(0.0, self.queue_workload_mi - processed)

    @property
    def reliability(self) -> float:
        return 1.0 - self.failure_probability

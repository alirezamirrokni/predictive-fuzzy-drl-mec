from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(slots=True)
class EdgeServer:
    """Heterogeneous edge server sampled from the documented ranges."""

    server_id: int
    cores: int
    cpu_frequency_ghz: float
    bandwidth_mhz: float
    availability: float
    position: Tuple[float, float]
    queue_workload_mi: float = 0.0

    def validate(self) -> None:
        if self.cores <= 0 or self.cpu_frequency_ghz <= 0:
            raise ValueError("server cores/frequency must be positive")
        if self.bandwidth_mhz <= 0:
            raise ValueError("bandwidth_mhz must be positive")
        if not 0.0 <= self.availability <= 1.0:
            raise ValueError("availability must be in [0, 1]")
        if self.queue_workload_mi < 0:
            raise ValueError("queue_workload_mi must be non-negative")

    @property
    def cpu_capacity_mips(self) -> float:
        return float(self.cores) * self.cpu_frequency_ghz * 1000.0

    @property
    def reliability(self) -> float:
        return self.availability

    def queue_delay(self) -> float:
        return self.queue_workload_mi / self.cpu_capacity_mips

    def add_workload(self, workload_mi: float) -> None:
        self.queue_workload_mi += max(0.0, workload_mi)

    def process_slot(self, slot_duration_s: float) -> None:
        processed = self.cpu_capacity_mips * slot_duration_s
        self.queue_workload_mi = max(0.0, self.queue_workload_mi - processed)

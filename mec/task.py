from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Task:
    task_id: int
    device_id: int
    data_size_mb: float
    cpu_cycles_mi: float
    deadline_s: float
    local_fraction: float
    release_interval_s: float
    release_time_s: float
    arrival_slot: int

    def validate(self) -> None:
        if self.data_size_mb <= 0 or self.cpu_cycles_mi <= 0:
            raise ValueError("task size and cycles must be positive")
        if self.deadline_s <= 0 or self.release_interval_s <= 0:
            raise ValueError("deadline/release interval must be positive")
        if not 0.0 < self.local_fraction < 1.0:
            raise ValueError("local_fraction must be in (0, 1)")
        if self.release_time_s < 0 or self.arrival_slot < 0:
            raise ValueError("release time must be non-negative")

    @property
    def output_size_mb(self) -> float:
        # No independent result-size parameter is supplied. A conservative
        # symmetric payload is used for returning the completed result.
        return self.data_size_mb

    @property
    def arrival_time(self) -> int:
        return self.arrival_slot

    @property
    def priority(self) -> int:
        # Priority is not present in the document/table; all tasks are equal.
        return 1

    @property
    def compute_density(self) -> float:
        return self.cpu_cycles_mi / self.data_size_mb

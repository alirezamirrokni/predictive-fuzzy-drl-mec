from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Task:
    task_id: int
    device_id: int
    data_size_mb: float
    output_size_mb: float
    cpu_cycles_mi: float
    deadline_s: float
    priority: int
    arrival_time: int

    def validate(self) -> None:
        if self.data_size_mb <= 0:
            raise ValueError("data_size_mb must be positive")
        if self.output_size_mb < 0:
            raise ValueError("output_size_mb must be non-negative")
        if self.cpu_cycles_mi <= 0:
            raise ValueError("cpu_cycles_mi must be positive")
        if self.deadline_s <= 0:
            raise ValueError("deadline_s must be positive")
        if self.priority < 1:
            raise ValueError("priority must be at least 1")
        if self.arrival_time < 0:
            raise ValueError("arrival_time must be non-negative")

    @property
    def compute_density(self) -> float:
        return self.cpu_cycles_mi / self.data_size_mb

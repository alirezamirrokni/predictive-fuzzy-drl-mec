from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from .task import Task


@dataclass(slots=True)
class IoTDevice:
    device_id: int
    cpu_capacity_mips: float
    battery_j: float
    tx_power_w: float
    local_power_w: float
    position: Tuple[float, float]
    mobility_speed_mps: float
    failure_probability: float
    task_queue: List[Task] = field(default_factory=list)

    def validate(self) -> None:
        if self.cpu_capacity_mips <= 0:
            raise ValueError("cpu_capacity_mips must be positive")
        if self.battery_j < 0:
            raise ValueError("battery_j must be non-negative")
        if self.tx_power_w <= 0:
            raise ValueError("tx_power_w must be positive")
        if self.local_power_w <= 0:
            raise ValueError("local_power_w must be positive")
        if self.mobility_speed_mps < 0:
            raise ValueError("mobility_speed_mps must be non-negative")
        if not 0 <= self.failure_probability <= 1:
            raise ValueError("failure_probability must be in [0, 1]")

    def add_task(self, task: Task) -> None:
        if task.device_id != self.device_id:
            raise ValueError("task device_id does not match device")
        self.task_queue.append(task)

    def pop_ready_tasks(self, time_slot: int) -> List[Task]:
        ready = [task for task in self.task_queue if task.arrival_time <= time_slot]
        self.task_queue = [task for task in self.task_queue if task.arrival_time > time_slot]
        return ready

    def consume_energy(self, energy_j: float) -> None:
        self.battery_j = max(0.0, self.battery_j - max(0.0, energy_j))

    @property
    def battery_empty(self) -> bool:
        return self.battery_j <= 0

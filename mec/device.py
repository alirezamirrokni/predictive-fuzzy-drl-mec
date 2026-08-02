from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from .task import Task


@dataclass(slots=True)
class IoTDevice:
    """IoT device whose physical fields come directly from the project table."""

    device_id: int
    cpu_frequency_ghz: float
    tx_power_w: float
    compute_power_w: float
    reliability_target: float
    position: Tuple[float, float]
    task_queue: List[Task] = field(default_factory=list)

    def validate(self) -> None:
        if self.cpu_frequency_ghz <= 0:
            raise ValueError("cpu_frequency_ghz must be positive")
        if self.tx_power_w <= 0 or self.compute_power_w <= 0:
            raise ValueError("power values must be positive")
        if not 0.0 <= self.reliability_target <= 1.0:
            raise ValueError("reliability_target must be in [0, 1]")

    @property
    def cpu_capacity_mips(self) -> float:
        # One CPU cycle per instruction: 1 GHz == 1000 MIPS.
        return self.cpu_frequency_ghz * 1000.0

    @property
    def local_power_w(self) -> float:
        return self.compute_power_w

    def add_task(self, task: Task) -> None:
        if task.device_id != self.device_id:
            raise ValueError("task device_id does not match device")
        self.task_queue.append(task)

    def pop_ready_tasks(self, time_slot: int) -> List[Task]:
        ready = [task for task in self.task_queue if task.arrival_slot <= time_slot]
        self.task_queue = [task for task in self.task_queue if task.arrival_slot > time_slot]
        return ready

    def consume_energy(self, energy_j: float) -> None:
        # Battery capacity is not specified in the document/table, so no invented
        # battery constraint is introduced. Energy is measured, not subtracted.
        _ = energy_j

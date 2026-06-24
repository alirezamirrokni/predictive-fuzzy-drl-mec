from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List


@dataclass(slots=True)
class MetricRecord:
    time_slot: int
    task_id: int
    device_id: int
    mode: str
    server_id: int
    partial_ratio: float
    latency_s: float
    energy_j: float
    reliability: float
    success: int
    deadline_violation: int
    failed_due_to_battery: int
    failed_due_to_channel: int
    failed_due_to_server: int
    data_size_mb: float = 0.0
    output_size_mb: float = 0.0
    cpu_cycles_mi: float = 0.0
    deadline_s: float = 0.0
    priority: int = 0
    arrival_time: int = 0
    tx_time_s: float = 0.0
    rx_time_s: float = 0.0
    queue_delay_s: float = 0.0
    edge_compute_time_s: float = 0.0
    local_compute_time_s: float = 0.0
    execution_overhead_s: float = 0.0
    policy_overhead_s: float = 0.0
    prediction_overhead_s: float = 0.0
    fuzzy_overhead_s: float = 0.0
    drl_overhead_s: float = 0.0
    simulator_apply_overhead_s: float = 0.0
    online_overhead_s: float = 0.0


class MetricsLogger:
    def __init__(self) -> None:
        self.records: List[MetricRecord] = []

    def log(self, record: MetricRecord) -> None:
        self.records.append(record)

    def summary(self) -> Dict[str, float]:
        if not self.records:
            return {
                "tasks": 0,
                "success_ratio": 0.0,
                "average_latency_s": 0.0,
                "average_energy_j": 0.0,
                "average_reliability": 0.0,
                "deadline_violation_rate": 0.0,
                "battery_failure_rate": 0.0,
                "channel_failure_rate": 0.0,
                "server_failure_rate": 0.0,
                "local_ratio": 0.0,
                "edge_ratio": 0.0,
                "partial_ratio": 0.0,
                "average_execution_overhead_s": 0.0,
                "average_online_overhead_s": 0.0,
                "total_online_overhead_s": 0.0,
            }
        n = len(self.records)
        return {
            "tasks": n,
            "success_ratio": sum(r.success for r in self.records) / n,
            "average_latency_s": mean(r.latency_s for r in self.records),
            "average_energy_j": mean(r.energy_j for r in self.records),
            "average_reliability": mean(r.reliability for r in self.records),
            "deadline_violation_rate": sum(r.deadline_violation for r in self.records) / n,
            "battery_failure_rate": sum(r.failed_due_to_battery for r in self.records) / n,
            "channel_failure_rate": sum(r.failed_due_to_channel for r in self.records) / n,
            "server_failure_rate": sum(r.failed_due_to_server for r in self.records) / n,
            "local_ratio": sum(r.mode == "local" for r in self.records) / n,
            "edge_ratio": sum(r.mode == "edge" for r in self.records) / n,
            "partial_ratio": sum(r.mode == "partial" for r in self.records) / n,
            "average_execution_overhead_s": mean(r.execution_overhead_s for r in self.records),
            "average_online_overhead_s": mean(r.online_overhead_s for r in self.records),
            "total_online_overhead_s": sum(r.online_overhead_s for r in self.records),
        }

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(MetricRecord.__dataclass_fields__.keys())
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.records:
                writer.writerow(asdict(record))

    def summary_to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(self.summary(), file, indent=2)

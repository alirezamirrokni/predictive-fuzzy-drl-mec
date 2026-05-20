from __future__ import annotations

from baselines.greedy_latency import GreedyLatencyPolicy
from baselines.local_only import LocalOnlyPolicy
from mec.simulator import OffloadingDecision, build_simulator_from_config, load_yaml_config


def tiny_config():
    return {
        "seed": 1,
        "scenario_name": "tiny",
        "simulation": {"time_slots": 5, "slot_duration_s": 1.0},
        "area": {"width_m": 100.0, "height_m": 100.0},
        "devices": {
            "count": 3,
            "cpu_capacity_mips": [400.0, 800.0],
            "battery_j": [1000.0, 2000.0],
            "tx_power_w": [0.3, 0.5],
            "local_power_w": [0.6, 1.0],
            "mobility_speed_mps": [0.0, 0.5],
            "failure_probability": [0.0, 0.001],
        },
        "servers": {
            "count": 2,
            "cpu_capacity_mips": [8000.0, 10000.0],
            "bandwidth_mhz": [20.0, 30.0],
            "failure_probability": [0.0, 0.001],
            "static_power_w": [60.0, 80.0],
            "dynamic_power_w": [30.0, 40.0],
        },
        "tasks": {
            "tasks_per_device": 4,
            "data_size_mb": [0.2, 1.0],
            "output_size_mb": [0.02, 0.1],
            "cpu_cycles_mi": [300.0, 800.0],
            "deadline_s": [1.0, 8.0],
            "priority": [1, 3],
        },
        "channel": {
            "noise_power_w": 1.0e-12,
            "path_loss_exponent": 3.0,
            "reference_gain": 1.0e-3,
            "min_rate_mbps": 0.1,
            "packet_loss_base": 0.0,
        },
    }


def test_simulator_generates_expected_number_of_tasks():
    simulator = build_simulator_from_config(tiny_config())
    total = sum(len(device.task_queue) for device in simulator.devices)
    assert total == 12


def test_local_only_runs_all_tasks():
    simulator = build_simulator_from_config(tiny_config())
    metrics = simulator.run(LocalOnlyPolicy())
    summary = metrics.summary()
    assert summary["tasks"] == 12
    assert 0.0 <= summary["success_ratio"] <= 1.0


def test_greedy_latency_returns_valid_decision():
    simulator = build_simulator_from_config(tiny_config())
    device = simulator.devices[0]
    task = device.task_queue[0]
    decision = GreedyLatencyPolicy().choose(simulator, task, device)
    assert isinstance(decision, OffloadingDecision)
    assert decision.mode in {"local", "edge", "partial"}

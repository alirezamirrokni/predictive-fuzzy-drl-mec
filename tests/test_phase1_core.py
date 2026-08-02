from __future__ import annotations

from baselines.local_only import LocalOnlyPolicy
from mec.simulator import OffloadingDecision, build_simulator_from_config, load_yaml_config


def debug_config():
    return load_yaml_config("configs/debug.yaml")


def test_documented_ranges_are_sampled_per_entity_and_task():
    simulator = build_simulator_from_config(debug_config(), seed_override=123)
    assert len(simulator.devices) == 5
    assert len(simulator.servers) == 3
    assert sum(len(d.task_queue) for d in simulator.devices) == 20
    assert all(0.5 <= d.cpu_frequency_ghz <= 1.0 for d in simulator.devices)
    assert all(0.05 <= d.tx_power_w <= 0.10 for d in simulator.devices)
    assert all(0.005 <= d.compute_power_w <= 0.010 for d in simulator.devices)
    assert all(0.93 <= d.reliability_target <= 0.98 for d in simulator.devices)
    assert all(2 <= s.cores <= 4 for s in simulator.servers)
    assert all(2.0 <= s.cpu_frequency_ghz <= 10.0 for s in simulator.servers)
    tasks = [t for d in simulator.devices for t in d.task_queue]
    assert all(5.0 <= t.data_size_mb <= 10.0 for t in tasks)
    assert all(1000.0 <= t.cpu_cycles_mi <= 2500.0 for t in tasks)
    assert all(0.1 <= t.local_fraction <= 0.5 for t in tasks)
    assert all(0.2 <= t.deadline_s <= 1.5 for t in tasks)


def test_seed_override_changes_sample_but_is_reproducible():
    a = build_simulator_from_config(debug_config(), seed_override=99)
    b = build_simulator_from_config(debug_config(), seed_override=99)
    c = build_simulator_from_config(debug_config(), seed_override=100)
    assert a.devices[0].cpu_frequency_ghz == b.devices[0].cpu_frequency_ghz
    assert a.devices[0].cpu_frequency_ghz != c.devices[0].cpu_frequency_ghz


def test_local_only_runs_all_tasks():
    simulator = build_simulator_from_config(debug_config())
    summary = simulator.run(LocalOnlyPolicy()).summary()
    assert summary["tasks"] == 20
    assert 0.0 <= summary["success_ratio"] <= 1.0
    assert summary["total_energy_j"] > 0.0


def test_partial_uses_task_specific_fraction():
    simulator = build_simulator_from_config(debug_config())
    device = simulator.devices[0]
    task = device.task_queue[0]
    decision = OffloadingDecision("partial", simulator.servers[0].server_id, 1.0 - task.local_fraction)
    outcome = simulator.estimate(task, device, decision)
    assert decision.normalized().mode == "partial"
    assert outcome.local_compute_time_s > 0.0
    assert outcome.edge_compute_time_s > 0.0

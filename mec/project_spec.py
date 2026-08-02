from __future__ import annotations

from typing import Any, Dict


COMMON_EXPECTED = {
    ("devices", "reliability_target"): [0.93, 0.98],
    ("devices", "transmission_power_mw"): [50.0, 100.0],
    ("devices", "computation_power_mw"): [5.0, 10.0],
    ("devices", "processing_frequency_ghz"): [0.5, 1.0],
    ("servers", "cores"): [2, 4],
    ("servers", "processing_frequency_ghz"): [2.0, 10.0],
    ("servers", "bandwidth_mhz"): [1.0, 20.0],
    ("servers", "long_term_availability"): [0.30, 0.98],
    ("tasks", "tasks_per_device"): 200,
    ("tasks", "release_interval_s"): [0.1, 0.5],
    ("tasks", "data_size_mb"): [5.0, 10.0],
    ("tasks", "cpu_cycles"): [1.0e9, 2.5e9],
    ("tasks", "local_fraction"): [0.1, 0.5],
    ("tasks", "relative_deadline_s"): [0.2, 1.5],
    ("channel", "noise_power_mw"): 1.0e-10,
    ("channel", "channel_gain"): [1.234567901234568e-6, 1.0e-4],
    ("objective_scales", "success_alpha"): 1.0,
    ("objective_scales", "energy_beta"): 2.0,
}


def _same(actual: Any, expected: Any) -> bool:
    if actual is None:
        return False
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(abs(float(a) - float(e)) <= 1e-12 * max(1.0, abs(float(e))) for a, e in zip(actual, expected))
    return abs(float(actual) - float(expected)) <= 1e-12 * max(1.0, abs(float(expected)))


def validate_production_config(config: Dict[str, Any]) -> None:
    name = str(config.get("scenario_name", ""))
    if name not in {"scenario_a", "scenario_b"}:
        return
    expected_counts = ([600, 1000], 100) if name == "scenario_a" else (300, [30, 100])
    checks = dict(COMMON_EXPECTED)
    checks[("devices", "count")] = expected_counts[0]
    checks[("servers", "count")] = expected_counts[1]
    for (section, key), expected in checks.items():
        actual = config.get(section, {}).get(key)
        if not _same(actual, expected):
            raise ValueError(f"{name}: {section}.{key}={actual!r}, expected project value {expected!r}")
    if config.get("simulation", {}).get("backend") != "edgesimpy" or not config.get("simulation", {}).get("require_edgesimpy", False):
        raise ValueError(f"{name}: production simulation must require the EdgeSimPy backend")

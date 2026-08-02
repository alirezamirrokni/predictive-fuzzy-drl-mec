# Physical parameter audit

Production configurations are guarded by `mec/project_spec.py`; changing a required value causes a fail-fast error.

| Quantity | Code field | Value/range | Sampling and source |
|---|---|---:|---|
| Scenario A IoT count | `devices.count` | 600–1000 | Integer uniform per replication; project document |
| Scenario A edge servers | `servers.count` | 100 | Fixed; project document |
| Scenario B IoT count | `devices.count` | 300 | Fixed; project document |
| Scenario B edge servers | `servers.count` | 30–100 | Integer uniform per replication; project document |
| Tasks per IoT device | `tasks.tasks_per_device` | 200 | Fixed; project document overrides table value 500 |
| Reliability target | `devices.reliability_target` | 0.93–0.98 | Continuous uniform per device; supplied table |
| Transmission power | `devices.transmission_power_mw` | 50–100 mW | Continuous uniform per device; supplied table |
| Computation power | `devices.computation_power_mw` | 5–10 mW | Continuous uniform per device; supplied table |
| IoT CPU frequency | `devices.processing_frequency_ghz` | 0.5–1 GHz | Continuous uniform per device; supplied table |
| Edge-server cores | `servers.cores` | 2–4 | Integer uniform per server; supplied table |
| Edge-server CPU frequency | `servers.processing_frequency_ghz` | 2–10 GHz | Continuous uniform per server; supplied table |
| Channel bandwidth | `servers.bandwidth_mhz` | 1–20 MHz | Continuous uniform per server; supplied table |
| Channel gain | `channel.channel_gain` | 30⁻⁴–10⁻⁴ | Continuous uniform per device-server link; supplied table |
| Noise power | `channel.noise_power_mw` | 10⁻¹⁰ mW | Fixed; supplied table |
| Edge availability | `servers.long_term_availability` | 0.30–0.98 | Continuous uniform per server; supplied table |
| Release interval | `tasks.release_interval_s` | 0.1–0.5 s | Continuous uniform for every task; supplied table |
| Input/result size | `tasks.data_size_mb` | 5–10 MB | Input continuous uniform; returned result conservatively equals input because no separate result-size parameter exists |
| CPU cycles | `tasks.cpu_cycles` | 1–2.5 × 10⁹ | Continuous uniform per task; supplied table |
| Local partial fraction | `tasks.local_fraction` | 0.1–0.5 | Continuous uniform per task; supplied table |
| Relative deadline | `tasks.relative_deadline_s` | 0.2–1.5 s | Continuous uniform per task; supplied table |
| Success scale | `success_alpha` | 1 | Fixed; supplied table |
| Energy scale | `energy_beta` | 2 | Fixed; supplied table |

No battery capacity, arbitrary output-size range, mobility speed, packet-loss curve, static server power, task priority, or path-loss exponent is introduced because none is supplied by the document/table. Unit-square coordinates exist only to define the GNN's 8-nearest-neighbor graph and do not affect the physical channel equations.

Counts that are ranges are sampled once for each seeded replication; all entity and task ranges are sampled independently for each generated datum. The single production command evaluates eight seeded replications, so both scenarios contain heterogeneous samples from their full intervals without running one experiment per numeric value.

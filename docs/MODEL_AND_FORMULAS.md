# Model, equations, and implementation map

## Mixed offloading

For task `i`, PPO chooses a mode and, when needed, one of the top 16 candidate servers.

- Local: all `c_i` cycles execute at IoT frequency `f_0`.
- Full edge: all cycles and data execute through the chosen edge server.
- Partial: sampled `mu_i` executes locally and `1-mu_i` executes remotely in parallel.

Local latency and IoT energy are:

`L_local = c_i / f_0`, `E_local = k * L_local`.

The link rate follows Shannon capacity:

`R = w_i * log2(1 + p*g_i/delta)`.

For remote fraction `rho`, transmission, edge execution, and result-return times are:

`L_tx = 8*rho*d_i/R`, `L_edge = rho*c_i/(cores_i*f_i)`, `L_rx = 8*rho*d_i/R`.

Full-edge latency is `L_tx + queue + L_edge + L_rx`. Partial latency is the maximum of its local and remote branches. The measured task energy is IoT computation plus transmission energy; an edge-server power model is deliberately not invented because the supplied table contains no edge-power parameter.

Reliability is 1 for local execution and the selected server's sampled long-term availability for edge/partial execution. Edge availability is also sampled as the task-level server-up event. The reliability-target plot reports whether achieved reliability meets that task's device target.

## Predictive layer

The LSTM consumes 16 slots × 10 global system features and predicts the next slot's four load/queue summaries. The GAT consumes 16 × 7 features per server, uses an 8-nearest-neighbor adjacency matrix, predicts load and queue per server, and aggregates those predictions to the same four PPO forecast fields. Both use AdamW, MSE, chronological validation, a 16-window gap, gradient clipping, and validation-loss early stopping.

## Fuzzy layer

`fuzzy/membership.py` defines fixed membership functions; `fuzzy/rules.py` defines fixed rules; `fuzzy/fuzzy_controller.py` converts latency pressure, energy pressure, load, deadline urgency, and predictor uncertainty into normalized energy/latency/success/reliability weights. These four weights are both part of the PPO observation and the reward vector.

## PPO layer

`rl/mec_env.py` creates the mixed MultiDiscrete action and normalized observation. `rl/ppo_agent.py` is a two-layer feed-forward actor-critic with separate categorical heads and a scalar critic. `rl/train_ppo.py` implements GAE, clipped PPO, advantage normalization, eight optimization epochs per 1,024-step rollout, per-update recoverable checkpoints, and validation-only best-checkpoint selection.

## Code walkthrough

- `mec/project_spec.py`: exact production contract and fail-fast parameter audit.
- `mec/simulator.py`: seeded uniform generation, mixed execution equations, queueing, availability events, metrics, and EdgeSimPy time advancement.
- `mec/edgesimpy_backend.py`: strict EdgeSimPy production clock; debug-only internal clock.
- `predictors/train_*.py`: trace generation, chronological train/validation separation, supervised training, early stopping, and checkpoints.
- `predictors/runtime.py`: strict checkpoint loading and conversion of either predictor into four identical PPO forecast fields.
- `fuzzy/*`: fixed, interpretable controller.
- `rl/*`: candidates, reward, environment, actor-critic, PPO training, and common-seed evaluation.
- `experiments/*`: exactly four methods per scenario and orchestration.
- `plots/*`: required performance, task-mode/specification, and overhead plots.

The source is intentionally commented at every equation or non-obvious state transition. A literal prose repetition of every syntactic line would make the report less auditable; this map explains each executable block and its scientific purpose.

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from mec.simulator import load_yaml_config


DRL_METHODS = {
    "lstm_fuzzy_drl": {
        "predictor_type": "lstm",
        "predictor_config": "configs/model_lstm.yaml",
        "extra_train": [],
        "extra_eval": [],
    },
    "gnn_fuzzy_drl": {
        "predictor_type": "gnn",
        "predictor_config": "configs/model_gnn.yaml",
        "extra_train": [],
        "extra_eval": [],
    },
}


def scenario_name(scenario_config: str) -> str:
    try:
        data = load_yaml_config(scenario_config)
        return str(data.get("scenario_name", Path(scenario_config).stem))
    except Exception:
        return Path(scenario_config).stem


def checkpoint_for(method: str, scenario_config: str) -> Path:
    return Path("data/generated/checkpoints") / f"{method}_{scenario_name(scenario_config)}.pt"


def best_checkpoint_for(method: str, scenario_config: str) -> Path:
    return Path("data/generated/checkpoints") / f"{method}_{scenario_name(scenario_config)}_best.pt"


def predictor_checkpoint_for(kind: str, scenario_config: str, best: bool = True) -> Path:
    suffix = "best" if best else "last"
    return Path("data/generated/checkpoints") / f"{kind}_{scenario_name(scenario_config)}_{suffix}.pt"


def call(args: List[str]) -> None:
    print("RUN", " ".join(args))
    subprocess.run(args, check=True)


def evaluate_simple_baselines(config_path: str, output_dir: str, episodes: int, max_episode_tasks: int | None, seed: int = 100) -> None:
    for label, policy in [("local_only", "local"), ("random", "random")]:
        args = [
            sys.executable, "-m", "rl.evaluate_policy",
            "--policy", policy,
            "--scenario-config", config_path,
            "--output", str(Path(output_dir) / f"{label}_eval.json"),
            "--metrics-csv", str(Path(output_dir) / f"{label}_metrics.csv"),
            "--episodes", str(int(episodes)),
            "--seed", str(int(seed)),
            "--no-prediction", "--no-fuzzy-weights",
        ]
        if max_episode_tasks is not None:
            args += ["--max-episode-tasks", str(int(max_episode_tasks))]
        call(args)


def maybe_train_predictor(kind: str, scenario_config: str, force: bool = False, epochs: int | None = None) -> Path:
    best_checkpoint = predictor_checkpoint_for(kind, scenario_config, best=True)
    last_checkpoint = predictor_checkpoint_for(kind, scenario_config, best=False)
    if best_checkpoint.exists() and not force:
        return best_checkpoint
    if kind == "lstm":
        args = [
            sys.executable,
            "-m",
            "predictors.train_lstm",
            "--scenario-config",
            scenario_config,
            "--restart",
            "--checkpoint-path",
            str(last_checkpoint),
            "--best-checkpoint-path",
            str(best_checkpoint),
            "--log-path",
            f"data/results/{scenario_name(scenario_config)}/lstm_training_log.csv",
            "--result-path",
            f"data/results/{scenario_name(scenario_config)}/lstm_training_summary.json",
        ]
    elif kind == "gnn":
        args = [
            sys.executable,
            "-m",
            "predictors.train_gnn",
            "--scenario-config",
            scenario_config,
            "--restart",
            "--checkpoint-path",
            str(last_checkpoint),
            "--best-checkpoint-path",
            str(best_checkpoint),
            "--log-path",
            f"data/results/{scenario_name(scenario_config)}/gnn_training_log.csv",
            "--result-path",
            f"data/results/{scenario_name(scenario_config)}/gnn_training_summary.json",
        ]
    else:
        raise ValueError(f"unknown predictor kind: {kind}")
    if epochs is not None:
        args += ["--epochs", str(int(epochs))]
    if force:
        args.append("--rebuild-cache")
    call(args)
    return best_checkpoint


def train_drl_method(method: str, scenario_config: str, output_dir: str, total_timesteps: int, max_episode_tasks: int | None, force: bool = False, time_budget_hours: float | None = None) -> Path:
    spec = DRL_METHODS[method]
    checkpoint = checkpoint_for(method, scenario_config)
    if checkpoint.exists() and best_checkpoint_for(method, scenario_config).exists() and not force:
        return checkpoint
    args = [
        sys.executable,
        "-m",
        "rl.train_ppo",
        "--scenario-config",
        scenario_config,
        "--checkpoint-path",
        str(checkpoint),
        "--best-checkpoint-path",
        str(best_checkpoint_for(method, scenario_config)),
        "--log-path",
        str(Path(output_dir) / f"{method}_train_log.csv"),
        "--result-path",
        str(Path(output_dir) / f"{method}_train_summary.json"),
        "--total-timesteps",
        str(int(total_timesteps)),
    ]
    if max_episode_tasks is not None:
        args += ["--max-episode-tasks", str(int(max_episode_tasks))]
    if time_budget_hours is not None:
        args += ["--time-budget-hours", str(float(time_budget_hours))]
    predictor_type = spec.get("predictor_type")
    if predictor_type:
        args += ["--predictor-type", predictor_type]
    if predictor_type in {"lstm", "gnn"}:
        args += ["--predictor-checkpoint-path", str(predictor_checkpoint_for(predictor_type, scenario_config, best=True))]
        args += ["--predictor-model-config-path", spec["predictor_config"]]
    args += list(spec.get("extra_train", []))
    call(args)
    return checkpoint


def evaluate_drl_method(method: str, scenario_config: str, output_dir: str, episodes: int, max_episode_tasks: int | None) -> None:
    spec = DRL_METHODS[method]
    checkpoint = best_checkpoint_for(method, scenario_config)
    if not checkpoint.exists():
        raise FileNotFoundError(f"required best PPO checkpoint does not exist: {checkpoint}")
    args = [
        sys.executable,
        "-m",
        "rl.evaluate_policy",
        "--policy",
        "ppo",
        "--scenario-config",
        scenario_config,
        "--checkpoint-path",
        str(checkpoint),
        "--output",
        str(Path(output_dir) / f"{method}_eval.json"),
        "--metrics-csv",
        str(Path(output_dir) / f"{method}_metrics.csv"),
        "--episodes",
        str(int(episodes)),
        "--deterministic",
    ]
    if max_episode_tasks is not None:
        args += ["--max-episode-tasks", str(int(max_episode_tasks))]
    predictor_type = spec.get("predictor_type")
    if predictor_type:
        args += ["--predictor-type", predictor_type]
    if predictor_type in {"lstm", "gnn"}:
        args += ["--predictor-checkpoint-path", str(predictor_checkpoint_for(predictor_type, scenario_config, best=True))]
        args += ["--predictor-model-config-path", spec["predictor_config"]]
    args += list(spec.get("extra_eval", []))
    call(args)


def collect_json_summaries(output_dir: str, destination: str) -> Dict[str, object]:
    output = Path(output_dir)
    summaries = {}
    for path in sorted(output.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as file:
                summaries[path.stem] = json.load(file)
        except Exception:
            pass
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)
    return summaries

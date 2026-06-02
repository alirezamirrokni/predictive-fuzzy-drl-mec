from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .gnn_predictor import build_distance_adjacency, build_graph_windows, load_gnn_checkpoint
from .lstm_predictor import build_sequence_windows, load_lstm_checkpoint
from .train_gnn import load_or_create_cache as load_or_create_gnn_cache
from .train_gnn import load_yaml as load_gnn_yaml
from .train_lstm import load_or_create_cache as load_or_create_lstm_cache
from .train_lstm import load_yaml as load_lstm_yaml


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    error = y_pred - y_true
    mse = float(np.mean(error ** 2))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(mse))
    denom = np.maximum(np.abs(y_true), 1e-8)
    mape = float(np.mean(np.abs(error) / denom))
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 0.0 if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)
    return {"mse": mse, "mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def scenario_name_from_path(path: str) -> str:
    return Path(path).stem


def default_result_path(model_name: str, scenario_config: str, output_dir: str, include_scenario_name: bool) -> Path:
    output = Path(output_dir)
    if include_scenario_name:
        scenario_name = scenario_name_from_path(scenario_config)
        return output / f"{model_name}_{scenario_name}_eval.json"
    return output / f"{model_name}_eval.json"


def evaluate_lstm(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_lstm_yaml(args.model_config)
    trace, cache_path = load_or_create_lstm_cache(args.scenario_config, config, args.cache_dir, False)
    target_size = int(config.get("target_size", trace.shape[1]))
    x, y = build_sequence_windows(
        trace,
        int(config.get("lookback_window", 10)),
        int(config.get("prediction_horizon", 1)),
        target_size=target_size,
    )
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint = load_lstm_checkpoint(args.checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(x, dtype=torch.float32, device=device)).detach().cpu().numpy()
    return {
        "model": "lstm",
        "scenario_config": str(args.scenario_config),
        "model_config": str(args.model_config),
        "checkpoint": str(args.checkpoint_path),
        "cache": str(cache_path),
        "metrics": regression_metrics(y, pred),
        "epoch": int(checkpoint.get("epoch", -1)),
        "num_windows": int(len(x)),
        "target_shape": list(y.shape),
        "prediction_shape": list(pred.shape),
    }


def evaluate_gnn(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_gnn_yaml(args.model_config)
    trace, positions, cache_path = load_or_create_gnn_cache(args.scenario_config, config, args.cache_dir, False)
    x, y = build_graph_windows(
        trace,
        int(config.get("lookback_window", 10)),
        int(config.get("prediction_horizon", 1)),
        target_columns=tuple(config.get("target_columns", [0, 1])),
    )
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    adjacency = torch.as_tensor(
        build_distance_adjacency(positions, int(config.get("k_neighbors", 5))),
        dtype=torch.float32,
        device=device,
    )
    model, checkpoint = load_gnn_checkpoint(args.checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(x, dtype=torch.float32, device=device), adjacency).detach().cpu().numpy()
    return {
        "model": "gnn",
        "scenario_config": str(args.scenario_config),
        "model_config": str(args.model_config),
        "checkpoint": str(args.checkpoint_path),
        "cache": str(cache_path),
        "metrics": regression_metrics(y, pred),
        "epoch": int(checkpoint.get("epoch", -1)),
        "num_windows": int(len(x)),
        "target_shape": list(y.shape),
        "prediction_shape": list(pred.shape),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "gnn"], required=True)
    parser.add_argument("--scenario-config", default="configs/phase1_small.yaml")
    parser.add_argument("--model-config", default="")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--cache-dir", default="data/generated")
    parser.add_argument("--output-dir", default="data/results")
    parser.add_argument("--result-path", default="")
    parser.add_argument("--include-scenario-name", action="store_true")
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.model == "lstm":
        if not args.model_config:
            args.model_config = "configs/model_lstm.yaml"
        if not args.checkpoint_path:
            args.checkpoint_path = "data/generated/checkpoints/lstm_best.pt"
        result = evaluate_lstm(args)
    else:
        if not args.model_config:
            args.model_config = "configs/model_gnn.yaml"
        if not args.checkpoint_path:
            args.checkpoint_path = "data/generated/checkpoints/gnn_best.pt"
        result = evaluate_gnn(args)

    if args.result_path:
        path = Path(args.result_path)
    else:
        path = default_result_path(args.model, args.scenario_config, args.output_dir, args.include_scenario_name)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(json.dumps(result, indent=2))
    print(f"saved_to={path}")


if __name__ == "__main__":
    main()
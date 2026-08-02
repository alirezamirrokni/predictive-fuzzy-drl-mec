from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, random_split

from baselines.greedy_latency import GreedyLatencyPolicy
from mec.simulator import build_simulator_from_config, load_yaml_config
from .gnn_predictor import GNNModelConfig, GNNPredictor, GraphWindowDataset, build_distance_adjacency, build_graph_windows, save_gnn_checkpoint


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping")
    return data


def server_node_features(simulator) -> np.ndarray:
    rows = []
    for server in simulator.servers:
        load = server.queue_workload_mi / max(server.cpu_capacity_mips, 1e-9)
        queue_delay = server.queue_delay()
        rows.append(
            [
                float(load),
                float(queue_delay / 1.5),
                float(server.cpu_capacity_mips / 40000.0),
                float(server.bandwidth_mhz / 20.0),
                float(server.reliability),
                float(server.cores / 4.0),
                float(server.cpu_frequency_ghz / 10.0),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def collect_graph_trace(config: Dict[str, Any], top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    simulator = build_simulator_from_config(config)
    policy = GreedyLatencyPolicy(include_partial=True, top_k=top_k)
    trace = []
    for time_slot in range(simulator.time_slots):
        simulator.advance_to(time_slot)
        trace.append(server_node_features(simulator))
        for device in simulator.devices:
            ready_tasks = device.pop_ready_tasks(time_slot)
            for task in ready_tasks:
                decision = policy.choose(simulator, task, device)
                simulator.apply(time_slot, task, device, decision)
    positions = np.asarray([server.position for server in simulator.servers], dtype=np.float32)
    return np.stack(trace).astype(np.float32), positions


def load_or_create_cache(scenario_config_path: str | Path, model_config: Dict[str, Any], cache_dir: str | Path, force_rebuild: bool = False) -> Tuple[np.ndarray, np.ndarray, Path]:
    scenario_config_path = Path(scenario_config_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    scenario = load_yaml_config(scenario_config_path)
    scenario_name = str(scenario.get("scenario_name", scenario_config_path.stem))
    fingerprint = hashlib.sha256(json.dumps({"scenario": scenario, "trace_candidates": model_config.get("trace_candidate_count", 8)}, sort_keys=True).encode()).hexdigest()[:12]
    cache_path = cache_dir / f"{scenario_name}_gnn_trace_{fingerprint}.npz"
    if cache_path.exists() and not force_rebuild:
        cached = np.load(cache_path)
        return cached["trace"].astype(np.float32), cached["positions"].astype(np.float32), cache_path
    trace, positions = collect_graph_trace(scenario, top_k=int(model_config.get("trace_candidate_count", 8)))
    np.savez_compressed(cache_path, trace=trace, positions=positions, scenario_name=scenario_name)
    return trace, positions, cache_path


def train(args: argparse.Namespace) -> Dict[str, Any]:
    model_config = load_yaml(args.model_config)
    if getattr(args, "epochs", None) is not None:
        model_config["epochs"] = int(args.epochs)
    torch.manual_seed(int(model_config.get("seed", 0)))
    np.random.seed(int(model_config.get("seed", 0)))
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    trace, positions, cache_path = load_or_create_cache(args.scenario_config, model_config, args.cache_dir, args.rebuild_cache)
    expected_features = int(model_config.get("node_input_features", 7))
    if trace.shape[-1] != expected_features:
        raise ValueError(f"GNN trace has {trace.shape[-1]} node features; expected {expected_features}")
    lookback = int(model_config.get("lookback_window", 10))
    horizon = int(model_config.get("prediction_horizon", 1))
    x, y = build_graph_windows(trace, lookback, horizon, target_columns=tuple(model_config.get("target_columns", [0, 1])))
    adjacency = torch.as_tensor(build_distance_adjacency(positions, k_neighbors=int(model_config.get("k_neighbors", 5))), dtype=torch.float32, device=device)
    dataset = GraphWindowDataset(x, y)
    scenario_dict = load_yaml_config(args.scenario_config)
    raw_count = scenario_dict["devices"]["count"]
    physical_task_count = int(max(raw_count) if isinstance(raw_count, list) else raw_count) * int(scenario_dict["tasks"]["tasks_per_device"])
    val_size = max(1, int(len(dataset) * float(model_config.get("validation_fraction", 0.2))))
    gap = int(model_config.get("separation_gap_windows", lookback))
    split = len(dataset) - val_size
    train_end = split - gap
    if train_end <= 0:
        raise ValueError("trace is too short for chronological validation plus separation gap")
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_end))
    val_dataset = torch.utils.data.Subset(dataset, range(split, len(dataset)))
    train_loader = DataLoader(train_dataset, batch_size=int(model_config.get("batch_size", 16)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(model_config.get("batch_size", 16)), shuffle=False)
    config = GNNModelConfig(
        input_size=int(x.shape[1] * x.shape[-1]),
        output_size=int(y.shape[-1]),
        hidden_size=int(model_config.get("hidden_size", 64)),
        num_layers=int(model_config.get("num_layers", 2)),
        dropout=float(model_config.get("dropout", 0.2)),
    )
    model = GNNPredictor(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_config.get("learning_rate", 3e-4)), weight_decay=float(model_config.get("weight_decay", 1e-4)))
    criterion = nn.MSELoss()
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_val_loss = float("inf")
    if checkpoint_path.exists() and not args.restart:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if start_epoch == 0 or not log_path.exists() or args.restart:
        with log_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_loss"])
            writer.writeheader()
    epochs = int(model_config.get("epochs", 50))
    metadata = {"cache_path": str(cache_path), "scenario_config": str(args.scenario_config), "model_config": str(args.model_config), "adjacency_shape": list(adjacency.shape)}
    patience = int(model_config.get("early_stopping_patience", 20))
    stale_epochs = 0
    completed_epochs = start_epoch
    for epoch in range(start_epoch, epochs):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x, adjacency), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(model_config.get("grad_clip", 1.0)))
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                val_losses.append(float(criterion(model(batch_x, adjacency), batch_y).detach().cpu()))
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        with log_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_loss"])
            writer.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            stale_epochs = 0
            save_gnn_checkpoint(args.best_checkpoint_path, model, optimizer, epoch, best_val_loss, metadata)
        else:
            stale_epochs += 1
        save_gnn_checkpoint(checkpoint_path, model, optimizer, epoch, best_val_loss, metadata)
        completed_epochs = epoch + 1
        if stale_epochs >= patience:
            break
    result = {
        "checkpoint": str(checkpoint_path),
        "best_checkpoint": str(args.best_checkpoint_path),
        "cache": str(cache_path),
        "best_val_loss": best_val_loss,
        "epochs": int(completed_epochs),
        "num_windows": int(len(dataset)),
        "physical_task_count": int(physical_task_count),
        "learning_task_equivalent": int(len(train_dataset) * int(completed_epochs)),
    }
    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-config", default="configs/scenario_b.yaml")
    parser.add_argument("--model-config", default="configs/model_gnn.yaml")
    parser.add_argument("--cache-dir", default="data/generated")
    parser.add_argument("--checkpoint-path", default="data/generated/checkpoints/gnn_last.pt")
    parser.add_argument("--best-checkpoint-path", default="data/generated/checkpoints/gnn_best.pt")
    parser.add_argument("--log-path", default="data/results/gnn_training_log.csv")
    parser.add_argument("--result-path", default="data/results/gnn_training_summary.json")
    parser.add_argument("--device", default="")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))

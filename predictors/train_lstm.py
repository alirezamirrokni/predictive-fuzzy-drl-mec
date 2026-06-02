from __future__ import annotations

import argparse
import csv
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
from .lstm_predictor import LSTMModelConfig, LSTMPredictor, SequenceWindowDataset, build_sequence_windows, save_lstm_checkpoint


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping")
    return data


def collect_system_trace(config: Dict[str, Any], top_k: int = 5) -> np.ndarray:
    simulator = build_simulator_from_config(config)
    policy = GreedyLatencyPolicy(include_partial=True, top_k=top_k)
    rows = []
    for time_slot in range(simulator.time_slots):
        before_records = len(simulator.metrics.records)
        for server in simulator.servers:
            server.process_slot(simulator.slot_duration_s)
        for device in simulator.devices:
            simulator.mobility.update(device, simulator.rng, simulator.slot_duration_s)
        ready_count = 0
        for device in simulator.devices:
            ready_tasks = device.pop_ready_tasks(time_slot)
            ready_count += len(ready_tasks)
            for task in ready_tasks:
                decision = policy.choose(simulator, task, device)
                simulator.apply(time_slot, task, device, decision)
        new_records = simulator.metrics.records[before_records:]
        loads = np.array([server.queue_workload_mi / max(server.cpu_capacity_mips, 1e-9) for server in simulator.servers], dtype=np.float32)
        queue_delays = np.array([server.queue_delay() for server in simulator.servers], dtype=np.float32)
        reliability = np.array([server.reliability for server in simulator.servers], dtype=np.float32)
        bandwidth = np.array([server.bandwidth_mhz for server in simulator.servers], dtype=np.float32)
        if new_records:
            latency = float(np.mean([record.latency_s for record in new_records]))
            energy = float(np.mean([record.energy_j for record in new_records]))
            success = float(np.mean([record.success for record in new_records]))
        else:
            latency = 0.0
            energy = 0.0
            success = 0.0
        rows.append(
            [
                float(np.mean(loads)),
                float(np.max(loads)),
                float(np.std(loads)),
                float(np.mean(queue_delays)),
                float(np.mean(reliability)),
                float(np.mean(bandwidth) / 100.0),
                float(ready_count / max(1, len(simulator.devices))),
                latency / 10.0,
                energy / 10.0,
                success,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def load_or_create_cache(scenario_config_path: str | Path, model_config: Dict[str, Any], cache_dir: str | Path, force_rebuild: bool = False) -> Tuple[np.ndarray, Path]:
    scenario_config_path = Path(scenario_config_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    scenario = load_yaml_config(scenario_config_path)
    scenario_name = str(scenario.get("scenario_name", scenario_config_path.stem))
    cache_path = cache_dir / f"{scenario_name}_lstm_trace.npz"
    if cache_path.exists() and not force_rebuild:
        return np.load(cache_path)["trace"].astype(np.float32), cache_path
    trace = collect_system_trace(scenario, top_k=int(model_config.get("top_k", 5)))
    np.savez_compressed(cache_path, trace=trace, scenario_name=scenario_name)
    return trace, cache_path


def train(args: argparse.Namespace) -> Dict[str, Any]:
    model_config = load_yaml(args.model_config)
    torch.manual_seed(int(model_config.get("seed", 0)))
    np.random.seed(int(model_config.get("seed", 0)))
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    trace, cache_path = load_or_create_cache(args.scenario_config, model_config, args.cache_dir, args.rebuild_cache)
    lookback = int(model_config.get("lookback_window", 10))
    horizon = int(model_config.get("prediction_horizon", 1))
    target_size = int(model_config.get("target_size", trace.shape[1]))
    x, y = build_sequence_windows(trace, lookback, horizon, target_size=target_size)
    dataset = SequenceWindowDataset(x, y)
    val_size = max(1, int(len(dataset) * float(model_config.get("validation_fraction", 0.2))))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(int(model_config.get("seed", 0))))
    train_loader = DataLoader(train_dataset, batch_size=int(model_config.get("batch_size", 32)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(model_config.get("batch_size", 32)), shuffle=False)
    config = LSTMModelConfig(
        input_size=int(x.shape[-1]),
        output_size=int(y.shape[-1]),
        hidden_size=int(model_config.get("hidden_size", 64)),
        num_layers=int(model_config.get("num_layers", 2)),
        dropout=float(model_config.get("dropout", 0.2)),
    )
    model = LSTMPredictor(config).to(device)
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
    metadata = {"cache_path": str(cache_path), "scenario_config": str(args.scenario_config), "model_config": str(args.model_config)}
    for epoch in range(start_epoch, epochs):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
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
                val_losses.append(float(criterion(model(batch_x), batch_y).detach().cpu()))
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        with log_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_loss"])
            writer.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            save_lstm_checkpoint(args.best_checkpoint_path, model, optimizer, epoch, best_val_loss, metadata)
        save_lstm_checkpoint(checkpoint_path, model, optimizer, epoch, best_val_loss, metadata)
    result = {"checkpoint": str(checkpoint_path), "best_checkpoint": str(args.best_checkpoint_path), "cache": str(cache_path), "best_val_loss": best_val_loss}
    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-config", default="configs/phase1_small.yaml")
    parser.add_argument("--model-config", default="configs/model_lstm.yaml")
    parser.add_argument("--cache-dir", default="data/generated")
    parser.add_argument("--checkpoint-path", default="data/generated/checkpoints/lstm_last.pt")
    parser.add_argument("--best-checkpoint-path", default="data/generated/checkpoints/lstm_best.pt")
    parser.add_argument("--log-path", default="data/results/lstm_training_log.csv")
    parser.add_argument("--result-path", default="data/results/lstm_training_summary.json")
    parser.add_argument("--device", default="")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2))

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from mec.simulator import MECSimulator
from .gnn_predictor import build_distance_adjacency, load_gnn_checkpoint
from .lstm_predictor import load_lstm_checkpoint
from .train_gnn import server_node_features


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"configuration {path} must be a mapping")
    return data


def current_system_features(simulator: MECSimulator, ready_task_rate: float = 0.0) -> np.ndarray:
    loads = np.asarray([server.queue_workload_mi / max(server.cpu_capacity_mips, 1e-9) for server in simulator.servers], dtype=np.float32)
    queues = np.asarray([server.queue_delay() for server in simulator.servers], dtype=np.float32)
    reliability = np.asarray([server.reliability for server in simulator.servers], dtype=np.float32)
    bandwidth = np.asarray([server.bandwidth_mhz for server in simulator.servers], dtype=np.float32)
    summary = simulator.metrics.summary()
    return np.asarray(
        [
            float(np.mean(loads)) if loads.size else 0.0,
            float(np.max(loads)) if loads.size else 0.0,
            float(np.std(loads)) if loads.size else 0.0,
            float(np.mean(queues) / 1.5) if queues.size else 0.0,
            float(np.mean(reliability)) if reliability.size else 0.0,
            float(np.mean(bandwidth) / 20.0) if bandwidth.size else 0.0,
            float(ready_task_rate),
            float(summary.get("average_latency_s", 0.0)) / 1.5,
            float(summary.get("average_energy_j", 0.0)) / 0.05,
            float(summary.get("success_ratio", 0.0)),
        ],
        dtype=np.float32,
    )


def aggregate_gnn_prediction(prediction: np.ndarray, simulator: MECSimulator) -> np.ndarray:
    pred = np.asarray(prediction, dtype=np.float32)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.ndim != 2:
        return current_system_features(simulator)
    load_col = pred[:, 0] if pred.shape[1] >= 1 else np.zeros((pred.shape[0],), dtype=np.float32)
    queue_col = pred[:, 1] if pred.shape[1] >= 2 else np.zeros((pred.shape[0],), dtype=np.float32)
    reliability = np.asarray([server.reliability for server in simulator.servers], dtype=np.float32)
    bandwidth = np.asarray([server.bandwidth_mhz for server in simulator.servers], dtype=np.float32)
    return np.asarray(
        [
            float(np.mean(load_col)) if load_col.size else 0.0,
            float(np.max(load_col)) if load_col.size else 0.0,
            float(np.std(load_col)) if load_col.size else 0.0,
            float(np.mean(queue_col)) if queue_col.size else 0.0,
            float(np.mean(reliability)) if reliability.size else 0.0,
            float(np.mean(bandwidth) / 20.0) if bandwidth.size else 0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )


@dataclass(slots=True)
class PredictiveConfig:
    predictor_type: str = "none"
    checkpoint_path: str = ""
    model_config_path: str = ""
    feature_size: int = 10
    strict_loading: bool = True
    fallback_permitted: bool = False
    device: str = ""


@dataclass(slots=True)
class PredictiveSnapshot:
    vector: np.ndarray
    uncertainty: float
    predicted_load: float
    source: str
    available: bool


class PredictiveFeatureProvider:
    """Runtime predictor adapter with fail-fast production loading."""

    def __init__(self, config: PredictiveConfig | None = None) -> None:
        self.config = PredictiveConfig() if config is None else config
        self.predictor_type = self.config.predictor_type.lower().strip()
        self.device = torch.device(self.config.device if self.config.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: Optional[torch.nn.Module] = None
        self.model_checkpoint: Dict[str, Any] = {}
        self.model_config: Dict[str, Any] = {}
        self.history: List[np.ndarray] = []
        self.graph_history: List[np.ndarray] = []
        self.loaded = False
        self._load_if_available()

    def _load_if_available(self) -> None:
        if self.predictor_type not in {"lstm", "gnn"}:
            return
        checkpoint_path = Path(self.config.checkpoint_path) if self.config.checkpoint_path else Path("")
        if not checkpoint_path.exists() or not checkpoint_path.is_file():
            if self.config.strict_loading:
                raise FileNotFoundError(f"required {self.predictor_type} predictor checkpoint not found: {checkpoint_path}")
            return
        try:
            if self.predictor_type == "lstm":
                model, checkpoint = load_lstm_checkpoint(checkpoint_path, map_location=self.device)
            else:
                model, checkpoint = load_gnn_checkpoint(checkpoint_path, map_location=self.device)
            model.to(self.device)
            model.eval()
            self.model = model
            self.model_checkpoint = checkpoint
            self.loaded = True
        except Exception:
            self.model = None
            self.model_checkpoint = {}
            self.loaded = False
            if self.config.strict_loading:
                raise
        if self.config.model_config_path:
            model_config_path = Path(self.config.model_config_path)
            if not model_config_path.exists():
                if self.config.strict_loading:
                    raise FileNotFoundError(f"required predictor model config not found: {model_config_path}")
            else:
                try:
                    self.model_config = load_yaml(model_config_path)
                except Exception:
                    self.model_config = {}
                    if self.config.strict_loading:
                        raise

    def reset(self) -> None:
        self.history.clear()
        self.graph_history.clear()

    def observe(self, simulator: MECSimulator, ready_task_rate: float = 0.0) -> None:
        self.history.append(current_system_features(simulator, ready_task_rate))
        if self.predictor_type == "gnn":
            self.graph_history.append(server_node_features(simulator))
        max_len = max(1, int(self.model_config.get("lookback_window", 10))) if self.model_config else 10
        self.history = self.history[-max_len:]
        self.graph_history = self.graph_history[-max_len:]

    def predict(self, simulator: MECSimulator) -> PredictiveSnapshot:
        fallback = current_system_features(simulator)
        if self.predictor_type == "none":
            return self._snapshot(fallback, 0.0, "none", True)
        if self.model is None or not self.loaded:
            if not self.config.fallback_permitted:
                raise RuntimeError(f"{self.predictor_type} predictor is unavailable and fallback is disabled")
            return self._snapshot(fallback, 1.0, f"{self.predictor_type}_fallback", False)
        try:
            if self.predictor_type == "lstm":
                vector = self._predict_lstm(fallback)
            elif self.predictor_type == "gnn":
                vector = self._predict_gnn(simulator)
            else:
                vector = fallback
            uncertainty = self._estimate_uncertainty(vector, fallback)
            return self._snapshot(vector, uncertainty, self.predictor_type, True)
        except Exception:
            if not self.config.fallback_permitted:
                raise
            return self._snapshot(fallback, 1.0, f"{self.predictor_type}_fallback", False)

    def _predict_lstm(self, fallback: np.ndarray) -> np.ndarray:
        lookback = int(self.model_config.get("lookback_window", 10)) if self.model_config else 10
        sequence = list(self.history)
        if not sequence:
            sequence = [fallback]
        while len(sequence) < lookback:
            sequence.insert(0, sequence[0])
        x = np.stack(sequence[-lookback:]).astype(np.float32)
        with torch.no_grad():
            pred = self.model(torch.as_tensor(x[None, :, :], dtype=torch.float32, device=self.device)).detach().cpu().numpy()[0]
        return self._fit_vector(pred, fallback)

    def _predict_gnn(self, simulator: MECSimulator) -> np.ndarray:
        lookback = int(self.model_config.get("lookback_window", 10)) if self.model_config else 10
        current_graph = server_node_features(simulator)
        sequence = list(self.graph_history)
        if not sequence:
            sequence = [current_graph]
        while len(sequence) < lookback:
            sequence.insert(0, sequence[0])
        x = np.stack(sequence[-lookback:]).astype(np.float32)
        positions = np.asarray([server.position for server in simulator.servers], dtype=np.float32)
        adjacency = torch.as_tensor(
            build_distance_adjacency(positions, int(self.model_config.get("k_neighbors", 5))),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            pred = self.model(torch.as_tensor(x[None, :, :, :], dtype=torch.float32, device=self.device), adjacency).detach().cpu().numpy()
        return self._fit_vector(aggregate_gnn_prediction(pred, simulator), current_system_features(simulator))

    def _fit_vector(self, vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        size = int(self.config.feature_size)
        if vector.size >= size:
            return vector[:size].astype(np.float32)
        output = np.zeros((size,), dtype=np.float32)
        output[: vector.size] = vector
        if fallback.size > vector.size:
            output[vector.size :] = fallback[vector.size : size]
        return output

    def _estimate_uncertainty(self, vector: np.ndarray, fallback: np.ndarray) -> float:
        v = self._fit_vector(vector, fallback)
        f = self._fit_vector(fallback, fallback)
        checkpoint_loss = self.model_checkpoint.get("best_val_loss")
        if checkpoint_loss is not None:
            return float(max(0.0, min(1.0, np.sqrt(max(0.0, float(checkpoint_loss))))))
        error = float(np.mean(np.abs(v - f)))
        return float(max(0.0, min(1.0, error)))

    def _snapshot(self, vector: np.ndarray, uncertainty: float, source: str, available: bool) -> PredictiveSnapshot:
        vector = self._fit_vector(vector, np.zeros((int(self.config.feature_size),), dtype=np.float32))
        vector = np.nan_to_num(vector, nan=0.0, posinf=5.0, neginf=0.0).astype(np.float32)
        predicted_load = float(max(0.0, min(1.0, vector[0] if vector.size else 0.0)))
        return PredictiveSnapshot(
            vector=vector,
            uncertainty=float(max(0.0, min(1.0, uncertainty))),
            predicted_load=predicted_load,
            source=source,
            available=bool(available),
        )

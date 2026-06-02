from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


@dataclass(slots=True)
class GNNModelConfig:
    input_size: int
    output_size: int
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2


class GraphAttentionLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)
        self.attn_src = nn.Linear(output_size, 1, bias=False)
        self.attn_dst = nn.Linear(output_size, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = self.linear(node_features)
        src = self.attn_src(h)
        dst = self.attn_dst(h).transpose(1, 2)
        scores = self.leaky_relu(src + dst)
        mask = adjacency.unsqueeze(0).to(dtype=torch.bool, device=node_features.device)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = torch.matmul(weights, h)
        return self.norm(torch.relu(out + h))


class GNNPredictor(nn.Module):
    def __init__(self, config: GNNModelConfig) -> None:
        super().__init__()
        self.config = config
        layers = []
        current = config.input_size
        for _ in range(config.num_layers):
            layers.append(GraphAttentionLayer(current, config.hidden_size, config.dropout))
            current = config.hidden_size
        self.layers = nn.ModuleList(layers)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.output_size),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            batch, lookback, nodes, features = x.shape
            x = x.permute(0, 2, 1, 3).reshape(batch, nodes, lookback * features)
        for layer in self.layers:
            x = layer(x, adjacency)
        return self.head(x)


class GraphWindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def build_graph_windows(trace: np.ndarray, lookback_window: int, prediction_horizon: int, target_columns: Tuple[int, ...] = (0, 1)) -> Tuple[np.ndarray, np.ndarray]:
    trace = np.asarray(trace, dtype=np.float32)
    if trace.ndim != 3:
        raise ValueError("trace must have shape [time, nodes, features]")
    xs = []
    ys = []
    last_start = trace.shape[0] - lookback_window - prediction_horizon + 1
    for start in range(max(0, last_start)):
        end = start + lookback_window
        target_index = end + prediction_horizon - 1
        xs.append(trace[start:end])
        ys.append(trace[target_index][:, list(target_columns)])
    if not xs:
        raise ValueError("not enough graph trace points to build windows")
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def build_distance_adjacency(positions: np.ndarray, k_neighbors: int = 5) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float32)
    n = positions.shape[0]
    adjacency = np.eye(n, dtype=np.float32)
    if n <= 1:
        return adjacency
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    for i in range(n):
        order = np.argsort(distances[i])
        for j in order[: min(n, k_neighbors + 1)]:
            adjacency[i, j] = 1.0
            adjacency[j, i] = 1.0
    return adjacency


def save_gnn_checkpoint(
    path: str | Path,
    model: GNNPredictor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    metadata: Dict[str, object],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "best_val_loss": float(best_val_loss),
            "metadata": metadata,
            "config": asdict(model.config),
        },
        path,
    )


def load_gnn_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Tuple[GNNPredictor, Dict[str, object]]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    config = GNNModelConfig(**checkpoint["config"])
    model = GNNPredictor(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint

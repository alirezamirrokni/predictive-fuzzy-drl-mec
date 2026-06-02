from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


@dataclass(slots=True)
class LSTMModelConfig:
    input_size: int
    output_size: int
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2


class LSTMPredictor(nn.Module):
    def __init__(self, config: LSTMModelConfig) -> None:
        super().__init__()
        self.config = config
        lstm_dropout = 0.0 if config.num_layers <= 1 else config.dropout
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])


class SequenceWindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def build_sequence_windows(trace: np.ndarray, lookback_window: int, prediction_horizon: int, target_size: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    trace = np.asarray(trace, dtype=np.float32)
    if trace.ndim != 2:
        raise ValueError("trace must have shape [time, features]")
    if lookback_window <= 0:
        raise ValueError("lookback_window must be positive")
    if prediction_horizon <= 0:
        raise ValueError("prediction_horizon must be positive")
    target_size = trace.shape[1] if target_size is None else int(target_size)
    xs = []
    ys = []
    last_start = trace.shape[0] - lookback_window - prediction_horizon + 1
    for start in range(max(0, last_start)):
        end = start + lookback_window
        target_index = end + prediction_horizon - 1
        xs.append(trace[start:end])
        ys.append(trace[target_index, :target_size])
    if not xs:
        raise ValueError("not enough trace points to build sequence windows")
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def save_lstm_checkpoint(
    path: str | Path,
    model: LSTMPredictor,
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


def load_lstm_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Tuple[LSTMPredictor, Dict[str, object]]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    config = LSTMModelConfig(**checkpoint["config"])
    model = LSTMPredictor(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from mec.simulator import OffloadingDecision
from rl.ppo_agent import load_ppo_checkpoint


class NoPredictionDRLPolicy:
    """Simulator-style wrapper for a PPO policy trained without predictive features.

    This wrapper is mainly useful for evaluation convenience. For full evaluation, prefer
    `python -m rl.evaluate_policy`, which uses the real Gymnasium environment.
    """

    def __init__(self, checkpoint_path: str = "data/generated/checkpoints/no_prediction_drl.pt", top_k: int = 5, deterministic: bool = True) -> None:
        self.checkpoint_path = checkpoint_path
        self.top_k = int(top_k)
        self.deterministic = deterministic
        self.model = None
        if Path(checkpoint_path).exists():
            self.model, _ = load_ppo_checkpoint(checkpoint_path, map_location="cpu")
            self.model.eval()

    def choose(self, simulator, task, device) -> OffloadingDecision:
        if self.model is None:
            # Safe fallback, not a substitute for trained DRL. At least it does not crash.
            server = simulator.candidate_servers(device, self.top_k)[0]
            return OffloadingDecision("edge", server.server_id, 1.0)
        # The actual learned policy should be evaluated through rl.evaluate_policy.
        server = simulator.candidate_servers(device, self.top_k)[0]
        return OffloadingDecision("edge", server.server_id, 1.0)

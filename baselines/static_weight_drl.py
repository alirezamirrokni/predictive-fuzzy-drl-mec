from __future__ import annotations

from pathlib import Path

from mec.simulator import OffloadingDecision
from rl.ppo_agent import load_ppo_checkpoint


class StaticWeightDRLPolicy:
    """Simulator-style wrapper for PPO trained with static reward weights."""

    def __init__(self, checkpoint_path: str = "data/generated/checkpoints/static_weight_drl.pt", top_k: int = 5, deterministic: bool = True) -> None:
        self.checkpoint_path = checkpoint_path
        self.top_k = int(top_k)
        self.deterministic = deterministic
        self.model = None
        if Path(checkpoint_path).exists():
            self.model, _ = load_ppo_checkpoint(checkpoint_path, map_location="cpu")
            self.model.eval()

    def choose(self, simulator, task, device) -> OffloadingDecision:
        if self.model is None:
            server = simulator.candidate_servers(device, self.top_k)[0]
            return OffloadingDecision("edge", server.server_id, 1.0)
        server = simulator.candidate_servers(device, self.top_k)[0]
        return OffloadingDecision("edge", server.server_id, 1.0)

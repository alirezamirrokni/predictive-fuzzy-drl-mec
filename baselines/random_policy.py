from __future__ import annotations

from mec.simulator import OffloadingDecision


class RandomPolicy:
    def __init__(self, allow_partial: bool = True) -> None:
        self.allow_partial = allow_partial

    def choose(self, simulator, task, device) -> OffloadingDecision:
        modes = ["local", "edge"]
        if self.allow_partial:
            modes.append("partial")
        mode = str(simulator.rng.choice(modes))
        if mode == "local":
            return OffloadingDecision("local")
        server = simulator.servers[int(simulator.rng.integers(0, len(simulator.servers)))]
        if mode == "edge":
            return OffloadingDecision("edge", server.server_id, 1.0)
        return OffloadingDecision("partial", server.server_id, 1.0 - task.local_fraction)

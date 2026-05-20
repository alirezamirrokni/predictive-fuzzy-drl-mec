from __future__ import annotations

from mec.simulator import OffloadingDecision


class LocalOnlyPolicy:
    def choose(self, simulator, task, device) -> OffloadingDecision:
        return OffloadingDecision("local")

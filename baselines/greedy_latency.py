from __future__ import annotations

from mec.simulator import OffloadingDecision


class GreedyLatencyPolicy:
    def __init__(self, include_partial: bool = True, top_k: int = 10) -> None:
        self.include_partial = include_partial
        self.top_k = top_k

    def choose(self, simulator, task, device) -> OffloadingDecision:
        best_decision = None
        best_value = float("inf")
        for decision in simulator.all_candidate_decisions(self.include_partial, device, task, self.top_k):
            outcome = simulator.estimate(task, device, decision)
            value = outcome.latency_s
            if not outcome.success:
                value += task.deadline_s * 10.0
            if value < best_value:
                best_value = value
                best_decision = decision
        return best_decision if best_decision is not None else OffloadingDecision("local")

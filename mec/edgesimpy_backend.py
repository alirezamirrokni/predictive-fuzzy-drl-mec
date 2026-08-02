from __future__ import annotations

from typing import Callable


class EdgeSimPyBackend:
    """Use EdgeSimPy's Simulator as the production simulation clock.

    The MEC task/offloading equations remain project-specific, while EdgeSimPy
    owns time advancement and invokes the per-tick resource-management callback.
    """

    def __init__(self, tick_duration_s: float, callback: Callable[[], None], required: bool) -> None:
        self.callback = callback
        self.name = "internal-debug"
        self.model = None
        try:
            from edge_sim_py import Simulator
        except ImportError as exc:
            if required:
                raise RuntimeError(
                    "EdgeSimPy is required by the production scenario. Install requirements.txt; "
                    "predictor/simulator fallback is intentionally disabled."
                ) from exc
            return

        def resource_management_algorithm(parameters):
            _ = parameters
            self.callback()

        self.model = Simulator(
            tick_duration=float(tick_duration_s),
            tick_unit="seconds",
            dump_interval=float("inf"),
            resource_management_algorithm=resource_management_algorithm,
            stopping_criterion=lambda model: False,
        )
        self.name = "edgesimpy"

    def step(self) -> None:
        if self.model is None:
            self.callback()
        else:
            self.model.step()

    @property
    def steps(self) -> int:
        if self.model is None:
            return 0
        return int(self.model.schedule.steps)

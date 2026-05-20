from __future__ import annotations

from typing import Tuple

import numpy as np

from .device import IoTDevice


class MobilityModel:
    def __init__(self, area_width_m: float, area_height_m: float) -> None:
        self.area_width_m = area_width_m
        self.area_height_m = area_height_m

    def update(self, device: IoTDevice, rng: np.random.Generator, slot_duration_s: float) -> None:
        if device.mobility_speed_mps == 0:
            return
        angle = rng.uniform(0.0, 2.0 * np.pi)
        dx = float(np.cos(angle) * device.mobility_speed_mps * slot_duration_s)
        dy = float(np.sin(angle) * device.mobility_speed_mps * slot_duration_s)
        x = min(self.area_width_m, max(0.0, device.position[0] + dx))
        y = min(self.area_height_m, max(0.0, device.position[1] + dy))
        device.position = (x, y)

    def random_position(self, rng: np.random.Generator) -> Tuple[float, float]:
        return (float(rng.uniform(0.0, self.area_width_m)), float(rng.uniform(0.0, self.area_height_m)))

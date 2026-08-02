from __future__ import annotations

from dataclasses import dataclass
from math import log2, sqrt
from typing import Tuple

import numpy as np


@dataclass(slots=True)
class WirelessChannel:
    """Shannon channel using the table's gain and noise ranges exactly."""

    noise_power_w: float
    gain_matrix: np.ndarray

    def distance_m(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        # Coordinates are only used to build the GNN topology.
        return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def channel_gain(self, device_id: int, server_id: int) -> float:
        return float(self.gain_matrix[int(device_id), int(server_id)])

    def snr(self, tx_power_w: float, device_id: int, server_id: int) -> float:
        return max(0.0, tx_power_w * self.channel_gain(device_id, server_id) / self.noise_power_w)

    def data_rate_mbps(self, bandwidth_mhz: float, tx_power_w: float, device_id: int, server_id: int) -> float:
        return max(1e-9, bandwidth_mhz * log2(1.0 + self.snr(tx_power_w, device_id, server_id)))

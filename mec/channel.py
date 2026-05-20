from __future__ import annotations

from dataclasses import dataclass
from math import log2, sqrt
from typing import Tuple


@dataclass(slots=True)
class WirelessChannel:
    noise_power_w: float
    path_loss_exponent: float
    reference_gain: float
    min_rate_mbps: float
    packet_loss_base: float

    def distance_m(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return max(1.0, sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))

    def channel_gain(self, distance_m: float) -> float:
        return self.reference_gain / (distance_m ** self.path_loss_exponent)

    def snr(self, tx_power_w: float, distance_m: float) -> float:
        return max(0.0, tx_power_w * self.channel_gain(distance_m) / self.noise_power_w)

    def data_rate_mbps(self, bandwidth_mhz: float, tx_power_w: float, device_position: Tuple[float, float], server_position: Tuple[float, float]) -> float:
        distance = self.distance_m(device_position, server_position)
        spectral_efficiency = log2(1.0 + self.snr(tx_power_w, distance))
        return max(self.min_rate_mbps, bandwidth_mhz * spectral_efficiency)

    def packet_loss_probability(self, device_position: Tuple[float, float], server_position: Tuple[float, float]) -> float:
        distance = self.distance_m(device_position, server_position)
        distance_component = min(0.45, distance / 2000.0)
        return min(0.95, max(0.0, self.packet_loss_base + distance_component))

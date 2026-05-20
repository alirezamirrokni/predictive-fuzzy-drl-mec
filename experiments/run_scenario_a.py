from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from main import run


if __name__ == "__main__":
    for policy in ["local_only", "random", "greedy_latency", "greedy_energy"]:
        run("configs/scenario_a.yaml", policy, "data/results/scenario_a")

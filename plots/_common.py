from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


def read_metrics(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def ensure_parent(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output

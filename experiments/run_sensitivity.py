from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from main import run as run_heuristic
from mec.simulator import load_yaml_config


def save_config(config: Dict[str, Any], path: str | Path) -> Path:
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    return path


def run_server_count_sweep(base_config_path: str, output_dir: str, server_counts: Iterable[int]) -> List[Dict[str, Any]]:
    base = load_yaml_config(base_config_path)
    rows = []
    for count in server_counts:
        config = copy.deepcopy(base)
        config["servers"]["count"] = int(count)
        config["scenario_name"] = f"{base.get('scenario_name', 'scenario')}_servers_{count}"
        path = save_config(config, Path(output_dir) / "configs" / f"servers_{count}.yaml")
        for policy in ["greedy_latency", "greedy_energy"]:
            run_heuristic(str(path), policy, str(Path(output_dir) / "metrics"))
            summary_path = Path(output_dir) / "metrics" / f"{config['scenario_name']}_{policy}_summary.json"
            with summary_path.open("r", encoding="utf-8") as file:
                summary = json.load(file)
            rows.append({"sweep": "server_count", "value": count, "policy": policy, **summary})
    return rows


def run_device_count_sweep(base_config_path: str, output_dir: str, device_counts: Iterable[int]) -> List[Dict[str, Any]]:
    base = load_yaml_config(base_config_path)
    rows = []
    for count in device_counts:
        config = copy.deepcopy(base)
        config["devices"]["count"] = int(count)
        config["scenario_name"] = f"{base.get('scenario_name', 'scenario')}_devices_{count}"
        path = save_config(config, Path(output_dir) / "configs" / f"devices_{count}.yaml")
        for policy in ["greedy_latency", "greedy_energy"]:
            run_heuristic(str(path), policy, str(Path(output_dir) / "metrics"))
            summary_path = Path(output_dir) / "metrics" / f"{config['scenario_name']}_{policy}_summary.json"
            with summary_path.open("r", encoding="utf-8") as file:
                summary = json.load(file)
            rows.append({"sweep": "device_count", "value": count, "policy": policy, **summary})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-a", default="configs/scenario_a.yaml")
    parser.add_argument("--scenario-b", default="configs/scenario_b.yaml")
    parser.add_argument("--output-dir", default="data/results/sensitivity")
    parser.add_argument("--device-counts", default="600,800,1000")
    parser.add_argument("--server-counts", default="30,50,100")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device_counts = [int(item.strip()) for item in args.device_counts.split(",") if item.strip()]
    server_counts = [int(item.strip()) for item in args.server_counts.split(",") if item.strip()]
    rows: List[Dict[str, Any]] = []
    rows.extend(run_device_count_sweep(args.scenario_a, args.output_dir, device_counts))
    rows.extend(run_server_count_sweep(args.scenario_b, args.output_dir, server_counts))
    result_path = output / "sensitivity_summary.json"
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)
    print(f"saved_to={result_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Union

from .membership import OUTPUT_TERM_VALUES, clamp, fuzzify_inputs
from .rules import FuzzyRule, default_rules, validate_rules


@dataclass(frozen=True, slots=True)
class FuzzyInputs:
    latency_pressure: float
    energy_pressure: float
    server_load: float
    deadline_urgency: float
    prediction_uncertainty: float = 0.0

    def normalized(self) -> "FuzzyInputs":
        return FuzzyInputs(
            latency_pressure=clamp(self.latency_pressure),
            energy_pressure=clamp(self.energy_pressure),
            server_load=clamp(self.server_load),
            deadline_urgency=clamp(self.deadline_urgency),
            prediction_uncertainty=clamp(self.prediction_uncertainty),
        )

    def to_dict(self) -> Dict[str, float]:
        item = self.normalized()
        return {
            "latency_pressure": item.latency_pressure,
            "energy_pressure": item.energy_pressure,
            "server_load": item.server_load,
            "deadline_urgency": item.deadline_urgency,
            "prediction_uncertainty": item.prediction_uncertainty,
        }


@dataclass(frozen=True, slots=True)
class FuzzyWeights:
    energy: float
    latency: float
    success: float
    reliability: float

    def normalized(self) -> "FuzzyWeights":
        values = [
            max(0.0, self.energy),
            max(0.0, self.latency),
            max(0.0, self.success),
            max(0.0, self.reliability),
        ]
        total = sum(values)
        if total <= 0.0:
            return FuzzyWeights(0.25, 0.25, 0.25, 0.25)
        return FuzzyWeights(
            energy=values[0] / total,
            latency=values[1] / total,
            success=values[2] / total,
            reliability=values[3] / total,
        )

    def to_dict(self) -> Dict[str, float]:
        item = self.normalized()
        return {
            "energy": item.energy,
            "latency": item.latency,
            "success": item.success,
            "reliability": item.reliability,
        }

    def as_reward_vector(self) -> Dict[str, float]:
        return self.to_dict()


class FuzzyController:
    def __init__(self, rules: Optional[Iterable[FuzzyRule]] = None) -> None:
        self.rules = list(default_rules() if rules is None else rules)
        validate_rules(self.rules)
        self.default_output = FuzzyWeights(0.25, 0.25, 0.25, 0.25)

    def compute(self, inputs: Union[FuzzyInputs, Mapping[str, float]]) -> FuzzyWeights:
        if isinstance(inputs, FuzzyInputs):
            values = inputs.to_dict()
        else:
            values = FuzzyInputs(
                latency_pressure=float(inputs.get("latency_pressure", 0.0)),
                energy_pressure=float(inputs.get("energy_pressure", 0.0)),
                server_load=float(inputs.get("server_load", 0.0)),
                deadline_urgency=float(inputs.get("deadline_urgency", 0.0)),
                prediction_uncertainty=float(inputs.get("prediction_uncertainty", 0.0)),
            ).to_dict()

        memberships = fuzzify_inputs(values)

        accumulators: Dict[str, float] = {
            "energy": 0.0,
            "latency": 0.0,
            "success": 0.0,
            "reliability": 0.0,
        }
        strengths: Dict[str, float] = {
            "energy": 0.0,
            "latency": 0.0,
            "success": 0.0,
            "reliability": 0.0,
        }

        for rule in self.rules:
            activation = rule.activation(memberships)
            if activation <= 0.0:
                continue
            for objective_name, term_name in rule.consequents:
                term_value = OUTPUT_TERM_VALUES.get(term_name)
                if term_value is None:
                    raise KeyError(f"unknown output term {term_name!r}")
                accumulators[objective_name] += activation * term_value
                strengths[objective_name] += activation

        raw = {}
        for objective_name in accumulators:
            if strengths[objective_name] > 0.0:
                raw[objective_name] = accumulators[objective_name] / strengths[objective_name]
            else:
                raw[objective_name] = getattr(self.default_output, objective_name)

        return FuzzyWeights(
            energy=raw["energy"],
            latency=raw["latency"],
            success=raw["success"],
            reliability=raw["reliability"],
        ).normalized()

    def explain(self, inputs: Union[FuzzyInputs, Mapping[str, float]]) -> Dict[str, object]:
        if isinstance(inputs, FuzzyInputs):
            values = inputs.to_dict()
        else:
            values = FuzzyInputs(
                latency_pressure=float(inputs.get("latency_pressure", 0.0)),
                energy_pressure=float(inputs.get("energy_pressure", 0.0)),
                server_load=float(inputs.get("server_load", 0.0)),
                deadline_urgency=float(inputs.get("deadline_urgency", 0.0)),
                prediction_uncertainty=float(inputs.get("prediction_uncertainty", 0.0)),
            ).to_dict()

        memberships = fuzzify_inputs(values)
        fired_rules = []

        for index, rule in enumerate(self.rules):
            activation = rule.activation(memberships)
            if activation > 0.0:
                fired_rules.append(
                    {
                        "rule_index": index,
                        "activation": activation,
                        "conditions": list(rule.conditions),
                        "consequents": list(rule.consequents),
                        "description": rule.description,
                    }
                )

        return {
            "inputs": values,
            "memberships": memberships,
            "weights": self.compute(values).to_dict(),
            "fired_rules": fired_rules,
        }

    def compute_from_task_context(
        self,
        task,
        device,
        candidate_servers,
        prediction_uncertainty: float = 0.0,
    ) -> FuzzyWeights:
        servers = list(candidate_servers)

        local_latency = task.cpu_cycles_mi / device.cpu_capacity_mips
        latency_pressure = local_latency / max(task.deadline_s, 1e-9)

        estimated_local_energy = device.local_power_w * local_latency
        energy_pressure = estimated_local_energy / max(device.battery_j, 1e-9)

        if servers:
            load_values = []
            for server in servers:
                capacity = max(server.cpu_capacity_mips, 1e-9)
                load_values.append(server.queue_workload_mi / capacity)
            server_load = sum(load_values) / len(load_values)
        else:
            server_load = 1.0

        deadline_urgency = 1.0 - min(
            1.0,
            task.deadline_s / max(local_latency + task.deadline_s, 1e-9),
        )

        return self.compute(
            FuzzyInputs(
                latency_pressure=latency_pressure,
                energy_pressure=energy_pressure,
                server_load=server_load,
                deadline_urgency=deadline_urgency,
                prediction_uncertainty=prediction_uncertainty,
            )
        )


def main() -> None:
    controller = FuzzyController()

    sample = FuzzyInputs(
        latency_pressure=0.82,
        energy_pressure=0.30,
        server_load=0.70,
        deadline_urgency=0.90,
        prediction_uncertainty=0.40,
    )

    weights = controller.compute(sample)
    print(weights.to_dict())


if __name__ == "__main__":
    main()

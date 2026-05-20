from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Sequence, Tuple


Operator = Literal["and", "or"]
ObjectiveName = Literal["energy", "latency", "success", "reliability"]
TermName = str
Condition = Tuple[str, str]
Consequent = Tuple[ObjectiveName, TermName]


@dataclass(frozen=True, slots=True)
class FuzzyRule:
    conditions: Sequence[Condition]
    consequents: Sequence[Consequent]
    operator: Operator = "and"
    weight: float = 1.0
    description: str = ""

    def activation(self, memberships: Dict[str, Dict[str, float]]) -> float:
        degrees = []
        for variable_name, term_name in self.conditions:
            degrees.append(float(memberships.get(variable_name, {}).get(term_name, 0.0)))
        if not degrees:
            return 0.0
        if self.operator == "or":
            value = max(degrees)
        else:
            value = min(degrees)
        return max(0.0, min(1.0, value * self.weight))


DEFAULT_RULES: List[FuzzyRule] = [
    FuzzyRule(
        conditions=(("deadline_urgency", "urgent"), ("latency_pressure", "high")),
        consequents=(("latency", "very_high"), ("success", "high"), ("energy", "low")),
        description="urgent deadlines and high latency pressure prioritize latency and completion",
    ),
    FuzzyRule(
        conditions=(("deadline_urgency", "urgent"), ("server_load", "high")),
        consequents=(("latency", "high"), ("reliability", "high"), ("success", "high")),
        description="urgent deadlines under heavy server load require reliable successful scheduling",
    ),
    FuzzyRule(
        conditions=(("energy_pressure", "high"),),
        consequents=(("energy", "very_high"), ("latency", "medium")),
        description="high energy pressure prioritizes energy saving",
    ),
    FuzzyRule(
        conditions=(("energy_pressure", "high"), ("deadline_urgency", "relaxed")),
        consequents=(("energy", "very_high"), ("latency", "low")),
        description="relaxed deadlines allow stronger energy optimization",
    ),
    FuzzyRule(
        conditions=(("server_load", "high"), ("prediction_uncertainty", "high")),
        consequents=(("reliability", "very_high"), ("success", "high"), ("latency", "medium")),
        description="high load and uncertain prediction increase reliability importance",
    ),
    FuzzyRule(
        conditions=(("prediction_uncertainty", "high"),),
        consequents=(("reliability", "high"), ("success", "high")),
        description="uncertain predictions require conservative reliable decisions",
    ),
    FuzzyRule(
        conditions=(("server_load", "low"), ("latency_pressure", "high")),
        consequents=(("latency", "high"), ("success", "medium")),
        description="low server load makes latency-oriented offloading attractive",
    ),
    FuzzyRule(
        conditions=(("server_load", "high"), ("latency_pressure", "low")),
        consequents=(("energy", "medium"), ("reliability", "high")),
        description="when latency is not critical, avoid risky overloaded servers",
    ),
    FuzzyRule(
        conditions=(("latency_pressure", "medium"), ("energy_pressure", "medium")),
        consequents=(("latency", "medium"), ("energy", "medium"), ("success", "medium"), ("reliability", "medium")),
        description="balanced conditions use balanced objective weights",
    ),
    FuzzyRule(
        conditions=(("latency_pressure", "low"), ("energy_pressure", "low"), ("server_load", "low")),
        consequents=(("success", "high"), ("reliability", "medium"), ("latency", "medium"), ("energy", "medium")),
        description="easy conditions prioritize stable successful execution",
    ),
    FuzzyRule(
        conditions=(("deadline_urgency", "normal"), ("server_load", "medium")),
        consequents=(("success", "high"), ("latency", "medium"), ("energy", "medium"), ("reliability", "medium")),
        description="normal deadlines and medium load prioritize success with balanced cost",
    ),
    FuzzyRule(
        conditions=(("deadline_urgency", "relaxed"), ("energy_pressure", "medium")),
        consequents=(("energy", "high"), ("latency", "low"), ("success", "medium")),
        description="relaxed deadlines allow energy-aware execution",
    ),
    FuzzyRule(
        conditions=(("latency_pressure", "high"), ("energy_pressure", "low")),
        consequents=(("latency", "very_high"), ("success", "high"), ("energy", "low")),
        description="low energy pressure permits aggressive latency reduction",
    ),
    FuzzyRule(
        conditions=(("energy_pressure", "high"), ("latency_pressure", "high")),
        consequents=(("success", "very_high"), ("latency", "high"), ("energy", "high"), ("reliability", "medium")),
        description="conflicting high energy and latency pressures prioritize successful compromise",
    ),
    FuzzyRule(
        conditions=(("server_load", "medium"), ("prediction_uncertainty", "medium")),
        consequents=(("reliability", "medium"), ("success", "high")),
        description="moderate uncertainty requires success-aware scheduling",
    ),
    FuzzyRule(
        conditions=(("prediction_uncertainty", "low"), ("server_load", "low")),
        consequents=(("latency", "high"), ("energy", "medium"), ("reliability", "medium")),
        description="confident low-load predictions support latency optimization",
    ),
    FuzzyRule(
        conditions=(("deadline_urgency", "urgent"),),
        consequents=(("success", "high"), ("latency", "high")),
        description="urgent tasks generally require latency and success focus",
    ),
    FuzzyRule(
        conditions=(("server_load", "high"),),
        consequents=(("reliability", "high"), ("success", "medium")),
        description="high server load increases risk and reliability weight",
    ),
    FuzzyRule(
        conditions=(("latency_pressure", "low"), ("deadline_urgency", "relaxed")),
        consequents=(("energy", "high"), ("reliability", "medium")),
        description="non-urgent low-latency-pressure tasks can save energy",
    ),
    FuzzyRule(
        conditions=(("energy_pressure", "low"), ("prediction_uncertainty", "low")),
        consequents=(("latency", "high"), ("success", "medium")),
        description="low energy pressure and confident predictions allow performance focus",
    ),
]


def default_rules() -> List[FuzzyRule]:
    return list(DEFAULT_RULES)


def validate_rules(rules: Iterable[FuzzyRule]) -> None:
    valid_objectives = {"energy", "latency", "success", "reliability"}
    for rule in rules:
        if rule.operator not in {"and", "or"}:
            raise ValueError(f"invalid rule operator {rule.operator!r}")
        if not 0.0 <= float(rule.weight) <= 1.0:
            raise ValueError("rule weight must be in [0, 1]")
        for objective_name, _ in rule.consequents:
            if objective_name not in valid_objectives:
                raise ValueError(f"invalid consequent objective {objective_name!r}")

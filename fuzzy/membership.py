from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Union


Number = Union[float, int]


def clamp(value: Number, lower: float = 0.0, upper: float = 1.0) -> float:
    value = float(value)
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


@dataclass(frozen=True, slots=True)
class TriangularMF:
    left: float
    center: float
    right: float

    def __call__(self, value: Number) -> float:
        x = float(value)
        if self.left == self.center and x <= self.center:
            return 1.0 if x == self.center else 0.0
        if self.center == self.right and x >= self.center:
            return 1.0 if x == self.center else 0.0
        if x <= self.left or x >= self.right:
            return 0.0
        if x == self.center:
            return 1.0
        if x < self.center:
            denominator = self.center - self.left
            return 0.0 if denominator == 0 else clamp((x - self.left) / denominator)
        denominator = self.right - self.center
        return 0.0 if denominator == 0 else clamp((self.right - x) / denominator)


@dataclass(frozen=True, slots=True)
class TrapezoidalMF:
    left: float
    left_top: float
    right_top: float
    right: float

    def __call__(self, value: Number) -> float:
        x = float(value)
        if x <= self.left or x >= self.right:
            if self.left == self.left_top and x == self.left:
                return 1.0
            if self.right_top == self.right and x == self.right:
                return 1.0
            return 0.0
        if self.left_top <= x <= self.right_top:
            return 1.0
        if self.left < x < self.left_top:
            denominator = self.left_top - self.left
            return 0.0 if denominator == 0 else clamp((x - self.left) / denominator)
        denominator = self.right - self.right_top
        return 0.0 if denominator == 0 else clamp((self.right - x) / denominator)


@dataclass(frozen=True, slots=True)
class LinguisticVariable:
    name: str
    terms: Mapping[str, Union[TriangularMF, TrapezoidalMF]]

    def fuzzify(self, value: Number) -> Dict[str, float]:
        x = clamp(value)
        return {term: mf(x) for term, mf in self.terms.items()}

    def degree(self, term: str, value: Number) -> float:
        if term not in self.terms:
            raise KeyError(f"unknown fuzzy term {term!r} for variable {self.name!r}")
        return self.terms[term](clamp(value))


INPUT_VARIABLES: Dict[str, LinguisticVariable] = {
    "latency_pressure": LinguisticVariable(
        "latency_pressure",
        {
            "low": TrapezoidalMF(0.0, 0.0, 0.20, 0.45),
            "medium": TriangularMF(0.25, 0.50, 0.75),
            "high": TrapezoidalMF(0.55, 0.80, 1.0, 1.0),
        },
    ),
    "energy_pressure": LinguisticVariable(
        "energy_pressure",
        {
            "low": TrapezoidalMF(0.0, 0.0, 0.20, 0.45),
            "medium": TriangularMF(0.25, 0.50, 0.75),
            "high": TrapezoidalMF(0.55, 0.80, 1.0, 1.0),
        },
    ),
    "server_load": LinguisticVariable(
        "server_load",
        {
            "low": TrapezoidalMF(0.0, 0.0, 0.25, 0.50),
            "medium": TriangularMF(0.30, 0.55, 0.80),
            "high": TrapezoidalMF(0.60, 0.82, 1.0, 1.0),
        },
    ),
    "deadline_urgency": LinguisticVariable(
        "deadline_urgency",
        {
            "relaxed": TrapezoidalMF(0.0, 0.0, 0.20, 0.45),
            "normal": TriangularMF(0.25, 0.50, 0.75),
            "urgent": TrapezoidalMF(0.55, 0.80, 1.0, 1.0),
        },
    ),
    "prediction_uncertainty": LinguisticVariable(
        "prediction_uncertainty",
        {
            "low": TrapezoidalMF(0.0, 0.0, 0.20, 0.45),
            "medium": TriangularMF(0.25, 0.50, 0.75),
            "high": TrapezoidalMF(0.55, 0.80, 1.0, 1.0),
        },
    ),
}


OUTPUT_TERM_VALUES: Dict[str, float] = {
    "very_low": 0.05,
    "low": 0.20,
    "medium": 0.50,
    "high": 0.80,
    "very_high": 0.95,
}


def fuzzify_inputs(values: Mapping[str, Number]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for variable_name, variable in INPUT_VARIABLES.items():
        result[variable_name] = variable.fuzzify(values.get(variable_name, 0.0))
    return result

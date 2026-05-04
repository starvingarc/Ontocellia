from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import log2
from typing import Any


@dataclass(slots=True)
class OrganValidationResult:
    name: str
    passed: bool
    score: float
    target: str = ""
    evidence: str = ""
    cost: float = 0.0
    risk: float = 0.0
    latency: float = 0.0
    output_digest: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OrganSelectionTarget:
    min_coverage: float = 0.5
    min_diversity: float = 0.3
    min_validation_score: float = 0.7
    max_risk: float = 0.4
    max_cost: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class OrganFeedbackSignal:
    selection_pressure: float = 0.0
    validation_pressure: float = 0.0
    risk_pressure: float = 0.0
    resource_pressure: float = 0.0
    reward_signal: float = 0.0
    by_fate: dict[str, float] = field(default_factory=dict)
    by_gene: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.selection_pressure = _clamp(self.selection_pressure)
        self.validation_pressure = _clamp(self.validation_pressure)
        self.risk_pressure = _clamp(self.risk_pressure)
        self.resource_pressure = _clamp(self.resource_pressure)
        self.reward_signal = _clamp(self.reward_signal)
        self.by_fate = {str(name): _clamp(value) for name, value in self.by_fate.items()}
        self.by_gene = {str(name): _clamp(value) for name, value in self.by_gene.items()}

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OrganSelectionReport:
    metrics: dict[str, float]
    targets: OrganSelectionTarget
    feedback: OrganFeedbackSignal
    validation_results: list[OrganValidationResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "targets": self.targets.as_dict(),
            "feedback": self.feedback.as_dict(),
            "validation_results": [result.as_dict() for result in self.validation_results],
        }


@dataclass(slots=True)
class OrganSelectionField:
    strength: float = 0.35

    def evaluate(
        self,
        tissue: Any,
        validation_results: list[OrganValidationResult] | None = None,
    ) -> OrganSelectionReport:
        results = list(validation_results or [])
        targets = getattr(tissue.environment, "selection_targets", OrganSelectionTarget())
        coverage = _coverage(tissue)
        diversity = _diversity(tissue.fate_counts())
        validation_score = _validation_score(results)
        risk = _validation_risk(results)
        cost = _validation_cost(results)

        coverage_deficit = _positive(targets.min_coverage - coverage)
        diversity_deficit = _positive(targets.min_diversity - diversity)
        validation_deficit = _positive(targets.min_validation_score - validation_score) if results else 0.0
        risk_excess = _positive(risk - targets.max_risk)
        cost_excess = _positive(cost - targets.max_cost)

        selection_pressure = _clamp(self.strength * (coverage_deficit + diversity_deficit + validation_deficit))
        validation_pressure = _clamp(self.strength * validation_deficit)
        risk_pressure = _clamp(self.strength * risk_excess)
        resource_pressure = _clamp(self.strength * cost_excess)
        reward_signal = _clamp(self.strength * min(coverage, validation_score) if results and validation_score >= targets.min_validation_score and coverage >= targets.min_coverage else 0.0)

        by_fate = {
            "repair": _clamp(validation_pressure + selection_pressure * 0.5),
            "reviewer": _clamp(validation_pressure + risk_pressure + selection_pressure * 0.25),
            "explorer": _clamp(coverage_deficit * self.strength + diversity_deficit * self.strength),
            "planner": _clamp(coverage_deficit * self.strength + diversity_deficit * self.strength * 0.5),
            "quiescent": _clamp(resource_pressure + risk_pressure * 0.5),
        }
        by_gene = _gene_feedback(getattr(tissue.genome, "genes", []), by_fate)

        feedback = OrganFeedbackSignal(
            selection_pressure=selection_pressure,
            validation_pressure=validation_pressure,
            risk_pressure=risk_pressure,
            resource_pressure=resource_pressure,
            reward_signal=reward_signal,
            by_fate=by_fate,
            by_gene=by_gene,
        )
        return OrganSelectionReport(
            metrics={
                "coverage": coverage,
                "diversity": diversity,
                "validation_score": validation_score,
                "risk": risk,
                "cost": cost,
                "validation_pass_rate": _validation_pass_rate(results),
            },
            targets=targets,
            feedback=feedback,
            validation_results=results,
        )


def _coverage(tissue: Any) -> float:
    niches = getattr(tissue.environment, "niches", [])
    if not niches:
        return 1.0
    tissue._refresh_niche_occupancy()
    demand = sum(max(1, int(getattr(niche, "demand", 1))) for niche in niches)
    occupied = sum(min(len(getattr(niche, "occupied_by", [])), max(1, int(getattr(niche, "demand", 1)))) for niche in niches)
    return _clamp(occupied / max(1, demand))


def _diversity(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * log2(probability)
    return _clamp(entropy / log2(len(counts)))


def _validation_score(results: list[OrganValidationResult]) -> float:
    if not results:
        return 1.0
    return _clamp(sum(_clamp(result.score) for result in results) / len(results))


def _validation_risk(results: list[OrganValidationResult]) -> float:
    if not results:
        return 0.0
    return _clamp(max(max(0.0, result.risk) for result in results))


def _validation_cost(results: list[OrganValidationResult]) -> float:
    if not results:
        return 0.0
    return max(0.0, sum(max(0.0, result.cost) for result in results) / len(results))


def _validation_pass_rate(results: list[OrganValidationResult]) -> float:
    if not results:
        return 1.0
    return _clamp(sum(1 for result in results if result.passed) / len(results))


def _gene_feedback(genes: list[Any], by_fate: dict[str, float]) -> dict[str, float]:
    feedback: dict[str, float] = {}
    for gene in genes:
        fate = str(getattr(gene, "fate_bias", getattr(gene, "category", "")))
        value = by_fate.get(fate, 0.0)
        if value > 0.0:
            feedback[str(gene.id)] = value
    return feedback


def _positive(value: float) -> float:
    return max(0.0, float(value))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

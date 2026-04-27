from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class FateAttractor:
    fate: str
    morphogens: list[str]
    threshold: float = 0.4
    commitment: float = 1.0
    hysteresis: float = 0.15
    competence_window: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FateDecision:
    fate: str
    score: float
    scores: dict[str, float]
    threshold: float
    niche_bias: str | None = None
    reason: str = "attractor"


@dataclass(slots=True)
class FateLandscape:
    attractors: list[FateAttractor] = field(default_factory=list)

    @classmethod
    def default(cls) -> "FateLandscape":
        return cls(
            attractors=[
                FateAttractor("explorer", ["ambiguity", "missing_context", "exploration_pressure"]),
                FateAttractor("repair", ["test_failure", "repair_pressure", "niche_vacancy", "damage"]),
                FateAttractor("reviewer", ["review_pressure", "risk", "verification_pressure"]),
                FateAttractor("builder", ["implementation_pressure"]),
                FateAttractor("planner", ["coordination_pressure", "planning_pressure"]),
                FateAttractor("memory", ["memory_pressure", "context_pressure"]),
                FateAttractor("quiescent", ["quiescence_pressure"], threshold=0.2, commitment=0.8),
            ]
        )

    def decide(
        self,
        cell: Any,
        genome: Any,
        local_signals: dict[str, float],
        niche_bias: str | None = None,
    ) -> FateDecision:
        attractors = self.attractors or self.default().attractors
        genome_fates = _genome_fates(genome)
        scores: dict[str, float] = {}
        thresholds: dict[str, float] = {}
        for attractor in attractors:
            if genome_fates and attractor.fate != "quiescent" and attractor.fate not in genome_fates:
                continue
            score = sum(float(local_signals.get(name, 0.0)) for name in attractor.morphogens) * attractor.commitment
            if str(cell.fate) == attractor.fate:
                score += attractor.hysteresis
            score *= float(getattr(cell.competence, "fate_scores", {}).get(attractor.fate, 1.0))
            if niche_bias == attractor.fate:
                score += 0.35
            scores[attractor.fate] = score
            thresholds[attractor.fate] = attractor.threshold
        if not scores:
            return FateDecision(fate="quiescent", score=0.0, scores={}, threshold=0.0, niche_bias=niche_bias, reason="no_attractor")
        fate = min(scores, key=lambda item: (-scores[item], item))
        score = scores[fate]
        threshold = thresholds[fate]
        if score < threshold:
            fallback = str(cell.fate) if getattr(cell, "differentiated", False) else "quiescent"
            return FateDecision(
                fate=fallback,
                score=score,
                scores=scores,
                threshold=threshold,
                niche_bias=niche_bias,
                reason="below_threshold",
            )
        return FateDecision(
            fate=fate,
            score=score,
            scores=scores,
            threshold=threshold,
            niche_bias=niche_bias,
            reason="attractor",
        )


def _genome_fates(genome: Any) -> set[str]:
    genes = getattr(genome, "genes", [])
    return {str(getattr(gene, "fate_bias", getattr(gene, "category", ""))) for gene in genes}

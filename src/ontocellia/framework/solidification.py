from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from ontocellia.framework.genome import AgentGenome, EpigeneticMarks, RegulatoryElement
from ontocellia.framework.mutation import genome_to_dict
from ontocellia.framework.structure_search import StructureSearchReport, StructureTrialResult, StructureVariant


@dataclass(slots=True)
class SelectionSolidificationPolicy:
    min_structure_score: float = 0.65
    min_margin: float = 0.03
    min_repetitions: int = 1
    regulatory_strength: float = 0.16
    max_tendencies: int = 3

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SolidifiedTendency:
    id: str
    variant_name: str
    confidence: float
    morphogen_bias: dict[str, float] = field(default_factory=dict)
    niche_demand_bias: dict[str, int] = field(default_factory=dict)
    population_delta: int = 0
    regulatory_bias: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SelectionSolidificationReport:
    decision: str
    decision_reason: str
    tendencies: list[SolidifiedTendency]
    selected_tendency: SolidifiedTendency | None
    solidified_genome: AgentGenome
    policy: SelectionSolidificationPolicy

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "decision_reason": self.decision_reason,
            "tendencies": [tendency.as_dict() for tendency in self.tendencies],
            "selected_tendency": self.selected_tendency.as_dict() if self.selected_tendency else None,
            "policy": self.policy.as_dict(),
        }

    def write(self, output: str | Path) -> dict[str, str]:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "tendencies": output_dir / "solidified_tendencies.json",
            "report": output_dir / "solidification_report.md",
            "genome": output_dir / "solidified_genome.yaml",
        }
        paths["tendencies"].write_text(
            json.dumps([tendency.as_dict() for tendency in self.tendencies], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        paths["report"].write_text(_markdown_report(self), encoding="utf-8")
        paths["genome"].write_text(yaml.safe_dump(genome_to_dict(self.solidified_genome), sort_keys=False), encoding="utf-8")
        return {key: str(path) for key, path in paths.items()}


class SelectionSolidificationRuntime:
    def solidify(
        self,
        reports: StructureSearchReport | list[StructureSearchReport] | dict[str, Any] | list[dict[str, Any]],
        *,
        genome: AgentGenome,
        policy: SelectionSolidificationPolicy | None = None,
    ) -> SelectionSolidificationReport:
        active_policy = policy or SelectionSolidificationPolicy()
        normalized = _normalize_reports(reports)
        candidates = _candidate_tendencies(normalized, genome, active_policy)
        selected = candidates[0] if candidates else None
        solidified = _apply_tendency_to_genome(genome, selected, active_policy) if selected is not None else _copy_genome(genome)
        if selected is None:
            return SelectionSolidificationReport(
                "not_selected",
                "No structure variant met the score, margin, and repetition thresholds.",
                [],
                None,
                solidified,
                active_policy,
            )
        return SelectionSolidificationReport(
            "selected",
            f"Variant {selected.variant_name} met structure score, margin, and repetition thresholds.",
            candidates[: active_policy.max_tendencies],
            selected,
            solidified,
            active_policy,
        )

    def solidify_summary_file(
        self,
        path: str | Path,
        *,
        genome: AgentGenome,
        policy: SelectionSolidificationPolicy | None = None,
    ) -> SelectionSolidificationReport:
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        return self.solidify(summary, genome=genome, policy=policy)


def _normalize_reports(reports: StructureSearchReport | list[StructureSearchReport] | dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(reports, (StructureSearchReport, dict)):
        items = [reports]
    else:
        items = list(reports)
    normalized: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, StructureSearchReport):
            normalized.append(
                {
                    "task": item.task,
                    "domain": item.domain,
                    "selected_variant": item.selected_variant,
                    "trials": [trial.as_dict() for trial in item.trials],
                }
            )
        elif isinstance(item, dict):
            normalized.append(dict(item))
    return normalized


def _candidate_tendencies(
    reports: list[dict[str, Any]],
    genome: AgentGenome,
    policy: SelectionSolidificationPolicy,
) -> list[SolidifiedTendency]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        selected = str(report.get("selected_variant", ""))
        if not selected:
            continue
        trial = _trial_by_variant(report, selected)
        baseline = _trial_by_variant(report, "baseline")
        if not trial:
            continue
        margin = _trial_score(trial) - _trial_score(baseline)
        enriched = {**trial, "margin": margin, "task": report.get("task", ""), "domain": report.get("domain", "")}
        grouped.setdefault(selected, []).append(enriched)
    tendencies: list[SolidifiedTendency] = []
    for variant_name, trials in grouped.items():
        repetitions = len(trials)
        average_score = sum(_trial_score(trial) for trial in trials) / max(1, repetitions)
        average_margin = sum(float(trial.get("margin", 0.0)) for trial in trials) / max(1, repetitions)
        if repetitions < policy.min_repetitions or average_score < policy.min_structure_score or average_margin < policy.min_margin:
            continue
        variant = _variant_from_trial(trials[0])
        confidence = min(1.0, average_score * 0.75 + average_margin * 1.25)
        regulatory_bias = _regulatory_bias(genome, variant, confidence, policy)
        tendencies.append(
            SolidifiedTendency(
                id=f"tendency:{variant.name}",
                variant_name=variant.name,
                confidence=round(confidence, 6),
                morphogen_bias=dict(variant.morphogen_patch),
                niche_demand_bias=dict(variant.niche_demand_patch),
                population_delta=variant.target_population_delta,
                regulatory_bias=regulatory_bias,
                evidence=[f"{trial.get('task', '')}: score={_trial_score(trial):.3f} margin={float(trial.get('margin', 0.0)):.3f}" for trial in trials],
                metrics={"average_score": round(average_score, 6), "average_margin": round(average_margin, 6), "repetitions": repetitions},
            )
        )
    tendencies.sort(key=lambda tendency: (-tendency.confidence, tendency.variant_name))
    return tendencies


def _trial_by_variant(report: dict[str, Any], variant_name: str) -> dict[str, Any]:
    for trial in report.get("trials", []) or []:
        variant = trial.get("variant", {})
        name = variant.get("name") if isinstance(variant, dict) else variant
        if name == variant_name:
            return dict(trial)
    return {}


def _trial_score(trial: dict[str, Any]) -> float:
    if not trial:
        return 0.0
    if "score" in trial:
        return float(trial.get("score", 0.0))
    metrics = trial.get("metrics", {}) if isinstance(trial.get("metrics"), dict) else {}
    return float(metrics.get("structure_score", 0.0))


def _variant_from_trial(trial: dict[str, Any]) -> StructureVariant:
    variant = trial.get("variant", {})
    if isinstance(variant, StructureVariant):
        return variant
    if not isinstance(variant, dict):
        return StructureVariant(str(variant))
    return StructureVariant(
        name=str(variant.get("name", "")),
        morphogen_patch={str(key): float(value) for key, value in dict(variant.get("morphogen_patch", {})).items()},
        niche_demand_patch={str(key): int(value) for key, value in dict(variant.get("niche_demand_patch", {})).items()},
        target_population_delta=int(variant.get("target_population_delta", 0)),
        min_population_before_differentiation=variant.get("min_population_before_differentiation"),
    )


def _regulatory_bias(genome: AgentGenome, variant: StructureVariant, confidence: float, policy: SelectionSolidificationPolicy) -> list[dict[str, Any]]:
    target_fates = set(variant.niche_demand_patch)
    if not target_fates:
        target_fates = _target_fates_from_morphogens(variant.morphogen_patch)
    signals = list(variant.morphogen_patch) or [f"{fate}_pressure" for fate in sorted(target_fates)]
    result = []
    for gene in genome.genes:
        if gene.fate_bias not in target_fates:
            continue
        result.append(
            {
                "id": f"solidified:{variant.name}:{gene.id}",
                "kind": "enhancer",
                "target_gene_id": gene.id,
                "signals": signals,
                "strength": round(policy.regulatory_strength * confidence, 6),
            }
        )
    return result


def _target_fates_from_morphogens(morphogens: dict[str, float]) -> set[str]:
    fates: set[str] = set()
    for name in morphogens:
        if "repair" in name or "test_failure" in name:
            fates.add("repair")
        if "review" in name or "risk" in name or "verification" in name:
            fates.add("reviewer")
        if "memory" in name or "context" in name:
            fates.add("memory")
    return fates


def _apply_tendency_to_genome(genome: AgentGenome, tendency: SolidifiedTendency, policy: SelectionSolidificationPolicy) -> AgentGenome:
    metadata = dict(genome.metadata)
    previous = list(metadata.get("solidified_tendencies", []))
    metadata["solidified_tendencies"] = [*previous, tendency.as_dict()]
    existing_ids = {element.id for element in genome.regulatory_elements}
    added = [
        RegulatoryElement(
            id=str(item["id"]),
            kind=str(item.get("kind", "enhancer")),
            target_gene_id=str(item["target_gene_id"]),
            signals=[str(signal) for signal in item.get("signals", [])],
            strength=float(item.get("strength", policy.regulatory_strength)),
        )
        for item in tendency.regulatory_bias
        if str(item["id"]) not in existing_ids
    ]
    return AgentGenome(
        genes=[replace(gene) for gene in genome.genes],
        metadata=metadata,
        regulatory_elements=[replace(element) for element in genome.regulatory_elements] + added,
        epigenetic_defaults=EpigeneticMarks(
            fate_locks=dict(genome.epigenetic_defaults.fate_locks),
            gene_locks=dict(genome.epigenetic_defaults.gene_locks),
        ),
        mutation_history=list(genome.mutation_history),
    )


def _copy_genome(genome: AgentGenome) -> AgentGenome:
    return AgentGenome(
        genes=[replace(gene) for gene in genome.genes],
        metadata=deepcopy(genome.metadata),
        regulatory_elements=[replace(element) for element in genome.regulatory_elements],
        epigenetic_defaults=EpigeneticMarks(
            fate_locks=dict(genome.epigenetic_defaults.fate_locks),
            gene_locks=dict(genome.epigenetic_defaults.gene_locks),
        ),
        mutation_history=list(genome.mutation_history),
    )


def _markdown_report(report: SelectionSolidificationReport) -> str:
    selected = report.selected_tendency.variant_name if report.selected_tendency else "none"
    lines = [
        "# Selection Solidification Report",
        "",
        f"- Decision: `{report.decision}`",
        f"- Reason: {report.decision_reason}",
        f"- Selected tendency: `{selected}`",
        f"- Tendencies: {len(report.tendencies)}",
        "",
    ]
    if report.tendencies:
        lines.extend(["| Variant | Confidence | Repetitions | Average score | Margin |", "| --- | ---: | ---: | ---: | ---: |"])
        for tendency in report.tendencies:
            metrics = tendency.metrics
            lines.append(
                f"| {tendency.variant_name} | {tendency.confidence:.3f} | {int(metrics.get('repetitions', 0))} | {float(metrics.get('average_score', 0.0)):.3f} | {float(metrics.get('average_margin', 0.0)):.3f} |"
            )
    return "\n".join(lines) + "\n"

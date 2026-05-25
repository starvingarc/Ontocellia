from __future__ import annotations

import csv
import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ontocellia.framework.attribution import ContributionAttributionRuntime
from ontocellia.framework.cell import CellPosition
from ontocellia.framework.core import Niche, TaskMicroenvironment, TissueRuntime
from ontocellia.framework.induction import InductionRequest, TemplateInductionCompiler
from ontocellia.framework.llm import EffectorRuntime, MockLLMProvider
from ontocellia.framework.model_config import resolve_effector_provider
from ontocellia.framework.selection import OrganValidationResult


@dataclass(slots=True)
class StructureVariant:
    name: str
    morphogen_patch: dict[str, float] = field(default_factory=dict)
    niche_demand_patch: dict[str, int] = field(default_factory=dict)
    target_population_delta: int = 0
    min_population_before_differentiation: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StructureTrialResult:
    variant: StructureVariant
    score: float
    metrics: dict[str, Any]
    artifacts: dict[str, str]
    trace_summary: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant.as_dict(),
            "score": self.score,
            "metrics": dict(self.metrics),
            "artifacts": dict(self.artifacts),
            "trace_summary": dict(self.trace_summary),
        }


@dataclass(slots=True)
class StructureSearchReport:
    task: str
    domain: str
    output_dir: Path
    trials: list[StructureTrialResult]
    selected_variant: str
    summary_path: Path
    csv_path: Path
    report_path: Path
    selected_path: Path


class StructureSearchRunner:
    def __init__(
        self,
        *,
        task: str,
        domain: str = "repo_repair",
        effector: str = "mock-llm",
        model_profile: str | None = None,
        steps: int = 6,
        seed: int = 7,
        variants: list[StructureVariant] | None = None,
        with_attribution: bool = False,
    ) -> None:
        self.task = task
        self.domain = domain
        self.effector = effector
        self.model_profile = model_profile
        self.steps = steps
        self.seed = seed
        self.variants = list(variants or builtin_structure_variants())
        self.with_attribution = with_attribution

    def run(self, output: str | Path) -> StructureSearchReport:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        draft = TemplateInductionCompiler().compile(
            InductionRequest(
                task=self.task,
                domain=self.domain,
                available_interfaces=["workspace", "pytest", "git"],
                seed=self.seed,
            )
        )
        provider = self._provider()
        validation = [OrganValidationResult("structure_search", False, 0.25, self.task, "structure search pressure", 0.1, 0.25, 0.0)]
        trials = [
            self._run_variant(
                variant,
                draft.genome,
                _clone_environment(draft.environment),
                provider,
                validation,
                output_dir / "variants" / variant.name,
            )
            for variant in self.variants
        ]
        selected = sorted(trials, key=lambda trial: (-trial.score, trial.variant.name))[0]
        summary_path = output_dir / "structure_search_summary.json"
        csv_path = output_dir / "structure_trials.csv"
        report_path = output_dir / "structure_search_report.md"
        selected_path = output_dir / "selected_variant.json"
        summary_path.write_text(json.dumps(_summary(self.task, self.domain, selected, trials), indent=2, sort_keys=True), encoding="utf-8")
        _write_csv(csv_path, trials)
        report_path.write_text(_markdown_report(self.task, selected, trials), encoding="utf-8")
        selected_path.write_text(json.dumps({"variant": selected.variant.name, "score": selected.score, "metrics": selected.metrics}, indent=2, sort_keys=True), encoding="utf-8")
        return StructureSearchReport(
            task=self.task,
            domain=self.domain,
            output_dir=output_dir,
            trials=trials,
            selected_variant=selected.variant.name,
            summary_path=summary_path,
            csv_path=csv_path,
            report_path=report_path,
            selected_path=selected_path,
        )

    def _run_variant(
        self,
        variant: StructureVariant,
        genome: Any,
        environment: TaskMicroenvironment,
        provider: Any,
        validation_results: list[OrganValidationResult],
        output_dir: Path,
    ) -> StructureTrialResult:
        started = time.perf_counter()
        output_dir.mkdir(parents=True, exist_ok=True)
        _apply_variant(environment, variant)
        tissue = TissueRuntime.seeded(genome, environment, seed=self.seed)
        if variant.min_population_before_differentiation is not None:
            tissue.min_population_before_differentiation = variant.min_population_before_differentiation
        if variant.target_population_delta:
            tissue.target_population = max(1, (tissue.target_population or len(tissue.cells)) + variant.target_population_delta)
        tissue.develop(ticks=self.steps, validation_results=validation_results)
        actions = tissue.execute(effectors=EffectorRuntime(provider))
        if variant.name in {"review_heavy", "repair_heavy"}:
            handoff = _handoff_action(tissue)
            if handoff is not None:
                actions.append(handoff)
                tissue.communicate([handoff])
        tissue.develop(ticks=1, validation_results=validation_results)
        metrics = _metrics(tissue, actions, validation_results, time.perf_counter() - started)
        score = float(metrics["structure_score"])
        artifacts = _write_variant_artifacts(output_dir, tissue, actions)
        if self.with_attribution:
            attribution = ContributionAttributionRuntime().analyze(
                tissue=tissue,
                actions=actions,
                validation_results=validation_results,
            )
            attribution_paths = attribution.write(output_dir / "attribution")
            metrics = {**metrics, "attribution": attribution.summary}
            artifacts = {**artifacts, "attribution": attribution_paths["summary"]}
        return StructureTrialResult(variant, score, metrics, artifacts, _trace_summary(tissue))

    def _provider(self) -> Any:
        if self.effector == "mock-llm":
            return MockLLMProvider()
        return resolve_effector_provider(self.effector, model_profile=self.model_profile)


def builtin_structure_variants() -> list[StructureVariant]:
    return [
        StructureVariant("baseline"),
        StructureVariant("repair_heavy", {"repair_pressure": 0.35, "test_failure": 0.25}, {"repair": 1}, target_population_delta=1),
        StructureVariant("review_heavy", {"review_pressure": 0.45, "verification_pressure": 0.35, "risk": 0.35}, {"reviewer": 1}, target_population_delta=1),
        StructureVariant("memory_heavy", {"memory_pressure": 0.45, "context_pressure": 0.3, "coordination_pressure": 0.25}, {"memory": 1}, target_population_delta=1),
        StructureVariant("lean", {"resource_pressure": 0.45}, {}, target_population_delta=-2, min_population_before_differentiation=3),
    ]


def _apply_variant(environment: TaskMicroenvironment, variant: StructureVariant) -> None:
    for name, amount in variant.morphogen_patch.items():
        environment.morphogens.emit(name, amount)
    for fate, delta in variant.niche_demand_patch.items():
        for niche in environment.niches:
            if niche.required_fate == fate:
                niche.demand = max(1, niche.demand + delta)
                break
        else:
            node = f"{fate}-variant-niche"
            environment.niches.append(Niche(node, fate, CellPosition(node, "variant")))


def _clone_environment(environment: TaskMicroenvironment) -> TaskMicroenvironment:
    return deepcopy(environment)


def _metrics(tissue: TissueRuntime, actions: list[dict[str, Any]], validation_results: list[OrganValidationResult], latency: float) -> dict[str, Any]:
    events = tissue.trace.events
    fate_counts = tissue.fate_counts()
    action_count = max(1, len(actions))
    matrix_records = len(tissue.environment.matrix.records)
    handoffs = sum(1 for event in events if event["type"] == "handoff_completed")
    regeneration_events = [event for event in events if event["type"] == "regeneration"]
    validation_score = sum(result.score for result in validation_results) / max(1, len(validation_results))
    fate_diversity = min(1.0, len(fate_counts) / 5.0)
    cost = float(len(actions) + len(tissue.cells) * 0.1)
    cost_efficiency = 1.0 / (1.0 + cost * 0.1)
    matrix_reuse = min(1.0, matrix_records / action_count)
    handoff_rate = min(1.0, handoffs / action_count)
    recovery_ticks = float(len(regeneration_events) * 4)
    fate_match = _fate_match(fate_counts)
    score_parts = {
        "validation_score": validation_score,
        "fate_match": fate_match,
        "matrix_reuse_rate": matrix_reuse,
        "handoff_completion_rate": handoff_rate,
        "regeneration_score": min(1.0, recovery_ticks / 4.0) if recovery_ticks else 0.5,
        "cost_efficiency": cost_efficiency,
        "fate_diversity": fate_diversity,
    }
    resource_report = tissue.last_resource_report.as_dict() if tissue.last_resource_report is not None else {}
    resource_efficiency = float(resource_report.get("resource_efficiency", 1.0))
    score = round(
        score_parts["validation_score"] * 0.18
        + score_parts["fate_match"] * 0.18
        + score_parts["matrix_reuse_rate"] * 0.15
        + score_parts["handoff_completion_rate"] * 0.15
        + score_parts["regeneration_score"] * 0.1
        + score_parts["cost_efficiency"] * 0.1
        + score_parts["fate_diversity"] * 0.09
        + resource_efficiency * 0.05,
        6,
    )
    return {
        **score_parts,
        "fate_distribution": fate_counts,
        "cost": round(cost, 6),
        "latency_seconds": round(latency, 6),
        "regeneration_recovery_ticks": recovery_ticks,
        "average_cell_energy": float(resource_report.get("average_cell_energy", 1.0)),
        "population_pressure": float(resource_report.get("population_pressure", 0.0)),
        "resource_efficiency": resource_efficiency,
        "structure_score": score,
    }


def _fate_match(fate_counts: dict[str, int]) -> float:
    expected = ["explorer", "repair", "reviewer", "memory"]
    return sum(1 for fate in expected if fate_counts.get(fate, 0) > 0) / len(expected)


def _trace_summary(tissue: TissueRuntime) -> dict[str, Any]:
    return {
        "population": len(tissue.cells),
        "target_population": tissue.target_population,
        "development_stage": tissue.development_stage,
        "stage_counts": tissue.stage_counts(),
        "fate_counts": tissue.fate_counts(),
        "matrix_records": len(tissue.environment.matrix.records),
        "handoffs": sum(1 for event in tissue.trace.events if event["type"] == "handoff_completed"),
        "proliferation_events": sum(1 for event in tissue.trace.events if event["type"] == "proliferation"),
        "resource_competition": tissue.last_resource_report.as_dict() if tissue.last_resource_report is not None else {},
    }


def _handoff_action(tissue: TissueRuntime) -> dict[str, Any] | None:
    repair = next((cell for cell in tissue.cells.values() if cell.fate == "repair"), None)
    if repair is None:
        return None
    return {
        "cell_id": repair.id,
        "fate": repair.fate,
        "expressed_gene_ids": list(repair.expressed_gene_ids),
        "intent_type": "propose_patch",
        "target": repair.niche_id or repair.position.node_id,
        "rationale": "Structure search handoff from repair to reviewer.",
        "required_interfaces": ["workspace"],
        "confidence": 0.8,
        "validation_hooks": [],
        "payload": {"message": "Patch ready for review.", "handoff_to_fate": "reviewer", "matrix_tags": ["patch", "review"]},
    }


def _write_variant_artifacts(output_dir: Path, tissue: TissueRuntime, actions: list[dict[str, Any]]) -> dict[str, str]:
    paths = {
        "summary": output_dir / "tissue_summary.json",
        "trace": output_dir / "tissue_trace.json",
        "actions": output_dir / "action_intents.json",
    }
    paths["summary"].write_text(json.dumps(_trace_summary(tissue), indent=2, sort_keys=True), encoding="utf-8")
    paths["trace"].write_text(json.dumps(tissue.trace.events, indent=2, sort_keys=True), encoding="utf-8")
    paths["actions"].write_text(json.dumps(actions, indent=2, sort_keys=True), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def _summary(task: str, domain: str, selected: StructureTrialResult, trials: list[StructureTrialResult]) -> dict[str, Any]:
    summary = {
        "task": task,
        "domain": domain,
        "variants": len(trials),
        "selected_variant": selected.variant.name,
        "selected_score": selected.score,
        "trials": [trial.as_dict() for trial in trials],
    }
    if isinstance(selected.metrics.get("attribution"), dict):
        summary["selected_variant_explanation"] = selected.metrics["attribution"]
    return summary


def _write_csv(path: Path, trials: list[StructureTrialResult]) -> None:
    fieldnames = [
        "variant",
        "structure_score",
        "validation_score",
        "fate_match",
        "matrix_reuse_rate",
        "handoff_completion_rate",
        "regeneration_recovery_ticks",
        "cost",
        "cost_efficiency",
        "resource_efficiency",
        "average_cell_energy",
        "population_pressure",
        "fate_diversity",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trial in trials:
            row = {"variant": trial.variant.name, **trial.metrics}
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _markdown_report(task: str, selected: StructureTrialResult, trials: list[StructureTrialResult]) -> str:
    lines = [
        "# Structure Search Report",
        "",
        f"- Task: {task}",
        f"- Selected variant: `{selected.variant.name}`",
        "",
        "| Variant | Score | Validation | Matrix | Handoff | Cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for trial in trials:
        metrics = trial.metrics
        lines.append(
            f"| {trial.variant.name} | {trial.score:.3f} | {metrics['validation_score']:.3f} | {metrics['matrix_reuse_rate']:.3f} | {metrics['handoff_completion_rate']:.3f} | {metrics['cost']:.3f} |"
        )
    return "\n".join(lines) + "\n"

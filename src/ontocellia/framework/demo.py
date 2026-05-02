from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ontocellia.framework.induction import InductionRequest, TemplateInductionCompiler
from ontocellia.framework.llm import EffectorRuntime, MockLLMProvider
from ontocellia.framework.mutation import MutationCandidateGenerator, MutationSelectionRuntime, write_mutation_outputs
from ontocellia.framework.selection import OrganValidationResult
from ontocellia.framework.core import TissueRuntime


@dataclass(slots=True)
class DemoResult:
    output_dir: Path
    summary_path: Path
    report_path: Path


def run_repo_repair_demo(
    output: str | Path,
    task: str = "Fix failing tests while preserving existing behavior.",
    steps: int = 4,
    seed: int = 7,
) -> DemoResult:
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    draft = TemplateInductionCompiler().compile(
        InductionRequest(task=task, domain="repo_repair", available_interfaces=["workspace", "pytest", "git"], seed=seed)
    )
    induction_dir = output_dir / "induction"
    induction_paths = draft.write(induction_dir)

    tissue = TissueRuntime.seeded(draft.genome, draft.environment, seed=seed)
    baseline_validation = [
        OrganValidationResult("pytest", False, 0.2, "repo", "3 failing regression tests", 0.1, 0.3, 1.2)
    ]
    candidate_validation = [
        OrganValidationResult("pytest", True, 0.92, "repo", "All tests passed after candidate repair.", 0.1, 0.1, 1.0)
    ]
    tissue.develop(ticks=steps, validation_results=baseline_validation)
    actions = tissue.execute(effectors=EffectorRuntime(MockLLMProvider()))
    tissue.develop(ticks=1, validation_results=baseline_validation)

    tissue_dir = output_dir / "tissue"
    tissue_dir.mkdir(parents=True, exist_ok=True)
    _write_json(tissue_dir / "tissue_summary.json", _tissue_summary(tissue, actions))
    _write_json(tissue_dir / "tissue_trace.json", tissue.trace.events)
    _write_json(tissue_dir / "action_intents.json", actions)
    _write_json(tissue_dir / "llm_trace.json", [event for event in tissue.trace.events if event["type"] == "llm_effector"])

    validation_dir = output_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    _write_json(validation_dir / "baseline_validation.json", [result.as_dict() for result in baseline_validation])
    _write_json(validation_dir / "candidate_validation.json", [result.as_dict() for result in candidate_validation])

    candidates = MutationCandidateGenerator().generate(draft.genome, baseline_validation, environment=draft.environment)
    mutation_report = MutationSelectionRuntime().select(draft.genome, candidates, baseline_validation, candidate_validation)
    mutation_paths = write_mutation_outputs(mutation_report, output_dir / "mutation")

    trace_counts = _trace_counts(tissue)
    summary = {
        "phase": "complete_repo_repair_demo",
        "task": task,
        "induction": {
            "domain": "repo_repair",
            "genome": str(induction_paths["genome"]),
            "environment": str(induction_paths["environment"]),
        },
        "tissue": {
            "ticks": tissue.tick_count,
            "population": len(tissue.cells),
            "fate_counts": tissue.fate_counts(),
            "stage_counts": tissue.stage_counts(),
            "development_stage": tissue.development_stage,
            "origin_cell_id": tissue.origin_cell_id,
            "proliferation_events": trace_counts["proliferation"],
            "actions": len(actions),
            "messages": trace_counts["messages"],
            "matrix_records": len(tissue.environment.matrix.records),
            "handoffs": trace_counts["handoffs"],
        },
        "validation": {
            "baseline_score": _score(baseline_validation),
            "candidate_score": _score(candidate_validation),
            "baseline_pass_rate": _pass_rate(baseline_validation),
            "candidate_pass_rate": _pass_rate(candidate_validation),
        },
        "mutation": {
            "decision": mutation_report.decision,
            "selected_gene": mutation_report.selected_candidate.source_gene_id if mutation_report.selected_candidate else None,
            "solidified_genome": str(mutation_paths["genome"]),
        },
    }
    summary_path = output_dir / "demo_summary.json"
    report_path = output_dir / "demo_report.md"
    _write_json(summary_path, summary)
    report_path.write_text(_demo_report(summary), encoding="utf-8")
    return DemoResult(output_dir=output_dir, summary_path=summary_path, report_path=report_path)


def _tissue_summary(tissue: TissueRuntime, actions: list[dict[str, Any]]) -> dict[str, Any]:
    trace_counts = _trace_counts(tissue)
    return {
        "objective": tissue.environment.objective,
        "ticks": tissue.tick_count,
        "population": len(tissue.cells),
        "fate_counts": tissue.fate_counts(),
        "stage_counts": tissue.stage_counts(),
        "development_stage": tissue.development_stage,
        "origin_cell_id": tissue.origin_cell_id,
        "proliferation_events": trace_counts["proliferation"],
        "niche_occupancy": tissue.niche_occupancy(),
        "organ_selection": tissue.last_organ_selection_report.as_dict() if tissue.last_organ_selection_report else {},
        "messages": trace_counts["messages"],
        "matrix_records": len(tissue.environment.matrix.records),
        "handoffs": trace_counts["handoffs"],
        "actions": actions,
    }


def _trace_counts(tissue: TissueRuntime) -> dict[str, int]:
    return {
        "messages": sum(1 for event in tissue.trace.events if event["type"] == "message_emitted"),
        "handoffs": sum(1 for event in tissue.trace.events if event["type"] == "handoff_completed"),
        "proliferation": sum(1 for event in tissue.trace.events if event["type"] == "proliferation"),
    }


def _demo_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Complete Repo Repair Tissue Demo",
            "",
            f"- Task: {summary['task']}",
            f"- Tissue actions: {summary['tissue']['actions']}",
            f"- Development stage: {summary['tissue']['development_stage']}",
            f"- Origin cell: `{summary['tissue']['origin_cell_id']}`",
            f"- Messages: {summary['tissue']['messages']}",
            f"- Matrix records: {summary['tissue']['matrix_records']}",
            f"- Baseline validation score: {summary['validation']['baseline_score']:.3f}",
            f"- Candidate validation score: {summary['validation']['candidate_score']:.3f}",
            f"- Mutation decision: `{summary['mutation']['decision']}`",
            f"- Selected gene: `{summary['mutation']['selected_gene']}`",
            "",
            "Artifacts include induced specs, tissue trace, LLM action intents, validation evidence, mutation candidates, and a solidified genome.",
            "",
        ]
    )


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _score(results: list[OrganValidationResult]) -> float:
    return sum(result.score for result in results) / max(1, len(results))


def _pass_rate(results: list[OrganValidationResult]) -> float:
    return sum(1 for result in results if result.passed) / max(1, len(results))

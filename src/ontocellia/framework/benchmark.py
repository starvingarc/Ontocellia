from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ontocellia.framework.core import TissueRuntime
from ontocellia.framework.execution import ExecutionPolicy, ExecutionRuntime
from ontocellia.framework.induction import InductionRequest, TemplateInductionCompiler
from ontocellia.framework.llm import EffectorRuntime, MockLLMProvider
from ontocellia.framework.model_config import resolve_effector_provider
from ontocellia.framework.selection import OrganValidationResult


@dataclass(slots=True)
class BenchmarkTask:
    id: str
    category: str
    prompt: str
    expected_signals: list[str] = field(default_factory=list)
    expected_fates: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BenchmarkResult:
    task: BenchmarkTask
    score: float
    metrics: dict[str, float]
    artifacts: dict[str, str]
    trace_summary: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task.id,
            "category": self.task.category,
            "score": self.score,
            "metrics": dict(self.metrics),
            "artifacts": dict(self.artifacts),
            "trace_summary": dict(self.trace_summary),
        }


@dataclass(slots=True)
class BenchmarkRunResult:
    suite: str
    output_dir: Path
    results: list[BenchmarkResult]
    summary_path: Path
    csv_path: Path
    report_path: Path


@dataclass(slots=True)
class BenchmarkSuite:
    tasks: list[BenchmarkTask]
    name: str = "custom"

    @classmethod
    def builtin(cls, name: str) -> "BenchmarkSuite":
        if name != "ontocellia_minibench_v1":
            raise ValueError(f"unknown benchmark suite: {name}")
        return cls(name=name, tasks=_minibench_tasks())

    def task(self, task_id: str) -> BenchmarkTask:
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise KeyError(task_id)


class TissueBenchmarkRunner:
    def __init__(
        self,
        *,
        suite: BenchmarkSuite,
        effector: str = "mock-llm",
        model_profile: str | None = None,
        steps: int = 6,
        seed: int = 7,
        execute_actions: bool = False,
        execution_dry_run: bool = True,
        allowed_interfaces: list[str] | None = None,
        allowed_commands: list[str] | None = None,
    ) -> None:
        self.suite = suite
        self.effector = effector
        self.model_profile = model_profile
        self.steps = steps
        self.seed = seed
        self.execute_actions = execute_actions
        self.execution_dry_run = execution_dry_run
        self.allowed_interfaces = list(allowed_interfaces or [])
        self.allowed_commands = list(allowed_commands or [])

    def run(self, output: str | Path) -> BenchmarkRunResult:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = [self._run_task(task, output_dir / "tasks" / task.id) for task in self.suite.tasks]
        summary_path = output_dir / "benchmark_summary.json"
        csv_path = output_dir / "benchmark_results.csv"
        report_path = output_dir / "benchmark_report.md"
        summary_path.write_text(json.dumps(_summary(self.suite.name, results), indent=2, sort_keys=True), encoding="utf-8")
        _write_csv(csv_path, results)
        report_path.write_text(_report(self.suite.name, results), encoding="utf-8")
        return BenchmarkRunResult(self.suite.name, output_dir, results, summary_path, csv_path, report_path)

    def _run_task(self, task: BenchmarkTask, output_dir: Path) -> BenchmarkResult:
        started = time.perf_counter()
        output_dir.mkdir(parents=True, exist_ok=True)
        draft = TemplateInductionCompiler().compile(
            InductionRequest(task=task.prompt, domain="repo_repair", available_interfaces=["workspace", "pytest", "git"], seed=self.seed)
        )
        tissue = TissueRuntime.seeded(draft.genome, draft.environment, seed=self.seed)
        validation = [OrganValidationResult("benchmark", False, 0.25, task.id, "synthetic benchmark pressure", 0.1, 0.25, 0.0)]
        if task.id in {"repo_repair_intent", "decentralized_handoff", "tool_gate_policy"}:
            tissue.environment.morphogens.emit("review_pressure", 1.0)
            tissue.environment.morphogens.emit("risk", 0.8)
        tissue.develop(ticks=self.steps, validation_results=validation)
        removed_id: int | None = None
        if task.id == "self_repair_recovery":
            removed_id = next((cell.id for cell in tissue.cells.values() if cell.fate == "repair"), None)
            if removed_id is not None:
                tissue.clear_cell(removed_id, reason="benchmark_self_repair")
                tissue.develop(ticks=4, validation_results=validation)
        provider = self._provider()
        actions = tissue.execute(effectors=EffectorRuntime(provider))
        if task.id == "decentralized_handoff":
            handoff_action = _handoff_action(tissue)
            if handoff_action is not None:
                actions.append(handoff_action)
                tissue.communicate([handoff_action])
        execution_results = None
        if self.execute_actions:
            execution_results = tissue.execute_actions(
                actions,
                ExecutionRuntime(),
                ExecutionPolicy(
                    allowed_interfaces=self.allowed_interfaces,
                    allowed_commands=self.allowed_commands,
                    dry_run=self.execution_dry_run,
                ),
            )
        tissue.develop(ticks=1, validation_results=validation)
        metrics = _metrics(task, tissue, actions, removed_id, time.perf_counter() - started)
        score = _score(metrics)
        artifacts = _write_task_artifacts(output_dir, tissue, actions, execution_results)
        return BenchmarkResult(task, score, metrics, artifacts, _trace_summary(tissue, actions))

    def _provider(self) -> Any:
        if self.effector == "mock-llm":
            return MockLLMProvider()
        return resolve_effector_provider(self.effector, model_profile=self.model_profile)


def _minibench_tasks() -> list[BenchmarkTask]:
    return [
        BenchmarkTask(
            "repo_repair_intent",
            "repo_repair",
            "Fix failing tests while preserving behavior.",
            ["test_failure", "repair_pressure", "review_pressure"],
            ["explorer", "repair", "reviewer"],
            ["repair intent", "review intent", "validation hook"],
        ),
        BenchmarkTask("tool_gate_policy", "tool_use", "Choose only allowed interfaces for a safe repo repair.", expected_fates=["repair", "reviewer"]),
        BenchmarkTask("matrix_memory", "memory", "Record failing test evidence and reuse it through matrix memory.", expected_fates=["memory", "repair"]),
        BenchmarkTask("self_repair_recovery", "regeneration", "Recover after a repair cell is removed.", expected_fates=["repair"]),
        BenchmarkTask("decentralized_handoff", "coordination", "Pass risky repair evidence to reviewer through local handoff.", expected_fates=["repair", "reviewer"]),
    ]


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
        "rationale": "Benchmark handoff from repair to reviewer.",
        "required_interfaces": ["workspace"],
        "confidence": 0.8,
        "validation_hooks": [],
        "payload": {"message": "Patch ready for review.", "handoff_to_fate": "reviewer", "matrix_tags": ["patch", "review"]},
    }


def _metrics(task: BenchmarkTask, tissue: TissueRuntime, actions: list[dict[str, Any]], removed_id: int | None, latency: float) -> dict[str, float]:
    fate_counts = tissue.fate_counts()
    events = tissue.trace.events
    expected_fate_score = _fraction_present(task.expected_fates, fate_counts)
    intent_quality = _intent_quality(actions)
    policy_compliance = _policy_compliance(actions, tissue)
    matrix_records = len(tissue.environment.matrix.records)
    handoffs = sum(1 for event in events if event["type"] == "handoff_completed")
    regeneration_events = [event for event in events if event["type"] == "regeneration" and (removed_id is None or event.get("replaced_cell_id") == removed_id)]
    lineage_ok = 0.0
    if removed_id is not None:
        lineage_ok = 1.0 if any(cell.replaces_cell_id == removed_id and cell.lineage.root_id == tissue.origin_cell_id for cell in tissue.cells.values()) else 0.0
    decentralization = min(1.0, (matrix_records > 0) * 0.35 + (handoffs > 0) * 0.35 + (len(actions) > 1) * 0.3)
    validation_score = 1.0 if any(action.get("validation_hooks") for action in actions) else 0.5
    task_success = 1.0 if expected_fate_score >= 0.75 and intent_quality > 0.0 else 0.0
    return {
        "task_success": task_success,
        "validation_score": validation_score,
        "intent_quality": intent_quality,
        "interface_policy_compliance": policy_compliance,
        "matrix_reuse_rate": min(1.0, matrix_records / max(1, len(actions))),
        "handoff_completion_rate": min(1.0, handoffs / max(1, len(actions))),
        "regeneration_recovery_ticks": float(len(regeneration_events) * 4),
        "lineage_traceability": lineage_ok if removed_id is not None else 1.0,
        "decentralization_score": decentralization,
        "cost_actions": float(len(actions)),
        "latency_seconds": round(latency, 6),
    }


def _intent_quality(actions: list[dict[str, Any]]) -> float:
    if not actions:
        return 0.0
    useful = sum(1 for action in actions if action.get("intent_type") in {"propose_patch", "inspect_context", "review_output", "record_memory"})
    return useful / len(actions)


def _policy_compliance(actions: list[dict[str, Any]], tissue: TissueRuntime) -> float:
    allowed = {interface.id for interface in tissue.environment.interfaces}
    requested = [interface for action in actions for interface in action.get("required_interfaces", [])]
    if not requested:
        return 1.0
    return sum(1 for interface in requested if interface in allowed) / len(requested)


def _fraction_present(expected: list[str], counts: dict[str, int]) -> float:
    if not expected:
        return 1.0
    return sum(1 for fate in expected if counts.get(fate, 0) > 0) / len(expected)


def _score(metrics: dict[str, float]) -> float:
    keys = [
        "task_success",
        "validation_score",
        "intent_quality",
        "interface_policy_compliance",
        "matrix_reuse_rate",
        "handoff_completion_rate",
        "lineage_traceability",
        "decentralization_score",
    ]
    return round(sum(float(metrics[key]) for key in keys) / len(keys), 6)


def _trace_summary(tissue: TissueRuntime, actions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "population": len(tissue.cells),
        "development_stage": tissue.development_stage,
        "origin_cell_id": tissue.origin_cell_id,
        "stage_counts": tissue.stage_counts(),
        "fate_counts": tissue.fate_counts(),
        "actions": len(actions),
        "messages": sum(1 for event in tissue.trace.events if event["type"] == "message_emitted"),
        "handoffs": sum(1 for event in tissue.trace.events if event["type"] == "handoff_completed"),
        "matrix_records": len(tissue.environment.matrix.records),
        "proliferation_events": sum(1 for event in tissue.trace.events if event["type"] == "proliferation"),
    }


def _write_task_artifacts(output_dir: Path, tissue: TissueRuntime, actions: list[dict[str, Any]], execution_results: list[Any] | None = None) -> dict[str, str]:
    summary = _trace_summary(tissue, actions)
    paths = {
        "summary": output_dir / "tissue_summary.json",
        "trace": output_dir / "tissue_trace.json",
        "actions": output_dir / "action_intents.json",
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    paths["trace"].write_text(json.dumps(tissue.trace.events, indent=2, sort_keys=True), encoding="utf-8")
    paths["actions"].write_text(json.dumps(actions, indent=2, sort_keys=True), encoding="utf-8")
    if execution_results is not None:
        paths["execution"] = output_dir / "execution_results.json"
        paths["execution"].write_text(json.dumps([result.as_dict() for result in execution_results], indent=2, sort_keys=True), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def _summary(suite: str, results: list[BenchmarkResult]) -> dict[str, Any]:
    return {
        "suite": suite,
        "tasks": len(results),
        "average_score": round(sum(result.score for result in results) / max(1, len(results)), 6),
        "results": [result.as_dict() for result in results],
    }


def _write_csv(path: Path, results: list[BenchmarkResult]) -> None:
    fieldnames = ["task_id", "category", "score", "task_success", "intent_quality", "interface_policy_compliance", "matrix_reuse_rate", "handoff_completion_rate", "regeneration_recovery_ticks", "lineage_traceability", "decentralization_score", "cost_actions", "latency_seconds"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {"task_id": result.task.id, "category": result.task.category, "score": result.score}
            row.update(result.metrics)
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _report(suite: str, results: list[BenchmarkResult]) -> str:
    lines = [
        f"# Tissue Benchmark Report: {suite}",
        "",
        "| Task | Category | Score | Success | Intents | Matrix | Handoff |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            "| {task} | {category} | {score:.3f} | {success:.3f} | {intent:.3f} | {matrix:.3f} | {handoff:.3f} |".format(
                task=result.task.id,
                category=result.task.category,
                score=result.score,
                success=result.metrics["task_success"],
                intent=result.metrics["intent_quality"],
                matrix=result.metrics["matrix_reuse_rate"],
                handoff=result.metrics["handoff_completion_rate"],
            )
        )
    return "\n".join(lines) + "\n"

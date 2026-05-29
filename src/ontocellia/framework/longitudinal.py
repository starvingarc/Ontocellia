from __future__ import annotations

import csv
import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ontocellia.framework.core import TaskMicroenvironment, TissueRuntime
from ontocellia.framework.genome import AgentGenome
from ontocellia.framework.induction import InductionRequest, TemplateInductionCompiler
from ontocellia.framework.llm import EffectorRuntime, MockLLMProvider
from ontocellia.framework.model_config import resolve_effector_provider
from ontocellia.framework.selection import OrganValidationResult
from ontocellia.framework.solidification import SelectionSolidificationPolicy, SelectionSolidificationReport, SelectionSolidificationRuntime
from ontocellia.framework.structure_search import StructureSearchReport, StructureSearchRunner


@dataclass(slots=True)
class ReplayTask:
    id: str
    prompt: str
    domain: str = "repo_repair"
    validation_score: float = 0.25
    tags: list[str] | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tags"] = list(self.tags or [])
        return data


@dataclass(slots=True)
class ReplayTrialResult:
    task: ReplayTask
    condition: str
    score: float
    metrics: dict[str, Any]
    artifacts: dict[str, str]
    selected_variant: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.as_dict(),
            "condition": self.condition,
            "score": self.score,
            "metrics": dict(self.metrics),
            "artifacts": dict(self.artifacts),
            "selected_variant": self.selected_variant,
        }


@dataclass(slots=True)
class LongitudinalReplayReport:
    output_dir: Path
    trials: list[ReplayTrialResult]
    summary_path: Path
    csv_path: Path
    report_path: Path
    solidification_path: Path


class LongitudinalReplayRunner:
    """Run repeated task families through controlled tissue baselines."""

    def __init__(
        self,
        *,
        tasks: list[ReplayTask] | list[str] | None = None,
        domain: str = "repo_repair",
        effector: str = "mock-llm",
        model_profile: str | None = None,
        steps: int = 6,
        seed: int = 7,
        solidification_policy: SelectionSolidificationPolicy | None = None,
    ) -> None:
        self.tasks = _normalize_tasks(tasks or builtin_replay_tasks(), domain)
        self.domain = domain
        self.effector = effector
        self.model_profile = model_profile
        self.steps = steps
        self.seed = seed
        self.solidification_policy = solidification_policy or SelectionSolidificationPolicy(min_repetitions=2)

    def run(self, output: str | Path) -> LongitudinalReplayReport:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        provider = self._provider()
        compiler = TemplateInductionCompiler()
        trials: list[ReplayTrialResult] = []
        structure_reports: list[StructureSearchReport] = []
        adaptive_genome: AgentGenome | None = None
        final_solidification: SelectionSolidificationReport | None = None

        for index, task in enumerate(self.tasks):
            task_dir = output_dir / "tasks" / task.id
            draft = compiler.compile(InductionRequest(task=task.prompt, domain=task.domain, seed=self.seed + index))
            validation = [_validation_for_task(task)]
            base_genome = draft.genome
            trials.append(_direct_agent_trial(task, validation, task_dir / "direct_agent"))
            trials.append(
                _run_tissue_trial(
                    task,
                    "single_cell",
                    base_genome,
                    _clone_environment(draft.environment),
                    provider,
                    validation,
                    self.steps,
                    self.seed + index,
                    task_dir / "single_cell",
                    single_cell=True,
                )
            )
            trials.append(
                _run_tissue_trial(
                    task,
                    "fixed_tissue",
                    base_genome,
                    _clone_environment(draft.environment),
                    provider,
                    validation,
                    self.steps,
                    self.seed + index,
                    task_dir / "fixed_tissue",
                    single_cell=False,
                )
            )
            adaptive_search = StructureSearchRunner(
                task=task.prompt,
                domain=task.domain,
                effector=self.effector,
                model_profile=self.model_profile,
                steps=self.steps,
                seed=self.seed + index,
            ).run(
                task_dir / "adaptive_tissue",
                genome=adaptive_genome or base_genome,
                environment=_clone_environment(draft.environment),
                validation_results=validation,
            )
            structure_reports.append(adaptive_search)
            selected_trial = next(trial for trial in adaptive_search.trials if trial.variant.name == adaptive_search.selected_variant)
            trials.append(
                ReplayTrialResult(
                    task=task,
                    condition="adaptive_tissue",
                    score=selected_trial.score,
                    metrics={**selected_trial.metrics, "selected_variant": adaptive_search.selected_variant},
                    artifacts={
                        "structure_search_summary": str(adaptive_search.summary_path),
                        "structure_search_report": str(adaptive_search.report_path),
                        **selected_trial.artifacts,
                    },
                    selected_variant=adaptive_search.selected_variant,
                )
            )
            final_solidification = SelectionSolidificationRuntime().solidify(
                structure_reports,
                genome=adaptive_genome or base_genome,
                policy=self.solidification_policy,
            )
            adaptive_genome = final_solidification.solidified_genome

        if final_solidification is None:
            draft = compiler.compile(InductionRequest(task="", domain=self.domain, seed=self.seed))
            final_solidification = SelectionSolidificationRuntime().solidify([], genome=draft.genome, policy=self.solidification_policy)
        solidification_paths = final_solidification.write(output_dir / "solidification")
        summary_path = output_dir / "longitudinal_replay_summary.json"
        csv_path = output_dir / "longitudinal_trials.csv"
        report_path = output_dir / "longitudinal_replay_report.md"
        summary = _summary(self.tasks, trials, final_solidification)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        _write_csv(csv_path, trials)
        report_path.write_text(_markdown_report(summary, trials), encoding="utf-8")
        return LongitudinalReplayReport(
            output_dir=output_dir,
            trials=trials,
            summary_path=summary_path,
            csv_path=csv_path,
            report_path=report_path,
            solidification_path=Path(solidification_paths["report"]),
        )

    def _provider(self) -> Any:
        if self.effector == "mock-llm":
            return MockLLMProvider()
        return resolve_effector_provider(self.effector, model_profile=self.model_profile)


def builtin_replay_tasks() -> list[ReplayTask]:
    return [
        ReplayTask(
            "repo_repair_failing_tests",
            "Fix failing tests while preserving existing behavior.",
            "repo_repair",
            0.25,
            ["pytest", "repair"],
        ),
        ReplayTask(
            "repo_repair_regression",
            "Fix a regression without broad rewrites or suppressing tests.",
            "repo_repair",
            0.3,
            ["regression", "review"],
        ),
        ReplayTask(
            "repo_repair_risk_review",
            "Repair risky behavior and produce review evidence before finalizing.",
            "repo_repair",
            0.35,
            ["risk", "review", "repair"],
        ),
    ]


def _normalize_tasks(tasks: list[ReplayTask] | list[str], domain: str) -> list[ReplayTask]:
    normalized: list[ReplayTask] = []
    for index, item in enumerate(tasks):
        if isinstance(item, ReplayTask):
            normalized.append(item)
        else:
            normalized.append(ReplayTask(f"task_{index + 1}", str(item), domain, 0.25, ["custom"]))
    return normalized


def _validation_for_task(task: ReplayTask) -> OrganValidationResult:
    return OrganValidationResult(
        name=f"validation:{task.id}",
        passed=False,
        score=float(task.validation_score),
        target=task.prompt,
        evidence=f"Replay validation pressure for {task.id}.",
        cost=0.1,
        risk=0.25,
        latency=0.0,
    )


def _direct_agent_trial(task: ReplayTask, validation: list[OrganValidationResult], output_dir: Path) -> ReplayTrialResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_score = _average_validation(validation)
    cost = 1.0
    score = round(validation_score * 0.55 + (1.0 / (1.0 + cost * 0.1)) * 0.25, 6)
    metrics = {
        "validation_score": validation_score,
        "fate_match": 0.0,
        "matrix_reuse_rate": 0.0,
        "handoff_completion_rate": 0.0,
        "cell_count": 0,
        "action_count": 1,
        "cost": cost,
        "cost_efficiency": round(1.0 / (1.0 + cost * 0.1), 6),
        "structure_score": score,
    }
    summary_path = output_dir / "baseline_summary.json"
    summary_path.write_text(json.dumps({"task": task.as_dict(), "metrics": metrics}, indent=2, sort_keys=True), encoding="utf-8")
    return ReplayTrialResult(task, "direct_agent", score, metrics, {"summary": str(summary_path)})


def _run_tissue_trial(
    task: ReplayTask,
    condition: str,
    genome: AgentGenome,
    environment: TaskMicroenvironment,
    provider: Any,
    validation: list[OrganValidationResult],
    steps: int,
    seed: int,
    output_dir: Path,
    *,
    single_cell: bool,
) -> ReplayTrialResult:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    tissue = TissueRuntime.seeded(genome, environment, seed=seed)
    if single_cell:
        tissue.target_population = 1
        tissue.min_population_before_differentiation = 1
    tissue.develop(ticks=steps, validation_results=validation)
    actions = tissue.execute(effectors=EffectorRuntime(provider))
    tissue.develop(ticks=1, validation_results=validation)
    metrics = _tissue_metrics(tissue, actions, validation, time.perf_counter() - started)
    artifacts = _write_tissue_artifacts(output_dir, tissue, actions, metrics)
    return ReplayTrialResult(task, condition, float(metrics["structure_score"]), metrics, artifacts)


def _tissue_metrics(tissue: TissueRuntime, actions: list[dict[str, Any]], validation: list[OrganValidationResult], latency: float) -> dict[str, Any]:
    fate_counts = tissue.fate_counts()
    action_count = max(1, len(actions))
    validation_score = _average_validation(validation)
    matrix_records = len(tissue.environment.matrix.records)
    handoffs = sum(1 for event in tissue.trace.events if event["type"] == "handoff_completed")
    fate_match = sum(1 for fate in ["explorer", "repair", "reviewer", "memory"] if fate_counts.get(fate, 0) > 0) / 4.0
    matrix_reuse = min(1.0, matrix_records / action_count)
    handoff_rate = min(1.0, handoffs / action_count)
    cost = float(len(actions) + len(tissue.cells) * 0.1)
    cost_efficiency = 1.0 / (1.0 + cost * 0.1)
    score = round(validation_score * 0.32 + fate_match * 0.24 + matrix_reuse * 0.14 + handoff_rate * 0.1 + cost_efficiency * 0.2, 6)
    resource_report = tissue.last_resource_report.as_dict() if tissue.last_resource_report is not None else {}
    return {
        "validation_score": validation_score,
        "fate_match": round(fate_match, 6),
        "matrix_reuse_rate": round(matrix_reuse, 6),
        "handoff_completion_rate": round(handoff_rate, 6),
        "cell_count": len(tissue.cells),
        "action_count": len(actions),
        "stage_counts": tissue.stage_counts(),
        "fate_distribution": fate_counts,
        "cost": round(cost, 6),
        "cost_efficiency": round(cost_efficiency, 6),
        "latency_seconds": round(latency, 6),
        "resource_efficiency": float(resource_report.get("resource_efficiency", 1.0)),
        "structure_score": score,
    }


def _write_tissue_artifacts(output_dir: Path, tissue: TissueRuntime, actions: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, str]:
    paths = {
        "summary": output_dir / "tissue_summary.json",
        "trace": output_dir / "tissue_trace.json",
        "actions": output_dir / "action_intents.json",
    }
    paths["summary"].write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    paths["trace"].write_text(json.dumps(tissue.trace.events, indent=2, sort_keys=True), encoding="utf-8")
    paths["actions"].write_text(json.dumps(actions, indent=2, sort_keys=True), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def _summary(tasks: list[ReplayTask], trials: list[ReplayTrialResult], solidification: SelectionSolidificationReport) -> dict[str, Any]:
    conditions = ["direct_agent", "single_cell", "fixed_tissue", "adaptive_tissue"]
    averages = {
        condition: _average([trial.score for trial in trials if trial.condition == condition])
        for condition in conditions
    }
    adaptive_variants = [trial.selected_variant for trial in trials if trial.condition == "adaptive_tissue" and trial.selected_variant]
    return {
        "task_count": len(tasks),
        "tasks": [task.as_dict() for task in tasks],
        "conditions": conditions,
        "trials": [trial.as_dict() for trial in trials],
        "comparison": {
            "conditions_compared": len(conditions),
            "average_scores": averages,
            "adaptive_average_score": averages["adaptive_tissue"],
            "fixed_average_score": averages["fixed_tissue"],
            "adaptive_gain_over_fixed": round(averages["adaptive_tissue"] - averages["fixed_tissue"], 6),
            "adaptive_gain_over_direct": round(averages["adaptive_tissue"] - averages["direct_agent"], 6),
        },
        "memory": {
            "replay_sessions": len(adaptive_variants),
            "selected_variants": adaptive_variants,
            "solidified_tendencies": len(solidification.tendencies),
        },
        "solidification": solidification.as_dict(),
    }


def _write_csv(path: Path, trials: list[ReplayTrialResult]) -> None:
    fieldnames = [
        "task_id",
        "condition",
        "score",
        "selected_variant",
        "validation_score",
        "fate_match",
        "matrix_reuse_rate",
        "handoff_completion_rate",
        "cell_count",
        "action_count",
        "cost",
        "cost_efficiency",
        "structure_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trial in trials:
            row = {
                "task_id": trial.task.id,
                "condition": trial.condition,
                "score": trial.score,
                "selected_variant": trial.selected_variant,
                **trial.metrics,
            }
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _markdown_report(summary: dict[str, Any], trials: list[ReplayTrialResult]) -> str:
    lines = [
        "# Longitudinal Replay Report",
        "",
        f"- Tasks: {summary['task_count']}",
        f"- Adaptive gain over fixed tissue: `{summary['comparison']['adaptive_gain_over_fixed']:.3f}`",
        f"- Solidification decision: `{summary['solidification']['decision']}`",
        "",
        "| Task | Condition | Score | Selected variant | Cost |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for trial in trials:
        lines.append(
            f"| {trial.task.id} | {trial.condition} | {trial.score:.3f} | {trial.selected_variant or '-'} | {float(trial.metrics.get('cost', 0.0)):.3f} |"
        )
    return "\n".join(lines) + "\n"


def _average(values: list[float]) -> float:
    return round(sum(values) / max(1, len(values)), 6)


def _average_validation(results: list[OrganValidationResult]) -> float:
    return _average([float(result.score) for result in results])


def _clone_environment(environment: TaskMicroenvironment) -> TaskMicroenvironment:
    return deepcopy(environment)

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from ontocellia.framework.attribution import ContributionAttributionRuntime
from ontocellia.framework.cell import CellPosition
from ontocellia.framework.communication import MatrixRecord
from ontocellia.framework.core import ExtracellularInterface, MorphogenField, Niche, TaskMicroenvironment, TissueRuntime
from ontocellia.framework.induction import InductionRequest, TemplateInductionCompiler
from ontocellia.framework.llm import ActionIntent, CellPrompt, EffectorRuntime, LLMResponse, MockLLMProvider, OpenAICompatibleProvider
from ontocellia.framework.model_config import load_secret_env, load_user_config, resolve_effector_provider
from ontocellia.framework.selection import OrganValidationResult
from ontocellia.framework.structure_search import StructureVariant, builtin_structure_variants


BFCL_DATASET = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_BASE_URL = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main"
BFCL_DEFAULT_CATEGORY = "BFCL_v3_simple"
SWE_BENCH_LITE_DATASET = "princeton-nlp/SWE-bench_Lite"
TERMINAL_BENCH_REPO = "https://github.com/laude-institute/terminal-bench.git"
TAU_BENCH_REPO = "https://github.com/sierra-research/tau-bench.git"


@dataclass(slots=True)
class AdaptiveBenchmarkTask:
    id: str
    source_benchmark: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AdaptiveBenchmarkReport:
    benchmark: str
    mode: str
    output_dir: Path
    summary_path: Path
    structure_path: Path
    report_path: Path


@dataclass(slots=True)
class OfficialScorerPlan:
    benchmark: str
    scorer: str
    status: str
    command: str
    cwd: str
    reason: str
    predictions_path: str | None = None
    requirements: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OfficialBenchmarkRunResult:
    benchmark: str
    output_dir: Path
    summary_path: Path
    report_path: Path
    official_results_path: Path


class OfficialBenchmarkAdapter(Protocol):
    name: str

    @staticmethod
    def for_benchmark(name: str) -> "GenericOfficialBenchmarkAdapter":
        return GenericOfficialBenchmarkAdapter(name)


@dataclass(slots=True)
class GenericOfficialBenchmarkAdapter:
    name: str

    def fixture_task(self) -> AdaptiveBenchmarkTask:
        fixtures = {
            "tau-bench": AdaptiveBenchmarkTask(
                "tau-fixture",
                "tau-bench",
                "Help a user update an order while following the support policy and using available tools.",
                {"policy": "Only update orders after lookup confirms eligibility.", "tools": [{"name": "lookup_order"}, {"name": "update_order"}], "user": "customer"},
            ),
            "terminal-bench": AdaptiveBenchmarkTask(
                "terminal-fixture",
                "terminal-bench",
                "Inspect the workspace and run the provided check command after reasoning about the task.",
                {"check_command": "python -m pytest -q", "files": ["README.md"], "tools": [{"name": "shell.run"}]},
            ),
            "multiagentbench": AdaptiveBenchmarkTask(
                "multiagent-fixture",
                "multiagentbench",
                "Coordinate specialist agents to gather evidence, propose a plan, review risk, and produce a final decision.",
                {"roles": ["scout", "planner", "builder", "reviewer"], "coordination": "handoff-required"},
            ),
            "swe-bench-lite": AdaptiveBenchmarkTask(
                "swe-fixture",
                "swe-bench-lite",
                "Fix a repository issue described by failing tests while preserving existing behavior.",
                {"issue": "Failing regression test in repository.", "tests": ["python -m pytest -q"], "repo": "fixture"},
            ),
        }
        if self.name not in fixtures:
            raise ValueError(f"unsupported adaptive benchmark: {self.name}")
        return fixtures[self.name]

    def load_tasks(
        self,
        *,
        limit: int | None = None,
        task_id: str | None = None,
        dry_run: bool = False,
        split: str = "test",
        source_dir: str | Path | None = None,
        tau_domain: str = "airline",
    ) -> list[AdaptiveBenchmarkTask]:
        if dry_run:
            task = self.fixture_task()
            if task_id is not None:
                task = AdaptiveBenchmarkTask(task_id, self.name, task.prompt, dict(task.metadata))
            tasks = [task]
            return tasks[:limit] if limit is not None else tasks
        if self.name == "swe-bench-lite":
            return _load_swe_bench_tasks(limit=limit, task_id=task_id, split=split)
        if self.name == "terminal-bench":
            return _load_terminal_bench_tasks(limit=limit, task_id=task_id, source_dir=source_dir)
        if self.name == "tau-bench":
            return _load_tau_bench_tasks(limit=limit, task_id=task_id, source_dir=source_dir, tau_domain=tau_domain)
        if self.name == "multiagentbench":
            raise ValueError("multiagentbench official source loading is not implemented; use --dry-run for fixture mode")
        task = self.fixture_task()
        tasks = [task]
        return tasks[:limit] if limit is not None else tasks

    def to_induction_request(self, task: AdaptiveBenchmarkTask) -> InductionRequest:
        environment = self.to_microenvironment(task)
        interfaces = [interface.id for interface in environment.interfaces]
        return InductionRequest(
            task=task.prompt,
            domain=_induction_domain_for_task(task),
            available_interfaces=interfaces,
            constraints={"source_benchmark": task.source_benchmark, **task.metadata},
        )

    def to_microenvironment(self, task: AdaptiveBenchmarkTask) -> TaskMicroenvironment:
        metadata = dict(task.metadata)
        tools = ["workspace", *[str(tool.get("name", tool)) for tool in metadata.get("tools", [])]]
        if metadata.get("check_command"):
            tools.append("shell.run")
        if metadata.get("tests"):
            tools.append("pytest.run")
        interfaces = [
            ExtracellularInterface(tool, "membrane_channel", ["explorer", "planner", "builder", "reviewer", "repair", "memory"])
            for tool in sorted(set(tools or ["workspace"]))
        ]
        records = _matrix_records_for_task(task)
        morphogens = _morphogens_for_task(task)
        niches = [
            Niche("exploration-front", "explorer", CellPosition("exploration-front", task.source_benchmark, ["planner-niche"], (0.0, 0.0, 0.0))),
            Niche("planner-niche", "planner", CellPosition("planner-niche", task.source_benchmark, ["exploration-front", "builder-niche"], (1.0, 0.0, 0.0))),
            Niche("builder-niche", "builder", CellPosition("builder-niche", task.source_benchmark, ["planner-niche", "review-boundary"], (2.0, 0.0, 0.0))),
            Niche("review-boundary", "reviewer", CellPosition("review-boundary", task.source_benchmark, ["builder-niche"], (3.0, 0.0, 0.0))),
            Niche("memory-niche", "memory", CellPosition("memory-niche", task.source_benchmark, ["planner-niche"], (1.0, 1.0, 0.0))),
        ]
        if _is_repo_repair_like_task(task):
            niches.append(Niche("repair-niche", "repair", CellPosition("repair-niche", task.source_benchmark, ["builder-niche"], (2.0, 1.0, 0.0))))
        environment = TaskMicroenvironment(task.prompt, morphogens, niches, interfaces)
        for record in records:
            environment.matrix.deposit(record)
        return environment


class AdaptiveTissueBenchmarkRunner:
    def __init__(
        self,
        *,
        model_profile: str | None = None,
        dry_run: bool = True,
        steps: int = 6,
        seed: int = 7,
        structure_search: bool = False,
        run_official_scorer: bool = False,
        official_scorer_command: str | None = None,
        official_scorer_timeout: float = 300.0,
        official_scorer_cwd: str | Path | None = None,
        split: str = "test",
        source_dir: str | Path | None = None,
        tau_domain: str = "airline",
        official_agent_adapter: str = "auto",
        bridge_url: str | None = None,
        max_agent_steps: int = 8,
        with_attribution: bool = False,
    ) -> None:
        self.model_profile = model_profile
        self.dry_run = dry_run
        self.steps = steps
        self.seed = seed
        self.structure_search = structure_search
        self.run_official_scorer = run_official_scorer
        self.official_scorer_command = official_scorer_command
        self.official_scorer_timeout = official_scorer_timeout
        self.official_scorer_cwd = Path(official_scorer_cwd) if official_scorer_cwd is not None else None
        self.split = split
        self.source_dir = Path(source_dir) if source_dir is not None else None
        self.tau_domain = tau_domain
        self.official_agent_adapter = official_agent_adapter
        self.bridge_url = bridge_url
        self.max_agent_steps = max_agent_steps
        self.with_attribution = with_attribution

    def run_tasks(self, tasks: list[AdaptiveBenchmarkTask], output: str | Path) -> AdaptiveBenchmarkReport:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        traces_dir = output_dir / "tissue_traces"
        traces_dir.mkdir(exist_ok=True)
        provider = MockLLMProvider() if self.dry_run else resolve_effector_provider("llm", model_profile=self.model_profile)
        scoring = _scoring_status(tasks, self.run_official_scorer, self.official_scorer_command)
        task_reports = []
        task_artifacts: list[tuple[Path, AdaptiveBenchmarkTask, dict[str, Any]]] = []
        predictions = []
        for task in tasks:
            task_dir = traces_dir / task.id
            task_dir.mkdir(parents=True, exist_ok=True)
            if self.structure_search:
                selected, variants = _run_structure_search_task(task, provider, self.steps, self.seed, scoring, task_dir, self.with_attribution)
                metrics = selected["metrics"]
                actions = selected["actions"]
                task_report = {"task_id": task.id, "source_benchmark": task.source_benchmark, "metrics": metrics, "variants": variants}
            else:
                selected = _run_single_task(task, provider, self.steps, self.seed, scoring, task_dir)
                metrics = selected["metrics"]
                actions = selected["actions"]
                if self.with_attribution:
                    attribution = ContributionAttributionRuntime().analyze(
                        trace=selected["trace"],
                        actions=actions,
                    )
                    attribution.write(task_dir / "attribution")
                    metrics = {**metrics, "attribution": attribution.summary}
                    selected["metrics"] = metrics
                task_report = {"task_id": task.id, "source_benchmark": task.source_benchmark, "metrics": metrics}
            task_artifacts.append((task_dir, task, metrics))
            (task_dir / "tissue_trace.json").write_text(json.dumps(selected["trace"], indent=2, sort_keys=True), encoding="utf-8")
            _write_jsonl(task_dir / "action_intents.jsonl", actions)
            task_reports.append(task_report)
            predictions.append({"id": task.id, "source_benchmark": task.source_benchmark, "actions": actions, "metrics": metrics})
        _write_jsonl(output_dir / "official_tasks.jsonl", [task.as_dict() for task in tasks])
        _write_json(output_dir / "official_task_manifest.json", {"tasks": [task.as_dict() for task in tasks]})
        _write_jsonl(output_dir / "ontocellia_predictions.jsonl", predictions)
        scoring = _finalize_scoring_status(
            output_dir=output_dir,
            tasks=tasks,
            run_official_scorer=self.run_official_scorer,
            official_scorer_command=self.official_scorer_command,
            official_scorer_timeout=self.official_scorer_timeout,
            official_scorer_cwd=self.official_scorer_cwd,
            predictions=predictions,
            model_profile=self.model_profile,
            split=self.split,
            source_dir=self.source_dir,
            tau_domain=self.tau_domain,
            official_agent_adapter=self.official_agent_adapter,
            bridge_url=self.bridge_url,
            max_agent_steps=self.max_agent_steps,
        )
        _apply_scoring_to_reports(task_reports, predictions, scoring)
        for task_dir, task, metrics in task_artifacts:
            _write_json(task_dir / "tissue_summary.json", {"task": task.as_dict(), "metrics": metrics})
        summary = {
            "benchmark": tasks[0].source_benchmark if tasks else "adaptive",
            "mode": "adaptive-tissue",
            "tasks": len(tasks),
            "dry_run": self.dry_run,
            "structure_search": self.structure_search,
            "official_score_status": scoring["official_score_status"],
            "average_final_task_success": _average([item["metrics"]["final_task_success"] for item in task_reports]),
            "average_structure_efficiency": _average([item["metrics"]["structure_efficiency"] for item in task_reports]),
        }
        structure = {"mode": "adaptive-tissue", "tasks": task_reports}
        _write_json(output_dir / "run_config.json", {"mode": "adaptive-tissue", "model_profile": self.model_profile, "dry_run": self.dry_run, "structure_search": self.structure_search})
        _write_json(output_dir / "scoring_status.json", scoring)
        _write_json(output_dir / "ontocellia_summary.json", summary)
        _write_json(output_dir / "structure_report.json", structure)
        _write_json(output_dir / "official_results.json", {"mode": "adaptive-tissue", "scoring_status": scoring, "tasks": task_reports})
        _write_jsonl(output_dir / "ontocellia_predictions.jsonl", predictions)
        (output_dir / "adaptation_report.md").write_text(_adaptive_report(summary, task_reports), encoding="utf-8")
        return AdaptiveBenchmarkReport(
            benchmark=summary["benchmark"],
            mode="adaptive-tissue",
            output_dir=output_dir,
            summary_path=output_dir / "ontocellia_summary.json",
            structure_path=output_dir / "structure_report.json",
            report_path=output_dir / "adaptation_report.md",
        )


class OfficialBenchmarkRunner:
    def prepare(self, benchmark: str, output: Path | str, dry_run: bool = False) -> dict[str, Any]:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        plan = {"benchmark": benchmark, "dry_run": dry_run, "mode": "adaptive-tissue" if benchmark != "bfcl" else "provider-baseline"}
        if benchmark == "bfcl":
            plan.update({"dataset": BFCL_DATASET, "category": BFCL_DEFAULT_CATEGORY})
        (output_dir / "prepare_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
        return plan

    def run(
        self,
        *,
        benchmark: str,
        output: Path | str,
        model_profile: str | None,
        limit: int | None = None,
        task_id: str | None = None,
        full: bool = False,
        dry_run: bool = False,
        category: str = BFCL_DEFAULT_CATEGORY,
        mode: str | None = None,
        split: str = "test",
        source_dir: str | Path | None = None,
        tau_domain: str = "airline",
        structure_search: bool = False,
        run_official_scorer: bool = False,
        official_scorer_command: str | None = None,
        official_scorer_timeout: float = 300.0,
        official_scorer_cwd: str | Path | None = None,
        official_agent_adapter: str = "auto",
        bridge_url: str | None = None,
        max_agent_steps: int = 8,
        with_attribution: bool = False,
    ) -> OfficialBenchmarkRunResult | AdaptiveBenchmarkReport:
        selected_mode = mode or ("provider-baseline" if benchmark == "bfcl" else "adaptive-tissue")
        if not full and limit is None and task_id is None:
            raise ValueError("official benchmark runs require --limit, --task-id, or --full")
        if selected_mode == "adaptive-tissue":
            adapter = OfficialBenchmarkAdapter.for_benchmark(benchmark)
            tasks = adapter.load_tasks(limit=limit, task_id=task_id, dry_run=dry_run, split=split, source_dir=source_dir, tau_domain=tau_domain)
            return AdaptiveTissueBenchmarkRunner(
                model_profile=model_profile,
                dry_run=dry_run,
                structure_search=structure_search,
                run_official_scorer=run_official_scorer,
                official_scorer_command=official_scorer_command,
                official_scorer_timeout=official_scorer_timeout,
                official_scorer_cwd=official_scorer_cwd,
                split=split,
                source_dir=source_dir,
                tau_domain=tau_domain,
                official_agent_adapter=official_agent_adapter,
                bridge_url=bridge_url,
                max_agent_steps=max_agent_steps,
                with_attribution=with_attribution,
            ).run_tasks(tasks, output)
        if benchmark != "bfcl":
            raise ValueError(f"{benchmark} only supports adaptive-tissue mode")
        return self._run_bfcl_provider_baseline(output, model_profile, limit, task_id, full, dry_run, category)

    def run_bfcl_records(
        self,
        *,
        tasks: list[dict[str, Any]],
        answers: dict[str, Any],
        output: Path | str,
        model_profile: str | None,
        dry_run: bool,
        category: str = BFCL_DEFAULT_CATEGORY,
        official_data_downloaded: bool = False,
        mock_predictions: list[dict[str, Any]] | None = None,
        mode: str = "provider-baseline",
    ) -> OfficialBenchmarkRunResult:
        if mode != "provider-baseline":
            raise ValueError("BFCL is a provider-baseline benchmark; use adaptive runner for tissue evaluation")
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        provider = _resolve_official_provider(model_profile) if not dry_run and mock_predictions is None else MockLLMProvider()
        mock_by_id = {item["id"]: item for item in mock_predictions or []}
        predictions = []
        for task in tasks:
            task_id = str(task.get("id", ""))
            if task_id in mock_by_id:
                prediction = mock_by_id[task_id]
            elif dry_run:
                prediction = {"id": task_id, "function_calls": [], "raw_response": "", "model": "dry-run", "source": "provider_baseline"}
            else:
                prediction = _call_bfcl_provider(provider, task)
            prediction.setdefault("source", "provider_baseline")
            predictions.append(prediction)
        scores = _score_bfcl(predictions, answers)
        summary = {
            "benchmark": "bfcl",
            "dataset": BFCL_DATASET,
            "category": category,
            "model_profile": model_profile or load_user_config().default_model,
            "dry_run": dry_run,
            "mode": "provider-baseline",
            "tasks": len(tasks),
            "official_data_downloaded": official_data_downloaded,
            **scores,
            "latency_seconds": round(time.perf_counter() - started, 6),
        }
        _write_json(output_dir / "run_config.json", {"benchmark": "bfcl", "mode": "provider-baseline", "category": category})
        _write_jsonl(output_dir / "official_tasks.jsonl", tasks)
        _write_json(output_dir / "official_task_manifest.json", {"tasks": tasks})
        _write_jsonl(output_dir / "official_answers.jsonl", [{"id": task.get("id", ""), "ground_truth": answers.get(str(task.get("id", "")), [])} for task in tasks])
        _write_jsonl(output_dir / "ontocellia_predictions.jsonl", predictions)
        scoring = {"official_score_status": "run", "scorer": "bfcl_exact_function_call", **scores}
        _write_json(output_dir / "scoring_status.json", scoring)
        _write_json(output_dir / "official_results.json", {"benchmark": "bfcl", "scores": scores, "scoring_status": scoring, "predictions": predictions})
        _write_json(output_dir / "ontocellia_summary.json", summary)
        (output_dir / "report.md").write_text(_bfcl_report(summary), encoding="utf-8")
        (output_dir / "official_command.sh").write_text(_bfcl_command_text(category), encoding="utf-8")
        (output_dir / "official_stdout.log").write_text("", encoding="utf-8")
        (output_dir / "official_stderr.log").write_text("", encoding="utf-8")
        return OfficialBenchmarkRunResult("bfcl", output_dir, output_dir / "ontocellia_summary.json", output_dir / "report.md", output_dir / "official_results.json")

    def _run_bfcl_provider_baseline(
        self,
        output: Path | str,
        model_profile: str | None,
        limit: int | None,
        task_id: str | None,
        full: bool,
        dry_run: bool,
        category: str,
    ) -> OfficialBenchmarkRunResult:
        if dry_run:
            tasks = [_dry_run_bfcl_task(task_id or "simple_0")]
            answers = {tasks[0]["id"]: [{"calculate_triangle_area": {"base": [10], "height": [5]}}]}
        else:
            tasks = _download_bfcl_tasks(category)
            answers = _download_bfcl_answers(category)
        if task_id is not None:
            tasks = [task for task in tasks if task.get("id") == task_id]
        if limit is not None:
            tasks = tasks[:limit]
        return self.run_bfcl_records(
            tasks=tasks,
            answers=answers,
            output=output,
            model_profile=model_profile,
            dry_run=dry_run,
            category=category,
            official_data_downloaded=not dry_run,
            mode="provider-baseline",
        )


def load_swe_bench_task_from_row(row: dict[str, Any], *, benchmark_id: str = "swe-bench-lite") -> AdaptiveBenchmarkTask:
    prompt = str(row.get("problem_statement", ""))
    tests = list(row.get("FAIL_TO_PASS") or [])
    return AdaptiveBenchmarkTask(
        id=str(row.get("instance_id", "")),
        source_benchmark=benchmark_id,
        prompt=prompt,
        metadata={
            "repo": row.get("repo"),
            "base_commit": row.get("base_commit"),
            "issue": prompt,
            "tests": tests,
            "pass_to_pass": list(row.get("PASS_TO_PASS") or []),
            "official_dataset": SWE_BENCH_LITE_DATASET,
        },
    )


def load_terminal_bench_task_from_yaml(path: str | Path, *, task_id: str | None = None) -> AdaptiveBenchmarkTask:
    task_path = Path(path)
    data = yaml.safe_load(task_path.read_text(encoding="utf-8")) or {}
    task_name = task_id or task_path.parent.name
    parser = str(data.get("parser_name", ""))
    return AdaptiveBenchmarkTask(
        id=task_name,
        source_benchmark="terminal-bench",
        prompt=str(data.get("instruction", "")),
        metadata={
            "difficulty": data.get("difficulty"),
            "category": data.get("category"),
            "tags": list(data.get("tags") or []),
            "check_command": f"official terminal-bench parser: {parser}" if parser else "official terminal-bench parser",
            "official_repo": "laude-institute/terminal-bench",
        },
    )


def load_tau_bench_tasks_from_text(
    text: str,
    *,
    benchmark_id: str = "tau-bench",
    tau_domain: str = "airline",
    limit: int | None = None,
    task_id: str | None = None,
) -> list[AdaptiveBenchmarkTask]:
    tree = ast.parse(text)
    tasks: list[AdaptiveBenchmarkTask] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "TASKS" for target in node.targets):
            continue
        for index, element in enumerate(getattr(node.value, "elts", [])):
            current_id = f"{tau_domain}-test-{index}"
            if task_id is not None and task_id != current_id:
                continue
            task = _tau_task_from_ast_call(element, current_id, benchmark_id, tau_domain)
            if task is not None:
                tasks.append(task)
            if limit is not None and len(tasks) >= limit:
                return tasks
        break
    return tasks


def _tau_task_from_ast_call(element: ast.AST, task_id: str, benchmark_id: str, tau_domain: str) -> AdaptiveBenchmarkTask | None:
    if not isinstance(element, ast.Call):
        return None
    values: dict[str, Any] = {}
    action_names: list[str] = []
    for keyword in element.keywords:
        if keyword.arg in {"user_id", "instruction", "outputs"}:
            values[keyword.arg] = ast.literal_eval(keyword.value)
        if keyword.arg == "actions" and isinstance(keyword.value, ast.List):
            for action_call in keyword.value.elts:
                if not isinstance(action_call, ast.Call):
                    continue
                for action_keyword in action_call.keywords:
                    if action_keyword.arg == "name":
                        action_names.append(str(ast.literal_eval(action_keyword.value)))
    instruction = str(values.get("instruction", ""))
    return AdaptiveBenchmarkTask(
        id=task_id,
        source_benchmark=benchmark_id,
        prompt=instruction,
        metadata={
            "user_id": values.get("user_id"),
            "expected_action_names": action_names,
            "outputs": list(values.get("outputs") or []),
            "policy": f"official tau-bench {tau_domain} environment task",
            "tools": [{"name": name} for name in sorted(set(action_names))],
            "official_repo": "sierra-research/tau-bench",
            "tau_domain": tau_domain,
        },
    )


def _load_swe_bench_tasks(*, limit: int | None, task_id: str | None, split: str) -> list[AdaptiveBenchmarkTask]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("SWE-bench official loading requires installing the benchmark extra: pip install 'ontocellia[benchmark]'") from error
    dataset = load_dataset(SWE_BENCH_LITE_DATASET, split=split)
    tasks = [load_swe_bench_task_from_row(dict(row)) for row in dataset]
    if task_id is not None:
        tasks = [task for task in tasks if task.id == task_id]
    return tasks[:limit] if limit is not None else tasks


def _load_terminal_bench_tasks(*, limit: int | None, task_id: str | None, source_dir: str | Path | None) -> list[AdaptiveBenchmarkTask]:
    root = _ensure_source_dir("terminal-bench", source_dir)
    task_paths = sorted((root / "original-tasks").glob("*/task.yaml"))
    tasks = [load_terminal_bench_task_from_yaml(path) for path in task_paths]
    if task_id is not None:
        tasks = [task for task in tasks if task.id == task_id]
    return tasks[:limit] if limit is not None else tasks


def _load_tau_bench_tasks(*, limit: int | None, task_id: str | None, source_dir: str | Path | None, tau_domain: str) -> list[AdaptiveBenchmarkTask]:
    root = _ensure_source_dir("tau-bench", source_dir)
    path = root / "tau_bench" / "envs" / tau_domain / "tasks_test.py"
    if not path.exists():
        raise FileNotFoundError(f"tau-bench task file not found: {path}")
    return load_tau_bench_tasks_from_text(path.read_text(encoding="utf-8"), tau_domain=tau_domain, limit=limit, task_id=task_id)


def _ensure_source_dir(benchmark: str, source_dir: str | Path | None) -> Path:
    root = Path(source_dir) if source_dir is not None else Path("artifacts") / "official_sources" / benchmark
    if root.exists():
        return root
    repo = {"terminal-bench": TERMINAL_BENCH_REPO, "tau-bench": TAU_BENCH_REPO}.get(benchmark)
    if repo is None:
        return root
    root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", repo, str(root)], check=True)
    return root


def _matrix_records_for_task(task: AdaptiveBenchmarkTask) -> list[MatrixRecord]:
    records = []
    metadata = task.metadata
    for key in ("policy", "check_command", "issue"):
        if key in metadata:
            records.append(MatrixRecord(f"{task.id}-{key}", 0, "policy" if key == "policy" else "observation", str(metadata[key]), [key], CellPosition("benchmark-input", task.source_benchmark), 0.8))
    if metadata.get("tests"):
        records.append(MatrixRecord(f"{task.id}-tests", 0, "validation", json.dumps(metadata["tests"]), ["tests", "check_command"], CellPosition("benchmark-input", task.source_benchmark), 0.8))
    return records


def _morphogens_for_task(task: AdaptiveBenchmarkTask) -> MorphogenField:
    signals = {"ambiguity": 0.7, "coordination_pressure": 0.75, "verification_pressure": 0.65, "memory_pressure": 0.45, "implementation_pressure": 0.6}
    if task.source_benchmark == "terminal-bench":
        signals.update({"execution_pressure": 0.8, "verification_pressure": 0.85})
    if _is_repo_repair_like_task(task):
        signals.update({"test_failure": 0.9, "repair_pressure": 0.85})
    if task.source_benchmark == "tau-bench":
        signals.update({"risk": 0.7, "planning_pressure": 0.75})
    return MorphogenField(signals)


def _run_single_task(task: AdaptiveBenchmarkTask, provider: Any, steps: int, seed: int, scoring: dict[str, Any], task_dir: Path) -> dict[str, Any]:
    tissue = _seed_official_tissue(task, seed)
    validation = [OrganValidationResult("official_task", False, 0.25, task.id, "official benchmark pressure", 0.1, 0.2, 0.0)]
    tissue.develop(ticks=steps, validation_results=validation)
    actions = _safe_execute_tissue(tissue, provider)
    tissue.develop(ticks=1, validation_results=validation)
    metrics = _structure_metrics(task, tissue, actions, scoring, selected_variant="baseline")
    return {"variant": "baseline", "tissue": tissue, "actions": actions, "metrics": metrics, "trace": tissue.trace.events}


def _run_structure_search_task(
    task: AdaptiveBenchmarkTask,
    provider: Any,
    steps: int,
    seed: int,
    scoring: dict[str, Any],
    task_dir: Path,
    with_attribution: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    variants = []
    for variant in builtin_structure_variants():
        variant_dir = task_dir / "variants" / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        result = _run_variant_task(task, provider, steps, seed, scoring, variant)
        _write_json(variant_dir / "tissue_summary.json", {"task": task.as_dict(), "metrics": result["metrics"]})
        (variant_dir / "tissue_trace.json").write_text(json.dumps(result["trace"], indent=2, sort_keys=True), encoding="utf-8")
        _write_jsonl(variant_dir / "action_intents.jsonl", result["actions"])
        if with_attribution:
            attribution = ContributionAttributionRuntime().analyze(
                trace=result["trace"],
                actions=result["actions"],
            )
            attribution.write(variant_dir / "attribution")
            result["metrics"] = {**result["metrics"], "attribution": attribution.summary}
        variants.append({key: result[key] for key in ("variant", "metrics", "actions", "trace")})
    selected = sorted(variants, key=lambda item: (-_structure_selection_score(item["metrics"]), str(item["variant"])))[0]
    selected["metrics"] = {**selected["metrics"], "selected_variant": selected["variant"]}
    return selected, [{"variant": item["variant"], "metrics": item["metrics"]} for item in variants]


def _run_variant_task(
    task: AdaptiveBenchmarkTask,
    provider: Any,
    steps: int,
    seed: int,
    scoring: dict[str, Any],
    variant: StructureVariant,
) -> dict[str, Any]:
    tissue = _seed_official_tissue(task, seed)
    _apply_structure_variant(tissue.environment, variant)
    if variant.min_population_before_differentiation is not None:
        tissue.min_population_before_differentiation = variant.min_population_before_differentiation
    if variant.target_population_delta:
        tissue.target_population = max(1, (tissue.target_population or len(tissue.cells)) + variant.target_population_delta)
    validation = [OrganValidationResult("official_task", False, 0.25, task.id, "official benchmark pressure", 0.1, 0.2, 0.0)]
    tissue.develop(ticks=steps, validation_results=validation)
    actions = _safe_execute_tissue(tissue, provider)
    tissue.develop(ticks=1, validation_results=validation)
    metrics = _structure_metrics(task, tissue, actions, scoring, selected_variant=variant.name)
    return {"variant": variant.name, "tissue": tissue, "actions": actions, "metrics": metrics, "trace": tissue.trace.events}


def _seed_official_tissue(task: AdaptiveBenchmarkTask, seed: int) -> TissueRuntime:
    adapter = GenericOfficialBenchmarkAdapter(task.source_benchmark)
    draft = TemplateInductionCompiler().compile(adapter.to_induction_request(task))
    official_environment = adapter.to_microenvironment(task)
    _merge_official_environment(draft.environment, official_environment)
    return TissueRuntime.seeded(draft.genome, draft.environment, seed=seed)


def _safe_execute_tissue(tissue: TissueRuntime, provider: Any) -> list[dict[str, Any]]:
    return tissue.execute(effectors=EffectorRuntime(_BenchmarkProviderGuard(provider, tissue)))


_PROVIDER_ERROR_TYPES = (
    TimeoutError,
    ConnectionError,
    RuntimeError,
    urllib.error.URLError,
    json.JSONDecodeError,
)


@dataclass(slots=True)
class _BenchmarkProviderGuard:
    provider: Any
    tissue: TissueRuntime

    @property
    def name(self) -> str:
        return getattr(self.provider, "name", self.provider.__class__.__name__)

    def complete(self, prompt: CellPrompt) -> LLMResponse:
        try:
            return self.provider.complete(prompt)
        except _PROVIDER_ERROR_TYPES as error:
            cell_id = int(prompt.context.get("cell_id", 0))
            fate = str(prompt.context.get("fate", "unknown"))
            self.tissue.trace.record(
                "official_benchmark_provider_error",
                provider=self.name,
                cell_id=cell_id,
                error_type=error.__class__.__name__,
                error=str(error),
            )
            intent = ActionIntent(
                cell_id=cell_id,
                fate=fate,
                expressed_gene_ids=[str(item) for item in prompt.context.get("expressed_gene_ids", [])],
                intent_type="record_memory",
                target=str(prompt.context.get("position", {}).get("node_id", "benchmark")),
                rationale=f"Provider {self.name} failed during official benchmark execution: {error.__class__.__name__}.",
                required_interfaces=[],
                confidence=0.0,
                validation_hooks=[str(item) for item in prompt.context.get("validation_hooks", [])],
                payload={"provider_error": str(error), "matrix_tags": ["provider_error", fate]},
            )
            return LLMResponse(
                content=intent.rationale,
                parsed_intent=intent,
                raw={"provider_error": error.__class__.__name__},
                model=self.name,
                usage={},
            )


def _merge_official_environment(target: TaskMicroenvironment, source: TaskMicroenvironment) -> None:
    for name, value in source.morphogens.signals.items():
        target.morphogens.emit(name, value)
    existing_interfaces = {interface.id for interface in target.interfaces}
    for interface in source.interfaces:
        if interface.id not in existing_interfaces:
            target.interfaces.append(interface)
            existing_interfaces.add(interface.id)
    existing_records = {record.id for record in target.matrix.records}
    for record in source.matrix.records:
        if record.id not in existing_records:
            target.matrix.deposit(record)
            existing_records.add(record.id)


def _apply_structure_variant(environment: TaskMicroenvironment, variant: StructureVariant) -> None:
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


def _structure_selection_score(metrics: dict[str, Any]) -> float:
    return float(metrics["structure_efficiency"]) + float(metrics.get("repair_presence", 0.0)) * 0.1 + float(metrics.get("expected_fate_coverage", 0.0)) * 0.1


def _structure_metrics(task: AdaptiveBenchmarkTask, tissue: TissueRuntime, actions: list[dict[str, Any]], scoring: dict[str, Any], *, selected_variant: str) -> dict[str, Any]:
    events = tissue.trace.events
    execution_events = [event for event in events if event["type"].startswith("execution_")]
    validation_cycles = [event for event in events if event["type"] == "organ_selection"]
    provider_calls = [event for event in events if event["type"] == "llm_effector"]
    provider_errors = [event for event in events if event["type"] == "official_benchmark_provider_error"]
    handoffs = [event for event in events if event["type"] == "handoff_completed"]
    matrix_records = len(tissue.environment.matrix.records)
    action_count = max(1, len(actions))
    fate_counts = tissue.fate_counts()
    expected_fate_coverage = _expected_fate_coverage(task, fate_counts)
    repair_presence = 1.0 if fate_counts.get("repair", 0) > 0 else 0.0
    resource_report = tissue.last_resource_report.as_dict() if tissue.last_resource_report is not None else {}
    resource_efficiency = float(resource_report.get("resource_efficiency", 1.0))
    return {
        "final_task_success": 0.0,
        "fate_distribution": fate_counts,
        "proliferation_events": sum(1 for event in events if event["type"] == "proliferation"),
        "handoff_completion_rate": len(handoffs) / action_count,
        "matrix_reuse_rate": min(1.0, matrix_records / action_count),
        "execution_success_rate": _execution_success_rate(execution_events),
        "validation_feedback_cycles": len(validation_cycles),
        "provider_call_count": len(provider_calls),
        "provider_call_errors": len(provider_errors),
        "structure_efficiency": round(
            min(
                1.0,
                (len(fate_counts) / 5.0) * 0.22
                + expected_fate_coverage * 0.23
                + min(1.0, matrix_records / action_count) * 0.22
                + (len(handoffs) > 0) * 0.13
                + repair_presence * 0.1
                + resource_efficiency * 0.1,
            ),
            6,
        ),
        "regeneration_events": sum(1 for event in events if event["type"] == "regeneration"),
        "selected_variant": selected_variant,
        "repair_presence": repair_presence,
        "expected_fate_coverage": expected_fate_coverage,
        "average_cell_energy": float(resource_report.get("average_cell_energy", 1.0)),
        "population_pressure": float(resource_report.get("population_pressure", 0.0)),
        "resource_efficiency": resource_efficiency,
        "official_score_status": scoring["official_score_status"],
        "scorer_pass_rate": float(scoring.get("scorer_pass_rate", 0.0)),
        "execution_attempted": bool(execution_events),
        "blocked_tool_requests": sum(1 for event in events if event["type"] in {"tool_invocation_skipped", "execution_skipped"}),
    }


def _execution_success_rate(events: list[dict[str, Any]]) -> float:
    completed = [event for event in events if event["type"] == "execution_completed"]
    if not completed:
        return 0.0
    return sum(1 for event in completed if event.get("passed")) / len(completed)


def _expected_fate_coverage(task: AdaptiveBenchmarkTask, fate_counts: dict[str, int]) -> float:
    expected = _expected_fates_for_task(task)
    if not expected:
        return 1.0
    return round(sum(1 for fate in expected if fate_counts.get(fate, 0) > 0) / len(expected), 6)


def _expected_fates_for_task(task: AdaptiveBenchmarkTask) -> list[str]:
    if _is_repo_repair_like_task(task):
        return ["explorer", "repair", "reviewer", "memory"]
    if task.source_benchmark == "tau-bench":
        return ["explorer", "builder", "reviewer", "memory"]
    return ["explorer", "builder", "reviewer", "memory"]


def _induction_domain_for_task(task: AdaptiveBenchmarkTask) -> str:
    if _is_repo_repair_like_task(task):
        return "repo_repair"
    return "generic"


def _is_repo_repair_like_task(task: AdaptiveBenchmarkTask) -> bool:
    metadata = task.metadata
    tags = {str(tag).lower() for tag in metadata.get("tags", [])}
    category = str(metadata.get("category", "")).lower()
    prompt = task.prompt.lower()
    if task.source_benchmark == "swe-bench-lite":
        return True
    if task.source_benchmark != "terminal-bench":
        return False
    repair_categories = {"debugging", "software-engineering"}
    repair_tags = {
        "coding",
        "debugging",
        "legacy-modernization",
        "optimization",
        "software-engineering",
        "swe-bench",
    }
    repair_tokens = {
        "bug",
        "compatibility",
        "debug",
        "failing",
        "fix",
        "legacy",
        "modernize",
        "pytest",
        "regression",
    }
    repair_phrases = {
        "failing test",
        "failing tests",
        "fix test",
        "fix tests",
        "test failure",
        "test failures",
    }
    prompt_tokens = set(re.findall(r"[a-z0-9_+-]+", prompt))
    return (
        category in repair_categories
        or bool(tags & repair_tags)
        or bool(prompt_tokens & repair_tokens)
        or any(phrase in prompt for phrase in repair_phrases)
    )


def _scoring_status(tasks: list[AdaptiveBenchmarkTask], run_official_scorer: bool, official_scorer_command: str | None = None) -> dict[str, Any]:
    benchmark = tasks[0].source_benchmark if tasks else "adaptive"
    if run_official_scorer and official_scorer_command:
        return {
            "benchmark": benchmark,
            "official_score_status": "pending",
            "scorer": "external_command",
            "scorer_pass_rate": 0.0,
            "reason": "Official scorer command requested and will run after predictions are written.",
        }
    if run_official_scorer:
        return {
            "benchmark": benchmark,
            "official_score_status": "unsupported",
            "scorer": None,
            "scorer_pass_rate": 0.0,
            "reason": "Official scorer execution is not wired for this adaptive tissue run yet.",
        }
    return {
        "benchmark": benchmark,
        "official_score_status": "not_run",
        "scorer": None,
        "scorer_pass_rate": 0.0,
        "reason": "Official scorer was not requested; this run reports Ontocellia adaptive tissue metrics only.",
    }


def _finalize_scoring_status(
    *,
    output_dir: Path,
    tasks: list[AdaptiveBenchmarkTask],
    run_official_scorer: bool,
    official_scorer_command: str | None,
    official_scorer_timeout: float,
    official_scorer_cwd: Path | None,
    predictions: list[dict[str, Any]],
    model_profile: str | None,
    split: str,
    source_dir: Path | None,
    tau_domain: str,
    official_agent_adapter: str,
    bridge_url: str | None,
    max_agent_steps: int,
) -> dict[str, Any]:
    if not run_official_scorer:
        return _scoring_status(tasks, run_official_scorer, official_scorer_command)
    if official_scorer_command:
        return _run_official_scorer_command(
            benchmark=tasks[0].source_benchmark if tasks else "adaptive",
            command=official_scorer_command,
            output_dir=output_dir,
            timeout=official_scorer_timeout,
            cwd=official_scorer_cwd or Path.cwd(),
        )
    return _run_official_scorer_adapter(
        benchmark=tasks[0].source_benchmark if tasks else "adaptive",
        output_dir=output_dir,
        tasks=tasks,
        predictions=predictions,
        model_profile=model_profile,
        split=split,
        source_dir=source_dir,
        tau_domain=tau_domain,
        official_agent_adapter=official_agent_adapter,
        bridge_url=bridge_url,
        max_agent_steps=max_agent_steps,
        timeout=official_scorer_timeout,
    )


def _run_official_scorer_adapter(
    *,
    benchmark: str,
    output_dir: Path,
    tasks: list[AdaptiveBenchmarkTask],
    predictions: list[dict[str, Any]],
    model_profile: str | None,
    split: str,
    source_dir: Path | None,
    tau_domain: str,
    official_agent_adapter: str,
    bridge_url: str | None,
    max_agent_steps: int,
    timeout: float,
) -> dict[str, Any]:
    plan = _official_scorer_plan(
        benchmark=benchmark,
        output_dir=output_dir,
        tasks=tasks,
        predictions=predictions,
        model_profile=model_profile,
        split=split,
        source_dir=source_dir,
        tau_domain=tau_domain,
        official_agent_adapter=official_agent_adapter,
        bridge_url=bridge_url,
        max_agent_steps=max_agent_steps,
    )
    _write_json(output_dir / "official_scorer_plan.json", plan.as_dict())
    if plan.status != "ready":
        return {
            "benchmark": benchmark,
            "official_score_status": plan.status,
            "scorer": plan.scorer,
            "command": plan.command,
            "cwd": plan.cwd,
            "scorer_pass_rate": 0.0,
            "reason": plan.reason,
            "requirements": list(plan.requirements),
            "predictions_path": plan.predictions_path,
            "environment": dict(plan.environment),
        }
    return _run_official_scorer_command(
        benchmark=benchmark,
        command=plan.command,
        output_dir=output_dir,
        timeout=timeout,
        cwd=Path(plan.cwd),
        environment=plan.environment,
    )


def _official_scorer_plan(
    *,
    benchmark: str,
    output_dir: Path,
    tasks: list[AdaptiveBenchmarkTask],
    predictions: list[dict[str, Any]],
    model_profile: str | None,
    split: str,
    source_dir: Path | None,
    tau_domain: str,
    official_agent_adapter: str,
    bridge_url: str | None,
    max_agent_steps: int,
) -> OfficialScorerPlan:
    if benchmark == "swe-bench-lite":
        predictions_path = _write_swe_bench_predictions(output_dir, predictions, model_profile)
        command = " ".join(
            [
                shlex.quote(sys.executable),
                "-m",
                "swebench.harness.run_evaluation",
                "--dataset_name",
                shlex.quote(SWE_BENCH_LITE_DATASET),
                "--split",
                shlex.quote(split),
                "--predictions_path",
                shlex.quote(str(predictions_path)),
                "--max_workers",
                "1",
                "--run_id",
                shlex.quote("ontocellia"),
            ]
        )
        if not _module_available("swebench"):
            return OfficialScorerPlan(
                benchmark,
                "swebench.harness.run_evaluation",
                "adapter_unavailable",
                command,
                str(Path.cwd()),
                "SWE-bench scorer adapter is configured, but the swebench package is not installed in this environment.",
                predictions_path=str(predictions_path),
                requirements=["pip install swe-bench", "Docker-compatible SWE-bench harness runtime"],
            )
        return OfficialScorerPlan(
            benchmark,
            "swebench.harness.run_evaluation",
            "ready",
            command,
            str(Path.cwd()),
            "SWE-bench official harness command is ready.",
            predictions_path=str(predictions_path),
        )
    if benchmark == "terminal-bench":
        root = source_dir or Path("artifacts") / "official_sources" / "terminal-bench"
        command = " ".join(
            [
                "tb",
                "run",
                "--dataset-path",
                shlex.quote(str(root / "original-tasks")),
                "--agent-import-path",
                "ontocellia.official_terminal_agent:OntocelliaTerminalAgent",
                "--agent-kwargs",
                shlex.quote(json.dumps({"max_steps": max_agent_steps, "use_mock": model_profile in {None, "mock"}})),
                "--output-path",
                shlex.quote(str(output_dir / "official_scorer")),
            ]
        )
        if _module_available("terminal_bench"):
            return OfficialScorerPlan(
                benchmark,
                "terminal-bench",
                "ready",
                command,
                str(root),
                "Terminal-Bench official custom agent command is ready.",
                requirements=["terminal-bench CLI"],
            )
        return OfficialScorerPlan(
            benchmark,
            "terminal-bench",
            "adapter_required",
            command,
            str(root),
            "Terminal-Bench custom agent adapter is required before the official scorer can drive Ontocellia end-to-end.",
            requirements=["terminal-bench CLI", "Ontocellia Terminal-Bench custom agent adapter"],
        )
    if benchmark == "tau-bench":
        root = source_dir or Path("artifacts") / "official_sources" / "tau-bench"
        env = {}
        if bridge_url:
            env = {"OPENAI_BASE_URL": bridge_url, "OPENAI_API_KEY": "ontocellia-local-bridge"}
        command = " ".join(
            [
                shlex.quote(sys.executable),
                "run.py",
                "--env",
                shlex.quote(tau_domain),
                "--agent-strategy",
                "tool-calling",
                "--model",
                "ontocellia-bridge",
                "--model-provider",
                "openai",
                "--user-model",
                shlex.quote(model_profile or "gpt-4o"),
                "--user-model-provider",
                "openai",
                "--user-strategy",
                "llm",
                "--max-concurrency",
                "1",
            ]
        )
        if bridge_url:
            return OfficialScorerPlan(
                benchmark,
                "tau-bench",
                "ready",
                command,
                str(root),
                "tau-bench command is ready to use the Ontocellia OpenAI-compatible bridge.",
                requirements=["tau-bench official repo", "running Ontocellia bridge server"],
                environment=env,
            )
        return OfficialScorerPlan(
            benchmark,
            "tau-bench",
            "bridge_required",
            command,
            str(root),
            "tau-bench requires an OpenAI-compatible bridge URL before the official scorer can drive Ontocellia.",
            requirements=["tau-bench official repo", "python -m ontocellia server --host 127.0.0.1 --port 8765"],
        )
    return OfficialScorerPlan(
        benchmark,
        "unsupported",
        "unsupported",
        "",
        str(Path.cwd()),
        f"No official scorer adapter is available for benchmark: {benchmark}.",
    )


def _write_swe_bench_predictions(output_dir: Path, predictions: list[dict[str, Any]], model_profile: str | None) -> Path:
    path = output_dir / "official_scorer_predictions.jsonl"
    rows = [
        {
            "instance_id": str(item.get("id", "")),
            "model_name_or_path": f"ontocellia/{model_profile or 'mock'}",
            "model_patch": _patch_from_actions(list(item.get("actions") or [])),
        }
        for item in predictions
    ]
    _write_jsonl(path, rows)
    return path


def _patch_from_actions(actions: list[Any]) -> str:
    for action in actions:
        data = action.as_dict() if hasattr(action, "as_dict") else dict(action)
        payload = data.get("payload", {})
        if isinstance(payload, dict) and payload.get("patch"):
            return str(payload["patch"])
    return ""


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _run_official_scorer_command(
    *,
    benchmark: str,
    command: str,
    output_dir: Path,
    timeout: float,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "official_command.sh").write_text(command + "\n", encoding="utf-8")
    try:
        completed = subprocess.run(
            shlex.split(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            env={**os.environ, **(environment or {})},
        )
        stdout = completed.stdout
        stderr = completed.stderr
        exit_code = completed.returncode
        status = "run" if exit_code == 0 else "run_failed"
        reason = "Official scorer command completed." if exit_code == 0 else "Official scorer command returned a non-zero exit code."
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout if isinstance(error.stdout, str) else (error.stdout or b"").decode("utf-8", errors="replace")
        stderr = error.stderr if isinstance(error.stderr, str) else (error.stderr or b"").decode("utf-8", errors="replace")
        exit_code = None
        status = "run_failed"
        reason = f"Official scorer command timed out after {timeout} seconds."
    (output_dir / "official_stdout.log").write_text(stdout, encoding="utf-8")
    (output_dir / "official_stderr.log").write_text(stderr, encoding="utf-8")
    return {
        "benchmark": benchmark,
        "official_score_status": status,
        "scorer": "external_command",
        "command": command,
        "cwd": str(cwd),
        "environment": dict(environment or {}),
        "exit_code": exit_code,
        "scorer_pass_rate": 1.0 if status == "run" else 0.0,
        "latency_seconds": round(time.perf_counter() - started, 6),
        "reason": reason,
    }


def _apply_scoring_to_reports(task_reports: list[dict[str, Any]], predictions: list[dict[str, Any]], scoring: dict[str, Any]) -> None:
    for item in [*task_reports, *predictions]:
        metrics = item.get("metrics")
        if isinstance(metrics, dict):
            metrics["official_score_status"] = scoring["official_score_status"]
            metrics["scorer_pass_rate"] = float(scoring.get("scorer_pass_rate", 0.0))
            if scoring["official_score_status"] == "run":
                metrics["final_task_success"] = float(scoring.get("scorer_pass_rate", 0.0))


def _average(values: list[float]) -> float:
    return round(sum(values) / max(1, len(values)), 6)


def _adaptive_report(summary: dict[str, Any], task_reports: list[dict[str, Any]]) -> str:
    lines = [
        f"# Adaptive Tissue Benchmark: {summary['benchmark']}",
        "",
        f"- Mode: `{summary['mode']}`",
        f"- Tasks: {summary['tasks']}",
        f"- Average structure efficiency: {summary['average_structure_efficiency']:.3f}",
        "",
        "| Task | Source | Structure Efficiency | Provider Calls | Matrix Reuse |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for item in task_reports:
        metrics = item["metrics"]
        lines.append(f"| {item['task_id']} | {item['source_benchmark']} | {metrics['structure_efficiency']:.3f} | {metrics['provider_call_count']} | {metrics['matrix_reuse_rate']:.3f} |")
    return "\n".join(lines) + "\n"


def _resolve_official_provider(model_profile: str | None) -> Any:
    config = load_user_config()
    profile = config.profile(model_profile)
    if profile.provider == "mock-llm":
        return MockLLMProvider()
    env = dict(load_secret_env())
    return OpenAICompatibleProvider.from_name(
        profile.provider,
        model=profile.model or None,
        base_url=profile.base_url or None,
        env={**env, **__import__("os").environ},
    )


def _call_bfcl_provider(provider: Any, task: dict[str, Any]) -> dict[str, Any]:
    prompt = CellPrompt(
        system="Return only JSON with shape {\"function_calls\":[{\"name\":\"...\",\"arguments\":{...}}]}.",
        context=build_bfcl_prompt_context(task),
        output_schema={"type": "BFCLFunctionCalls"},
    )
    if isinstance(provider, MockLLMProvider):
        return {"id": task.get("id", ""), "function_calls": [], "raw_response": "", "model": provider.name, "source": "provider_baseline"}
    payload = {
        "model": provider.model,
        "messages": [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": json.dumps(prompt.context, ensure_ascii=True)},
        ],
        "temperature": 0.0,
        "stream": False,
    }
    raw = (provider.transport or _post_json)(provider._chat_url(), {"Authorization": f"Bearer {provider.api_key}", "Content-Type": "application/json"}, payload, provider.timeout)
    content = str(raw["choices"][0]["message"].get("content", ""))
    return {
        "id": task.get("id", ""),
        "function_calls": _extract_function_calls(content),
        "raw_response": content,
        "model": str(raw.get("model", provider.model)),
        "usage": raw.get("usage", {}),
        "source": "provider_baseline",
    }


def build_bfcl_prompt_context(task: dict[str, Any]) -> dict[str, Any]:
    candidate_functions = []
    for function in task.get("function", []):
        parameters = function.get("parameters", {}) if isinstance(function, dict) else {}
        properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        candidate = dict(function)
        candidate["argument_keys"] = sorted(str(key) for key in properties)
        candidate["required_argument_keys"] = [str(key) for key in parameters.get("required", [])] if isinstance(parameters, dict) else []
        candidate_functions.append(candidate)
    return {"question": task.get("question", []), "functions": task.get("function", []), "candidate_functions": candidate_functions}


def _download_bfcl_tasks(category: str) -> list[dict[str, Any]]:
    return parse_bfcl_task_text(_read_url(f"{BFCL_BASE_URL}/{category}.json"))


def parse_bfcl_task_text(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []
    data = json.loads(stripped) if stripped.startswith("[") else [json.loads(line) for line in stripped.splitlines() if line.strip()]
    if not isinstance(data, list):
        raise ValueError("unexpected BFCL task shape")
    return [dict(item) for item in data]


def _download_bfcl_answers(category: str) -> dict[str, Any]:
    answers: dict[str, Any] = {}
    for line in _read_url(f"{BFCL_BASE_URL}/possible_answer/{category}.json").splitlines():
        if line.strip():
            item = json.loads(line)
            answers[str(item["id"])] = item.get("ground_truth", [])
    return answers


def _read_url(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_function_calls(content: str) -> list[dict[str, Any]]:
    text = content.strip()
    if "```" in text:
        parts = text.split("```")
        text = parts[1].removeprefix("json").strip() if len(parts) > 1 else text
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    calls = data.get("function_calls", data.get("tool_calls", [])) if isinstance(data, dict) else data
    normalized = []
    for call in calls if isinstance(calls, list) else []:
        if isinstance(call, dict):
            arguments = call.get("arguments") or call.get("parameters") or {}
            normalized.append({"name": str(call.get("name") or call.get("function") or call.get("tool_name")), "arguments": arguments if isinstance(arguments, dict) else {}})
    return normalized


def _score_bfcl(predictions: list[dict[str, Any]], answers: dict[str, Any]) -> dict[str, Any]:
    function_hits = 0
    argument_hits = 0
    scored = 0
    for prediction in predictions:
        expected = answers.get(str(prediction.get("id", "")), [])
        if not expected:
            continue
        scored += 1
        expected_name = next(iter(expected[0].keys()))
        expected_args = set(next(iter(expected[0].values())).keys())
        predicted = (prediction.get("function_calls") or [{}])[0]
        if predicted.get("name") == expected_name:
            function_hits += 1
        if expected_args <= set((predicted.get("arguments") or {}).keys()):
            argument_hits += 1
    denominator = max(1, scored)
    return {"tasks": len(predictions), "scored_tasks": scored, "function_name_accuracy": round(function_hits / denominator, 6), "argument_key_accuracy": round(argument_hits / denominator, 6)}


def _dry_run_bfcl_task(task_id: str) -> dict[str, Any]:
    return {
        "id": task_id,
        "question": [[{"role": "user", "content": "Find the area of a triangle with a base of 10 units and height of 5 units."}]],
        "function": [{"name": "calculate_triangle_area", "parameters": {"properties": {"base": {}, "height": {}, "unit": {}}}}],
    }


def _bfcl_command_text(category: str) -> str:
    return f"# BFCL provider-baseline data source: https://huggingface.co/datasets/{BFCL_DATASET}\n# Category: {category}\n"


def _bfcl_report(summary: dict[str, Any]) -> str:
    return (
        "# Provider Baseline Report: BFCL\n\n"
        f"- Mode: `{summary['mode']}`\n"
        f"- Tasks: {summary['tasks']}\n"
        f"- Function name accuracy: {summary['function_name_accuracy']:.3f}\n"
        f"- Argument key accuracy: {summary['argument_key_accuracy']:.3f}\n"
    )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

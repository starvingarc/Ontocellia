from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ontocellia.framework.cell import CellPosition
from ontocellia.framework.communication import MatrixRecord
from ontocellia.framework.core import ExtracellularInterface, MorphogenField, Niche, TaskMicroenvironment, TissueRuntime
from ontocellia.framework.induction import InductionRequest, TemplateInductionCompiler
from ontocellia.framework.llm import CellPrompt, EffectorRuntime, MockLLMProvider, OpenAICompatibleProvider
from ontocellia.framework.model_config import load_secret_env, load_user_config, resolve_effector_provider
from ontocellia.framework.selection import OrganValidationResult


BFCL_DATASET = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_BASE_URL = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main"
BFCL_DEFAULT_CATEGORY = "BFCL_v3_simple"


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

    def load_tasks(self, *, limit: int | None = None, task_id: str | None = None, dry_run: bool = False) -> list[AdaptiveBenchmarkTask]:
        task = self.fixture_task()
        if task_id is not None:
            task = AdaptiveBenchmarkTask(task_id, self.name, task.prompt, dict(task.metadata))
        tasks = [task]
        return tasks[:limit] if limit is not None else tasks

    def to_induction_request(self, task: AdaptiveBenchmarkTask) -> InductionRequest:
        environment = self.to_microenvironment(task)
        interfaces = [interface.id for interface in environment.interfaces]
        return InductionRequest(
            task=task.prompt,
            domain="generic",
            available_interfaces=interfaces,
            constraints={"source_benchmark": task.source_benchmark, **task.metadata},
        )

    def to_microenvironment(self, task: AdaptiveBenchmarkTask) -> TaskMicroenvironment:
        metadata = dict(task.metadata)
        tools = [str(tool.get("name", tool)) for tool in metadata.get("tools", [])]
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
        if task.source_benchmark == "swe-bench-lite":
            niches.append(Niche("repair-niche", "repair", CellPosition("repair-niche", task.source_benchmark, ["builder-niche"], (2.0, 1.0, 0.0))))
        environment = TaskMicroenvironment(task.prompt, morphogens, niches, interfaces)
        for record in records:
            environment.matrix.deposit(record)
        return environment


class AdaptiveTissueBenchmarkRunner:
    def __init__(self, *, model_profile: str | None = None, dry_run: bool = True, steps: int = 6, seed: int = 7) -> None:
        self.model_profile = model_profile
        self.dry_run = dry_run
        self.steps = steps
        self.seed = seed

    def run_tasks(self, tasks: list[AdaptiveBenchmarkTask], output: str | Path) -> AdaptiveBenchmarkReport:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        traces_dir = output_dir / "tissue_traces"
        traces_dir.mkdir(exist_ok=True)
        provider = MockLLMProvider() if self.dry_run else resolve_effector_provider("llm", model_profile=self.model_profile)
        task_reports = []
        predictions = []
        for task in tasks:
            adapter = GenericOfficialBenchmarkAdapter(task.source_benchmark)
            draft = TemplateInductionCompiler().compile(adapter.to_induction_request(task))
            tissue = TissueRuntime.seeded(draft.genome, draft.environment, seed=self.seed)
            validation = [OrganValidationResult("official_task", False, 0.25, task.id, "official benchmark pressure", 0.1, 0.2, 0.0)]
            tissue.develop(ticks=self.steps, validation_results=validation)
            actions = tissue.execute(effectors=EffectorRuntime(provider))
            tissue.develop(ticks=1, validation_results=validation)
            task_dir = traces_dir / task.id
            task_dir.mkdir(parents=True, exist_ok=True)
            metrics = _structure_metrics(tissue, actions)
            _write_json(task_dir / "tissue_summary.json", {"task": task.as_dict(), "metrics": metrics})
            (task_dir / "tissue_trace.json").write_text(json.dumps(tissue.trace.events, indent=2, sort_keys=True), encoding="utf-8")
            _write_jsonl(task_dir / "action_intents.jsonl", actions)
            task_reports.append({"task_id": task.id, "source_benchmark": task.source_benchmark, "metrics": metrics})
            predictions.append({"id": task.id, "source_benchmark": task.source_benchmark, "actions": actions, "metrics": metrics})
        summary = {
            "benchmark": tasks[0].source_benchmark if tasks else "adaptive",
            "mode": "adaptive-tissue",
            "tasks": len(tasks),
            "dry_run": self.dry_run,
            "average_structure_efficiency": _average([item["metrics"]["structure_efficiency"] for item in task_reports]),
        }
        structure = {"mode": "adaptive-tissue", "tasks": task_reports}
        _write_json(output_dir / "run_config.json", {"mode": "adaptive-tissue", "model_profile": self.model_profile, "dry_run": self.dry_run})
        _write_json(output_dir / "ontocellia_summary.json", summary)
        _write_json(output_dir / "structure_report.json", structure)
        _write_json(output_dir / "official_results.json", {"mode": "adaptive-tissue", "tasks": task_reports})
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
    ) -> OfficialBenchmarkRunResult | AdaptiveBenchmarkReport:
        selected_mode = mode or ("provider-baseline" if benchmark == "bfcl" else "adaptive-tissue")
        if not full and limit is None and task_id is None:
            raise ValueError("official benchmark runs require --limit, --task-id, or --full")
        if selected_mode == "adaptive-tissue":
            adapter = OfficialBenchmarkAdapter.for_benchmark(benchmark)
            tasks = adapter.load_tasks(limit=limit, task_id=task_id, dry_run=dry_run)
            return AdaptiveTissueBenchmarkRunner(model_profile=model_profile, dry_run=dry_run).run_tasks(tasks, output)
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
        _write_jsonl(output_dir / "official_answers.jsonl", [{"id": task.get("id", ""), "ground_truth": answers.get(str(task.get("id", "")), [])} for task in tasks])
        _write_jsonl(output_dir / "ontocellia_predictions.jsonl", predictions)
        _write_json(output_dir / "official_results.json", {"benchmark": "bfcl", "scores": scores, "predictions": predictions})
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
    if task.source_benchmark == "swe-bench-lite":
        signals.update({"test_failure": 0.9, "repair_pressure": 0.85})
    if task.source_benchmark == "tau-bench":
        signals.update({"risk": 0.7, "planning_pressure": 0.75})
    return MorphogenField(signals)


def _structure_metrics(tissue: TissueRuntime, actions: list[dict[str, Any]]) -> dict[str, Any]:
    events = tissue.trace.events
    execution_events = [event for event in events if event["type"].startswith("execution_")]
    validation_cycles = [event for event in events if event["type"] == "organ_selection"]
    provider_calls = [event for event in events if event["type"] == "llm_effector"]
    handoffs = [event for event in events if event["type"] == "handoff_completed"]
    matrix_records = len(tissue.environment.matrix.records)
    action_count = max(1, len(actions))
    return {
        "final_task_success": 0.0,
        "fate_distribution": tissue.fate_counts(),
        "proliferation_events": sum(1 for event in events if event["type"] == "proliferation"),
        "handoff_completion_rate": len(handoffs) / action_count,
        "matrix_reuse_rate": min(1.0, matrix_records / action_count),
        "execution_success_rate": _execution_success_rate(execution_events),
        "validation_feedback_cycles": len(validation_cycles),
        "provider_call_count": len(provider_calls),
        "structure_efficiency": round(min(1.0, (len(tissue.fate_counts()) / 5.0) * 0.4 + min(1.0, matrix_records / action_count) * 0.3 + (len(handoffs) > 0) * 0.3), 6),
        "regeneration_events": sum(1 for event in events if event["type"] == "regeneration"),
    }


def _execution_success_rate(events: list[dict[str, Any]]) -> float:
    completed = [event for event in events if event["type"] == "execution_completed"]
    if not completed:
        return 0.0
    return sum(1 for event in completed if event.get("passed")) / len(completed)


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

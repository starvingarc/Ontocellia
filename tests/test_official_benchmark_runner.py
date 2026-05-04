from __future__ import annotations

import json
from pathlib import Path

import pytest

from ontocellia.__main__ import main
from ontocellia.framework.official_benchmark import (
    AdaptiveBenchmarkTask,
    AdaptiveTissueBenchmarkRunner,
    OfficialBenchmarkAdapter,
    OfficialBenchmarkRunner,
)
from ontocellia.framework.model_config import ModelProfile, OntocelliaUserConfig, save_user_config


def test_bfcl_default_mode_is_provider_baseline(tmp_path: Path) -> None:
    result = OfficialBenchmarkRunner().run(
        benchmark="bfcl",
        output=tmp_path,
        model_profile="mock",
        limit=1,
        dry_run=True,
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["benchmark"] == "bfcl"
    assert summary["mode"] == "provider-baseline"
    assert "tissue_traces" not in {path.name for path in tmp_path.iterdir()}


def test_adaptive_task_conversion_produces_generic_microenvironment() -> None:
    task = AdaptiveBenchmarkTask(
        id="tau-fixture",
        source_benchmark="tau-bench",
        prompt="Help the user change a reservation while following policy.",
        metadata={
            "policy": "Only change refundable reservations.",
            "tools": [{"name": "lookup_reservation"}, {"name": "change_reservation"}],
            "user": "traveler",
        },
    )

    adapter = OfficialBenchmarkAdapter.for_benchmark("tau-bench")
    draft = adapter.to_induction_request(task)

    assert draft.domain == "generic"
    assert "reservation" in draft.task
    assert "lookup_reservation" in draft.available_interfaces
    assert draft.constraints["source_benchmark"] == "tau-bench"


def test_adaptive_runner_generates_tissue_traces_and_structure_metrics(tmp_path: Path) -> None:
    task = AdaptiveBenchmarkTask(
        id="terminal-fixture",
        source_benchmark="terminal-bench",
        prompt="Inspect project files and run the provided check command.",
        metadata={"check_command": "python -m pytest -q", "files": ["README.md"]},
    )

    result = AdaptiveTissueBenchmarkRunner(model_profile="mock", dry_run=True).run_tasks([task], tmp_path)

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    structure = json.loads((tmp_path / "structure_report.json").read_text(encoding="utf-8"))
    trace = json.loads((tmp_path / "tissue_traces" / "terminal-fixture" / "tissue_trace.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "adaptive-tissue"
    assert summary["tasks"] == 1
    assert structure["tasks"][0]["source_benchmark"] == "terminal-bench"
    assert structure["tasks"][0]["metrics"]["proliferation_events"] > 0
    assert "fate_distribution" in structure["tasks"][0]["metrics"]
    assert any(event["type"] == "development_stage_changed" for event in trace)


def test_tau_bench_fixture_maps_policy_tools_into_matrix_and_interfaces() -> None:
    adapter = OfficialBenchmarkAdapter.for_benchmark("tau-bench")
    task = adapter.fixture_task()
    environment = adapter.to_microenvironment(task)

    assert any(record.kind == "policy" for record in environment.matrix.records)
    assert {interface.id for interface in environment.interfaces} >= {"lookup_order", "update_order"}


def test_terminal_bench_fixture_maps_check_command_into_execution_metadata() -> None:
    adapter = OfficialBenchmarkAdapter.for_benchmark("terminal-bench")
    task = adapter.fixture_task()
    environment = adapter.to_microenvironment(task)

    assert environment.matrix.query(tags=["check_command"], limit=1)
    assert any(interface.id == "shell.run" for interface in environment.interfaces)


def test_multiagentbench_fixture_produces_collaboration_pressure() -> None:
    adapter = OfficialBenchmarkAdapter.for_benchmark("multiagentbench")
    task = adapter.fixture_task()
    environment = adapter.to_microenvironment(task)

    assert environment.morphogens.signal("coordination_pressure") > 0
    assert len(environment.niches) >= 4


def test_swe_bench_fixture_maps_issue_tests_to_repo_repair_environment() -> None:
    adapter = OfficialBenchmarkAdapter.for_benchmark("swe-bench-lite")
    task = adapter.fixture_task()
    environment = adapter.to_microenvironment(task)

    assert environment.morphogens.signal("test_failure") > 0
    assert any(interface.id == "pytest.run" for interface in environment.interfaces)


def test_official_benchmark_cli_runs_adaptive_tissue_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONTOCELLIA_CONFIG_DIR", str(tmp_path / "config"))
    save_user_config(
        OntocelliaUserConfig(
            models={"default": "mock", "profiles": {"mock": ModelProfile(provider="mock-llm", model="mock-llm")}}
        )
    )
    output = tmp_path / "cli"

    main(
        [
            "official-benchmark",
            "run",
            "--benchmark",
            "tau-bench",
            "--model-profile",
            "mock",
            "--limit",
            "1",
            "--mode",
            "adaptive-tissue",
            "--dry-run",
            "--output",
            str(output),
        ]
    )

    summary = json.loads((output / "ontocellia_summary.json").read_text(encoding="utf-8"))
    assert summary["benchmark"] == "tau-bench"
    assert summary["mode"] == "adaptive-tissue"
    assert (output / "adaptation_report.md").exists()


def test_official_adaptive_run_writes_official_tasks_and_scoring_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONTOCELLIA_CONFIG_DIR", str(tmp_path / "config"))
    save_user_config(
        OntocelliaUserConfig(
            models={"default": "mock", "profiles": {"mock": ModelProfile(provider="mock-llm", model="mock-llm")}}
        )
    )
    source = tmp_path / "terminal-source"
    task_dir = source / "original-tasks" / "jsonl-aggregator"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        "instruction: aggregate JSONL records\ncategory: file-operations\nparser_name: pytest\n",
        encoding="utf-8",
    )

    result = OfficialBenchmarkRunner().run(
        benchmark="terminal-bench",
        output=tmp_path / "run",
        model_profile="mock",
        limit=1,
        dry_run=False,
        source_dir=source,
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    scoring = json.loads((tmp_path / "run" / "scoring_status.json").read_text(encoding="utf-8"))
    assert summary["benchmark"] == "terminal-bench"
    assert summary["official_score_status"] == "not_run"
    assert scoring["official_score_status"] == "not_run"
    assert (tmp_path / "run" / "official_tasks.jsonl").exists()
    assert (tmp_path / "run" / "official_task_manifest.json").exists()


def test_official_adaptive_run_with_structure_search_writes_selected_variant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONTOCELLIA_CONFIG_DIR", str(tmp_path / "config"))
    save_user_config(
        OntocelliaUserConfig(
            models={"default": "mock", "profiles": {"mock": ModelProfile(provider="mock-llm", model="mock-llm")}}
        )
    )
    source = tmp_path / "terminal-source"
    task_dir = source / "original-tasks" / "debug-task"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        """
instruction: Fix a failing pytest regression.
category: debugging
tags:
  - debugging
parser_name: pytest
""",
        encoding="utf-8",
    )

    OfficialBenchmarkRunner().run(
        benchmark="terminal-bench",
        output=tmp_path / "run",
        model_profile="mock",
        limit=1,
        dry_run=False,
        source_dir=source,
        structure_search=True,
    )

    structure = json.loads((tmp_path / "run" / "structure_report.json").read_text(encoding="utf-8"))
    task = structure["tasks"][0]
    assert task["metrics"]["selected_variant"] in {"baseline", "lean", "memory_heavy", "repair_heavy", "review_heavy"}
    assert "repair_presence" in task["metrics"]
    assert (tmp_path / "run" / "tissue_traces" / "debug-task" / "variants" / "baseline" / "tissue_trace.json").exists()


def test_official_runner_records_provider_errors_without_aborting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class TimeoutProvider:
        name = "timeout-provider"

        def complete(self, prompt):  # type: ignore[no-untyped-def]
            raise TimeoutError("provider timed out")

    monkeypatch.setattr("ontocellia.framework.official_benchmark.resolve_effector_provider", lambda *args, **kwargs: TimeoutProvider())
    task = AdaptiveBenchmarkTask(
        id="tau-timeout",
        source_benchmark="tau-bench",
        prompt="Book a reservation using official tau-bench tools.",
        metadata={"tools": [{"name": "book_reservation"}], "policy": "official tau-bench airline environment task"},
    )

    result = AdaptiveTissueBenchmarkRunner(model_profile="timeout", dry_run=False).run_tasks([task], tmp_path)

    structure = json.loads(result.structure_path.read_text(encoding="utf-8"))
    trace = json.loads((tmp_path / "tissue_traces" / "tau-timeout" / "tissue_trace.json").read_text(encoding="utf-8"))
    assert structure["tasks"][0]["metrics"]["provider_call_errors"] > 0
    assert any(event["type"] == "official_benchmark_provider_error" for event in trace)


def test_provider_baseline_scores_bfcl_mock_predictions(tmp_path: Path) -> None:
    tasks = [
        {
            "id": "simple_0",
            "question": [[{"role": "user", "content": "Find area."}]],
            "function": [{"name": "calculate_triangle_area", "parameters": {"properties": {"base": {}, "height": {}}}}],
        }
    ]
    answers = {"simple_0": [{"calculate_triangle_area": {"base": [10], "height": [5]}}]}

    result = OfficialBenchmarkRunner().run_bfcl_records(
        tasks=tasks,
        answers=answers,
        output=tmp_path,
        model_profile="mock",
        dry_run=True,
        mode="provider-baseline",
        mock_predictions=[{"id": "simple_0", "function_calls": [{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}]}],
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["mode"] == "provider-baseline"
    assert summary["function_name_accuracy"] == 1.0
    assert summary["argument_key_accuracy"] == 1.0


def test_official_outputs_do_not_contain_api_key_pattern(tmp_path: Path) -> None:
    result = OfficialBenchmarkRunner().run(
        benchmark="bfcl",
        output=tmp_path,
        model_profile="mock",
        limit=1,
        dry_run=True,
    )

    combined = "\n".join(path.read_text(encoding="utf-8") for path in tmp_path.iterdir() if path.is_file())
    assert "sk-" not in combined
    assert result.report_path.exists()

from __future__ import annotations

import csv
import json
from pathlib import Path

from ontocellia.__main__ import main
from ontocellia.framework.benchmark import BenchmarkSuite, TissueBenchmarkRunner


def test_minibench_suite_lists_five_native_tasks() -> None:
    suite = BenchmarkSuite.builtin("ontocellia_minibench_v1")

    assert [task.id for task in suite.tasks] == [
        "repo_repair_intent",
        "tool_gate_policy",
        "matrix_memory",
        "self_repair_recovery",
        "decentralized_handoff",
    ]
    assert suite.tasks[0].expected_fates == ["explorer", "repair", "reviewer"]


def test_mock_runner_writes_summary_csv_report_and_task_artifacts(tmp_path: Path) -> None:
    suite = BenchmarkSuite.builtin("ontocellia_minibench_v1")
    result = TissueBenchmarkRunner(suite=suite, effector="mock-llm").run(tmp_path)

    assert result.summary_path == tmp_path / "benchmark_summary.json"
    assert result.csv_path == tmp_path / "benchmark_results.csv"
    assert result.report_path == tmp_path / "benchmark_report.md"
    assert len(result.results) == 5
    assert all(item.score >= 0.0 for item in result.results)

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["suite"] == "ontocellia_minibench_v1"
    assert summary["tasks"] == 5
    assert summary["average_score"] > 0.0

    rows = list(csv.DictReader(result.csv_path.open(encoding="utf-8")))
    assert {row["task_id"] for row in rows} == {task.id for task in suite.tasks}

    for task in suite.tasks:
        task_dir = tmp_path / "tasks" / task.id
        assert (task_dir / "tissue_summary.json").exists()
        assert (task_dir / "tissue_trace.json").exists()
        assert (task_dir / "action_intents.json").exists()


def test_repo_repair_task_scores_fates_and_intents(tmp_path: Path) -> None:
    result = TissueBenchmarkRunner(
        suite=BenchmarkSuite([BenchmarkSuite.builtin("ontocellia_minibench_v1").task("repo_repair_intent")]),
        effector="mock-llm",
    ).run(tmp_path)

    metrics = result.results[0].metrics

    assert metrics["intent_quality"] > 0.0
    assert metrics["task_success"] == 1.0
    assert "repair" in result.results[0].trace_summary["fate_counts"]
    assert result.results[0].trace_summary["actions"] > 0


def test_tool_gate_matrix_memory_self_repair_and_handoff_metrics(tmp_path: Path) -> None:
    result = TissueBenchmarkRunner(suite=BenchmarkSuite.builtin("ontocellia_minibench_v1"), effector="mock-llm").run(tmp_path)
    by_id = {item.task.id: item for item in result.results}

    assert by_id["tool_gate_policy"].metrics["interface_policy_compliance"] == 1.0
    assert by_id["matrix_memory"].metrics["matrix_reuse_rate"] > 0.0
    assert by_id["self_repair_recovery"].metrics["regeneration_recovery_ticks"] > 0.0
    assert by_id["self_repair_recovery"].metrics["lineage_traceability"] == 1.0
    assert by_id["decentralized_handoff"].metrics["handoff_completion_rate"] > 0.0
    assert by_id["decentralized_handoff"].metrics["decentralization_score"] > 0.0


def test_benchmark_cli_writes_expected_outputs(tmp_path: Path) -> None:
    output = tmp_path / "bench"

    main(["benchmark", "--suite", "ontocellia_minibench_v1", "--effector", "mock-llm", "--output", str(output)])

    assert (output / "benchmark_summary.json").exists()
    assert (output / "benchmark_results.csv").exists()
    assert (output / "benchmark_report.md").exists()
    summary = json.loads((output / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert summary["tasks"] == 5


def test_benchmark_execution_smoke_writes_execution_artifacts(tmp_path: Path) -> None:
    task = BenchmarkSuite.builtin("ontocellia_minibench_v1").task("repo_repair_intent")
    result = TissueBenchmarkRunner(
        suite=BenchmarkSuite([task], name="execution_smoke"),
        effector="mock-llm",
        execute_actions=True,
        allowed_interfaces=["workspace.search", "git.diff", "pytest.run"],
        execution_dry_run=True,
    ).run(tmp_path)

    task_dir = tmp_path / "tasks" / "repo_repair_intent"
    execution_path = task_dir / "execution_results.json"
    assert result.results[0].artifacts["execution"] == str(execution_path)
    assert execution_path.exists()
    execution_results = json.loads(execution_path.read_text(encoding="utf-8"))
    assert execution_results

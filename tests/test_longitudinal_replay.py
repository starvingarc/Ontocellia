from __future__ import annotations

import csv
import json
from pathlib import Path

from ontocellia.__main__ import main
from ontocellia.framework.longitudinal import LongitudinalReplayRunner, builtin_replay_tasks


def test_builtin_replay_tasks_are_stable() -> None:
    tasks = builtin_replay_tasks()

    assert [task.id for task in tasks] == [
        "repo_repair_failing_tests",
        "repo_repair_regression",
        "repo_repair_risk_review",
    ]
    assert all(task.domain == "repo_repair" for task in tasks)
    assert all(task.prompt for task in tasks)


def test_longitudinal_replay_writes_reports_and_baselines(tmp_path: Path) -> None:
    report = LongitudinalReplayRunner(
        tasks=builtin_replay_tasks()[:2],
        effector="mock-llm",
        steps=4,
        seed=7,
    ).run(tmp_path)

    assert report.summary_path == tmp_path / "longitudinal_replay_summary.json"
    assert report.csv_path == tmp_path / "longitudinal_trials.csv"
    assert report.report_path == tmp_path / "longitudinal_replay_report.md"
    assert report.solidification_path == tmp_path / "solidification" / "solidification_report.md"
    assert report.summary_path.exists()
    assert report.csv_path.exists()
    assert report.report_path.exists()
    assert report.solidification_path.exists()

    summary = json.loads(report.summary_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(report.csv_path.open(encoding="utf-8")))

    assert summary["task_count"] == 2
    assert set(summary["conditions"]) == {"direct_agent", "single_cell", "fixed_tissue", "adaptive_tissue"}
    assert "adaptive_gain_over_fixed" in summary["comparison"]
    assert len(rows) == 8
    assert {row["condition"] for row in rows} == set(summary["conditions"])
    assert (tmp_path / "tasks" / "repo_repair_failing_tests" / "adaptive_tissue" / "structure_search_summary.json").exists()


def test_adaptive_replay_records_structure_memory_and_selected_variants(tmp_path: Path) -> None:
    report = LongitudinalReplayRunner(
        tasks=builtin_replay_tasks()[:2],
        effector="mock-llm",
        steps=5,
        seed=11,
    ).run(tmp_path)

    summary = json.loads(report.summary_path.read_text(encoding="utf-8"))
    adaptive_trials = [trial for trial in summary["trials"] if trial["condition"] == "adaptive_tissue"]

    assert len(adaptive_trials) == 2
    assert all(trial["selected_variant"] for trial in adaptive_trials)
    assert summary["solidification"]["decision"] in {"selected", "not_selected"}
    assert "solidified_tendencies" in summary["memory"]
    assert summary["memory"]["replay_sessions"] == 2
    assert summary["comparison"]["adaptive_average_score"] >= 0.0


def test_longitudinal_replay_cli_writes_expected_outputs(tmp_path: Path) -> None:
    output = tmp_path / "replay"

    main(
        [
            "longitudinal-replay",
            "--task",
            "Fix failing tests while preserving behavior.",
            "--task",
            "Fix a regression without broad rewrites.",
            "--domain",
            "repo_repair",
            "--effector",
            "mock-llm",
            "--steps",
            "4",
            "--seed",
            "7",
            "--output",
            str(output),
        ]
    )

    assert (output / "longitudinal_replay_summary.json").exists()
    assert (output / "longitudinal_trials.csv").exists()
    assert (output / "longitudinal_replay_report.md").exists()
    summary = json.loads((output / "longitudinal_replay_summary.json").read_text(encoding="utf-8"))
    assert summary["task_count"] == 2
    assert summary["comparison"]["conditions_compared"] == 4

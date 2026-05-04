from __future__ import annotations

import csv
import json
from pathlib import Path

from ontocellia.__main__ import main
from ontocellia.framework.structure_search import StructureSearchRunner, builtin_structure_variants


def test_builtin_structure_variants_are_stable() -> None:
    variants = builtin_structure_variants()

    assert [variant.name for variant in variants] == [
        "baseline",
        "repair_heavy",
        "review_heavy",
        "memory_heavy",
        "lean",
    ]
    assert variants[1].niche_demand_patch["repair"] > 0
    assert variants[2].morphogen_patch["review_pressure"] > variants[0].morphogen_patch.get("review_pressure", 0.0)
    assert variants[3].niche_demand_patch["memory"] > 0
    assert variants[4].target_population_delta < 0


def test_structure_search_runner_writes_trials_and_selects_winner(tmp_path: Path) -> None:
    report = StructureSearchRunner(
        task="Fix failing tests while preserving behavior.",
        domain="repo_repair",
        effector="mock-llm",
        steps=6,
        seed=7,
    ).run(tmp_path)

    assert report.summary_path == tmp_path / "structure_search_summary.json"
    assert report.csv_path == tmp_path / "structure_trials.csv"
    assert report.report_path == tmp_path / "structure_search_report.md"
    assert report.selected_path == tmp_path / "selected_variant.json"
    assert [trial.variant.name for trial in report.trials] == [variant.name for variant in builtin_structure_variants()]
    assert report.selected_variant in {trial.variant.name for trial in report.trials}

    summary = json.loads(report.summary_path.read_text(encoding="utf-8"))
    selected = json.loads(report.selected_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(report.csv_path.open(encoding="utf-8")))

    assert summary["selected_variant"] == selected["variant"]
    assert len(summary["trials"]) == 5
    assert len(rows) == 5
    assert "structure_score" in rows[0]
    assert (tmp_path / "structure_search_report.md").read_text(encoding="utf-8").startswith("# Structure Search Report")


def test_each_structure_trial_contains_required_metrics_and_artifacts(tmp_path: Path) -> None:
    report = StructureSearchRunner(task="Fix failing tests while preserving behavior.", steps=5).run(tmp_path)

    for trial in report.trials:
        metrics = trial.metrics
        assert "fate_distribution" in metrics
        assert "validation_score" in metrics
        assert "cost" in metrics
        assert "handoff_completion_rate" in metrics
        assert "matrix_reuse_rate" in metrics
        assert "regeneration_recovery_ticks" in metrics
        assert "structure_score" in metrics
        assert 0.0 <= float(metrics["structure_score"]) <= 1.0
        variant_dir = tmp_path / "variants" / trial.variant.name
        assert (variant_dir / "tissue_summary.json").exists()
        assert (variant_dir / "tissue_trace.json").exists()
        assert (variant_dir / "action_intents.json").exists()


def test_structure_search_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    first = StructureSearchRunner(task="Fix failing tests while preserving behavior.", seed=11).run(tmp_path / "first")
    second = StructureSearchRunner(task="Fix failing tests while preserving behavior.", seed=11).run(tmp_path / "second")

    assert first.selected_variant == second.selected_variant
    assert [trial.score for trial in first.trials] == [trial.score for trial in second.trials]


def test_heavy_variants_change_expected_structure_pressure(tmp_path: Path) -> None:
    report = StructureSearchRunner(task="Fix failing tests while preserving behavior.", steps=6).run(tmp_path)
    by_name = {trial.variant.name: trial for trial in report.trials}

    assert by_name["repair_heavy"].trace_summary["target_population"] >= by_name["baseline"].trace_summary["target_population"]
    assert by_name["review_heavy"].metrics["fate_distribution"].get("reviewer", 0) >= by_name["baseline"].metrics["fate_distribution"].get("reviewer", 0)
    assert by_name["memory_heavy"].metrics["fate_distribution"].get("memory", 0) >= by_name["baseline"].metrics["fate_distribution"].get("memory", 0)
    assert by_name["lean"].trace_summary["target_population"] <= by_name["baseline"].trace_summary["target_population"]


def test_structure_search_cli_writes_expected_outputs(tmp_path: Path) -> None:
    output = tmp_path / "search"

    main(
        [
            "structure-search",
            "--task",
            "Fix failing tests while preserving behavior.",
            "--domain",
            "repo_repair",
            "--effector",
            "mock-llm",
            "--steps",
            "5",
            "--seed",
            "7",
            "--output",
            str(output),
        ]
    )

    assert (output / "structure_search_summary.json").exists()
    assert (output / "structure_trials.csv").exists()
    assert (output / "structure_search_report.md").exists()
    assert (output / "selected_variant.json").exists()
    summary = json.loads((output / "structure_search_summary.json").read_text(encoding="utf-8"))
    assert summary["task"] == "Fix failing tests while preserving behavior."
    assert summary["selected_variant"]

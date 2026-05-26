from __future__ import annotations

import json
from pathlib import Path

from ontocellia.__main__ import main
from ontocellia.framework import (
    DevelopmentalAnnealingPolicy,
    DevelopmentalAnnealingRuntime,
    DifferentiatedCellState,
    OrganValidationResult,
    load_task_microenvironment,
)

from tests.test_agent_tissue_framework import make_repo_repair_tissue


def failed_validation() -> list[OrganValidationResult]:
    return [
        OrganValidationResult(
            name="pytest",
            passed=False,
            score=0.0,
            target="repo",
            evidence="repeated failing tests",
            cost=0.1,
            risk=0.4,
            latency=0.0,
        )
    ]


def test_annealing_temperature_cools_as_tissue_develops() -> None:
    tissue = make_repo_repair_tissue(stem_cells=None)
    runtime = DevelopmentalAnnealingRuntime()

    early = runtime.apply(tissue, policy=DevelopmentalAnnealingPolicy(warmup_ticks=1, stabilization_ticks=4))
    tissue.tick_count = 6
    late = runtime.apply(tissue, policy=DevelopmentalAnnealingPolicy(warmup_ticks=1, stabilization_ticks=4))

    assert early.temperature > late.temperature
    assert late.commitment > early.commitment
    assert tissue.environment.morphogens.signal("exploration_pressure") > 0.0
    assert any(event["type"] == "developmental_annealing" for event in tissue.trace.events)


def test_stable_mature_tissue_increases_fate_lock() -> None:
    tissue = make_repo_repair_tissue(stem_cells=None)
    tissue.develop(ticks=7)
    repair = next(cell for cell in tissue.cells.values() if cell.fate == "repair")
    assert isinstance(repair.stage_state, DifferentiatedCellState)
    before = repair.stage_state.fate_lock

    DevelopmentalAnnealingRuntime().apply(
        tissue,
        policy=DevelopmentalAnnealingPolicy(warmup_ticks=0, stabilization_ticks=1, fate_lock_growth=0.2),
        validation_results=[OrganValidationResult("pytest", True, 1.0, "repo", "passed", 0.1, 0.0, 0.0)],
    )

    assert repair.stage_state.fate_lock > before
    assert repair.epigenetic_marks.fate_locks["repair"] == repair.stage_state.fate_lock


def test_repeated_validation_failure_unlocks_and_reprograms_local_cell() -> None:
    tissue = make_repo_repair_tissue(stem_cells=None)
    tissue.develop(ticks=7)
    explorer = next(cell for cell in tissue.cells.values() if cell.fate == "explorer")
    explorer.energy = 0.9
    assert explorer.stage == "differentiated"

    policy = DevelopmentalAnnealingPolicy(
        repeated_failure_threshold=2,
        max_reprogramming_per_tick=1,
        failure_unlock=0.25,
        reprogramming_energy_cost=0.2,
    )
    runtime = DevelopmentalAnnealingRuntime()
    runtime.apply(tissue, policy=policy, validation_results=failed_validation())
    report = runtime.apply(tissue, policy=policy, validation_results=failed_validation())

    assert report.reprogrammed_cell_ids
    reprogrammed = tissue.cells[int(report.reprogrammed_cell_ids[0])]
    assert reprogrammed.stage == "progenitor"
    assert reprogrammed.fate in {"repair", "reviewer"}
    assert reprogrammed.energy < 0.9
    assert any(event["type"] == "annealing_reprogramming" for event in tissue.trace.events)


def test_environment_yaml_accepts_annealing_policy(tmp_path: Path) -> None:
    environment_path = tmp_path / "environment.yaml"
    environment_path.write_text(
        """
objective: annealing policy test
morphogens:
  ambiguity: 0.5
niches:
  - id: repair-niche
    required_fate: repair
    position: {node_id: repair-niche}
annealing:
  warmup_ticks: 2
  stabilization_ticks: 9
  final_temperature: 0.2
  fate_lock_growth: 0.07
  repeated_failure_threshold: 3
""",
        encoding="utf-8",
    )

    environment = load_task_microenvironment(environment_path)

    assert environment.annealing_policy.warmup_ticks == 2
    assert environment.annealing_policy.stabilization_ticks == 9
    assert environment.annealing_policy.final_temperature == 0.2
    assert environment.annealing_policy.fate_lock_growth == 0.07
    assert environment.annealing_policy.repeated_failure_threshold == 3


def test_tissue_cli_summary_includes_annealing(tmp_path: Path) -> None:
    output = tmp_path / "tissue"

    main(
        [
            "tissue",
            "--genome-spec",
            "examples/framework/repo_repair_genome.yaml",
            "--environment-spec",
            "examples/framework/failing_tests_environment.yaml",
            "--steps",
            "4",
            "--output",
            str(output),
        ]
    )

    summary = json.loads((output / "tissue_summary.json").read_text(encoding="utf-8"))
    assert "annealing" in summary
    assert "temperature" in summary["annealing"]
    assert "average_fate_lock" in summary["annealing"]


def test_structure_search_metrics_include_annealing_state(tmp_path: Path) -> None:
    output = tmp_path / "search"

    main(
        [
            "structure-search",
            "--task",
            "Fix failing tests while preserving behavior.",
            "--steps",
            "4",
            "--output",
            str(output),
        ]
    )

    summary = json.loads((output / "structure_search_summary.json").read_text(encoding="utf-8"))
    for trial in summary["trials"]:
        metrics = trial["metrics"]
        assert "annealing_temperature" in metrics
        assert "average_fate_lock" in metrics
        assert "reprogramming_events" in metrics

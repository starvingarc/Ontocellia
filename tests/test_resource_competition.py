from __future__ import annotations

import json
from pathlib import Path

from ontocellia.__main__ import main
from ontocellia.framework import (
    ContributionAttributionRuntime,
    ResourceCompetitionPolicy,
    ResourceCompetitionRuntime,
    load_task_microenvironment,
)

from tests.test_agent_tissue_framework import make_repo_repair_tissue
from tests.test_contribution_attribution import sample_action, sample_trace


def test_resource_competition_applies_cell_maintenance_cost() -> None:
    tissue = make_repo_repair_tissue(stem_cells=None)
    tissue.develop(ticks=1)
    before = {cell.id: cell.energy for cell in tissue.cells.values()}

    report = ResourceCompetitionRuntime().apply(
        tissue,
        policy=ResourceCompetitionPolicy(maintenance_cost=0.08, differentiated_cost=0.04),
    )

    assert report.average_cell_energy < 1.0
    assert any(tissue.cells[cell_id].energy < energy for cell_id, energy in before.items())
    assert any(event["type"] == "resource_competition" for event in tissue.trace.events)
    assert tissue.environment.morphogens.signal("resource_pressure") >= 0.0


def test_contribution_report_rewards_helpful_cells_and_penalizes_failed_paths() -> None:
    tissue = make_repo_repair_tissue(stem_cells=None)
    tissue.develop(ticks=6)
    repair_cell = next(cell for cell in tissue.cells.values() if cell.fate == "repair")
    repair_cell.energy = 0.5
    attribution = ContributionAttributionRuntime().analyze(
        trace=sample_trace(),
        actions=[sample_action()],
        validation_results=[{"name": "pytest", "passed": False, "score": 0.0, "target": "repo", "evidence": "pytest failed"}],
    )

    report = ResourceCompetitionRuntime().apply(
        tissue,
        contribution_report=attribution,
        policy=ResourceCompetitionPolicy(maintenance_cost=0.0, contribution_reward=0.25, negative_contribution_penalty=0.2),
    )

    assert tissue.cells[2].energy > 0.5
    assert tissue.cells[2].stress > 0.0
    assert report.cell_deltas["2"].contribution_reward > 0.0
    assert report.cell_deltas["2"].negative_penalty > 0.0


def test_population_pressure_suppresses_growth_above_cap() -> None:
    tissue = make_repo_repair_tissue(stem_cells=6)
    tissue.develop(ticks=2)

    report = ResourceCompetitionRuntime().apply(
        tissue,
        policy=ResourceCompetitionPolicy(population_cap=3, maintenance_cost=0.0, over_cap_pressure_weight=0.5),
    )

    assert report.population_pressure > 0.0
    assert tissue.environment.morphogens.signal("resource_pressure") > 0.0
    assert tissue.target_population <= 3


def test_low_energy_non_origin_cell_can_be_removed_under_population_pressure() -> None:
    tissue = make_repo_repair_tissue(stem_cells=5)
    tissue.develop(ticks=1)
    removable_id = max(tissue.cells)
    tissue.cells[removable_id].energy = 0.01

    report = ResourceCompetitionRuntime().apply(
        tissue,
        policy=ResourceCompetitionPolicy(population_cap=2, apoptosis_threshold=0.05, allow_apoptosis=True, maintenance_cost=0.0),
    )

    assert removable_id not in tissue.cells
    assert str(removable_id) in report.removed_cell_ids
    assert any(event["type"] == "resource_apoptosis" and event["cell_id"] == removable_id for event in tissue.trace.events)


def test_runtime_develop_records_resource_report() -> None:
    tissue = make_repo_repair_tissue(stem_cells=None)

    tissue.develop(ticks=2)

    assert tissue.last_resource_report is not None
    assert "average_cell_energy" in tissue.last_resource_report.as_dict()
    assert any(event["type"] == "resource_competition" for event in tissue.trace.events)


def test_environment_yaml_accepts_resource_policy(tmp_path: Path) -> None:
    environment_path = tmp_path / "environment.yaml"
    environment_path.write_text(
        """
objective: resource pressure test
morphogens:
  ambiguity: 0.5
niches:
  - id: repair-niche
    required_fate: repair
    position: {node_id: repair-niche}
resources:
  population_cap: 3
  maintenance_cost: 0.02
  differentiated_cost: 0.03
  contribution_reward: 0.2
""",
        encoding="utf-8",
    )

    environment = load_task_microenvironment(environment_path)

    assert environment.resource_policy.population_cap == 3
    assert environment.resource_policy.maintenance_cost == 0.02
    assert environment.resource_policy.differentiated_cost == 0.03
    assert environment.resource_policy.contribution_reward == 0.2


def test_tissue_cli_summary_includes_resource_competition(tmp_path: Path) -> None:
    output = tmp_path / "tissue"

    main(
        [
            "tissue",
            "--genome-spec",
            "examples/framework/repo_repair_genome.yaml",
            "--environment-spec",
            "examples/framework/failing_tests_environment.yaml",
            "--steps",
            "3",
            "--output",
            str(output),
        ]
    )

    summary = json.loads((output / "tissue_summary.json").read_text(encoding="utf-8"))
    assert "resource_competition" in summary
    assert summary["resource_competition"]["average_cell_energy"] <= 1.0


def test_structure_search_metrics_include_resource_efficiency(tmp_path: Path) -> None:
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
        assert "resource_efficiency" in metrics
        assert "average_cell_energy" in metrics
        assert "population_pressure" in metrics

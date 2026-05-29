from __future__ import annotations

import json
from pathlib import Path

from ontocellia.__main__ import main
from ontocellia.framework import (
    AgentGenome,
    Gene,
    SelectionSolidificationPolicy,
    SelectionSolidificationRuntime,
    StructureSearchReport,
    StructureTrialResult,
    StructureVariant,
)


def repo_genome() -> AgentGenome:
    return AgentGenome(
        genes=[
            Gene("gene_repair_from_test_failures", "regeneration", ["repair_pressure"], ["repair"]),
            Gene("gene_review_boundary", "verification", ["review_pressure"], ["review"]),
            Gene("gene_memory_context", "memory", ["memory_pressure"], ["remember"]),
        ]
    )


def report_for(selected: str = "repair_heavy", selected_score: float = 0.72, baseline_score: float = 0.58) -> StructureSearchReport:
    trials = [
        StructureTrialResult(
            StructureVariant("baseline"),
            baseline_score,
            {"structure_score": baseline_score, "validation_score": 0.4},
            {},
            {},
        ),
        StructureTrialResult(
            StructureVariant("repair_heavy", {"repair_pressure": 0.35, "test_failure": 0.25}, {"repair": 1}, target_population_delta=1),
            selected_score if selected == "repair_heavy" else 0.55,
            {"structure_score": selected_score if selected == "repair_heavy" else 0.55, "validation_score": 0.55},
            {},
            {},
        ),
        StructureTrialResult(
            StructureVariant("review_heavy", {"review_pressure": 0.45}, {"reviewer": 1}, target_population_delta=1),
            selected_score if selected == "review_heavy" else 0.54,
            {"structure_score": selected_score if selected == "review_heavy" else 0.54, "validation_score": 0.5},
            {},
            {},
        ),
    ]
    return StructureSearchReport(
        task="Fix failing tests.",
        domain="repo_repair",
        output_dir=Path("unused"),
        trials=trials,
        selected_variant=selected,
        summary_path=Path("unused/summary.json"),
        csv_path=Path("unused/trials.csv"),
        report_path=Path("unused/report.md"),
        selected_path=Path("unused/selected.json"),
    )


def test_solidification_selects_repeated_effective_structure() -> None:
    genome = repo_genome()

    report = SelectionSolidificationRuntime().solidify(
        [report_for("repair_heavy"), report_for("repair_heavy", selected_score=0.74)],
        genome=genome,
        policy=SelectionSolidificationPolicy(min_repetitions=2, min_structure_score=0.65, min_margin=0.05),
    )

    assert report.decision == "selected"
    assert report.selected_tendency is not None
    assert report.selected_tendency.variant_name == "repair_heavy"
    assert report.selected_tendency.morphogen_bias["repair_pressure"] > 0
    assert report.selected_tendency.niche_demand_bias["repair"] == 1
    assert report.solidified_genome is not genome
    assert genome.regulatory_elements == []
    assert any(element.target_gene_id == "gene_repair_from_test_failures" for element in report.solidified_genome.regulatory_elements)


def test_solidification_requires_score_margin() -> None:
    report = SelectionSolidificationRuntime().solidify(
        [report_for("repair_heavy", selected_score=0.61, baseline_score=0.6)],
        genome=repo_genome(),
        policy=SelectionSolidificationPolicy(min_structure_score=0.55, min_margin=0.05),
    )

    assert report.decision == "not_selected"
    assert report.selected_tendency is None
    assert report.solidified_genome.regulatory_elements == []


def test_solidification_writes_report_and_genome(tmp_path: Path) -> None:
    report = SelectionSolidificationRuntime().solidify([report_for()], genome=repo_genome())

    paths = report.write(tmp_path)

    assert set(paths) == {"tendencies", "report", "genome"}
    tendencies = json.loads((tmp_path / "solidified_tendencies.json").read_text(encoding="utf-8"))
    assert tendencies[0]["variant_name"] == "repair_heavy"
    assert (tmp_path / "solidification_report.md").read_text(encoding="utf-8").startswith("# Selection Solidification Report")
    assert "regulatory_elements:" in (tmp_path / "solidified_genome.yaml").read_text(encoding="utf-8")


def test_solidify_cli_reads_structure_summary_and_writes_outputs(tmp_path: Path) -> None:
    summary = tmp_path / "structure_search_summary.json"
    output = tmp_path / "solidified"
    summary.write_text(
        json.dumps(
            {
                "task": "Fix failing tests.",
                "domain": "repo_repair",
                "selected_variant": "repair_heavy",
                "trials": [trial.as_dict() for trial in report_for().trials],
            }
        ),
        encoding="utf-8",
    )
    genome = tmp_path / "genome.yaml"
    genome.write_text(
        """
genes:
  - id: gene_repair_from_test_failures
    category: regeneration
    morphogen_affinity: [repair_pressure]
    encoded_response: [repair]
""",
        encoding="utf-8",
    )

    main(["solidify", "--structure-search", str(summary), "--genome-spec", str(genome), "--output", str(output)])

    assert (output / "solidified_tendencies.json").exists()
    assert (output / "solidification_report.md").exists()
    assert (output / "solidified_genome.yaml").exists()
    result = json.loads((output / "solidified_tendencies.json").read_text(encoding="utf-8"))
    assert result[0]["variant_name"] == "repair_heavy"

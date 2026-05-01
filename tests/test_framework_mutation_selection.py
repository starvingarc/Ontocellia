from __future__ import annotations

import json
from pathlib import Path

import yaml

from ontocellia.__main__ import main
from ontocellia.framework import (
    AgentGenome,
    CellPosition,
    ExtracellularMatrix,
    Gene,
    MatrixRecord,
    MutationCandidateGenerator,
    MutationSelectionRuntime,
    OrganValidationResult,
    TaskMicroenvironment,
    load_agent_genome,
)


def make_genome() -> AgentGenome:
    return AgentGenome(
        genes=[
            Gene("gene_repair", "regeneration", ["repair_pressure"], ["repair"], suppression_cues=["broad rewrites"]),
            Gene("gene_review", "verification", ["review_pressure"], ["review"]),
            Gene("gene_memory", "memory", ["memory_pressure"], ["remember"]),
        ]
    )


def failed_validation() -> list[OrganValidationResult]:
    return [OrganValidationResult("pytest", False, 0.2, "repo", "3 failing regression tests", 0.1, 0.3, 1.0)]


def passing_validation() -> list[OrganValidationResult]:
    return [OrganValidationResult("pytest", True, 0.9, "repo", "tests passed", 0.1, 0.1, 1.0)]


def test_failed_validation_generates_repair_and_reviewer_candidates() -> None:
    candidates = MutationCandidateGenerator().generate(make_genome(), failed_validation())

    assert {candidate.source_gene_id for candidate in candidates} == {"gene_repair", "gene_review"}
    repair = next(candidate for candidate in candidates if candidate.source_gene_id == "gene_repair")
    reviewer = next(candidate for candidate in candidates if candidate.source_gene_id == "gene_review")
    assert "validation_pressure" in repair.changed_fields["morphogen_affinity"]
    assert "test_failure" in repair.changed_fields["morphogen_affinity"]
    assert "risk_pressure" in reviewer.changed_fields["morphogen_affinity"]
    assert any("3 failing regression tests" in item for item in repair.changed_fields["suppression_cues"])


def test_candidate_mutation_is_shallow_and_original_genome_is_unchanged() -> None:
    genome = make_genome()
    candidate = MutationCandidateGenerator().generate(genome, failed_validation())[0]

    mutated = candidate.to_lineage_mutation("lineage-1")
    new_genome = genome.mutate(mutated)

    assert genome.gene_by_id("gene_repair").morphogen_affinity == ["repair_pressure"]
    assert new_genome.gene_by_id(candidate.source_gene_id).morphogen_affinity == candidate.changed_fields["morphogen_affinity"]
    assert new_genome is not genome


def test_validation_improvement_selects_and_solidifies_mutation() -> None:
    genome = make_genome()
    candidates = MutationCandidateGenerator().generate(genome, failed_validation())

    report = MutationSelectionRuntime().select(genome, candidates, failed_validation(), passing_validation())

    assert report.selected_candidate is not None
    assert report.solidified_genome is not genome
    assert report.solidified_genome.mutation_history
    assert report.decision == "selected"
    assert report.candidate_score > report.baseline_score


def test_no_validation_improvement_keeps_original_genome() -> None:
    genome = make_genome()
    candidates = MutationCandidateGenerator().generate(genome, failed_validation())

    report = MutationSelectionRuntime().select(genome, candidates, passing_validation(), failed_validation())

    assert report.selected_candidate is None
    assert report.solidified_genome is genome
    assert report.decision == "not_selected"


def test_matrix_evidence_is_included_in_candidate_evidence() -> None:
    environment = TaskMicroenvironment(
        "Fix tests",
        matrix=ExtracellularMatrix(
            [
                MatrixRecord(
                    "m1",
                    1,
                    "observation",
                    "Failure happens around parser tests.",
                    ["test_failure", "parser"],
                    CellPosition("repair-niche"),
                    0.8,
                    1,
                )
            ]
        ),
    )

    candidates = MutationCandidateGenerator().generate(make_genome(), failed_validation(), environment=environment)

    assert any("parser" in item for item in candidates[0].evidence)
    assert any("Failure happens" in item for item in candidates[0].evidence)


def test_mutation_report_contains_scores_and_reason() -> None:
    genome = make_genome()
    candidates = MutationCandidateGenerator().generate(genome, failed_validation())

    report = MutationSelectionRuntime().select(genome, candidates, failed_validation(), passing_validation())
    data = report.as_dict()

    assert data["baseline_score"] == report.baseline_score
    assert data["candidate_score"] == report.candidate_score
    assert data["decision_reason"]
    assert data["selected_candidate"]["source_gene_id"] in {"gene_repair", "gene_review"}


def test_mutate_cli_writes_candidates_report_and_solidified_genome(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "mutation"
    baseline.write_text(json.dumps([result.as_dict() for result in failed_validation()]), encoding="utf-8")
    candidate.write_text(json.dumps([result.as_dict() for result in passing_validation()]), encoding="utf-8")

    main(
        [
            "mutate",
            "--genome-spec",
            "examples/framework/repo_repair_genome.yaml",
            "--environment-spec",
            "examples/framework/failing_tests_environment.yaml",
            "--baseline-validation",
            str(baseline),
            "--candidate-validation",
            str(candidate),
            "--output",
            str(output),
        ]
    )

    assert (output / "mutation_candidates.json").exists()
    assert (output / "mutation_report.md").exists()
    solidified = yaml.safe_load((output / "solidified_genome.yaml").read_text(encoding="utf-8"))
    assert solidified["mutation_history"]
    assert load_agent_genome(output / "solidified_genome.yaml").gene_by_id("gene_repair_from_test_failures")

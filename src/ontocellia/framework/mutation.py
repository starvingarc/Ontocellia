from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ontocellia.framework.core import TaskMicroenvironment
from ontocellia.framework.genome import AgentGenome, Gene, LineageMutation
from ontocellia.framework.selection import OrganValidationResult


@dataclass(slots=True)
class MutationCandidate:
    source_gene_id: str
    changed_fields: dict[str, Any]
    objective: str
    evidence: list[str]
    expected_effect: str
    confidence: float = 0.5

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_lineage_mutation(self, lineage_id: str, validation_result: dict[str, Any] | None = None) -> LineageMutation:
        return LineageMutation(
            source_gene_id=self.source_gene_id,
            changed_fields=dict(self.changed_fields),
            objective=self.objective,
            validation_result=dict(validation_result or {}),
            lineage_id=lineage_id,
        )


@dataclass(slots=True)
class MutationSelectionReport:
    baseline_score: float
    baseline_pass_rate: float
    candidate_score: float
    candidate_pass_rate: float
    decision: str
    decision_reason: str
    candidates: list[MutationCandidate]
    selected_candidate: MutationCandidate | None
    solidified_genome: AgentGenome

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline_score": self.baseline_score,
            "baseline_pass_rate": self.baseline_pass_rate,
            "candidate_score": self.candidate_score,
            "candidate_pass_rate": self.candidate_pass_rate,
            "decision": self.decision,
            "decision_reason": self.decision_reason,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "selected_candidate": self.selected_candidate.as_dict() if self.selected_candidate else None,
        }


class MutationCandidateGenerator:
    def generate(
        self,
        genome: AgentGenome,
        validation_results: list[OrganValidationResult],
        environment: TaskMicroenvironment | None = None,
    ) -> list[MutationCandidate]:
        if not _has_failure_signal(validation_results):
            return []
        evidence = _validation_evidence(validation_results)
        if environment is not None:
            evidence.extend(_matrix_evidence(environment))
        candidates: list[MutationCandidate] = []
        for gene in genome.genes:
            if gene.fate_bias == "repair":
                candidates.append(_candidate_for_gene(gene, ["validation_pressure", "test_failure", "repair_pressure"], evidence))
            elif gene.fate_bias == "reviewer":
                candidates.append(_candidate_for_gene(gene, ["risk_pressure", "validation_pressure"], evidence))
        candidates.sort(key=lambda candidate: (-candidate.confidence, candidate.source_gene_id))
        return candidates


class MutationSelectionRuntime:
    def select(
        self,
        genome: AgentGenome,
        candidates: list[MutationCandidate],
        baseline_validation: list[OrganValidationResult],
        candidate_validation: list[OrganValidationResult],
    ) -> MutationSelectionReport:
        baseline_score = _validation_score(baseline_validation)
        baseline_pass_rate = _pass_rate(baseline_validation)
        candidate_score = _validation_score(candidate_validation)
        candidate_pass_rate = _pass_rate(candidate_validation)
        improved = candidate_score > baseline_score or candidate_pass_rate > baseline_pass_rate
        selected = candidates[0] if improved and candidates else None
        if selected is None:
            return MutationSelectionReport(
                baseline_score,
                baseline_pass_rate,
                candidate_score,
                candidate_pass_rate,
                "not_selected",
                "Candidate validation did not improve over baseline." if candidates else "No mutation candidates were generated.",
                candidates,
                None,
                genome,
            )
        mutation = selected.to_lineage_mutation(
            lineage_id=f"mutation-{selected.source_gene_id}",
            validation_result={
                "baseline_score": baseline_score,
                "baseline_pass_rate": baseline_pass_rate,
                "candidate_score": candidate_score,
                "candidate_pass_rate": candidate_pass_rate,
            },
        )
        return MutationSelectionReport(
            baseline_score,
            baseline_pass_rate,
            candidate_score,
            candidate_pass_rate,
            "selected",
            "Candidate validation improved over baseline.",
            candidates,
            selected,
            genome.mutate(mutation),
        )


def write_mutation_outputs(report: MutationSelectionReport, output: str | Path) -> dict[str, Path]:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "candidates": output_path / "mutation_candidates.json",
        "report": output_path / "mutation_report.md",
        "genome": output_path / "solidified_genome.yaml",
    }
    import json

    paths["candidates"].write_text(json.dumps([candidate.as_dict() for candidate in report.candidates], indent=2, sort_keys=True), encoding="utf-8")
    paths["report"].write_text(_report_markdown(report), encoding="utf-8")
    paths["genome"].write_text(yaml.safe_dump(genome_to_dict(report.solidified_genome), sort_keys=False), encoding="utf-8")
    return paths


def genome_to_dict(genome: AgentGenome) -> dict[str, Any]:
    return {
        "metadata": dict(genome.metadata),
        "genes": [_gene_to_dict(gene) for gene in genome.genes],
        "regulatory_elements": [
            {
                "id": element.id,
                "kind": element.kind,
                "target_gene_id": element.target_gene_id,
                "signals": list(element.signals),
                "strength": element.strength,
            }
            for element in genome.regulatory_elements
        ],
        "epigenetic_defaults": {
            "fate_locks": dict(genome.epigenetic_defaults.fate_locks),
            "gene_locks": dict(genome.epigenetic_defaults.gene_locks),
        },
        "mutation_history": [asdict(mutation) for mutation in genome.mutation_history],
    }


def _gene_to_dict(gene: Gene) -> dict[str, Any]:
    return {
        "type": "Gene",
        "id": gene.id,
        "category": gene.category,
        "morphogen_affinity": list(gene.morphogen_affinity),
        "expression_window": list(gene.expression_window),
        "encoded_response": list(gene.encoded_response),
        "inhibitors": list(gene.inhibitors),
        "suppression_cues": list(gene.suppression_cues),
        "constraints": dict(gene.constraints),
        "validation_hooks": list(gene.validation_hooks),
        "heritability": dict(gene.heritability),
    }


def _candidate_for_gene(gene: Gene, affinity: list[str], evidence: list[str]) -> MutationCandidate:
    return MutationCandidate(
        source_gene_id=gene.id,
        changed_fields={
            "morphogen_affinity": _append_unique(gene.morphogen_affinity, affinity),
            "suppression_cues": _append_unique(gene.suppression_cues, [_failure_summary(evidence)]),
        },
        objective=f"Improve {gene.fate_bias} response under failed validation pressure.",
        evidence=list(evidence),
        expected_effect=f"Increase {gene.fate_bias} expression when validation failures recur.",
        confidence=0.75 if gene.fate_bias == "repair" else 0.65,
    )


def _append_unique(existing: list[str], additions: list[str]) -> list[str]:
    return list(dict.fromkeys([*existing, *[item for item in additions if item]]))


def _has_failure_signal(results: list[OrganValidationResult]) -> bool:
    keywords = ("test", "failure", "failing", "regression")
    return any(not result.passed and any(keyword in result.evidence.lower() for keyword in keywords) for result in results)


def _validation_evidence(results: list[OrganValidationResult]) -> list[str]:
    return [f"{result.name}: {result.evidence}" for result in results if result.evidence]


def _matrix_evidence(environment: TaskMicroenvironment) -> list[str]:
    evidence: list[str] = []
    for record in environment.matrix.records:
        if record.tags:
            evidence.append(f"matrix tags: {', '.join(record.tags)}")
        if record.content:
            evidence.append(f"matrix evidence: {record.content}")
    return evidence


def _failure_summary(evidence: list[str]) -> str:
    if not evidence:
        return "repeated validation failure"
    return evidence[0][:180]


def _validation_score(results: list[OrganValidationResult]) -> float:
    if not results:
        return 0.0
    return sum(max(0.0, min(1.0, result.score)) for result in results) / len(results)


def _pass_rate(results: list[OrganValidationResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for result in results if result.passed) / len(results)


def _report_markdown(report: MutationSelectionReport) -> str:
    selected = report.selected_candidate.source_gene_id if report.selected_candidate else "none"
    return "\n".join(
        [
            "# Mutation Selection Report",
            "",
            f"- Decision: `{report.decision}`",
            f"- Reason: {report.decision_reason}",
            f"- Baseline score: {report.baseline_score:.3f}",
            f"- Baseline pass rate: {report.baseline_pass_rate:.3f}",
            f"- Candidate score: {report.candidate_score:.3f}",
            f"- Candidate pass rate: {report.candidate_pass_rate:.3f}",
            f"- Selected mutation: `{selected}`",
            f"- Candidates: {len(report.candidates)}",
            "",
        ]
    )

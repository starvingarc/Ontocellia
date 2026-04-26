from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import yaml

from ontocellia.config import GeneAsset, GeneKind
from ontocellia.framework.cell import CellPosition
from ontocellia.framework.core import ExtracellularInterface, MorphogenField, Niche, TaskMicroenvironment
from ontocellia.framework.genome import AgentGenome, EpigeneticMarks, Gene, RegulatoryElement


@dataclass(slots=True)
class InductionRequest:
    task: str
    domain: str = "repo_repair"
    constraints: dict[str, object] = field(default_factory=dict)
    available_interfaces: list[str] = field(default_factory=list)
    seed: int = 7


@dataclass(slots=True)
class InductionDraft:
    genome: AgentGenome
    environment: TaskMicroenvironment
    gene_assets: list[GeneAsset] = field(default_factory=list)
    experiment: dict[str, object] | None = None
    rationale: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def write(self, output: str | Path) -> dict[str, Path]:
        return InductionWriter().write(self, output)


class InductionCompiler(Protocol):
    def compile(self, request: InductionRequest) -> InductionDraft:
        ...


class TemplateInductionCompiler:
    """Deterministic task-to-tissue induction compiler."""

    def compile(self, request: InductionRequest) -> InductionDraft:
        domain = request.domain.lower().replace("-", "_")
        if domain == "repo_repair":
            return self._repo_repair(request)
        if domain == "research_synthesis":
            return self._research_synthesis(request)
        return self._generic(request)

    def _repo_repair(self, request: InductionRequest) -> InductionDraft:
        interfaces = _interfaces(request.available_interfaces, ["workspace", "pytest", "git"])
        genome = AgentGenome(
            metadata={"name": "induced-repo-repair-genome", "version": "0.1.0"},
            genes=[
                Gene("gene_inspect_context", "exploration", ["ambiguity", "missing_context"], ["Inspect failing logs and related files."]),
                Gene(
                    "gene_repair_from_test_failures",
                    "regeneration",
                    ["test_failure", "repair_pressure", "niche_vacancy"],
                    ["Patch the narrowest cause.", "Rerun validation hooks."],
                    suppression_cues=["broad rewrites", "suppressing tests"],
                    constraints={"max_files": 5},
                    validation_hooks=["python -m pytest -q"],
                ),
                Gene("gene_review_boundary", "verification", ["review_pressure", "risk"], ["Check blast radius and evidence."], validation_hooks=["git diff --check"]),
                Gene("gene_record_lineage_memory", "memory", ["coordination_pressure", "repair_pressure"], ["Record hypotheses, failures, and validation evidence."]),
            ],
            regulatory_elements=[
                RegulatoryElement("promoter_repair_from_failure", "promoter", "gene_repair_from_test_failures", ["test_failure"], 0.25),
                RegulatoryElement("enhancer_review_under_risk", "enhancer", "gene_review_boundary", ["review_pressure"], 0.15),
            ],
            epigenetic_defaults=EpigeneticMarks(fate_locks={"repair": 0.2, "reviewer": 0.15}),
        )
        environment = TaskMicroenvironment(
            objective=request.task,
            morphogens=MorphogenField(
                {"ambiguity": 0.75, "missing_context": 0.55, "test_failure": 0.9, "repair_pressure": 0.75, "review_pressure": 0.55}
            ),
            niches=[
                Niche("exploration-front", "explorer", CellPosition("exploration-front", "repo", ["repair-niche"], (2.0, 8.0, 0.0))),
                Niche("repair-niche", "repair", CellPosition("repair-niche", "repo", ["exploration-front", "review-boundary"], (10.0, 4.0, 0.0)), demand=2),
                Niche("review-boundary", "reviewer", CellPosition("review-boundary", "repo", ["repair-niche"], (7.0, 7.0, 0.0))),
                Niche("memory-niche", "memory", CellPosition("memory-niche", "repo", ["repair-niche"], (5.0, 5.0, 1.0))),
            ],
            interfaces=[
                ExtracellularInterface(interface_id, _interface_kind(interface_id), _accepted_fates(interface_id))
                for interface_id in interfaces
            ],
        )
        return InductionDraft(
            genome=genome,
            environment=environment,
            gene_assets=[
                GeneAsset(
                    kind=GeneKind.STRATEGY,
                    name="repo-repair-induction",
                    signals=["test_failure", "repair_pressure"],
                    summary="Culture protocol for inducing a repair tissue around failing validation signals.",
                    validation_hooks=["python -m pytest -q"],
                    provenance="induction_compiler",
                )
            ],
            experiment=_experiment("repo-repair-induction", request.seed),
            rationale=["Detected repo repair domain.", "Generated repair, exploration, review, and memory niches."],
            warnings=[],
        )

    def _research_synthesis(self, request: InductionRequest) -> InductionDraft:
        interfaces = _interfaces(request.available_interfaces, ["web", "workspace", "citation_store"])
        genome = AgentGenome(
            metadata={"name": "induced-research-synthesis-genome", "version": "0.1.0"},
            genes=[
                Gene("gene_search_sources", "exploration", ["ambiguity", "citation_pressure"], ["Search and collect candidate sources."]),
                Gene("gene_verify_claims", "verification", ["verification_pressure", "citation_pressure"], ["Check claims against sources."]),
                Gene("gene_synthesize_report", "implementation", ["synthesis_pressure"], ["Produce a structured synthesis."]),
                Gene("gene_critique_summary", "review", ["verification_pressure"], ["Critique coverage and uncertainty."]),
            ],
        )
        environment = TaskMicroenvironment(
            objective=request.task,
            morphogens=MorphogenField({"ambiguity": 0.8, "citation_pressure": 0.75, "verification_pressure": 0.65, "synthesis_pressure": 0.7}),
            niches=[
                Niche("scout-niche", "explorer", CellPosition("scout-niche", "research", ["verifier-niche"], (2.0, 6.0, 0.0))),
                Niche("verifier-niche", "reviewer", CellPosition("verifier-niche", "research", ["scout-niche", "synthesizer-niche"], (5.0, 6.0, 0.0))),
                Niche("synthesizer-niche", "builder", CellPosition("synthesizer-niche", "research", ["verifier-niche"], (8.0, 6.0, 0.0))),
                Niche("critic-niche", "reviewer", CellPosition("critic-niche", "research", ["synthesizer-niche"], (8.0, 3.0, 0.0))),
            ],
            interfaces=[ExtracellularInterface(interface_id, _interface_kind(interface_id), _accepted_fates(interface_id)) for interface_id in interfaces],
        )
        return InductionDraft(genome=genome, environment=environment, gene_assets=[], experiment=_experiment("research-synthesis-induction", request.seed), rationale=["Detected research synthesis domain."], warnings=[])

    def _generic(self, request: InductionRequest) -> InductionDraft:
        interfaces = _interfaces(request.available_interfaces, ["workspace"])
        genome = AgentGenome(
            metadata={"name": "induced-generic-agent-tissue-genome", "version": "0.1.0"},
            genes=[
                Gene("gene_explore_task", "exploration", ["ambiguity"], ["Inspect task context."]),
                Gene("gene_plan_task", "planning", ["ambiguity", "coordination_pressure"], ["Draft a task decomposition."]),
                Gene("gene_build_output", "implementation", ["implementation_pressure"], ["Produce candidate output."]),
                Gene("gene_review_output", "review", ["verification_pressure"], ["Review evidence and risk."]),
                Gene("gene_remember_context", "memory", ["coordination_pressure"], ["Record assumptions and lineage evidence."]),
            ],
        )
        environment = TaskMicroenvironment(
            objective=request.task,
            morphogens=MorphogenField({"ambiguity": 0.7, "implementation_pressure": 0.55, "verification_pressure": 0.55, "coordination_pressure": 0.45}),
            niches=[
                Niche("exploration-front", "explorer", CellPosition("exploration-front", "generic", ["builder-niche"], (2.0, 5.0, 0.0))),
                Niche("builder-niche", "builder", CellPosition("builder-niche", "generic", ["review-boundary"], (5.0, 5.0, 0.0))),
                Niche("review-boundary", "reviewer", CellPosition("review-boundary", "generic", ["builder-niche"], (8.0, 5.0, 0.0))),
                Niche("memory-niche", "memory", CellPosition("memory-niche", "generic", ["builder-niche"], (5.0, 2.0, 0.0))),
            ],
            interfaces=[ExtracellularInterface(interface_id, _interface_kind(interface_id), _accepted_fates(interface_id)) for interface_id in interfaces],
        )
        return InductionDraft(genome=genome, environment=environment, gene_assets=[], experiment=_experiment("generic-induction", request.seed), rationale=["Used generic induction template."], warnings=[f"Unknown domain: {request.domain}"])


class InductionWriter:
    def write(self, draft: InductionDraft, output: str | Path) -> dict[str, Path]:
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        paths = {
            "genome": output_path / "genome.yaml",
            "environment": output_path / "environment.yaml",
            "report": output_path / "induction_report.md",
            "experiment": output_path / "experiment.yaml",
        }
        paths["genome"].write_text(yaml.safe_dump(_genome_data(draft.genome), sort_keys=False), encoding="utf-8")
        paths["environment"].write_text(yaml.safe_dump(_environment_data(draft.environment), sort_keys=False), encoding="utf-8")
        paths["report"].write_text(_report(draft), encoding="utf-8")
        if draft.experiment is not None:
            paths["experiment"].write_text(yaml.safe_dump(draft.experiment, sort_keys=False), encoding="utf-8")
        return paths


def _interfaces(available: list[str], defaults: list[str]) -> list[str]:
    return list(dict.fromkeys(available or defaults))


def _interface_kind(interface_id: str) -> str:
    if interface_id in {"workspace", "citation_store"}:
        return "extracellular_matrix"
    if interface_id in {"web"}:
        return "extracellular_niche"
    return "membrane_channel"


def _accepted_fates(interface_id: str) -> list[str]:
    mapping = {
        "pytest": ["repair", "reviewer"],
        "git": ["reviewer", "repair"],
        "workspace": ["explorer", "repair", "builder", "memory"],
        "web": ["explorer", "reviewer"],
        "citation_store": ["memory", "reviewer", "builder"],
    }
    return mapping.get(interface_id, ["explorer", "builder", "reviewer", "repair", "memory"])


def _experiment(name: str, seed: int) -> dict[str, object]:
    return {
        "metadata": {"name": name},
        "base": {"genome": "genome.yaml", "environment": "environment.yaml", "steps": 8, "seed": seed},
        "variants": [{"name": "baseline"}],
        "outputs": {"summary": True, "report": True},
    }


def _genome_data(genome: AgentGenome) -> dict[str, object]:
    return {
        "metadata": genome.metadata,
        "genes": [
            {
                "type": "Gene",
                "id": gene.id,
                "category": gene.category,
                "morphogen_affinity": gene.morphogen_affinity,
                "expression_window": gene.expression_window,
                "encoded_response": gene.encoded_response,
                "inhibitors": gene.inhibitors,
                "suppression_cues": gene.suppression_cues,
                "constraints": gene.constraints,
                "validation_hooks": gene.validation_hooks,
                "heritability": gene.heritability,
            }
            for gene in genome.genes
        ],
        "regulatory_elements": [
            {
                "id": element.id,
                "kind": element.kind,
                "target_gene_id": element.target_gene_id,
                "signals": element.signals,
                "strength": element.strength,
            }
            for element in genome.regulatory_elements
        ],
        "epigenetic_defaults": {
            "fate_locks": genome.epigenetic_defaults.fate_locks,
            "gene_locks": genome.epigenetic_defaults.gene_locks,
        },
    }


def _environment_data(environment: TaskMicroenvironment) -> dict[str, object]:
    return {
        "task": {"objective": environment.objective},
        "morphogens": environment.morphogens.signals,
        "niches": [
            {
                "id": niche.id,
                "required_fate": niche.required_fate,
                "position": {
                    "node_id": niche.position.node_id,
                    "region": niche.position.region,
                    "neighbors": niche.position.neighbors,
                    "embedding": list(niche.position.embedding),
                },
                "demand": niche.demand,
            }
            for niche in environment.niches
        ],
        "interfaces": [
            {
                "id": interface.id,
                "kind": interface.kind,
                "accepts_fates": interface.accepts_fates,
                "metadata": interface.metadata,
            }
            for interface in environment.interfaces
        ],
    }


def _report(draft: InductionDraft) -> str:
    lines = [
        "# Induction Report",
        "",
        f"Objective: {draft.environment.objective}",
        "",
        "## Rationale",
        *[f"- {item}" for item in draft.rationale],
        "",
        "## Warnings",
        *[f"- {item}" for item in draft.warnings or ["None"]],
        "",
        "## Generated Tissue",
        f"- Genes: {len(draft.genome.genes)}",
        f"- Niches: {len(draft.environment.niches)}",
        f"- Interfaces: {len(draft.environment.interfaces)}",
    ]
    return "\n".join(lines) + "\n"

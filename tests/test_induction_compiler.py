from __future__ import annotations

from pathlib import Path

from ontocellia.__main__ import main
from ontocellia.framework import (
    InductionRequest,
    TemplateInductionCompiler,
    TissueRuntime,
    load_agent_genome,
    load_task_microenvironment,
)


def test_repo_repair_induction_generates_repair_tissue_specs() -> None:
    compiler = TemplateInductionCompiler()
    draft = compiler.compile(
        InductionRequest(
            task="Fix failing tests while preserving behavior",
            domain="repo_repair",
            available_interfaces=["workspace", "pytest", "git"],
        )
    )

    gene_ids = {gene.id for gene in draft.genome.genes}
    niche_ids = {niche.id for niche in draft.environment.niches}

    assert {"gene_inspect_context", "gene_repair_from_test_failures", "gene_review_boundary"}.issubset(gene_ids)
    assert {"exploration-front", "repair-niche", "review-boundary"}.issubset(niche_ids)
    assert draft.environment.morphogens.signal("repair_pressure") > 0
    assert any(asset.name == "repo-repair-induction" for asset in draft.gene_assets)
    assert draft.experiment is not None


def test_research_induction_generates_synthesis_tissue_specs() -> None:
    compiler = TemplateInductionCompiler()
    draft = compiler.compile(
        InductionRequest(
            task="Research recent approaches and synthesize a cited summary",
            domain="research_synthesis",
            available_interfaces=["web", "workspace", "citation_store"],
        )
    )

    assert {gene.fate_bias for gene in draft.genome.genes} >= {"explorer", "reviewer"}
    assert draft.environment.morphogens.signal("citation_pressure") > 0
    assert "synthesizer-niche" in {niche.id for niche in draft.environment.niches}


def test_induction_writer_outputs_loadable_specs_and_report(tmp_path: Path) -> None:
    compiler = TemplateInductionCompiler()
    draft = compiler.compile(InductionRequest(task="Fix failing tests", domain="repo_repair"))

    paths = draft.write(tmp_path)
    genome = load_agent_genome(paths["genome"])
    environment = load_task_microenvironment(paths["environment"])
    tissue = TissueRuntime.seeded(genome=genome, environment=environment, stem_cells=5, seed=3)
    tissue.develop(ticks=4)

    assert paths["report"].exists()
    assert paths["experiment"].exists()
    assert tissue.niche_occupancy()["repair-niche"] >= 1


def test_induce_cli_writes_specs_that_tissue_cli_can_run(tmp_path: Path) -> None:
    output = tmp_path / "induced"
    tissue_output = tmp_path / "tissue"

    main(["induce", "--task", "Fix failing tests", "--domain", "repo_repair", "--output", str(output)])
    main(
        [
            "tissue",
            "--genome-spec",
            str(output / "genome.yaml"),
            "--environment-spec",
            str(output / "environment.yaml"),
            "--steps",
            "4",
            "--output",
            str(tissue_output),
        ]
    )

    assert (output / "genome.yaml").exists()
    assert (output / "environment.yaml").exists()
    assert (output / "induction_report.md").exists()
    assert (tissue_output / "tissue_summary.json").exists()

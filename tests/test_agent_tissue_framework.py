from __future__ import annotations

from ontocellia.framework import (
    AgentGenome,
    ExtracellularInterface,
    Gene,
    MorphogenField,
    Niche,
    TaskMicroenvironment,
    TissueRuntime,
)


def make_repo_repair_tissue() -> TissueRuntime:
    genome = AgentGenome(
        genes=[
            Gene(
                id="gene_inspect_context",
                category="exploration",
                morphogen_affinity=["ambiguity", "missing_context"],
                encoded_response=["inspect nearby evidence", "emit structured findings"],
                validation_hooks=[],
            ),
            Gene(
                id="gene_repair_from_test_failures",
                category="regeneration",
                morphogen_affinity=["test_failure", "repair_pressure", "niche_vacancy"],
                encoded_response=["patch narrow cause", "rerun validation"],
                validation_hooks=["python -m pytest -q"],
            ),
            Gene(
                id="gene_review_boundary",
                category="verification",
                morphogen_affinity=["review_pressure", "risk"],
                encoded_response=["check blast radius", "validate evidence"],
                validation_hooks=["git diff --check"],
            ),
        ]
    )
    environment = TaskMicroenvironment(
        objective="Fix failing tests while preserving existing behavior.",
        morphogens=MorphogenField(
            signals={
                "ambiguity": 0.8,
                "missing_context": 0.5,
                "test_failure": 0.9,
                "repair_pressure": 0.7,
                "review_pressure": 0.55,
            }
        ),
        niches=[
            Niche(id="repair-niche", required_fate="repair", position=(10.0, 4.0), demand=2),
            Niche(id="exploration-front", required_fate="explorer", position=(2.0, 8.0), demand=1),
            Niche(id="review-boundary", required_fate="reviewer", position=(7.0, 7.0), demand=1),
        ],
        interfaces=[
            ExtracellularInterface(id="pytest", kind="membrane_channel", accepts_fates=["repair", "reviewer"]),
            ExtracellularInterface(id="workspace", kind="extracellular_matrix", accepts_fates=["explorer", "repair"]),
        ],
    )
    return TissueRuntime.seeded(genome=genome, environment=environment, stem_cells=5, seed=11)


def test_task_induces_tissue_with_positioned_functional_domains() -> None:
    tissue = make_repo_repair_tissue()

    tissue.develop(ticks=4)

    assert tissue.fate_counts()["repair"] >= 2
    assert tissue.fate_counts()["explorer"] >= 1
    assert tissue.fate_counts()["reviewer"] >= 1
    assert tissue.niche_occupancy()["repair-niche"] >= 2
    assert tissue.niche_occupancy()["exploration-front"] >= 1


def test_cleared_repair_cell_is_replaced_by_stem_progenitor_lineage() -> None:
    tissue = make_repo_repair_tissue()
    tissue.develop(ticks=4)
    removed = next(cell.id for cell in tissue.cells.values() if cell.fate == "repair")

    tissue.clear_cell(removed, reason="manual_clear")
    tissue.develop(ticks=3)

    replacement_events = [
        event
        for event in tissue.trace.events
        if event["type"] == "regeneration" and event["replaced_cell_id"] == removed
    ]
    assert replacement_events
    assert replacement_events[-1]["source_stage"] in {"stem", "progenitor", "transit_amplifying"}
    assert tissue.niche_occupancy()["repair-niche"] >= 2


def test_expressed_genes_use_membrane_channels_without_encoding_external_tools_as_genes() -> None:
    tissue = make_repo_repair_tissue()
    tissue.develop(ticks=4)

    actions = tissue.execute()

    assert any(action["interface_id"] == "pytest" for action in actions)
    assert any(action["gene_id"] == "gene_repair_from_test_failures" for action in actions)
    assert "pytest" not in {gene.id for gene in tissue.genome.genes}

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


genome = AgentGenome(
    genes=[
        Gene(
            id="gene_inspect_context",
            category="exploration",
            morphogen_affinity=["ambiguity", "missing_context"],
            encoded_response=["inspect nearby evidence", "emit structured findings"],
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

tissue = TissueRuntime.seeded(genome=genome, environment=environment, stem_cells=5, seed=11)
tissue.develop(ticks=4)

print("fates", tissue.fate_counts())
print("niches", tissue.niche_occupancy())
print("actions", tissue.execute())

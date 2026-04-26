from __future__ import annotations

from pathlib import Path

import pytest

from ontocellia.framework import (
    AgentGenome,
    EpigeneticMarks,
    ExpressionContext,
    Gene,
    LineageMutation,
    RegulatoryElement,
    load_agent_genome,
)


def make_genome() -> AgentGenome:
    return AgentGenome(
        genes=[
            Gene(
                id="gene_a_explore",
                category="exploration",
                morphogen_affinity=["ambiguity"],
                encoded_response=["inspect context"],
            ),
            Gene(
                id="gene_b_repair",
                category="regeneration",
                morphogen_affinity=["test_failure", "repair_pressure"],
                encoded_response=["repair narrow cause"],
                inhibitors=["low_confidence"],
            ),
            Gene(
                id="gene_c_review",
                category="verification",
                morphogen_affinity=["review_pressure"],
                encoded_response=["review risk"],
            ),
        ],
        regulatory_elements=[
            RegulatoryElement(id="promote-repair", kind="promoter", target_gene_id="gene_b_repair", signals=["test_failure"], strength=0.5),
            RegulatoryElement(id="silence-review", kind="silencer", target_gene_id="gene_c_review", signals=["review_pressure"], strength=0.8),
        ],
        epigenetic_defaults=EpigeneticMarks(fate_locks={"repair": 0.4}),
    )


def test_load_agent_genome_reads_regulatory_elements_and_epigenetic_defaults(tmp_path: Path) -> None:
    path = tmp_path / "genome.yaml"
    path.write_text(
        """
metadata:
  name: regulated-genome
genes:
  - id: gene_repair
    category: regeneration
    morphogen_affinity: [test_failure]
    encoded_response: [repair]
regulatory_elements:
  - id: promoter_repair
    kind: promoter
    target_gene_id: gene_repair
    signals: [test_failure]
    strength: 0.25
epigenetic_defaults:
  fate_locks:
    repair: 0.35
  gene_locks:
    gene_repair: 0.1
""",
        encoding="utf-8",
    )

    genome = load_agent_genome(path)

    assert genome.metadata["name"] == "regulated-genome"
    assert genome.regulatory_elements[0].target_gene_id == "gene_repair"
    assert genome.epigenetic_defaults.fate_locks["repair"] == 0.35
    assert genome.epigenetic_defaults.gene_locks["gene_repair"] == 0.1


def test_load_agent_genome_rejects_unknown_regulatory_target(tmp_path: Path) -> None:
    path = tmp_path / "bad_genome.yaml"
    path.write_text(
        """
genes:
  - id: gene_repair
    category: regeneration
    morphogen_affinity: [test_failure]
    encoded_response: [repair]
regulatory_elements:
  - id: promoter_missing
    kind: promoter
    target_gene_id: gene_missing
    signals: [test_failure]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown target gene"):
        load_agent_genome(path)


def test_express_orders_repair_and_exploration_by_local_morphogens() -> None:
    genome = make_genome()

    repair_programs = genome.express(ExpressionContext(morphogens={"test_failure": 0.9, "repair_pressure": 0.4}, current_fate="repair"))
    explore_programs = genome.express(ExpressionContext(morphogens={"ambiguity": 1.0}, current_fate="explorer"))

    assert repair_programs[0].gene.id == "gene_b_repair"
    assert repair_programs[0].fate_bias == "repair"
    assert explore_programs[0].gene.id == "gene_a_explore"
    assert explore_programs[0].fate_bias == "explorer"


def test_express_uses_stable_gene_id_order_for_equal_scores() -> None:
    genome = AgentGenome(
        genes=[
            Gene(id="gene_b", category="memory", morphogen_affinity=["coordination"], encoded_response=["remember b"]),
            Gene(id="gene_a", category="memory", morphogen_affinity=["coordination"], encoded_response=["remember a"]),
        ]
    )

    programs = genome.express(ExpressionContext(morphogens={"coordination": 1.0}), limit=2)

    assert [program.gene.id for program in programs] == ["gene_a", "gene_b"]


def test_regulatory_elements_and_epigenetic_lock_shape_expression() -> None:
    genome = make_genome()
    programs = genome.express(
        ExpressionContext(
            morphogens={"test_failure": 0.9, "review_pressure": 0.9},
            current_fate="repair",
            epigenetic_marks=EpigeneticMarks(fate_locks={"repair": 0.7}),
        ),
        limit=3,
    )

    scores = {program.gene.id: program.score for program in programs}

    assert scores["gene_b_repair"] > 1.0
    assert "gene_c_review" not in scores


def test_low_energy_and_high_stress_suppress_high_cost_gene() -> None:
    genome = AgentGenome(
        genes=[
            Gene(
                id="gene_costly_builder",
                category="implementation",
                morphogen_affinity=["implementation_pressure"],
                encoded_response=["write patch"],
                constraints={"cost": 0.9},
            )
        ]
    )

    programs = genome.express(ExpressionContext(morphogens={"implementation_pressure": 0.6}, energy=0.15, stress=0.95))

    assert programs == []


def test_mutate_returns_new_genome_and_records_lineage_mutation() -> None:
    genome = make_genome()
    mutation = LineageMutation(
        source_gene_id="gene_b_repair",
        changed_fields={"suppression_cues": ["broad rewrite"]},
        objective="suppress unsafe repair",
        validation_result={"passed": True},
        lineage_id="lineage-7",
    )

    mutated = genome.mutate(mutation)

    assert mutated is not genome
    assert genome.gene_by_id("gene_b_repair").suppression_cues == []
    assert mutated.gene_by_id("gene_b_repair").suppression_cues == ["broad rewrite"]
    assert mutated.mutation_history[-1].lineage_id == "lineage-7"

from __future__ import annotations

from ontocellia.framework import (
    AdhesionProfile,
    AgentCell,
    AgentGenome,
    CellPosition,
    CompetenceProfile,
    DifferentiatedCellState,
    ExtracellularInterface,
    Gene,
    MorphogenField,
    Niche,
    ReceptorProfile,
    StemCellState,
    TissueRuntime,
)


def make_genome() -> AgentGenome:
    return AgentGenome(
        genes=[
            Gene(
                id="gene_repair",
                category="regeneration",
                morphogen_affinity=["repair_pressure", "niche_vacancy"],
                encoded_response=["repair"],
            ),
            Gene(
                id="gene_explore",
                category="exploration",
                morphogen_affinity=["ambiguity"],
                encoded_response=["explore"],
            ),
        ]
    )


def test_stem_cell_expression_context_contains_cell_state() -> None:
    cell = AgentCell(
        id=1,
        stage="stem",
        fate="stem",
        position=CellPosition(node_id="stem-reserve-1", region="stem-reserve", embedding=(1.0, 2.0, 3.0)),
        stage_state=StemCellState(plasticity=0.9, division_potential=0.8),
        competence=CompetenceProfile(fate_scores={"repair": 0.7}, plasticity=0.9),
    )
    cell.record_event("stress", damage=0.2)

    context = cell.expression_context({"repair_pressure": 0.8}, organ_feedback={"repair": 0.1})

    assert context.cell_stage == "stem"
    assert context.current_fate == "stem"
    assert context.energy == cell.energy
    assert context.competence["repair"] == 0.7
    assert context.lineage_history[-1]["type"] == "stress"
    assert context.epigenetic_marks.fate_locks == {}
    assert context.organ_feedback["repair"] == 0.1


def test_cell_position_supports_graph_topology_and_3d_embedding() -> None:
    position = CellPosition(node_id="repair-niche", region="repo/tests", neighbors=["review-boundary"], embedding=(1.0, 2.0, 3.0))

    assert position.node_id == "repair-niche"
    assert position.region == "repo/tests"
    assert position.neighbors == ["review-boundary"]
    assert position.embedding == (1.0, 2.0, 3.0)


def test_niche_tuple_position_is_normalized_to_three_dimensional_embedding() -> None:
    niche = Niche(id="repair-niche", required_fate="repair", position=(10.0, 4.0), demand=1)

    assert isinstance(niche.position, CellPosition)
    assert niche.position.node_id == "coord:10.0:4.0:0.0"
    assert niche.position.embedding == (10.0, 4.0, 0.0)


def test_spawn_child_increments_generation_and_preserves_lineage_root() -> None:
    parent = AgentCell(id=1, stage="stem", fate="stem", position=CellPosition(node_id="stem", embedding=(0.0, 0.0, 0.0)))

    child = parent.spawn_child(child_id=2, stage="progenitor", fate="repair", position=CellPosition(node_id="repair", embedding=(1.0, 0.0, 0.0)))

    assert child.lineage_parent == parent.id
    assert child.lineage.root_id == parent.id
    assert child.lineage.generation == 1
    assert child.stage == "progenitor"
    assert child.fate == "repair"
    assert child.receptor.signal_sensitivities == parent.receptor.signal_sensitivities


def test_progenitor_can_enter_transit_amplifying_and_differentiate() -> None:
    genome = make_genome()
    progenitor = AgentCell(id=1, stage="progenitor", fate="repair", position=CellPosition(node_id="repair", embedding=(1.0, 0.0, 0.0)))
    transit = progenitor.spawn_child(child_id=2, stage="transit_amplifying", fate="repair", position=progenitor.position)

    committed = transit.commit_to_fate("repair", "repair-niche", genome, MorphogenField({"repair_pressure": 1.0}))

    assert committed is transit
    assert committed.stage == "differentiated"
    assert isinstance(committed.stage_state, DifferentiatedCellState)
    assert committed.expressed_gene_ids == ["gene_repair"]
    assert committed.position.node_id == "repair-niche"


def test_differentiated_cell_division_and_reprogramming_rules() -> None:
    stable = AgentCell(
        id=1,
        stage="differentiated",
        fate="reviewer",
        position=CellPosition(node_id="review", embedding=(0.0, 0.0, 0.0)),
        stage_state=DifferentiatedCellState(fate_lock=0.6, reprogrammable=True),
    )

    assert not stable.can_divide()
    assert not stable.can_reprogram(0.5)
    assert stable.can_reprogram(1.5)


def test_receptor_profile_limits_extracellular_interfaces() -> None:
    cell = AgentCell(
        id=1,
        stage="differentiated",
        fate="repair",
        position=CellPosition(node_id="repair", embedding=(0.0, 0.0, 0.0)),
        expressed_gene_ids=["gene_repair"],
        receptor=ReceptorProfile(accepted_interfaces=["workspace"]),
    )
    runtime = TissueRuntime(
        genome=make_genome(),
        environment=type(
            "Environment",
            (),
            {
                "interfaces": [
                    ExtracellularInterface(id="pytest", kind="membrane_channel", accepts_fates=["repair"]),
                    ExtracellularInterface(id="workspace", kind="extracellular_matrix", accepts_fates=["repair"]),
                ]
            },
        )(),
        cells={1: cell},
    )

    actions = runtime.execute()

    assert [action["interface_id"] for action in actions] == ["workspace"]


def test_graph_neighbor_source_beats_far_embedding_for_replacement() -> None:
    genome = make_genome()
    niche = Niche(
        id="repair-niche",
        required_fate="repair",
        position=CellPosition(node_id="repair", region="repo", neighbors=["near-stem"], embedding=(100.0, 100.0, 100.0)),
        demand=1,
    )
    near = AgentCell(id=1, stage="stem", fate="stem", position=CellPosition(node_id="near-stem", embedding=(1000.0, 1000.0, 1000.0)))
    far = AgentCell(id=2, stage="stem", fate="stem", position=CellPosition(node_id="far-stem", embedding=(100.0, 100.0, 101.0)))
    runtime = TissueRuntime(
        genome=genome,
        environment=type("Environment", (), {"niches": [niche], "morphogens": MorphogenField({"repair_pressure": 1.0}), "interfaces": [], "niche_by_id": lambda _, niche_id: niche})(),
        cells={1: near, 2: far},
    )

    selected = runtime._select_plastic_cell(prefer_near=niche.position)

    assert selected.id == 1


def test_adhesion_profile_scores_compatible_neighbor_fates() -> None:
    cell = AgentCell(
        id=1,
        stage="differentiated",
        fate="repair",
        position=CellPosition(node_id="repair", embedding=(0.0, 0.0, 0.0)),
        adhesion=AdhesionProfile(compatible_fates=["memory"], strength=0.8),
    )

    assert cell.adhesion_score("repair") == 1.0
    assert cell.adhesion_score("memory") == 0.8
    assert cell.adhesion_score("reviewer") == 0.0

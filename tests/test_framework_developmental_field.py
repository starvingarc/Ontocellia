from __future__ import annotations

from pathlib import Path

from ontocellia.framework import (
    AgentCell,
    AgentGenome,
    CellPosition,
    FateAttractor,
    FateLandscape,
    Gene,
    MorphogenField,
    Niche,
    TaskMicroenvironment,
    TissueRuntime,
    TissueTopology,
    TopologyNode,
    load_task_microenvironment,
)


def make_genome() -> AgentGenome:
    return AgentGenome(
        genes=[
            Gene("gene_repair", "regeneration", ["test_failure", "repair_pressure", "niche_vacancy"], ["repair"]),
            Gene("gene_explore", "exploration", ["ambiguity", "missing_context"], ["inspect"]),
            Gene("gene_review", "verification", ["review_pressure"], ["review"]),
        ]
    )


def make_environment() -> TaskMicroenvironment:
    repair = TopologyNode("repair-niche", "repo/tests", ["stem-reserve", "review-boundary"], (10.0, 4.0, 0.0))
    stem = TopologyNode("stem-reserve", "repo/tests", ["repair-niche"], (8.0, 4.0, 0.0))
    review = TopologyNode("review-boundary", "repo/tests", ["repair-niche"], (12.0, 4.0, 0.0))
    return TaskMicroenvironment(
        objective="Fix tests",
        morphogens=MorphogenField({"ambiguity": 0.2, "test_failure": 0.2, "repair_pressure": 0.2}),
        niches=[
            Niche("repair-niche", "repair", CellPosition("repair-niche", "repo/tests", ["stem-reserve"], (10.0, 4.0, 0.0)), demand=2),
            Niche("review-boundary", "reviewer", CellPosition("review-boundary", "repo/tests", ["repair-niche"], (12.0, 4.0, 0.0)), demand=1),
        ],
        topology=TissueTopology(nodes={node.id: node for node in [repair, stem, review]}),
        fate_landscape=FateLandscape.default(),
    )


def test_morphogen_field_combines_global_and_local_source_signal() -> None:
    topology = TissueTopology(
        nodes={
            "repair": TopologyNode("repair", "repo", ["review"], (0.0, 0.0, 0.0)),
            "review": TopologyNode("review", "repo", ["repair"], (0.0, 10.0, 0.0)),
        }
    )
    field = MorphogenField({"repair_pressure": 0.5})
    field.emit_at("repair_pressure", 1.0, CellPosition("repair", "repo"), radius=2.0, source_id="local-repair")

    near = field.signal_at("repair_pressure", CellPosition("repair", "repo"), topology)
    neighbor = field.signal_at("repair_pressure", CellPosition("review", "repo"), topology)

    assert near == 1.5
    assert neighbor == 1.0


def test_morphogen_field_decay_updates_sources_deterministically() -> None:
    field = MorphogenField({"damage": 1.0})
    field.emit_at("damage", 0.001, CellPosition("repair", "repo"), source_id="tiny")
    field.emit_at("damage", 1.0, CellPosition("repair", "repo"), source_id="strong")

    field.decay(rate=0.5)

    assert field.signal("damage") == 0.5
    assert [source.id for source in field.sources] == ["strong"]
    assert field.sources[0].amount == 0.5


def test_tissue_topology_distance_prefers_graph_path_over_embedding() -> None:
    topology = TissueTopology(
        nodes={
            "a": TopologyNode("a", "repo", ["b"], (0.0, 0.0, 0.0)),
            "b": TopologyNode("b", "repo", ["a"], (100.0, 100.0, 100.0)),
            "c": TopologyNode("c", "other", [], (0.0, 0.0, 0.1)),
        }
    )

    assert topology.distance(CellPosition("a"), CellPosition("b")) == 1.0
    assert topology.distance(CellPosition("a"), CellPosition("c")) > 1.0


def test_tissue_topology_nearest_step_toward_is_deterministic() -> None:
    topology = TissueTopology(
        nodes={
            "a": TopologyNode("a", "repo", ["b", "c"], (0.0, 0.0, 0.0)),
            "b": TopologyNode("b", "repo", ["a", "d"], (1.0, 0.0, 0.0)),
            "c": TopologyNode("c", "repo", ["a", "d"], (0.0, 1.0, 0.0)),
            "d": TopologyNode("d", "repo", ["b", "c"], (2.0, 0.0, 0.0)),
        }
    )

    assert topology.nearest_step_toward(CellPosition("a"), CellPosition("d")).node_id == "b"


def test_fate_landscape_selects_repair_and_explorer_from_local_signals() -> None:
    genome = make_genome()
    cell = AgentCell(1, "stem", "stem", CellPosition("stem"))
    landscape = FateLandscape.default()

    repair = landscape.decide(cell, genome, {"test_failure": 0.9, "repair_pressure": 0.7})
    explorer = landscape.decide(cell, genome, {"ambiguity": 0.9, "missing_context": 0.6})

    assert repair.fate == "repair"
    assert repair.score > 0
    assert explorer.fate == "explorer"


def test_fate_landscape_hysteresis_keeps_committed_repair_cell() -> None:
    genome = make_genome()
    cell = AgentCell(1, "differentiated", "repair", CellPosition("repair"))

    decision = FateLandscape.default().decide(cell, genome, {"repair_pressure": 0.45, "review_pressure": 0.5})

    assert decision.fate == "repair"
    assert decision.reason == "attractor"


def test_fate_landscape_niche_bias_wins_close_plastic_decision() -> None:
    genome = make_genome()
    cell = AgentCell(1, "stem", "stem", CellPosition("stem"))
    landscape = FateLandscape(
        attractors=[
            FateAttractor("repair", ["repair_pressure"], threshold=0.2, commitment=1.0, hysteresis=0.0),
            FateAttractor("explorer", ["ambiguity"], threshold=0.2, commitment=1.0, hysteresis=0.0),
        ]
    )

    decision = landscape.decide(cell, genome, {"repair_pressure": 0.3, "ambiguity": 0.4}, niche_bias="repair")

    assert decision.fate == "repair"
    assert decision.niche_bias == "repair"


def test_task_microenvironment_loader_accepts_phase4_fields(tmp_path: Path) -> None:
    spec = tmp_path / "environment.yaml"
    spec.write_text(
        """
task:
  objective: Fix tests
morphogens:
  ambiguity: 0.2
topology:
  nodes:
    - id: repair-niche
      region: repo/tests
      neighbors: [stem-reserve]
      embedding: [10.0, 4.0, 0.0]
    - id: stem-reserve
      region: repo/tests
      neighbors: [repair-niche]
      embedding: [8.0, 4.0, 0.0]
morphogen_sources:
  - id: failing-tests
    signal: test_failure
    amount: 0.9
    position:
      node_id: repair-niche
      region: repo/tests
    radius: 2.0
fate_landscape:
  attractors:
    - fate: repair
      morphogens: [test_failure, repair_pressure, niche_vacancy]
      threshold: 0.4
      commitment: 1.0
      hysteresis: 0.2
niches:
  - id: repair-niche
    required_fate: repair
    position:
      node_id: repair-niche
      region: repo/tests
    demand: 1
""",
        encoding="utf-8",
    )

    environment = load_task_microenvironment(spec)

    assert environment.topology.node("repair-niche").neighbors == ["stem-reserve"]
    assert environment.morphogens.signal_at("test_failure", CellPosition("repair-niche", "repo/tests"), environment.topology) == 0.9
    assert environment.fate_landscape.attractors[0].fate == "repair"


def test_clearing_cell_emits_local_vacancy_and_records_fate_decision() -> None:
    tissue = TissueRuntime.seeded(make_genome(), make_environment(), stem_cells=4, seed=3)
    tissue.develop(ticks=3)
    removed = next(cell.id for cell in tissue.cells.values() if cell.fate == "repair")

    tissue.clear_cell(removed, reason="manual_clear")
    niche = tissue.environment.niche_by_id("repair-niche")
    local_vacancy = tissue.environment.morphogens.signal_at("niche_vacancy", niche.position, tissue.environment.topology)
    tissue.develop(ticks=2)

    assert local_vacancy >= 1.0
    assert any(event["type"] == "fate_decision" and event["niche_id"] == "repair-niche" for event in tissue.trace.events)
    assert any(event["type"] == "regeneration" and event["replaced_cell_id"] == removed for event in tissue.trace.events)


def test_repair_cells_cluster_on_repair_niche_graph_node_after_development() -> None:
    tissue = TissueRuntime.seeded(make_genome(), make_environment(), stem_cells=5, seed=5)

    tissue.develop(ticks=4)

    repair_cells = [cell for cell in tissue.cells.values() if cell.fate == "repair"]
    assert len(repair_cells) >= 2
    assert all(cell.position.node_id == "repair-niche" for cell in repair_cells)

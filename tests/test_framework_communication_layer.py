from __future__ import annotations

import json
from pathlib import Path

from ontocellia.__main__ import main
from ontocellia.framework import (
    AgentCell,
    AgentGenome,
    CellPosition,
    CommunicationPolicy,
    CommunicationRuntime,
    ExtracellularMatrix,
    Gene,
    MatrixRecord,
    MorphogenField,
    Niche,
    TaskMicroenvironment,
    TissueMessage,
    TissueRuntime,
    TissueTopology,
    TopologyNode,
    load_task_microenvironment,
)


def make_genome() -> AgentGenome:
    return AgentGenome(
        genes=[
            Gene("gene_repair", "regeneration", ["repair_pressure"], ["repair"]),
            Gene("gene_review", "verification", ["review_pressure"], ["review"]),
            Gene("gene_memory", "memory", ["memory_pressure"], ["remember"]),
        ]
    )


def make_tissue() -> TissueRuntime:
    topology = TissueTopology(
        {
            "repair-node": TopologyNode("repair-node", "repo/tests", ["review-node"], (0.0, 0.0, 0.0)),
            "review-node": TopologyNode("review-node", "repo/tests", ["repair-node"], (1.0, 0.0, 0.0)),
            "memory-node": TopologyNode("memory-node", "repo/memory", [], (10.0, 0.0, 0.0)),
        }
    )
    environment = TaskMicroenvironment(
        objective="Fix tests",
        morphogens=MorphogenField({"repair_pressure": 1.0, "review_pressure": 1.0, "memory_pressure": 1.0}),
        niches=[
            Niche("repair-niche", "repair", CellPosition("repair-node", "repo/tests", ["review-node"], (0.0, 0.0, 0.0))),
            Niche("review-niche", "reviewer", CellPosition("review-node", "repo/tests", ["repair-node"], (1.0, 0.0, 0.0))),
            Niche("memory-niche", "memory", CellPosition("memory-node", "repo/memory", [], (10.0, 0.0, 0.0))),
        ],
        topology=topology,
        communication_policy=CommunicationPolicy(matrix_query_limit=5, default_ttl=3, promote_confidence_threshold=0.6, allow_broadcast=True),
    )
    cells = {
        1: AgentCell(1, "differentiated", "repair", CellPosition("repair-node", "repo/tests", ["review-node"], (0.0, 0.0, 0.0)), niche_id="repair-niche"),
        2: AgentCell(2, "differentiated", "reviewer", CellPosition("review-node", "repo/tests", ["repair-node"], (1.0, 0.0, 0.0)), niche_id="review-niche"),
        3: AgentCell(3, "differentiated", "memory", CellPosition("memory-node", "repo/memory", [], (10.0, 0.0, 0.0)), niche_id="memory-niche"),
    }
    return TissueRuntime(genome=make_genome(), environment=environment, cells=cells)


def test_direct_message_routes_only_to_target_cell() -> None:
    tissue = make_tissue()
    message = TissueMessage(
        id="m1",
        sender_cell_id=1,
        recipient_cell_id=2,
        scope="direct",
        kind="observation",
        content="Patch candidate ready.",
        confidence=0.8,
    )

    deliveries = CommunicationRuntime().route(tissue, [message])

    assert [(delivery.message_id, delivery.recipient_cell_id) for delivery in deliveries] == [("m1", 2)]
    assert any(event["type"] == "message_delivered" and event["recipient_cell_id"] == 2 for event in tissue.trace.events)


def test_local_message_routes_to_same_region_graph_neighbors() -> None:
    tissue = make_tissue()
    message = TissueMessage("m1", sender_cell_id=1, scope="local", kind="observation", content="Failure localized.", confidence=0.8)

    deliveries = CommunicationRuntime().route(tissue, [message])

    assert [delivery.recipient_cell_id for delivery in deliveries] == [2]


def test_fate_scoped_message_routes_to_matching_fate_cells() -> None:
    tissue = make_tissue()
    message = TissueMessage("m1", sender_cell_id=1, recipient_fate="reviewer", scope="fate", kind="request", content="Please review.", confidence=0.7)

    deliveries = CommunicationRuntime().route(tissue, [message])

    assert [delivery.recipient_cell_id for delivery in deliveries] == [2]


def test_broadcast_messages_are_capped_by_policy() -> None:
    tissue = make_tissue()
    tissue.environment.communication_policy.broadcast_limit = 1
    message = TissueMessage("m1", sender_cell_id=1, scope="broadcast", kind="observation", content="Global note.", confidence=0.8)

    deliveries = CommunicationRuntime().route(tissue, [message])

    assert len(deliveries) == 1


def test_high_confidence_observation_promotes_to_matrix() -> None:
    tissue = make_tissue()
    message = TissueMessage("m1", sender_cell_id=1, scope="local", kind="observation", content="Failure in test_api.", confidence=0.9, references=["test_api"])

    CommunicationRuntime().route(tissue, [message])

    records = tissue.environment.matrix.query(tags=["test_api"], limit=5)
    assert len(records) == 1
    assert records[0].content == "Failure in test_api."
    assert any(event["type"] == "matrix_deposit" for event in tissue.trace.events)


def test_low_confidence_message_is_not_promoted_unless_memory() -> None:
    tissue = make_tissue()
    low = TissueMessage("low", sender_cell_id=1, scope="local", kind="hypothesis", content="Maybe parser.", confidence=0.2)
    memory = TissueMessage("mem", sender_cell_id=3, scope="broadcast", kind="memory", content="Keep this clue.", confidence=0.2)

    CommunicationRuntime().route(tissue, [low, memory])

    assert [record.content for record in tissue.environment.matrix.records] == ["Keep this clue."]


def test_record_memory_action_intent_creates_matrix_record() -> None:
    tissue = make_tissue()
    actions = [
        {
            "cell_id": 3,
            "fate": "memory",
            "intent_type": "record_memory",
            "target": "memory-node",
            "confidence": 0.72,
            "payload": {"message": "Remember failed approach.", "matrix_tags": ["memory", "failed_attempt"]},
        }
    ]

    tissue.communicate(actions)

    records = tissue.environment.matrix.query(tags=["failed_attempt"], limit=5)
    assert len(records) == 1
    assert records[0].source_cell_id == 3


def test_handoff_request_to_reviewer_receives_receipt() -> None:
    tissue = make_tissue()
    actions = [
        {
            "cell_id": 1,
            "fate": "repair",
            "intent_type": "propose_patch",
            "target": "repair-node",
            "confidence": 0.82,
            "payload": {"message": "Patch ready for review.", "handoff_to_fate": "reviewer", "matrix_tags": ["patch"]},
        }
    ]

    tissue.communicate(actions)

    assert any(event["type"] == "handoff_requested" and event["target_fate"] == "reviewer" for event in tissue.trace.events)
    assert any(event["type"] == "handoff_completed" and event["recipient_cell_id"] == 2 for event in tissue.trace.events)


def test_matrix_query_filters_by_tags_and_position() -> None:
    matrix = ExtracellularMatrix()
    matrix.deposit(MatrixRecord("r1", 1, "observation", "near", ["failure"], CellPosition("repair-node", "repo/tests"), 0.8, 1))
    matrix.deposit(MatrixRecord("r2", 2, "observation", "far", ["failure"], CellPosition("memory-node", "repo/memory"), 0.8, 1))
    topology = TissueTopology(
        {
            "repair-node": TopologyNode("repair-node", "repo/tests", ["review-node"], (0.0, 0.0, 0.0)),
            "memory-node": TopologyNode("memory-node", "repo/memory", [], (10.0, 0.0, 0.0)),
        }
    )

    records = matrix.query(tags=["failure"], position=CellPosition("repair-node", "repo/tests"), topology=topology, limit=1)

    assert [record.id for record in records] == ["r1"]


def test_expired_matrix_records_decay_deterministically() -> None:
    matrix = ExtracellularMatrix()
    matrix.deposit(MatrixRecord("old", 1, "memory", "old", ["memory"], CellPosition("repair-node"), 0.8, 1, expires_tick=2))
    matrix.deposit(MatrixRecord("fresh", 1, "memory", "fresh", ["memory"], CellPosition("repair-node"), 0.8, 1, expires_tick=5))

    matrix.decay(current_tick=3)

    assert [record.id for record in matrix.records] == ["fresh"]


def test_loader_accepts_communication_and_matrix_records(tmp_path: Path) -> None:
    spec = tmp_path / "environment.yaml"
    spec.write_text(
        """
task:
  objective: Fix tests
communication:
  matrix_query_limit: 7
  default_ttl: 4
  promote_confidence_threshold: 0.65
  allow_broadcast: true
matrix:
  records:
    - kind: observation
      content: Existing failing test evidence.
      tags: [test_failure, repo]
      confidence: 0.8
niches:
  - id: repair-niche
    required_fate: repair
    position:
      node_id: repair-node
    demand: 1
""",
        encoding="utf-8",
    )

    environment = load_task_microenvironment(spec)

    assert environment.communication_policy.matrix_query_limit == 7
    assert environment.communication_policy.default_ttl == 4
    assert environment.matrix.records[0].content == "Existing failing test evidence."


def test_tissue_cli_writes_communication_summary_fields(tmp_path: Path) -> None:
    output = tmp_path / "communication"

    main(
        [
            "tissue",
            "--genome-spec",
            "examples/framework/repo_repair_genome.yaml",
            "--environment-spec",
            "examples/framework/failing_tests_environment.yaml",
            "--steps",
            "4",
            "--effector",
            "mock-llm",
            "--output",
            str(output),
        ]
    )

    summary = json.loads((output / "tissue_summary.json").read_text(encoding="utf-8"))
    trace = json.loads((output / "tissue_trace.json").read_text(encoding="utf-8"))

    assert "messages" in summary
    assert "matrix_records" in summary
    assert "handoffs" in summary
    assert any(event["type"] in {"message_emitted", "matrix_deposit", "handoff_requested"} for event in trace)

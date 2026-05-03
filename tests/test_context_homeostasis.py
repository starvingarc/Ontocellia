from __future__ import annotations

from pathlib import Path

from ontocellia.framework import (
    AgentCell,
    AgentGenome,
    CellPosition,
    CellPromptBuilder,
    ContextHomeostasisRuntime,
    ContextRetrievalPolicy,
    ExecutionPolicy,
    ExecutionRuntime,
    ExtracellularInterface,
    ExtracellularMatrix,
    Gene,
    MatrixRecord,
    MockLLMProvider,
    MorphogenField,
    Niche,
    OrganValidationResult,
    TaskMicroenvironment,
    TissueRuntime,
    TissueTopology,
    TopologyNode,
    EffectorRuntime,
    load_task_microenvironment,
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
        objective="Fix failing tests",
        morphogens=MorphogenField({"repair_pressure": 1.0, "validation_pressure": 0.7}),
        niches=[Niche("repair-niche", "repair", CellPosition("repair-node", "repo/tests", ["review-node"]))],
        interfaces=[
            ExtracellularInterface("workspace", "extracellular_matrix", ["repair", "reviewer"]),
            ExtracellularInterface("pytest", "membrane_channel", ["repair", "reviewer"]),
        ],
        topology=topology,
    )
    cell = AgentCell(
        1,
        "differentiated",
        "repair",
        CellPosition("repair-node", "repo/tests", ["review-node"]),
        niche_id="repair-niche",
        expressed_gene_ids=["gene_repair"],
    )
    return TissueRuntime(
        genome=AgentGenome([Gene("gene_repair", "regeneration", ["repair_pressure"], ["repair"], validation_hooks=["python -m pytest -q"])]),
        environment=environment,
        cells={1: cell},
    )


def test_matrix_records_load_with_context_defaults(tmp_path: Path) -> None:
    spec = tmp_path / "environment.yaml"
    spec.write_text(
        """
task:
  objective: Fix tests
matrix:
  records:
    - id: seed-evidence
      kind: observation
      content: Failure in test_api.
      tags: [test_failure, repo]
      confidence: 0.8
      status: active
      validation_status: unverified
      salience: 0.9
niches:
  - id: repair-niche
    required_fate: repair
    demand: 1
    position:
      node_id: repair-node
""",
        encoding="utf-8",
    )

    environment = load_task_microenvironment(spec)
    record = environment.matrix.records[0]

    assert record.status == "active"
    assert record.validation_status == "unverified"
    assert record.references == []
    assert record.lineage_id is None
    assert record.salience == 0.9
    assert record.decay_rate == 0.05


def test_context_retrieval_prefers_relevant_validated_local_records() -> None:
    tissue = make_tissue()
    matrix = tissue.environment.matrix
    matrix.deposit(
        MatrixRecord(
            "validated-local",
            1,
            "execution",
            "pytest failure points to repair target",
            ["test_failure", "workspace"],
            CellPosition("repair-node", "repo/tests"),
            confidence=0.7,
            created_tick=3,
            fate="repair",
            validation_status="validated",
            salience=0.9,
        )
    )
    matrix.deposit(
        MatrixRecord(
            "far-old",
            2,
            "hypothesis",
            "old memory note",
            ["memory"],
            CellPosition("memory-node", "repo/memory"),
            confidence=0.9,
            created_tick=1,
            fate="memory",
            salience=0.9,
        )
    )

    packet = matrix.query_context(
        fate="repair",
        position=tissue.cells[1].position,
        topology=tissue.environment.topology,
        tags=["test_failure"],
        accepted_interfaces=["workspace"],
        current_tick=4,
        policy=ContextRetrievalPolicy(limit=2),
    )

    assert packet.record_ids[0] == "validated-local"
    assert packet.records[0]["score"] > packet.records[-1]["score"]


def test_context_decay_suppresses_stale_low_salience_records() -> None:
    matrix = ExtracellularMatrix(
        [
            MatrixRecord(
                "stale",
                1,
                "hypothesis",
                "weak hunch",
                ["hypothesis"],
                CellPosition("repair-node"),
                confidence=0.15,
                created_tick=1,
                salience=0.1,
                decay_rate=0.2,
            )
        ]
    )

    ContextHomeostasisRuntime().decay(matrix, current_tick=4)

    record = matrix.records[0]
    assert record.status == "suppressed"
    assert record.confidence < 0.15


def test_correction_record_references_and_suppresses_failed_hypothesis() -> None:
    matrix = ExtracellularMatrix(
        [
            MatrixRecord(
                "bad-hypothesis",
                1,
                "hypothesis",
                "The failing test is unrelated to parsing.",
                ["hypothesis", "test_failure"],
                CellPosition("repair-node"),
                confidence=0.6,
                created_tick=1,
            )
        ]
    )

    correction = ContextHomeostasisRuntime().correct(
        matrix,
        record_id="bad-hypothesis",
        correction_id="correction-1",
        content="Validation contradicted the parser hypothesis.",
        source_cell_id=2,
        position=CellPosition("review-node"),
        created_tick=3,
    )

    assert matrix.records[0].status == "corrected"
    assert correction.corrects_record_id == "bad-hypothesis"
    assert correction.references == ["bad-hypothesis"]


def test_cell_prompt_includes_budgeted_relevant_matrix_context() -> None:
    tissue = make_tissue()
    tissue.environment.matrix.deposit(
        MatrixRecord(
            "short-evidence",
            1,
            "observation",
            "Failure: expected 2 but got 1.",
            ["test_failure", "workspace"],
            CellPosition("repair-node", "repo/tests"),
            confidence=0.8,
            created_tick=1,
            fate="repair",
        )
    )
    tissue.environment.matrix.deposit(
        MatrixRecord(
            "long-evidence",
            1,
            "observation",
            "x" * 1000,
            ["test_failure"],
            CellPosition("repair-node", "repo/tests"),
            confidence=0.8,
            created_tick=2,
            fate="repair",
        )
    )

    prompt = CellPromptBuilder(ContextRetrievalPolicy(limit=5, max_context_chars=220)).build(tissue, tissue.cells[1])

    assert prompt.context["context_record_ids"]
    assert "relevant_matrix" in prompt.context
    assert sum(len(record["content"]) for record in prompt.context["relevant_matrix"]) <= 220


def test_effector_trace_and_intent_payload_include_context_record_ids() -> None:
    tissue = make_tissue()
    tissue.environment.matrix.deposit(
        MatrixRecord(
            "repair-evidence",
            1,
            "observation",
            "Failure evidence",
            ["test_failure", "workspace"],
            CellPosition("repair-node", "repo/tests"),
            confidence=0.8,
            created_tick=1,
            fate="repair",
        )
    )

    intents = EffectorRuntime(MockLLMProvider()).emit_intents(tissue)
    event = next(event for event in tissue.trace.events if event["type"] == "llm_effector")

    assert intents[0].payload["context_record_ids"] == ["repair-evidence"]
    assert event["context_record_ids"] == ["repair-evidence"]


def test_execution_result_deposits_validated_context_metadata(tmp_path: Path) -> None:
    test_file = tmp_path / "test_ok.py"
    test_file.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    tissue = make_tissue()
    command = f"python -m pytest -q {test_file.name}"

    results = ExecutionRuntime().execute(
        tissue,
        [{"cell_id": 1, "intent_type": "review_output", "target": "tests", "payload": {"command": command}}],
        ExecutionPolicy(
            workspace_root=tmp_path,
            allowed_interfaces=["pytest.run"],
            allowed_commands=[command],
            dry_run=False,
            timeout_seconds=10,
        ),
    )

    record = tissue.environment.matrix.records[-1]
    assert results[0].passed is True
    assert record.kind == "execution"
    assert record.validation_status == "validated"
    assert record.references == [results[0].request.id]


def test_validation_feedback_updates_related_context_status() -> None:
    tissue = make_tissue()
    tissue.environment.matrix.deposit(
        MatrixRecord(
            "execution-1",
            1,
            "execution",
            "pytest failed",
            ["execution", "pytest.run", "failed"],
            CellPosition("repair-node"),
            confidence=0.0,
            created_tick=1,
            validation_status="failed",
        )
    )

    ContextHomeostasisRuntime().apply_validation_feedback(
        tissue.environment.matrix,
        [OrganValidationResult("execution:pytest.run", False, 0.0, target="tests", evidence="pytest failed")],
        current_tick=2,
    )

    record = tissue.environment.matrix.records[0]
    assert record.status == "suppressed"
    assert record.validation_status == "failed"
    assert any(item.kind == "validation" and item.references == ["execution-1"] for item in tissue.environment.matrix.records)

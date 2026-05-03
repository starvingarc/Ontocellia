from __future__ import annotations

from pathlib import Path

from ontocellia.framework import (
    AgentCell,
    AgentGenome,
    CellPosition,
    CellPromptBuilder,
    CommunicationPolicy,
    ContextMetabolismPolicy,
    ContextMetabolismRuntime,
    ExtracellularInterface,
    Gene,
    MatrixRecord,
    MorphogenField,
    Niche,
    OrganValidationResult,
    TaskMicroenvironment,
    TissueRuntime,
    load_task_microenvironment,
)


def make_tissue() -> TissueRuntime:
    environment = TaskMicroenvironment(
        objective="Fix failing tests while preserving behavior.",
        morphogens=MorphogenField({"repair_pressure": 1.0, "validation_pressure": 0.7}),
        niches=[Niche("repair-niche", "repair", CellPosition("repair-node", "repo/tests"))],
        interfaces=[
            ExtracellularInterface("workspace", "extracellular_matrix", ["repair", "reviewer"]),
            ExtracellularInterface("pytest", "membrane_channel", ["repair", "reviewer"]),
        ],
        communication_policy=CommunicationPolicy(
            matrix_query_limit=6,
            context_budget_chars=1600,
            context_metabolism=ContextMetabolismPolicy(window_ticks=3, min_source_records=2),
        ),
    )
    cell = AgentCell(
        1,
        "differentiated",
        "repair",
        CellPosition("repair-node", "repo/tests"),
        niche_id="repair-niche",
        expressed_gene_ids=["gene_repair"],
    )
    return TissueRuntime(
        genome=AgentGenome([Gene("gene_repair", "regeneration", ["repair_pressure"], ["repair"])]),
        environment=environment,
        cells={1: cell},
    )


def metabolite_records(tissue: TissueRuntime, kind: str | None = None) -> list[MatrixRecord]:
    records = [
        record
        for record in tissue.environment.matrix.records
        if record.metadata.get("metabolite_kind")
    ]
    if kind is not None:
        records = [record for record in records if record.metadata.get("metabolite_kind") == kind]
    return records


def test_failed_validation_and_test_failure_records_generate_failure_signature() -> None:
    tissue = make_tissue()
    tissue.tick_count = 4
    tissue.environment.matrix.deposit(
        MatrixRecord(
            "test-failure",
            1,
            "observation",
            "pytest failure in test_api_parse",
            ["test_failure", "pytest"],
            CellPosition("repair-node", "repo/tests"),
            confidence=0.8,
            created_tick=3,
            salience=0.9,
        )
    )
    tissue.environment.matrix.deposit(
        MatrixRecord(
            "validation-failed",
            0,
            "validation",
            "pytest failed: test_api_parse",
            ["validation", "failed"],
            CellPosition("repair-node", "repo/tests"),
            confidence=0.1,
            created_tick=4,
            validation_status="failed",
            salience=0.8,
        )
    )

    report = ContextMetabolismRuntime().metabolize(tissue)

    signatures = metabolite_records(tissue, "failure_signature")
    assert report.deposited_record_ids == [signatures[0].id]
    assert signatures[0].metadata["source_record_ids"] == ["test-failure", "validation-failed"]
    assert signatures[0].metadata["compression_level"] == "metabolite"
    assert signatures[0].metadata["lossiness"] == "bounded"
    assert signatures[0].metadata["source_count"] == 2
    assert tissue.environment.matrix.records[0].salience < 0.9


def test_recent_window_generates_episode_summary() -> None:
    tissue = make_tissue()
    tissue.tick_count = 8
    for index in range(3):
        tissue.environment.matrix.deposit(
            MatrixRecord(
                f"episode-{index}",
                1,
                "observation",
                f"recent tissue event {index}",
                ["observation", "repair"],
                CellPosition("repair-node", "repo/tests"),
                created_tick=7 + (index % 2),
                confidence=0.7,
                salience=0.7,
            )
        )
    tissue.trace.record("message_emitted", tick=8, message_id="m1")
    tissue.trace.record("handoff_completed", tick=8, request_id="h1")

    ContextMetabolismRuntime().metabolize(tissue)

    summaries = metabolite_records(tissue, "episode_summary")
    assert len(summaries) == 1
    assert summaries[0].metadata["source_record_ids"] == ["episode-0", "episode-1", "episode-2"]
    assert summaries[0].metadata["source_trace_event_ids"]


def test_trace_and_execution_chain_generates_causal_chain() -> None:
    tissue = make_tissue()
    tissue.tick_count = 5
    tissue.trace.record("llm_effector", tick=5, cell_id=1, intent={"intent_type": "propose_patch"})
    tissue.trace.record("message_emitted", tick=5, message_id="m1")
    tissue.trace.record("handoff_completed", tick=5, request_id="h1")
    tissue.environment.matrix.deposit(
        MatrixRecord(
            "execution-1",
            1,
            "execution",
            "pytest run failed",
            ["execution", "pytest.run", "failed"],
            CellPosition("repair-node"),
            created_tick=5,
            validation_status="failed",
        )
    )
    tissue.environment.matrix.deposit(
        MatrixRecord(
            "validation-1",
            0,
            "validation",
            "validation failed after patch",
            ["validation", "failed"],
            CellPosition("repair-node"),
            created_tick=5,
            validation_status="failed",
        )
    )

    ContextMetabolismRuntime().metabolize(tissue)

    chains = metabolite_records(tissue, "causal_chain")
    assert len(chains) == 1
    assert "execution-1" in chains[0].metadata["source_record_ids"]
    assert chains[0].metadata["source_trace_event_ids"]


def test_medium_and_constraint_records_generate_constraint_digest() -> None:
    tissue = make_tissue()
    tissue.tick_count = 2
    tissue.environment.matrix.deposit(
        MatrixRecord(
            "medium-1",
            -1,
            "medium_change",
            "Preserve public behavior while repairing failing tests.",
            ["medium", "task_change", "constraint"],
            CellPosition("culture-medium", "global"),
            created_tick=2,
            confidence=0.8,
        )
    )
    tissue.environment.matrix.deposit(
        MatrixRecord(
            "policy-1",
            0,
            "policy",
            "Do not run non-allowlisted commands.",
            ["policy", "constraint"],
            CellPosition("culture-medium", "global"),
            created_tick=2,
            confidence=0.8,
        )
    )

    ContextMetabolismRuntime().metabolize(tissue)

    digests = metabolite_records(tissue, "constraint_digest")
    assert len(digests) == 1
    assert digests[0].metadata["source_record_ids"] == ["medium-1", "policy-1"]


def test_failed_contradicted_and_suppressed_records_generate_toxic_context() -> None:
    tissue = make_tissue()
    tissue.tick_count = 3
    for record_id, status, validation_status in [
        ("failed-1", "active", "failed"),
        ("contradicted-1", "corrected", "contradicted"),
        ("suppressed-1", "suppressed", "unverified"),
    ]:
        tissue.environment.matrix.deposit(
            MatrixRecord(
                record_id,
                1,
                "hypothesis",
                f"bad context {record_id}",
                ["hypothesis"],
                CellPosition("repair-node"),
                created_tick=2,
                status=status,
                validation_status=validation_status,
                salience=0.5,
            )
        )

    ContextMetabolismRuntime().metabolize(tissue)

    toxic = metabolite_records(tissue, "toxic_context")
    assert len(toxic) == 1
    assert toxic[0].metadata["source_record_ids"] == ["contradicted-1", "failed-1", "suppressed-1"]


def test_repeated_metabolism_does_not_duplicate_same_source_set() -> None:
    tissue = make_tissue()
    tissue.tick_count = 4
    tissue.environment.matrix.deposit(
        MatrixRecord("a", 1, "observation", "failure a", ["test_failure"], CellPosition("repair-node"), created_tick=4)
    )
    tissue.environment.matrix.deposit(
        MatrixRecord("b", 1, "validation", "failure b", ["failed"], CellPosition("repair-node"), created_tick=4, validation_status="failed")
    )

    runtime = ContextMetabolismRuntime()
    runtime.metabolize(tissue)
    runtime.metabolize(tissue)

    assert len(metabolite_records(tissue, "failure_signature")) == 1


def test_cell_prompt_groups_context_metabolites_and_raw_context_records() -> None:
    tissue = make_tissue()
    tissue.tick_count = 4
    tissue.environment.matrix.deposit(
        MatrixRecord("raw-1", 1, "observation", "raw failure evidence", ["test_failure", "workspace"], CellPosition("repair-node"), created_tick=4, fate="repair")
    )
    tissue.environment.matrix.deposit(
        MatrixRecord(
            "metabolite-1",
            0,
            "context_metabolite",
            "Failure signature: pytest API parse",
            ["metabolite", "failure_signature", "test_failure", "workspace"],
            CellPosition("repair-node"),
            created_tick=4,
            fate="repair",
            salience=1.0,
            metadata={"metabolite_kind": "failure_signature", "source_record_ids": ["raw-1"]},
        )
    )

    prompt = CellPromptBuilder().build(tissue, tissue.cells[1])

    assert [record["id"] for record in prompt.context["context_metabolites"]] == ["metabolite-1"]
    assert "raw-1" in prompt.context["raw_context_record_ids"]
    assert "relevant_matrix" in prompt.context


def test_yaml_loader_accepts_context_metabolism_policy(tmp_path: Path) -> None:
    spec = tmp_path / "environment.yaml"
    spec.write_text(
        """
task:
  objective: Fix tests
communication:
  matrix_query_limit: 7
  context_budget_chars: 1200
  context_metabolism:
    enabled: false
    window_ticks: 5
    max_metabolites_per_tick: 2
    max_metabolite_chars: 300
    min_source_records: 3
    source_salience_decay: 0.25
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
    policy = environment.communication_policy.context_metabolism

    assert policy.enabled is False
    assert policy.window_ticks == 5
    assert policy.max_metabolites_per_tick == 2
    assert policy.max_metabolite_chars == 300
    assert policy.min_source_records == 3
    assert policy.source_salience_decay == 0.25


def test_develop_automatically_runs_context_metabolism_after_validation_feedback() -> None:
    tissue = make_tissue()
    tissue.environment.matrix.deposit(
        MatrixRecord("raw-failure", 1, "observation", "pytest failed", ["test_failure"], CellPosition("repair-node"), created_tick=0)
    )

    tissue.develop(
        ticks=1,
        validation_results=[OrganValidationResult("pytest", False, 0.0, target="tests", evidence="pytest failed")],
    )

    assert metabolite_records(tissue, "failure_signature")
    assert any(event["type"] == "context_metabolism" for event in tissue.trace.events)
    assert any(event["type"] == "context_metabolite_deposited" for event in tissue.trace.events)

from __future__ import annotations

import json
from pathlib import Path

from ontocellia.__main__ import main
from ontocellia.framework import (
    AgentGenome,
    CellPromptBuilder,
    CellPosition,
    EffectorRuntime,
    ExtracellularInterface,
    Gene,
    MockLLMProvider,
    MorphogenField,
    Niche,
    ReceptorProfile,
    TaskMicroenvironment,
    TissueRuntime,
)


def make_tissue() -> TissueRuntime:
    genome = AgentGenome(
        genes=[
            Gene("gene_repair_from_test_failures", "regeneration", ["repair_pressure"], ["repair"], validation_hooks=["python -m pytest -q"]),
            Gene("gene_inspect_context", "exploration", ["ambiguity"], ["inspect"]),
            Gene("gene_review_boundary", "verification", ["review_pressure"], ["review"], validation_hooks=["git diff --check"]),
        ]
    )
    environment = TaskMicroenvironment(
        objective="Fix failing tests",
        morphogens=MorphogenField({"repair_pressure": 1.0, "ambiguity": 1.0, "review_pressure": 1.0}),
        niches=[
            Niche("repair-niche", "repair", CellPosition("repair-niche", "repo", [], (1.0, 0.0, 0.0))),
            Niche("exploration-front", "explorer", CellPosition("exploration-front", "repo", [], (0.0, 1.0, 0.0))),
            Niche("review-boundary", "reviewer", CellPosition("review-boundary", "repo", [], (0.0, 0.0, 1.0))),
        ],
        interfaces=[
            ExtracellularInterface("pytest", "membrane_channel", ["repair", "reviewer"]),
            ExtracellularInterface("workspace", "extracellular_matrix", ["repair", "explorer"]),
            ExtracellularInterface("git", "membrane_channel", ["reviewer"]),
        ],
    )
    tissue = TissueRuntime.seeded(genome, environment, stem_cells=3, seed=2)
    tissue.cells[0].receptor = ReceptorProfile(accepted_interfaces=["pytest", "workspace"])
    tissue.cells[1].receptor = ReceptorProfile(accepted_interfaces=["workspace"])
    tissue.cells[2].receptor = ReceptorProfile(accepted_interfaces=["git"])
    tissue._differentiate(tissue.cells[0], environment.niche_by_id("repair-niche"))
    tissue._differentiate(tissue.cells[1], environment.niche_by_id("exploration-front"))
    tissue._differentiate(tissue.cells[2], environment.niche_by_id("review-boundary"))
    return tissue


def test_mock_llm_effector_emits_structured_action_intents() -> None:
    tissue = make_tissue()
    effectors = EffectorRuntime(provider=MockLLMProvider())

    intents = effectors.emit_intents(tissue)

    by_fate = {intent.fate: intent for intent in intents}
    assert by_fate["repair"].intent_type == "propose_patch"
    assert by_fate["explorer"].intent_type == "inspect_context"
    assert by_fate["reviewer"].intent_type == "review_output"
    assert by_fate["repair"].expressed_gene_ids == ["gene_repair_from_test_failures"]
    assert by_fate["repair"].validation_hooks == ["python -m pytest -q"]
    assert by_fate["repair"].confidence > 0


def test_effector_respects_receptor_interface_gate() -> None:
    tissue = make_tissue()
    tissue.cells[0].receptor = ReceptorProfile(accepted_interfaces=["workspace"])

    intents = EffectorRuntime(provider=MockLLMProvider()).emit_intents(tissue)
    repair = next(intent for intent in intents if intent.fate == "repair")

    assert repair.required_interfaces == ["workspace"]


def test_effector_records_prompt_and_response_trace() -> None:
    tissue = make_tissue()

    EffectorRuntime(provider=MockLLMProvider(), prompt_builder=CellPromptBuilder()).emit_intents(tissue)

    llm_events = [event for event in tissue.trace.events if event["type"] == "llm_effector"]
    assert llm_events
    assert llm_events[0]["provider"] == "mock-llm"
    assert "expressed_gene_ids" in llm_events[0]["prompt"]["context"]
    assert llm_events[0]["intent"]["cell_id"] == llm_events[0]["cell_id"]


def test_tissue_execute_accepts_optional_effector_runtime() -> None:
    tissue = make_tissue()

    intents = tissue.execute(effectors=EffectorRuntime(provider=MockLLMProvider()))

    assert any(intent["intent_type"] == "propose_patch" for intent in intents)


def test_tissue_cli_mock_llm_writes_action_intents(tmp_path: Path) -> None:
    output = tmp_path / "llm_tissue"

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

    intents = json.loads((output / "action_intents.json").read_text(encoding="utf-8"))
    trace = json.loads((output / "llm_trace.json").read_text(encoding="utf-8"))
    assert intents
    assert trace

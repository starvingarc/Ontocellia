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
    OpenAICompatibleProvider,
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


def test_tissue_cli_real_provider_requires_configured_api_key(tmp_path: Path, monkeypatch) -> None:
    for name in ("DEEPSEEK_API_KEY", "MOONSHOT_API_KEY", "KIMI_API_KEY", "MINIMAX_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    try:
        main(
            [
                "tissue",
                "--genome-spec",
                "examples/framework/repo_repair_genome.yaml",
                "--environment-spec",
                "examples/framework/failing_tests_environment.yaml",
                "--steps",
                "1",
                "--effector",
                "deepseek",
                "--output",
                str(tmp_path / "deepseek_tissue"),
            ]
        )
    except ValueError as error:
        assert "DEEPSEEK_API_KEY" in str(error)
    else:
        raise AssertionError("expected CLI real provider mode to require an API key")


def test_openai_compatible_provider_uses_deepseek_defaults_and_parses_intent() -> None:
    captured: dict[str, object] = {}

    def fake_transport(url: str, headers: dict[str, str], payload: dict[str, object], timeout: float) -> dict[str, object]:
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        captured["timeout"] = timeout
        return {
            "model": payload["model"],
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "intent_type": "propose_patch",
                                "target": "repair-niche",
                                "rationale": "Patch tests after inspecting failure signals.",
                                "required_interfaces": ["pytest", "workspace"],
                                "confidence": 0.81,
                                "payload": {"plan": ["inspect", "patch", "test"]},
                            }
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 16, "total_tokens": 28},
        }

    tissue = make_tissue()
    prompt = CellPromptBuilder().build(tissue, tissue.cells[0])
    provider = OpenAICompatibleProvider.from_name(
        "deepseek",
        env={"DEEPSEEK_API_KEY": "test-key"},
        transport=fake_transport,
    )

    response = provider.complete(prompt)

    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer test-key"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "deepseek-v4-flash"
    assert response.parsed_intent.intent_type == "propose_patch"
    assert response.parsed_intent.cell_id == tissue.cells[0].id
    assert response.parsed_intent.expressed_gene_ids == tissue.cells[0].expressed_gene_ids
    assert response.usage["total_tokens"] == 28


def test_openai_compatible_provider_parses_minimax_markdown_action_shape() -> None:
    content = """<think>Reason about the cell state.</think>

```json
{
  "action": "propose_patch",
  "target": "repair-niche",
  "metadata": {
    "rationale": "Repair the failing test niche.",
    "confidence": 0.67,
    "required_interfaces": ["pytest", "workspace"]
  },
  "patch_proposal": {
    "description": "Inspect failures and propose a minimal patch."
  }
}
```"""

    def fake_transport(url: str, headers: dict[str, str], payload: dict[str, object], timeout: float) -> dict[str, object]:
        return {"model": "MiniMax-M2.7", "choices": [{"message": {"content": content}}], "usage": {"total_tokens": 42}}

    tissue = make_tissue()
    prompt = CellPromptBuilder().build(tissue, tissue.cells[0])
    provider = OpenAICompatibleProvider.from_name(
        "minimax",
        env={"MINIMAX_API_KEY": "test-key"},
        base_url="https://api.minimax.chat/v1",
        transport=fake_transport,
    )

    response = provider.complete(prompt)

    assert response.parsed_intent.intent_type == "propose_patch"
    assert response.parsed_intent.rationale == "Repair the failing test niche."
    assert response.parsed_intent.required_interfaces == ["pytest", "workspace"]
    assert response.parsed_intent.confidence == 0.67
    assert response.parsed_intent.payload["patch_proposal"]["description"] == "Inspect failures and propose a minimal patch."


def test_openai_compatible_provider_requires_api_key() -> None:
    try:
        OpenAICompatibleProvider.from_name("kimi", env={})
    except ValueError as error:
        assert "MOONSHOT_API_KEY" in str(error)
        assert "KIMI_API_KEY" in str(error)
    else:
        raise AssertionError("expected missing Kimi API key to fail")


def test_openai_compatible_provider_supports_kimi_and_minimax_defaults() -> None:
    kimi = OpenAICompatibleProvider.from_name("kimi", env={"KIMI_API_KEY": "kimi-key"})
    minimax = OpenAICompatibleProvider.from_name("minimax", env={"MINIMAX_API_KEY": "minimax-key"})

    assert kimi.base_url == "https://api.moonshot.ai/v1"
    assert kimi.model == "kimi-k2.6"
    assert minimax.base_url == "https://api.minimax.io/v1"
    assert minimax.model == "MiniMax-M2.7"

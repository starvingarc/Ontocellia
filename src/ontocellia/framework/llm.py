from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class ActionIntent:
    cell_id: int
    fate: str
    expressed_gene_ids: list[str]
    intent_type: str
    target: str
    rationale: str
    required_interfaces: list[str]
    confidence: float
    validation_hooks: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CellPrompt:
    system: str
    context: dict[str, Any]
    output_schema: dict[str, Any]


@dataclass(slots=True)
class LLMResponse:
    content: str
    parsed_intent: ActionIntent
    raw: dict[str, Any]
    model: str
    usage: dict[str, int] = field(default_factory=dict)


class LLMProvider(Protocol):
    name: str

    def complete(self, prompt: CellPrompt) -> LLMResponse:
        ...


class CellPromptBuilder:
    def build(self, tissue: Any, cell: Any) -> CellPrompt:
        programs = [tissue.genome.gene_by_id(gene_id) for gene_id in cell.expressed_gene_ids]
        allowed_interfaces = [
            interface.id
            for interface in tissue.environment.interfaces
            if interface.accepts(cell) and cell.accepts_interface(interface.id)
        ]
        validation_hooks: list[str] = []
        for program in programs:
            validation_hooks.extend(program.validation_hooks)
        return CellPrompt(
            system="You are an Ontocellia cell effector. Translate expressed genes into one structured action intent.",
            context={
                "cell_id": cell.id,
                "fate": cell.fate,
                "stage": str(cell.stage),
                "position": {
                    "node_id": cell.position.node_id,
                    "region": cell.position.region,
                    "neighbors": cell.position.neighbors,
                    "embedding": list(cell.position.embedding),
                },
                "objective": tissue.environment.objective,
                "morphogens": dict(tissue.environment.morphogens.signals),
                "expressed_gene_ids": list(cell.expressed_gene_ids),
                "encoded_responses": [response for gene in programs for response in gene.encoded_response],
                "allowed_interfaces": allowed_interfaces,
                "validation_hooks": validation_hooks,
            },
            output_schema={
                "type": "ActionIntent",
                "required": ["intent_type", "target", "rationale", "required_interfaces", "confidence"],
            },
        )


class MockLLMProvider:
    name = "mock-llm"

    def complete(self, prompt: CellPrompt) -> LLMResponse:
        context = prompt.context
        gene_ids = list(context["expressed_gene_ids"])
        fate = str(context["fate"])
        allowed = list(context["allowed_interfaces"])
        intent_type = _intent_type(fate, gene_ids)
        required = _required_interfaces(intent_type, allowed)
        intent = ActionIntent(
            cell_id=int(context["cell_id"]),
            fate=fate,
            expressed_gene_ids=gene_ids,
            intent_type=intent_type,
            target=str(context["position"]["node_id"]),
            rationale=f"Mock provider translated {', '.join(gene_ids)} for {fate}.",
            required_interfaces=required,
            confidence=0.72,
            validation_hooks=list(context["validation_hooks"]),
            payload={"encoded_responses": list(context["encoded_responses"])},
        )
        return LLMResponse(
            content=intent.rationale,
            parsed_intent=intent,
            raw={"provider": self.name, "prompt_context": context},
            model=self.name,
            usage={"prompt_tokens": 0, "completion_tokens": 0},
        )


class EffectorRuntime:
    def __init__(self, provider: LLMProvider, prompt_builder: CellPromptBuilder | None = None):
        self.provider = provider
        self.prompt_builder = prompt_builder or CellPromptBuilder()

    def emit_intents(self, tissue: Any) -> list[ActionIntent]:
        intents: list[ActionIntent] = []
        for cell in sorted(tissue.cells.values(), key=lambda item: item.id):
            if not cell.differentiated or not cell.expressed_gene_ids:
                continue
            prompt = self.prompt_builder.build(tissue, cell)
            if not prompt.context["allowed_interfaces"]:
                tissue.trace.record("llm_effector_skipped", cell_id=cell.id, reason="no accepted interfaces")
                continue
            response = self.provider.complete(prompt)
            intent = response.parsed_intent
            intent.required_interfaces = [
                interface_id for interface_id in intent.required_interfaces if interface_id in prompt.context["allowed_interfaces"]
            ]
            tissue.trace.record(
                "llm_effector",
                cell_id=cell.id,
                provider=getattr(self.provider, "name", response.model),
                model=response.model,
                prompt={"system": prompt.system, "context": prompt.context, "output_schema": prompt.output_schema},
                intent=intent.as_dict(),
                usage=response.usage,
            )
            intents.append(intent)
        return intents


def _intent_type(fate: str, gene_ids: list[str]) -> str:
    joined = " ".join(gene_ids)
    if fate == "repair" or "repair" in joined:
        return "propose_patch"
    if fate == "explorer" or "inspect" in joined or "search" in joined:
        return "inspect_context"
    if fate == "reviewer" or "review" in joined:
        return "review_output"
    if fate == "memory":
        return "record_memory"
    return "propose_action"


def _required_interfaces(intent_type: str, allowed: list[str]) -> list[str]:
    preferences = {
        "propose_patch": ["pytest", "workspace"],
        "inspect_context": ["workspace", "web"],
        "review_output": ["git", "pytest", "workspace"],
        "record_memory": ["citation_store", "workspace"],
    }
    preferred = preferences.get(intent_type, allowed)
    selected = [interface_id for interface_id in preferred if interface_id in allowed]
    return selected or allowed[:1]

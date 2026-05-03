from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol

from ontocellia.framework.communication import ContextRetrievalPolicy


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
    def __init__(self, retrieval_policy: ContextRetrievalPolicy | None = None):
        self.retrieval_policy = retrieval_policy

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
        retrieval_policy = self.retrieval_policy or ContextRetrievalPolicy(
            limit=int(getattr(tissue.environment.communication_policy, "matrix_query_limit", 5)),
            max_context_chars=int(getattr(tissue.environment.communication_policy, "context_budget_chars", 1600)),
        )
        lineage_id = str(cell.lineage.root_id) if getattr(cell, "lineage", None) is not None else None
        context_packet = tissue.environment.matrix.query_context(
            fate=cell.fate,
            position=cell.position,
            topology=tissue.environment.topology,
            tags=_context_tags(cell.fate, cell.expressed_gene_ids, [gene.category for gene in programs]),
            accepted_interfaces=allowed_interfaces,
            current_tick=int(getattr(tissue, "tick_count", 0)),
            lineage_id=lineage_id,
            policy=retrieval_policy,
        )
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
                "relevant_matrix": list(context_packet.records),
                "context_record_ids": context_packet.record_ids,
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
            payload={
                "encoded_responses": list(context["encoded_responses"]),
                "message": f"{fate} cell reports {intent_type}.",
                "matrix_tags": [fate, intent_type],
                **({"handoff_to_fate": "reviewer"} if intent_type == "propose_patch" else {}),
            },
        )
        return LLMResponse(
            content=intent.rationale,
            parsed_intent=intent,
            raw={"provider": self.name, "prompt_context": context},
            model=self.name,
            usage={"prompt_tokens": 0, "completion_tokens": 0},
        )


ProviderTransport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


@dataclass(slots=True)
class OpenAICompatibleProvider:
    """Real OpenAI-compatible chat-completions provider."""

    name: str
    api_key: str
    base_url: str
    model: str
    timeout: float = 60.0
    transport: ProviderTransport | None = None

    @classmethod
    def from_name(
        cls,
        name: str,
        *,
        model: str | None = None,
        base_url: str | None = None,
        env: dict[str, str] | None = None,
        transport: ProviderTransport | None = None,
    ) -> "OpenAICompatibleProvider":
        environ = env if env is not None else os.environ
        provider = _provider_defaults(name)
        api_key = _first_env(environ, provider["key_env"])
        if not api_key:
            raise ValueError(f"{name} provider requires one of: {', '.join(provider['key_env'])}")
        return cls(
            name=name,
            api_key=api_key,
            base_url=base_url or str(provider["base_url"]),
            model=model or str(provider["model"]),
            transport=transport,
        )

    def complete(self, prompt: CellPrompt) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt.system},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "context": prompt.context,
                            "output_schema": prompt.output_schema,
                            "instruction": "Return exactly one JSON object matching the ActionIntent shape.",
                        },
                        ensure_ascii=True,
                    ),
                },
            ],
            "temperature": 0.2,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        raw = (self.transport or _post_json)(self._chat_url(), headers, payload, self.timeout)
        content = str(raw["choices"][0]["message"].get("content", ""))
        intent = _parse_action_intent(content, prompt.context)
        return LLMResponse(
            content=content,
            parsed_intent=intent,
            raw=_redacted_raw(raw),
            model=str(raw.get("model", self.model)),
            usage={key: int(value) for key, value in raw.get("usage", {}).items() if isinstance(value, int)},
        )

    def _chat_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"


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
            context_record_ids = [str(record_id) for record_id in prompt.context.get("context_record_ids", [])]
            intent.payload = {**dict(intent.payload), "context_record_ids": context_record_ids}
            tissue.trace.record(
                "llm_effector",
                cell_id=cell.id,
                provider=getattr(self.provider, "name", response.model),
                model=response.model,
                prompt={"system": prompt.system, "context": prompt.context, "output_schema": prompt.output_schema},
                intent=intent.as_dict(),
                context_record_ids=context_record_ids,
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


def _context_tags(fate: str, gene_ids: list[str], categories: list[str]) -> list[str]:
    tags = [fate, *gene_ids, *categories]
    fate_tags = {
        "repair": ["test_failure", "repair_pressure", "validation_pressure", "execution", "workspace", "pytest"],
        "reviewer": ["review", "validation", "risk", "execution", "git", "pytest"],
        "explorer": ["ambiguity", "missing_context", "workspace", "observation", "hypothesis"],
        "planner": ["policy", "constraint", "coordination", "handoff"],
        "memory": ["memory", "handoff", "observation"],
    }
    tags.extend(fate_tags.get(fate, []))
    return list(dict.fromkeys(str(tag) for tag in tags if tag))


def _provider_defaults(name: str) -> dict[str, object]:
    providers = {
        "deepseek": {
            "key_env": ["DEEPSEEK_API_KEY"],
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-v4-flash",
        },
        "kimi": {
            "key_env": ["MOONSHOT_API_KEY", "KIMI_API_KEY"],
            "base_url": "https://api.moonshot.ai/v1",
            "model": "kimi-k2.6",
        },
        "minimax": {
            "key_env": ["MINIMAX_API_KEY"],
            "base_url": "https://api.minimax.io/v1",
            "model": "MiniMax-M2.7",
        },
        "openai": {
            "key_env": ["OPENAI_API_KEY"],
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4.1-mini",
        },
        "openrouter": {
            "key_env": ["OPENROUTER_API_KEY"],
            "base_url": "https://openrouter.ai/api/v1",
            "model": "openai/gpt-4.1-mini",
        },
        "ollama": {
            "key_env": ["OLLAMA_API_KEY"],
            "base_url": "http://localhost:11434/v1",
            "model": "llama3.1",
        },
        "custom-openai-compatible": {
            "key_env": ["ONTOCELLIA_CUSTOM_API_KEY"],
            "base_url": "https://example.com/v1",
            "model": "custom-model",
        },
    }
    if name not in providers:
        raise ValueError(f"unknown LLM provider: {name}")
    return providers[name]


def _first_env(environ: dict[str, str], names: object) -> str:
    for name in names:
        value = environ.get(str(name), "")
        if value:
            return value
    return ""


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM provider request failed with HTTP {error.code}: {body}") from error


def _parse_action_intent(content: str, context: dict[str, Any]) -> ActionIntent:
    data = _json_object(content)
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    payload = data.get("payload")
    if not isinstance(payload, dict):
        known_keys = {
            "cell_id",
            "fate",
            "expressed_gene_ids",
            "intent_type",
            "action",
            "type",
            "target",
            "rationale",
            "required_interfaces",
            "confidence",
            "validation_hooks",
            "metadata",
        }
        payload = {key: value for key, value in data.items() if key not in known_keys}
    return ActionIntent(
        cell_id=int(data.get("cell_id", context["cell_id"])),
        fate=str(data.get("fate", context["fate"])),
        expressed_gene_ids=[str(gene_id) for gene_id in data.get("expressed_gene_ids", context["expressed_gene_ids"])],
        intent_type=str(data.get("intent_type") or data.get("action") or data.get("type") or _intent_type(str(context["fate"]), list(context["expressed_gene_ids"]))),
        target=str(data.get("target", context["position"]["node_id"])),
        rationale=str(data.get("rationale") or metadata.get("rationale") or content),
        required_interfaces=[
            str(item)
            for item in (data.get("required_interfaces") or metadata.get("required_interfaces") or context["allowed_interfaces"][:1])
        ],
        confidence=float(data.get("confidence") or metadata.get("confidence") or 0.5),
        validation_hooks=[str(item) for item in data.get("validation_hooks", context.get("validation_hooks", []))],
        payload=payload,
    )


def _json_object(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = _first_json_object(content)
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _first_json_object(content: str) -> dict[str, Any]:
    for start, char in enumerate(content):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(content)):
            current = content[index]
            if escape:
                escape = False
                continue
            if current == "\\":
                escape = True
                continue
            if current == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(content[start : index + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
    return {}


def _redacted_raw(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in raw.items() if key.lower() not in {"authorization", "api_key"}}

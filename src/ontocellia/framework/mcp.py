from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ontocellia.framework.cell import CellPosition
from ontocellia.framework.communication import MatrixRecord
from ontocellia.framework.core import ExtracellularInterface, TaskMicroenvironment


@dataclass(slots=True)
class MCPToolSpec:
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    accepts_fates: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MCPResourceSpec:
    id: str
    uri: str
    content: str = ""
    tags: list[str] = field(default_factory=list)
    position: CellPosition | tuple[float, ...] | list[float] | dict[str, Any] = field(default_factory=lambda: CellPosition(""))
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.position = CellPosition.from_value(self.position)


@dataclass(slots=True)
class MCPPromptSpec:
    id: str
    template: str
    tags: list[str] = field(default_factory=list)
    accepts_fates: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MCPServerSpec:
    id: str
    name: str = ""
    tools: list[MCPToolSpec] = field(default_factory=list)
    resources: list[MCPResourceSpec] = field(default_factory=list)
    prompts: list[MCPPromptSpec] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MCPToolResult:
    server_id: str
    tool_name: str
    content: str
    tags: list[str] = field(default_factory=list)
    morphogens: dict[str, float] = field(default_factory=dict)
    position: CellPosition | tuple[float, ...] | list[float] | dict[str, Any] = field(default_factory=lambda: CellPosition(""))
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.position = CellPosition.from_value(self.position)


@dataclass(slots=True)
class MCPInterfaceAdapter:
    servers: list[MCPServerSpec] = field(default_factory=list)

    def interfaces(self) -> list[ExtracellularInterface]:
        interfaces: list[ExtracellularInterface] = []
        for server in self.servers:
            for tool in server.tools:
                interfaces.append(
                    ExtracellularInterface(
                        id=f"mcp:{server.id}:tool:{tool.name}",
                        kind="membrane_channel",
                        accepts_fates=list(tool.accepts_fates),
                        metadata={
                            "mcp": {"server_id": server.id, "tool_name": tool.name},
                            "description": tool.description,
                            "receptor_schema": dict(tool.input_schema),
                            **dict(tool.metadata),
                        },
                    )
                )
            for prompt in server.prompts:
                interfaces.append(
                    ExtracellularInterface(
                        id=f"mcp:{server.id}:prompt:{prompt.id}",
                        kind="induction_factor",
                        accepts_fates=list(prompt.accepts_fates or _fates_from_tags(prompt.tags)),
                        metadata={
                            "mcp": {"server_id": server.id, "prompt_id": prompt.id},
                            "template": prompt.template,
                            "tags": list(prompt.tags),
                            **dict(prompt.metadata),
                        },
                    )
                )
        return interfaces

    def matrix_records(self) -> list[MatrixRecord]:
        records: list[MatrixRecord] = []
        for server in self.servers:
            for resource in server.resources:
                records.append(
                    MatrixRecord(
                        id=f"mcp:{server.id}:resource:{resource.id}",
                        source_cell_id=0,
                        kind="mcp_resource",
                        content=resource.content or resource.uri,
                        tags=list(resource.tags),
                        position=resource.position,
                        confidence=resource.confidence,
                        created_tick=0,
                    )
                )
        return records

    def apply_to_environment(self, environment: TaskMicroenvironment) -> None:
        existing = {interface.id for interface in environment.interfaces}
        environment.interfaces.extend([interface for interface in self.interfaces() if interface.id not in existing])
        existing_records = {record.id for record in environment.matrix.records}
        for record in self.matrix_records():
            if record.id not in existing_records:
                environment.matrix.deposit(record)
        environment.mcp_adapter = self

    def apply_tool_result(self, environment: TaskMicroenvironment, result: MCPToolResult) -> MatrixRecord:
        record = MatrixRecord(
            id=f"mcp:{result.server_id}:result:{result.tool_name}:{len(environment.matrix.records)}",
            source_cell_id=0,
            kind="mcp_tool_result",
            content=result.content,
            tags=list(result.tags or [result.tool_name]),
            position=result.position,
            confidence=result.confidence,
            created_tick=0,
        )
        environment.matrix.deposit(record)
        for name, amount in result.morphogens.items():
            environment.morphogens.emit(str(name), float(amount))
        return record


def _fates_from_tags(tags: list[str]) -> list[str]:
    mapping = {
        "repair": "repair",
        "review": "reviewer",
        "verification": "reviewer",
        "explore": "explorer",
        "exploration": "explorer",
        "memory": "memory",
        "build": "builder",
        "implementation": "builder",
    }
    fates = [mapping[tag] for tag in tags if tag in mapping]
    return list(dict.fromkeys(fates)) or ["explorer", "repair", "reviewer", "builder", "memory"]

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ontocellia.framework.cell import CellPosition
from ontocellia.framework.communication import CommunicationPolicy, ContextMetabolismPolicy, ExtracellularMatrix, MatrixRecord
from ontocellia.framework.core import ExtracellularInterface, MorphogenField, MorphogenSource, Niche, TaskMicroenvironment
from ontocellia.framework.fate import FateAttractor, FateLandscape
from ontocellia.framework.genome import AgentGenome, EpigeneticMarks, Gene, RegulatoryElement
from ontocellia.framework.mcp import MCPInterfaceAdapter, MCPPromptSpec, MCPResourceSpec, MCPServerSpec, MCPToolSpec
from ontocellia.framework.resources import ResourceCompetitionPolicy
from ontocellia.framework.selection import OrganSelectionTarget
from ontocellia.framework.topology import TissueTopology, TopologyNode


def load_agent_genome(path: str | Path) -> AgentGenome:
    data = _load_yaml(path)
    genes = [Gene(**_strip_type(gene_data)) for gene_data in data.get("genes", [])]
    regulatory_elements = [RegulatoryElement(**element_data) for element_data in data.get("regulatory_elements", [])]
    return AgentGenome(
        genes=genes,
        metadata=dict(data.get("metadata", {})),
        regulatory_elements=regulatory_elements,
        epigenetic_defaults=_epigenetic_marks(data.get("epigenetic_defaults", {})),
    )


def load_task_microenvironment(path: str | Path) -> TaskMicroenvironment:
    data = _load_yaml(path)
    task = data.get("task", {})
    objective = str(task.get("objective", data.get("objective", "")))
    morphogens = MorphogenField(
        signals={str(name): float(value) for name, value in data.get("morphogens", data.get("signals", {})).items()},
        sources=[_morphogen_source(source) for source in data.get("morphogen_sources", [])],
    )
    niches = [
        Niche(
            id=str(niche["id"]),
            required_fate=str(niche["required_fate"]),
            position=_position(niche.get("position", (0.0, 0.0))),
            demand=int(niche.get("demand", 1)),
        )
        for niche in data.get("niches", [])
    ]
    interfaces = [
        ExtracellularInterface(
            id=str(interface["id"]),
            kind=str(interface.get("kind", "membrane_channel")),
            accepts_fates=[str(fate) for fate in interface.get("accepts_fates", [])],
            metadata=dict(interface.get("metadata", {})),
        )
        for interface in data.get("interfaces", [])
    ]
    environment = TaskMicroenvironment(
        objective=objective,
        morphogens=morphogens,
        niches=niches,
        interfaces=interfaces,
        topology=_topology(data.get("topology"), niches),
        fate_landscape=_fate_landscape(data.get("fate_landscape")),
        selection_targets=_selection_targets(data.get("organ_selection")),
        resource_policy=_resource_policy(data.get("resources")),
        matrix=_matrix(data.get("matrix")),
        communication_policy=_communication_policy(data.get("communication")),
    )
    mcp_adapter = _mcp_adapter(data.get("mcp"))
    if mcp_adapter is not None:
        mcp_adapter.apply_to_environment(environment)
    return environment


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def _strip_type(data: dict[str, Any]) -> dict[str, Any]:
    result = dict(data)
    result.pop("type", None)
    return result


def _position(value: Any) -> CellPosition:
    return CellPosition.from_value(value)


def _morphogen_source(data: dict[str, Any]) -> MorphogenSource:
    return MorphogenSource(
        id=str(data.get("id", data["signal"])),
        signal=str(data["signal"]),
        amount=float(data.get("amount", 0.0)),
        position=_position(data.get("position", (0.0, 0.0, 0.0))),
        radius=float(data.get("radius", 1.0)),
    )


def _topology(data: Any, niches: list[Niche]) -> TissueTopology:
    if not isinstance(data, dict):
        return TissueTopology.from_niches(niches)
    nodes = {
        str(node["id"]): TopologyNode(
            id=str(node["id"]),
            region=str(node.get("region", "")),
            neighbors=[str(neighbor) for neighbor in node.get("neighbors", [])],
            embedding=_position(node.get("embedding", (0.0, 0.0, 0.0))).embedding,
            metadata=dict(node.get("metadata", {})),
        )
        for node in data.get("nodes", [])
    }
    topology = TissueTopology(nodes=nodes)
    for niche in niches:
        topology.ensure_node(niche.position)
    return topology


def _fate_landscape(data: Any) -> FateLandscape:
    if not isinstance(data, dict):
        return FateLandscape.default()
    attractors = [
        FateAttractor(
            fate=str(attractor["fate"]),
            morphogens=[str(name) for name in attractor.get("morphogens", [])],
            threshold=float(attractor.get("threshold", 0.4)),
            commitment=float(attractor.get("commitment", 1.0)),
            hysteresis=float(attractor.get("hysteresis", 0.15)),
            competence_window=[str(item) for item in attractor.get("competence_window", [])],
        )
        for attractor in data.get("attractors", [])
    ]
    return FateLandscape(attractors=attractors) if attractors else FateLandscape.default()


def _selection_targets(data: Any) -> OrganSelectionTarget:
    if not isinstance(data, dict):
        return OrganSelectionTarget()
    return OrganSelectionTarget(
        min_coverage=float(data.get("min_coverage", 0.5)),
        min_diversity=float(data.get("min_diversity", 0.3)),
        min_validation_score=float(data.get("min_validation_score", 0.7)),
        max_risk=float(data.get("max_risk", 0.4)),
        max_cost=float(data.get("max_cost", 1.0)),
    )


def _communication_policy(data: Any) -> CommunicationPolicy:
    if not isinstance(data, dict):
        return CommunicationPolicy()
    return CommunicationPolicy(
        matrix_query_limit=int(data.get("matrix_query_limit", 5)),
        default_ttl=int(data.get("default_ttl", 3)),
        promote_confidence_threshold=float(data.get("promote_confidence_threshold", 0.6)),
        allow_broadcast=bool(data.get("allow_broadcast", True)),
        broadcast_limit=int(data.get("broadcast_limit", 8)),
        context_budget_chars=int(data.get("context_budget_chars", 1600)),
        context_metabolism=_context_metabolism_policy(data.get("context_metabolism")),
    )


def _context_metabolism_policy(data: Any) -> ContextMetabolismPolicy:
    if not isinstance(data, dict):
        return ContextMetabolismPolicy()
    return ContextMetabolismPolicy(
        enabled=bool(data.get("enabled", True)),
        window_ticks=int(data.get("window_ticks", 3)),
        max_metabolites_per_tick=int(data.get("max_metabolites_per_tick", 4)),
        max_metabolite_chars=int(data.get("max_metabolite_chars", 700)),
        min_source_records=int(data.get("min_source_records", 2)),
        source_salience_decay=float(data.get("source_salience_decay", 0.15)),
    )


def _resource_policy(data: Any) -> ResourceCompetitionPolicy:
    if not isinstance(data, dict):
        return ResourceCompetitionPolicy()
    return ResourceCompetitionPolicy(
        enabled=bool(data.get("enabled", True)),
        population_cap=int(data["population_cap"]) if data.get("population_cap") is not None else None,
        maintenance_cost=float(data.get("maintenance_cost", 0.01)),
        differentiated_cost=float(data.get("differentiated_cost", 0.01)),
        action_intent_cost=float(data.get("action_intent_cost", 0.015)),
        tool_cost_weight=float(data.get("tool_cost_weight", 0.1)),
        latency_cost_weight=float(data.get("latency_cost_weight", 0.01)),
        contribution_reward=float(data.get("contribution_reward", 0.08)),
        negative_contribution_penalty=float(data.get("negative_contribution_penalty", 0.08)),
        low_energy_threshold=float(data.get("low_energy_threshold", 0.35)),
        quiescence_threshold=float(data.get("quiescence_threshold", 0.08)),
        apoptosis_threshold=float(data.get("apoptosis_threshold", 0.02)),
        allow_quiescence=bool(data.get("allow_quiescence", False)),
        allow_apoptosis=bool(data.get("allow_apoptosis", False)),
        over_cap_pressure_weight=float(data.get("over_cap_pressure_weight", 0.25)),
        low_energy_pressure_weight=float(data.get("low_energy_pressure_weight", 0.35)),
        max_energy=float(data.get("max_energy", 1.2)),
    )


def _matrix(data: Any) -> ExtracellularMatrix:
    if not isinstance(data, dict):
        return ExtracellularMatrix()
    records = [
        MatrixRecord(
            id=str(record.get("id", f"matrix-seed-{index}")),
            source_cell_id=int(record.get("source_cell_id", 0)),
            kind=str(record.get("kind", "observation")),
            content=str(record.get("content", "")),
            tags=[str(tag) for tag in record.get("tags", [])],
            position=_position(record.get("position", {"node_id": ""})),
            confidence=float(record.get("confidence", 0.5)),
            created_tick=int(record.get("created_tick", 0)),
            expires_tick=record.get("expires_tick"),
            fate=str(record["fate"]) if "fate" in record else None,
            status=str(record.get("status", "active")),
            validation_status=str(record.get("validation_status", "unverified")),
            lineage_id=str(record["lineage_id"]) if "lineage_id" in record else None,
            references=[str(reference) for reference in record.get("references", [])],
            salience=float(record.get("salience", 0.5)),
            decay_rate=float(record.get("decay_rate", 0.05)),
            corrects_record_id=str(record["corrects_record_id"]) if "corrects_record_id" in record else None,
            metadata=dict(record.get("metadata", {})),
        )
        for index, record in enumerate(data.get("records", []))
    ]
    return ExtracellularMatrix(records=records)


def _mcp_adapter(data: Any) -> MCPInterfaceAdapter | None:
    if not isinstance(data, dict):
        return None
    servers = [_mcp_server(server) for server in data.get("servers", [])]
    return MCPInterfaceAdapter(servers=servers)


def _mcp_server(data: dict[str, Any]) -> MCPServerSpec:
    return MCPServerSpec(
        id=str(data["id"]),
        name=str(data.get("name", "")),
        tools=[_mcp_tool(tool) for tool in data.get("tools", [])],
        resources=[_mcp_resource(resource) for resource in data.get("resources", [])],
        prompts=[_mcp_prompt(prompt) for prompt in data.get("prompts", [])],
        metadata=dict(data.get("metadata", {})),
    )


def _mcp_tool(data: dict[str, Any]) -> MCPToolSpec:
    return MCPToolSpec(
        name=str(data["name"]),
        description=str(data.get("description", "")),
        input_schema=dict(data.get("input_schema", {})),
        accepts_fates=[str(fate) for fate in data.get("accepts_fates", [])],
        metadata=dict(data.get("metadata", {})),
    )


def _mcp_resource(data: dict[str, Any]) -> MCPResourceSpec:
    return MCPResourceSpec(
        id=str(data["id"]),
        uri=str(data.get("uri", data["id"])),
        content=str(data.get("content", "")),
        tags=[str(tag) for tag in data.get("tags", [])],
        position=_position(data.get("position", {"node_id": ""})),
        confidence=float(data.get("confidence", 0.5)),
        metadata=dict(data.get("metadata", {})),
    )


def _mcp_prompt(data: dict[str, Any]) -> MCPPromptSpec:
    return MCPPromptSpec(
        id=str(data["id"]),
        template=str(data.get("template", "")),
        tags=[str(tag) for tag in data.get("tags", [])],
        accepts_fates=[str(fate) for fate in data.get("accepts_fates", [])],
        metadata=dict(data.get("metadata", {})),
    )


def _epigenetic_marks(data: Any) -> EpigeneticMarks:
    if data is None:
        return EpigeneticMarks()
    if not isinstance(data, dict):
        raise ValueError("epigenetic_defaults must be a mapping")
    return EpigeneticMarks(
        fate_locks={str(name): float(value) for name, value in data.get("fate_locks", {}).items()},
        gene_locks={str(name): float(value) for name, value in data.get("gene_locks", {}).items()},
    )

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ontocellia.framework.cell import CellPosition
from ontocellia.framework.core import ExtracellularInterface, MorphogenField, MorphogenSource, Niche, TaskMicroenvironment
from ontocellia.framework.fate import FateAttractor, FateLandscape
from ontocellia.framework.genome import AgentGenome, EpigeneticMarks, Gene, RegulatoryElement
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
    return TaskMicroenvironment(
        objective=objective,
        morphogens=morphogens,
        niches=niches,
        interfaces=interfaces,
        topology=_topology(data.get("topology"), niches),
        fate_landscape=_fate_landscape(data.get("fate_landscape")),
        matrix=dict(data.get("matrix", {})),
    )


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


def _epigenetic_marks(data: Any) -> EpigeneticMarks:
    if data is None:
        return EpigeneticMarks()
    if not isinstance(data, dict):
        raise ValueError("epigenetic_defaults must be a mapping")
    return EpigeneticMarks(
        fate_locks={str(name): float(value) for name, value in data.get("fate_locks", {}).items()},
        gene_locks={str(name): float(value) for name, value in data.get("gene_locks", {}).items()},
    )

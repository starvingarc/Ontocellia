from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ontocellia.framework.core import ExtracellularInterface, MorphogenField, Niche, TaskMicroenvironment
from ontocellia.framework.genome import AgentGenome, EpigeneticMarks, Gene, RegulatoryElement


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
    morphogens = MorphogenField(signals={str(name): float(value) for name, value in data.get("morphogens", data.get("signals", {})).items()})
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


def _position(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("niche.position must be a two-item list")
    return (float(value[0]), float(value[1]))


def _epigenetic_marks(data: Any) -> EpigeneticMarks:
    if data is None:
        return EpigeneticMarks()
    if not isinstance(data, dict):
        raise ValueError("epigenetic_defaults must be a mapping")
    return EpigeneticMarks(
        fate_locks={str(name): float(value) for name, value in data.get("fate_locks", {}).items()},
        gene_locks={str(name): float(value) for name, value in data.get("gene_locks", {}).items()},
    )

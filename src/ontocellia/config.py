from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np


FATE_NAMES = ("stem", "progenitor", "specialist", "repair-active")
FIELD_NAMES = (
    "nutrient",
    "damage",
    "morphogen_a",
    "morphogen_b",
    "task_pressure",
    "crowding",
)

LEGACY_MODE = "legacy"
SPEC_MODE = "spec"


class GeneKind(StrEnum):
    STRATEGY = "strategy_gene"
    WARNING = "warning_gene"


@dataclass(slots=True)
class GeneAsset:
    kind: GeneKind
    name: str
    signals: list[str]
    summary: str
    steps: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    validation_hooks: list[str] = field(default_factory=list)
    version: str = "0.1.0"
    provenance: str = "manual"
    magnitude: float = 0.15

    def matches(self, context: dict[str, float]) -> bool:
        return any(context.get(signal, 0.0) > 0.45 for signal in self.signals)


@dataclass(slots=True)
class OntocelliaConfig:
    width: int = 32
    height: int = 32
    initial_cells: int = 28
    seed: int = 7
    hidden_dim: int = 8
    local_memory_dim: int = 4
    spatial_radius: float = 3.25
    communication_radius: float = 5.0
    max_neighbors: int = 6
    diffusion_rate: float = 0.15
    decay_rate: float = 0.02
    emission_scale: float = 0.22
    movement_scale: float = 0.7
    division_threshold: float = 0.5
    death_threshold: float = 1.02
    energy_floor: float = 0.05
    desired_local_density: float = 0.28
    commitment_decay: float = 0.72
    commitment_threshold: float = 0.5
    lock_strength: float = 0.28
    repair_boost: float = 0.24
    damage_repair_decay: float = 0.12
    enable_spatial: bool = True
    enable_graph: bool = True
    enable_competence: bool = True
    enable_epigenetic_lock: bool = True
    resource_driven_division: bool = True
    enable_organ_feedback: bool = True
    mutation_noise: float = 0.03
    lineage_cooldown_steps: int = 5
    community_min_size: int = 2
    community_edge_threshold: float = 0.55
    gene_evolution_period: int | None = None
    default_genes: list[GeneAsset] = field(default_factory=list)

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "initial_cells": self.initial_cells,
            "seed": self.seed,
            "enable_spatial": self.enable_spatial,
            "enable_graph": self.enable_graph,
            "enable_competence": self.enable_competence,
            "enable_epigenetic_lock": self.enable_epigenetic_lock,
            "enable_organ_feedback": self.enable_organ_feedback,
            "resource_driven_division": self.resource_driven_division,
        }

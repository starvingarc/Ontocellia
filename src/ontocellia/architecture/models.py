from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ontocellia.config import FATE_NAMES


@dataclass(slots=True)
class NeighborhoodState:
    hidden_mean: np.ndarray
    fate_mean: np.ndarray
    development_mean: np.ndarray
    graph_density: float
    local_demand: float
    contact_area_mean: float = 0.0
    contact_persistence_mean: float = 0.0
    relative_density_mean: float = 0.0
    neighbor_quiescence: float = 0.0
    contact_inhibition: float = 0.0
    community_signal: float = 0.0

    @classmethod
    def from_summary(cls, summary: dict[str, np.ndarray | float]) -> "NeighborhoodState":
        return cls(
            hidden_mean=np.asarray(summary["hidden_mean"], dtype=float),
            fate_mean=np.asarray(summary["fate_mean"], dtype=float),
            development_mean=np.asarray(summary["development_mean"], dtype=float),
            graph_density=float(summary["graph_density"]),
            local_demand=float(summary["local_demand"]),
            contact_area_mean=float(summary.get("contact_area_mean", 0.0)),
            contact_persistence_mean=float(summary.get("contact_persistence_mean", 0.0)),
            relative_density_mean=float(summary.get("relative_density_mean", 0.0)),
            neighbor_quiescence=float(summary.get("neighbor_quiescence", 0.0)),
            contact_inhibition=float(summary.get("contact_inhibition", 0.0)),
            community_signal=float(summary.get("community_signal", 0.0)),
        )


@dataclass(slots=True)
class LocalContext:
    diffusive_fields: dict[str, float]
    gradient_fields: dict[str, np.ndarray]
    neighbor_messages: dict[str, float]
    mechanical_resource_context: dict[str, float]
    local_risk: float
    global_signals: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class GenomeInput:
    self_state: "CellStateModel"
    neighborhood_state: NeighborhoodState
    local_context: LocalContext
    history_state: np.ndarray


@dataclass(slots=True)
class CellTransition:
    hidden_state: np.ndarray
    local_memory: np.ndarray
    signal_emission: dict[str, float]
    neighbor_signal: np.ndarray | None
    movement: np.ndarray
    division_score: float
    death_score: float
    fate_logits: np.ndarray
    edge_intent: float
    energy_delta: float
    stress_delta: float
    development_delta: np.ndarray | None = None
    repair_score: float = 0.0
    probe_scores: dict[str, float] | None = None
    attractor_potentials: dict[str, float] | None = None
    quiescence_drive: float = 0.0


@dataclass(slots=True)
class LifeProcessDecision:
    action_scores: dict[str, float]
    should_divide: bool = False
    should_die: bool = False
    should_dedifferentiate: bool = False
    should_quiesce: bool = False
    form_community_score: float = 0.0
    fuse_score: float = 0.0


@dataclass(slots=True)
class OrganFeedback:
    task_pressure_bias: float = 0.0
    resource_pressure_bias: float = 0.0
    damage_tolerance_bias: float = 0.0
    reward_field_bias: float = 0.0
    selection_pressure: float = 0.0
    target_regions: list[tuple[float, float, float]] = field(default_factory=list)


@dataclass(slots=True)
class CommunityState:
    id: int
    member_ids: list[int]
    centroid: np.ndarray
    shared_development: np.ndarray
    signal_pool: float
    cohesion: float


@dataclass(slots=True)
class CellStateModel:
    id: int
    pos: np.ndarray
    hidden_state: np.ndarray
    fate_dist: np.ndarray
    commitment: np.ndarray
    energy: float
    stress: float
    age: int
    competence: np.ndarray
    epigenetic_lock: float
    lineage_parent: int | None
    local_memory: np.ndarray
    active_genes: list[str] = field(default_factory=list)
    lineage_cooldown: int = 0
    current_fate: str = "stem"
    previous_fate: str = "stem"
    alive: bool = True
    trust: float = 0.5
    development_state: np.ndarray | None = None
    competence_state: np.ndarray | None = None
    phenotype_label: str = "unlabeled"
    phenotype_scores: dict[str, float] = field(default_factory=dict)
    repair_signal: float = 0.0
    contact_state: np.ndarray | None = None
    reprogramming_cost: float = 0.0
    commitment_timer: dict[str, int] = field(default_factory=dict)
    quiescence_state: float = 0.0
    attractor_potentials: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, float | str]] = field(default_factory=list)
    receptor_profile: np.ndarray | None = None
    community_id: int | None = None

    @property
    def internal_state(self) -> np.ndarray:
        return self.hidden_state

    @property
    def fate_state(self) -> np.ndarray:
        if self.development_state is not None:
            return self.development_state
        return self.commitment

    def append_history(self, event: dict[str, float | str], limit: int = 8) -> None:
        self.history.append(event)
        if len(self.history) > limit:
            del self.history[:-limit]

    def summarize_history(self, width: int | None = None) -> np.ndarray:
        target = width if width is not None else max(4, self.local_memory.size)
        if not self.history:
            return np.zeros(target, dtype=float)
        pressure = np.mean([float(event.get("task_pressure", 0.0)) for event in self.history])
        damage = np.mean([float(event.get("damage", 0.0)) for event in self.history])
        energy = np.mean([float(event.get("energy", self.energy)) for event in self.history])
        stress = np.mean([float(event.get("stress", self.stress)) for event in self.history])
        vector = np.array([pressure, damage, energy, stress], dtype=float)
        if target <= 4:
            return vector[:target]
        padded = np.zeros(target, dtype=float)
        padded[:4] = vector
        if self.development_state is not None:
            remainder = min(target - 4, self.development_state.size)
            padded[4 : 4 + remainder] = self.development_state[:remainder]
        return padded

    def clone(self, child_id: int, pos: np.ndarray, noise: np.ndarray) -> "CellStateModel":
        child_hidden = np.clip(self.hidden_state + noise[: self.hidden_state.size], -1.0, 1.0)
        child_memory = np.clip(self.local_memory + noise[: self.local_memory.size], -1.0, 1.0)
        child_development = None
        child_commitment = self.commitment.copy() * 0.92
        child_competence = np.clip(self.competence + noise[: self.competence.size], 0.05, 1.0)
        child_contact_state = self.contact_state.copy() if self.contact_state is not None else None
        child_receptors = self.receptor_profile.copy() if self.receptor_profile is not None else None
        if self.development_state is not None:
            dev_noise = noise[: self.development_state.size]
            child_development = np.clip(self.development_state + dev_noise, -1.0, 1.0)
            positive = np.maximum(child_development, 0.0) + 1e-6
            child_commitment = positive / positive.sum()
            child_competence = np.clip(self.competence + dev_noise[: self.competence.size], 0.05, 1.0)
            if child_contact_state is None:
                child_contact_state = np.zeros_like(child_development)
        return CellStateModel(
            id=child_id,
            pos=pos.astype(float),
            hidden_state=child_hidden,
            fate_dist=self.fate_dist.copy(),
            commitment=child_commitment,
            energy=max(0.12, self.energy * 0.45),
            stress=min(1.0, self.stress * 1.05),
            age=0,
            competence=child_competence,
            epigenetic_lock=max(0.05, self.epigenetic_lock * 0.9),
            lineage_parent=self.id,
            local_memory=child_memory,
            active_genes=list(self.active_genes),
            lineage_cooldown=self.lineage_cooldown,
            current_fate=self.current_fate,
            previous_fate=self.current_fate,
            alive=True,
            trust=self.trust,
            development_state=child_development,
            competence_state=self.competence_state.copy() if self.competence_state is not None else None,
            phenotype_label=self.phenotype_label,
            phenotype_scores=dict(self.phenotype_scores),
            repair_signal=self.repair_signal,
            contact_state=child_contact_state,
            reprogramming_cost=self.reprogramming_cost,
            commitment_timer=dict(self.commitment_timer),
            quiescence_state=self.quiescence_state,
            attractor_potentials=dict(self.attractor_potentials),
            history=list(self.history),
            receptor_profile=child_receptors,
            community_id=None,
        )

    @property
    def fate_index(self) -> int:
        return FATE_NAMES.index(self.current_fate)

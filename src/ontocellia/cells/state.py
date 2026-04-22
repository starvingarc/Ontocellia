from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ontocellia.config import FATE_NAMES


@dataclass(slots=True)
class CellState:
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

    def clone(self, child_id: int, pos: np.ndarray, noise: np.ndarray) -> "CellState":
        child_hidden = np.clip(self.hidden_state + noise[: self.hidden_state.size], -1.0, 1.0)
        child_memory = np.clip(self.local_memory + noise[: self.local_memory.size], -1.0, 1.0)
        child_development = None
        child_commitment = self.commitment.copy() * 0.92
        child_competence = np.clip(self.competence + noise[: self.competence.size], 0.05, 1.0)
        child_contact_state = self.contact_state.copy() if self.contact_state is not None else None
        if self.development_state is not None:
            dev_noise = noise[: self.development_state.size]
            child_development = np.clip(self.development_state + dev_noise, -1.0, 1.0)
            positive = np.maximum(child_development, 0.0) + 1e-6
            child_commitment = positive / positive.sum()
            child_competence = np.clip(self.competence + dev_noise[: self.competence.size], 0.05, 1.0)
            if child_contact_state is None:
                child_contact_state = np.zeros_like(child_development)
        return CellState(
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
        )

    @property
    def fate_index(self) -> int:
        return FATE_NAMES.index(self.current_fate)

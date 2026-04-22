from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ontocellia.compiler import CompiledGenome
from ontocellia.config import OntocelliaConfig


@dataclass(slots=True)
class FateLandscape:
    config: OntocelliaConfig
    compiled_genome: CompiledGenome

    def update_competence(self, cell, fields: dict[str, float]) -> np.ndarray:
        gates = np.ones(self.compiled_genome.development_dim, dtype=float) * 0.8
        for gate in self.compiled_genome.competence_gates:
            signal = 0.0
            signal += sum(fields.get(name, 0.0) * weight for name, weight in gate.promoters.items())
            signal -= sum(fields.get(name, 0.0) * weight for name, weight in gate.inhibitors.items())
            signal += gate.age_weight * np.clip(1.0 - cell.age / 60, 0.0, 1.0)
            signal += gate.energy_weight * cell.energy
            signal -= gate.stress_weight * cell.stress
            signal += gate.ecm_weight * fields.get("ECM", 0.0)
            signal += gate.mechanical_weight * fields.get("mechanical_stress", 0.0)
            if cell.receptor_profile is not None and gate.dimension < cell.receptor_profile.size:
                signal *= max(0.1, float(cell.receptor_profile[gate.dimension]))
            gates[gate.dimension] = 1.0 / (1.0 + np.exp(-signal))
        return np.clip(gates, 0.05, 1.0)

    def apply_development_transition(self, cell, delta: np.ndarray) -> None:
        lock_spec = self.compiled_genome.spec.epigenetic_lock
        candidate = 0.66 * cell.development_state + 0.5 * delta
        if cell.contact_state is not None:
            candidate += 0.15 * cell.contact_state
        if self.config.enable_epigenetic_lock:
            preserve = np.sign(cell.development_state) * np.maximum(0.0, np.abs(cell.development_state) - np.abs(candidate))
            candidate += lock_spec.strength * cell.epigenetic_lock * preserve
            sign_flip = (np.sign(cell.development_state) != np.sign(candidate)).astype(float)
            candidate -= (lock_spec.revert_cost + cell.reprogramming_cost) * sign_flip * np.abs(cell.development_state)
            cell.epigenetic_lock = float(np.clip(lock_spec.decay * cell.epigenetic_lock + lock_spec.strength * np.mean(np.abs(candidate)), 0.05, 1.0))
        else:
            cell.epigenetic_lock = max(0.05, cell.epigenetic_lock * 0.98)
        cell.development_state = np.tanh(candidate)
        positive = np.maximum(cell.development_state, 0.0) + 1e-6
        cell.commitment = positive / positive.sum()
        cell.reprogramming_cost = float(
            np.clip(0.75 * cell.reprogramming_cost + 0.25 * lock_spec.reprogramming_penalty * cell.epigenetic_lock, 0.0, 1.0)
        )

    def update_commitment(self, cell, attractor_potentials: dict[str, float]) -> None:
        cell.attractor_potentials = attractor_potentials
        best_name = None
        best_value = -1.0
        for attractor in self.compiled_genome.attractors:
            value = float(attractor_potentials.get(attractor.name, 0.0))
            competent = 1.0
            index = self.compiled_genome.development_names.index(attractor.name) if attractor.name in self.compiled_genome.development_names else None
            if index is not None and cell.competence_state is not None:
                competent = float(cell.competence_state[index])
            threshold = max(self.compiled_genome.spec.fate_landscape.global_commitment_threshold, attractor.commitment_threshold)
            if value >= threshold and competent >= 0.2:
                cell.commitment_timer[attractor.name] = cell.commitment_timer.get(attractor.name, 0) + 1
            else:
                cell.commitment_timer[attractor.name] = max(0, cell.commitment_timer.get(attractor.name, 0) - 1)
            if value > best_value:
                best_name = attractor.name
                best_value = value
            if cell.commitment_timer.get(attractor.name, 0) >= attractor.stability_horizon:
                cell.previous_fate = cell.current_fate
                cell.current_fate = attractor.name
        if best_name is not None and cell.current_fate == "unlabeled":
            cell.current_fate = best_name

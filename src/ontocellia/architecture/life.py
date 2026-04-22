from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ontocellia.architecture.models import LifeProcessDecision, LocalContext, NeighborhoodState
from ontocellia.compiler import CompiledGenome
from ontocellia.config import OntocelliaConfig


@dataclass(slots=True)
class LifeProcessModel:
    config: OntocelliaConfig
    compiled_genome: CompiledGenome | None = None

    def resolve(self, cell, transition, local_context: LocalContext, neighborhood: NeighborhoodState) -> LifeProcessDecision:
        fields = local_context.diffusive_fields
        action_scores = {
            "divide": float(transition.division_score),
            "differentiate": float(max(transition.attractor_potentials.values()) if transition.attractor_potentials else 0.0),
            "migrate": float(np.linalg.norm(transition.movement)),
            "apoptose": float(transition.death_score),
            "dedifferentiate": float(fields.get("damage", 0.0) + cell.repair_signal + cell.reprogramming_cost),
            "quiesce": float(transition.quiescence_drive),
            "secrete": float(sum(transition.signal_emission.values())),
            "signal_neighbor": float(np.linalg.norm(transition.neighbor_signal) if transition.neighbor_signal is not None else 0.0),
        }
        form_community = float(
            np.clip(
                0.35 * neighborhood.graph_density
                + 0.3 * neighborhood.contact_area_mean
                + 0.2 * neighborhood.contact_persistence_mean
                + 0.1 * (1.0 - neighborhood.relative_density_mean)
                + 0.05 * cell.trust,
                0.0,
                1.0,
            )
        )
        fuse_score = float(np.clip(form_community * neighborhood.community_signal, 0.0, 1.0))
        should_divide = self.should_divide(cell, float(transition.division_score), fields, neighborhood.local_demand)
        should_die = self.should_die(cell, float(transition.death_score), fields)
        should_dedifferentiate = bool(
            fields.get("damage", 0.0) > 0.48 and cell.epigenetic_lock < 0.75 and cell.reprogramming_cost < 0.7
        )
        should_quiesce = bool(transition.quiescence_drive > 0.62)
        return LifeProcessDecision(
            action_scores=action_scores,
            should_divide=should_divide,
            should_die=should_die,
            should_dedifferentiate=should_dedifferentiate,
            should_quiesce=should_quiesce,
            form_community_score=form_community,
            fuse_score=fuse_score,
        )

    def should_divide(self, cell, score: float, fields: dict[str, float], local_demand: float) -> bool:
        if self.compiled_genome is None:
            if not self.config.resource_driven_division:
                return score > 0.45 and cell.energy > self.config.energy_floor
            if cell.energy < 0.42 or cell.lineage_cooldown > 0:
                return False
            if fields.get("crowding", 0.0) > 0.8 or fields.get("damage", 0.0) > 0.72:
                return False
            return score > self.config.division_threshold and local_demand > 0.02 and cell.energy > 0.52 and cell.current_fate in {"stem", "progenitor", "repair-active"}
        lineage_rules = self.compiled_genome.spec.lineage_rules
        if cell.lineage_cooldown > 0 or cell.energy < lineage_rules.division_energy_threshold:
            return False
        if fields.get("crowding", 0.0) > 0.86 or fields.get("damage", 0.0) > 0.8 or cell.quiescence_state > 0.82:
            return False
        divide_axis = float(np.dot(self.compiled_genome.divide_bias, np.maximum(cell.development_state, 0.0)))
        return score > self.config.division_threshold and (local_demand > 0.02 or divide_axis > 0.35)

    def should_die(self, cell, score: float, fields: dict[str, float]) -> bool:
        if self.compiled_genome is None:
            return (
                score > self.config.death_threshold
                or cell.energy < self.config.energy_floor
                or (fields.get("damage", 0.0) > 0.65 and cell.energy < 0.95)
                or (fields.get("damage", 0.0) > 0.92 and cell.stress > 0.35)
            )
        return (
            score > self.config.death_threshold
            or cell.energy < self.config.energy_floor
            or (fields.get("damage", 0.0) > 0.72 and cell.energy < 0.9)
            or (fields.get("crowding", 0.0) > 0.92 and cell.stress > 0.45)
        )

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ontocellia.architecture.models import CellTransition, GenomeInput
from ontocellia.compiler import CompiledGenome
from ontocellia.config import FIELD_NAMES, OntocelliaConfig


@dataclass(slots=True)
class GenomeProgram:
    config: OntocelliaConfig
    compiled_genome: CompiledGenome | None = None
    fate_basis: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.fate_basis = np.array(
            [
                [0.48, 0.1, -0.12, -0.25, 0.05, -0.3],
                [0.22, 0.08, 0.42, 0.18, 0.25, -0.05],
                [0.08, -0.08, 0.2, 0.58, 0.48, 0.02],
                [-0.05, 0.72, 0.24, 0.2, 0.58, -0.18],
            ],
            dtype=float,
        )

    def step(self, genome_input: GenomeInput, gene_context: dict[str, float]) -> CellTransition:
        cell = genome_input.self_state
        local_fields = genome_input.local_context.diffusive_fields
        gradients = genome_input.local_context.gradient_fields
        neighbor = genome_input.neighborhood_state
        history_state = genome_input.history_state
        if self.compiled_genome is not None and cell.development_state is not None:
            return self._step_spec(cell, local_fields, gradients, neighbor, history_state, gene_context)
        return self._step_legacy(cell, local_fields, gradients, neighbor, history_state, gene_context)

    def _step_legacy(
        self,
        cell,
        local_fields: dict[str, float],
        gradients: dict[str, np.ndarray],
        neighbor,
        history_state: np.ndarray,
        gene_context: dict[str, float],
    ) -> CellTransition:
        field_vector = np.array([local_fields.get(name, 0.0) for name in FIELD_NAMES], dtype=float)
        fate_logits = self.fate_basis @ field_vector
        fate_logits += 0.35 * neighbor.fate_mean
        fate_logits += np.array(
            [
                0.35 * (cell.energy - cell.stress),
                0.15 * neighbor.local_demand,
                0.25 * neighbor.graph_density,
                0.45 * local_fields.get("damage", 0.0) + 0.2 * neighbor.local_demand,
            ],
            dtype=float,
        )
        fate_logits += np.array(
            [
                -0.12 * cell.age / 25,
                0.12 * cell.energy + 0.08 * local_fields.get("task_pressure", 0.0),
                0.15 * np.linalg.norm(neighbor.hidden_mean[:2]) + 0.12 * neighbor.graph_density + 0.08 * max(0.0, cell.age - 4) / 15,
                0.2 * cell.local_memory[1],
            ],
            dtype=float,
        )
        if history_state.size:
            fate_logits += np.pad(history_state[:4], (0, max(0, fate_logits.size - min(4, history_state.size))))[: fate_logits.size] * 0.05
        fate_logits += self._gene_bias(gene_context)
        if self.config.enable_competence:
            fate_logits *= np.clip(cell.competence, 0.1, 1.0)

        movement = self._movement_vector(gradients, local_fields, gene_context)
        division_score = self._division_score(cell, local_fields, neighbor.local_demand, neighbor.graph_density, gene_context)
        death_score = self._death_score(cell, local_fields) - 0.08 * gene_context.get("warning", 0.0)
        edge_intent = float(
            np.clip(
                0.4 + 0.25 * neighbor.graph_density + 0.1 * cell.trust - 0.2 * cell.stress - 0.1 * gene_context.get("warning", 0.0),
                0.0,
                1.0,
            )
        )
        hidden_state = np.tanh(
            0.72 * cell.hidden_state
            + 0.1 * np.pad(field_vector[: cell.hidden_state.size], (0, max(0, cell.hidden_state.size - field_vector.size)))[: cell.hidden_state.size]
            + 0.08 * neighbor.hidden_mean[: cell.hidden_state.size]
        )
        local_memory = np.tanh(
            0.7 * cell.local_memory
            + np.pad(
                np.array(
                    [
                        local_fields.get("task_pressure", 0.0) - local_fields.get("crowding", 0.0),
                        local_fields.get("damage", 0.0) + neighbor.local_demand,
                        cell.energy - cell.stress,
                        neighbor.graph_density - 0.5,
                    ]
                ),
                (0, max(0, cell.local_memory.size - 4)),
            )[: cell.local_memory.size]
            * 0.2
        )
        signal_emission = {
            "morphogen_a": max(0.0, 0.6 * fate_logits[0] + 0.4 * cell.energy) * self.config.emission_scale,
            "morphogen_b": max(0.0, 0.8 * fate_logits[2] + 0.7 * neighbor.local_demand + 0.2 * neighbor.graph_density) * self.config.emission_scale,
            "task_pressure": max(0.0, 0.3 * neighbor.local_demand + 0.2 * local_fields.get("damage", 0.0)) * self.config.emission_scale,
            "damage": max(0.0, death_score - 0.65) * self.config.emission_scale * 0.7,
            "nutrient": max(0.0, 0.35 * cell.energy - 0.15 * local_fields.get("damage", 0.0)) * self.config.emission_scale * 0.5,
        }
        energy_delta = 0.11 * local_fields.get("nutrient", 0.0) - 0.04 * local_fields.get("crowding", 0.0) - 0.02 * np.linalg.norm(movement)
        stress_delta = 0.18 * local_fields.get("damage", 0.0) + 0.05 * local_fields.get("task_pressure", 0.0) - 0.07 * local_fields.get("nutrient", 0.0)
        return CellTransition(
            hidden_state=hidden_state,
            local_memory=local_memory,
            signal_emission=signal_emission,
            neighbor_signal=None,
            movement=movement,
            division_score=float(np.clip(division_score, 0.0, 1.5)),
            death_score=float(np.clip(death_score, 0.0, 1.5)),
            fate_logits=fate_logits,
            edge_intent=edge_intent,
            energy_delta=float(energy_delta),
            stress_delta=float(stress_delta),
        )

    def _step_spec(
        self,
        cell,
        local_fields: dict[str, float],
        gradients: dict[str, np.ndarray],
        neighbor,
        history_state: np.ndarray,
        gene_context: dict[str, float],
    ) -> CellTransition:
        compiled = self.compiled_genome
        assert compiled is not None
        field_vector = np.array([local_fields.get(name, 0.0) for name in compiled.field_order], dtype=float)
        if cell.receptor_profile is not None:
            receptor = np.pad(cell.receptor_profile, (0, max(0, field_vector.size - cell.receptor_profile.size)))[: field_vector.size]
            field_vector *= np.clip(receptor, 0.0, 2.0)
        gradient_norms = np.array([float(np.linalg.norm(gradients.get(name, np.zeros(2, dtype=float)))) for name in compiled.field_order], dtype=float)

        development_delta = compiled.field_matrix @ field_vector
        development_delta += compiled.gradient_matrix @ gradient_norms
        development_delta += compiled.background_matrix @ field_vector
        development_delta += compiled.neighbor_development_weight * neighbor.development_mean[: compiled.development_dim]
        development_delta += compiled.graph_weight * neighbor.graph_density
        development_delta += compiled.local_demand_weight * neighbor.local_demand
        development_delta += compiled.task_bias * local_fields.get("task_pressure", 0.0)
        development_delta += compiled.hidden_feedback_weight * np.pad(neighbor.hidden_mean, (0, max(0, compiled.development_dim - neighbor.hidden_mean.size)))[: compiled.development_dim]
        development_delta += compiled.memory_feedback_weight * np.pad(cell.local_memory, (0, max(0, compiled.development_dim - cell.local_memory.size)))[: compiled.development_dim]
        development_delta += 0.08 * cell.development_state
        if history_state.size:
            development_delta += 0.04 * np.pad(history_state, (0, max(0, compiled.development_dim - history_state.size)))[: compiled.development_dim]

        contact_signal = np.zeros(compiled.development_dim, dtype=float)
        for program in compiled.contact_programs:
            signal = program.sender_weights * neighbor.development_mean[: compiled.development_dim]
            signal += program.receiver_weights * cell.development_state
            signal *= 1.0 + program.contact_area_weight * neighbor.contact_area_mean + program.persistence_weight * neighbor.contact_persistence_mean
            signal -= program.crowding_weight * neighbor.relative_density_mean
            signal -= program.inhibition_strength * neighbor.contact_inhibition
            signal += program.activation_strength * (1.0 - neighbor.neighbor_quiescence + 0.25 * neighbor.community_signal)
            if program.quiescence_threshold and neighbor.neighbor_quiescence >= program.quiescence_threshold:
                signal -= 0.15 * np.abs(signal)
            contact_signal += signal
        development_delta += contact_signal

        if self.config.enable_competence and cell.competence_state is not None:
            development_delta *= np.clip(cell.competence_state, 0.05, 1.0)

        move_propensity = float(np.dot(compiled.move_bias, np.maximum(cell.development_state, 0.0)))
        quiescence_drive = float(
            np.clip(
                0.3 * np.dot(compiled.quiesce_bias, np.maximum(cell.development_state, 0.0))
                + 0.25 * neighbor.neighbor_quiescence
                + 0.2 * neighbor.contact_inhibition
                + 0.08 * neighbor.community_signal,
                0.0,
                1.0,
            )
        )
        movement = self._movement_vector(
            gradients,
            local_fields,
            gene_context,
            move_scale=np.clip(0.55 + 0.35 * move_propensity - 0.25 * quiescence_drive, 0.1, 1.1),
        )
        divide_score = (
            0.32 * cell.energy
            + 0.28 * local_fields.get("nutrient", 0.0)
            + 0.18 * neighbor.local_demand
            - 0.3 * local_fields.get("damage", 0.0)
            - 0.22 * local_fields.get("crowding", 0.0)
            + 0.24 * float(np.dot(compiled.divide_bias, cell.development_state))
            - 0.18 * neighbor.contact_inhibition
            + 0.12 * gene_context.get("strategy", 0.0)
            - 0.18 * gene_context.get("warning", 0.0)
            + 0.08 * neighbor.community_signal
        )
        death_score = (
            0.46 * local_fields.get("damage", 0.0)
            + 0.22 * local_fields.get("crowding", 0.0)
            + 0.18 * cell.stress
            - 0.22 * local_fields.get("nutrient", 0.0)
            + 0.2 * float(np.dot(compiled.die_bias, np.maximum(cell.development_state, 0.0)))
            - 0.06 * gene_context.get("warning", 0.0)
        )
        edge_intent = float(
            np.clip(
                0.35
                + 0.25 * neighbor.graph_density
                + 0.18 * float(np.dot(compiled.rewire_bias, np.maximum(cell.development_state, 0.0)))
                + 0.15 * neighbor.contact_area_mean
                + 0.1 * neighbor.contact_persistence_mean
                + 0.12 * neighbor.community_signal
                - 0.08 * gene_context.get("warning", 0.0),
                0.0,
                1.0,
            )
        )

        hidden_state = np.tanh(
            0.72 * cell.hidden_state
            + 0.08 * np.pad(field_vector, (0, max(0, cell.hidden_state.size - field_vector.size)))[: cell.hidden_state.size]
            + 0.06 * neighbor.hidden_mean[: cell.hidden_state.size]
            + 0.08 * np.pad(cell.development_state, (0, max(0, cell.hidden_state.size - cell.development_state.size)))[: cell.hidden_state.size]
            + 0.06 * np.pad(contact_signal, (0, max(0, cell.hidden_state.size - contact_signal.size)))[: cell.hidden_state.size]
        )
        local_memory = np.tanh(
            0.74 * cell.local_memory
            + np.pad(
                np.array(
                    [
                        local_fields.get("task_pressure", 0.0) - local_fields.get("crowding", 0.0),
                        local_fields.get("damage", 0.0) + neighbor.local_demand,
                        cell.energy - cell.stress,
                        neighbor.graph_density - 0.5 + neighbor.contact_inhibition + neighbor.community_signal,
                    ]
                ),
                (0, max(0, cell.local_memory.size - 4)),
            )[: cell.local_memory.size]
            * 0.22
        )

        secretion_base = 0.45 + 0.2 * float(np.dot(compiled.secrete_bias, np.maximum(cell.development_state, 0.0)))
        emissions: dict[str, float] = {}
        for program in compiled.secretion_programs:
            emission = float(np.dot(program.development_weights, np.maximum(cell.development_state, 0.0)))
            emission += sum(local_fields.get(name, 0.0) * weight for name, weight in program.field_weights.items())
            emission += float(np.dot(program.memory_weights, cell.local_memory[: program.memory_weights.size]))
            emission = max(0.0, emission * secretion_base) * self.config.emission_scale
            if emission > 0:
                emissions[program.field] = emissions.get(program.field, 0.0) + emission

        attractor_potentials = self._attractor_potentials(compiled, cell.development_state + development_delta * 0.15)
        repair_score = float(
            0.4 * local_fields.get("damage", 0.0)
            + 0.25 * local_fields.get("task_pressure", 0.0)
            + 0.28 * np.dot(compiled.repair_bias, np.maximum(cell.development_state, 0.0))
        )
        energy_delta = (
            0.1 * local_fields.get("nutrient", 0.0)
            - 0.035 * local_fields.get("crowding", 0.0)
            - 0.02 * np.linalg.norm(movement)
            - 0.03 * max(0.0, divide_score - self.config.division_threshold)
            - 0.025 * quiescence_drive
        )
        stress_delta = (
            0.16 * local_fields.get("damage", 0.0)
            + 0.05 * local_fields.get("task_pressure", 0.0)
            + 0.05 * neighbor.contact_inhibition
            - 0.08 * local_fields.get("nutrient", 0.0)
        )
        probe_scores = self._probe_scores(compiled, cell.development_state, local_fields, cell.energy, cell.stress)
        return CellTransition(
            hidden_state=hidden_state,
            local_memory=local_memory,
            signal_emission=emissions,
            neighbor_signal=contact_signal,
            movement=movement,
            division_score=float(np.clip(divide_score, 0.0, 1.5)),
            death_score=float(np.clip(death_score, 0.0, 1.5)),
            fate_logits=np.zeros_like(cell.commitment),
            edge_intent=edge_intent,
            energy_delta=float(energy_delta),
            stress_delta=float(stress_delta),
            development_delta=development_delta,
            repair_score=repair_score,
            probe_scores=probe_scores,
            attractor_potentials=attractor_potentials,
            quiescence_drive=quiescence_drive,
        )

    def _movement_vector(
        self,
        gradients: dict[str, np.ndarray],
        local_fields: dict[str, float],
        gene_context: dict[str, float],
        *,
        move_scale: float | None = None,
    ) -> np.ndarray:
        nutrient_vec = gradients.get("nutrient", np.zeros(2, dtype=float)) + 0.4 * gradients.get("M2", np.zeros(2, dtype=float)) + 0.5 * gradients.get("morphogen_b", np.zeros(2, dtype=float))
        damage_vec = gradients.get("damage", np.zeros(2, dtype=float)) * (1.0 + 1.8 * gene_context.get("warning", 0.0)) + 0.3 * gradients.get("crowding", np.zeros(2, dtype=float))
        task_vec = gradients.get("task_pressure", np.zeros(2, dtype=float)) + 0.35 * gradients.get("M3", np.zeros(2, dtype=float)) + 0.3 * gradients.get("morphogen_a", np.zeros(2, dtype=float))
        move = nutrient_vec + task_vec - 1.15 * damage_vec
        if np.linalg.norm(move) > 1e-6:
            move = move / np.linalg.norm(move)
        scale = move_scale if move_scale is not None else np.clip(0.4 + local_fields.get("task_pressure", 0.0), 0.2, 1.0)
        return move * self.config.movement_scale * scale

    def _division_score(self, cell, local_fields: dict[str, float], local_demand: float, graph_density: float, gene_context: dict[str, float]) -> float:
        score = 0.45 * cell.energy + 0.35 * local_demand + 0.15 * graph_density
        score += 0.08 * local_fields.get("nutrient", 0.0) - 0.35 * local_fields.get("damage", 0.0) - 0.3 * local_fields.get("crowding", 0.0)
        score += 0.05 * max(cell.current_fate == "progenitor", cell.current_fate == "repair-active")
        score += 0.15 * gene_context.get("strategy", 0.0)
        score -= 0.25 * gene_context.get("warning", 0.0)
        if cell.lineage_cooldown > 0:
            score -= 0.3
        return score

    def _death_score(self, cell, local_fields: dict[str, float]) -> float:
        return (
            0.82 * local_fields.get("damage", 0.0)
            + 0.25 * cell.stress
            + 0.15 * local_fields.get("crowding", 0.0)
            + 0.15 * max(0.0, self.config.energy_floor - cell.energy)
            - 0.2 * local_fields.get("nutrient", 0.0)
        )

    def _gene_bias(self, gene_context: dict[str, float]) -> np.ndarray:
        strategy = gene_context.get("strategy", 0.0)
        warning = gene_context.get("warning", 0.0)
        return np.array(
            [-0.05 * warning, 0.08 * strategy, 0.05 * strategy, 0.2 * strategy + 0.12 * warning],
            dtype=float,
        )

    def _probe_scores(self, compiled: CompiledGenome, development_state: np.ndarray, local_fields: dict[str, float], energy: float, stress: float) -> dict[str, float]:
        scores: dict[str, float] = {}
        for probe in compiled.probes:
            score = float(np.dot(probe.development_weights, development_state))
            score += sum(local_fields.get(name, 0.0) * weight for name, weight in probe.field_weights.items())
            score += probe.energy_weight * energy
            score -= probe.stress_weight * stress
            scores[probe.name] = 1.0 / (1.0 + np.exp(-score))
        return scores

    def _attractor_potentials(self, compiled: CompiledGenome, development_state: np.ndarray) -> dict[str, float]:
        potentials: dict[str, float] = {}
        for attractor in compiled.attractors:
            distance = float(np.linalg.norm(development_state - attractor.center))
            potentials[attractor.name] = float(np.exp(-attractor.basin_sharpness * distance))
        return potentials

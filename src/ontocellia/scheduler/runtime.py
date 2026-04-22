from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ontocellia.cells import CellState, GenomeKernel
from ontocellia.compiler import BuiltEnvironment, CompiledGenome, EnvironmentBuilder, GenomeSpecCompiler
from ontocellia.config import FATE_NAMES, GeneAsset, LEGACY_MODE, OntocelliaConfig, SPEC_MODE
from ontocellia.fate import FateEngine
from ontocellia.fields import Microenvironment
from ontocellia.genes import GeneRegistry
from ontocellia.graph import InteractionGraph
from ontocellia.metrics import MetricsRecorder
from ontocellia.specs import EnvironmentSpec, GenomeSpec, load_environment_spec, load_genome_spec
from ontocellia.substrate import SpatialSubstrate


@dataclass(slots=True)
class OntocelliaRuntime:
    config: OntocelliaConfig
    genome_spec: GenomeSpec | None = None
    environment_spec: EnvironmentSpec | None = None
    rng: np.random.Generator = field(init=False)
    substrate: SpatialSubstrate = field(init=False)
    environment: Microenvironment = field(init=False)
    graph: InteractionGraph = field(init=False)
    kernel: GenomeKernel = field(init=False)
    fate_engine: FateEngine | None = field(init=False, default=None)
    gene_registry: GeneRegistry = field(init=False)
    metrics: MetricsRecorder = field(init=False)
    cells: dict[int, CellState] = field(default_factory=dict, init=False)
    tick_count: int = 0
    next_cell_id: int = 0
    total_deaths: int = 0
    division_events: int = 0
    risky_divisions: int = 0
    fate_switches: int = 0
    lineage_edges: list[tuple[int, int]] = field(default_factory=list, init=False)
    goals: list[dict[str, float | tuple[float, float]]] = field(default_factory=list, init=False)
    mode: str = field(init=False, default=LEGACY_MODE)
    compiled_genome: CompiledGenome | None = field(init=False, default=None)
    built_environment: BuiltEnvironment | None = field(init=False, default=None)

    @classmethod
    def from_specs(
        cls,
        genome_spec: GenomeSpec,
        environment_spec: EnvironmentSpec,
        sim_config: OntocelliaConfig | None = None,
    ) -> "OntocelliaRuntime":
        config = sim_config or OntocelliaConfig()
        config.width = environment_spec.grid.width
        config.height = environment_spec.grid.height
        config.hidden_dim = genome_spec.state_dims.hidden_dim
        config.local_memory_dim = genome_spec.state_dims.memory_dim
        return cls(config=config, genome_spec=genome_spec, environment_spec=environment_spec)

    @classmethod
    def from_spec_files(
        cls,
        genome_path: str | Path,
        environment_path: str | Path,
        sim_config: OntocelliaConfig | None = None,
    ) -> "OntocelliaRuntime":
        genome_spec = load_genome_spec(genome_path)
        environment_spec = load_environment_spec(environment_path)
        return cls.from_specs(genome_spec, environment_spec, sim_config=sim_config)

    def __post_init__(self) -> None:
        self.rng = self.config.rng()
        self.substrate = SpatialSubstrate(self.config)
        self.graph = InteractionGraph(self.config)
        self.gene_registry = GeneRegistry(list(self.config.default_genes))
        self.metrics = MetricsRecorder()
        if self.genome_spec is not None and self.environment_spec is not None:
            self.mode = SPEC_MODE
            self._initialize_spec_mode()
        else:
            self.mode = LEGACY_MODE
            self._initialize_legacy_mode()
        self.metrics.record(self)

    def _initialize_legacy_mode(self) -> None:
        self.environment = Microenvironment(self.config)
        self.kernel = GenomeKernel(self.config)
        self.fate_engine = FateEngine(self.config)
        self.seed_cells(self.config.initial_cells)

    def _initialize_spec_mode(self) -> None:
        assert self.genome_spec is not None and self.environment_spec is not None
        builder = EnvironmentBuilder()
        self.built_environment = builder.build(self.environment_spec)
        self.config.communication_radius = self.built_environment.contact_context.get("contact_radius", self.config.communication_radius)
        compiler = GenomeSpecCompiler()
        self.compiled_genome = compiler.compile(self.genome_spec, self.built_environment.field_names)
        self.environment = Microenvironment(
            self.config,
            initial_fields=self.built_environment.initial_fields,
            field_params=self.built_environment.field_params,
        )
        self.kernel = GenomeKernel(self.config, compiled_genome=self.compiled_genome)
        self.fate_engine = None
        self.seed_spec_cells(self.config.initial_cells)
        self.goals.append({"task": self.built_environment.global_task["text"], "objective": self.built_environment.global_task["objective"]})

    def seed_cells(self, count: int) -> None:
        center = np.array([self.config.width / 2, self.config.height / 2], dtype=float)
        for _ in range(count):
            pos = self.substrate.nearby_position(self.rng, center, scale=3.2)
            hidden = self.rng.normal(0.0, 0.15, size=self.config.hidden_dim)
            commitment = np.array([0.68, 0.18, 0.08, 0.06], dtype=float)
            cell = CellState(
                id=self.next_cell_id,
                pos=pos,
                hidden_state=hidden,
                fate_dist=commitment.copy(),
                commitment=commitment,
                energy=float(self.rng.uniform(0.45, 0.7)),
                stress=float(self.rng.uniform(0.05, 0.12)),
                age=0,
                competence=np.clip(self.rng.normal(0.85, 0.05, size=4), 0.35, 1.0),
                epigenetic_lock=float(self.rng.uniform(0.1, 0.2)),
                lineage_parent=None,
                local_memory=self.rng.normal(0.0, 0.05, size=self.config.local_memory_dim),
            )
            self.cells[cell.id] = cell
            self.next_cell_id += 1

    def seed_spec_cells(self, count: int) -> None:
        assert self.compiled_genome is not None
        center = np.array([self.config.width / 2, self.config.height / 2], dtype=float)
        development_dim = self.compiled_genome.development_dim
        for _ in range(count):
            pos = self.substrate.nearby_position(self.rng, center, scale=3.2)
            hidden = self.rng.normal(0.0, 0.12, size=self.config.hidden_dim)
            development_state = self.rng.normal(0.0, 0.08, size=development_dim)
            positive = np.maximum(development_state, 0.0) + 1e-6
            commitment = positive / positive.sum()
            cell = CellState(
                id=self.next_cell_id,
                pos=pos,
                hidden_state=hidden,
                fate_dist=np.zeros(4, dtype=float),
                commitment=commitment,
                energy=float(self.rng.uniform(0.45, 0.72)),
                stress=float(self.rng.uniform(0.04, 0.1)),
                age=0,
                competence=np.ones(development_dim, dtype=float) * 0.9,
                epigenetic_lock=float(self.rng.uniform(0.1, 0.2)),
                lineage_parent=None,
                local_memory=self.rng.normal(0.0, 0.04, size=self.config.local_memory_dim),
                current_fate="unlabeled",
                previous_fate="unlabeled",
                development_state=development_state,
                competence_state=np.ones(development_dim, dtype=float),
                phenotype_label="unlabeled",
                contact_state=np.zeros(development_dim, dtype=float),
                commitment_timer={attractor.name: 0 for attractor in self.compiled_genome.attractors},
                attractor_potentials={attractor.name: 0.0 for attractor in self.compiled_genome.attractors},
            )
            self.cells[cell.id] = cell
            self.next_cell_id += 1

    def step(self, steps: int = 1) -> None:
        for _ in range(steps):
            if not self.cells:
                self.tick_count += 1
                self.metrics.record(self)
                continue
            if self.mode == SPEC_MODE:
                self._step_once_spec()
            else:
                self._step_once_legacy()

    def _step_once_legacy(self) -> None:
        self.tick_count += 1
        crowding = self.substrate.crowding_map(self.cells)
        self.environment.set_crowding(crowding)
        self.environment.diffuse()
        self.graph.rebuild(self.cells)

        actions: dict[int, dict[str, object]] = {}
        for cell_id, cell in list(self.cells.items()):
            local_fields = self.environment.sample(cell.pos)
            gradients = self.environment.gradients(cell.pos)
            neighbor_summary = self.graph.summary_for(cell_id, self.cells, self.config.desired_local_density)
            gene_state = self.gene_registry.evaluate(local_fields)
            cell.active_genes = list(gene_state["active_names"])
            kernel_output = self.kernel.step(cell, local_fields, gradients, neighbor_summary, gene_state)
            actions[cell_id] = {
                "fields": local_fields,
                "kernel": kernel_output,
                "gene": gene_state,
                "neighbor_summary": neighbor_summary,
            }

        deaths: list[tuple[float, float]] = []
        pending_children: list[CellState] = []
        for cell_id, record in actions.items():
            if cell_id not in self.cells:
                continue
            cell = self.cells[cell_id]
            fields = record["fields"]
            output = record["kernel"]
            prev_fate = cell.current_fate
            assert self.fate_engine is not None
            self.fate_engine.update(cell, output.fate_logits)
            if cell.current_fate != prev_fate:
                self.fate_switches += 1

            cell.hidden_state = output.hidden_state
            cell.local_memory = output.local_memory
            cell.age += 1
            cell.lineage_cooldown = max(0, cell.lineage_cooldown - 1)
            cell.energy = float(np.clip(cell.energy + output.energy_delta, 0.0, 1.2))
            cell.stress = float(np.clip(cell.stress + output.stress_delta, 0.0, 1.2))
            cell.trust = float(np.clip(0.9 * cell.trust + 0.1 * output.edge_intent, 0.0, 1.0))

            if self.config.enable_competence:
                cell.competence = self._update_competence_legacy(cell, fields)

            if self.config.enable_spatial:
                cell.pos = self.substrate.clamp(cell.pos + output.movement)

            self.environment.emit(cell.pos, output.signal_emission)

            if self._should_divide_legacy(cell, output.division_score, fields, record["neighbor_summary"]["local_demand"]):
                self.division_events += 1
                if fields.get("damage", 0.0) > 0.12 or fields.get("crowding", 0.0) > 0.35:
                    self.risky_divisions += 1
                child_pos = self.substrate.nearby_position(self.rng, cell.pos, scale=1.2)
                child_noise = self.rng.normal(0.0, self.config.mutation_noise, size=max(self.config.hidden_dim, self.config.local_memory_dim))
                child = cell.clone(self.next_cell_id, child_pos, child_noise)
                child.current_fate = "progenitor" if cell.current_fate == "stem" else cell.current_fate
                pending_children.append(child)
                self.lineage_edges.append((cell.id, child.id))
                self.next_cell_id += 1
                cell.energy = max(0.12, cell.energy * 0.55)
                cell.lineage_cooldown = self.config.lineage_cooldown_steps

            if self._should_die_legacy(cell, output.death_score, fields):
                deaths.append(tuple(cell.pos.tolist()))
                self.total_deaths += 1
                del self.cells[cell_id]
                continue

        for child in pending_children:
            self.cells[child.id] = child

        self.graph.rebuild(self.cells)
        self._repair_response_legacy(deaths)
        self.metrics.record(self)

    def _step_once_spec(self) -> None:
        self.tick_count += 1
        self._dispatch_spec_events()
        crowding = self.substrate.crowding_map(self.cells)
        self.environment.set_crowding(crowding)
        self.environment.diffuse()
        self.graph.rebuild(self.cells)

        actions: dict[int, dict[str, object]] = {}
        for cell_id, cell in list(self.cells.items()):
            local_fields = self.environment.sample(cell.pos)
            gradients = self.environment.gradients(cell.pos)
            neighbor_summary = self.graph.summary_for(cell_id, self.cells, self.config.desired_local_density)
            gene_state = self.gene_registry.evaluate(local_fields)
            cell.active_genes = list(gene_state["active_names"])
            kernel_output = self.kernel.step(cell, local_fields, gradients, neighbor_summary, gene_state)
            actions[cell_id] = {
                "fields": local_fields,
                "kernel": kernel_output,
                "neighbor_summary": neighbor_summary,
            }

        deaths: list[tuple[float, float]] = []
        pending_children: list[CellState] = []
        for cell_id, record in actions.items():
            if cell_id not in self.cells:
                continue
            cell = self.cells[cell_id]
            fields = record["fields"]
            output = record["kernel"]

            cell.hidden_state = output.hidden_state
            cell.local_memory = output.local_memory
            cell.age += 1
            cell.lineage_cooldown = max(0, cell.lineage_cooldown - 1)
            cell.energy = float(np.clip(cell.energy + output.energy_delta, 0.0, 1.3))
            cell.stress = float(np.clip(cell.stress + output.stress_delta, 0.0, 1.2))
            cell.trust = float(np.clip(0.88 * cell.trust + 0.12 * output.edge_intent, 0.0, 1.0))
            cell.repair_signal = float(np.clip(0.65 * cell.repair_signal + 0.35 * output.repair_score, 0.0, 1.0))
            cell.competence_state = self._update_competence_spec(cell, fields)
            if output.neighbor_signal is not None:
                cell.contact_state = output.neighbor_signal
            cell.quiescence_state = float(np.clip(0.7 * cell.quiescence_state + 0.3 * output.quiescence_drive, 0.0, 1.0))
            self._update_development_state(cell, output.development_delta if output.development_delta is not None else np.zeros_like(cell.development_state))
            self._update_attractor_commitment(cell, output.attractor_potentials or {})
            self._update_probe_labels(cell, output.probe_scores or {})

            if self.config.enable_spatial:
                cell.pos = self.substrate.clamp(cell.pos + output.movement)

            self.environment.emit(cell.pos, output.signal_emission)

            local_demand = float(record["neighbor_summary"]["local_demand"])
            if self._should_divide_spec(cell, output.division_score, fields, local_demand):
                self.division_events += 1
                if fields.get("damage", 0.0) > 0.12 or fields.get("crowding", 0.0) > 0.35:
                    self.risky_divisions += 1
                child_pos = self.substrate.nearby_position(self.rng, cell.pos, scale=1.15)
                noise_scale = self.compiled_genome.spec.lineage_rules.noise_scale
                child_noise = self.rng.normal(0.0, noise_scale, size=max(self.config.hidden_dim, self.config.local_memory_dim, self.compiled_genome.development_dim))
                child = cell.clone(self.next_cell_id, child_pos, child_noise)
                pending_children.append(child)
                self.lineage_edges.append((cell.id, child.id))
                self.next_cell_id += 1
                parent_share = self.compiled_genome.spec.lineage_rules.parent_energy_share
                child.energy = max(0.08, min(1.0, cell.energy * self.compiled_genome.spec.lineage_rules.child_energy_share))
                cell.energy = max(0.08, cell.energy * parent_share)
                cell.lineage_cooldown = self.compiled_genome.spec.lineage_rules.cooldown_steps

            if self._should_die_spec(cell, output.death_score, fields):
                deaths.append(tuple(cell.pos.tolist()))
                self.total_deaths += 1
                del self.cells[cell_id]
                continue

        for child in pending_children:
            self.cells[child.id] = child

        self.graph.rebuild(self.cells)
        self._repair_response_spec(deaths)
        self.metrics.record(self)

    def _dispatch_spec_events(self) -> None:
        if self.built_environment is None:
            return
        for event in self.built_environment.events:
            if self.tick_count == event.step or (event.repeat_every and self.tick_count >= event.step and (self.tick_count - event.step) % event.repeat_every == 0):
                self.environment.apply_event(event.action, event.center, event.radius, event.intensity, field=event.field)

    def _update_competence_legacy(self, cell: CellState, fields: dict[str, float]) -> np.ndarray:
        age_factor = np.clip(1.0 - cell.age / 55, 0.05, 1.0)
        stem_window = np.clip(age_factor + 0.15 * fields.get("morphogen_a", 0.0) - 0.35 * fields.get("damage", 0.0), 0.05, 1.0)
        progenitor_window = np.clip(0.12 + 0.75 * fields.get("task_pressure", 0.0) + 0.15 * fields.get("damage", 0.0), 0.05, 1.0)
        specialist_window = np.clip(
            0.1 + 0.7 * fields.get("morphogen_b", 0.0) + 0.2 * cell.energy + 0.15 * min(cell.age / 20, 1.0),
            0.05,
            1.0,
        )
        repair_window = np.clip(0.05 + 1.15 * fields.get("damage", 0.0) + 0.35 * fields.get("task_pressure", 0.0) - 0.2 * fields.get("crowding", 0.0), 0.05, 1.0)
        windows = np.array([stem_window, progenitor_window, specialist_window, repair_window], dtype=float)
        return np.power(windows, 1.6)

    def _update_competence_spec(self, cell: CellState, fields: dict[str, float]) -> np.ndarray:
        if self.compiled_genome is None:
            return np.ones_like(cell.competence)
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
            gates[gate.dimension] = 1.0 / (1.0 + np.exp(-signal))
        return np.clip(gates, 0.05, 1.0)

    def _update_development_state(self, cell: CellState, delta: np.ndarray) -> None:
        assert self.compiled_genome is not None and cell.development_state is not None
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
        cell.reprogramming_cost = float(np.clip(0.75 * cell.reprogramming_cost + 0.25 * lock_spec.reprogramming_penalty * cell.epigenetic_lock, 0.0, 1.0))

    def _update_attractor_commitment(self, cell: CellState, attractor_potentials: dict[str, float]) -> None:
        if self.compiled_genome is None:
            return
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

    def _update_probe_labels(self, cell: CellState, probe_scores: dict[str, float]) -> None:
        cell.phenotype_scores = probe_scores
        if not probe_scores:
            cell.phenotype_label = cell.current_fate or "unlabeled"
            return
        label, score = max(probe_scores.items(), key=lambda item: item[1])
        if score >= 0.5:
            cell.phenotype_label = label
        elif cell.current_fate and cell.current_fate != "unlabeled":
            cell.phenotype_label = cell.current_fate
        else:
            cell.phenotype_label = "unlabeled"

    def _should_divide_legacy(self, cell: CellState, score: float, fields: dict[str, float], local_demand: float) -> bool:
        if not self.config.resource_driven_division:
            return score > 0.45 and cell.energy > self.config.energy_floor
        if cell.energy < 0.42 or cell.lineage_cooldown > 0:
            return False
        if fields.get("crowding", 0.0) > 0.8 or fields.get("damage", 0.0) > 0.72:
            return False
        return (
            score > self.config.division_threshold
            and local_demand > 0.02
            and cell.energy > 0.52
            and cell.current_fate in {"stem", "progenitor", "repair-active"}
        )

    def _should_divide_spec(self, cell: CellState, score: float, fields: dict[str, float], local_demand: float) -> bool:
        assert self.compiled_genome is not None
        lineage_rules = self.compiled_genome.spec.lineage_rules
        if cell.lineage_cooldown > 0 or cell.energy < lineage_rules.division_energy_threshold:
            return False
        if fields.get("crowding", 0.0) > 0.86 or fields.get("damage", 0.0) > 0.8 or cell.quiescence_state > 0.82:
            return False
        divide_axis = float(np.dot(self.compiled_genome.divide_bias, np.maximum(cell.development_state, 0.0)))
        return score > self.config.division_threshold and (local_demand > 0.02 or divide_axis > 0.35)

    def _should_die_legacy(self, cell: CellState, score: float, fields: dict[str, float]) -> bool:
        return (
            score > self.config.death_threshold
            or cell.energy < self.config.energy_floor
            or (fields.get("damage", 0.0) > 0.65 and cell.energy < 0.95)
            or (fields.get("damage", 0.0) > 0.92 and cell.stress > 0.35)
        )

    def _should_die_spec(self, cell: CellState, score: float, fields: dict[str, float]) -> bool:
        return (
            score > self.config.death_threshold
            or cell.energy < self.config.energy_floor
            or (fields.get("damage", 0.0) > 0.72 and cell.energy < 0.9)
            or (fields.get("crowding", 0.0) > 0.92 and cell.stress > 0.45)
        )

    def _repair_response_legacy(self, deaths: list[tuple[float, float]]) -> None:
        if not deaths:
            return
        for x, y in deaths:
            self.environment.inject_goal_pressure((x, y), radius=3.0, intensity=0.2)
            self.environment.inject_resource_pulse((x, y), radius=2.0, intensity=0.12)
            if "damage" in self.environment.fields:
                self.environment.fields["damage"] = np.clip(self.environment.fields["damage"] * (1.0 - self.config.damage_repair_decay), 0.0, 1.0)
            death_pos = np.array([x, y], dtype=float)
            for cell in self.cells.values():
                distance = float(np.linalg.norm(cell.pos - death_pos))
                if distance <= self.config.communication_radius:
                    cell.local_memory[1] = float(np.clip(cell.local_memory[1] + self.config.repair_boost * (1.0 - distance / self.config.communication_radius), -1.0, 1.0))
                    cell.commitment[3] += 0.08 * (1.0 - distance / self.config.communication_radius)
                    cell.commitment /= cell.commitment.sum()
                    if cell.current_fate in {"stem", "progenitor"} and distance <= self.config.spatial_radius:
                        cell.current_fate = "repair-active"
                        cell.energy = min(1.0, cell.energy + 0.08)

    def _repair_response_spec(self, deaths: list[tuple[float, float]]) -> None:
        if not deaths or self.compiled_genome is None:
            return
        for x, y in deaths:
            self.environment.inject_goal_pressure((x, y), radius=3.0, intensity=0.18)
            self.environment.inject_resource_pulse((x, y), radius=2.2, intensity=0.14)
            death_pos = np.array([x, y], dtype=float)
            for cell in self.cells.values():
                distance = float(np.linalg.norm(cell.pos - death_pos))
                if distance <= self.config.communication_radius and cell.development_state is not None:
                    proximity = 1.0 - distance / self.config.communication_radius
                    cell.local_memory[0] = float(np.clip(cell.local_memory[0] + 0.18 * proximity, -1.0, 1.0))
                    cell.repair_signal = float(np.clip(cell.repair_signal + 0.25 * proximity, 0.0, 1.0))
                    cell.development_state = np.tanh(cell.development_state + self.compiled_genome.repair_bias * 0.12 * proximity)
                    cell.reprogramming_cost = float(np.clip(cell.reprogramming_cost * (1.0 - 0.25 * proximity), 0.0, 1.0))
                    positive = np.maximum(cell.development_state, 0.0) + 1e-6
                    cell.commitment = positive / positive.sum()

    def inject_damage(self, center: tuple[float, float], radius: float, intensity: float) -> None:
        self.environment.inject_damage(center, radius, intensity)

    def inject_resource_pulse(self, center: tuple[float, float], radius: float, intensity: float) -> None:
        self.environment.inject_resource_pulse(center, radius, intensity)

    def inject_goal(self, center: tuple[float, float], radius: float, intensity: float) -> None:
        self.goals.append({"center": center, "radius": radius, "intensity": intensity})
        self.environment.inject_goal_pressure(center, radius, intensity)

    def add_gene(self, gene: GeneAsset) -> None:
        self.gene_registry.add(gene)

    def phenotype_counts(self) -> dict[str, int]:
        if self.mode == SPEC_MODE:
            counts: dict[str, int] = {}
            for cell in self.cells.values():
                counts[cell.phenotype_label] = counts.get(cell.phenotype_label, 0) + 1
            return counts
        counts = {name: 0 for name in FATE_NAMES}
        for cell in self.cells.values():
            counts[cell.current_fate] += 1
        return counts

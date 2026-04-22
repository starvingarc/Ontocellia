from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ontocellia.specs.schema import GenomeSpec


@dataclass(slots=True)
class CompiledSecretionProgram:
    field: str
    development_weights: np.ndarray
    field_weights: dict[str, float]
    memory_weights: np.ndarray


@dataclass(slots=True)
class CompiledProbe:
    name: str
    development_weights: np.ndarray
    field_weights: dict[str, float]
    energy_weight: float
    stress_weight: float
    threshold: float


@dataclass(slots=True)
class CompiledCompetenceGate:
    dimension: int
    promoters: dict[str, float]
    inhibitors: dict[str, float]
    age_weight: float
    energy_weight: float
    stress_weight: float
    ecm_weight: float
    mechanical_weight: float


@dataclass(slots=True)
class CompiledMorphogen:
    name: str
    diffusion: float
    decay: float
    interpreter_weights: np.ndarray


@dataclass(slots=True)
class CompiledContactProgram:
    name: str
    sender_weights: np.ndarray
    receiver_weights: np.ndarray
    inhibition_strength: float
    activation_strength: float
    contact_area_weight: float
    crowding_weight: float
    persistence_weight: float
    quiescence_threshold: float


@dataclass(slots=True)
class CompiledAttractor:
    name: str
    center: np.ndarray
    basin_sharpness: float
    commitment_threshold: float
    stability_horizon: int


@dataclass(slots=True)
class CompiledGenome:
    spec: GenomeSpec
    field_order: list[str]
    field_matrix: np.ndarray
    gradient_matrix: np.ndarray
    background_matrix: np.ndarray
    move_bias: np.ndarray
    divide_bias: np.ndarray
    die_bias: np.ndarray
    secrete_bias: np.ndarray
    rewire_bias: np.ndarray
    repair_bias: np.ndarray
    quiesce_bias: np.ndarray
    differentiate_bias: np.ndarray
    silence_bias: np.ndarray
    neighbor_development_weight: float
    neighbor_hidden_weight: float
    hidden_feedback_weight: float
    memory_feedback_weight: float
    graph_weight: float
    local_demand_weight: float
    task_bias: np.ndarray
    secretion_programs: list[CompiledSecretionProgram]
    competence_gates: list[CompiledCompetenceGate]
    probes: list[CompiledProbe]
    morphogens: list[CompiledMorphogen]
    contact_programs: list[CompiledContactProgram]
    attractors: list[CompiledAttractor]

    @property
    def hidden_dim(self) -> int:
        return self.spec.state_dims.hidden_dim

    @property
    def memory_dim(self) -> int:
        return self.spec.state_dims.memory_dim

    @property
    def development_dim(self) -> int:
        return self.spec.state_dims.development_dim

    @property
    def development_names(self) -> list[str]:
        return self.spec.state_dims.development_names


class GenomeSpecCompiler:
    def compile(self, spec: GenomeSpec, field_order: list[str]) -> CompiledGenome:
        development_dim = spec.state_dims.development_dim
        field_matrix = np.zeros((development_dim, len(field_order)), dtype=float)
        gradient_matrix = np.zeros((development_dim, len(field_order)), dtype=float)
        background_matrix = np.zeros((development_dim, len(field_order)), dtype=float)

        for field_index, field_name in enumerate(field_order):
            if field_name in spec.sensing.field_weights:
                field_matrix[:, field_index] = np.array(spec.sensing.field_weights[field_name], dtype=float)
            if field_name in spec.sensing.gradient_weights:
                gradient_matrix[:, field_index] = np.array(spec.sensing.gradient_weights[field_name], dtype=float)
            if field_name in spec.background_context.weights:
                background_matrix[:, field_index] = np.array(spec.background_context.weights[field_name], dtype=float)

        secretion_programs = [
            CompiledSecretionProgram(
                field=program.field,
                development_weights=np.array(program.development_weights, dtype=float),
                field_weights=dict(program.field_weights),
                memory_weights=np.array(program.memory_weights, dtype=float),
            )
            for program in spec.secretion_programs
        ]
        competence_gates = [
            CompiledCompetenceGate(
                dimension=gate.dimension,
                promoters=dict(gate.promoters),
                inhibitors=dict(gate.inhibitors),
                age_weight=gate.age_weight,
                energy_weight=gate.energy_weight,
                stress_weight=gate.stress_weight,
                ecm_weight=gate.ecm_weight,
                mechanical_weight=gate.mechanical_weight,
            )
            for gate in spec.competence_gates
        ]
        probes = [
            CompiledProbe(
                name=probe.name,
                development_weights=np.array(probe.development_weights, dtype=float),
                field_weights=dict(probe.field_weights),
                energy_weight=probe.energy_weight,
                stress_weight=probe.stress_weight,
                threshold=probe.threshold,
            )
            for probe in spec.reporting_probes
        ]
        morphogens = [
            CompiledMorphogen(
                name=morphogen.name,
                diffusion=morphogen.diffusion,
                decay=morphogen.decay,
                interpreter_weights=np.array(morphogen.interpreter_weights, dtype=float),
            )
            for morphogen in spec.morphogens
        ]
        contact_programs = [
            CompiledContactProgram(
                name=program.name,
                sender_weights=np.array(program.sender_weights, dtype=float),
                receiver_weights=np.array(program.receiver_weights, dtype=float),
                inhibition_strength=program.inhibition_strength,
                activation_strength=program.activation_strength,
                contact_area_weight=program.contact_area_weight,
                crowding_weight=program.crowding_weight,
                persistence_weight=program.persistence_weight,
                quiescence_threshold=program.quiescence_threshold,
            )
            for program in spec.contact_programs
        ]
        attractors = [
            CompiledAttractor(
                name=attractor.name,
                center=np.array(attractor.center, dtype=float),
                basin_sharpness=attractor.basin_sharpness,
                commitment_threshold=attractor.commitment_threshold,
                stability_horizon=attractor.stability_horizon,
            )
            for attractor in spec.fate_landscape.attractors
        ]

        return CompiledGenome(
            spec=spec,
            field_order=field_order,
            field_matrix=field_matrix,
            gradient_matrix=gradient_matrix,
            background_matrix=background_matrix,
            move_bias=np.array(spec.behavior_biases.move, dtype=float),
            divide_bias=np.array(spec.behavior_biases.divide, dtype=float),
            die_bias=np.array(spec.behavior_biases.die, dtype=float),
            secrete_bias=np.array(spec.behavior_biases.secrete, dtype=float),
            rewire_bias=np.array(spec.behavior_biases.rewire, dtype=float),
            repair_bias=np.array(spec.behavior_biases.repair, dtype=float),
            quiesce_bias=np.array(spec.behavior_biases.quiesce, dtype=float),
            differentiate_bias=np.array(spec.behavior_biases.differentiate, dtype=float),
            silence_bias=np.array(spec.behavior_biases.silence, dtype=float),
            neighbor_development_weight=spec.sensing.neighbor_development_weight,
            neighbor_hidden_weight=spec.sensing.neighbor_hidden_weight,
            hidden_feedback_weight=spec.sensing.hidden_feedback_weight,
            memory_feedback_weight=spec.sensing.memory_feedback_weight,
            graph_weight=spec.sensing.graph_weight,
            local_demand_weight=spec.sensing.local_demand_weight,
            task_bias=np.array(spec.task_coupling.development_biases, dtype=float),
            secretion_programs=secretion_programs,
            competence_gates=competence_gates,
            probes=probes,
            morphogens=morphogens,
            contact_programs=contact_programs,
            attractors=attractors,
        )

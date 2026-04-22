from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _required_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid required mapping: {key}")
    return value


def _optional_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _list_of_floats(values: list[Any], expected: int | None = None, *, label: str) -> list[float]:
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list")
    converted = [float(value) for value in values]
    if expected is not None and len(converted) != expected:
        raise ValueError(f"{label} must have length {expected}, got {len(converted)}")
    return converted


@dataclass(slots=True)
class MetadataSpec:
    name: str
    version: str = "0.1.0"
    description: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetadataSpec":
        if "name" not in data:
            raise ValueError("metadata.name is required")
        return cls(
            name=str(data["name"]),
            version=str(data.get("version", "0.1.0")),
            description=str(data.get("description", "")),
        )


@dataclass(slots=True)
class StateDimsSpec:
    hidden_dim: int
    memory_dim: int
    development_dim: int
    development_names: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateDimsSpec":
        hidden_dim = int(data["hidden_dim"])
        memory_dim = int(data["memory_dim"])
        development_dim = int(data["development_dim"])
        development_names = [str(name) for name in data.get("development_names", [])]
        if development_names and len(development_names) != development_dim:
            raise ValueError("state_dims.development_names must match development_dim length")
        if hidden_dim <= 0 or memory_dim <= 0 or development_dim <= 0:
            raise ValueError("state dims must be positive")
        if not development_names:
            development_names = [f"axis_{index}" for index in range(development_dim)]
        return cls(
            hidden_dim=hidden_dim,
            memory_dim=memory_dim,
            development_dim=development_dim,
            development_names=development_names,
        )


@dataclass(slots=True)
class SensingSpec:
    field_weights: dict[str, list[float]]
    gradient_weights: dict[str, list[float]]
    hidden_feedback_weight: float = 0.12
    memory_feedback_weight: float = 0.14
    neighbor_development_weight: float = 0.28
    neighbor_hidden_weight: float = 0.08
    graph_weight: float = 0.16
    local_demand_weight: float = 0.22

    @classmethod
    def from_dict(cls, data: dict[str, Any], development_dim: int) -> "SensingSpec":
        field_weights = {
            str(name): _list_of_floats(values, development_dim, label=f"sensing.field_weights.{name}")
            for name, values in data.get("field_weights", {}).items()
        }
        gradient_weights = {
            str(name): _list_of_floats(values, development_dim, label=f"sensing.gradient_weights.{name}")
            for name, values in data.get("gradient_weights", {}).items()
        }
        if not field_weights:
            raise ValueError("sensing.field_weights is required")
        return cls(
            field_weights=field_weights,
            gradient_weights=gradient_weights,
            hidden_feedback_weight=float(data.get("hidden_feedback_weight", 0.12)),
            memory_feedback_weight=float(data.get("memory_feedback_weight", 0.14)),
            neighbor_development_weight=float(data.get("neighbor_development_weight", 0.28)),
            neighbor_hidden_weight=float(data.get("neighbor_hidden_weight", 0.08)),
            graph_weight=float(data.get("graph_weight", 0.16)),
            local_demand_weight=float(data.get("local_demand_weight", 0.22)),
        )


@dataclass(slots=True)
class BehaviorBiasesSpec:
    move: list[float]
    divide: list[float]
    die: list[float]
    secrete: list[float]
    rewire: list[float]
    repair: list[float]
    quiesce: list[float]
    differentiate: list[float]
    silence: list[float]

    @classmethod
    def from_dict(cls, data: dict[str, Any], development_dim: int) -> "BehaviorBiasesSpec":
        defaults = {
            "quiesce": [0.0] * development_dim,
            "differentiate": [0.0] * development_dim,
            "silence": [0.0] * development_dim,
        }
        required = ("move", "divide", "die", "secrete", "rewire", "repair")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Missing behavior_biases fields: {', '.join(missing)}")
        merged = {**defaults, **data}
        return cls(
            move=_list_of_floats(merged["move"], development_dim, label="behavior_biases.move"),
            divide=_list_of_floats(merged["divide"], development_dim, label="behavior_biases.divide"),
            die=_list_of_floats(merged["die"], development_dim, label="behavior_biases.die"),
            secrete=_list_of_floats(merged["secrete"], development_dim, label="behavior_biases.secrete"),
            rewire=_list_of_floats(merged["rewire"], development_dim, label="behavior_biases.rewire"),
            repair=_list_of_floats(merged["repair"], development_dim, label="behavior_biases.repair"),
            quiesce=_list_of_floats(merged["quiesce"], development_dim, label="behavior_biases.quiesce"),
            differentiate=_list_of_floats(merged["differentiate"], development_dim, label="behavior_biases.differentiate"),
            silence=_list_of_floats(merged["silence"], development_dim, label="behavior_biases.silence"),
        )


@dataclass(slots=True)
class SecretionProgramSpec:
    field: str
    development_weights: list[float]
    field_weights: dict[str, float] = field(default_factory=dict)
    memory_weights: list[float] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any], development_dim: int, memory_dim: int) -> "SecretionProgramSpec":
        if "field" not in data:
            raise ValueError("secretion_programs[].field is required")
        memory_weights = data.get("memory_weights", [0.0] * memory_dim)
        return cls(
            field=str(data["field"]),
            development_weights=_list_of_floats(
                data["development_weights"],
                development_dim,
                label=f"secretion_programs[{data['field']}].development_weights",
            ),
            field_weights={str(name): float(value) for name, value in data.get("field_weights", {}).items()},
            memory_weights=_list_of_floats(
                memory_weights,
                memory_dim,
                label=f"secretion_programs[{data['field']}].memory_weights",
            ),
        )


@dataclass(slots=True)
class CompetenceGateSpec:
    dimension: int
    promoters: dict[str, float] = field(default_factory=dict)
    inhibitors: dict[str, float] = field(default_factory=dict)
    age_weight: float = 0.0
    energy_weight: float = 0.0
    stress_weight: float = 0.0
    ecm_weight: float = 0.0
    mechanical_weight: float = 0.0

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        development_dim: int,
        development_names: list[str],
    ) -> "CompetenceGateSpec":
        raw_dimension = data.get("dimension")
        if raw_dimension is None:
            raise ValueError("competence_windows[].dimension is required")
        if isinstance(raw_dimension, str):
            if raw_dimension not in development_names:
                raise ValueError(f"Unknown development dimension name: {raw_dimension}")
            dimension = development_names.index(raw_dimension)
        else:
            dimension = int(raw_dimension)
        if not 0 <= dimension < development_dim:
            raise ValueError("competence_windows[].dimension is out of range")
        return cls(
            dimension=dimension,
            promoters={str(name): float(value) for name, value in data.get("promoters", {}).items()},
            inhibitors={str(name): float(value) for name, value in data.get("inhibitors", {}).items()},
            age_weight=float(data.get("age_weight", 0.0)),
            energy_weight=float(data.get("energy_weight", 0.0)),
            stress_weight=float(data.get("stress_weight", 0.0)),
            ecm_weight=float(data.get("ecm_weight", 0.0)),
            mechanical_weight=float(data.get("mechanical_weight", 0.0)),
        )


@dataclass(slots=True)
class EpigeneticLockSpec:
    strength: float
    decay: float
    revert_cost: float
    commitment_half_life: float
    reprogramming_penalty: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpigeneticLockSpec":
        return cls(
            strength=float(data.get("strength", 0.22)),
            decay=float(data.get("decay", 0.82)),
            revert_cost=float(data.get("revert_cost", 0.18)),
            commitment_half_life=float(data.get("commitment_half_life", 4.0)),
            reprogramming_penalty=float(data.get("reprogramming_penalty", 0.25)),
        )


@dataclass(slots=True)
class LineageRulesSpec:
    cooldown_steps: int
    parent_energy_share: float
    child_energy_share: float
    noise_scale: float
    division_energy_threshold: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LineageRulesSpec":
        parent_share = float(data.get("parent_energy_share", 0.58))
        child_share = float(data.get("child_energy_share", 0.42))
        if parent_share + child_share > 1.05:
            raise ValueError("lineage_rules parent/child energy shares must be <= 1.05 in total")
        return cls(
            cooldown_steps=int(data.get("cooldown_steps", 5)),
            parent_energy_share=parent_share,
            child_energy_share=child_share,
            noise_scale=float(data.get("noise_scale", 0.03)),
            division_energy_threshold=float(data.get("division_energy_threshold", 0.52)),
        )


@dataclass(slots=True)
class TaskCouplingSpec:
    field_biases: dict[str, float]
    development_biases: list[float]
    goal_zone_bias: float = 0.22

    @classmethod
    def from_dict(cls, data: dict[str, Any], development_dim: int) -> "TaskCouplingSpec":
        biases = data.get("development_biases", [0.0] * development_dim)
        return cls(
            field_biases={str(name): float(value) for name, value in data.get("field_biases", {}).items()},
            development_biases=_list_of_floats(biases, development_dim, label="task_coupling.development_biases"),
            goal_zone_bias=float(data.get("goal_zone_bias", 0.22)),
        )


@dataclass(slots=True)
class ReportingProbeSpec:
    name: str
    development_weights: list[float]
    field_weights: dict[str, float] = field(default_factory=dict)
    energy_weight: float = 0.0
    stress_weight: float = 0.0
    threshold: float = 0.5

    @classmethod
    def from_dict(cls, data: dict[str, Any], development_dim: int) -> "ReportingProbeSpec":
        if "name" not in data:
            raise ValueError("reporting_probes[].name is required")
        return cls(
            name=str(data["name"]),
            development_weights=_list_of_floats(
                data["development_weights"],
                development_dim,
                label=f"reporting_probes[{data['name']}].development_weights",
            ),
            field_weights={str(name): float(value) for name, value in data.get("field_weights", {}).items()},
            energy_weight=float(data.get("energy_weight", 0.0)),
            stress_weight=float(data.get("stress_weight", 0.0)),
            threshold=float(data.get("threshold", 0.5)),
        )


@dataclass(slots=True)
class MorphogenSpec:
    name: str
    diffusion: float
    decay: float
    interpreter_weights: list[float]

    @classmethod
    def from_dict(cls, data: dict[str, Any], development_dim: int) -> "MorphogenSpec":
        if "name" not in data:
            raise ValueError("morphogens[].name is required")
        return cls(
            name=str(data["name"]),
            diffusion=float(data.get("diffusion", 0.12)),
            decay=float(data.get("decay", 0.01)),
            interpreter_weights=_list_of_floats(
                data.get("interpreter_weights", [0.0] * development_dim),
                development_dim,
                label=f"morphogens[{data['name']}].interpreter_weights",
            ),
        )


@dataclass(slots=True)
class ContactProgramSpec:
    name: str
    sender_weights: list[float]
    receiver_weights: list[float]
    inhibition_strength: float = 0.0
    activation_strength: float = 0.0
    contact_area_weight: float = 0.0
    crowding_weight: float = 0.0
    persistence_weight: float = 0.0
    quiescence_threshold: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any], development_dim: int) -> "ContactProgramSpec":
        if "name" not in data:
            raise ValueError("contact_programs[].name is required")
        return cls(
            name=str(data["name"]),
            sender_weights=_list_of_floats(
                data.get("sender_weights", [0.0] * development_dim),
                development_dim,
                label=f"contact_programs[{data['name']}].sender_weights",
            ),
            receiver_weights=_list_of_floats(
                data.get("receiver_weights", [0.0] * development_dim),
                development_dim,
                label=f"contact_programs[{data['name']}].receiver_weights",
            ),
            inhibition_strength=float(data.get("inhibition_strength", 0.0)),
            activation_strength=float(data.get("activation_strength", 0.0)),
            contact_area_weight=float(data.get("contact_area_weight", 0.0)),
            crowding_weight=float(data.get("crowding_weight", 0.0)),
            persistence_weight=float(data.get("persistence_weight", 0.0)),
            quiescence_threshold=float(data.get("quiescence_threshold", 0.0)),
        )


@dataclass(slots=True)
class BackgroundContextSpec:
    weights: dict[str, list[float]]

    @classmethod
    def from_dict(cls, data: dict[str, Any], development_dim: int) -> "BackgroundContextSpec":
        weights = {
            str(name): _list_of_floats(values, development_dim, label=f"background_context.weights.{name}")
            for name, values in data.get("weights", {}).items()
        }
        return cls(weights=weights)


@dataclass(slots=True)
class FateAttractorSpec:
    name: str
    center: list[float]
    basin_sharpness: float
    commitment_threshold: float
    stability_horizon: int

    @classmethod
    def from_dict(cls, data: dict[str, Any], development_dim: int) -> "FateAttractorSpec":
        if "name" not in data:
            raise ValueError("fate_landscape.attractors[].name is required")
        return cls(
            name=str(data["name"]),
            center=_list_of_floats(
                data["center"],
                development_dim,
                label=f"fate_landscape.attractors[{data['name']}].center",
            ),
            basin_sharpness=float(data.get("basin_sharpness", 4.0)),
            commitment_threshold=float(data.get("commitment_threshold", 0.6)),
            stability_horizon=int(data.get("stability_horizon", 4)),
        )


@dataclass(slots=True)
class FateLandscapeSpec:
    attractors: list[FateAttractorSpec]
    global_commitment_threshold: float = 0.6

    @classmethod
    def from_dict(cls, data: dict[str, Any], development_dim: int, development_names: list[str]) -> "FateLandscapeSpec":
        attractors = [FateAttractorSpec.from_dict(item, development_dim) for item in data.get("attractors", [])]
        if not attractors:
            for index, name in enumerate(development_names):
                center = [0.0] * development_dim
                center[index] = 1.0
                attractors.append(
                    FateAttractorSpec(
                        name=name,
                        center=center,
                        basin_sharpness=4.0,
                        commitment_threshold=0.6,
                        stability_horizon=4,
                    )
                )
        return cls(
            attractors=attractors,
            global_commitment_threshold=float(data.get("global_commitment_threshold", 0.6)),
        )


@dataclass(slots=True)
class GenomeSpec:
    metadata: MetadataSpec
    state_dims: StateDimsSpec
    sensing: SensingSpec
    behavior_biases: BehaviorBiasesSpec
    secretion_programs: list[SecretionProgramSpec]
    competence_gates: list[CompetenceGateSpec]
    epigenetic_lock: EpigeneticLockSpec
    lineage_rules: LineageRulesSpec
    task_coupling: TaskCouplingSpec
    reporting_probes: list[ReportingProbeSpec]
    morphogens: list[MorphogenSpec]
    contact_programs: list[ContactProgramSpec]
    background_context: BackgroundContextSpec
    fate_landscape: FateLandscapeSpec
    source_path: Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source_path: Path | None = None) -> "GenomeSpec":
        metadata = MetadataSpec.from_dict(_required_mapping(data, "metadata"))
        state_dims = StateDimsSpec.from_dict(_required_mapping(data, "state_dims"))
        sensing = SensingSpec.from_dict(_required_mapping(data, "sensing"), state_dims.development_dim)
        behavior_biases = BehaviorBiasesSpec.from_dict(_required_mapping(data, "behavior_biases"), state_dims.development_dim)
        secretion_programs = [
            SecretionProgramSpec.from_dict(item, state_dims.development_dim, state_dims.memory_dim)
            for item in data.get("secretion_programs", [])
        ]
        competence_source = data.get("competence_windows", data.get("competence_gates", []))
        competence_gates = [
            CompetenceGateSpec.from_dict(item, state_dims.development_dim, state_dims.development_names)
            for item in competence_source
        ]
        epigenetic_lock = EpigeneticLockSpec.from_dict(_required_mapping(data, "epigenetic_lock"))
        lineage_rules = LineageRulesSpec.from_dict(_required_mapping(data, "lineage_rules"))
        task_coupling = TaskCouplingSpec.from_dict(_required_mapping(data, "task_coupling"), state_dims.development_dim)
        reporting_probes = [
            ReportingProbeSpec.from_dict(item, state_dims.development_dim)
            for item in data.get("reporting_probes", [])
        ]
        morphogens = [
            MorphogenSpec.from_dict(item, state_dims.development_dim)
            for item in data.get("morphogens", [])
        ]
        contact_programs = [
            ContactProgramSpec.from_dict(item, state_dims.development_dim)
            for item in data.get("contact_programs", [])
        ]
        background_context = BackgroundContextSpec.from_dict(
            _optional_mapping(data, "background_context"),
            state_dims.development_dim,
        )
        fate_landscape = FateLandscapeSpec.from_dict(
            _optional_mapping(data, "fate_landscape"),
            state_dims.development_dim,
            state_dims.development_names,
        )
        return cls(
            metadata=metadata,
            state_dims=state_dims,
            sensing=sensing,
            behavior_biases=behavior_biases,
            secretion_programs=secretion_programs,
            competence_gates=competence_gates,
            epigenetic_lock=epigenetic_lock,
            lineage_rules=lineage_rules,
            task_coupling=task_coupling,
            reporting_probes=reporting_probes,
            morphogens=morphogens,
            contact_programs=contact_programs,
            background_context=background_context,
            fate_landscape=fate_landscape,
            source_path=source_path,
        )


@dataclass(slots=True)
class GridSpec:
    width: int
    height: int
    boundary: str = "clamp"
    spatial_scale: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GridSpec":
        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            boundary=str(data.get("boundary", "clamp")),
            spatial_scale=float(data.get("spatial_scale", 1.0)),
        )


@dataclass(slots=True)
class FieldSpec:
    initial: float | dict[str, Any]
    diffusion: float = 0.15
    decay: float = 0.02

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FieldSpec":
        if "initial" not in data:
            raise ValueError("field.initial is required")
        initial = data["initial"]
        if not isinstance(initial, (int, float, dict)):
            raise ValueError("field.initial must be a number or mapping")
        return cls(
            initial=initial,
            diffusion=float(data.get("diffusion", 0.15)),
            decay=float(data.get("decay", 0.02)),
        )


@dataclass(slots=True)
class ZoneSpec:
    center: tuple[float, float]
    radius: float
    intensity: float
    field: str | None = None
    mode: str = "add"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ZoneSpec":
        center = data.get("center")
        if not isinstance(center, list) or len(center) != 2:
            raise ValueError("Zone center must be a 2-item list")
        return cls(
            center=(float(center[0]), float(center[1])),
            radius=float(data["radius"]),
            intensity=float(data["intensity"]),
            field=str(data["field"]) if data.get("field") is not None else None,
            mode=str(data.get("mode", "add")),
        )


@dataclass(slots=True)
class EventSpec:
    step: int
    action: str
    center: tuple[float, float]
    radius: float
    intensity: float
    field: str | None = None
    repeat_every: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventSpec":
        center = data.get("center")
        if not isinstance(center, list) or len(center) != 2:
            raise ValueError("Event center must be a 2-item list")
        return cls(
            step=int(data["step"]),
            action=str(data["action"]),
            center=(float(center[0]), float(center[1])),
            radius=float(data["radius"]),
            intensity=float(data["intensity"]),
            field=str(data["field"]) if data.get("field") is not None else None,
            repeat_every=int(data["repeat_every"]) if data.get("repeat_every") is not None else None,
        )


@dataclass(slots=True)
class GlobalTaskSpec:
    text: str
    objective: str = ""
    success_signals: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GlobalTaskSpec":
        if "text" not in data:
            raise ValueError("global_task.text is required")
        return cls(
            text=str(data["text"]),
            objective=str(data.get("objective", "")),
            success_signals=[str(item) for item in data.get("success_signals", [])],
        )


@dataclass(slots=True)
class TaskTranslationSpec:
    goal_zones: list[ZoneSpec] = field(default_factory=list)
    risk_zones: list[ZoneSpec] = field(default_factory=list)
    resource_zones: list[ZoneSpec] = field(default_factory=list)
    periodic_events: list[EventSpec] = field(default_factory=list)
    field_biases: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskTranslationSpec":
        return cls(
            goal_zones=[ZoneSpec.from_dict(item) for item in data.get("goal_zones", [])],
            risk_zones=[ZoneSpec.from_dict(item) for item in data.get("risk_zones", [])],
            resource_zones=[ZoneSpec.from_dict(item) for item in data.get("resource_zones", [])],
            periodic_events=[EventSpec.from_dict(item) for item in data.get("periodic_events", [])],
            field_biases={str(name): float(value) for name, value in data.get("field_biases", {}).items()},
        )


@dataclass(slots=True)
class EvaluationSpec:
    metrics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationSpec":
        return cls(metrics={str(name): float(value) for name, value in data.get("metrics", {}).items()})


@dataclass(slots=True)
class EnvironmentSpec:
    metadata: MetadataSpec
    grid: GridSpec
    diffusive_fields: dict[str, FieldSpec]
    background_context: dict[str, FieldSpec]
    contact_context: dict[str, float]
    sources: list[ZoneSpec]
    events: list[EventSpec]
    global_task: GlobalTaskSpec
    task_translation: TaskTranslationSpec
    evaluation: EvaluationSpec
    source_path: Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source_path: Path | None = None) -> "EnvironmentSpec":
        metadata = MetadataSpec.from_dict(_required_mapping(data, "metadata"))
        grid = GridSpec.from_dict(_required_mapping(data, "grid"))

        if "diffusive_fields" in data or "background_context" in data:
            diffusive_mapping = _required_mapping(data, "diffusive_fields")
            background_mapping = _optional_mapping(data, "background_context")
        else:
            raw_fields = _required_mapping(data, "fields")
            diffusive_keys = {"M1", "M2", "M3", "morphogen_a", "morphogen_b", "task_pressure", "nutrient", "damage"}
            background_keys = {"ECM", "mechanical_stress", "crowding"}
            diffusive_mapping = {key: value for key, value in raw_fields.items() if key in diffusive_keys}
            background_mapping = {key: value for key, value in raw_fields.items() if key in background_keys}
            if "crowding" not in background_mapping:
                background_mapping["crowding"] = {"initial": 0.0, "diffusion": 0.05, "decay": 0.02}

        diffusive_fields = {str(name): FieldSpec.from_dict(field_data) for name, field_data in diffusive_mapping.items()}
        background_context = {str(name): FieldSpec.from_dict(field_data) for name, field_data in background_mapping.items()}
        if "task_pressure" not in diffusive_fields:
            raise ValueError("Environment diffusive_fields must include task_pressure")
        if "nutrient" not in diffusive_fields:
            raise ValueError("Environment diffusive_fields must include nutrient")
        if "damage" not in diffusive_fields:
            raise ValueError("Environment diffusive_fields must include damage")
        contact_context = {
            "contact_radius": float(data.get("contact_context", {}).get("contact_radius", 5.0)),
            "contact_inhibition_strength": float(data.get("contact_context", {}).get("contact_inhibition_strength", 0.4)),
            "default_contact_persistence": float(data.get("contact_context", {}).get("default_contact_persistence", 0.6)),
        }
        return cls(
            metadata=metadata,
            grid=grid,
            diffusive_fields=diffusive_fields,
            background_context=background_context,
            contact_context=contact_context,
            sources=[ZoneSpec.from_dict(item) for item in data.get("sources", [])],
            events=[EventSpec.from_dict(item) for item in data.get("events", [])],
            global_task=GlobalTaskSpec.from_dict(_required_mapping(data, "global_task")),
            task_translation=TaskTranslationSpec.from_dict(_required_mapping(data, "task_translation")),
            evaluation=EvaluationSpec.from_dict(_required_mapping(data, "evaluation")),
            source_path=source_path,
        )

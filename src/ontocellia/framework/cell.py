from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ontocellia.framework.genome import AgentGenome, EpigeneticMarks, ExpressionContext


CellStage = Literal["stem", "progenitor", "transit_amplifying", "differentiated"]


@dataclass(slots=True)
class CellPosition:
    node_id: str
    region: str = ""
    neighbors: list[str] = field(default_factory=list)
    embedding: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_value(cls, value: "CellPosition | tuple[float, ...] | list[float] | dict[str, Any]") -> "CellPosition":
        if isinstance(value, CellPosition):
            return value
        if isinstance(value, dict):
            embedding = value.get("embedding", (0.0, 0.0, 0.0))
            return cls(
                node_id=str(value.get("node_id", "")),
                region=str(value.get("region", "")),
                neighbors=[str(neighbor) for neighbor in value.get("neighbors", [])],
                embedding=_embedding3(embedding),
            )
        if isinstance(value, (tuple, list)):
            embedding = _embedding3(value)
            return cls(node_id=f"coord:{embedding[0]}:{embedding[1]}:{embedding[2]}", embedding=embedding)
        raise ValueError("position must be CellPosition, mapping, tuple, or list")


@dataclass(slots=True)
class AdhesionProfile:
    compatible_fates: list[str] = field(default_factory=list)
    strength: float = 0.5


@dataclass(slots=True)
class ReceptorProfile:
    signal_sensitivities: dict[str, float] = field(default_factory=dict)
    accepted_interfaces: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CompetenceProfile:
    fate_scores: dict[str, float] = field(default_factory=dict)
    plasticity: float = 1.0


@dataclass(slots=True)
class LineageRecord:
    parent_id: int | None
    root_id: int
    generation: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    def child(self, parent_id: int) -> "LineageRecord":
        return LineageRecord(
            parent_id=parent_id,
            root_id=self.root_id,
            generation=self.generation + 1,
            events=list(self.events),
        )


@dataclass(slots=True)
class StemCellState:
    plasticity: float = 1.0
    division_potential: float = 1.0


@dataclass(slots=True)
class ProgenitorCellState:
    target_fate: str = ""
    amplification_potential: float = 1.0


@dataclass(slots=True)
class TransitAmplifyingCellState:
    target_fate: str = ""
    remaining_divisions: int = 1


@dataclass(slots=True)
class DifferentiatedCellState:
    fate_lock: float = 0.6
    reprogrammable: bool = True


@dataclass(slots=True)
class AgentCell:
    """One autonomous cell-agent in the tissue."""

    id: int
    stage: CellStage | str
    fate: str
    position: CellPosition | tuple[float, ...] | list[float] | dict[str, Any]
    lineage_parent: int | None = None
    niche_id: str | None = None
    expressed_gene_ids: list[str] = field(default_factory=list)
    age: int = 0
    energy: float = 1.0
    alive: bool = True
    replaces_cell_id: int | None = None
    stress: float = 0.0
    adhesion: AdhesionProfile = field(default_factory=AdhesionProfile)
    receptor: ReceptorProfile = field(default_factory=ReceptorProfile)
    competence: CompetenceProfile = field(default_factory=CompetenceProfile)
    epigenetic_marks: EpigeneticMarks = field(default_factory=EpigeneticMarks)
    lineage: LineageRecord | None = None
    stage_state: StemCellState | ProgenitorCellState | TransitAmplifyingCellState | DifferentiatedCellState | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.position = CellPosition.from_value(self.position)
        if self.lineage is None:
            root_id = self.id if self.lineage_parent is None else self.lineage_parent
            self.lineage = LineageRecord(parent_id=self.lineage_parent, root_id=root_id)
        if self.stage_state is None:
            self.stage_state = _default_stage_state(str(self.stage), self.fate)

    @property
    def differentiated(self) -> bool:
        return self.stage == "differentiated"

    def expression_context(self, morphogens: dict[str, float] | Any, organ_feedback: dict[str, float] | None = None) -> ExpressionContext:
        signal_source = morphogens if isinstance(morphogens, dict) else getattr(morphogens, "signals", {})
        signals = {
            str(name): float(value) * float(self.receptor.signal_sensitivities.get(str(name), 1.0))
            for name, value in signal_source.items()
        }
        return ExpressionContext(
            morphogens=signals,
            cell_stage=str(self.stage),
            current_fate=self.fate,
            energy=self.energy,
            stress=self.stress,
            competence=dict(self.competence.fate_scores),
            lineage_history=list(self.history),
            epigenetic_marks=self.epigenetic_marks,
            organ_feedback=dict(organ_feedback or {}),
        )

    def can_divide(self) -> bool:
        if self.stage == "stem" and isinstance(self.stage_state, StemCellState):
            return self.alive and self.energy >= 0.25 and self.stage_state.division_potential > 0.0
        if self.stage == "progenitor" and isinstance(self.stage_state, ProgenitorCellState):
            return self.alive and self.energy >= 0.25 and self.stage_state.amplification_potential > 0.0
        if self.stage == "transit_amplifying" and isinstance(self.stage_state, TransitAmplifyingCellState):
            return self.alive and self.energy >= 0.25 and self.stage_state.remaining_divisions > 0
        return False

    def can_reprogram(self, regeneration_pressure: float) -> bool:
        return (
            self.differentiated
            and isinstance(self.stage_state, DifferentiatedCellState)
            and self.stage_state.reprogrammable
            and regeneration_pressure >= 1.0 + (self.stage_state.fate_lock * 0.5)
        )

    def commit_to_fate(
        self,
        fate: str,
        niche_id: str,
        genome: AgentGenome,
        morphogens: dict[str, float] | Any,
        organ_feedback: dict[str, float] | None = None,
    ) -> "AgentCell":
        self.stage = "differentiated"
        self.fate = fate
        self.niche_id = niche_id
        self.stage_state = DifferentiatedCellState(fate_lock=0.6, reprogrammable=True)
        programs = genome.express(self.expression_context(morphogens, organ_feedback=organ_feedback))
        self.expressed_gene_ids = [program.gene.id for program in programs if program.fate_bias == fate]
        self.position = CellPosition(
            node_id=niche_id,
            region=self.position.region or niche_id,
            neighbors=list(self.position.neighbors),
            embedding=self.position.embedding,
        )
        self.record_event("differentiation", fate=fate, niche_id=niche_id)
        return self

    def spawn_child(
        self,
        child_id: int,
        stage: CellStage | str,
        fate: str,
        position: CellPosition | tuple[float, ...] | list[float] | dict[str, Any],
    ) -> "AgentCell":
        assert self.lineage is not None
        child_lineage = self.lineage.child(parent_id=self.id)
        child = AgentCell(
            id=child_id,
            stage=stage,
            fate=fate,
            position=position,
            lineage_parent=self.id,
            energy=max(0.25, self.energy * 0.55),
            stress=self.stress,
            adhesion=AdhesionProfile(compatible_fates=list(self.adhesion.compatible_fates), strength=self.adhesion.strength),
            receptor=ReceptorProfile(
                signal_sensitivities=dict(self.receptor.signal_sensitivities),
                accepted_interfaces=list(self.receptor.accepted_interfaces),
            ),
            competence=CompetenceProfile(fate_scores=dict(self.competence.fate_scores), plasticity=self.competence.plasticity),
            epigenetic_marks=EpigeneticMarks(
                fate_locks=dict(self.epigenetic_marks.fate_locks),
                gene_locks=dict(self.epigenetic_marks.gene_locks),
            ),
            lineage=child_lineage,
            history=list(self.history),
        )
        child.record_event("division", parent_cell_id=self.id, child_stage=str(stage), fate=fate)
        return child

    def record_event(self, event_type: str, **payload: Any) -> None:
        event = {"type": event_type, **payload}
        self.history.append(event)
        assert self.lineage is not None
        self.lineage.events.append(event)

    def accepts_interface(self, interface_id: str) -> bool:
        return not self.receptor.accepted_interfaces or interface_id in self.receptor.accepted_interfaces

    def adhesion_score(self, neighbor_fate: str) -> float:
        if neighbor_fate == self.fate:
            return 1.0
        if neighbor_fate in self.adhesion.compatible_fates:
            return self.adhesion.strength
        return 0.0


def _default_stage_state(stage: str, fate: str) -> StemCellState | ProgenitorCellState | TransitAmplifyingCellState | DifferentiatedCellState:
    if stage == "stem":
        return StemCellState()
    if stage == "progenitor":
        return ProgenitorCellState(target_fate=fate)
    if stage == "transit_amplifying":
        return TransitAmplifyingCellState(target_fate=fate)
    return DifferentiatedCellState()


def _embedding3(value: tuple[float, ...] | list[float]) -> tuple[float, float, float]:
    if len(value) == 2:
        return (float(value[0]), float(value[1]), 0.0)
    if len(value) == 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    raise ValueError("embedding must have two or three items")

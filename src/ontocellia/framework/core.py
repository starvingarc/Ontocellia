from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from random import Random
from typing import Any

from ontocellia.framework.cell import AgentCell, CellPosition, StemCellState
from ontocellia.framework.genome import AgentGenome, Gene


@dataclass(slots=True)
class MorphogenField:
    """Task and tissue signals that induce gene expression."""

    signals: dict[str, float] = field(default_factory=dict)

    def signal(self, name: str) -> float:
        return float(self.signals.get(name, 0.0))

    def emit(self, name: str, amount: float) -> None:
        self.signals[name] = max(0.0, self.signal(name) + amount)

    def decay(self, rate: float = 0.92) -> None:
        for name, value in list(self.signals.items()):
            self.signals[name] = max(0.0, float(value) * rate)


@dataclass(slots=True)
class Niche:
    """A local functional region that attracts compatible cells."""

    id: str
    required_fate: str
    position: CellPosition | tuple[float, ...] | list[float] | dict[str, Any]
    demand: int = 1
    occupied_by: list[int] = field(default_factory=list)
    vacant_replacements: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.position = CellPosition.from_value(self.position)
        if not self.position.node_id:
            self.position.node_id = self.id
        if not self.position.region:
            self.position.region = self.id


@dataclass(slots=True)
class ExtracellularInterface:
    """Biological interface that may be implemented by MCP, shell, LLM, or local code."""

    id: str
    kind: str
    accepts_fates: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def accepts(self, cell: "AgentCell") -> bool:
        return cell.fate in self.accepts_fates


@dataclass(slots=True)
class TaskMicroenvironment:
    """Task substrate that provides morphogens, niches, and extracellular interfaces."""

    objective: str
    morphogens: MorphogenField = field(default_factory=MorphogenField)
    niches: list[Niche] = field(default_factory=list)
    interfaces: list[ExtracellularInterface] = field(default_factory=list)
    matrix: dict[str, Any] = field(default_factory=dict)

    def niche_by_id(self, niche_id: str) -> Niche:
        for niche in self.niches:
            if niche.id == niche_id:
                return niche
        raise KeyError(f"unknown niche: {niche_id}")


@dataclass(slots=True)
class TissueTrace:
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, event_type: str, **payload: Any) -> None:
        self.events.append({"type": event_type, **payload})


@dataclass(slots=True)
class TissueRuntime:
    """Deterministic developmental harness for task-induced agent tissues."""

    genome: AgentGenome
    environment: TaskMicroenvironment
    cells: dict[int, AgentCell]
    trace: TissueTrace = field(default_factory=TissueTrace)
    rng: Random = field(default_factory=Random)
    tick_count: int = 0
    next_cell_id: int = 0

    @classmethod
    def seeded(
        cls,
        genome: AgentGenome,
        environment: TaskMicroenvironment,
        stem_cells: int = 6,
        seed: int = 0,
    ) -> "TissueRuntime":
        rng = Random(seed)
        cells: dict[int, AgentCell] = {}
        for cell_id in range(stem_cells):
            cells[cell_id] = AgentCell(
                id=cell_id,
                stage="stem",
                fate="stem",
                position=CellPosition(
                    node_id=f"stem-reserve-{cell_id}",
                    region="stem-reserve",
                    embedding=(rng.uniform(3.5, 6.5), rng.uniform(3.5, 6.5), rng.uniform(0.0, 2.0)),
                ),
                stage_state=StemCellState(plasticity=1.0, division_potential=1.0),
            )
        runtime = cls(
            genome=genome,
            environment=environment,
            cells=cells,
            rng=rng,
            next_cell_id=stem_cells,
        )
        runtime.trace.record("seed", stem_cells=stem_cells)
        return runtime

    def develop(self, ticks: int = 1) -> None:
        for _ in range(ticks):
            self.tick_count += 1
            self._refresh_niche_occupancy()
            self._resolve_vacancies()
            self._fill_open_niches()
            self._update_cell_positions()
            self._age_cells()
            self.environment.morphogens.decay()
            self.trace.record("tick", tick=self.tick_count, fate_counts=self.fate_counts())

    def fate_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for cell in self.cells.values():
            if cell.alive:
                counts[cell.fate] = counts.get(cell.fate, 0) + 1
        return counts

    def niche_occupancy(self) -> dict[str, int]:
        self._refresh_niche_occupancy()
        return {niche.id: len(niche.occupied_by) for niche in self.environment.niches}

    def clear_cell(self, cell_id: int, reason: str = "cleared") -> None:
        cell = self.cells.pop(cell_id)
        if cell.niche_id is not None:
            niche = self.environment.niche_by_id(cell.niche_id)
            niche.vacant_replacements.append(cell.id)
        self.environment.morphogens.emit("damage", 1.0)
        self.environment.morphogens.emit("niche_vacancy", 1.0)
        self.environment.morphogens.emit("repair_pressure", 0.6)
        self.trace.record(
            "apoptosis",
            cell_id=cell.id,
            fate=cell.fate,
            niche_id=cell.niche_id,
            reason=reason,
        )

    def execute(self, effectors: Any | None = None) -> list[dict[str, Any]]:
        if effectors is not None:
            return [intent.as_dict() for intent in effectors.emit_intents(self)]
        actions: list[dict[str, Any]] = []
        for cell in sorted(self.cells.values(), key=lambda item: item.id):
            if not cell.differentiated:
                continue
            for interface in self.environment.interfaces:
                if not interface.accepts(cell):
                    continue
                if not cell.accepts_interface(interface.id):
                    continue
                for gene_id in cell.expressed_gene_ids:
                    action = {
                        "cell_id": cell.id,
                        "fate": cell.fate,
                        "gene_id": gene_id,
                        "interface_id": interface.id,
                        "interface_kind": interface.kind,
                    }
                    actions.append(action)
                    self.trace.record("effector_action", **action)
        return actions

    def _refresh_niche_occupancy(self) -> None:
        for niche in self.environment.niches:
            niche.occupied_by = []
        for cell in self.cells.values():
            if cell.niche_id is None or not cell.alive:
                continue
            try:
                niche = self.environment.niche_by_id(cell.niche_id)
            except KeyError:
                cell.niche_id = None
                continue
            if cell.fate == niche.required_fate:
                niche.occupied_by.append(cell.id)

    def _resolve_vacancies(self) -> None:
        for niche in self.environment.niches:
            while niche.vacant_replacements and len(niche.occupied_by) < niche.demand:
                source = self._select_plastic_cell(prefer_near=niche.position)
                if source is None:
                    source = self._reprogram_for_regeneration(niche)
                if source is None:
                    break
                replaced_cell_id = niche.vacant_replacements.pop(0)
                replacement = self._regenerate_from(source, niche, replaced_cell_id)
                niche.occupied_by.append(replacement.id)

    def _fill_open_niches(self) -> None:
        for niche in sorted(self.environment.niches, key=lambda item: item.id):
            while len(niche.occupied_by) < niche.demand:
                source = self._select_plastic_cell(prefer_near=niche.position)
                if source is None:
                    return
                committed = self._differentiate(source, niche)
                niche.occupied_by.append(committed.id)

    def _select_plastic_cell(self, prefer_near: CellPosition | tuple[float, ...] | list[float] | dict[str, Any]) -> AgentCell | None:
        target = CellPosition.from_value(prefer_near)
        candidates = [
            cell
            for cell in self.cells.values()
            if cell.stage in {"stem", "progenitor", "transit_amplifying"} and cell.niche_id is None
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda cell: (_graph_distance(cell.position, target), _stage_rank(str(cell.stage)), cell.id))
        return candidates[0]

    def _differentiate(self, cell: AgentCell, niche: Niche) -> AgentCell:
        cell.position = _move_position_toward(cell.position, niche.position, fraction=0.85)
        cell.commit_to_fate(niche.required_fate, niche.id, self.genome, self.environment.morphogens)
        cell.position = CellPosition(
            node_id=niche.position.node_id,
            region=niche.position.region,
            neighbors=list(niche.position.neighbors),
            embedding=cell.position.embedding,
        )
        self.trace.record(
            "differentiation",
            cell_id=cell.id,
            fate=cell.fate,
            niche_id=niche.id,
            expressed_gene_ids=list(cell.expressed_gene_ids),
        )
        return cell

    def _regenerate_from(self, source: AgentCell, niche: Niche, replaced_cell_id: int) -> AgentCell:
        source_stage = source.stage
        if source.stage == "stem":
            progenitor = self._spawn_child(source, stage="progenitor", fate=niche.required_fate, position=source.position)
            transit = self._spawn_child(progenitor, stage="transit_amplifying", fate=niche.required_fate, position=progenitor.position)
            replacement = transit
        elif source.stage == "progenitor":
            replacement = self._spawn_child(source, stage="transit_amplifying", fate=niche.required_fate, position=source.position)
        else:
            replacement = source
        replacement.replaces_cell_id = replaced_cell_id
        self._differentiate(replacement, niche)
        self.trace.record(
            "regeneration",
            cell_id=replacement.id,
            replaced_cell_id=replaced_cell_id,
            niche_id=niche.id,
            source_cell_id=source.id,
            source_stage=source_stage,
            fate=niche.required_fate,
        )
        return replacement

    def _spawn_child(self, parent: AgentCell, stage: str, fate: str, position: tuple[float, float]) -> AgentCell:
        base_position = CellPosition.from_value(position)
        child_position = CellPosition(
            node_id=base_position.neighbors[0] if base_position.neighbors else base_position.node_id,
            region=base_position.region,
            neighbors=list(base_position.neighbors),
            embedding=(
                base_position.embedding[0] + self.rng.uniform(-0.25, 0.25),
                base_position.embedding[1] + self.rng.uniform(-0.25, 0.25),
                base_position.embedding[2] + self.rng.uniform(-0.25, 0.25),
            ),
        )
        child = parent.spawn_child(self.next_cell_id, stage=stage, fate=fate, position=child_position)
        self.cells[child.id] = child
        self.next_cell_id += 1
        self.trace.record("division", parent_cell_id=parent.id, child_cell_id=child.id, child_stage=stage, fate=fate)
        return child

    def _reprogram_for_regeneration(self, niche: Niche) -> AgentCell | None:
        regeneration_pressure = self.environment.morphogens.signal("repair_pressure") + self.environment.morphogens.signal("niche_vacancy")
        candidates = [cell for cell in self.cells.values() if cell.fate != niche.required_fate and cell.can_reprogram(regeneration_pressure)]
        if not candidates:
            return None
        candidates.sort(key=lambda cell: (_graph_distance(cell.position, niche.position), cell.id))
        cell = candidates[0]
        cell.stage = "progenitor"
        cell.fate = niche.required_fate
        cell.niche_id = None
        cell.expressed_gene_ids = []
        cell.energy *= 0.6
        cell.record_event("reprogramming", target_fate=niche.required_fate, niche_id=niche.id)
        self.trace.record("reprogramming", cell_id=cell.id, target_fate=niche.required_fate, niche_id=niche.id)
        return cell

    def _update_cell_positions(self) -> None:
        for cell in self.cells.values():
            if cell.niche_id is None:
                continue
            niche = self.environment.niche_by_id(cell.niche_id)
            cell.position = _move_position_toward(cell.position, niche.position, fraction=0.35)

    def _age_cells(self) -> None:
        for cell in self.cells.values():
            cell.age += 1
            if cell.stage == "stem" and cell.niche_id is None:
                cell.fate = "stem"


def _embedding_distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return hypot(hypot(left[0] - right[0], left[1] - right[1]), left[2] - right[2])


def _graph_distance(left: CellPosition, right: CellPosition) -> float:
    if left.node_id == right.node_id:
        return 0.0
    if left.node_id in right.neighbors or right.node_id in left.neighbors:
        return 1.0
    if left.region and right.region and left.region == right.region:
        return 2.0
    return 3.0 + _embedding_distance(left.embedding, right.embedding)


def _move_position_toward(left: CellPosition, right: CellPosition, fraction: float) -> CellPosition:
    return CellPosition(
        node_id=left.node_id,
        region=left.region,
        neighbors=list(left.neighbors),
        embedding=(
            left.embedding[0] + (right.embedding[0] - left.embedding[0]) * fraction,
            left.embedding[1] + (right.embedding[1] - left.embedding[1]) * fraction,
            left.embedding[2] + (right.embedding[2] - left.embedding[2]) * fraction,
        ),
    )


def _stage_rank(stage: str) -> int:
    return {"transit_amplifying": 0, "progenitor": 1, "stem": 2}.get(stage, 3)

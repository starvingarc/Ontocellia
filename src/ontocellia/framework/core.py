from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from random import Random
from typing import Any

from ontocellia.framework.cell import AgentCell, CellPosition, StemCellState
from ontocellia.framework.communication import CommunicationPolicy, CommunicationRuntime, ContextHomeostasisRuntime, ContextMetabolismRuntime, ExtracellularMatrix
from ontocellia.framework.fate import FateLandscape
from ontocellia.framework.genome import AgentGenome, Gene
from ontocellia.framework.resources import ResourceCompetitionPolicy, ResourceCompetitionReport, ResourceCompetitionRuntime
from ontocellia.framework.selection import OrganFeedbackSignal, OrganSelectionField, OrganSelectionReport, OrganSelectionTarget, OrganValidationResult
from ontocellia.framework.topology import TissueTopology


@dataclass(slots=True)
class MorphogenSource:
    id: str
    signal: str
    amount: float
    position: CellPosition | tuple[float, ...] | list[float] | dict[str, Any]
    radius: float = 1.0

    def __post_init__(self) -> None:
        self.position = CellPosition.from_value(self.position)


@dataclass(slots=True)
class MorphogenGradient:
    signal: str
    source_id: str
    position: CellPosition
    amount: float


@dataclass(slots=True)
class MorphogenField:
    """Task and tissue signals that induce gene expression."""

    signals: dict[str, float] = field(default_factory=dict)
    sources: list[MorphogenSource] = field(default_factory=list)

    def signal(self, name: str) -> float:
        return float(self.signals.get(name, 0.0))

    def emit(self, name: str, amount: float) -> None:
        self.signals[name] = max(0.0, self.signal(name) + amount)

    def emit_at(
        self,
        name: str,
        amount: float,
        position: CellPosition | tuple[float, ...] | list[float] | dict[str, Any],
        radius: float = 1.0,
        source_id: str | None = None,
    ) -> None:
        self.sources.append(
            MorphogenSource(
                id=source_id or f"{name}-{len(self.sources)}",
                signal=name,
                amount=max(0.0, float(amount)),
                position=position,
                radius=max(0.0, float(radius)),
            )
        )

    def local_signals(self, position: CellPosition | tuple[float, ...] | list[float] | dict[str, Any], topology: TissueTopology | None = None) -> dict[str, float]:
        target = CellPosition.from_value(position)
        signals = {str(name): float(value) for name, value in self.signals.items()}
        for source in self.sources:
            distance = _field_distance(source.position, target, topology)
            if source.radius and distance > source.radius:
                continue
            signals[source.signal] = signals.get(source.signal, 0.0) + source.amount / (1.0 + distance)
        return signals

    def signal_at(self, name: str, position: CellPosition | tuple[float, ...] | list[float] | dict[str, Any], topology: TissueTopology | None = None) -> float:
        return float(self.local_signals(position, topology).get(name, 0.0))

    def decay(self, rate: float = 0.92) -> None:
        for name, value in list(self.signals.items()):
            self.signals[name] = max(0.0, float(value) * rate)
        decayed: list[MorphogenSource] = []
        for source in self.sources:
            source.amount = max(0.0, float(source.amount) * rate)
            if source.amount >= 0.001:
                decayed.append(source)
        self.sources = decayed


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
    topology: TissueTopology | None = None
    fate_landscape: FateLandscape = field(default_factory=FateLandscape.default)
    selection_targets: OrganSelectionTarget = field(default_factory=OrganSelectionTarget)
    organ_feedback: OrganFeedbackSignal | None = None
    resource_policy: ResourceCompetitionPolicy = field(default_factory=ResourceCompetitionPolicy)
    matrix: ExtracellularMatrix | dict[str, Any] = field(default_factory=ExtracellularMatrix)
    communication_policy: CommunicationPolicy = field(default_factory=CommunicationPolicy)
    mcp_adapter: Any | None = None

    def __post_init__(self) -> None:
        if self.topology is None:
            self.topology = TissueTopology.from_niches(self.niches)
        if isinstance(self.matrix, dict):
            self.matrix = ExtracellularMatrix()

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
    organ_selection_field: OrganSelectionField | None = field(default_factory=OrganSelectionField)
    last_organ_selection_report: OrganSelectionReport | None = None
    resource_runtime: ResourceCompetitionRuntime | None = field(default_factory=ResourceCompetitionRuntime)
    last_resource_report: ResourceCompetitionReport | None = None
    communication_runtime: CommunicationRuntime | None = field(default_factory=CommunicationRuntime)
    development_stage: str = "proliferating"
    origin_cell_id: int = 0
    min_population_before_differentiation: int = 4
    target_population: int | None = None

    @classmethod
    def seeded(
        cls,
        genome: AgentGenome,
        environment: TaskMicroenvironment,
        stem_cells: int = 1,
        seed: int = 0,
    ) -> "TissueRuntime":
        rng = Random(seed)
        cells: dict[int, AgentCell] = {}
        for cell_id in range(stem_cells):
            is_origin = cell_id == 0
            cells[cell_id] = AgentCell(
                id=cell_id,
                stage="stem",
                fate="stem",
                position=CellPosition(
                    node_id="zygote-origin" if is_origin else f"stem-reserve-{cell_id}",
                    region="stem-origin" if is_origin else "stem-reserve",
                    embedding=(rng.uniform(3.5, 6.5), rng.uniform(3.5, 6.5), rng.uniform(0.0, 2.0)),
                ),
                stage_state=StemCellState(plasticity=1.0, division_potential=1.0),
            )
            if is_origin:
                cells[cell_id].record_event("stem_origin", potency="totipotent")
        runtime = cls(
            genome=genome,
            environment=environment,
            cells=cells,
            rng=rng,
            next_cell_id=stem_cells,
            origin_cell_id=0,
            target_population=max(1, sum(max(1, niche.demand) for niche in environment.niches) + 1),
        )
        assert runtime.environment.topology is not None
        for cell in runtime.cells.values():
            runtime.environment.topology.ensure_node(cell.position)
        runtime.trace.record("seed", stem_cells=stem_cells, origin_cell_id=runtime.origin_cell_id)
        return runtime

    def develop(
        self,
        ticks: int = 1,
        validation_results: list[OrganValidationResult] | None = None,
        contribution_report: Any | None = None,
    ) -> None:
        for _ in range(ticks):
            self.tick_count += 1
            self._refresh_niche_occupancy()
            self._resolve_vacancies()
            self._proliferate()
            if self._can_differentiate():
                self._fill_open_niches()
            self._update_cell_positions()
            self._age_cells()
            self._apply_organ_selection(validation_results)
            self._apply_resource_competition(contribution_report=contribution_report, validation_results=validation_results)
            self._apply_context_metabolism()
            self.environment.morphogens.decay()
            self._update_development_stage()
            self.trace.record(
                "tick",
                tick=self.tick_count,
                development_stage=self.development_stage,
                fate_counts=self.fate_counts(),
                stage_counts=self.stage_counts(),
            )

    def fate_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for cell in self.cells.values():
            if cell.alive:
                counts[cell.fate] = counts.get(cell.fate, 0) + 1
        return counts

    def stage_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for cell in self.cells.values():
            if cell.alive:
                stage = str(cell.stage)
                counts[stage] = counts.get(stage, 0) + 1
        return counts

    def niche_occupancy(self) -> dict[str, int]:
        self._refresh_niche_occupancy()
        return {niche.id: len(niche.occupied_by) for niche in self.environment.niches}

    def clear_cell(self, cell_id: int, reason: str = "cleared") -> None:
        cell = self.cells.pop(cell_id)
        signal_position = cell.position
        if cell.niche_id is not None:
            niche = self.environment.niche_by_id(cell.niche_id)
            niche.vacant_replacements.append(cell.id)
            signal_position = niche.position
        self.environment.morphogens.emit("damage", 1.0)
        self.environment.morphogens.emit("niche_vacancy", 1.0)
        self.environment.morphogens.emit("repair_pressure", 0.6)
        self.environment.morphogens.emit_at("damage", 1.0, signal_position, radius=2.0, source_id=f"damage-{cell.id}-{self.tick_count}")
        self.environment.morphogens.emit_at("niche_vacancy", 1.0, signal_position, radius=2.0, source_id=f"vacancy-{cell.id}-{self.tick_count}")
        self.environment.morphogens.emit_at("repair_pressure", 0.6, signal_position, radius=2.0, source_id=f"repair-{cell.id}-{self.tick_count}")
        self.trace.record(
            "apoptosis",
            cell_id=cell.id,
            fate=cell.fate,
            niche_id=cell.niche_id,
            reason=reason,
        )

    def execute(self, effectors: Any | None = None) -> list[dict[str, Any]]:
        if effectors is not None:
            actions = [intent.as_dict() for intent in effectors.emit_intents(self)]
            self.communicate(actions)
            self._apply_resource_competition(actions=actions)
            return actions
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
        self.communicate(actions)
        self._apply_resource_competition(actions=actions)
        return actions

    def execute_actions(self, actions: list[dict[str, Any]], execution_runtime: Any, policy: Any) -> list[Any]:
        results = execution_runtime.execute(self, actions, policy)
        self._apply_resource_competition(actions=actions, tool_results=results)
        return results

    def communicate(self, actions: list[dict[str, Any]] | None = None) -> dict[str, list[dict[str, Any]]]:
        if (
            self.communication_runtime is None
            or not hasattr(self.environment, "communication_policy")
            or not hasattr(self.environment, "matrix")
        ):
            return {"messages": [], "deliveries": [], "handoffs": []}
        messages = self.communication_runtime.emit_from_actions(self, list(actions or []))
        deliveries = self.communication_runtime.route(self, messages)
        receipts = self.communication_runtime.resolve_handoffs(self)
        self.environment.matrix.decay(self.tick_count)
        return {
            "messages": [message.as_dict() for message in messages],
            "deliveries": [delivery.as_dict() for delivery in deliveries],
            "handoffs": [receipt.as_dict() for receipt in receipts],
        }

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
        for niche in sorted(self.environment.niches, key=_niche_priority):
            while len(niche.occupied_by) < niche.demand:
                source = self._select_plastic_cell(prefer_near=niche.position)
                if source is None:
                    return
                committed = self._differentiate(source, niche)
                niche.occupied_by.append(committed.id)

    def _proliferate(self) -> None:
        target_population = self.target_population or max(1, sum(max(1, niche.demand) for niche in self.environment.niches) + 1)
        force_regeneration = any(niche.vacant_replacements for niche in self.environment.niches)
        if len(self.cells) >= target_population and not force_regeneration:
            return
        available = [cell for cell in self.cells.values() if cell.niche_id is None and cell.can_divide()]
        if not available:
            return
        available.sort(key=lambda cell: (_stage_rank_for_proliferation(str(cell.stage)), cell.id))
        parent = available[0]
        parent_stage = str(parent.stage)
        if parent_stage == "stem":
            child = self._spawn_child(parent, stage="progenitor", fate="stem", position=parent.position)
            self.trace.record("asymmetric_division", parent_cell_id=parent.id, child_cell_id=child.id)
        elif parent_stage == "progenitor":
            child = self._spawn_child(parent, stage="transit_amplifying", fate=parent.fate, position=parent.position)
            self.trace.record("progenitor_amplification", parent_cell_id=parent.id, child_cell_id=child.id)
        else:
            child = self._spawn_child(parent, stage="transit_amplifying", fate=parent.fate, position=parent.position)
            self.trace.record("progenitor_amplification", parent_cell_id=parent.id, child_cell_id=child.id)
        self.trace.record(
            "proliferation",
            parent_cell_id=parent.id,
            child_cell_id=child.id,
            parent_stage=parent_stage,
            child_stage=str(child.stage),
            population=len(self.cells),
        )

    def _can_differentiate(self) -> bool:
        if any(niche.vacant_replacements for niche in self.environment.niches):
            return True
        return len(self.cells) >= self.min_population_before_differentiation

    def _select_plastic_cell(self, prefer_near: CellPosition | tuple[float, ...] | list[float] | dict[str, Any]) -> AgentCell | None:
        target = CellPosition.from_value(prefer_near)
        candidates = [
            cell
            for cell in self.cells.values()
            if cell.stage in {"stem", "progenitor", "transit_amplifying"} and cell.niche_id is None
        ]
        if not candidates:
            return None
        topology = _environment_topology(self.environment)
        candidates.sort(
            key=lambda cell: (
                _field_distance(cell.position, target, topology),
                -_gradient_strength(self.environment.morphogens, cell.position, topology),
                -_nearby_adhesion_score(cell, self.cells.values(), topology),
                _stage_rank(str(cell.stage)),
                cell.id,
            )
        )
        return candidates[0]

    def _update_development_stage(self) -> None:
        previous = self.development_stage
        if any(niche.vacant_replacements for niche in self.environment.niches):
            next_stage = "regenerating"
        elif len(self.cells) < self.min_population_before_differentiation:
            next_stage = "proliferating"
        elif any(cell.stage == "differentiated" for cell in self.cells.values()):
            if all(len(niche.occupied_by) >= niche.demand for niche in self.environment.niches):
                next_stage = "mature"
            else:
                next_stage = "differentiating"
        else:
            next_stage = "differentiating"
        self.development_stage = next_stage
        if next_stage != previous:
            self.trace.record("development_stage_changed", tick=self.tick_count, previous=previous, current=next_stage)

    def _differentiate(self, cell: AgentCell, niche: Niche) -> AgentCell:
        topology = _environment_topology(self.environment)
        local_signals = self.environment.morphogens.local_signals(cell.position, topology)
        decision = _environment_fate_landscape(self.environment).decide(cell, self.genome, local_signals, niche_bias=niche.required_fate)
        cell.position = _move_position_toward(cell.position, niche.position, fraction=0.85)
        cell.commit_to_fate(niche.required_fate, niche.id, self.genome, local_signals, organ_feedback=_organ_feedback_for_expression(self.environment))
        cell.position = CellPosition(
            node_id=niche.position.node_id,
            region=niche.position.region,
            neighbors=list(niche.position.neighbors),
            embedding=cell.position.embedding,
        )
        self.trace.record(
            "fate_decision",
            cell_id=cell.id,
            niche_id=niche.id,
            selected_fate=decision.fate,
            committed_fate=niche.required_fate,
            score=decision.score,
            scores=decision.scores,
            threshold=decision.threshold,
            reason=decision.reason,
            niche_bias=decision.niche_bias,
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
        topology = _environment_topology(self.environment)
        if topology is not None:
            topology.ensure_node(child.position)
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
            topology = _environment_topology(self.environment)
            if topology is not None and cell.position.node_id != niche.position.node_id:
                next_position = topology.nearest_step_toward(cell.position, niche.position)
                if next_position.node_id != cell.position.node_id:
                    cell.position = CellPosition(
                        node_id=next_position.node_id,
                        region=next_position.region,
                        neighbors=list(next_position.neighbors),
                        embedding=_move_position_toward(cell.position, next_position, fraction=1.0).embedding,
                    )
                    continue
            cell.position = _move_position_toward(cell.position, niche.position, fraction=0.35)

    def _age_cells(self) -> None:
        for cell in self.cells.values():
            cell.age += 1
            if cell.stage == "stem" and cell.niche_id is None:
                cell.fate = "stem"

    def _apply_organ_selection(self, validation_results: list[OrganValidationResult] | None = None) -> None:
        if self.organ_selection_field is None:
            return
        report = self.organ_selection_field.evaluate(self, validation_results)
        self.last_organ_selection_report = report
        self.environment.organ_feedback = report.feedback
        self.environment.morphogens.emit("selection_pressure", report.feedback.selection_pressure)
        self.environment.morphogens.emit("validation_pressure", report.feedback.validation_pressure)
        self.environment.morphogens.emit("risk_pressure", report.feedback.risk_pressure)
        self.environment.morphogens.emit("resource_pressure", report.feedback.resource_pressure)
        self.environment.morphogens.emit("reward_signal", report.feedback.reward_signal)
        if validation_results:
            ContextHomeostasisRuntime().apply_validation_feedback(self.environment.matrix, validation_results, current_tick=self.tick_count)
        self.trace.record("organ_selection", tick=self.tick_count, **report.as_dict())

    def _apply_context_metabolism(self) -> None:
        if not hasattr(self.environment, "matrix") or not hasattr(self.environment, "communication_policy"):
            return
        ContextMetabolismRuntime().metabolize(self)

    def _apply_resource_competition(
        self,
        *,
        contribution_report: Any | None = None,
        actions: list[dict[str, Any]] | None = None,
        tool_results: list[Any] | None = None,
        validation_results: list[OrganValidationResult] | None = None,
    ) -> None:
        if self.resource_runtime is None:
            return
        self.resource_runtime.apply(
            self,
            policy=getattr(self.environment, "resource_policy", None),
            contribution_report=contribution_report,
            actions=actions,
            tool_results=tool_results,
            validation_results=validation_results,
        )


def _embedding_distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return hypot(hypot(left[0] - right[0], left[1] - right[1]), left[2] - right[2])


def _environment_topology(environment: Any) -> TissueTopology | None:
    return getattr(environment, "topology", None)


def _environment_fate_landscape(environment: Any) -> FateLandscape:
    return getattr(environment, "fate_landscape", FateLandscape.default())


def _organ_feedback_for_expression(environment: Any) -> dict[str, float]:
    feedback = getattr(environment, "organ_feedback", None)
    if feedback is None:
        return {}
    by_fate = dict(getattr(feedback, "by_fate", {}))
    by_gene = dict(getattr(feedback, "by_gene", {}))
    return {**by_fate, **by_gene}


def _field_distance(left: CellPosition, right: CellPosition, topology: TissueTopology | None) -> float:
    if topology is not None:
        return topology.distance(left, right)
    return _graph_distance(left, right)


def _gradient_strength(morphogens: MorphogenField, position: CellPosition, topology: TissueTopology | None) -> float:
    return sum(morphogens.local_signals(position, topology).values())


def _nearby_adhesion_score(cell: AgentCell, cells: Any, topology: TissueTopology | None) -> float:
    score = 0.0
    for other in cells:
        if other.id == cell.id or not other.alive:
            continue
        distance = _field_distance(cell.position, other.position, topology)
        if distance > 1.0:
            continue
        score += cell.adhesion_score(other.fate)
    return score


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


def _stage_rank_for_proliferation(stage: str) -> int:
    return {"stem": 0, "progenitor": 1, "transit_amplifying": 2}.get(stage, 3)


def _niche_priority(niche: Niche) -> tuple[int, str]:
    fate_order = {
        "repair": 0,
        "explorer": 1,
        "reviewer": 2,
        "builder": 3,
        "planner": 4,
        "memory": 5,
        "quiescent": 6,
    }
    return (fate_order.get(niche.required_fate, 10), niche.id)

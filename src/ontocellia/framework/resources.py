from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(slots=True)
class ResourceCompetitionPolicy:
    enabled: bool = True
    population_cap: int | None = None
    maintenance_cost: float = 0.01
    differentiated_cost: float = 0.01
    action_intent_cost: float = 0.015
    tool_cost_weight: float = 0.1
    latency_cost_weight: float = 0.01
    contribution_reward: float = 0.08
    negative_contribution_penalty: float = 0.08
    low_energy_threshold: float = 0.35
    quiescence_threshold: float = 0.08
    apoptosis_threshold: float = 0.02
    allow_quiescence: bool = False
    allow_apoptosis: bool = False
    over_cap_pressure_weight: float = 0.25
    low_energy_pressure_weight: float = 0.35
    max_energy: float = 1.2

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CellResourceDelta:
    cell_id: int
    stage: str
    fate: str
    energy_before: float
    energy_after: float
    maintenance_cost: float = 0.0
    action_cost: float = 0.0
    tool_cost: float = 0.0
    contribution_reward: float = 0.0
    negative_penalty: float = 0.0
    stress_delta: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResourceCompetitionReport:
    tick: int
    population: int
    population_cap: int | None
    average_cell_energy: float
    population_pressure: float
    low_energy_pressure: float
    resource_pressure: float
    resource_efficiency: float
    cell_deltas: dict[str, CellResourceDelta] = field(default_factory=dict)
    removed_cell_ids: list[str] = field(default_factory=list)
    quiescent_cell_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "population": self.population,
            "population_cap": self.population_cap,
            "average_cell_energy": self.average_cell_energy,
            "population_pressure": self.population_pressure,
            "low_energy_pressure": self.low_energy_pressure,
            "resource_pressure": self.resource_pressure,
            "resource_efficiency": self.resource_efficiency,
            "cell_deltas": {cell_id: delta.as_dict() for cell_id, delta in self.cell_deltas.items()},
            "removed_cell_ids": list(self.removed_cell_ids),
            "quiescent_cell_ids": list(self.quiescent_cell_ids),
        }


class ResourceCompetitionRuntime:
    def apply(
        self,
        tissue: Any,
        *,
        policy: ResourceCompetitionPolicy | None = None,
        contribution_report: Any | None = None,
        actions: list[dict[str, Any]] | None = None,
        tool_results: list[Any] | None = None,
        validation_results: list[Any] | None = None,
    ) -> ResourceCompetitionReport:
        active_policy = policy or getattr(getattr(tissue, "environment", None), "resource_policy", None) or ResourceCompetitionPolicy()
        if not active_policy.enabled:
            report = _empty_report(tissue, active_policy)
            _set_last_report(tissue, report)
            return report

        cells = [cell for cell in getattr(tissue, "cells", {}).values() if getattr(cell, "alive", True)]
        population = len(cells)
        cap = active_policy.population_cap
        if cap is not None:
            tissue.target_population = min(int(cap), int(getattr(tissue, "target_population", cap) or cap))
        population_pressure = _population_pressure(population, cap, active_policy)
        contribution = _cell_contribution(contribution_report)
        action_counts = _action_counts(actions or [])
        tool_costs = _tool_costs(tool_results or [], active_policy)

        deltas: dict[str, CellResourceDelta] = {}
        for cell in sorted(cells, key=lambda item: item.id):
            before = float(getattr(cell, "energy", 0.0))
            score = contribution.get(int(cell.id), {"positive": 0.0, "negative": 0.0})
            maintenance = active_policy.maintenance_cost
            if getattr(cell, "stage", "") == "differentiated":
                maintenance += active_policy.differentiated_cost
            action_cost = action_counts.get(int(cell.id), 0) * active_policy.action_intent_cost
            tool_cost = tool_costs.get(int(cell.id), 0.0)
            reward = float(score.get("positive", 0.0)) * active_policy.contribution_reward
            penalty = float(score.get("negative", 0.0)) * active_policy.negative_contribution_penalty
            after = _clamp(before - maintenance - action_cost - tool_cost - penalty + reward, 0.0, active_policy.max_energy)
            cell.energy = after
            stress_delta = _clamp(penalty + population_pressure * 0.25, 0.0, 1.0)
            cell.stress = _clamp(float(getattr(cell, "stress", 0.0)) + stress_delta, 0.0, 1.0)
            deltas[str(cell.id)] = CellResourceDelta(
                cell_id=int(cell.id),
                stage=str(getattr(cell, "stage", "")),
                fate=str(getattr(cell, "fate", "")),
                energy_before=round(before, 6),
                energy_after=round(after, 6),
                maintenance_cost=round(maintenance, 6),
                action_cost=round(action_cost, 6),
                tool_cost=round(tool_cost, 6),
                contribution_reward=round(reward, 6),
                negative_penalty=round(penalty, 6),
                stress_delta=round(stress_delta, 6),
            )

        quiescent = _apply_quiescence(tissue, active_policy)
        removed = _apply_apoptosis(tissue, active_policy)
        remaining = [cell for cell in getattr(tissue, "cells", {}).values() if getattr(cell, "alive", True)]
        average_energy = sum(float(getattr(cell, "energy", 0.0)) for cell in remaining) / max(1, len(remaining))
        low_energy_pressure = _clamp(max(0.0, active_policy.low_energy_threshold - average_energy) * active_policy.low_energy_pressure_weight)
        resource_pressure = _clamp(population_pressure + low_energy_pressure)
        resource_efficiency = _clamp(average_energy * (1.0 - population_pressure))
        _emit_resource_signals(tissue, resource_pressure, population_pressure, resource_efficiency)
        report = ResourceCompetitionReport(
            tick=int(getattr(tissue, "tick_count", 0)),
            population=len(remaining),
            population_cap=cap,
            average_cell_energy=round(average_energy, 6),
            population_pressure=round(population_pressure, 6),
            low_energy_pressure=round(low_energy_pressure, 6),
            resource_pressure=round(resource_pressure, 6),
            resource_efficiency=round(resource_efficiency, 6),
            cell_deltas=deltas,
            removed_cell_ids=removed,
            quiescent_cell_ids=quiescent,
        )
        _set_last_report(tissue, report)
        trace = getattr(tissue, "trace", None)
        if trace is not None:
            trace.record("resource_competition", **report.as_dict())
        return report


def _empty_report(tissue: Any, policy: ResourceCompetitionPolicy) -> ResourceCompetitionReport:
    cells = [cell for cell in getattr(tissue, "cells", {}).values() if getattr(cell, "alive", True)]
    average_energy = sum(float(getattr(cell, "energy", 0.0)) for cell in cells) / max(1, len(cells))
    return ResourceCompetitionReport(
        tick=int(getattr(tissue, "tick_count", 0)),
        population=len(cells),
        population_cap=policy.population_cap,
        average_cell_energy=round(average_energy, 6),
        population_pressure=0.0,
        low_energy_pressure=0.0,
        resource_pressure=0.0,
        resource_efficiency=round(_clamp(average_energy), 6),
    )


def _population_pressure(population: int, cap: int | None, policy: ResourceCompetitionPolicy) -> float:
    if cap is None or cap <= 0 or population <= cap:
        return 0.0
    return _clamp(((population - cap) / cap) * policy.over_cap_pressure_weight)


def _cell_contribution(report: Any | None) -> dict[int, dict[str, float]]:
    if report is None:
        return {}
    if hasattr(report, "as_dict"):
        data = report.as_dict()
    elif isinstance(report, dict):
        data = report
    else:
        return {}
    scores = data.get("scores", [])
    result: dict[int, dict[str, float]] = {}
    for score in scores:
        node_id = str(score.get("node_id", ""))
        if not node_id.startswith("cell:"):
            continue
        try:
            cell_id = int(node_id.split(":", 1)[1])
        except ValueError:
            continue
        result[cell_id] = {
            "positive": float(score.get("positive", 0.0)),
            "negative": float(score.get("negative", 0.0)),
        }
    return result


def _action_counts(actions: list[dict[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for action in actions:
        try:
            cell_id = int(action.get("cell_id"))
        except (TypeError, ValueError):
            continue
        counts[cell_id] = counts.get(cell_id, 0) + 1
    return counts


def _tool_costs(tool_results: list[Any], policy: ResourceCompetitionPolicy) -> dict[int, float]:
    costs: dict[int, float] = {}
    for result in tool_results:
        data = result.as_dict() if hasattr(result, "as_dict") else dict(result)
        invocation = data.get("invocation") if isinstance(data.get("invocation"), dict) else data.get("request", {})
        try:
            cell_id = int(invocation.get("cell_id"))
        except (AttributeError, TypeError, ValueError):
            continue
        base_cost = float(data.get("cost", 0.0)) * policy.tool_cost_weight
        latency_cost = float(data.get("latency", data.get("latency_seconds", 0.0))) * policy.latency_cost_weight
        costs[cell_id] = costs.get(cell_id, 0.0) + base_cost + latency_cost
    return costs


def _apply_quiescence(tissue: Any, policy: ResourceCompetitionPolicy) -> list[str]:
    if not policy.allow_quiescence:
        return []
    quiescent: list[str] = []
    for cell in getattr(tissue, "cells", {}).values():
        if getattr(cell, "energy", 1.0) > policy.quiescence_threshold:
            continue
        if getattr(cell, "stage", "") != "differentiated":
            continue
        cell.fate = "quiescent"
        cell.record_event("resource_quiescence", energy=cell.energy)
        quiescent.append(str(cell.id))
    return quiescent


def _apply_apoptosis(tissue: Any, policy: ResourceCompetitionPolicy) -> list[str]:
    if not policy.allow_apoptosis or policy.population_cap is None:
        return []
    cells = getattr(tissue, "cells", {})
    if len(cells) <= policy.population_cap:
        return []
    origin = getattr(tissue, "origin_cell_id", None)
    candidates = [
        cell
        for cell in cells.values()
        if getattr(cell, "id", None) != origin and float(getattr(cell, "energy", 1.0)) <= policy.apoptosis_threshold
    ]
    candidates.sort(key=lambda cell: (float(getattr(cell, "energy", 0.0)), int(getattr(cell, "id", 0))))
    removed: list[str] = []
    for cell in candidates:
        if len(cells) <= policy.population_cap:
            break
        removed.append(str(cell.id))
        niche_id = getattr(cell, "niche_id", None)
        if niche_id is not None:
            try:
                niche = tissue.environment.niche_by_id(niche_id)
                niche.vacant_replacements.append(cell.id)
            except Exception:
                pass
        cells.pop(cell.id, None)
        if getattr(tissue, "trace", None) is not None:
            tissue.trace.record("resource_apoptosis", cell_id=cell.id, fate=getattr(cell, "fate", ""), niche_id=niche_id)
    return removed


def _emit_resource_signals(tissue: Any, resource_pressure: float, population_pressure: float, resource_efficiency: float) -> None:
    morphogens = getattr(getattr(tissue, "environment", None), "morphogens", None)
    if morphogens is None:
        return
    morphogens.emit("resource_pressure", resource_pressure)
    morphogens.emit("population_pressure", population_pressure)
    morphogens.emit("resource_efficiency", resource_efficiency * 0.1)


def _set_last_report(tissue: Any, report: ResourceCompetitionReport) -> None:
    try:
        tissue.last_resource_report = report
    except Exception:
        pass

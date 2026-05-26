from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ontocellia.framework.cell import DifferentiatedCellState, ProgenitorCellState


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(slots=True)
class DevelopmentalAnnealingPolicy:
    enabled: bool = True
    warmup_ticks: int = 3
    stabilization_ticks: int = 8
    initial_temperature: float = 1.0
    final_temperature: float = 0.15
    diversity_pressure: float = 0.25
    commitment_pressure: float = 0.35
    fate_lock_min: float = 0.25
    fate_lock_max: float = 0.9
    fate_lock_growth: float = 0.04
    failure_unlock: float = 0.12
    repeated_failure_threshold: int = 2
    reprogramming_pressure_threshold: float = 0.85
    reprogramming_energy_cost: float = 0.2
    max_reprogramming_per_tick: int = 1

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FateLockDelta:
    cell_id: int
    fate: str
    before: float
    after: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DevelopmentalAnnealingReport:
    tick: int
    temperature: float
    commitment: float
    diversity_pressure: float
    average_fate_lock: float
    failure_streak: int
    fate_lock_deltas: dict[str, FateLockDelta] = field(default_factory=dict)
    reprogrammed_cell_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "temperature": self.temperature,
            "commitment": self.commitment,
            "diversity_pressure": self.diversity_pressure,
            "average_fate_lock": self.average_fate_lock,
            "failure_streak": self.failure_streak,
            "fate_lock_deltas": {cell_id: delta.as_dict() for cell_id, delta in self.fate_lock_deltas.items()},
            "reprogrammed_cell_ids": list(self.reprogrammed_cell_ids),
        }


@dataclass(slots=True)
class DevelopmentalAnnealingRuntime:
    failure_streak: int = 0

    def apply(
        self,
        tissue: Any,
        *,
        policy: DevelopmentalAnnealingPolicy | None = None,
        validation_results: list[Any] | None = None,
    ) -> DevelopmentalAnnealingReport:
        active_policy = policy or getattr(getattr(tissue, "environment", None), "annealing_policy", None) or DevelopmentalAnnealingPolicy()
        self._update_failure_streak(validation_results)
        if not active_policy.enabled:
            report = self._report(tissue, active_policy, {}, [])
            _set_last_report(tissue, report)
            return report
        temperature = _temperature(int(getattr(tissue, "tick_count", 0)), active_policy)
        commitment = _clamp(1.0 - temperature)
        diversity = _clamp(temperature * active_policy.diversity_pressure)
        self._emit_annealing_signals(tissue, diversity, commitment, active_policy)
        fate_lock_deltas = self._update_fate_locks(tissue, active_policy, commitment)
        reprogrammed = self._maybe_reprogram(tissue, active_policy)
        report = self._report(tissue, active_policy, fate_lock_deltas, reprogrammed)
        _set_last_report(tissue, report)
        trace = getattr(tissue, "trace", None)
        if trace is not None:
            trace.record("developmental_annealing", **report.as_dict())
        return report

    def _update_failure_streak(self, validation_results: list[Any] | None) -> None:
        if not validation_results:
            return
        passed = [bool(getattr(result, "passed", False) if not isinstance(result, dict) else result.get("passed", False)) for result in validation_results]
        if any(not item for item in passed):
            self.failure_streak += 1
        elif passed and all(passed):
            self.failure_streak = 0

    def _emit_annealing_signals(self, tissue: Any, diversity: float, commitment: float, policy: DevelopmentalAnnealingPolicy) -> None:
        morphogens = getattr(getattr(tissue, "environment", None), "morphogens", None)
        if morphogens is None:
            return
        morphogens.emit("exploration_pressure", diversity)
        morphogens.emit("commitment_pressure", commitment * policy.commitment_pressure)
        if self.failure_streak >= policy.repeated_failure_threshold:
            morphogens.emit("reprogramming_pressure", min(1.0, self.failure_streak * 0.2))

    def _update_fate_locks(self, tissue: Any, policy: DevelopmentalAnnealingPolicy, commitment: float) -> dict[str, FateLockDelta]:
        deltas: dict[str, FateLockDelta] = {}
        for cell in sorted(getattr(tissue, "cells", {}).values(), key=lambda item: item.id):
            state = getattr(cell, "stage_state", None)
            if not isinstance(state, DifferentiatedCellState):
                continue
            before = float(state.fate_lock)
            if self.failure_streak >= policy.repeated_failure_threshold:
                after = _clamp(before - policy.failure_unlock, policy.fate_lock_min, policy.fate_lock_max)
            else:
                after = _clamp(before + policy.fate_lock_growth * max(0.1, commitment), policy.fate_lock_min, policy.fate_lock_max)
            if after == before:
                continue
            state.fate_lock = after
            cell.epigenetic_marks.fate_locks[cell.fate] = after
            deltas[str(cell.id)] = FateLockDelta(cell.id, cell.fate, round(before, 6), round(after, 6))
        return deltas

    def _maybe_reprogram(self, tissue: Any, policy: DevelopmentalAnnealingPolicy) -> list[str]:
        if self.failure_streak < policy.repeated_failure_threshold or policy.max_reprogramming_per_tick <= 0:
            return []
        pressure = _reprogramming_pressure(tissue, self.failure_streak)
        if pressure < policy.reprogramming_pressure_threshold:
            return []
        target_fate = _target_fate_for_pressure(tissue)
        candidates = [
            cell
            for cell in getattr(tissue, "cells", {}).values()
            if getattr(cell, "stage", "") == "differentiated"
            and getattr(cell, "fate", "") != target_fate
            and isinstance(getattr(cell, "stage_state", None), DifferentiatedCellState)
            and getattr(cell.stage_state, "reprogrammable", False)
        ]
        candidates.sort(key=lambda cell: (-float(getattr(cell, "stress", 0.0)), float(getattr(cell, "energy", 1.0)), cell.id))
        reprogrammed: list[str] = []
        for cell in candidates[: policy.max_reprogramming_per_tick]:
            previous_fate = cell.fate
            previous_lock = cell.stage_state.fate_lock
            cell.stage = "progenitor"
            cell.fate = target_fate
            cell.niche_id = None
            cell.expressed_gene_ids = []
            cell.stage_state = ProgenitorCellState(target_fate=target_fate, amplification_potential=0.6)
            cell.energy = _clamp(float(cell.energy) * (1.0 - policy.reprogramming_energy_cost))
            cell.record_event("annealing_reprogramming", previous_fate=previous_fate, target_fate=target_fate, previous_fate_lock=previous_lock)
            reprogrammed.append(str(cell.id))
            trace = getattr(tissue, "trace", None)
            if trace is not None:
                trace.record(
                    "annealing_reprogramming",
                    cell_id=cell.id,
                    previous_fate=previous_fate,
                    target_fate=target_fate,
                    previous_fate_lock=previous_lock,
                    failure_streak=self.failure_streak,
                )
        return reprogrammed

    def _report(
        self,
        tissue: Any,
        policy: DevelopmentalAnnealingPolicy,
        fate_lock_deltas: dict[str, FateLockDelta],
        reprogrammed: list[str],
    ) -> DevelopmentalAnnealingReport:
        temperature = _temperature(int(getattr(tissue, "tick_count", 0)), policy)
        locks = [
            float(cell.stage_state.fate_lock)
            for cell in getattr(tissue, "cells", {}).values()
            if isinstance(getattr(cell, "stage_state", None), DifferentiatedCellState)
        ]
        average = sum(locks) / max(1, len(locks))
        return DevelopmentalAnnealingReport(
            tick=int(getattr(tissue, "tick_count", 0)),
            temperature=round(temperature, 6),
            commitment=round(_clamp(1.0 - temperature), 6),
            diversity_pressure=round(_clamp(temperature * policy.diversity_pressure), 6),
            average_fate_lock=round(average, 6),
            failure_streak=self.failure_streak,
            fate_lock_deltas=fate_lock_deltas,
            reprogrammed_cell_ids=reprogrammed,
        )


def _temperature(tick: int, policy: DevelopmentalAnnealingPolicy) -> float:
    if tick <= policy.warmup_ticks:
        return _clamp(policy.initial_temperature)
    span = max(1, policy.stabilization_ticks)
    progress = _clamp((tick - policy.warmup_ticks) / span)
    return _clamp(policy.initial_temperature + (policy.final_temperature - policy.initial_temperature) * progress)


def _reprogramming_pressure(tissue: Any, failure_streak: int) -> float:
    morphogens = getattr(getattr(tissue, "environment", None), "morphogens", None)
    if morphogens is None:
        return failure_streak * 0.2
    return (
        morphogens.signal("repair_pressure")
        + morphogens.signal("validation_pressure")
        + morphogens.signal("risk_pressure")
        + morphogens.signal("reprogramming_pressure")
        + failure_streak * 0.2
    )


def _target_fate_for_pressure(tissue: Any) -> str:
    morphogens = getattr(getattr(tissue, "environment", None), "morphogens", None)
    if morphogens is None:
        return "repair"
    repair = morphogens.signal("repair_pressure") + morphogens.signal("test_failure")
    review = morphogens.signal("review_pressure") + morphogens.signal("risk") + morphogens.signal("validation_pressure")
    return "reviewer" if review > repair else "repair"


def _set_last_report(tissue: Any, report: DevelopmentalAnnealingReport) -> None:
    try:
        tissue.last_annealing_report = report
    except Exception:
        pass

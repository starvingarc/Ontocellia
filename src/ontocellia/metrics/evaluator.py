from __future__ import annotations

from dataclasses import dataclass, field
from math import log

import numpy as np

from ontocellia.config import FATE_NAMES, LEGACY_MODE


def _entropy(distribution: np.ndarray) -> float:
    safe = distribution[distribution > 0]
    if safe.size == 0:
        return 0.0
    return float(-(safe * np.log(safe)).sum() / log(max(2, len(distribution))))


@dataclass(slots=True)
class MetricsRecorder:
    history: list[dict[str, object]] = field(default_factory=list)

    def record(self, runtime) -> dict[str, object]:
        population = len(runtime.cells)
        if population == 0:
            snapshot: dict[str, object] = {
                "tick": runtime.tick_count,
                "mode": runtime.mode,
                "population": 0,
                "heterogeneity": 0.0,
                "avg_energy": 0.0,
                "avg_stress": 0.0,
                "repair_fraction": 0.0,
                "graph_density": 0.0,
                "coverage": 0.0,
                "fate_switches": runtime.fate_switches,
                "deaths": runtime.total_deaths,
                "divisions": runtime.division_events,
                "risky_divisions": runtime.risky_divisions,
                "fate_counts": {name: 0 for name in FATE_NAMES},
                "probe_counts": {},
                "repair_activation": 0.0,
                "division_pressure": 0.0,
                "risk_exposure": 0.0,
                "contact_inhibition": 0.0,
                "attractor_occupancy": {},
            }
            self.history.append(snapshot)
            return snapshot

        energies = np.array([cell.energy for cell in runtime.cells.values()], dtype=float)
        stresses = np.array([cell.stress for cell in runtime.cells.values()], dtype=float)
        coverage = float(np.count_nonzero(runtime.substrate.crowding_map(runtime.cells) > 0.15) / (runtime.config.width * runtime.config.height))
        graph_density = 0.0
        if runtime.graph.graph.number_of_nodes() > 1:
            nodes = runtime.graph.graph.number_of_nodes()
            graph_density = 2 * runtime.graph.graph.number_of_edges() / (nodes * (nodes - 1))

        if runtime.mode == LEGACY_MODE:
            commitments = np.stack([cell.commitment for cell in runtime.cells.values()])
            repair_fraction = float(np.mean([cell.current_fate == "repair-active" for cell in runtime.cells.values()]))
            fate_counts = {name: 0 for name in FATE_NAMES}
            for cell in runtime.cells.values():
                fate_counts[cell.current_fate] += 1
            snapshot = {
                "tick": runtime.tick_count,
                "mode": runtime.mode,
                "population": population,
                "heterogeneity": _entropy(commitments.mean(axis=0)),
                "development_diversity": _entropy(commitments.mean(axis=0)),
                "avg_energy": float(energies.mean()),
                "avg_stress": float(stresses.mean()),
                "repair_fraction": repair_fraction,
                "repair_activation": repair_fraction,
                "division_pressure": float(runtime.division_events / max(1, runtime.tick_count)),
                "risk_exposure": float(runtime.risky_divisions / max(1, runtime.division_events)),
                "graph_density": float(graph_density),
                "coverage": coverage,
                "fate_switches": runtime.fate_switches,
                "deaths": runtime.total_deaths,
                "divisions": runtime.division_events,
                "risky_divisions": runtime.risky_divisions,
                "fate_counts": fate_counts,
                "probe_counts": {},
            }
            self.history.append(snapshot)
            return snapshot

        development = np.stack([cell.development_state for cell in runtime.cells.values() if cell.development_state is not None])
        probe_counts = runtime.phenotype_counts()
        repair_activation = float(np.mean([cell.repair_signal for cell in runtime.cells.values()]))
        risk_exposure = float(np.mean([runtime.environment.sample(cell.pos).get("damage", 0.0) for cell in runtime.cells.values()]))
        contact_inhibition = float(
            np.mean(
                [
                    runtime.graph.summary_for(cell.id, runtime.cells, runtime.config.desired_local_density).get("contact_inhibition", 0.0)
                    for cell in runtime.cells.values()
                ]
            )
        )
        attractor_names = sorted({name for cell in runtime.cells.values() for name in cell.attractor_potentials})
        attractor_occupancy = {
            name: float(np.mean([cell.attractor_potentials.get(name, 0.0) for cell in runtime.cells.values()]))
            for name in attractor_names
        }
        positive_mean = np.maximum(development, 0.0).mean(axis=0) + 1e-6
        snapshot = {
            "tick": runtime.tick_count,
            "mode": runtime.mode,
            "population": population,
            "heterogeneity": _entropy(positive_mean / positive_mean.sum()),
            "development_diversity": _entropy(positive_mean / positive_mean.sum()),
            "avg_energy": float(energies.mean()),
            "avg_stress": float(stresses.mean()),
            "repair_fraction": float(np.mean([cell.repair_signal > 0.35 for cell in runtime.cells.values()])),
            "repair_activation": repair_activation,
            "division_pressure": float(runtime.division_events / max(1, runtime.tick_count)),
            "risk_exposure": risk_exposure,
            "contact_inhibition": contact_inhibition,
            "graph_density": float(graph_density),
            "coverage": coverage,
            "fate_switches": runtime.fate_switches,
            "deaths": runtime.total_deaths,
            "divisions": runtime.division_events,
            "risky_divisions": runtime.risky_divisions,
            "fate_counts": {},
            "probe_counts": probe_counts,
            "attractor_occupancy": attractor_occupancy,
        }
        self.history.append(snapshot)
        return snapshot

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from ontocellia.config import OntocelliaConfig


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(left, right) / denom)


@dataclass(slots=True)
class InteractionGraph:
    config: OntocelliaConfig
    graph: nx.Graph = field(default_factory=nx.Graph)
    previous_edges: dict[tuple[int, int], dict[str, float]] = field(default_factory=dict)

    def rebuild(self, cells: dict[int, object]) -> None:
        previous = {
            tuple(sorted((left, right))): dict(data)
            for left, right, data in self.graph.edges(data=True)
        }
        self.graph.clear()
        self.graph.add_nodes_from(cells)
        if not self.config.enable_graph:
            self.previous_edges = previous
            return

        ids = list(cells)
        for index, cell_id in enumerate(ids):
            distances: list[tuple[float, int]] = []
            for other_id in ids[index + 1 :]:
                dist = float(np.linalg.norm(cells[cell_id].pos - cells[other_id].pos))
                if dist <= self.config.communication_radius:
                    distances.append((dist, other_id))
            distances.sort(key=lambda item: item[0])
            for dist, other_id in distances[: self.config.max_neighbors]:
                left = cells[cell_id]
                right = cells[other_id]
                similarity = self._similarity(left, right)
                area = float(np.clip(1.0 - dist / self.config.communication_radius, 0.0, 1.0))
                crowding = float(np.clip((left.quiescence_state + right.quiescence_state) * 0.5, 0.0, 1.0))
                persistence = previous.get(tuple(sorted((cell_id, other_id))), {}).get("persistence", area)
                weight = float(np.clip(0.25 + 0.3 * similarity + 0.25 * area + 0.2 * persistence - 0.1 * crowding, 0.05, 1.0))
                self.graph.add_edge(
                    cell_id,
                    other_id,
                    weight=weight,
                    contact_area=area,
                    relative_density=crowding,
                    persistence=float(np.clip(0.6 * persistence + 0.4 * area, 0.0, 1.0)),
                )
        self.previous_edges = previous

    def _similarity(self, left, right) -> float:
        if left.development_state is not None and right.development_state is not None:
            return 0.5 + 0.5 * _cosine_similarity(left.development_state, right.development_state)
        return 1.0 - float(np.abs(left.commitment.argmax() - right.commitment.argmax())) / max(1, left.commitment.size - 1)

    def summary_for(self, cell_id: int, cells: dict[int, object], desired_local_density: float) -> dict[str, np.ndarray | float]:
        if not cells:
            return {
                "hidden_mean": np.zeros(self.config.hidden_dim, dtype=float),
                "fate_mean": np.zeros(4, dtype=float),
                "development_mean": np.zeros(4, dtype=float),
                "graph_density": 0.0,
                "local_demand": 0.0,
                "contact_area_mean": 0.0,
                "contact_persistence_mean": 0.0,
                "relative_density_mean": 0.0,
                "neighbor_quiescence": 0.0,
                "contact_inhibition": 0.0,
                "community_signal": 0.0,
            }
        template = next(iter(cells.values()))
        hidden = np.zeros_like(template.hidden_state)
        fate = np.zeros_like(template.commitment)
        development = np.zeros_like(template.development_state if template.development_state is not None else template.commitment)
        graph_density = 0.0
        local_demand = 0.0
        contact_area_mean = 0.0
        contact_persistence_mean = 0.0
        relative_density_mean = 0.0
        neighbor_quiescence = 0.0
        contact_inhibition = 0.0
        community_signal = 0.0
        if self.graph.has_node(cell_id) and self.graph.degree(cell_id) > 0:
            neighbors = list(self.graph.neighbors(cell_id))
            weights = np.array([self.graph[cell_id][neighbor]["weight"] for neighbor in neighbors], dtype=float)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            areas = []
            persistences = []
            densities = []
            quiescence_values = []
            community_values = []
            for idx, neighbor in enumerate(neighbors):
                edge = self.graph[cell_id][neighbor]
                hidden += cells[neighbor].hidden_state * weights[idx]
                fate += cells[neighbor].commitment * weights[idx]
                if cells[neighbor].development_state is not None:
                    development += cells[neighbor].development_state * weights[idx]
                else:
                    development += cells[neighbor].commitment * weights[idx]
                areas.append(edge.get("contact_area", 0.0))
                persistences.append(edge.get("persistence", 0.0))
                densities.append(edge.get("relative_density", 0.0))
                quiescence_values.append(getattr(cells[neighbor], "quiescence_state", 0.0))
                same_community = (
                    getattr(cells[cell_id], "community_id", None) is not None
                    and getattr(cells[cell_id], "community_id", None) == getattr(cells[neighbor], "community_id", None)
                )
                community_values.append(1.0 if same_community else 0.0)
            graph_density = float(np.clip(len(neighbors) / max(1, self.config.max_neighbors), 0.0, 1.0))
            contact_area_mean = float(np.mean(areas))
            contact_persistence_mean = float(np.mean(persistences))
            relative_density_mean = float(np.mean(densities))
            neighbor_quiescence = float(np.mean(quiescence_values))
            contact_inhibition = float(np.clip(contact_area_mean * (1.0 - development.max()) + neighbor_quiescence * 0.4, 0.0, 1.0))
            community_signal = float(np.mean(community_values))
        cell = cells[cell_id]
        local_neighbors = sum(
            1
            for other in cells.values()
            if other.id != cell.id and np.linalg.norm(other.pos - cell.pos) <= self.config.spatial_radius
        )
        local_density = local_neighbors / max(1.0, np.pi * self.config.spatial_radius**2)
        local_demand = float(np.clip(desired_local_density - local_density, 0.0, 1.0))
        return {
            "hidden_mean": hidden,
            "fate_mean": fate,
            "development_mean": development,
            "graph_density": graph_density,
            "local_demand": local_demand,
            "contact_area_mean": contact_area_mean,
            "contact_persistence_mean": contact_persistence_mean,
            "relative_density_mean": relative_density_mean,
            "neighbor_quiescence": neighbor_quiescence,
            "contact_inhibition": contact_inhibition,
            "community_signal": community_signal,
        }

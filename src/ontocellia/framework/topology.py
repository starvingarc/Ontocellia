from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot
from typing import Any

from ontocellia.framework.cell import CellPosition


@dataclass(slots=True)
class TopologyNode:
    id: str
    region: str = ""
    neighbors: list[str] = field(default_factory=list)
    embedding: tuple[float, float, float] = (0.0, 0.0, 0.0)
    metadata: dict[str, Any] = field(default_factory=dict)

    def position(self) -> CellPosition:
        return CellPosition(
            node_id=self.id,
            region=self.region,
            neighbors=list(self.neighbors),
            embedding=self.embedding,
        )


@dataclass(slots=True)
class TissueTopology:
    nodes: dict[str, TopologyNode] = field(default_factory=dict)

    @classmethod
    def from_niches(cls, niches: list[Any]) -> "TissueTopology":
        nodes: dict[str, TopologyNode] = {}
        for niche in niches:
            position = CellPosition.from_value(niche.position)
            node_id = position.node_id or str(niche.id)
            neighbors = list(position.neighbors)
            nodes[node_id] = TopologyNode(
                id=node_id,
                region=position.region or str(niche.id),
                neighbors=neighbors,
                embedding=position.embedding,
                metadata={"niche_id": str(niche.id), "required_fate": str(niche.required_fate)},
            )
        return cls(nodes=nodes)

    def ensure_node(self, position: CellPosition) -> None:
        if not position.node_id or position.node_id in self.nodes:
            return
        self.nodes[position.node_id] = TopologyNode(
            id=position.node_id,
            region=position.region,
            neighbors=list(position.neighbors),
            embedding=position.embedding,
        )

    def node(self, node_id: str) -> TopologyNode:
        try:
            return self.nodes[node_id]
        except KeyError as error:
            raise KeyError(f"unknown topology node: {node_id}") from error

    def neighbors(self, node_id: str) -> list[str]:
        if node_id not in self.nodes:
            return []
        return sorted(self.nodes[node_id].neighbors)

    def distance(self, left: CellPosition | str, right: CellPosition | str) -> float:
        left_position = _position(left, self)
        right_position = _position(right, self)
        if left_position.node_id and left_position.node_id == right_position.node_id:
            return 0.0
        graph_distance = self._graph_distance(left_position.node_id, right_position.node_id)
        if graph_distance is not None:
            return float(graph_distance)
        embedding_distance = _embedding_distance(left_position.embedding, right_position.embedding)
        if left_position.region and right_position.region and left_position.region == right_position.region:
            return 2.0 + embedding_distance * 0.01
        return 10.0 + embedding_distance

    def nearest_step_toward(self, start: CellPosition, target: CellPosition) -> CellPosition:
        if not start.node_id or not target.node_id or start.node_id == target.node_id:
            return start
        if start.node_id not in self.nodes or target.node_id not in self.nodes:
            return start
        candidates = self.neighbors(start.node_id)
        if not candidates:
            return start
        best = min(candidates, key=lambda node_id: (self.distance(node_id, target), node_id))
        return self.node(best).position()

    def _graph_distance(self, left: str, right: str) -> int | None:
        if not left or not right or left not in self.nodes or right not in self.nodes:
            return None
        queue: deque[tuple[str, int]] = deque([(left, 0)])
        seen = {left}
        while queue:
            node_id, distance = queue.popleft()
            if node_id == right:
                return distance
            for neighbor in self.neighbors(node_id):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append((neighbor, distance + 1))
        return None


def _position(value: CellPosition | str, topology: TissueTopology) -> CellPosition:
    if isinstance(value, CellPosition):
        if value.node_id in topology.nodes:
            node = topology.nodes[value.node_id]
            return CellPosition(
                node_id=value.node_id,
                region=value.region or node.region,
                neighbors=list(value.neighbors or node.neighbors),
                embedding=value.embedding if value.embedding != (0.0, 0.0, 0.0) else node.embedding,
            )
        return value
    if value in topology.nodes:
        return topology.nodes[value].position()
    return CellPosition(str(value))


def _embedding_distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return hypot(hypot(left[0] - right[0], left[1] - right[1]), left[2] - right[2])

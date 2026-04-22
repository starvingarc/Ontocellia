from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ontocellia.config import OntocelliaConfig


@dataclass(slots=True)
class SpatialSubstrate:
    config: OntocelliaConfig

    def clamp(self, pos: np.ndarray) -> np.ndarray:
        x = float(np.clip(pos[0], 0, self.config.width - 1))
        y = float(np.clip(pos[1], 0, self.config.height - 1))
        return np.array([x, y], dtype=float)

    def nearby_position(self, rng: np.random.Generator, pos: np.ndarray, scale: float = 1.5) -> np.ndarray:
        offset = rng.normal(0.0, scale, size=2)
        return self.clamp(pos + offset)

    def crowding_map(self, cells: dict[int, object]) -> np.ndarray:
        crowding = np.zeros((self.config.height, self.config.width), dtype=float)
        if not cells:
            return crowding
        yy, xx = np.indices((self.config.height, self.config.width))
        radius_sq = max(1.0, self.config.spatial_radius**2)
        for cell in cells.values():
            dx = xx - cell.pos[0]
            dy = yy - cell.pos[1]
            crowding += np.exp(-(dx * dx + dy * dy) / (2 * radius_sq))
        peak = crowding.max()
        if peak > 0:
            crowding /= peak
        return np.clip(crowding, 0.0, 1.0)

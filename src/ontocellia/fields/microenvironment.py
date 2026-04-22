from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ontocellia.config import FIELD_NAMES, OntocelliaConfig


@dataclass(slots=True)
class Microenvironment:
    config: OntocelliaConfig
    initial_fields: dict[str, np.ndarray] | None = None
    field_params: dict[str, dict[str, float]] | None = None
    fields: dict[str, np.ndarray] = field(init=False)
    pending_sources: dict[str, np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        shape = (self.config.height, self.config.width)
        if self.initial_fields is not None:
            self.fields = {name: np.array(value, dtype=float) for name, value in self.initial_fields.items()}
        else:
            self.fields = {name: np.zeros(shape, dtype=float) for name in FIELD_NAMES}
            self.fields["nutrient"] += 0.65
            x_coords = np.linspace(-1, 1, self.config.width)
            y_coords = np.linspace(-1, 1, self.config.height)
            xx, yy = np.meshgrid(x_coords, y_coords)
            self.fields["morphogen_a"] = np.clip(0.6 - np.abs(xx), 0.0, 1.0)
            self.fields["morphogen_b"] = np.clip(0.6 - np.abs(yy), 0.0, 1.0)
            self.fields["task_pressure"] = np.clip(0.12 + 0.18 * (xx + 1) * 0.5, 0.0, 1.0)
            self.fields["damage"] = np.zeros(shape, dtype=float)
            self.fields["crowding"] = np.zeros(shape, dtype=float)
        self.pending_sources = {name: np.zeros_like(field) for name, field in self.fields.items()}
        if self.field_params is None:
            self.field_params = {
                name: {"diffusion": self.config.diffusion_rate, "decay": self.config.damage_repair_decay if name == "damage" else self.config.decay_rate}
                for name in self.fields
            }
        else:
            self.field_params = {
                name: {
                    "diffusion": float(params.get("diffusion", self.config.diffusion_rate)),
                    "decay": float(params.get("decay", self.config.decay_rate)),
                }
                for name, params in self.field_params.items()
            }
            for name in self.fields:
                self.field_params.setdefault(
                    name,
                    {"diffusion": self.config.diffusion_rate, "decay": self.config.damage_repair_decay if name == "damage" else self.config.decay_rate},
                )

    def diffuse(self) -> None:
        if not self.config.enable_spatial:
            for name, source in self.pending_sources.items():
                if np.any(source):
                    self.fields[name][:] = np.clip(self.fields[name].mean() + source.mean(), 0.0, 1.0)
                    source.fill(0.0)
            if "crowding" in self.fields:
                self.fields["crowding"].fill(self.fields["crowding"].mean())
            return

        for name, field in self.fields.items():
            with_sources = np.clip(field + self.pending_sources[name], 0.0, 1.0)
            laplace = (
                np.roll(with_sources, 1, axis=0)
                + np.roll(with_sources, -1, axis=0)
                + np.roll(with_sources, 1, axis=1)
                + np.roll(with_sources, -1, axis=1)
                - 4 * with_sources
            )
            params = self.field_params.get(name, {})
            diffusion = float(params.get("diffusion", self.config.diffusion_rate))
            decay = float(params.get("decay", self.config.decay_rate))
            updated = with_sources + diffusion * laplace
            self.fields[name] = np.clip(updated * (1.0 - decay), 0.0, 1.0)
            self.pending_sources[name].fill(0.0)

    def set_crowding(self, crowding: np.ndarray) -> None:
        if "crowding" not in self.fields:
            self.fields["crowding"] = np.zeros((self.config.height, self.config.width), dtype=float)
            self.pending_sources["crowding"] = np.zeros_like(self.fields["crowding"])
            self.field_params["crowding"] = {"diffusion": self.config.diffusion_rate, "decay": self.config.decay_rate}
        self.fields["crowding"] = np.clip(crowding, 0.0, 1.0)

    def emit(self, pos: np.ndarray, emissions: dict[str, float]) -> None:
        x, y = self._index(pos)
        for name, value in emissions.items():
            if name not in self.pending_sources:
                self.pending_sources[name] = np.zeros((self.config.height, self.config.width), dtype=float)
                self.fields[name] = np.zeros((self.config.height, self.config.width), dtype=float)
                self.field_params[name] = {"diffusion": self.config.diffusion_rate, "decay": self.config.decay_rate}
            if value:
                self.pending_sources[name][y, x] += value

    def sample(self, pos: np.ndarray) -> dict[str, float]:
        x, y = self._index(pos)
        return {name: float(field[y, x]) for name, field in self.fields.items()}

    def gradients(self, pos: np.ndarray) -> dict[str, np.ndarray]:
        x, y = self._index(pos)
        grads: dict[str, np.ndarray] = {}
        for name, field in self.fields.items():
            left = field[y, max(0, x - 1)]
            right = field[y, min(self.config.width - 1, x + 1)]
            up = field[max(0, y - 1), x]
            down = field[min(self.config.height - 1, y + 1), x]
            grads[name] = np.array([(right - left) * 0.5, (down - up) * 0.5], dtype=float)
        return grads

    def inject_damage(self, center: tuple[float, float], radius: float, intensity: float) -> None:
        self._inject_radial("damage", center, radius, intensity)
        if "task_pressure" in self.fields:
            self._inject_radial("task_pressure", center, radius, intensity * 0.45)
        if "nutrient" in self.fields:
            cx, cy = center
            yy, xx = np.indices((self.config.height, self.config.width))
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            mask = np.clip(1 - dist / max(radius, 1e-6), 0.0, 1.0)
            self.fields["nutrient"] = np.clip(self.fields["nutrient"] - mask * intensity * 0.55, 0.0, 1.0)

    def inject_resource_pulse(self, center: tuple[float, float], radius: float, intensity: float) -> None:
        self._inject_radial("nutrient", center, radius, intensity)

    def inject_goal_pressure(self, center: tuple[float, float], radius: float, intensity: float) -> None:
        self._inject_radial("task_pressure", center, radius, intensity)

    def apply_event(self, action: str, center: tuple[float, float], radius: float, intensity: float, field: str | None = None) -> None:
        if action == "damage":
            self.inject_damage(center, radius, intensity)
        elif action == "resource":
            self.inject_resource_pulse(center, radius, intensity)
        elif action == "goal":
            self.inject_goal_pressure(center, radius, intensity)
        elif action == "field" and field is not None:
            self._inject_radial(field, center, radius, intensity)
        else:
            raise ValueError(f"Unsupported environment event action: {action}")

    def _inject_radial(self, name: str, center: tuple[float, float], radius: float, intensity: float) -> None:
        if name not in self.fields:
            self.fields[name] = np.zeros((self.config.height, self.config.width), dtype=float)
            self.pending_sources[name] = np.zeros_like(self.fields[name])
            self.field_params[name] = {"diffusion": self.config.diffusion_rate, "decay": self.config.decay_rate}
        cx, cy = center
        yy, xx = np.indices((self.config.height, self.config.width))
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask = np.clip(1 - dist / max(radius, 1e-6), 0.0, 1.0)
        self.fields[name] = np.clip(self.fields[name] + mask * intensity, 0.0, 1.0)

    def _index(self, pos: np.ndarray) -> tuple[int, int]:
        x = int(np.clip(round(float(pos[0])), 0, self.config.width - 1))
        y = int(np.clip(round(float(pos[1])), 0, self.config.height - 1))
        return x, y

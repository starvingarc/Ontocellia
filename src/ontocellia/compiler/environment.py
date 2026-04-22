from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ontocellia.architecture import EnvironmentModel
from ontocellia.specs.schema import EnvironmentSpec, EventSpec, ZoneSpec


@dataclass(slots=True)
class BuiltEnvironment:
    field_names: list[str]
    diffusive_field_names: list[str]
    background_field_names: list[str]
    initial_fields: dict[str, np.ndarray]
    field_params: dict[str, dict[str, float]]
    contact_context: dict[str, float]
    sources: list[ZoneSpec]
    events: list[EventSpec]
    global_task: dict[str, object]
    evaluation: dict[str, float]
    environment_model: EnvironmentModel | None = None


class TaskCompiler:
    def apply(self, spec: EnvironmentSpec, initial_fields: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[EventSpec]]:
        events = list(spec.events)
        translation = spec.task_translation

        for field_name, value in translation.field_biases.items():
            if field_name in initial_fields:
                initial_fields[field_name] = np.clip(initial_fields[field_name] + value, 0.0, 1.0)

        for zone in translation.goal_zones:
            self._apply_zone(initial_fields, zone.field or "task_pressure", zone)
        for zone in translation.risk_zones:
            target_field = zone.field or "damage"
            self._apply_zone(initial_fields, target_field, zone)
            if "task_pressure" in initial_fields:
                self._apply_zone(
                    initial_fields,
                    "task_pressure",
                    ZoneSpec(center=zone.center, radius=zone.radius, intensity=zone.intensity * 0.4),
                )
        for zone in translation.resource_zones:
            self._apply_zone(initial_fields, zone.field or "nutrient", zone)

        events.extend(translation.periodic_events)
        return initial_fields, events

    def _apply_zone(self, fields: dict[str, np.ndarray], field_name: str, zone: ZoneSpec) -> None:
        if field_name not in fields:
            return
        yy, xx = np.indices(fields[field_name].shape)
        dist = np.sqrt((xx - zone.center[0]) ** 2 + (yy - zone.center[1]) ** 2)
        mask = np.clip(1 - dist / max(zone.radius, 1e-6), 0.0, 1.0)
        if zone.mode == "set":
            fields[field_name] = np.clip(fields[field_name] * (1 - mask) + mask * zone.intensity, 0.0, 1.0)
        else:
            fields[field_name] = np.clip(fields[field_name] + mask * zone.intensity, 0.0, 1.0)


class EnvironmentBuilder:
    def __init__(self) -> None:
        self.task_compiler = TaskCompiler()

    def build(self, spec: EnvironmentSpec) -> BuiltEnvironment:
        shape = (spec.grid.height, spec.grid.width)
        initial_fields: dict[str, np.ndarray] = {}
        field_params: dict[str, dict[str, float]] = {}

        for field_name, field_spec in {**spec.diffusive_fields, **spec.background_context}.items():
            initial_fields[field_name] = self._build_initial(shape, field_spec.initial)
            field_params[field_name] = {"diffusion": field_spec.diffusion, "decay": field_spec.decay}

        for source in spec.sources:
            self.task_compiler._apply_zone(initial_fields, source.field or "task_pressure", source)
        initial_fields, events = self.task_compiler.apply(spec, initial_fields)
        global_task = {
            "text": spec.global_environment.task_goal.text,
            "objective": spec.global_environment.task_goal.objective,
            "success_signals": spec.global_environment.task_goal.success_signals,
        }
        built = BuiltEnvironment(
            field_names=list(initial_fields),
            diffusive_field_names=list(spec.diffusive_fields),
            background_field_names=list(spec.background_context),
            initial_fields=initial_fields,
            field_params=field_params,
            contact_context=dict(spec.contact_context),
            sources=spec.sources,
            events=events,
            global_task=global_task,
            evaluation=dict(spec.global_environment.evaluation.metrics),
        )
        built.environment_model = EnvironmentModel.from_built_environment(spec, built)
        return built

    def _build_initial(self, shape: tuple[int, int], initial: float | dict[str, object]) -> np.ndarray:
        if isinstance(initial, (int, float)):
            return np.full(shape, float(initial), dtype=float)

        pattern = str(initial.get("pattern", "constant"))
        if pattern == "constant":
            return np.full(shape, float(initial.get("value", 0.0)), dtype=float)
        if pattern == "x_gradient":
            start = float(initial.get("start", 0.0))
            end = float(initial.get("end", 1.0))
            gradient = np.linspace(start, end, shape[1], dtype=float)
            return np.repeat(gradient[None, :], shape[0], axis=0)
        if pattern == "y_gradient":
            start = float(initial.get("start", 0.0))
            end = float(initial.get("end", 1.0))
            gradient = np.linspace(start, end, shape[0], dtype=float)
            return np.repeat(gradient[:, None], shape[1], axis=1)
        if pattern == "radial":
            center = initial.get("center", [shape[1] / 2, shape[0] / 2])
            if not isinstance(center, list) or len(center) != 2:
                raise ValueError("radial initial.center must be a 2-item list")
            yy, xx = np.indices(shape)
            radius = float(initial.get("radius", min(shape) / 3))
            intensity = float(initial.get("intensity", 1.0))
            dist = np.sqrt((xx - float(center[0])) ** 2 + (yy - float(center[1])) ** 2)
            return np.clip((1 - dist / max(radius, 1e-6)) * intensity, 0.0, 1.0)
        raise ValueError(f"Unsupported field initial pattern: {pattern}")

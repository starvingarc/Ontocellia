from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ontocellia.specs import EnvironmentSpec

if TYPE_CHECKING:
    from ontocellia.compiler.environment import BuiltEnvironment


@dataclass(slots=True)
class GlobalEnvironmentModel:
    task_goal: str
    objective: str
    resource_budget: dict[str, float] = field(default_factory=dict)
    time_limit: int | None = None
    tools: list[str] = field(default_factory=list)
    evaluation: dict[str, float] = field(default_factory=dict)
    constraints: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class LocalMicroenvironment:
    diffusive_fields: dict[str, float]
    neighbor_messages: dict[str, float]
    mechanical_resource_context: dict[str, float]
    local_risk: float


@dataclass(slots=True)
class EnvironmentModel:
    global_environment: GlobalEnvironmentModel
    diffusive_field_names: list[str]
    background_field_names: list[str]
    contact_context: dict[str, float]
    initial_fields: dict[str, np.ndarray]
    field_params: dict[str, dict[str, float]]
    sources: list[object]
    events: list[object]

    @classmethod
    def from_built_environment(cls, spec: EnvironmentSpec, built: BuiltEnvironment) -> "EnvironmentModel":
        global_model = GlobalEnvironmentModel(
            task_goal=spec.global_environment.task_goal.text,
            objective=spec.global_environment.task_goal.objective,
            resource_budget=dict(spec.global_environment.resource_budget),
            time_limit=spec.global_environment.time_limit,
            tools=list(spec.global_environment.tools),
            evaluation=dict(spec.global_environment.evaluation.metrics),
            constraints=dict(spec.global_environment.constraints),
        )
        return cls(
            global_environment=global_model,
            diffusive_field_names=list(built.diffusive_field_names),
            background_field_names=list(built.background_field_names),
            contact_context=dict(built.contact_context),
            initial_fields={name: field.copy() for name, field in built.initial_fields.items()},
            field_params={name: dict(params) for name, params in built.field_params.items()},
            sources=list(built.sources),
            events=list(built.events),
        )

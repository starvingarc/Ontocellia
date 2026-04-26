from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ontocellia.config import SPEC_MODE

if TYPE_CHECKING:
    from ontocellia.scheduler.runtime import OntocelliaRuntime


@dataclass(slots=True)
class StepPipeline:
    """Deterministic runtime step orchestration.

    The first v0.2 pipeline keeps the mature legacy/spec transition logic on
    the runtime while giving new experiment tooling a stable orchestration
    boundary to depend on.
    """

    def step(self, runtime: OntocelliaRuntime, steps: int = 1) -> None:
        for _ in range(steps):
            if not runtime.cells:
                runtime.tick_count += 1
                runtime.metrics.record(runtime)
                continue
            self.step_once(runtime)

    def step_once(self, runtime: OntocelliaRuntime) -> None:
        if runtime.mode == SPEC_MODE:
            runtime._step_once_spec()
        else:
            runtime._step_once_legacy()

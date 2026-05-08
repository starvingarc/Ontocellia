from __future__ import annotations

from pathlib import Path
from typing import Any

from ontocellia.framework.official_agents import OfficialAgentRunConfig, run_official_terminal_agent

try:  # pragma: no cover - exercised only when terminal-bench is installed.
    from terminal_bench.agents import AgentResult as _TerminalBenchAgentResult
    from terminal_bench.agents import BaseAgent as _TerminalBenchBaseAgent
except Exception:  # pragma: no cover - fallback is covered in unit tests.
    _TerminalBenchBaseAgent = object
    _TerminalBenchAgentResult = None


class _FallbackAgentResult:
    def __init__(self, *, failure_mode: str | None = None, total_input_tokens: int = 0, total_output_tokens: int = 0):
        self.failure_mode = failure_mode
        self.total_input_tokens = total_input_tokens
        self.total_output_tokens = total_output_tokens


class OntocelliaTerminalAgent(_TerminalBenchBaseAgent):  # type: ignore[misc]
    def __init__(self, model_name: str | None = None, max_steps: int = 8, use_mock: bool = True, *args: Any, **kwargs: Any):
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            super().__init__()
        self.model_name = model_name
        self.max_steps = max_steps
        self.use_mock = use_mock

    @staticmethod
    def name() -> str:
        return "Ontocellia"

    def perform_task(self, task_description: str, session: Any, logging_dir: Path | None = None) -> Any:
        result = run_official_terminal_agent(
            task_description,
            session,
            logging_dir=logging_dir,
            config=OfficialAgentRunConfig(
                model_profile=self.model_name,
                max_steps=self.max_steps,
                use_mock=self.use_mock,
            ),
        )
        return _agent_result(result.failure_mode)


def _agent_result(failure_mode: str | None) -> Any:
    if _TerminalBenchAgentResult is None:
        return _FallbackAgentResult(failure_mode=failure_mode)
    for kwargs in (
        {"failure_mode": failure_mode, "total_input_tokens": 0, "total_output_tokens": 0},
        {"failure_mode": failure_mode, "input_tokens": 0, "output_tokens": 0},
        {"failure_mode": failure_mode},
        {},
    ):
        try:
            return _TerminalBenchAgentResult(**kwargs)
        except TypeError:
            continue
    return _FallbackAgentResult(failure_mode=failure_mode)

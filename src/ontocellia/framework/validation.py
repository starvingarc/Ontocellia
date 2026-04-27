from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ontocellia.framework.selection import OrganValidationResult


@dataclass(slots=True)
class ValidationHookRequest:
    name: str
    command: str
    source_gene_id: str | None = None
    source_action_id: str | None = None
    source_cell_id: int | None = None
    cwd: Path | str | None = None
    timeout: float | None = None


@dataclass(slots=True)
class ValidationHookPolicy:
    allowed_commands: list[str] = field(default_factory=list)
    timeout_seconds: float = 60.0
    max_output_chars: int = 12000

    def is_allowed(self, command: str) -> bool:
        return command in set(self.allowed_commands)


class ValidationHookRunner:
    def run(
        self,
        requests: list[ValidationHookRequest],
        policy: ValidationHookPolicy,
        trace: Any | None = None,
    ) -> list[OrganValidationResult]:
        return [self._run_one(request, policy, trace) for request in requests]

    def _run_one(
        self,
        request: ValidationHookRequest,
        policy: ValidationHookPolicy,
        trace: Any | None,
    ) -> OrganValidationResult:
        if not policy.is_allowed(request.command):
            result = OrganValidationResult(
                name=request.name,
                passed=False,
                score=0.0,
                target=request.command,
                evidence=f"Validation hook not allowlisted: {request.command}",
                cost=0.0,
                risk=0.2,
                latency=0.0,
            )
            _record(trace, "validation_hook_skipped", request=_request_dict(request), result=result.as_dict())
            return result

        started = time.monotonic()
        _record(trace, "validation_hook_started", request=_request_dict(request))
        try:
            completed = subprocess.run(
                shlex.split(request.command),
                cwd=str(request.cwd) if request.cwd is not None else None,
                capture_output=True,
                check=False,
                text=True,
                timeout=request.timeout or policy.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            latency = time.monotonic() - started
            result = OrganValidationResult(
                name=request.name,
                passed=False,
                score=0.0,
                target=request.command,
                evidence=_truncate(f"Validation hook timed out after {latency:.2f}s.\n{_timeout_output(error)}", policy.max_output_chars),
                cost=latency,
                risk=0.6,
                latency=latency,
            )
            _record(trace, "validation_hook_completed", request=_request_dict(request), result=result.as_dict())
            return result
        except (OSError, ValueError) as error:
            latency = time.monotonic() - started
            result = OrganValidationResult(
                name=request.name,
                passed=False,
                score=0.0,
                target=request.command,
                evidence=_truncate(f"Validation hook failed to start: {error}", policy.max_output_chars),
                cost=latency,
                risk=0.5,
                latency=latency,
            )
            _record(trace, "validation_hook_completed", request=_request_dict(request), result=result.as_dict())
            return result

        latency = time.monotonic() - started
        passed = completed.returncode == 0
        output = _combined_output(completed.stdout, completed.stderr)
        status = "passed" if passed else f"failed with exit code {completed.returncode}"
        result = OrganValidationResult(
            name=request.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            target=request.command,
            evidence=_truncate(f"Validation hook {status}.\n{output}", policy.max_output_chars),
            cost=latency,
            risk=0.0 if passed else 0.5,
            latency=latency,
        )
        _record(trace, "validation_hook_completed", request=_request_dict(request), result=result.as_dict())
        return result


def _combined_output(stdout: str, stderr: str) -> str:
    parts = []
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    return "\n".join(parts).strip()


def _timeout_output(error: subprocess.TimeoutExpired) -> str:
    stdout = error.stdout.decode("utf-8", errors="replace") if isinstance(error.stdout, bytes) else error.stdout or ""
    stderr = error.stderr.decode("utf-8", errors="replace") if isinstance(error.stderr, bytes) else error.stderr or ""
    return _combined_output(stdout, stderr)


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 24)] + "\n[output truncated]"


def _record(trace: Any | None, event_type: str, **payload: Any) -> None:
    if trace is not None:
        trace.record(event_type, **payload)


def _request_dict(request: ValidationHookRequest) -> dict[str, Any]:
    return {
        "name": request.name,
        "command": request.command,
        "source_gene_id": request.source_gene_id,
        "source_action_id": request.source_action_id,
        "source_cell_id": request.source_cell_id,
        "cwd": str(request.cwd) if request.cwd is not None else None,
        "timeout": request.timeout,
    }

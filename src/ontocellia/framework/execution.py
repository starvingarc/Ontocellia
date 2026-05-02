from __future__ import annotations

import fnmatch
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ontocellia.framework.communication import MatrixRecord
from ontocellia.framework.selection import OrganValidationResult


@dataclass(slots=True)
class ExecutionRequest:
    id: str
    cell_id: int | None
    intent_type: str
    target: str
    interface: str
    command: str | None = None
    patch: str | None = None
    path: str | None = None
    query: str | None = None
    dry_run: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExecutionPolicy:
    workspace_root: Path | str = Path(".")
    allowed_interfaces: list[str] = field(default_factory=list)
    allowed_commands: list[str] = field(default_factory=list)
    allowed_write_globs: list[str] = field(default_factory=list)
    timeout_seconds: float = 60.0
    max_output_chars: int = 12000
    dry_run: bool = True

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root).resolve()

    def allows_interface(self, interface: str) -> bool:
        return interface in set(self.allowed_interfaces)

    def allows_command(self, command: str) -> bool:
        return command in set(self.allowed_commands)

    def allows_write(self, relative_path: str) -> bool:
        normalized = relative_path.replace("\\", "/")
        return any(fnmatch.fnmatch(normalized, pattern) or _globstar_single_level(normalized, pattern) for pattern in self.allowed_write_globs)


@dataclass(slots=True)
class ExecutionResult:
    request: ExecutionRequest
    status: str
    passed: bool
    score: float
    evidence: str
    stdout: str = ""
    stderr: str = ""
    changed_files: list[str] = field(default_factory=list)
    risk: float = 0.0
    cost: float = 0.0
    latency: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["request"] = self.request.as_dict()
        return data

    def to_validation_result(self) -> OrganValidationResult:
        return OrganValidationResult(
            name=f"execution:{self.request.interface}",
            passed=self.passed,
            score=self.score,
            target=self.request.target,
            evidence=self.evidence,
            cost=self.cost,
            risk=self.risk,
            latency=self.latency,
        )


class ExecutionRuntime:
    def execute(self, tissue: Any, actions: list[dict[str, Any]], policy: ExecutionPolicy) -> list[ExecutionResult]:
        results: list[ExecutionResult] = []
        for index, action in enumerate(actions):
            request = _request_from_action(index, action, policy)
            result = self._execute_one(tissue, request, policy)
            results.append(result)
            _deposit_result(tissue, result)
        return results

    def _execute_one(self, tissue: Any, request: ExecutionRequest, policy: ExecutionPolicy) -> ExecutionResult:
        if not policy.allows_interface(request.interface):
            result = _result(request, "skipped", False, 0.0, f"Execution interface not allowlisted: {request.interface}", risk=0.2)
            _record(tissue, "execution_skipped", result)
            return result
        if request.interface in {"pytest.run", "shell.run"} and request.command and not policy.allows_command(request.command):
            result = _result(request, "skipped", False, 0.0, f"Execution command not allowlisted: {request.command}", risk=0.3)
            _record(tissue, "execution_skipped", result)
            return result
        if request.interface == "workspace.apply_patch" and policy.dry_run:
            result = _result(request, "dry_run", False, 0.0, "Dry-run: patch was not applied.", risk=0.1)
            _record(tissue, "execution_skipped", result)
            return result

        _record_request(tissue, "execution_started", request)
        started = time.monotonic()
        if request.interface == "workspace.read":
            result = _read_file(request, policy)
        elif request.interface == "workspace.search":
            result = _search_workspace(request, policy)
        elif request.interface == "workspace.apply_patch":
            result = _apply_patch(request, policy)
        elif request.interface == "git.diff":
            result = _run_command(request, policy, request.command or "git diff", require_allowlist=False)
        elif request.interface in {"pytest.run", "shell.run"}:
            result = _run_command(request, policy, request.command or "", require_allowlist=True)
        elif request.intent_type == "record_memory":
            result = _result(request, "passed", True, 1.0, "Memory intent recorded without external execution.")
        else:
            result = _result(request, "skipped", False, 0.0, f"No executor for interface: {request.interface}", risk=0.2)
        result.latency = time.monotonic() - started
        _record(tissue, "execution_completed" if result.status != "skipped" else "execution_skipped", result)
        return result


def _request_from_action(index: int, action: dict[str, Any], policy: ExecutionPolicy) -> ExecutionRequest:
    payload = dict(action.get("payload", {}))
    intent_type = str(action.get("intent_type") or action.get("gene_id") or "propose_action")
    target = str(action.get("target") or payload.get("path") or payload.get("query") or ".")
    command = payload.get("command")
    interface = _infer_interface(intent_type, payload, command)
    return ExecutionRequest(
        id=str(action.get("id") or f"action-{index}"),
        cell_id=int(action["cell_id"]) if "cell_id" in action else None,
        intent_type=intent_type,
        target=target,
        interface=interface,
        command=str(command) if command else None,
        patch=str(payload.get("patch")) if payload.get("patch") is not None else None,
        path=str(payload.get("path") or target) if payload.get("path") or intent_type == "inspect_context" else None,
        query=str(payload.get("query") or target) if payload.get("query") or intent_type == "inspect_context" else None,
        dry_run=policy.dry_run,
    )


def _infer_interface(intent_type: str, payload: dict[str, Any], command: Any) -> str:
    if intent_type == "record_memory":
        return "matrix.record"
    if intent_type == "inspect_context":
        return "workspace.read" if payload.get("path") else "workspace.search"
    if intent_type == "propose_patch":
        return "workspace.apply_patch"
    command_text = str(command or "")
    if command_text.startswith("git diff") or intent_type == "review_output" and not command_text:
        return "git.diff"
    if "pytest" in shlex.split(command_text) if command_text else False:
        return "pytest.run"
    return "shell.run"


def _read_file(request: ExecutionRequest, policy: ExecutionPolicy) -> ExecutionResult:
    path_value = request.path or request.target
    try:
        path = _resolve_inside(policy.workspace_root, path_value)
    except ValueError as error:
        return _result(request, "failed", False, 0.0, str(error), risk=0.4)
    if not path.is_file():
        return _result(request, "failed", False, 0.0, f"Workspace file not found: {path_value}", risk=0.2)
    content = _truncate(path.read_text(encoding="utf-8", errors="replace"), policy.max_output_chars)
    return _result(request, "passed", True, 1.0, content)


def _search_workspace(request: ExecutionRequest, policy: ExecutionPolicy) -> ExecutionResult:
    query = request.query or request.target
    command = ["rg", "-n", "--", query, str(policy.workspace_root)]
    try:
        completed = subprocess.run(command, capture_output=True, check=False, text=True, timeout=policy.timeout_seconds)
    except FileNotFoundError:
        command = ["grep", "-R", "-n", "--", query, str(policy.workspace_root)]
        completed = subprocess.run(command, capture_output=True, check=False, text=True, timeout=policy.timeout_seconds)
    except subprocess.TimeoutExpired as error:
        return _result(request, "failed", False, 0.0, _truncate(f"Search timed out.\n{_timeout_output(error)}", policy.max_output_chars), risk=0.3)
    output = _combined_output(completed.stdout, completed.stderr)
    passed = completed.returncode == 0
    return _result(request, "passed" if passed else "failed", passed, 1.0 if passed else 0.0, _truncate(output or "No matches.", policy.max_output_chars), stdout=completed.stdout, stderr=completed.stderr, risk=0.0 if passed else 0.1)


def _apply_patch(request: ExecutionRequest, policy: ExecutionPolicy) -> ExecutionResult:
    if not request.patch:
        return _result(request, "skipped", False, 0.0, "Patch payload missing.", risk=0.2)
    changed_files = _patch_paths(request.patch)
    blocked = [path for path in changed_files if not policy.allows_write(path)]
    if blocked:
        return _result(request, "skipped", False, 0.0, f"Patch paths not allowlisted: {', '.join(blocked)}", changed_files=changed_files, risk=0.5)
    try:
        _apply_simple_unified_diff(policy.workspace_root, request.patch)
    except (OSError, ValueError) as error:
        return _result(request, "failed", False, 0.0, f"Patch failed: {error}", changed_files=changed_files, risk=0.5)
    return _result(request, "passed", True, 1.0, f"Patch applied to: {', '.join(changed_files)}", changed_files=changed_files)


def _run_command(request: ExecutionRequest, policy: ExecutionPolicy, command: str, *, require_allowlist: bool) -> ExecutionResult:
    if not command:
        return _result(request, "skipped", False, 0.0, "Command payload missing.", risk=0.2)
    if require_allowlist and not policy.allows_command(command):
        return _result(request, "skipped", False, 0.0, f"Execution command not allowlisted: {command}", risk=0.3)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            shlex.split(command),
            cwd=policy.workspace_root,
            capture_output=True,
            check=False,
            text=True,
            timeout=policy.timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        latency = time.monotonic() - started
        return _result(request, "failed", False, 0.0, _truncate(f"Command timed out after {latency:.2f}s.\n{_timeout_output(error)}", policy.max_output_chars), risk=0.5, latency=latency)
    except (OSError, ValueError) as error:
        latency = time.monotonic() - started
        return _result(request, "failed", False, 0.0, f"Command failed to start: {error}", risk=0.5, latency=latency)
    latency = time.monotonic() - started
    passed = completed.returncode == 0
    output = _truncate(_combined_output(completed.stdout, completed.stderr), policy.max_output_chars)
    status = "passed" if passed else "failed"
    return _result(
        request,
        status,
        passed,
        1.0 if passed else 0.0,
        output or f"Command {status} with exit code {completed.returncode}.",
        stdout=_truncate(completed.stdout, policy.max_output_chars),
        stderr=_truncate(completed.stderr, policy.max_output_chars),
        risk=0.0 if passed else 0.5,
        latency=latency,
    )


def _deposit_result(tissue: Any, result: ExecutionResult) -> None:
    if not hasattr(tissue.environment, "matrix"):
        return
    cell = tissue.cells.get(result.request.cell_id) if result.request.cell_id is not None else None
    position = cell.position if cell is not None else None
    if position is None:
        position = getattr(tissue.environment.niches[0], "position", None) if getattr(tissue.environment, "niches", []) else ""
    record = MatrixRecord(
        id=f"execution-{len(tissue.environment.matrix.records) + 1}",
        source_cell_id=result.request.cell_id or 0,
        kind="execution",
        content=result.evidence,
        tags=["execution", result.request.interface, result.status],
        position=position,
        confidence=result.score,
        created_tick=int(getattr(tissue, "tick_count", 0)),
        fate=getattr(cell, "fate", None),
    )
    tissue.environment.matrix.deposit(record)
    tissue.trace.record("matrix_deposit", **record.as_dict())


def _resolve_inside(root: Path, value: str) -> Path:
    candidate = Path(value)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Path outside workspace: {value}")
    return path


def _patch_paths(patch_text: str) -> list[str]:
    paths: list[str] = []
    for line in patch_text.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path == "/dev/null":
                continue
            if path.startswith("b/"):
                path = path[2:]
            paths.append(path)
    return paths


def _apply_simple_unified_diff(root: Path, patch_text: str) -> None:
    current_path: str | None = None
    old_lines: list[str] = []
    new_lines: list[str] = []
    for line in patch_text.splitlines():
        if line.startswith("+++ "):
            current_path = line[4:].strip()
            if current_path.startswith("b/"):
                current_path = current_path[2:]
            continue
        if current_path is None or line.startswith("--- ") or line.startswith("@@"):
            continue
        if line.startswith("-"):
            old_lines.append(line[1:] + "\n")
        elif line.startswith("+"):
            new_lines.append(line[1:] + "\n")
        elif line.startswith(" "):
            old_lines.append(line[1:] + "\n")
            new_lines.append(line[1:] + "\n")
    if current_path is None:
        raise ValueError("patch target missing")
    path = _resolve_inside(root, current_path)
    content = path.read_text(encoding="utf-8")
    old = "".join(old_lines)
    new = "".join(new_lines)
    if old not in content:
        raise ValueError(f"patch context not found in {current_path}")
    path.write_text(content.replace(old, new, 1), encoding="utf-8")


def _globstar_single_level(path: str, pattern: str) -> bool:
    if "/**/" not in pattern:
        return False
    return fnmatch.fnmatch(path, pattern.replace("/**/", "/"))


def _result(
    request: ExecutionRequest,
    status: str,
    passed: bool,
    score: float,
    evidence: str,
    *,
    stdout: str = "",
    stderr: str = "",
    changed_files: list[str] | None = None,
    risk: float = 0.0,
    latency: float = 0.0,
) -> ExecutionResult:
    return ExecutionResult(
        request=request,
        status=status,
        passed=passed,
        score=score,
        evidence=evidence,
        stdout=stdout,
        stderr=stderr,
        changed_files=list(changed_files or []),
        risk=risk,
        latency=latency,
    )


def _record_request(tissue: Any, event_type: str, request: ExecutionRequest) -> None:
    tissue.trace.record(event_type, request=request.as_dict())


def _record(tissue: Any, event_type: str, result: ExecutionResult) -> None:
    tissue.trace.record(event_type, **result.as_dict())


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

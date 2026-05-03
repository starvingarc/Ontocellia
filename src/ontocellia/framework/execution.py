from __future__ import annotations

import fnmatch
import json
import shlex
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ontocellia.framework.communication import MatrixRecord
from ontocellia.framework.selection import OrganValidationResult


@dataclass(slots=True)
class ToolInvocation:
    id: str
    action_id: str
    cell_id: int | None
    intent_type: str
    target: str
    interface: str
    adapter: str
    operation: str
    command: str | None = None
    patch: str | None = None
    path: str | None = None
    query: str | None = None
    url: str | None = None
    method: str = "GET"
    arguments: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    required_interfaces: list[str] = field(default_factory=list)
    context_record_ids: list[str] = field(default_factory=list)
    dry_run: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


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
class ToolPolicy:
    workspace_root: Path | str = Path(".")
    allowed_interfaces: list[str] = field(default_factory=list)
    allowed_commands: list[str] = field(default_factory=list)
    allowed_write_globs: list[str] = field(default_factory=list)
    allowed_network_hosts: list[str] = field(default_factory=list)
    allowed_mcp_tools: list[str] = field(default_factory=list)
    allowed_git_commands: list[str] = field(default_factory=list)
    allowed_http_methods: list[str] = field(default_factory=lambda: ["GET", "POST"])
    enable_http_tools: bool = False
    enable_browser_tools: bool = False
    timeout_seconds: float = 60.0
    max_output_chars: int = 12000
    dry_run: bool = True

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root).resolve()

    def allows_interface(self, interface: str) -> bool:
        return _any_interface_matches(interface, self.allowed_interfaces)

    def allows_command(self, command: str) -> bool:
        return command in set(self.allowed_commands)

    def allows_git_command(self, command: str) -> bool:
        return not self.allowed_git_commands or command in set(self.allowed_git_commands)

    def allows_network_host(self, host: str) -> bool:
        return host in set(self.allowed_network_hosts)

    def allows_mcp_tool(self, interface: str) -> bool:
        return interface in set(self.allowed_mcp_tools)

    def allows_write(self, relative_path: str) -> bool:
        normalized = relative_path.replace("\\", "/")
        return any(fnmatch.fnmatch(normalized, pattern) or _globstar_single_level(normalized, pattern) for pattern in self.allowed_write_globs)


@dataclass(slots=True)
class ExecutionPolicy(ToolPolicy):
    pass


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


@dataclass(slots=True)
class ToolResult:
    invocation: ToolInvocation
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
    matrix_tags: list[str] = field(default_factory=list)

    @property
    def request(self) -> ToolInvocation:
        return self.invocation

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["invocation"] = self.invocation.as_dict()
        return data

    def to_validation_result(self) -> OrganValidationResult:
        return OrganValidationResult(
            name=f"tool:{self.invocation.interface}",
            passed=self.passed,
            score=self.score,
            target=self.invocation.target,
            evidence=self.evidence,
            cost=self.cost,
            risk=self.risk,
            latency=self.latency,
        )

    def to_execution_result(self) -> ExecutionResult:
        return ExecutionResult(
            request=ExecutionRequest(
                id=self.invocation.id,
                cell_id=self.invocation.cell_id,
                intent_type=self.invocation.intent_type,
                target=self.invocation.target,
                interface=self.invocation.interface,
                command=self.invocation.command,
                patch=self.invocation.patch,
                path=self.invocation.path,
                query=self.invocation.query,
                dry_run=self.invocation.dry_run,
            ),
            status=self.status,
            passed=self.passed,
            score=self.score,
            evidence=self.evidence,
            stdout=self.stdout,
            stderr=self.stderr,
            changed_files=list(self.changed_files),
            risk=self.risk,
            cost=self.cost,
            latency=self.latency,
        )


class ExtracellularToolRuntime:
    def plan_invocations(self, actions: list[dict[str, Any]], policy: ToolPolicy) -> list[ToolInvocation]:
        return [_invocation_from_action(index, action, policy) for index, action in enumerate(actions)]

    def execute(self, tissue: Any, actions: list[dict[str, Any]], policy: ToolPolicy) -> list[ToolResult]:
        results: list[ToolResult] = []
        for invocation in self.plan_invocations(actions, policy):
            result = self._execute_invocation(tissue, invocation, policy)
            results.append(result)
            _deposit_tool_result(tissue, result)
            _emit_tool_feedback(tissue, result)
        return results

    def _execute_invocation(self, tissue: Any, invocation: ToolInvocation, policy: ToolPolicy) -> ToolResult:
        gate = _gate_invocation(tissue, invocation, policy)
        if gate is not None:
            result = _tool_result(invocation, "skipped", False, 0.0, gate, risk=0.25)
            _record_tool_result(tissue, "tool_invocation_skipped", result)
            _record_execution_result(tissue, "execution_skipped", result)
            return result
        if invocation.interface == "workspace.apply_patch" and policy.dry_run:
            result = _tool_result(invocation, "dry_run", False, 0.0, "Dry-run: patch was not applied.", risk=0.1)
            _record_tool_result(tissue, "tool_invocation_skipped", result)
            _record_execution_result(tissue, "execution_skipped", result)
            return result

        _record_tool_invocation(tissue, "tool_invocation_started", invocation)
        _record_execution_request(tissue, "execution_started", invocation)
        started = time.monotonic()
        if invocation.adapter == "workspace":
            result = _workspace_tool(invocation, policy)
        elif invocation.adapter == "git":
            result = _git_tool(invocation, policy)
        elif invocation.adapter == "validation":
            result = _command_tool(invocation, policy, invocation.command or "", require_allowlist=True)
        elif invocation.adapter == "shell":
            result = _command_tool(invocation, policy, invocation.command or "", require_allowlist=True)
        elif invocation.adapter == "mcp":
            result = _mcp_tool(tissue, invocation, policy)
        elif invocation.adapter == "http":
            result = _http_tool(invocation, policy)
        elif invocation.adapter == "browser":
            result = _browser_tool(invocation, policy)
        elif invocation.adapter == "matrix":
            result = _tool_result(invocation, "passed", True, 1.0, "Memory intent recorded without external execution.", matrix_tags=["memory"])
        else:
            result = _tool_result(invocation, "skipped", False, 0.0, f"No tool adapter for interface: {invocation.interface}", risk=0.2)
        result.latency = time.monotonic() - started
        _record_tool_result(tissue, "tool_invocation_completed" if result.status != "skipped" else "tool_invocation_skipped", result)
        _record_execution_result(tissue, "execution_completed" if result.status != "skipped" else "execution_skipped", result)
        return result


class ExecutionRuntime:
    def execute(self, tissue: Any, actions: list[dict[str, Any]], policy: ExecutionPolicy) -> list[ExecutionResult]:
        return [result.to_execution_result() for result in ExtracellularToolRuntime().execute(tissue, actions, policy)]

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


def _invocation_from_action(index: int, action: dict[str, Any] | Any, policy: ToolPolicy) -> ToolInvocation:
    data = action.as_dict() if hasattr(action, "as_dict") else dict(action)
    payload = dict(data.get("payload", {}))
    intent_type = str(data.get("intent_type") or data.get("gene_id") or "propose_action")
    target = str(payload.get("target") or data.get("target") or payload.get("path") or payload.get("query") or payload.get("url") or ".")
    command = payload.get("command")
    interface = str(payload.get("interface") or _infer_interface(intent_type, payload, command))
    adapter, operation = _adapter_operation(interface)
    return ToolInvocation(
        id=str(data.get("id") or f"tool-{index}"),
        action_id=str(data.get("id") or f"action-{index}"),
        cell_id=int(data["cell_id"]) if "cell_id" in data else None,
        intent_type=intent_type,
        target=target,
        interface=interface,
        adapter=adapter,
        operation=operation,
        command=str(command) if command else None,
        patch=str(payload.get("patch")) if payload.get("patch") is not None else None,
        path=str(payload.get("path") or target) if payload.get("path") or interface in {"workspace.read", "workspace.list"} else None,
        query=str(payload.get("query") or target) if payload.get("query") or interface == "workspace.search" else None,
        url=str(payload.get("url") or target) if interface in {"http.request", "browser.open"} else None,
        method=str(payload.get("method", "GET")).upper(),
        arguments=dict(payload.get("arguments", {})) if isinstance(payload.get("arguments", {}), dict) else {},
        payload=payload,
        required_interfaces=[str(item) for item in data.get("required_interfaces", [])],
        context_record_ids=[str(item) for item in payload.get("context_record_ids", [])],
        dry_run=policy.dry_run,
    )


def _adapter_operation(interface: str) -> tuple[str, str]:
    if interface.startswith("workspace."):
        return "workspace", interface.split(".", 1)[1]
    if interface.startswith("git."):
        return "git", interface.split(".", 1)[1]
    if interface == "pytest.run":
        return "validation", "pytest"
    if interface == "shell.run":
        return "shell", "run"
    if interface.startswith("mcp:"):
        return "mcp", "tool"
    if interface == "http.request":
        return "http", "request"
    if interface == "browser.open":
        return "browser", "open"
    if interface == "matrix.record":
        return "matrix", "record"
    return "shell", "run"


def _gate_invocation(tissue: Any, invocation: ToolInvocation, policy: ToolPolicy) -> str | None:
    if not policy.allows_interface(invocation.interface):
        return f"Tool interface not allowlisted: {invocation.interface}"
    if invocation.required_interfaces and not _any_interface_matches(invocation.interface, invocation.required_interfaces):
        return f"Tool interface not requested by action: {invocation.interface}"
    cell = tissue.cells.get(invocation.cell_id) if invocation.cell_id is not None and hasattr(tissue, "cells") else None
    if cell is not None and not _cell_accepts_interface(cell, invocation.interface):
        return f"Cell receptor does not accept tool interface: {invocation.interface}"
    if not _environment_accepts_interface(tissue, cell, invocation.interface):
        return f"Tool interface not available in environment: {invocation.interface}"
    if invocation.adapter in {"shell", "validation"} and invocation.command and not policy.allows_command(invocation.command):
        return f"Execution command not allowlisted: {invocation.command}"
    if invocation.adapter == "git" and invocation.command and not policy.allows_git_command(invocation.command):
        return f"Git command not allowlisted: {invocation.command}"
    if invocation.adapter == "mcp" and not policy.allows_mcp_tool(invocation.interface):
        return f"MCP tool not allowlisted: {invocation.interface}"
    if invocation.adapter == "http":
        if not policy.enable_http_tools:
            return "HTTP tools are disabled by policy."
        host = _url_host(invocation.url or invocation.target)
        if not host or not policy.allows_network_host(host):
            return f"Network host not allowlisted: {host or invocation.url or invocation.target}"
        if invocation.method.upper() not in {method.upper() for method in policy.allowed_http_methods}:
            return f"HTTP method not allowlisted: {invocation.method}"
    if invocation.adapter == "browser" and not policy.enable_browser_tools:
        return "Browser tools are disabled by policy."
    return None


def _workspace_tool(invocation: ToolInvocation, policy: ToolPolicy) -> ToolResult:
    request = _execution_request_from_invocation(invocation)
    if invocation.operation == "read":
        return _tool_from_execution(invocation, _read_file(request, policy), ["workspace", "read"])
    if invocation.operation == "search":
        return _tool_from_execution(invocation, _search_workspace(request, policy), ["workspace", "search"])
    if invocation.operation == "list":
        return _list_workspace(invocation, policy)
    if invocation.operation == "apply_patch":
        return _tool_from_execution(invocation, _apply_patch(request, policy), ["workspace", "patch"])
    return _tool_result(invocation, "skipped", False, 0.0, f"Unknown workspace operation: {invocation.operation}", risk=0.2)


def _list_workspace(invocation: ToolInvocation, policy: ToolPolicy) -> ToolResult:
    path_value = invocation.path or invocation.target or "."
    try:
        path = _resolve_inside(policy.workspace_root, path_value)
    except ValueError as error:
        return _tool_result(invocation, "failed", False, 0.0, str(error), risk=0.4, matrix_tags=["workspace", "list"])
    if not path.is_dir():
        return _tool_result(invocation, "failed", False, 0.0, f"Workspace directory not found: {path_value}", risk=0.2, matrix_tags=["workspace", "list"])
    entries = sorted(item.name + ("/" if item.is_dir() else "") for item in path.iterdir())
    return _tool_result(invocation, "passed", True, 1.0, _truncate("\n".join(entries), policy.max_output_chars), matrix_tags=["workspace", "list"])


def _git_tool(invocation: ToolInvocation, policy: ToolPolicy) -> ToolResult:
    command = _git_command(invocation)
    if not policy.allows_git_command(command):
        return _tool_result(invocation, "skipped", False, 0.0, f"Git command not allowlisted: {command}", risk=0.25, matrix_tags=["git", invocation.operation])
    execution = _run_command(_execution_request_from_invocation(invocation, command=command), policy, command, require_allowlist=False)
    return _tool_from_execution(invocation, execution, ["git", invocation.operation])


def _git_command(invocation: ToolInvocation) -> str:
    if invocation.operation == "status":
        return "git status --short"
    if invocation.operation == "show":
        target = str(invocation.payload.get("target") or invocation.target or "HEAD")
        return f"git show {target}"
    if invocation.operation == "log":
        limit = int(invocation.payload.get("limit", 5))
        return f"git log --oneline -n {max(1, min(limit, 50))}"
    path = invocation.path or invocation.payload.get("path")
    return f"git diff -- {path}" if path else "git diff"


def _command_tool(invocation: ToolInvocation, policy: ToolPolicy, command: str, *, require_allowlist: bool) -> ToolResult:
    execution = _run_command(_execution_request_from_invocation(invocation, command=command), policy, command, require_allowlist=require_allowlist)
    tags = ["validation", invocation.operation] if invocation.adapter == "validation" else ["shell", "command"]
    return _tool_from_execution(invocation, execution, tags)


def _mcp_tool(tissue: Any, invocation: ToolInvocation, policy: ToolPolicy) -> ToolResult:
    adapter = getattr(tissue.environment, "mcp_adapter", None)
    if adapter is None:
        return _tool_result(invocation, "skipped", False, 0.0, "No MCP adapter configured.", risk=0.2, matrix_tags=["mcp"])
    tool = _declared_mcp_tool(adapter, invocation.interface)
    if tool is None:
        return _tool_result(invocation, "skipped", False, 0.0, f"MCP tool is not declared: {invocation.interface}", risk=0.3, matrix_tags=["mcp"])
    content = str(tool.metadata.get("mock_result") or tool.description or f"MCP tool invoked: {invocation.interface}")
    tags = [str(tag) for tag in tool.metadata.get("tags", [])] if isinstance(tool.metadata.get("tags", []), list) else []
    return _tool_result(invocation, "passed", True, 1.0, content, matrix_tags=["mcp", tool.name, *tags])


def _declared_mcp_tool(adapter: Any, interface: str) -> Any | None:
    parts = interface.split(":")
    if len(parts) != 4 or parts[0] != "mcp" or parts[2] != "tool":
        return None
    server_id, tool_name = parts[1], parts[3]
    for server in getattr(adapter, "servers", []):
        if server.id != server_id:
            continue
        for tool in server.tools:
            if tool.name == tool_name:
                return tool
    return None


def _http_tool(invocation: ToolInvocation, policy: ToolPolicy) -> ToolResult:
    url = invocation.url or invocation.target
    data = None
    headers = {"Content-Type": "application/json"}
    if invocation.method.upper() == "POST":
        data = json.dumps(invocation.arguments or invocation.payload.get("body", {})).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=invocation.method.upper())
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=policy.timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
            latency = time.monotonic() - started
            return _tool_result(invocation, "passed", True, 1.0, _truncate(body, policy.max_output_chars), latency=latency, matrix_tags=["http", _url_host(url)])
    except (urllib.error.URLError, OSError) as error:
        latency = time.monotonic() - started
        return _tool_result(invocation, "failed", False, 0.0, f"HTTP request failed: {error}", risk=0.4, latency=latency, matrix_tags=["http", _url_host(url)])


def _browser_tool(invocation: ToolInvocation, policy: ToolPolicy) -> ToolResult:
    return _tool_result(invocation, "passed", True, 0.5, f"Browser adapter accepted URL for external browser runtime: {invocation.url or invocation.target}", matrix_tags=["browser"])


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
    if payload.get("interface"):
        return str(payload["interface"])
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


def _execution_request_from_invocation(invocation: ToolInvocation, command: str | None = None) -> ExecutionRequest:
    return ExecutionRequest(
        id=invocation.id,
        cell_id=invocation.cell_id,
        intent_type=invocation.intent_type,
        target=invocation.target,
        interface=invocation.interface,
        command=command or invocation.command,
        patch=invocation.patch,
        path=invocation.path,
        query=invocation.query,
        dry_run=invocation.dry_run,
    )


def _tool_from_execution(invocation: ToolInvocation, result: ExecutionResult, matrix_tags: list[str]) -> ToolResult:
    return ToolResult(
        invocation=invocation,
        status=result.status,
        passed=result.passed,
        score=result.score,
        evidence=result.evidence,
        stdout=result.stdout,
        stderr=result.stderr,
        changed_files=list(result.changed_files),
        risk=result.risk,
        cost=result.cost,
        latency=result.latency,
        matrix_tags=list(matrix_tags),
    )


def _tool_result(
    invocation: ToolInvocation,
    status: str,
    passed: bool,
    score: float,
    evidence: str,
    *,
    stdout: str = "",
    stderr: str = "",
    changed_files: list[str] | None = None,
    risk: float = 0.0,
    cost: float = 0.0,
    latency: float = 0.0,
    matrix_tags: list[str] | None = None,
) -> ToolResult:
    return ToolResult(
        invocation=invocation,
        status=status,
        passed=passed,
        score=score,
        evidence=evidence,
        stdout=stdout,
        stderr=stderr,
        changed_files=list(changed_files or []),
        risk=risk,
        cost=cost,
        latency=latency,
        matrix_tags=list(matrix_tags or []),
    )


def _deposit_tool_result(tissue: Any, result: ToolResult) -> None:
    if not hasattr(tissue.environment, "matrix"):
        return
    cell = tissue.cells.get(result.invocation.cell_id) if result.invocation.cell_id is not None else None
    position = cell.position if cell is not None else None
    if position is None:
        position = getattr(tissue.environment.niches[0], "position", None) if getattr(tissue.environment, "niches", []) else ""
    references = [result.invocation.id, *result.invocation.context_record_ids]
    tags = list(dict.fromkeys(["tool", "execution", result.invocation.adapter, result.invocation.interface, result.status, *result.matrix_tags]))
    record = MatrixRecord(
        id=f"tool-{len(tissue.environment.matrix.records) + 1}",
        source_cell_id=result.invocation.cell_id or 0,
        kind="execution",
        content=result.evidence,
        tags=tags,
        position=position,
        confidence=result.score,
        created_tick=int(getattr(tissue, "tick_count", 0)),
        fate=getattr(cell, "fate", None),
        validation_status="validated" if result.passed else "failed",
        lineage_id=str(getattr(getattr(cell, "lineage", None), "root_id", "")) or None,
        references=references,
        salience=max(0.5, result.score),
    )
    tissue.environment.matrix.deposit(record)
    tissue.trace.record("matrix_deposit", **record.as_dict())


def _emit_tool_feedback(tissue: Any, result: ToolResult) -> None:
    if result.passed and result.risk <= 0.2:
        return
    if hasattr(tissue.environment, "morphogens"):
        tissue.environment.morphogens.emit("risk_pressure", min(1.0, max(0.1, result.risk)))
        tissue.environment.morphogens.emit("validation_pressure", 0.2)
        tissue.environment.morphogens.emit("repair_pressure", 0.2)


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
        validation_status="validated" if result.passed else "failed",
        lineage_id=str(getattr(getattr(cell, "lineage", None), "root_id", "")) or None,
        references=[result.request.id],
        salience=max(0.5, result.score),
    )
    tissue.environment.matrix.deposit(record)
    tissue.trace.record("matrix_deposit", **record.as_dict())


def _record_tool_invocation(tissue: Any, event_type: str, invocation: ToolInvocation) -> None:
    tissue.trace.record(event_type, invocation=invocation.as_dict())


def _record_tool_result(tissue: Any, event_type: str, result: ToolResult) -> None:
    tissue.trace.record(event_type, **result.as_dict())


def _record_execution_request(tissue: Any, event_type: str, invocation: ToolInvocation) -> None:
    tissue.trace.record(event_type, request=_execution_request_from_invocation(invocation).as_dict())


def _record_execution_result(tissue: Any, event_type: str, result: ToolResult) -> None:
    execution_result = result.to_execution_result()
    tissue.trace.record(event_type, **execution_result.as_dict())


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


def _any_interface_matches(interface: str, candidates: list[str]) -> bool:
    return any(_interface_matches(interface, candidate) for candidate in candidates)


def _interface_matches(interface: str, candidate: str) -> bool:
    if interface == candidate:
        return True
    if ":" in interface or ":" in candidate:
        return False
    return interface.startswith(f"{candidate}.") or candidate.startswith(f"{interface}.")


def _cell_accepts_interface(cell: Any, interface: str) -> bool:
    accepted = list(getattr(getattr(cell, "receptor", None), "accepted_interfaces", []))
    return not accepted or _any_interface_matches(interface, accepted)


def _environment_accepts_interface(tissue: Any, cell: Any, interface: str) -> bool:
    interfaces = list(getattr(tissue.environment, "interfaces", []))
    if not interfaces:
        return True
    for candidate in interfaces:
        if not _interface_matches(interface, str(candidate.id)):
            continue
        accepts_fates = list(getattr(candidate, "accepts_fates", []))
        if cell is None or not accepts_fates or getattr(cell, "fate", None) in accepts_fates:
            return True
    return False


def _url_host(url: str | None) -> str:
    if not url:
        return ""
    return urllib.parse.urlparse(url).hostname or ""

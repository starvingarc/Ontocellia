from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ontocellia.framework.core import TissueRuntime
from ontocellia.framework.induction import InductionRequest, TemplateInductionCompiler
from ontocellia.framework.llm import EffectorRuntime, MockLLMProvider


@dataclass(slots=True)
class OfficialAgentRunConfig:
    model_profile: str | None = None
    domain: str = "generic"
    max_steps: int = 8
    seed: int = 7
    use_mock: bool = True


@dataclass(slots=True)
class OfficialAgentRunResult:
    command_count: int
    action_count: int
    artifacts: dict[str, str]
    failure_mode: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_official_terminal_agent(
    task_description: str,
    session: Any,
    *,
    logging_dir: str | Path | None = None,
    config: OfficialAgentRunConfig | None = None,
) -> OfficialAgentRunResult:
    active = config or OfficialAgentRunConfig()
    output_dir = Path(logging_dir or Path.cwd()) / "ontocellia"
    output_dir.mkdir(parents=True, exist_ok=True)
    tissue, actions = _run_tissue_for_task(
        task_description,
        domain=active.domain,
        available_interfaces=["workspace", "shell.run", "git", "pytest"],
        ticks=max(1, active.max_steps),
        seed=active.seed,
    )
    commands = terminal_commands_from_actions(actions, max_commands=max(0, active.max_steps))
    sent = 0
    for command in commands:
        if _send_terminal_command(session, command):
            sent += 1
    failure_mode = None if sent else "no_terminal_command"
    artifacts = _write_terminal_artifacts(output_dir, tissue, actions, commands, failure_mode)
    return OfficialAgentRunResult(sent, len(actions), artifacts, failure_mode)


def terminal_commands_from_actions(actions: list[dict[str, Any]], *, max_commands: int = 8) -> list[str]:
    commands: list[str] = []
    for action in actions:
        if len(commands) >= max_commands:
            break
        intent = str(action.get("intent_type") or action.get("gene_id") or "")
        if intent == "inspect_context":
            commands.append("pwd")
            commands.append("ls -la")
        elif intent == "review_output":
            commands.append("git status --short")
        elif intent == "propose_patch":
            commands.append("printf '%s\\n' 'Ontocellia produced a patch intent; review action_intents.json for details.'")
        elif intent == "record_memory":
            continue
        else:
            commands.append("printf '%s\\n' 'Ontocellia completed a bounded reasoning step.'")
    return _dedupe_commands(commands)[:max_commands]


def openai_compatible_bridge_completion(payload: dict[str, Any], *, output_root: str | Path) -> dict[str, Any]:
    messages = list(payload.get("messages") or [])
    task = _last_user_message(messages) or "Continue the official benchmark task."
    tools = _tool_names(payload.get("tools") or [])
    tissue, actions = _run_tissue_for_task(
        task,
        domain="generic",
        available_interfaces=tools or ["workspace"],
        ticks=4,
        seed=7,
    )
    session_id = f"bridge-{int(time.time() * 1000)}"
    session_dir = Path(output_root) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    _write_json(session_dir / "bridge_summary.json", {"task": task, "actions": actions, "fate_counts": tissue.fate_counts()})
    _write_json(session_dir / "tissue_trace.json", tissue.trace.events)
    _write_json(session_dir / "action_intents.json", actions)
    tool_call = _tool_call_for_actions(actions, tools)
    message: dict[str, Any] = {"role": "assistant", "content": "Ontocellia tissue processed the request."}
    finish_reason = "stop"
    if tool_call is not None:
        message["content"] = None
        message["tool_calls"] = [tool_call]
        finish_reason = "tool_calls"
    return {
        "id": session_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": str(payload.get("model") or "ontocellia-bridge"),
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "ontocellia": {"session_id": session_id, "artifacts": str(session_dir)},
    }


def _run_tissue_for_task(
    task: str,
    *,
    domain: str,
    available_interfaces: list[str],
    ticks: int,
    seed: int,
) -> tuple[TissueRuntime, list[dict[str, Any]]]:
    draft = TemplateInductionCompiler().compile(
        InductionRequest(task=task, domain=domain, available_interfaces=available_interfaces, seed=seed)
    )
    tissue = TissueRuntime.seeded(draft.genome, draft.environment, seed=seed)
    tissue.develop(ticks=max(1, ticks))
    _develop_until_actions(tissue)
    actions = tissue.execute(effectors=EffectorRuntime(MockLLMProvider()))
    tissue.develop(ticks=1)
    return tissue, actions


def _develop_until_actions(tissue: TissueRuntime) -> None:
    for _ in range(6):
        if any(cell.differentiated and cell.expressed_gene_ids for cell in tissue.cells.values()):
            return
        tissue.develop(ticks=1)


def _send_terminal_command(session: Any, command: str) -> bool:
    for name in ("send_command", "run_command", "sendline"):
        method = getattr(session, name, None)
        if callable(method):
            method(command)
            return True
    send_keys = getattr(session, "send_keys", None)
    if callable(send_keys):
        try:
            send_keys(command, enter=True)
        except TypeError:
            send_keys(command)
        return True
    return False


def _write_terminal_artifacts(
    output_dir: Path,
    tissue: TissueRuntime,
    actions: list[dict[str, Any]],
    commands: list[str],
    failure_mode: str | None,
) -> dict[str, str]:
    summary = {
        "commands": commands,
        "command_count": len(commands),
        "action_count": len(actions),
        "failure_mode": failure_mode,
        "fate_counts": tissue.fate_counts(),
        "stage_counts": tissue.stage_counts(),
    }
    paths = {
        "summary": output_dir / "ontocellia_terminal_summary.json",
        "trace": output_dir / "tissue_trace.json",
        "actions": output_dir / "action_intents.json",
    }
    _write_json(paths["summary"], summary)
    _write_json(paths["trace"], tissue.trace.events)
    _write_json(paths["actions"], actions)
    return {key: str(path) for key, path in paths.items()}


def _tool_names(tools: list[Any]) -> list[str]:
    names: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function", {})
        if isinstance(function, dict) and function.get("name"):
            names.append(str(function["name"]))
    return list(dict.fromkeys(names))


def _tool_call_for_actions(actions: list[dict[str, Any]], tools: list[str]) -> dict[str, Any] | None:
    if not tools:
        return None
    preferred = tools[0]
    for action in actions:
        required = [str(item) for item in action.get("required_interfaces", [])]
        match = next((tool for tool in tools if tool in required), None)
        if match is not None:
            preferred = match
            break
    return {
        "id": "call_ontocellia_0",
        "type": "function",
        "function": {"name": preferred, "arguments": "{}"},
    }


def _last_user_message(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str):
                return content.strip()
    return ""


def _dedupe_commands(commands: list[str]) -> list[str]:
    return list(dict.fromkeys(command for command in commands if "\n" not in command))


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

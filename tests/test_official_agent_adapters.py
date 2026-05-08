from __future__ import annotations

import json
from pathlib import Path

from ontocellia.framework.official_agents import (
    OfficialAgentRunConfig,
    run_official_terminal_agent,
    terminal_commands_from_actions,
)
from ontocellia.official_terminal_agent import OntocelliaTerminalAgent


class FakeTmuxSession:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def send_command(self, command: str) -> None:
        self.commands.append(command)


def test_terminal_agent_perform_task_writes_artifacts_and_sends_bounded_command(tmp_path: Path) -> None:
    session = FakeTmuxSession()
    agent = OntocelliaTerminalAgent(max_steps=4, use_mock=True)

    result = agent.perform_task("Inspect the workspace and report the evidence.", session, logging_dir=tmp_path)

    output = tmp_path / "ontocellia"
    assert session.commands
    assert all("\n" not in command for command in session.commands)
    assert (output / "ontocellia_terminal_summary.json").exists()
    assert (output / "tissue_trace.json").exists()
    assert (output / "action_intents.json").exists()
    assert getattr(result, "failure_mode", None) in {None, ""}


def test_terminal_runner_returns_failure_mode_when_no_command_can_be_sent(tmp_path: Path) -> None:
    result = run_official_terminal_agent(
        "Remember the task without touching the terminal.",
        FakeTmuxSession(),
        logging_dir=tmp_path,
        config=OfficialAgentRunConfig(max_steps=0, use_mock=True),
    )

    assert result.command_count == 0
    assert result.failure_mode == "no_terminal_command"
    summary = json.loads((tmp_path / "ontocellia" / "ontocellia_terminal_summary.json").read_text(encoding="utf-8"))
    assert summary["failure_mode"] == "no_terminal_command"


def test_terminal_commands_ignore_unsafe_action_shell_payloads() -> None:
    commands = terminal_commands_from_actions(
        [
            {
                "intent_type": "review_output",
                "target": "repo",
                "payload": {"command": "rm -rf /"},
            },
            {
                "intent_type": "inspect_context",
                "target": "workspace",
                "payload": {"command": "cat /etc/passwd"},
            },
        ],
        max_commands=4,
    )

    assert commands
    assert not any("rm -rf" in command or "/etc/passwd" in command for command in commands)

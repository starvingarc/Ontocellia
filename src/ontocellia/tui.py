from __future__ import annotations

from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Input, Static, TabPane, TabbedContent

from ontocellia.framework.benchmark import BenchmarkSuite, TissueBenchmarkRunner
from ontocellia.framework.interactive import InteractiveSessionSnapshot, InteractiveTissueSession
from ontocellia.framework.llm import LLMProvider, MockLLMProvider
from ontocellia.framework.model_config import load_user_config


class OntocelliaTUI(App[None]):
    """Soft Lab Console TUI for interactive tissue sessions."""

    theme = "rose-pine-dawn"

    CSS = """
    Screen {
        background: #f6fbf8;
        color: #25342f;
    }

    #status {
        height: 4;
        padding: 0 2;
        background: #e4f4ec;
        color: #26473c;
        text-style: bold;
    }

    #main {
        height: 1fr;
        padding: 1 1 0 1;
    }

    #agents {
        width: 36;
        border: round #7bbf9d;
        background: #fbfffd;
        color: #263630;
    }

    #events {
        width: 1fr;
        border: round #8dc6d6;
        padding: 0 1;
        background: #fbfdff;
        color: #263845;
    }

    #summary {
        width: 38;
        border: round #e3c88c;
        padding: 0 1;
        background: #fffdf6;
        color: #413824;
    }

    #bottom {
        height: 12;
        margin: 0 1;
        border: round #b7a4df;
        background: #fdfbff;
        color: #332f42;
    }

    #command-input {
        height: 3;
        margin: 0 1;
        border: round #7bbf9d;
        background: #ffffff;
        color: #25342f;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_events", "Clear"),
    ]

    def __init__(
        self,
        *,
        output_root: str | Path = Path("artifacts/tui_sessions"),
        provider: LLMProvider | None = None,
    ) -> None:
        super().__init__()
        self.session = InteractiveTissueSession(output_root=output_root, provider=provider)
        self.use_mock = isinstance(provider, MockLLMProvider)
        self.event_lines: list[str] = [
            "Welcome. Type a task to culture a tissue.",
            "The first sentence becomes induction; cells then express intents and share matrix memory.",
        ]
        self.active_tab = "intents"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("", id="status")
        with Horizontal(id="main"):
            agents = DataTable(id="agents")
            agents.border_title = "Cell Culture"
            yield agents
            yield Static("", id="events")
            yield Static("", id="summary")
        with TabbedContent(id="bottom"):
            with TabPane("Intents", id="intents"):
                yield Static("", id="intent-view")
            with TabPane("Matrix", id="matrix"):
                yield Static("", id="matrix-view")
            with TabPane("Handoffs", id="handoffs"):
                yield Static("", id="handoff-view")
            with TabPane("Tools", id="tools"):
                yield Static("", id="tool-view")
            with TabPane("Report", id="report"):
                yield Static("", id="report-view")
        yield Input(placeholder="task to culture, or /help", id="command-input")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Ontocellia"
        self.sub_title = "Soft Lab"
        self.query_one("#events", Static).border_title = "Culture Log"
        self.query_one("#summary", Static).border_title = "Task Niche"
        self.query_one("#bottom", TabbedContent).border_title = "Shared Matrix"
        table = self.query_one("#agents", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns("cell", "stage", "fate", "niche", "genes")
        self._refresh()
        self.query_one("#command-input", Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        event.input.value = ""
        if not value:
            return
        if value.startswith("/"):
            await self._handle_command(value[1:])
            event.input.focus()
            return
        snapshot = self.session.submit_task(value, ticks=2, use_mock=self.use_mock)
        self._log(f"induction: {snapshot.task}")
        self._log(snapshot.notice)
        if snapshot.status == "ran":
            self._log(f"collaboration: {snapshot.actions} intents, {snapshot.messages} messages, {snapshot.matrix_records} matrix records.")
        self._refresh()
        event.input.focus()

    async def _handle_command(self, command: str) -> None:
        if command in {"exit", "quit"}:
            self.exit()
            return
        if command == "help":
            self._show_help()
        elif command == "clear":
            self.event_lines = []
        elif command == "setup":
            self._log("setup: run `python -m ontocellia config setup` in another terminal, then return here with /models.")
        elif command == "models":
            self._log(_models_text())
        elif command.startswith("use "):
            self._log("use: run `python -m ontocellia config models set NAME` for now.")
        elif command == "mock":
            self.use_mock = True
            self._log("mock effector enabled for this session.")
        elif command == "benchmark":
            output = self.session.output_root.parent / "benchmarks" / "minibench"
            result = TissueBenchmarkRunner(suite=BenchmarkSuite.builtin("ontocellia_minibench_v1"), effector="mock-llm").run(output)
            average = sum(item.score for item in result.results) / max(1, len(result.results))
            self._log(f"benchmark ontocellia_minibench_v1 average {average:.3f} across {len(result.results)} tasks")
            self._log(f"benchmark report: {result.report_path}")
        elif command.startswith("new "):
            snapshot = self.session.submit_task(command.split(" ", 1)[1], ticks=2, use_mock=self.use_mock)
            self._log(f"induction: {snapshot.task}")
            self._log(snapshot.notice)
            if snapshot.status == "ran":
                self._log(f"collaboration: {snapshot.actions} intents, {snapshot.messages} messages, {snapshot.matrix_records} matrix records.")
        elif command.startswith("run"):
            ticks = _ticks(command, default=2)
            snapshot = self.session.run(ticks=ticks, use_mock=self.use_mock)
            self._log(snapshot.notice)
            if snapshot.status == "ran":
                self._log(f"collaboration: {snapshot.actions} intents, {snapshot.messages} messages, {snapshot.matrix_records} matrix records.")
        elif command == "step":
            snapshot = self.session.step(use_mock=self.use_mock)
            self._log(snapshot.notice)
        elif command in {"agents", "intents", "matrix", "handoffs", "tools", "report", "config"}:
            self._log(f"view: {command}")
        else:
            self._log(f"unknown command: /{command}")
            self._show_help()
        self._refresh()

    def action_clear_events(self) -> None:
        self.event_lines = []
        self._refresh()

    def _show_help(self) -> None:
        self._log(
            "commands: /new <task>, /run [ticks], /step, /agents, /intents, /matrix, /handoffs, /tools, /report, /models, /mock, /setup, /clear, /exit"
        )

    def _refresh(self) -> None:
        snapshot = self.session.snapshot()
        self.query_one("#status", Static).update(_status(snapshot, self.use_mock))
        self.query_one("#events", Static).update("\n".join(self.event_lines[-14:]))
        self.query_one("#summary", Static).update(_summary(snapshot))
        self._refresh_agents()
        self.query_one("#intent-view", Static).update(_intents(self.session.actions))
        self.query_one("#matrix-view", Static).update(_matrix(self.session.matrix_records()))
        self.query_one("#handoff-view", Static).update(_handoffs(self.session.handoffs()))
        self.query_one("#tool-view", Static).update(_tools(self.session.tool_invocations()))
        self.query_one("#report-view", Static).update(self.session.report())

    def _refresh_agents(self) -> None:
        table = self.query_one("#agents", DataTable)
        table.clear()
        for agent in self.session.agents():
            genes = ",".join(agent["genes"]) or "-"
            if len(genes) > 22:
                genes = genes[:21] + "…"
            table.add_row(
                str(agent["id"]),
                str(agent["stage"]),
                str(agent["fate"]),
                str(agent["niche"]),
                genes,
            )

    def _log(self, line: str) -> None:
        self.event_lines.append(line)


def run_tui() -> None:
    OntocelliaTUI().run()


def _status(snapshot: InteractiveSessionSnapshot, use_mock: bool) -> str:
    model = "mock-llm" if use_mock else snapshot.model_profile or "unconfigured"
    session = snapshot.session_id or "no active tissue"
    return "\n".join(
        [
            "Ontocellia Soft Lab",
            f"culture repo_repair   model {model}   session {session}",
            "plain text starts a tissue; slash commands inspect or guide it",
        ]
    )


def _summary(snapshot: InteractiveSessionSnapshot) -> str:
    return "\n".join(
        [
            f"task: {snapshot.task or '-'}",
            f"status: {snapshot.status}",
            f"development: {snapshot.development_stage}",
            f"origin: cell {snapshot.origin_cell_id if snapshot.origin_cell_id is not None else '-'}",
            f"cells: {snapshot.population}",
            f"stages: {snapshot.stage_counts or {}}",
            f"fates: {snapshot.fate_counts or {}}",
            f"proliferation: {snapshot.proliferation_events}",
            f"actions: {snapshot.actions}",
            f"messages: {snapshot.messages}",
            f"matrix: {snapshot.matrix_records}",
            f"handoffs: {snapshot.handoffs}",
            f"dir: {snapshot.session_dir}",
            "",
            snapshot.notice or "A quiet dish, waiting for induction.",
        ]
    )


def _intents(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return "No action intents yet. Type a task and the tissue will begin."
    lines = []
    for action in actions[-10:]:
        lines.append(
            f"cell {action.get('cell_id')} | {action.get('fate')} | {action.get('intent_type')} -> {action.get('target')}"
        )
        rationale = str(action.get("rationale", ""))
        if rationale:
            lines.append(f"  {rationale[:88]}")
    return "\n".join(lines)


def _matrix(records: list[dict[str, Any]]) -> str:
    if not records:
        return "Matrix is clear. Evidence will settle here after cells communicate."
    return "\n".join(f"{record.get('kind')} | {', '.join(record.get('tags', []))}: {record.get('content')}" for record in records)


def _handoffs(events: list[dict[str, Any]]) -> str:
    if not events:
        return "No handoffs yet."
    return "\n".join(str(event) for event in events)


def _tools(invocations: list[dict[str, Any]]) -> str:
    if not invocations:
        return "No tool invocations planned yet. Tool execution stays behind explicit CLI policy."
    lines = []
    for invocation in invocations:
        lines.append(
            f"{invocation.get('id')} | {invocation.get('adapter')}:{invocation.get('operation')} | {invocation.get('interface')} -> {invocation.get('target')}"
        )
    return "\n".join(lines)


def _models_text() -> str:
    config = load_user_config()
    if not config.profiles:
        return "No model profiles. Use /setup or `python -m ontocellia config setup`."
    lines = [f"default: {config.default_model}"]
    for name, profile in sorted(config.profiles.items()):
        marker = "*" if name == config.default_model else "-"
        lines.append(f"{marker} {name}: {profile.provider} / {profile.model}")
    return "\n".join(lines)


def _ticks(command: str, *, default: int) -> int:
    parts = command.split()
    if len(parts) < 2:
        return default
    try:
        return max(1, int(parts[1]))
    except ValueError:
        return default

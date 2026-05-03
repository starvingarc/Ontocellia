from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ontocellia.framework.core import TissueRuntime
from ontocellia.framework.execution import ExtracellularToolRuntime, ToolPolicy
from ontocellia.framework.induction import InductionRequest, TemplateInductionCompiler
from ontocellia.framework.llm import EffectorRuntime, LLMProvider, MockLLMProvider
from ontocellia.framework.model_config import load_user_config, resolve_effector_provider


@dataclass(slots=True)
class InteractiveSessionSnapshot:
    status: str
    task: str
    session_id: str
    session_dir: Path
    model_profile: str
    population: int = 0
    fate_counts: dict[str, int] | None = None
    actions: int = 0
    messages: int = 0
    matrix_records: int = 0
    handoffs: int = 0
    development_stage: str = "idle"
    origin_cell_id: int | None = None
    stage_counts: dict[str, int] | None = None
    proliferation_events: int = 0
    notice: str = ""

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["session_dir"] = str(self.session_dir)
        data["fate_counts"] = self.fate_counts or {}
        data["stage_counts"] = self.stage_counts or {}
        return data


class InteractiveTissueSession:
    """Stateful interactive harness used by the TUI and fallback shell."""

    def __init__(
        self,
        *,
        output_root: str | Path = Path("artifacts/tui_sessions"),
        provider: LLMProvider | None = None,
        seed: int = 7,
    ) -> None:
        self.output_root = Path(output_root)
        self.provider = provider
        self.seed = seed
        self.task = ""
        self.session_id = ""
        self.session_dir = self.output_root
        self.draft: Any | None = None
        self.tissue: TissueRuntime | None = None
        self.actions: list[dict[str, Any]] = []
        self.status = "idle"
        self.notice = ""

    def new_task(self, task: str, *, domain: str = "repo_repair") -> InteractiveSessionSnapshot:
        self.task = task.strip()
        self.session_id = self.session_id or _session_id(self.task)
        if self.session_dir == self.output_root:
            self.session_dir = self.output_root / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.draft = TemplateInductionCompiler().compile(
            InductionRequest(
                task=self.task,
                domain=domain,
                available_interfaces=["workspace", "pytest", "git"],
                seed=self.seed,
            )
        )
        self.draft.write(self.session_dir / "induction")
        self.tissue = TissueRuntime.seeded(self.draft.genome, self.draft.environment, seed=self.seed)
        self.tissue.develop(ticks=1)
        self.actions = []
        self.status = "ready"
        self.notice = "Tissue induced. Use /run or /step to let cells collaborate."
        self._write_artifacts()
        return self.snapshot()

    def submit_task(self, task: str, *, ticks: int = 2, domain: str = "repo_repair", use_mock: bool = False) -> InteractiveSessionSnapshot:
        self.new_task(task, domain=domain)
        return self.run(ticks=ticks, use_mock=use_mock)

    def run(self, *, ticks: int = 2, use_mock: bool = False) -> InteractiveSessionSnapshot:
        if self.tissue is None:
            self.status = "idle"
            self.notice = "Enter a task first, or use /new <task>."
            return self.snapshot()
        provider = self._provider(use_mock=use_mock)
        if provider is None:
            self.status = "setup_required"
            self.notice = "Run /setup to configure a model profile, or use /mock for a deterministic local provider."
            self._write_artifacts()
            return self.snapshot()
        self.tissue.develop(ticks=max(1, int(ticks)))
        self._develop_until_executable()
        self.actions = self.tissue.execute(effectors=EffectorRuntime(provider))
        self.tissue.develop(ticks=1)
        self.status = "ran"
        self.notice = "Cells expressed intents, exchanged messages, and updated matrix memory."
        self._write_artifacts()
        return self.snapshot()

    def step(self, *, use_mock: bool = False) -> InteractiveSessionSnapshot:
        return self.run(ticks=1, use_mock=use_mock)

    def snapshot(self) -> InteractiveSessionSnapshot:
        trace_counts = self._trace_counts()
        config = load_user_config()
        return InteractiveSessionSnapshot(
            status=self.status,
            task=self.task,
            session_id=self.session_id,
            session_dir=self.session_dir,
            model_profile=config.default_model,
            population=len(self.tissue.cells) if self.tissue is not None else 0,
            fate_counts=self.tissue.fate_counts() if self.tissue is not None else {},
            actions=len(self.actions),
            messages=trace_counts["messages"],
            matrix_records=len(self.tissue.environment.matrix.records) if self.tissue is not None else 0,
            handoffs=trace_counts["handoffs"],
            development_stage=getattr(self.tissue, "development_stage", "idle") if self.tissue is not None else "idle",
            origin_cell_id=getattr(self.tissue, "origin_cell_id", None) if self.tissue is not None else None,
            stage_counts=self.tissue.stage_counts() if self.tissue is not None else {},
            proliferation_events=trace_counts["proliferation"],
            notice=self.notice,
        )

    def agents(self) -> list[dict[str, Any]]:
        if self.tissue is None:
            return []
        return [
            {
                "id": cell.id,
                "stage": str(cell.stage),
                "fate": cell.fate,
                "niche": cell.niche_id or "-",
                "energy": round(float(cell.energy), 3),
                "genes": list(cell.expressed_gene_ids),
            }
            for cell in sorted(self.tissue.cells.values(), key=lambda item: item.id)
        ]

    def matrix_records(self, limit: int = 8) -> list[dict[str, Any]]:
        if self.tissue is None:
            return []
        records = self.tissue.environment.matrix.records[-limit:]
        return [record.as_dict() for record in records]

    def handoffs(self, limit: int = 8) -> list[dict[str, Any]]:
        if self.tissue is None:
            return []
        return [event for event in self.tissue.trace.events if event["type"].startswith("handoff")][-limit:]

    def tool_invocations(self, limit: int = 8) -> list[dict[str, Any]]:
        if not self.actions:
            return []
        invocations = ExtracellularToolRuntime().plan_invocations(self.actions, ToolPolicy())
        return [invocation.as_dict() for invocation in invocations[-limit:]]

    def report(self) -> str:
        if self.tissue is None:
            return "No active tissue session."
        snapshot = self.snapshot()
        return "\n".join(
            [
                "# Ontocellia Tissue Session",
                "",
                f"- Task: {snapshot.task}",
                f"- Status: {snapshot.status}",
                f"- Population: {snapshot.population}",
                f"- Development stage: {snapshot.development_stage}",
                f"- Stage counts: {snapshot.stage_counts}",
                f"- Fates: {snapshot.fate_counts}",
                f"- Actions: {snapshot.actions}",
                f"- Messages: {snapshot.messages}",
                f"- Matrix records: {snapshot.matrix_records}",
                f"- Handoffs: {snapshot.handoffs}",
                "",
            ]
        )

    def _provider(self, *, use_mock: bool) -> LLMProvider | None:
        if use_mock:
            return MockLLMProvider()
        if self.provider is not None:
            return self.provider
        try:
            provider = resolve_effector_provider("llm")
        except Exception:
            return None
        return provider

    def _develop_until_executable(self) -> None:
        if self.tissue is None:
            return
        guard = max(1, self.tissue.min_population_before_differentiation + len(self.tissue.environment.niches))
        while guard > 0 and not any(cell.differentiated for cell in self.tissue.cells.values()):
            self.tissue.develop(ticks=1)
            guard -= 1

    def _write_artifacts(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        snapshot = self.snapshot().as_dict()
        (self.session_dir / "session.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
        (self.session_dir / "report.md").write_text(self.report(), encoding="utf-8")
        if self.tissue is None:
            return
        summary = {
            "objective": self.tissue.environment.objective,
            "ticks": self.tissue.tick_count,
            "population": len(self.tissue.cells),
            "fate_counts": self.tissue.fate_counts(),
            "niche_occupancy": self.tissue.niche_occupancy(),
            "actions": self.actions,
            "messages": snapshot["messages"],
            "matrix_records": snapshot["matrix_records"],
            "handoffs": snapshot["handoffs"],
            "development_stage": snapshot["development_stage"],
            "origin_cell_id": snapshot["origin_cell_id"],
            "stage_counts": snapshot["stage_counts"],
            "proliferation_events": snapshot["proliferation_events"],
        }
        (self.session_dir / "tissue_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        (self.session_dir / "tissue_trace.json").write_text(
            json.dumps(self.tissue.trace.events, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (self.session_dir / "action_intents.json").write_text(json.dumps(self.actions, indent=2, sort_keys=True), encoding="utf-8")
        llm_trace = [event for event in self.tissue.trace.events if event["type"] == "llm_effector"]
        (self.session_dir / "llm_trace.json").write_text(json.dumps(llm_trace, indent=2, sort_keys=True), encoding="utf-8")

    def _trace_counts(self) -> dict[str, int]:
        if self.tissue is None:
            return {"messages": 0, "handoffs": 0, "proliferation": 0}
        return {
            "messages": sum(1 for event in self.tissue.trace.events if event["type"] == "message_emitted"),
            "handoffs": sum(1 for event in self.tissue.trace.events if event["type"] == "handoff_completed"),
            "proliferation": sum(1 for event in self.tissue.trace.events if event["type"] == "proliferation"),
        }


def _session_id(task: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", task.lower()).strip("-")
    return (slug or "untitled")[:48]

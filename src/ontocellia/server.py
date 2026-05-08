from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from ontocellia.framework.execution import ToolPolicy
from ontocellia.framework.interactive import InteractiveTissueSession
from ontocellia.framework.llm import MockLLMProvider
from ontocellia.framework.official_agents import openai_compatible_bridge_completion


@dataclass(slots=True)
class TissueEventBus:
    subscribers: dict[str, list[asyncio.Queue[dict[str, Any]]]] = field(default_factory=dict)

    async def connect(self, session_id: str) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.subscribers.setdefault(session_id, []).append(queue)
        return queue

    def disconnect(self, session_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        queues = self.subscribers.get(session_id, [])
        if queue in queues:
            queues.remove(queue)
        if not queues and session_id in self.subscribers:
            del self.subscribers[session_id]

    async def publish(self, session_id: str, event: dict[str, Any]) -> None:
        payload = {"session_id": session_id, **event}
        for queue in list(self.subscribers.get(session_id, [])):
            await queue.put(payload)

    async def publish_many(self, session_id: str, events: list[dict[str, Any]]) -> None:
        for event in events:
            await self.publish(session_id, event)


@dataclass(slots=True)
class ManagedSession:
    id: str
    session: InteractiveTissueSession
    trace_cursor: int = 0


@dataclass(slots=True)
class TissueSessionManager:
    output_root: Path = Path("artifacts/server_sessions")
    use_mock: bool = True
    sessions: dict[str, ManagedSession] = field(default_factory=dict)
    counter: int = 0

    def create(self) -> ManagedSession:
        self.counter += 1
        session_id = f"session-{self.counter}"
        session_dir = self.output_root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        provider = MockLLMProvider() if self.use_mock else None
        session = InteractiveTissueSession(output_root=self.output_root, provider=provider)
        session.session_id = session_id
        session.session_dir = session_dir
        session._write_artifacts()
        managed = ManagedSession(id=session_id, session=session)
        self.sessions[session_id] = managed
        return managed

    def get(self, session_id: str) -> ManagedSession:
        try:
            return self.sessions[session_id]
        except KeyError as error:
            raise HTTPException(status_code=404, detail=f"unknown session: {session_id}") from error

    def list(self) -> list[ManagedSession]:
        return [self.sessions[key] for key in sorted(self.sessions)]

    def trace_delta(self, managed: ManagedSession) -> list[dict[str, Any]]:
        tissue = managed.session.tissue
        if tissue is None:
            return []
        events = tissue.trace.events[managed.trace_cursor :]
        managed.trace_cursor = len(tissue.trace.events)
        return [dict(event) for event in events]


def create_app(
    *,
    output_root: str | Path = Path("artifacts/server_sessions"),
    use_mock: bool = True,
) -> FastAPI:
    manager = TissueSessionManager(output_root=Path(output_root), use_mock=use_mock)
    bus = TissueEventBus()
    app = FastAPI(title="Ontocellia Living Tissue Server")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
        return openai_compatible_bridge_completion(payload, output_root=manager.output_root / "bridge_sessions")

    @app.get("/projects")
    async def list_projects() -> dict[str, list[dict[str, Any]]]:
        sessions = [_snapshot(managed) for managed in manager.list()]
        return {"projects": [_project_snapshot(manager.output_root, sessions)]}

    @app.get("/projects/{project_id}/sessions")
    async def list_project_sessions(project_id: str) -> dict[str, Any]:
        if project_id != "local":
            raise HTTPException(status_code=404, detail=f"unknown project: {project_id}")
        return {"project_id": project_id, "sessions": [_snapshot(managed) for managed in manager.list()]}

    @app.post("/sessions")
    async def create_session(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        managed = manager.create()
        snapshot = _snapshot(managed)
        await bus.publish(managed.id, {"type": "session_created", "snapshot": snapshot})
        return {"session_id": managed.id, "snapshot": snapshot}

    @app.get("/sessions")
    async def list_sessions() -> dict[str, list[dict[str, Any]]]:
        return {"sessions": [_snapshot(managed) for managed in manager.list()]}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str) -> dict[str, Any]:
        managed = manager.get(session_id)
        return {"session_id": managed.id, "snapshot": _snapshot(managed)}

    @app.post("/sessions/{session_id}/task")
    async def submit_task(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        managed = manager.get(session_id)
        task = str(payload.get("task", "")).strip()
        if not task:
            raise HTTPException(status_code=400, detail="task is required")
        domain = str(payload.get("domain", "repo_repair"))
        snapshot_before = managed.session.snapshot()
        managed.session.new_task(task, domain=domain)
        events = [{"type": "induction", "task": task, "snapshot": _snapshot(managed)}]
        events.extend(_trace_events(session_id, manager.trace_delta(managed)))
        if snapshot_before.status != managed.session.status:
            events.append({"type": "session_update", "snapshot": _snapshot(managed)})
        await bus.publish_many(session_id, events)
        return {"session_id": managed.id, "snapshot": _snapshot(managed)}

    @app.post("/sessions/{session_id}/change-medium")
    async def change_medium(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        managed = manager.get(session_id)
        text = str(payload.get("text") or payload.get("task") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is required")
        ticks = int(payload.get("ticks", 1))
        snapshot = managed.session.change_medium(text, ticks=max(1, ticks))
        events = [
            {
                "type": "medium_changed",
                "text": text,
                "snapshot": snapshot.as_dict(),
            }
        ]
        events.extend(_trace_events(session_id, manager.trace_delta(managed)))
        await bus.publish_many(session_id, events)
        return {"session_id": managed.id, "snapshot": _snapshot(managed)}

    @app.post("/sessions/{session_id}/run")
    async def run_session(session_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        managed = manager.get(session_id)
        ticks = int((payload or {}).get("ticks", 2))
        managed.session.run(ticks=max(1, ticks), use_mock=manager.use_mock)
        events = [{"type": "session_update", "snapshot": _snapshot(managed)}]
        events.extend(_trace_events(session_id, manager.trace_delta(managed)))
        await bus.publish_many(session_id, events)
        return {"session_id": managed.id, "snapshot": _snapshot(managed)}

    @app.post("/sessions/{session_id}/step")
    async def step_session(session_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        managed = manager.get(session_id)
        managed.session.step(use_mock=manager.use_mock)
        events = [{"type": "session_update", "snapshot": _snapshot(managed)}]
        events.extend(_trace_events(session_id, manager.trace_delta(managed)))
        await bus.publish_many(session_id, events)
        return {"session_id": managed.id, "snapshot": _snapshot(managed)}

    @app.post("/sessions/{session_id}/interventions")
    async def intervene(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        managed = manager.get(session_id)
        intervention_type = str(payload.get("type") or "").strip()
        if not intervention_type:
            raise HTTPException(status_code=400, detail="type is required")
        try:
            snapshot = managed.session.intervene(intervention_type, **{key: value for key, value in payload.items() if key != "type"})
        except (KeyError, ValueError, TypeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        events = [{"type": "intervention", "intervention": intervention_type, "snapshot": snapshot.as_dict()}]
        events.extend(_trace_events(session_id, manager.trace_delta(managed)))
        await bus.publish_many(session_id, events)
        return {"session_id": managed.id, "snapshot": _snapshot(managed)}

    @app.get("/sessions/{session_id}/agents")
    async def agents(session_id: str) -> dict[str, Any]:
        managed = manager.get(session_id)
        return {"session_id": managed.id, "agents": managed.session.agents()}

    @app.get("/sessions/{session_id}/intents")
    async def intents(session_id: str) -> dict[str, Any]:
        managed = manager.get(session_id)
        return {"session_id": managed.id, "intents": list(managed.session.actions)}

    @app.get("/sessions/{session_id}/matrix")
    async def matrix(session_id: str) -> dict[str, Any]:
        managed = manager.get(session_id)
        return {"session_id": managed.id, "matrix": managed.session.matrix_records(limit=50)}

    @app.get("/sessions/{session_id}/handoffs")
    async def handoffs(session_id: str) -> dict[str, Any]:
        managed = manager.get(session_id)
        return {"session_id": managed.id, "handoffs": managed.session.handoffs(limit=50)}

    @app.get("/sessions/{session_id}/tools")
    async def tools(session_id: str) -> dict[str, Any]:
        managed = manager.get(session_id)
        return {
            "session_id": managed.id,
            "tools": managed.session.tool_invocations(limit=50),
            "results": managed.session.tool_results_summary(limit=50),
        }

    @app.get("/sessions/{session_id}/tool-approvals")
    async def tool_approvals(session_id: str) -> dict[str, Any]:
        managed = manager.get(session_id)
        return {
            "session_id": managed.id,
            "pending": [tool for tool in managed.session.tool_invocations(limit=100) if tool.get("approval_status") == "pending"],
            "results": managed.session.tool_results_summary(limit=100),
        }

    @app.post("/sessions/{session_id}/tool-approvals")
    async def approve_tools(session_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        managed = manager.get(session_id)
        data = payload or {}
        action_ids = [str(item) for item in data.get("action_ids", [])] if data.get("action_ids") else None
        approve = bool(data.get("approve", True))
        policy = _tool_policy_from_payload(data["policy"]) if "policy" in data else None
        results = managed.session.approve_tools(action_ids=action_ids, approve=approve, policy=policy)
        events = [{"type": "tool_approval", "approved": approve, "results": results, "snapshot": _snapshot(managed)}]
        events.extend(_trace_events(session_id, manager.trace_delta(managed)))
        await bus.publish_many(session_id, events)
        return {"session_id": managed.id, "results": results, "snapshot": _snapshot(managed)}

    @app.get("/sessions/{session_id}/artifacts/{name}")
    async def artifact(session_id: str, name: str) -> Response:
        managed = manager.get(session_id)
        if name not in _ALLOWED_ARTIFACTS:
            raise HTTPException(status_code=404, detail=f"artifact not found: {name}")
        path = managed.session.session_dir / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"artifact not found: {name}")
        text = path.read_text(encoding="utf-8")
        if name.endswith(".json"):
            return JSONResponse(content=json.loads(text))
        return PlainTextResponse(content=text)

    @app.websocket("/sessions/{session_id}/events")
    async def events(websocket: WebSocket, session_id: str) -> None:
        managed = manager.get(session_id)
        await websocket.accept()
        queue = await bus.connect(session_id)
        await websocket.send_json({"type": "snapshot", "session_id": session_id, "snapshot": _snapshot(managed)})
        try:
            while True:
                await websocket.send_json(await queue.get())
        except WebSocketDisconnect:
            bus.disconnect(session_id, queue)

    return app


def _snapshot(managed: ManagedSession) -> dict[str, Any]:
    data = managed.session.snapshot().as_dict()
    data["session_id"] = managed.id
    return data


def _trace_events(session_id: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"type": "trace", "trace_event": event, "session_id": session_id} for event in events]


_ALLOWED_ARTIFACTS = {
    "session.json",
    "report.md",
    "tissue_summary.json",
    "tissue_trace.json",
    "action_intents.json",
    "llm_trace.json",
    "tool_results.json",
}


def _project_snapshot(output_root: Path, sessions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "local",
        "name": Path.cwd().name or "Ontocellia",
        "root": str(Path.cwd()),
        "artifact_root": str(output_root),
        "sessions": sessions,
        "session_count": len(sessions),
    }


def _tool_policy_from_payload(payload: Any) -> ToolPolicy:
    data = payload if isinstance(payload, dict) else {}
    return ToolPolicy(
        workspace_root=Path(data.get("workspace_root") or Path.cwd()),
        allowed_interfaces=[str(item) for item in data.get("allowed_interfaces", [])],
        allowed_commands=[str(item) for item in data.get("allowed_commands", [])],
        allowed_write_globs=[str(item) for item in data.get("allowed_write_globs", [])],
        allowed_network_hosts=[str(item) for item in data.get("allowed_network_hosts", [])],
        allowed_mcp_tools=[str(item) for item in data.get("allowed_mcp_tools", [])],
        allowed_git_commands=[str(item) for item in data.get("allowed_git_commands", [])],
        enable_http_tools=bool(data.get("enable_http_tools", False)),
        enable_browser_tools=bool(data.get("enable_browser_tools", False)),
        timeout_seconds=float(data.get("timeout_seconds", 60.0)),
        dry_run=bool(data.get("dry_run", True)),
    )

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ontocellia.__main__ import build_parser
from ontocellia.server import create_app


def make_client(tmp_path: Path) -> TestClient:
    app = create_app(output_root=tmp_path / "server_sessions", use_mock=True)
    return TestClient(app)


def test_health_endpoint_returns_ok(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_create_session_returns_idle_snapshot(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.post("/sessions", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"]
    assert payload["snapshot"]["status"] == "idle"
    assert payload["snapshot"]["session_id"] == payload["session_id"]


def test_project_listing_reads_live_sessions(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    session_id = client.post("/sessions", json={}).json()["session_id"]
    client.post(f"/sessions/{session_id}/task", json={"task": "Fix failing tests."})

    projects = client.get("/projects").json()["projects"]
    project_sessions = client.get("/projects/local/sessions").json()["sessions"]

    assert projects[0]["id"] == "local"
    assert projects[0]["session_count"] == 1
    assert project_sessions[0]["session_id"] == session_id


def test_submit_task_induces_tissue_and_writes_artifacts(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    session_id = client.post("/sessions", json={}).json()["session_id"]

    response = client.post(f"/sessions/{session_id}/task", json={"task": "Fix failing tests while preserving behavior."})

    assert response.status_code == 200
    snapshot = response.json()["snapshot"]
    assert snapshot["status"] in {"ready", "ran"}
    assert snapshot["population"] >= 1
    assert (Path(snapshot["session_dir"]) / "session.json").exists()


def test_change_medium_updates_tissue_and_broadcasts_traceable_event(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    session_id = client.post("/sessions", json={}).json()["session_id"]
    client.post(f"/sessions/{session_id}/task", json={"task": "Fix failing tests."})

    response = client.post(
        f"/sessions/{session_id}/change-medium",
        json={"text": "Add review pressure and repair the failing pytest regression.", "ticks": 1},
    )

    snapshot = response.json()["snapshot"]
    trace = client.get(f"/sessions/{session_id}/artifacts/tissue_trace.json").json()
    matrix = client.get(f"/sessions/{session_id}/matrix").json()["matrix"]
    assert response.status_code == 200
    assert snapshot["status"] == "medium_changed"
    assert any(event["type"] == "medium_changed" for event in trace)
    assert any(record["kind"] == "medium_change" for record in matrix)


def test_run_and_step_update_live_session_views(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    session_id = client.post("/sessions", json={}).json()["session_id"]
    client.post(f"/sessions/{session_id}/task", json={"task": "Fix failing tests."})

    run_response = client.post(f"/sessions/{session_id}/run", json={"ticks": 2})
    step_response = client.post(f"/sessions/{session_id}/step", json={})

    assert run_response.status_code == 200
    assert step_response.status_code == 200
    assert run_response.json()["snapshot"]["actions"] > 0
    assert client.get(f"/sessions/{session_id}/agents").json()["agents"]
    assert client.get(f"/sessions/{session_id}/intents").json()["intents"]
    assert "matrix" in client.get(f"/sessions/{session_id}/matrix").json()
    assert "handoffs" in client.get(f"/sessions/{session_id}/handoffs").json()
    assert "tools" in client.get(f"/sessions/{session_id}/tools").json()


def test_intervention_can_clear_cell_and_trigger_regeneration_signals(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    session_id = client.post("/sessions", json={}).json()["session_id"]
    client.post(f"/sessions/{session_id}/task", json={"task": "Fix failing tests."})
    client.post(f"/sessions/{session_id}/run", json={"ticks": 3})
    agents = client.get(f"/sessions/{session_id}/agents").json()["agents"]
    target = next(agent for agent in agents if agent["stage"] == "differentiated")

    response = client.post(
        f"/sessions/{session_id}/interventions",
        json={"type": "clear_cell", "cell_id": target["id"], "reason": "test_clear"},
    )

    trace = client.get(f"/sessions/{session_id}/artifacts/tissue_trace.json").json()
    assert response.status_code == 200
    assert response.json()["snapshot"]["notice"].startswith("Cleared cell")
    assert any(event["type"] == "apoptosis" and event["cell_id"] == target["id"] for event in trace)


def test_tool_approval_queue_defaults_to_explicit_dry_run(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    session_id = client.post("/sessions", json={}).json()["session_id"]
    client.post(f"/sessions/{session_id}/task", json={"task": "Fix failing tests."})
    client.post(f"/sessions/{session_id}/run", json={"ticks": 2})
    pending = client.get(f"/sessions/{session_id}/tool-approvals").json()["pending"]
    assert pending

    response = client.post(
        f"/sessions/{session_id}/tool-approvals",
        json={
            "action_ids": [pending[0]["action_id"]],
            "approve": True,
            "policy": {"allowed_interfaces": [pending[0]["interface"]], "dry_run": True},
        },
    )

    tools = client.get(f"/sessions/{session_id}/tools").json()
    assert response.status_code == 200
    assert response.json()["results"]
    assert tools["results"]


def test_artifact_endpoint_returns_known_json_and_rejects_unknown(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    session_id = client.post("/sessions", json={}).json()["session_id"]
    client.post(f"/sessions/{session_id}/task", json={"task": "Fix failing tests."})

    artifact = client.get(f"/sessions/{session_id}/artifacts/session.json")
    missing = client.get(f"/sessions/{session_id}/artifacts/secrets.env")

    assert artifact.status_code == 200
    assert artifact.json()["session_id"] == session_id
    assert missing.status_code == 404


def test_websocket_receives_initial_snapshot_and_task_events(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    session_id = client.post("/sessions", json={}).json()["session_id"]

    with client.websocket_connect(f"/sessions/{session_id}/events") as websocket:
        initial = websocket.receive_json()
        assert initial["type"] == "snapshot"
        assert initial["session_id"] == session_id

        client.post(f"/sessions/{session_id}/task", json={"task": "Fix failing tests."})
        event = websocket.receive_json()

        assert event["type"] in {"induction", "session_update"}
        assert event["session_id"] == session_id


def test_invalid_session_returns_404(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.get("/sessions/missing")

    assert response.status_code == 404


def test_server_cli_parser_defaults_to_localhost() -> None:
    args = build_parser().parse_args(["server"])

    assert args.command == "server"
    assert args.host == "127.0.0.1"
    assert args.port == 8765


def test_openai_compatible_bridge_returns_chat_completion_message(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "ontocellia-bridge",
            "messages": [{"role": "user", "content": "Help the user follow policy and inspect context."}],
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["choices"][0]["message"]["content"]
    assert "api_key" not in response.text.lower()


def test_openai_compatible_bridge_returns_allowed_tool_call(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "ontocellia-bridge",
            "messages": [{"role": "user", "content": "Look up the order and update it if policy allows."}],
            "tools": [
                {"type": "function", "function": {"name": "lookup_order", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "update_order", "parameters": {"type": "object"}}},
            ],
        },
    )

    message = response.json()["choices"][0]["message"]
    tool_calls = message["tool_calls"]
    assert response.status_code == 200
    assert tool_calls
    assert tool_calls[0]["function"]["name"] in {"lookup_order", "update_order"}

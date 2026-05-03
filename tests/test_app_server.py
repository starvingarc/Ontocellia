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


def test_submit_task_induces_tissue_and_writes_artifacts(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    session_id = client.post("/sessions", json={}).json()["session_id"]

    response = client.post(f"/sessions/{session_id}/task", json={"task": "Fix failing tests while preserving behavior."})

    assert response.status_code == 200
    snapshot = response.json()["snapshot"]
    assert snapshot["status"] in {"ready", "ran"}
    assert snapshot["population"] >= 1
    assert (Path(snapshot["session_dir"]) / "session.json").exists()


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

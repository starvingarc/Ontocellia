from __future__ import annotations

import json
from pathlib import Path

from ontocellia.framework.interactive import InteractiveTissueSession
from ontocellia.framework.llm import MockLLMProvider
from ontocellia.framework.model_config import ModelProfile, OntocelliaUserConfig, save_user_config


def test_plain_task_creates_induced_tissue_session(tmp_path: Path) -> None:
    session = InteractiveTissueSession(output_root=tmp_path, provider=MockLLMProvider())

    snapshot = session.new_task("Fix failing tests while preserving behavior.")

    assert snapshot.status == "ready"
    assert snapshot.task == "Fix failing tests while preserving behavior."
    assert snapshot.session_dir.exists()
    assert (snapshot.session_dir / "induction" / "genome.yaml").exists()
    assert (snapshot.session_dir / "induction" / "environment.yaml").exists()
    assert snapshot.origin_cell_id == 0
    assert snapshot.population >= 1
    assert snapshot.stage_counts.get("stem", 0) >= 1


def test_mock_run_generates_actions_messages_matrix_and_artifacts(tmp_path: Path) -> None:
    session = InteractiveTissueSession(output_root=tmp_path, provider=MockLLMProvider())
    session.new_task("Fix failing tests while preserving behavior.")

    snapshot = session.run(ticks=2, use_mock=True)

    assert snapshot.status == "ran"
    assert snapshot.actions > 0
    assert snapshot.messages > 0
    assert snapshot.matrix_records > 0
    assert (snapshot.session_dir / "tissue_summary.json").exists()
    assert (snapshot.session_dir / "tissue_trace.json").exists()
    assert (snapshot.session_dir / "action_intents.json").exists()
    assert (snapshot.session_dir / "llm_trace.json").exists()
    assert (snapshot.session_dir / "session.json").exists()
    assert (snapshot.session_dir / "report.md").exists()

    persisted = json.loads((snapshot.session_dir / "session.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "ran"
    assert persisted["actions"] == snapshot.actions


def test_submit_task_auto_runs_collaboration_with_mock_provider(tmp_path: Path) -> None:
    session = InteractiveTissueSession(output_root=tmp_path, provider=MockLLMProvider())

    snapshot = session.submit_task("Fix failing tests while preserving behavior.", ticks=2)

    assert snapshot.status == "ran"
    assert snapshot.actions > 0
    assert snapshot.messages > 0
    assert snapshot.matrix_records > 0


def test_unconfigured_llm_profile_returns_setup_required(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ONTOCELLIA_CONFIG_DIR", str(tmp_path / "config"))
    session = InteractiveTissueSession(output_root=tmp_path / "sessions")
    session.new_task("Fix failing tests while preserving behavior.")

    snapshot = session.run(ticks=1)

    assert snapshot.status == "setup_required"
    assert "Run /setup" in snapshot.notice


def test_configured_mock_profile_runs_without_real_provider(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ONTOCELLIA_CONFIG_DIR", str(tmp_path / "config"))
    save_user_config(
        OntocelliaUserConfig(
            models={
                "default": "mock",
                "profiles": {"mock": ModelProfile(provider="mock-llm", model="mock-llm")},
            }
        )
    )
    session = InteractiveTissueSession(output_root=tmp_path / "sessions")
    session.new_task("Fix failing tests while preserving behavior.")

    snapshot = session.run(ticks=1)

    assert snapshot.status == "ran"
    assert snapshot.model_profile == "mock"
    assert snapshot.actions > 0

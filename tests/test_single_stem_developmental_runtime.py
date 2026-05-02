from __future__ import annotations

from ontocellia.framework import TissueRuntime
from ontocellia.framework.interactive import InteractiveTissueSession
from ontocellia.framework.llm import MockLLMProvider

from tests.test_agent_tissue_framework import make_repo_repair_tissue


def test_seeded_runtime_defaults_to_single_stem_origin() -> None:
    tissue = make_repo_repair_tissue(stem_cells=None)

    assert len(tissue.cells) == 1
    origin = tissue.cells[0]
    assert origin.stage == "stem"
    assert origin.fate == "stem"
    assert origin.position.node_id == "zygote-origin"
    assert tissue.origin_cell_id == 0
    assert tissue.development_stage == "proliferating"
    assert tissue.stage_counts() == {"stem": 1}


def test_single_stem_proliferates_before_differentiating() -> None:
    tissue = make_repo_repair_tissue(stem_cells=None)

    tissue.develop(ticks=1)

    assert len(tissue.cells) > 1
    assert tissue.stage_counts().get("progenitor", 0) >= 1
    assert "repair" not in tissue.fate_counts()

    tissue.develop(ticks=5)

    assert len(tissue.cells) >= 4
    assert tissue.fate_counts()["repair"] >= 2
    assert tissue.fate_counts()["explorer"] >= 1
    assert tissue.fate_counts()["reviewer"] >= 1
    assert tissue.development_stage in {"differentiating", "mature"}
    assert any(event["type"] == "proliferation" for event in tissue.trace.events)
    assert any(event["type"] == "development_stage_changed" for event in tissue.trace.events)


def test_repair_regeneration_traces_back_to_single_origin() -> None:
    tissue = make_repo_repair_tissue(stem_cells=None)
    tissue.develop(ticks=6)
    removed = next(cell.id for cell in tissue.cells.values() if cell.fate == "repair")

    tissue.clear_cell(removed, reason="manual_clear")
    tissue.develop(ticks=4)

    replacements = [
        cell
        for cell in tissue.cells.values()
        if cell.replaces_cell_id == removed and cell.fate == "repair"
    ]
    assert replacements
    assert replacements[-1].lineage.root_id == tissue.origin_cell_id
    assert any(event["type"] == "regeneration" and event["replaced_cell_id"] == removed for event in tissue.trace.events)


def test_interactive_session_summary_exposes_single_stem_development(tmp_path) -> None:
    session = InteractiveTissueSession(output_root=tmp_path, provider=MockLLMProvider())

    snapshot = session.submit_task("Fix failing tests while preserving behavior.", ticks=5)

    assert snapshot.status == "ran"
    assert snapshot.origin_cell_id == 0
    assert snapshot.development_stage in {"differentiating", "mature"}
    assert snapshot.stage_counts.get("differentiated", 0) >= 1
    assert snapshot.proliferation_events >= 1
    assert snapshot.actions > 0

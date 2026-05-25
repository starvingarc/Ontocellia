from __future__ import annotations

import json
from pathlib import Path

from ontocellia.__main__ import main
from ontocellia.framework.attribution import ContributionAttributionRuntime


def sample_action() -> dict[str, object]:
    return {
        "cell_id": 2,
        "fate": "repair",
        "expressed_gene_ids": ["gene_repair_from_test_failures"],
        "intent_type": "propose_patch",
        "target": "repair-niche",
        "rationale": "Patch narrow failure.",
        "required_interfaces": ["workspace"],
        "confidence": 0.8,
        "validation_hooks": ["python -m pytest -q"],
        "payload": {"context_record_ids": ["matrix-evidence-1"], "message": "Patch ready.", "handoff_to_fate": "reviewer"},
    }


def sample_trace() -> list[dict[str, object]]:
    action = sample_action()
    return [
        {
            "type": "matrix_deposit",
            "tick": 1,
            "id": "matrix-evidence-1",
            "source_cell_id": 1,
            "kind": "observation",
            "content": "pytest failed",
            "tags": ["test_failure"],
            "validation_status": "unverified",
        },
        {
            "type": "llm_effector",
            "tick": 2,
            "cell_id": 2,
            "context_record_ids": ["matrix-evidence-1"],
            "intent": action,
        },
        {
            "type": "message_emitted",
            "tick": 2,
            "id": "message-1",
            "sender_cell_id": 2,
            "kind": "handoff",
            "content": "Patch ready.",
            "recipient_fate": "reviewer",
        },
        {
            "type": "handoff_requested",
            "tick": 2,
            "request_id": "handoff-1",
            "source_cell_id": 2,
            "target_fate": "reviewer",
            "message_id": "message-1",
        },
        {
            "type": "handoff_completed",
            "tick": 3,
            "request_id": "handoff-1",
            "recipient_cell_id": 3,
            "accepted": True,
        },
        {
            "type": "tool_completed",
            "tick": 4,
            "invocation": {
                "id": "tool-1",
                "cell_id": 2,
                "intent_type": "propose_patch",
                "interface": "pytest.run",
                "context_record_ids": ["matrix-evidence-1"],
            },
            "status": "failed",
            "passed": False,
            "score": 0.0,
            "evidence": "pytest failed",
        },
    ]


def test_attribution_builds_graph_and_scores_from_trace() -> None:
    report = ContributionAttributionRuntime().analyze(
        trace=sample_trace(),
        actions=[sample_action()],
        tool_results=[
            {
                "invocation": {
                    "id": "tool-1",
                    "cell_id": 2,
                    "intent_type": "propose_patch",
                    "interface": "pytest.run",
                    "context_record_ids": ["matrix-evidence-1"],
                },
                "status": "failed",
                "passed": False,
                "score": 0.0,
                "evidence": "pytest failed",
            }
        ],
        validation_results=[{"name": "pytest", "passed": False, "score": 0.0, "target": "repo", "evidence": "pytest failed"}],
    )

    graph = report.as_dict()["graph"]
    node_ids = {node["id"] for node in graph["nodes"]}
    relations = {edge["relation"] for edge in graph["edges"]}
    scores = {score["node_id"]: score for score in report.as_dict()["scores"]}

    assert "cell:2" in node_ids
    assert "gene:gene_repair_from_test_failures" in node_ids
    assert "matrix:matrix-evidence-1" in node_ids
    assert "action:cell-2:propose_patch:repair-niche" in node_ids
    assert {"expressed_by", "emitted", "referenced", "validated", "handoff_completed"} <= relations
    assert scores["matrix:matrix-evidence-1"]["positive"] > 0
    assert scores["tool:tool-1"]["negative"] > 0
    assert report.summary["top_cell_id"] == 2


def test_attribution_writes_report_artifacts(tmp_path: Path) -> None:
    report = ContributionAttributionRuntime().analyze(trace=sample_trace(), actions=[sample_action()])

    outputs = report.write(tmp_path)

    assert set(outputs) == {
        "graph",
        "report",
        "cell_contributions",
        "gene_contributions",
        "matrix_contributions",
        "summary",
    }
    assert (tmp_path / "contribution_graph.json").exists()
    assert (tmp_path / "contribution_report.md").read_text(encoding="utf-8").startswith("# Contribution Attribution Report")
    assert "cell_id" in (tmp_path / "cell_contributions.csv").read_text(encoding="utf-8").splitlines()[0]


def test_attribute_cli_reads_artifacts_and_writes_outputs(tmp_path: Path) -> None:
    trace = tmp_path / "tissue_trace.json"
    actions = tmp_path / "action_intents.json"
    summary = tmp_path / "tissue_summary.json"
    output = tmp_path / "attribution"
    trace.write_text(json.dumps(sample_trace()), encoding="utf-8")
    actions.write_text(json.dumps([sample_action()]), encoding="utf-8")
    summary.write_text(json.dumps({"population": 3}), encoding="utf-8")

    main(["attribute", "--trace", str(trace), "--summary", str(summary), "--actions", str(actions), "--output", str(output)])

    assert (output / "contribution_summary.json").exists()
    assert (output / "contribution_graph.json").exists()
    assert json.loads((output / "contribution_summary.json").read_text(encoding="utf-8"))["top_cell_id"] == 2


def test_tissue_cli_with_attribution_updates_summary(tmp_path: Path) -> None:
    output = tmp_path / "tissue"

    main(
        [
            "tissue",
            "--genome-spec",
            "examples/framework/repo_repair_genome.yaml",
            "--environment-spec",
            "examples/framework/failing_tests_environment.yaml",
            "--steps",
            "4",
            "--effector",
            "mock-llm",
            "--with-attribution",
            "--output",
            str(output),
        ]
    )

    summary = json.loads((output / "tissue_summary.json").read_text(encoding="utf-8"))
    assert "attribution" in summary
    assert (output / "attribution" / "contribution_summary.json").exists()


def test_structure_search_with_attribution_writes_variant_reports(tmp_path: Path) -> None:
    output = tmp_path / "search"

    main(
        [
            "structure-search",
            "--task",
            "Fix failing tests while preserving behavior.",
            "--steps",
            "4",
            "--with-attribution",
            "--output",
            str(output),
        ]
    )

    summary = json.loads((output / "structure_search_summary.json").read_text(encoding="utf-8"))
    selected = summary["selected_variant"]
    assert "selected_variant_explanation" in summary
    assert (output / "variants" / selected / "attribution" / "contribution_summary.json").exists()

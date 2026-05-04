from __future__ import annotations

import json
import sys
from pathlib import Path

from ontocellia.__main__ import main
from ontocellia.framework import (
    AgentCell,
    AgentGenome,
    CellPosition,
    ExecutionPolicy,
    ExecutionRuntime,
    Gene,
    MorphogenField,
    Niche,
    TaskMicroenvironment,
    TissueRuntime,
)


def make_tissue(tmp_path: Path) -> TissueRuntime:
    environment = TaskMicroenvironment(
        objective="Execute structured intents safely.",
        morphogens=MorphogenField({"repair_pressure": 1.0}),
        niches=[Niche("repair-niche", "repair", CellPosition("repair-node", "repo"))],
    )
    cells = {
        1: AgentCell(
            1,
            "differentiated",
            "repair",
            CellPosition("repair-node", "repo"),
            niche_id="repair-niche",
            expressed_gene_ids=["gene_repair"],
        )
    }
    return TissueRuntime(
        genome=AgentGenome([Gene("gene_repair", "regeneration", ["repair_pressure"], ["repair"])]),
        environment=environment,
        cells=cells,
    )


def policy(tmp_path: Path, **kwargs: object) -> ExecutionPolicy:
    defaults = {
        "workspace_root": tmp_path,
        "allowed_interfaces": ["workspace.read", "workspace.search", "workspace.apply_patch", "pytest.run", "git.diff", "shell.run"],
        "allowed_commands": [],
        "allowed_write_globs": [],
        "dry_run": True,
        "timeout_seconds": 5.0,
    }
    defaults.update(kwargs)
    return ExecutionPolicy(**defaults)


def test_dry_run_patch_does_not_modify_file(tmp_path: Path) -> None:
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir()
    target.write_text("value = 1\n", encoding="utf-8")
    tissue = make_tissue(tmp_path)
    actions = [
        {
            "cell_id": 1,
            "intent_type": "propose_patch",
            "target": "src/app.py",
            "confidence": 0.8,
            "required_interfaces": ["workspace"],
            "payload": {"patch": "--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"},
        }
    ]

    results = ExecutionRuntime().execute(tissue, actions, policy(tmp_path, allowed_write_globs=["src/**/*.py"]))

    assert target.read_text(encoding="utf-8") == "value = 1\n"
    assert results[0].status == "dry_run"
    assert results[0].passed is False
    assert any(event["type"] == "execution_skipped" and event["status"] == "dry_run" for event in tissue.trace.events)


def test_workspace_read_rejects_path_outside_workspace(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("secret", encoding="utf-8")
    tissue = make_tissue(tmp_path)
    actions = [{"cell_id": 1, "intent_type": "inspect_context", "target": str(outside), "payload": {"path": str(outside)}}]

    results = ExecutionRuntime().execute(tissue, actions, policy(tmp_path, allowed_interfaces=["workspace.read"]))

    assert results[0].status == "failed"
    assert results[0].passed is False
    assert "outside workspace" in results[0].evidence


def test_workspace_search_uses_rg_and_returns_evidence(tmp_path: Path) -> None:
    source = tmp_path / "src" / "module.py"
    source.parent.mkdir()
    source.write_text("def repair_target():\n    return 'needle'\n", encoding="utf-8")
    tissue = make_tissue(tmp_path)
    actions = [{"cell_id": 1, "intent_type": "inspect_context", "target": "needle", "payload": {"query": "needle"}}]

    results = ExecutionRuntime().execute(tissue, actions, policy(tmp_path, allowed_interfaces=["workspace.search"]))

    assert results[0].passed is True
    assert "module.py" in results[0].evidence
    assert "needle" in results[0].evidence


def test_git_diff_reads_without_modifying_workspace(tmp_path: Path) -> None:
    (tmp_path / "tracked.py").write_text("value = 1\n", encoding="utf-8")
    tissue = make_tissue(tmp_path)
    actions = [{"cell_id": 1, "intent_type": "review_output", "target": "repo", "payload": {"command": "git diff -- tracked.py"}}]

    results = ExecutionRuntime().execute(tissue, actions, policy(tmp_path, allowed_interfaces=["git.diff"]))

    assert results[0].status in {"passed", "failed"}
    assert (tmp_path / "tracked.py").read_text(encoding="utf-8") == "value = 1\n"
    assert results[0].request.interface == "git.diff"


def test_allowlisted_pytest_command_runs(tmp_path: Path) -> None:
    test_file = tmp_path / "test_ok.py"
    test_file.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    command = f"{sys.executable} -m pytest -q {test_file.name}"
    tissue = make_tissue(tmp_path)
    actions = [{"cell_id": 1, "intent_type": "review_output", "target": "tests", "payload": {"command": command}}]

    results = ExecutionRuntime().execute(
        tissue,
        actions,
        policy(tmp_path, allowed_interfaces=["pytest.run"], allowed_commands=[command], dry_run=False),
    )

    assert results[0].passed is True
    assert results[0].score == 1.0
    assert "passed" in results[0].evidence.lower()


def test_unallowlisted_command_is_skipped(tmp_path: Path) -> None:
    tissue = make_tissue(tmp_path)
    actions = [{"cell_id": 1, "intent_type": "review_output", "target": "tests", "payload": {"command": f"{sys.executable} -c \"print('run')\""}}]

    results = ExecutionRuntime().execute(tissue, actions, policy(tmp_path, allowed_interfaces=["shell.run"], dry_run=False))

    assert results[0].status == "skipped"
    assert "not allowlisted" in results[0].evidence
    assert any(event["type"] == "execution_skipped" for event in tissue.trace.events)


def test_shell_run_does_not_execute_shell_metacharacters(tmp_path: Path) -> None:
    marker = tmp_path / "marker.txt"
    command = f"{sys.executable} -c \"print('safe')\" ; touch {marker}"
    tissue = make_tissue(tmp_path)
    actions = [{"cell_id": 1, "intent_type": "review_output", "target": "shell", "payload": {"command": command}}]

    results = ExecutionRuntime().execute(
        tissue,
        actions,
        policy(tmp_path, allowed_interfaces=["shell.run"], allowed_commands=[command], dry_run=False),
    )

    assert results[0].passed is True
    assert not marker.exists()


def test_apply_patch_requires_write_allowlist(tmp_path: Path) -> None:
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir()
    target.write_text("value = 1\n", encoding="utf-8")
    patch = "--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
    tissue = make_tissue(tmp_path)
    actions = [{"cell_id": 1, "intent_type": "propose_patch", "target": "src/app.py", "payload": {"patch": patch}}]

    blocked = ExecutionRuntime().execute(
        tissue,
        actions,
        policy(tmp_path, allowed_interfaces=["workspace.apply_patch"], dry_run=False),
    )
    allowed = ExecutionRuntime().execute(
        tissue,
        actions,
        policy(tmp_path, allowed_interfaces=["workspace.apply_patch"], allowed_write_globs=["src/**/*.py"], dry_run=False),
    )

    assert blocked[0].status == "skipped"
    assert allowed[0].passed is True
    assert target.read_text(encoding="utf-8") == "value = 2\n"


def test_execution_result_deposits_matrix_and_validation_feedback(tmp_path: Path) -> None:
    test_file = tmp_path / "test_ok.py"
    test_file.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    command = f"{sys.executable} -m pytest -q {test_file.name}"
    tissue = make_tissue(tmp_path)
    actions = [{"cell_id": 1, "intent_type": "review_output", "target": "tests", "payload": {"command": command}}]

    results = ExecutionRuntime().execute(
        tissue,
        actions,
        policy(tmp_path, allowed_interfaces=["pytest.run"], allowed_commands=[command], dry_run=False),
    )
    tissue.develop(ticks=1, validation_results=[result.to_validation_result() for result in results])

    assert tissue.environment.matrix.records
    assert tissue.environment.matrix.records[0].kind == "execution"
    assert any(event["type"] == "matrix_deposit" for event in tissue.trace.events)
    assert any(event["type"] == "organ_selection" for event in tissue.trace.events)


def test_long_command_output_uses_output_metabolism_artifact(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    code = "import sys; [print(f'head-{i}') for i in range(80)]; print('Traceback: important failure', file=sys.stderr); [print(f'tail-{i}') for i in range(80)]"
    command = f"{sys.executable} -c {code!r}"
    tissue = make_tissue(tmp_path)
    actions = [{"cell_id": 1, "intent_type": "review_output", "target": "shell", "payload": {"command": command}}]

    results = ExecutionRuntime().execute(
        tissue,
        actions,
        policy(
            tmp_path,
            allowed_interfaces=["shell.run"],
            allowed_commands=[command],
            dry_run=False,
            max_output_chars=360,
            artifact_root=artifact_root,
        ),
    )

    assert results[0].passed is True
    assert len(results[0].evidence) <= 360
    assert results[0].output_digest["truncated"] is True
    assert results[0].output_digest["raw_output_path"] == "raw_outputs/tool-0-evidence.txt"
    assert "Traceback: important failure" in results[0].evidence
    assert (artifact_root / "raw_outputs" / "tool-0-evidence.txt").exists()
    record = tissue.environment.matrix.records[-1]
    assert record.metadata["raw_output_path"] == "raw_outputs/tool-0-evidence.txt"
    assert record.metadata["source_result_id"] == "tool-0"


def test_tissue_cli_writes_execution_results_in_dry_run(tmp_path: Path) -> None:
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
            "--execute-actions",
            "--execution-dry-run",
            "--allow-interface",
            "workspace.search",
            "--allow-interface",
            "git.diff",
            "--output",
            str(output),
        ]
    )

    results = json.loads((output / "execution_results.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "tissue_summary.json").read_text(encoding="utf-8"))
    assert results
    assert summary["execution_results"] == len(results)
    assert "changed_files" in summary
    assert summary["raw_outputs"] == 0
    assert summary["truncated_outputs"] == 0


def test_tissue_cli_without_execute_actions_does_not_write_execution_results(tmp_path: Path) -> None:
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
            "--output",
            str(output),
        ]
    )

    assert not (output / "execution_results.json").exists()

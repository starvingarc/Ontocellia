from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ontocellia.__main__ import main
from ontocellia.framework import (
    AgentCell,
    AgentGenome,
    CellPosition,
    ExtracellularInterface,
    ExtracellularToolRuntime,
    Gene,
    MCPInterfaceAdapter,
    MCPServerSpec,
    MCPToolSpec,
    MorphogenField,
    Niche,
    ReceptorProfile,
    TaskMicroenvironment,
    TissueRuntime,
    ToolPolicy,
)


def make_tissue(tmp_path: Path, *, interfaces: list[str] | None = None, receptor: list[str] | None = None) -> TissueRuntime:
    interface_ids = interfaces or [
        "workspace.read",
        "workspace.search",
        "workspace.list",
        "workspace.apply_patch",
        "git.status",
        "git.diff",
        "git.show",
        "git.log",
        "shell.run",
        "pytest.run",
        "http.request",
        "browser.open",
        "mcp:repo:tool:lookup_order",
    ]
    environment = TaskMicroenvironment(
        objective="Use tools safely.",
        morphogens=MorphogenField({"repair_pressure": 1.0}),
        niches=[Niche("repair-niche", "repair", CellPosition("repair-node", "repo"))],
        interfaces=[ExtracellularInterface(interface_id, "membrane_channel", ["repair"]) for interface_id in interface_ids],
    )
    environment.mcp_adapter = MCPInterfaceAdapter(
        [
            MCPServerSpec(
                id="repo",
                tools=[
                    MCPToolSpec(
                        name="lookup_order",
                        description="Look up an order.",
                        accepts_fates=["repair"],
                        metadata={"mock_result": "order is eligible", "tags": ["order", "lookup"]},
                    )
                ],
            )
        ]
    )
    cell = AgentCell(
        1,
        "differentiated",
        "repair",
        CellPosition("repair-node", "repo"),
        niche_id="repair-niche",
        expressed_gene_ids=["gene_repair"],
        receptor=ReceptorProfile(accepted_interfaces=list(receptor or interface_ids)),
    )
    return TissueRuntime(
        genome=AgentGenome([Gene("gene_repair", "regeneration", ["repair_pressure"], ["repair"])]),
        environment=environment,
        cells={1: cell},
    )


def policy(tmp_path: Path, **kwargs: object) -> ToolPolicy:
    defaults = {
        "workspace_root": tmp_path,
        "allowed_interfaces": [],
        "allowed_commands": [],
        "allowed_write_globs": [],
        "allowed_network_hosts": [],
        "allowed_mcp_tools": [],
        "dry_run": True,
        "timeout_seconds": 5.0,
    }
    defaults.update(kwargs)
    return ToolPolicy(**defaults)


def test_tool_runtime_requires_receptor_environment_and_policy_gates(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    tissue = make_tissue(tmp_path, interfaces=["workspace.read"], receptor=["workspace.read"])
    actions = [
        {"cell_id": 1, "intent_type": "inspect_context", "target": "note.txt", "payload": {"interface": "workspace.read", "path": "note.txt"}},
        {"cell_id": 1, "intent_type": "review_output", "target": "repo", "payload": {"interface": "git.diff"}},
    ]

    results = ExtracellularToolRuntime().execute(
        tissue,
        actions,
        policy(tmp_path, allowed_interfaces=["workspace.read", "git.diff"]),
    )

    assert results[0].passed is True
    assert results[1].status == "skipped"
    assert "not available" in results[1].evidence or "not accept" in results[1].evidence
    assert any(event["type"] == "tool_invocation_skipped" for event in tissue.trace.events)


def test_workspace_list_and_read_reject_paths_outside_workspace(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("value = 1\n", encoding="utf-8")
    outside = tmp_path.parent / "outside-tool-runtime.txt"
    outside.write_text("secret", encoding="utf-8")
    tissue = make_tissue(tmp_path, interfaces=["workspace.list", "workspace.read"])

    results = ExtracellularToolRuntime().execute(
        tissue,
        [
            {"cell_id": 1, "intent_type": "inspect_context", "target": "src", "payload": {"interface": "workspace.list", "path": "src"}},
            {"cell_id": 1, "intent_type": "inspect_context", "target": str(outside), "payload": {"interface": "workspace.read", "path": str(outside)}},
        ],
        policy(tmp_path, allowed_interfaces=["workspace.list", "workspace.read"]),
    )

    assert results[0].passed is True
    assert "app.py" in results[0].evidence
    assert results[1].passed is False
    assert "outside workspace" in results[1].evidence


def test_patch_requires_non_dry_run_and_write_allowlist(tmp_path: Path) -> None:
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir()
    target.write_text("value = 1\n", encoding="utf-8")
    patch = "--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
    tissue = make_tissue(tmp_path, interfaces=["workspace.apply_patch"])
    action = {"cell_id": 1, "intent_type": "propose_patch", "target": "src/app.py", "payload": {"patch": patch}}

    dry = ExtracellularToolRuntime().execute(
        tissue,
        [action],
        policy(tmp_path, allowed_interfaces=["workspace.apply_patch"], allowed_write_globs=["src/**/*.py"], dry_run=True),
    )
    blocked = ExtracellularToolRuntime().execute(
        tissue,
        [action],
        policy(tmp_path, allowed_interfaces=["workspace.apply_patch"], dry_run=False),
    )
    allowed = ExtracellularToolRuntime().execute(
        tissue,
        [action],
        policy(tmp_path, allowed_interfaces=["workspace.apply_patch"], allowed_write_globs=["src/**/*.py"], dry_run=False),
    )

    assert dry[0].status == "dry_run"
    assert blocked[0].status == "skipped"
    assert allowed[0].passed is True
    assert target.read_text(encoding="utf-8") == "value = 2\n"


def test_git_read_adapters_are_read_only(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / "tracked.py").write_text("value = 2\n", encoding="utf-8")
    tissue = make_tissue(tmp_path, interfaces=["git.status", "git.diff", "git.show", "git.log"])

    results = ExtracellularToolRuntime().execute(
        tissue,
        [
            {"cell_id": 1, "intent_type": "review_output", "target": "repo", "payload": {"interface": "git.status"}},
            {"cell_id": 1, "intent_type": "review_output", "target": "tracked.py", "payload": {"interface": "git.diff", "path": "tracked.py"}},
            {"cell_id": 1, "intent_type": "review_output", "target": "HEAD:tracked.py", "payload": {"interface": "git.show", "target": "HEAD:tracked.py"}},
            {"cell_id": 1, "intent_type": "review_output", "target": "repo", "payload": {"interface": "git.log"}},
        ],
        policy(tmp_path, allowed_interfaces=["git.status", "git.diff", "git.show", "git.log"]),
    )

    assert all(result.status in {"passed", "failed"} for result in results)
    assert (tmp_path / "tracked.py").read_text(encoding="utf-8") == "value = 2\n"
    assert "modified" in results[0].evidence.lower() or "tracked.py" in results[0].evidence


def test_shell_command_requires_exact_allowlist_and_no_shell_execution(tmp_path: Path) -> None:
    marker = tmp_path / "marker.txt"
    command = f"{sys.executable} -c \"print('safe')\" ; touch {marker}"
    tissue = make_tissue(tmp_path, interfaces=["shell.run"])
    action = {"cell_id": 1, "intent_type": "review_output", "target": "shell", "payload": {"interface": "shell.run", "command": command}}

    blocked = ExtracellularToolRuntime().execute(tissue, [action], policy(tmp_path, allowed_interfaces=["shell.run"], dry_run=False))
    allowed = ExtracellularToolRuntime().execute(
        tissue,
        [action],
        policy(tmp_path, allowed_interfaces=["shell.run"], allowed_commands=[command], dry_run=False),
    )

    assert blocked[0].status == "skipped"
    assert allowed[0].passed is True
    assert not marker.exists()


def test_mcp_tool_invocation_requires_declared_and_allowlisted_tool(tmp_path: Path) -> None:
    tissue = make_tissue(tmp_path, interfaces=["mcp:repo:tool:lookup_order"])
    allowed = {
        "cell_id": 1,
        "intent_type": "tool_call",
        "target": "order-1",
        "payload": {"interface": "mcp:repo:tool:lookup_order", "arguments": {"order_id": "order-1"}},
    }
    undeclared = {
        "cell_id": 1,
        "intent_type": "tool_call",
        "target": "order-1",
        "payload": {"interface": "mcp:repo:tool:update_order", "arguments": {"order_id": "order-1"}},
    }

    results = ExtracellularToolRuntime().execute(
        tissue,
        [allowed, undeclared],
        policy(tmp_path, allowed_interfaces=["mcp:repo:tool:lookup_order", "mcp:repo:tool:update_order"], allowed_mcp_tools=["mcp:repo:tool:lookup_order"]),
    )

    assert results[0].passed is True
    assert "order is eligible" in results[0].evidence
    assert results[1].status == "skipped"


def test_http_and_browser_adapters_are_disabled_until_explicitly_enabled(tmp_path: Path) -> None:
    tissue = make_tissue(tmp_path, interfaces=["http.request", "browser.open"])
    actions = [
        {"cell_id": 1, "intent_type": "tool_call", "target": "https://example.com", "payload": {"interface": "http.request", "url": "https://example.com"}},
        {"cell_id": 1, "intent_type": "tool_call", "target": "https://example.com", "payload": {"interface": "browser.open", "url": "https://example.com"}},
    ]

    results = ExtracellularToolRuntime().execute(
        tissue,
        actions,
        policy(tmp_path, allowed_interfaces=["http.request", "browser.open"], allowed_network_hosts=["example.com"]),
    )

    assert results[0].status == "skipped"
    assert "HTTP tools are disabled" in results[0].evidence
    assert results[1].status == "skipped"
    assert "Browser tools are disabled" in results[1].evidence


def test_tool_results_deposit_matrix_context_and_emit_pressure(tmp_path: Path) -> None:
    tissue = make_tissue(tmp_path, interfaces=["http.request"])
    action = {
        "cell_id": 1,
        "intent_type": "tool_call",
        "target": "https://blocked.example",
        "payload": {"interface": "http.request", "url": "https://blocked.example", "context_record_ids": ["ctx-1"]},
    }

    results = ExtracellularToolRuntime().execute(tissue, [action], policy(tmp_path, allowed_interfaces=["http.request"]))

    record = tissue.environment.matrix.records[-1]
    assert results[0].passed is False
    assert record.references == [results[0].invocation.id, "ctx-1"]
    assert record.validation_status == "failed"
    assert tissue.environment.morphogens.signal("risk_pressure") > 0
    assert tissue.environment.morphogens.signal("repair_pressure") > 1.0


def test_tissue_cli_writes_tool_runtime_artifacts(tmp_path: Path) -> None:
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

    summary = json.loads((output / "tissue_summary.json").read_text(encoding="utf-8"))
    tool_invocations = json.loads((output / "tool_invocations.json").read_text(encoding="utf-8"))
    tool_results = json.loads((output / "tool_results.json").read_text(encoding="utf-8"))
    execution_results = json.loads((output / "execution_results.json").read_text(encoding="utf-8"))

    assert tool_invocations
    assert tool_results
    assert execution_results
    assert summary["tool_results"] == len(tool_results)
    assert "blocked_tool_invocations" in summary

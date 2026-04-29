from __future__ import annotations

import json
from pathlib import Path

from ontocellia.__main__ import main
from ontocellia.framework import (
    MCPInterfaceAdapter,
    MCPPromptSpec,
    MCPResourceSpec,
    MCPServerSpec,
    MCPToolResult,
    MCPToolSpec,
    load_task_microenvironment,
)


def test_mcp_tool_maps_to_membrane_channel_interface() -> None:
    server = MCPServerSpec(
        id="repo",
        tools=[
            MCPToolSpec(
                name="read_file",
                description="Read a workspace file.",
                input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
                accepts_fates=["explorer", "repair"],
            )
        ],
    )

    interfaces = MCPInterfaceAdapter([server]).interfaces()

    assert len(interfaces) == 1
    assert interfaces[0].id == "mcp:repo:tool:read_file"
    assert interfaces[0].kind == "membrane_channel"
    assert interfaces[0].accepts_fates == ["explorer", "repair"]
    assert interfaces[0].metadata["mcp"]["server_id"] == "repo"
    assert interfaces[0].metadata["mcp"]["tool_name"] == "read_file"
    assert interfaces[0].metadata["receptor_schema"]["type"] == "object"


def test_mcp_resource_maps_to_extracellular_matrix_record() -> None:
    server = MCPServerSpec(
        id="repo",
        resources=[
            MCPResourceSpec(
                id="failing-log",
                uri="file://pytest.log",
                content="3 failing tests",
                tags=["test_failure", "repo"],
                position={"node_id": "repair-niche", "region": "repo/tests"},
                confidence=0.9,
            )
        ],
    )

    records = MCPInterfaceAdapter([server]).matrix_records()

    assert len(records) == 1
    assert records[0].id == "mcp:repo:resource:failing-log"
    assert records[0].kind == "mcp_resource"
    assert records[0].tags == ["test_failure", "repo"]
    assert records[0].position.node_id == "repair-niche"


def test_mcp_prompt_maps_to_induction_factor_interface_metadata() -> None:
    server = MCPServerSpec(
        id="repo",
        prompts=[MCPPromptSpec(id="repair-protocol", template="Inspect failure, patch narrow, validate.", tags=["repair"])],
    )

    interfaces = MCPInterfaceAdapter([server]).interfaces()

    assert interfaces[0].id == "mcp:repo:prompt:repair-protocol"
    assert interfaces[0].kind == "induction_factor"
    assert interfaces[0].metadata["mcp"]["prompt_id"] == "repair-protocol"
    assert interfaces[0].metadata["template"] == "Inspect failure, patch narrow, validate."


def test_mcp_tool_result_deposits_matrix_record_and_emits_morphogens() -> None:
    server = MCPServerSpec(id="repo")
    adapter = MCPInterfaceAdapter([server])
    environment = load_task_microenvironment("examples/framework/failing_tests_environment.yaml")
    result = MCPToolResult(
        server_id="repo",
        tool_name="pytest",
        content="2 tests still failing",
        tags=["validation", "test_failure"],
        morphogens={"test_failure": 0.4, "repair_pressure": 0.2},
        position={"node_id": "repair-niche", "region": "repo/tests"},
        confidence=0.8,
    )

    adapter.apply_tool_result(environment, result)

    records = environment.matrix.query(tags=["validation", "test_failure"], limit=5)
    assert len(records) == 1
    assert records[0].content == "2 tests still failing"
    assert environment.morphogens.signal("test_failure") > 0.9
    assert environment.morphogens.signal("repair_pressure") > 0.7


def test_environment_loader_accepts_mcp_section(tmp_path: Path) -> None:
    spec = tmp_path / "environment.yaml"
    spec.write_text(
        """
task:
  objective: Inspect repository.
mcp:
  servers:
    - id: repo
      tools:
        - name: read_file
          description: Read a workspace file.
          accepts_fates: [explorer]
          input_schema:
            type: object
      resources:
        - id: failing-log
          uri: file://pytest.log
          content: 3 failing tests
          tags: [test_failure]
          position:
            node_id: repair-niche
      prompts:
        - id: repair-protocol
          template: Inspect failure, patch narrow, validate.
          tags: [repair]
niches:
  - id: repair-niche
    required_fate: repair
    position:
      node_id: repair-niche
""",
        encoding="utf-8",
    )

    environment = load_task_microenvironment(spec)

    assert "mcp:repo:tool:read_file" in {interface.id for interface in environment.interfaces}
    assert "mcp:repo:prompt:repair-protocol" in {interface.id for interface in environment.interfaces}
    assert environment.matrix.query(tags=["test_failure"], limit=5)[0].id == "mcp:repo:resource:failing-log"


def test_tissue_cli_writes_mcp_summary_when_spec_contains_mcp(tmp_path: Path) -> None:
    genome = tmp_path / "genome.yaml"
    environment = tmp_path / "environment.yaml"
    output = tmp_path / "out"
    genome.write_text(
        """
genes:
  - id: gene_inspect
    category: exploration
    morphogen_affinity: [ambiguity]
    encoded_response: [inspect]
""",
        encoding="utf-8",
    )
    environment.write_text(
        """
task:
  objective: Inspect repository.
morphogens:
  ambiguity: 1.0
mcp:
  servers:
    - id: repo
      tools:
        - name: read_file
          accepts_fates: [explorer]
niches:
  - id: exploration-front
    required_fate: explorer
    position:
      node_id: exploration-front
interfaces: []
""",
        encoding="utf-8",
    )

    main(
        [
            "tissue",
            "--genome-spec",
            str(genome),
            "--environment-spec",
            str(environment),
            "--steps",
            "1",
            "--output",
            str(output),
        ]
    )

    summary = json.loads((output / "tissue_summary.json").read_text(encoding="utf-8"))
    assert summary["mcp_interfaces"] == 1

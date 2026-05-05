from __future__ import annotations

from pathlib import Path

from ontocellia.framework.official_benchmark import (
    OfficialBenchmarkAdapter,
    load_swe_bench_task_from_row,
    load_tau_bench_tasks_from_text,
    load_terminal_bench_task_from_yaml,
)


def test_swe_bench_row_maps_to_adaptive_task() -> None:
    task = load_swe_bench_task_from_row(
        {
            "instance_id": "astropy__astropy-12907",
            "repo": "astropy/astropy",
            "base_commit": "abc123",
            "problem_statement": "Fix nested separability_matrix behavior.",
            "FAIL_TO_PASS": ["test_separable.py::test_nested"],
            "PASS_TO_PASS": ["test_existing.py::test_still_passes"],
        }
    )

    assert task.id == "astropy__astropy-12907"
    assert task.source_benchmark == "swe-bench-lite"
    assert task.metadata["repo"] == "astropy/astropy"
    assert task.metadata["tests"] == ["test_separable.py::test_nested"]
    assert task.metadata["official_dataset"] == "princeton-nlp/SWE-bench_Lite"


def test_terminal_bench_yaml_maps_instruction_and_metadata(tmp_path: Path) -> None:
    task_yaml = tmp_path / "task.yaml"
    task_yaml.write_text(
        """
instruction: |-
  Aggregate JSONL files and write aggregates.json.
difficulty: easy
category: file-operations
tags:
  - data-processing
parser_name: pytest
""",
        encoding="utf-8",
    )

    task = load_terminal_bench_task_from_yaml(task_yaml, task_id="jsonl-aggregator")

    assert task.id == "jsonl-aggregator"
    assert task.source_benchmark == "terminal-bench"
    assert "Aggregate JSONL" in task.prompt
    assert task.metadata["difficulty"] == "easy"
    assert task.metadata["category"] == "file-operations"
    assert task.metadata["tags"] == ["data-processing"]
    assert task.metadata["check_command"] == "official terminal-bench parser: pytest"


def test_tau_bench_task_text_maps_instruction_and_expected_actions() -> None:
    text = """
from tau_bench.types import Action, Task

TASKS = [
    Task(
        annotator="0",
        user_id="mia_li_3668",
        instruction="Book the cheapest eligible flight without insurance.",
        actions=[
            Action(name="book_reservation", kwargs={"user_id": "mia_li_3668"}),
            Action(name="update_reservation_baggages", kwargs={"reservation_id": "ABC123"}),
        ],
        outputs=["done"],
    )
]
"""

    tasks = load_tau_bench_tasks_from_text(text, benchmark_id="tau-bench", tau_domain="airline", limit=1)

    assert len(tasks) == 1
    assert tasks[0].id == "airline-test-0"
    assert tasks[0].source_benchmark == "tau-bench"
    assert tasks[0].metadata["user_id"] == "mia_li_3668"
    assert tasks[0].metadata["expected_action_names"] == ["book_reservation", "update_reservation_baggages"]
    assert tasks[0].metadata["official_repo"] == "sierra-research/tau-bench"


def test_non_dry_terminal_adapter_loads_official_source_dir(tmp_path: Path) -> None:
    task_dir = tmp_path / "original-tasks" / "jsonl-aggregator"
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text("instruction: collect records\nparser_name: pytest\n", encoding="utf-8")

    tasks = OfficialBenchmarkAdapter.for_benchmark("terminal-bench").load_tasks(
        limit=1,
        dry_run=False,
        source_dir=tmp_path,
    )

    assert [task.id for task in tasks] == ["jsonl-aggregator"]
    assert tasks[0].prompt == "collect records"


def test_terminal_software_engineering_task_induces_repo_repair_tissue() -> None:
    task = load_terminal_bench_task_from_yaml(
        _terminal_task_yaml(
            """
instruction: Modernize a legacy library, build it, and fix compatibility issues.
category: software-engineering
tags:
  - coding
  - legacy-modernization
parser_name: pytest
"""
        ),
        task_id="3d-model-format-legacy",
    )
    adapter = OfficialBenchmarkAdapter.for_benchmark("terminal-bench")

    request = adapter.to_induction_request(task)
    environment = adapter.to_microenvironment(task)

    assert request.domain == "repo_repair"
    assert environment.morphogens.signal("repair_pressure") > 0
    assert environment.morphogens.signal("test_failure") > 0
    assert any(niche.required_fate == "repair" for niche in environment.niches)


def test_terminal_data_processing_task_stays_generic() -> None:
    task = load_terminal_bench_task_from_yaml(
        _terminal_task_yaml(
            """
instruction: Aggregate JSONL files and write aggregates.json.
category: file-operations
tags:
  - data-processing
parser_name: pytest
"""
        ),
        task_id="jsonl-aggregator",
    )
    adapter = OfficialBenchmarkAdapter.for_benchmark("terminal-bench")

    request = adapter.to_induction_request(task)
    environment = adapter.to_microenvironment(task)

    assert request.domain == "generic"
    assert environment.morphogens.signal("repair_pressure") == 0.0
    assert not any(niche.required_fate == "repair" for niche in environment.niches)


def _terminal_task_yaml(content: str) -> Path:
    import tempfile

    directory = Path(tempfile.mkdtemp())
    path = directory / "task.yaml"
    path.write_text(content, encoding="utf-8")
    return path

from __future__ import annotations

import json
from pathlib import Path

from ontocellia.__main__ import main
from ontocellia.framework import run_repo_repair_demo


def test_repo_repair_demo_writes_end_to_end_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "demo"

    result = run_repo_repair_demo(output=output, task="Fix failing tests while preserving behavior.", steps=4, seed=7)

    assert result.summary_path == output / "demo_summary.json"
    assert result.report_path == output / "demo_report.md"
    assert result.summary_path.exists()
    assert result.report_path.exists()
    assert (output / "induction" / "genome.yaml").exists()
    assert (output / "induction" / "environment.yaml").exists()
    assert (output / "tissue" / "tissue_summary.json").exists()
    assert (output / "tissue" / "tissue_trace.json").exists()
    assert (output / "tissue" / "action_intents.json").exists()
    assert (output / "tissue" / "llm_trace.json").exists()
    assert (output / "validation" / "baseline_validation.json").exists()
    assert (output / "validation" / "candidate_validation.json").exists()
    assert (output / "mutation" / "mutation_candidates.json").exists()
    assert (output / "mutation" / "solidified_genome.yaml").exists()

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["phase"] == "complete_repo_repair_demo"
    assert summary["induction"]["domain"] == "repo_repair"
    assert summary["tissue"]["actions"] > 0
    assert summary["tissue"]["messages"] > 0
    assert summary["tissue"]["matrix_records"] > 0
    assert summary["validation"]["baseline_score"] < summary["validation"]["candidate_score"]
    assert summary["mutation"]["decision"] == "selected"


def test_demo_cli_writes_summary_and_report(tmp_path: Path) -> None:
    output = tmp_path / "demo_cli"

    main(
        [
            "demo",
            "--task",
            "Fix failing tests while preserving behavior.",
            "--steps",
            "4",
            "--output",
            str(output),
        ]
    )

    summary = json.loads((output / "demo_summary.json").read_text(encoding="utf-8"))
    assert summary["phase"] == "complete_repo_repair_demo"
    assert (output / "demo_report.md").exists()


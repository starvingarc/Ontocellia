from __future__ import annotations

import sys
from pathlib import Path

import json
import yaml

from ontocellia.__main__ import main
from ontocellia.framework import ValidationHookPolicy, ValidationHookRequest, ValidationHookRunner


def test_allowlisted_hook_success_generates_passed_result() -> None:
    command = f"{sys.executable} -c \"print('ok')\""
    runner = ValidationHookRunner()

    results = runner.run(
        [ValidationHookRequest(name="pytest", command=command)],
        ValidationHookPolicy(allowed_commands=[command]),
    )

    assert len(results) == 1
    assert results[0].name == "pytest"
    assert results[0].passed is True
    assert results[0].score == 1.0
    assert "ok" in results[0].evidence


def test_failing_hook_generates_failed_result_with_output() -> None:
    command = f"{sys.executable} -c \"import sys; print('bad'); sys.exit(3)\""

    results = ValidationHookRunner().run(
        [ValidationHookRequest(name="failure", command=command)],
        ValidationHookPolicy(allowed_commands=[command]),
    )

    assert results[0].passed is False
    assert results[0].score == 0.0
    assert "exit code 3" in results[0].evidence
    assert "bad" in results[0].evidence
    assert results[0].risk >= 0.4


def test_unallowlisted_hook_is_skipped_without_execution(tmp_path: Path) -> None:
    marker = tmp_path / "marker.txt"
    command = f"{sys.executable} -c \"from pathlib import Path; Path({str(marker)!r}).write_text('ran')\""

    results = ValidationHookRunner().run(
        [ValidationHookRequest(name="unsafe", command=command)],
        ValidationHookPolicy(allowed_commands=[]),
    )

    assert results[0].passed is False
    assert results[0].score == 0.0
    assert results[0].risk == 0.2
    assert "not allowlisted" in results[0].evidence
    assert not marker.exists()


def test_timeout_generates_failed_result() -> None:
    command = f"{sys.executable} -c \"import time; time.sleep(2)\""

    results = ValidationHookRunner().run(
        [ValidationHookRequest(name="slow", command=command)],
        ValidationHookPolicy(allowed_commands=[command], timeout_seconds=0.1),
    )

    assert results[0].passed is False
    assert results[0].score == 0.0
    assert "timed out" in results[0].evidence
    assert results[0].latency > 0.0


def test_runner_does_not_execute_shell_metacharacters(tmp_path: Path) -> None:
    marker = tmp_path / "shell-marker.txt"
    command = f"{sys.executable} -c \"print('safe')\" ; touch {marker}"

    results = ValidationHookRunner().run(
        [ValidationHookRequest(name="metachar", command=command)],
        ValidationHookPolicy(allowed_commands=[command]),
    )

    assert results[0].passed is True
    assert not marker.exists()


def test_tissue_cli_does_not_run_validation_hooks_by_default(tmp_path: Path) -> None:
    command = f"{sys.executable} -c \"print('validation ok')\""
    genome_path, environment_path = _write_specs(tmp_path, command)
    output = tmp_path / "default"

    main(
        [
            "tissue",
            "--genome-spec",
            str(genome_path),
            "--environment-spec",
            str(environment_path),
            "--steps",
            "1",
            "--output",
            str(output),
        ]
    )

    summary = json.loads((output / "tissue_summary.json").read_text(encoding="utf-8"))
    assert summary["validation_results"] == 0
    assert not (output / "validation_results.json").exists()


def test_tissue_cli_runs_allowlisted_validation_hooks(tmp_path: Path) -> None:
    command = f"{sys.executable} -c \"print('validation ok')\""
    genome_path, environment_path = _write_specs(tmp_path, command)
    output = tmp_path / "runner"

    main(
        [
            "tissue",
            "--genome-spec",
            str(genome_path),
            "--environment-spec",
            str(environment_path),
            "--steps",
            "1",
            "--run-validation-hooks",
            "--allow-validation-hook",
            command,
            "--output",
            str(output),
        ]
    )

    results = json.loads((output / "validation_results.json").read_text(encoding="utf-8"))
    trace = json.loads((output / "tissue_trace.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "tissue_summary.json").read_text(encoding="utf-8"))
    assert len(results) == 1
    assert results[0]["passed"] is True
    assert "validation ok" in results[0]["evidence"]
    assert summary["validation_results"] == 1
    assert any(event["type"] == "validation_hook_started" for event in trace)
    assert any(event["type"] == "validation_hook_completed" for event in trace)


def test_tissue_cli_merges_external_and_runner_validation_results(tmp_path: Path) -> None:
    command = f"{sys.executable} -c \"print('validation ok')\""
    genome_path, environment_path = _write_specs(tmp_path, command)
    external = tmp_path / "external.json"
    external.write_text(
        json.dumps(
            [
                {
                    "name": "external",
                    "passed": False,
                    "score": 0.2,
                    "target": "repo",
                    "evidence": "external failure",
                    "cost": 0.1,
                    "risk": 0.3,
                    "latency": 0.2,
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "merged"

    main(
        [
            "tissue",
            "--genome-spec",
            str(genome_path),
            "--environment-spec",
            str(environment_path),
            "--steps",
            "1",
            "--validation-result",
            str(external),
            "--run-validation-hooks",
            "--allow-validation-hook",
            command,
            "--output",
            str(output),
        ]
    )

    results = json.loads((output / "validation_results.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "tissue_summary.json").read_text(encoding="utf-8"))
    assert [result["name"] for result in results] == ["external", "gene_validate"]
    assert summary["validation_results"] == 2
    assert summary["organ_selection"]["validation_results"][0]["name"] == "external"


def _write_specs(tmp_path: Path, command: str) -> tuple[Path, Path]:
    genome_path = tmp_path / "genome.yaml"
    environment_path = tmp_path / "environment.yaml"
    genome_path.write_text(
        yaml.safe_dump(
            {
                "genes": [
                    {
                        "id": "gene_validate",
                        "category": "regeneration",
                        "morphogen_affinity": ["repair_pressure"],
                        "encoded_response": ["repair"],
                        "validation_hooks": [command],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    environment_path.write_text(
        yaml.safe_dump(
            {
                "task": {"objective": "Validate hook runner."},
                "morphogens": {"repair_pressure": 1.0},
                "niches": [
                    {
                        "id": "repair-niche",
                        "required_fate": "repair",
                        "position": {"node_id": "repair-niche"},
                        "demand": 1,
                    }
                ],
                "interfaces": [
                    {
                        "id": "workspace",
                        "kind": "extracellular_matrix",
                        "accepts_fates": ["repair"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return genome_path, environment_path

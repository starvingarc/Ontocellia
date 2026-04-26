from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GENOME_PATH = ROOT / "examples" / "specs" / "minimal_genome.yaml"
ENVIRONMENT_PATH = ROOT / "examples" / "specs" / "minimal_environment.yaml"


def run_cli(args: list[str], cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "ontocellia", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


def test_run_subcommand_writes_summary(tmp_path: Path) -> None:
    output = tmp_path / "run"
    result = run_cli(
        [
            "run",
            "--steps",
            "4",
            "--genome-spec",
            str(GENOME_PATH),
            "--environment-spec",
            str(ENVIRONMENT_PATH),
            "--output",
            str(output),
        ]
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "spec"
    assert summary["tick"] == 4


def test_experiment_subcommand_writes_comparison_report(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        yaml.safe_dump(
            {
                "metadata": {"name": "cli-experiment"},
                "base": {
                    "genome": str(GENOME_PATH),
                    "environment": str(ENVIRONMENT_PATH),
                    "steps": 4,
                    "seed": 5,
                },
                "variants": [
                    {"name": "baseline"},
                    {"name": "no_contact", "genome_patch": {"contact_programs": []}},
                ],
                "outputs": {"summary": True, "plots": False, "metrics_csv": True, "report": True},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "experiment"

    result = run_cli(["experiment", "--experiment-spec", str(experiment_path), "--output", str(output)])

    assert result.returncode == 0, result.stderr
    assert (output / "comparison.json").exists()
    assert (output / "report.md").exists()


def test_validate_subcommand_success_and_failure(tmp_path: Path) -> None:
    valid = run_cli(["validate", "--genome-spec", str(GENOME_PATH), "--environment-spec", str(ENVIRONMENT_PATH)])
    assert valid.returncode == 0, valid.stderr
    assert "Validation passed" in valid.stdout

    invalid_genome = tmp_path / "invalid_genome.yaml"
    invalid_genome.write_text(yaml.safe_dump({"metadata": {"name": "broken"}}), encoding="utf-8")
    invalid = run_cli(["validate", "--genome-spec", str(invalid_genome), "--environment-spec", str(ENVIRONMENT_PATH)])

    assert invalid.returncode != 0
    assert "state_dims" in invalid.stderr


def test_legacy_cli_arguments_still_work(tmp_path: Path) -> None:
    output = tmp_path / "legacy_args"
    result = run_cli(["--steps", "4", "--output", str(output)])

    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "legacy"
    assert summary["tick"] == 4

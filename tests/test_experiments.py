from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from ontocellia.experiments import ExperimentRunner, load_experiment_spec
from ontocellia.specs import validate_experiment_spec


ROOT = Path(__file__).resolve().parents[1]
GENOME_PATH = ROOT / "examples" / "specs" / "minimal_genome.yaml"
ENVIRONMENT_PATH = ROOT / "examples" / "specs" / "minimal_environment.yaml"


def write_experiment(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "metadata": {"name": "contact-ablation-test"},
                "base": {
                    "genome": str(GENOME_PATH),
                    "environment": str(ENVIRONMENT_PATH),
                    "steps": 6,
                    "seed": 31,
                },
                "variants": [
                    {"name": "baseline"},
                    {
                        "name": "no_contact",
                        "genome_patch": {"contact_programs": []},
                    },
                ],
                "outputs": {"summary": True, "plots": False, "metrics_csv": True, "report": True},
            }
        ),
        encoding="utf-8",
    )


def test_experiment_spec_loads_and_resolves_paths(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiment.yaml"
    write_experiment(experiment_path)

    spec = load_experiment_spec(experiment_path)

    assert spec.metadata.name == "contact-ablation-test"
    assert spec.base.steps == 6
    assert spec.base.seed == 31
    assert spec.variants[1].genome_patch == {"contact_programs": []}
    assert spec.resolved_genome_path.exists()
    assert spec.resolved_environment_path.exists()


def test_example_experiment_specs_validate() -> None:
    example_dir = ROOT / "examples" / "experiments"

    errors = []
    for path in sorted(example_dir.glob("*.yaml")):
        errors.extend(validate_experiment_spec(path))

    assert errors == []


def test_validate_experiment_reports_patch_type_errors(tmp_path: Path) -> None:
    experiment_path = tmp_path / "bad_experiment.yaml"
    experiment_path.write_text(
        yaml.safe_dump(
            {
                "metadata": {"name": "bad"},
                "base": {"genome": str(GENOME_PATH), "environment": str(ENVIRONMENT_PATH), "steps": 4},
                "variants": [{"name": "broken", "genome_patch": []}],
            }
        ),
        encoding="utf-8",
    )

    errors = validate_experiment_spec(experiment_path)

    assert errors
    assert "variants[0].genome_patch" in errors[0]
    assert "mapping" in errors[0]


def test_experiment_runner_writes_variant_and_comparison_artifacts(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiment.yaml"
    write_experiment(experiment_path)

    result = ExperimentRunner.from_spec_file(experiment_path).run(tmp_path / "artifacts")

    assert [run.name for run in result.runs] == ["baseline", "no_contact"]
    for variant in ("baseline", "no_contact"):
        run_dir = tmp_path / "artifacts" / "runs" / variant
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "metrics.csv").exists()
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["mode"] == "spec"
        assert summary["tick"] == 6

    comparison = json.loads((tmp_path / "artifacts" / "comparison.json").read_text(encoding="utf-8"))
    assert {row["variant"] for row in comparison["runs"]} == {"baseline", "no_contact"}
    for row in comparison["runs"]:
        assert "population" in row
        assert "development_diversity" in row
        assert "repair_activation" in row
        assert "risk_exposure" in row
        assert "phenotypes" in row

    with (tmp_path / "artifacts" / "comparison.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["variant"] for row in rows} == {"baseline", "no_contact"}
    assert (tmp_path / "artifacts" / "comparison.png").exists()
    report = (tmp_path / "artifacts" / "report.md").read_text(encoding="utf-8")
    assert "# Experiment Report: contact-ablation-test" in report
    assert "baseline" in report
    assert "no_contact" in report

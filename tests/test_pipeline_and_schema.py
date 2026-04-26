from __future__ import annotations

from pathlib import Path

from ontocellia import OntocelliaConfig, OntocelliaRuntime
from ontocellia.scheduler.pipeline import StepPipeline
from ontocellia.specs import export_schema_docs


ROOT = Path(__file__).resolve().parents[1]
GENOME_PATH = ROOT / "examples" / "specs" / "minimal_genome.yaml"
ENVIRONMENT_PATH = ROOT / "examples" / "specs" / "minimal_environment.yaml"


def test_runtime_uses_step_pipeline_for_legacy_and_spec_modes() -> None:
    legacy = OntocelliaRuntime(OntocelliaConfig(seed=3))
    spec = OntocelliaRuntime.from_spec_files(GENOME_PATH, ENVIRONMENT_PATH, sim_config=OntocelliaConfig(seed=3))

    assert isinstance(legacy.pipeline, StepPipeline)
    assert isinstance(spec.pipeline, StepPipeline)

    legacy.step(2)
    spec.step(2)

    assert legacy.metrics.history[-1]["mode"] == "legacy"
    assert spec.metrics.history[-1]["mode"] == "spec"


def test_empty_population_pipeline_records_metrics() -> None:
    runtime = OntocelliaRuntime(OntocelliaConfig(seed=10))
    runtime.cells.clear()

    runtime.step(3)

    assert runtime.tick_count == 3
    assert runtime.metrics.history[-1]["population"] == 0


def test_schema_docs_export_contains_core_specs(tmp_path: Path) -> None:
    paths = export_schema_docs(tmp_path)

    assert (tmp_path / "genome-spec.md").exists()
    assert (tmp_path / "environment-spec.md").exists()
    assert (tmp_path / "experiment-spec.md").exists()
    assert "GenomeSpec" in (tmp_path / "genome-spec.md").read_text(encoding="utf-8")
    assert "EnvironmentSpec" in (tmp_path / "environment-spec.md").read_text(encoding="utf-8")
    assert "ExperimentSpec" in (tmp_path / "experiment-spec.md").read_text(encoding="utf-8")
    assert len(paths) == 3

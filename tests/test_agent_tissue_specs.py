from __future__ import annotations

from pathlib import Path

from ontocellia.framework import TissueRuntime, load_agent_genome, load_task_microenvironment


def test_agent_tissue_specs_load_into_runtime() -> None:
    genome = load_agent_genome(Path("examples/framework/repo_repair_genome.yaml"))
    environment = load_task_microenvironment(Path("examples/framework/failing_tests_environment.yaml"))

    tissue = TissueRuntime.seeded(genome=genome, environment=environment, stem_cells=5, seed=11)
    tissue.develop(ticks=4)

    assert tissue.fate_counts()["repair"] >= 2
    assert tissue.niche_occupancy()["repair-niche"] >= 2


def test_tissue_cli_runs_framework_specs(tmp_path: Path) -> None:
    from ontocellia.__main__ import main

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
            "--output",
            str(output),
        ]
    )

    assert (output / "tissue_summary.json").exists()
    assert (output / "tissue_trace.json").exists()

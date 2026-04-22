from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from ontocellia import (
    EnvironmentModel,
    FateLandscape,
    GenomeProgram,
    OntocelliaConfig,
    OntocelliaRuntime,
    OrganSelectionField,
    load_environment_spec,
    load_genome_spec,
)
from ontocellia.compiler import EnvironmentBuilder, GenomeSpecCompiler
from ontocellia.specs.schema import EnvironmentSpec, GenomeSpec


ROOT = Path(__file__).resolve().parents[1]
GENOME_PATH = ROOT / "examples" / "specs" / "minimal_genome.yaml"
ENVIRONMENT_PATH = ROOT / "examples" / "specs" / "minimal_environment.yaml"


def test_yaml_and_json_specs_load_and_reproduce(tmp_path: Path) -> None:
    genome = load_genome_spec(GENOME_PATH)
    environment = load_environment_spec(ENVIRONMENT_PATH)

    env_json = tmp_path / "environment.json"
    env_json.write_text(json.dumps(yaml.safe_load(ENVIRONMENT_PATH.read_text(encoding="utf-8"))), encoding="utf-8")
    environment_from_json = load_environment_spec(env_json)

    assert genome.metadata.name == "minimal-developmental-genome"
    assert environment.metadata.name == environment_from_json.metadata.name

    runtime_a = OntocelliaRuntime.from_specs(genome, environment, sim_config=OntocelliaConfig(seed=9))
    runtime_b = OntocelliaRuntime.from_specs(genome, environment_from_json, sim_config=OntocelliaConfig(seed=9))
    runtime_a.step(18)
    runtime_b.step(18)

    assert runtime_a.metrics.history[-1]["population"] == runtime_b.metrics.history[-1]["population"]
    assert runtime_a.metrics.history[-1]["development_diversity"] == runtime_b.metrics.history[-1]["development_diversity"]


def test_invalid_genome_spec_missing_required_section_fails(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid_genome.yaml"
    invalid_path.write_text(
        yaml.safe_dump(
            {
                "metadata": {"name": "broken"},
                "behavior_biases": {"move": [1, 0], "divide": [1, 0], "die": [0, 1], "secrete": [1, 0], "rewire": [0, 1], "repair": [0, 1]},
            }
        ),
        encoding="utf-8",
    )

    try:
        load_genome_spec(invalid_path)
    except ValueError as exc:
        assert "state_dims" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid genome spec")


def test_different_genomes_produce_different_behavior_under_same_environment() -> None:
    genome_data = yaml.safe_load(GENOME_PATH.read_text(encoding="utf-8"))
    environment = load_environment_spec(ENVIRONMENT_PATH)

    conservative = GenomeSpec.from_dict(genome_data)
    aggressive_data = yaml.safe_load(GENOME_PATH.read_text(encoding="utf-8"))
    aggressive_data["metadata"]["name"] = "aggressive-division"
    aggressive_data["behavior_biases"]["divide"] = [1.0, 0.9, 0.55, 0.4]
    aggressive_data["lineage_rules"]["division_energy_threshold"] = 0.45
    aggressive = GenomeSpec.from_dict(aggressive_data)

    runtime_a = OntocelliaRuntime.from_specs(conservative, environment, sim_config=OntocelliaConfig(seed=11))
    runtime_b = OntocelliaRuntime.from_specs(aggressive, environment, sim_config=OntocelliaConfig(seed=11))
    runtime_a.step(24)
    runtime_b.step(24)

    assert runtime_b.division_events != runtime_a.division_events
    assert runtime_b.metrics.history[-1]["population"] != runtime_a.metrics.history[-1]["population"]


def test_environment_task_translation_materially_changes_fields_and_coverage() -> None:
    genome = load_genome_spec(GENOME_PATH)
    environment = load_environment_spec(ENVIRONMENT_PATH)
    calm_environment_data = yaml.safe_load(ENVIRONMENT_PATH.read_text(encoding="utf-8"))
    calm_environment_data["metadata"]["name"] = "calm-environment"
    calm_environment_data["task_translation"]["goal_zones"] = []
    calm_environment_data["task_translation"]["risk_zones"] = []
    calm_environment_data["spatial_environment"]["events"] = []
    calm_environment = EnvironmentSpec.from_dict(calm_environment_data)

    active_runtime = OntocelliaRuntime.from_specs(genome, environment, sim_config=OntocelliaConfig(seed=13))
    calm_runtime = OntocelliaRuntime.from_specs(genome, calm_environment, sim_config=OntocelliaConfig(seed=13))

    assert active_runtime.environment.fields["task_pressure"].max() > calm_runtime.environment.fields["task_pressure"].max()

    active_runtime.step(20)
    calm_runtime.step(20)

    assert active_runtime.metrics.history[-1]["risk_exposure"] != calm_runtime.metrics.history[-1]["risk_exposure"]


def test_spec_mode_generates_probe_counts_and_continuous_diversity() -> None:
    runtime = OntocelliaRuntime.from_spec_files(GENOME_PATH, ENVIRONMENT_PATH, sim_config=OntocelliaConfig(seed=6))
    runtime.step(22)

    last = runtime.metrics.history[-1]
    assert runtime.mode == "spec"
    assert last["development_diversity"] > 0.2
    assert last["probe_counts"]
    assert last["attractor_occupancy"]
    assert any(label != "unlabeled" for label in runtime.phenotype_counts())


def test_biological_spec_sections_are_loaded_and_built() -> None:
    genome = load_genome_spec(GENOME_PATH)
    environment = load_environment_spec(ENVIRONMENT_PATH)
    runtime = OntocelliaRuntime.from_specs(genome, environment, sim_config=OntocelliaConfig(seed=7))

    assert {m.name for m in genome.morphogens} == {"M1", "M2", "M3"}
    assert any(program.name == "NotchLike" for program in genome.contact_programs)
    assert {"ECM", "mechanical_stress", "crowding"} <= set(environment.background_context)
    assert {"M1", "M2", "M3", "task_pressure"} <= set(runtime.environment.fields)
    assert runtime.built_environment.contact_context["contact_radius"] > 0


def test_contact_program_toggle_changes_contact_inhibition_behavior() -> None:
    environment = load_environment_spec(ENVIRONMENT_PATH)
    contact_data = yaml.safe_load(GENOME_PATH.read_text(encoding="utf-8"))
    no_contact_data = yaml.safe_load(GENOME_PATH.read_text(encoding="utf-8"))
    no_contact_data["metadata"]["name"] = "no-contact"
    no_contact_data["contact_programs"] = []

    with_contact = OntocelliaRuntime.from_specs(GenomeSpec.from_dict(contact_data), environment, sim_config=OntocelliaConfig(seed=5))
    without_contact = OntocelliaRuntime.from_specs(GenomeSpec.from_dict(no_contact_data), environment, sim_config=OntocelliaConfig(seed=5))
    with_contact.step(20)
    without_contact.step(20)

    assert with_contact.metrics.history[-1]["contact_inhibition"] != without_contact.metrics.history[-1]["contact_inhibition"]
    assert with_contact.phenotype_counts() != without_contact.phenotype_counts()


def test_mechanical_background_context_changes_repair_activation() -> None:
    genome = load_genome_spec(GENOME_PATH)
    base_environment_data = yaml.safe_load(ENVIRONMENT_PATH.read_text(encoding="utf-8"))
    high_mech_environment_data = yaml.safe_load(ENVIRONMENT_PATH.read_text(encoding="utf-8"))
    high_mech_environment_data["metadata"]["name"] = "high-mechanics"
    high_mech_environment_data["spatial_environment"]["background_context"]["mechanical_stress"]["initial"] = {"pattern": "constant", "value": 0.8}

    base_runtime = OntocelliaRuntime.from_specs(genome, EnvironmentSpec.from_dict(base_environment_data), sim_config=OntocelliaConfig(seed=8))
    high_mech_runtime = OntocelliaRuntime.from_specs(genome, EnvironmentSpec.from_dict(high_mech_environment_data), sim_config=OntocelliaConfig(seed=8))
    base_runtime.step(18)
    high_mech_runtime.step(18)

    assert base_runtime.metrics.history[-1]["repair_activation"] != high_mech_runtime.metrics.history[-1]["repair_activation"]


def test_cli_supports_legacy_and_spec_modes(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    legacy_output = tmp_path / "legacy_out"
    spec_output = tmp_path / "spec_out"

    subprocess.run(
        [sys.executable, "-m", "ontocellia", "--steps", "4", "--output", str(legacy_output)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ontocellia",
            "--steps",
            "4",
            "--genome-spec",
            str(GENOME_PATH),
            "--environment-spec",
            str(ENVIRONMENT_PATH),
            "--output",
            str(spec_output),
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    legacy_summary = json.loads((legacy_output / "summary.json").read_text(encoding="utf-8"))
    spec_summary = json.loads((spec_output / "summary.json").read_text(encoding="utf-8"))

    assert legacy_summary["mode"] == "legacy"
    assert spec_summary["mode"] == "spec"
    assert "genome_spec" in spec_summary


def test_framework_components_construct_without_runtime() -> None:
    genome = load_genome_spec(GENOME_PATH)
    environment = load_environment_spec(ENVIRONMENT_PATH)
    built = EnvironmentBuilder().build(environment)
    compiled = GenomeSpecCompiler().compile(genome, built.field_names)

    model = EnvironmentModel.from_built_environment(environment, built)
    program = GenomeProgram(OntocelliaConfig(seed=3), compiled)
    landscape = FateLandscape(OntocelliaConfig(seed=3), compiled)
    selection = OrganSelectionField()

    assert model.global_environment.task_goal
    assert program.compiled_genome is compiled
    assert landscape.compiled_genome is compiled
    assert selection.strength > 0


def test_environment_spec_exposes_global_and_spatial_layers() -> None:
    environment = load_environment_spec(ENVIRONMENT_PATH)

    assert environment.global_environment.task_goal.text
    assert environment.global_environment.evaluation.metrics["min_coverage"] == 0.25
    assert environment.spatial_environment.grid.width == 32
    assert {"M1", "M2", "M3"} <= set(environment.spatial_environment.diffusive_fields)


def test_receptor_profile_and_history_influence_state_update() -> None:
    runtime = OntocelliaRuntime.from_spec_files(GENOME_PATH, ENVIRONMENT_PATH, sim_config=OntocelliaConfig(seed=17))
    runtime.graph.rebuild(runtime.cells)
    cell = next(iter(runtime.cells.values()))
    local_fields = runtime.environment.sample(cell.pos)
    gradients = runtime.environment.gradients(cell.pos)
    neighbor_summary = runtime.graph.summary_for(cell.id, runtime.cells, runtime.config.desired_local_density)
    baseline = runtime.kernel.step(cell, local_fields, gradients, neighbor_summary, {})

    cell_low = cell.clone(9000, cell.pos.copy(), runtime.rng.normal(0.0, 0.0, size=max(runtime.config.hidden_dim, runtime.config.local_memory_dim, runtime.compiled_genome.development_dim)))
    cell_low.receptor_profile = np.ones_like(cell.receptor_profile) * 0.2
    altered_receptor = runtime.kernel.step(cell_low, local_fields, gradients, neighbor_summary, {})

    cell_hist = cell.clone(9001, cell.pos.copy(), runtime.rng.normal(0.0, 0.0, size=max(runtime.config.hidden_dim, runtime.config.local_memory_dim, runtime.compiled_genome.development_dim)))
    cell_hist.append_history({"task_pressure": 1.0, "damage": 0.8, "energy": 0.3, "stress": 0.7})
    altered_history = runtime.kernel.step(cell_hist, local_fields, gradients, neighbor_summary, {})

    assert not np.allclose(baseline.development_delta, altered_receptor.development_delta)
    assert not np.allclose(baseline.development_delta, altered_history.development_delta)


def test_organ_feedback_forms_weak_closed_loop() -> None:
    with_feedback = OntocelliaRuntime.from_spec_files(GENOME_PATH, ENVIRONMENT_PATH, sim_config=OntocelliaConfig(seed=19, enable_organ_feedback=True))
    without_feedback = OntocelliaRuntime.from_spec_files(GENOME_PATH, ENVIRONMENT_PATH, sim_config=OntocelliaConfig(seed=19, enable_organ_feedback=False))

    with_feedback.step(14)
    without_feedback.step(14)

    assert with_feedback.metrics.history[-1]["organ_feedback"]["selection_pressure"] >= 0.0
    assert "selection_pressure" in with_feedback.environment.fields
    assert "selection_pressure" not in without_feedback.environment.fields


def test_community_formation_creates_shared_organization_unit() -> None:
    runtime = OntocelliaRuntime.from_spec_files(GENOME_PATH, ENVIRONMENT_PATH, sim_config=OntocelliaConfig(seed=21))
    runtime.step(10)

    assert runtime.communities
    community = next(iter(runtime.communities.values()))
    assert len(community.member_ids) >= 2
    assert all(runtime.cells[cell_id].community_id == community.id for cell_id in community.member_ids if cell_id in runtime.cells)
    signal = runtime.graph.summary_for(community.member_ids[0], runtime.cells, runtime.config.desired_local_density)["community_signal"]
    assert signal > 0.0

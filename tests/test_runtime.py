from __future__ import annotations

import numpy as np

from ontocellia import GeneAsset, OntocelliaConfig, OntocelliaRuntime
from ontocellia.config import GeneKind


def make_runtime(**overrides) -> OntocelliaRuntime:
    return OntocelliaRuntime(OntocelliaConfig(seed=4, **overrides))


def densest_position(runtime: OntocelliaRuntime) -> tuple[float, float]:
    cell = max(
        runtime.cells.values(),
        key=lambda current: sum(
            1
            for other in runtime.cells.values()
            if other.id != current.id and np.linalg.norm(other.pos - current.pos) <= runtime.config.spatial_radius
        ),
    )
    return float(cell.pos[0]), float(cell.pos[1])


def run_damage_sequence(runtime: OntocelliaRuntime, pulses: int = 2, settle_steps: int = 12) -> tuple[int, int]:
    runtime.step(settle_steps)
    for _ in range(pulses):
        runtime.inject_damage(densest_position(runtime), 5, 1.0)
        runtime.step(2)
    return runtime.tick_count, len(runtime.cells)


def test_self_organization_produces_stable_heterogeneity() -> None:
    runtime = make_runtime()
    runtime.step(40)

    counts = runtime.phenotype_counts()
    active_fates = sum(1 for count in counts.values() if count > 0)

    assert len(runtime.cells) > runtime.config.initial_cells
    assert active_fates >= 3
    assert runtime.metrics.history[-1]["heterogeneity"] > 0.65
    assert counts["specialist"] > 0


def test_local_damage_triggers_loss_then_repair() -> None:
    runtime = make_runtime()
    runtime.step(12)
    before_damage = len(runtime.cells)

    runtime.inject_damage(densest_position(runtime), 5, 1.0)
    runtime.step(1)
    after_first_pulse = len(runtime.cells)
    runtime.inject_damage(densest_position(runtime), 5, 1.0)
    runtime.step(2)
    after_damage = len(runtime.cells)
    repair_fraction_after_damage = runtime.metrics.history[-1]["repair_fraction"]

    runtime.step(15)
    recovered = len(runtime.cells)

    assert runtime.total_deaths >= 4
    assert after_damage < before_damage
    assert recovered > after_damage
    assert repair_fraction_after_damage > 0.05


def test_epigenetic_lock_prevents_fate_collapse_under_disturbance() -> None:
    locked = make_runtime(enable_epigenetic_lock=True)
    unlocked = make_runtime(enable_epigenetic_lock=False)

    for runtime in (locked, unlocked):
        for step in range(30):
            if step in (10, 18):
                runtime.inject_damage(densest_position(runtime), 5, 1.0)
            runtime.step(1)

    locked_counts = locked.phenotype_counts()
    unlocked_counts = unlocked.phenotype_counts()

    assert unlocked.fate_switches > locked.fate_switches
    assert unlocked_counts["specialist"] / len(unlocked.cells) > 0.65
    assert locked_counts["stem"] > 0


def test_competence_window_limits_global_overreaction() -> None:
    competent = make_runtime(enable_competence=True)
    unbounded = make_runtime(enable_competence=False)

    competent.step(12)
    unbounded.step(12)

    competent_baseline = {cell_id: cell.commitment.copy() for cell_id, cell in competent.cells.items()}
    unbounded_baseline = {cell_id: cell.commitment.copy() for cell_id, cell in unbounded.cells.items()}

    competent.inject_damage(densest_position(competent), 5, 1.0)
    unbounded.inject_damage(densest_position(unbounded), 5, 1.0)
    competent.step(3)
    unbounded.step(3)

    competent_delta = np.mean(
        [np.abs(competent.cells[cell_id].commitment - competent_baseline[cell_id]).sum() for cell_id in competent.cells if cell_id in competent_baseline]
    )
    unbounded_delta = np.mean(
        [np.abs(unbounded.cells[cell_id].commitment - unbounded_baseline[cell_id]).sum() for cell_id in unbounded.cells if cell_id in unbounded_baseline]
    )

    assert unbounded_delta > competent_delta
    assert competent.metrics.history[-1]["repair_fraction"] > unbounded.metrics.history[-1]["repair_fraction"]


def test_hybrid_substrate_outperforms_single_channel_variants() -> None:
    full = make_runtime(enable_graph=True, enable_spatial=True)
    no_graph = make_runtime(enable_graph=False, enable_spatial=True)
    no_spatial = make_runtime(enable_graph=True, enable_spatial=False)

    variants = [full, no_graph, no_spatial]
    for runtime in variants:
        run_damage_sequence(runtime, pulses=1, settle_steps=12)
        runtime.step(15)

    assert len(full.cells) > len(no_graph.cells)
    assert no_spatial.metrics.history[-1]["repair_fraction"] == 0.0
    assert len(no_spatial.cells) > len(full.cells) * 1.5
    assert full.metrics.history[-1]["graph_density"] > no_graph.metrics.history[-1]["graph_density"]


def test_resource_driven_division_bounds_population_growth() -> None:
    controlled = make_runtime(resource_driven_division=True)
    free = make_runtime(resource_driven_division=False)

    controlled.step(50)
    free.step(50)

    assert len(free.cells) > len(controlled.cells) * 3
    assert free.metrics.history[-1]["avg_energy"] < controlled.metrics.history[-1]["avg_energy"]


def test_warning_gene_reduces_risky_divisions() -> None:
    baseline = make_runtime()
    guarded = make_runtime()
    guarded.add_gene(
        GeneAsset(
            kind=GeneKind.WARNING,
            name="avoid-damage-division",
            signals=["damage", "crowding"],
            summary="Suppress risky replication under local danger and crowding.",
            avoid=["divide into damaged zones", "amplify crowding loops"],
            magnitude=0.1,
        )
    )

    for runtime in (baseline, guarded):
        runtime.step(12)
        for _ in range(3):
            runtime.inject_damage(densest_position(runtime), 5, 1.0)
            runtime.step(2)

    assert guarded.risky_divisions < baseline.risky_divisions
    assert guarded.division_events < baseline.division_events

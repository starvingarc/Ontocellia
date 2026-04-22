from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from ontocellia.config import LEGACY_MODE


def export_summary(runtime, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": runtime.mode,
        "project_kind": "developmental_agent_framework",
        "config": runtime.config.as_dict(),
        "tick": runtime.tick_count,
        "population": len(runtime.cells),
        "total_deaths": runtime.total_deaths,
        "fate_switches": runtime.fate_switches,
        "metrics": runtime.metrics.history,
        "lineage": runtime.lineage_edges,
        "genes": [gene.name for gene in runtime.gene_registry.genes],
        "phenotypes": runtime.phenotype_counts(),
        "communities": {
            str(community_id): {
                "member_ids": community.member_ids,
                "signal_pool": community.signal_pool,
                "cohesion": community.cohesion,
            }
            for community_id, community in getattr(runtime, "communities", {}).items()
        },
    }
    if runtime.mode != LEGACY_MODE and runtime.genome_spec is not None and runtime.environment_spec is not None:
        payload["framework"] = {
            "genome_program": "GenomeProgram",
            "fate_landscape": "FateLandscape",
            "life_process_model": "LifeProcessModel",
            "organ_selection_field": "OrganSelectionField",
            "runtime_adapter": type(runtime).__name__,
        }
        payload["genome_spec"] = {
            "name": runtime.genome_spec.metadata.name,
            "path": str(runtime.genome_spec.source_path) if runtime.genome_spec.source_path else None,
            "morphogens": [m.name for m in runtime.genome_spec.morphogens],
            "contact_programs": [p.name for p in runtime.genome_spec.contact_programs],
            "attractors": [a.name for a in runtime.genome_spec.fate_landscape.attractors],
        }
        payload["environment_spec"] = {
            "name": runtime.environment_spec.metadata.name,
            "path": str(runtime.environment_spec.source_path) if runtime.environment_spec.source_path else None,
            "global_task": runtime.built_environment.global_task if runtime.built_environment is not None else {},
        }
    summary_path = output_path / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


def plot_fields(runtime, output_dir: str | Path) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results: list[Path] = []
    for name, field in runtime.environment.fields.items():
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(field, cmap="viridis", origin="lower")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        path = output_path / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        results.append(path)
    return results


def plot_interaction_graph(runtime, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    positions = {cell_id: runtime.cells[cell_id].pos for cell_id in runtime.graph.graph.nodes if cell_id in runtime.cells}
    if runtime.mode == LEGACY_MODE:
        colors = [runtime.cells[cell_id].fate_index for cell_id in positions]
    else:
        labels = sorted({runtime.cells[cell_id].phenotype_label for cell_id in positions})
        index_map = {label: index for index, label in enumerate(labels)}
        colors = [index_map[runtime.cells[cell_id].phenotype_label] for cell_id in positions]
    nx.draw_networkx(
        runtime.graph.graph.subgraph(list(positions)),
        pos=positions,
        node_size=70,
        node_color=colors,
        edge_color="gray",
        width=0.8,
        with_labels=False,
        ax=ax,
        cmap="plasma",
    )
    ax.set_title("Interaction Graph")
    ax.set_xticks([])
    ax.set_yticks([])
    path = output_path / "interaction_graph.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_fate_timeline(runtime, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    history = runtime.metrics.history
    fig, ax = plt.subplots(figsize=(7, 4))
    ticks = [row["tick"] for row in history]
    if runtime.mode == LEGACY_MODE:
        labels = list(history[-1]["fate_counts"].keys()) if history and history[-1]["fate_counts"] else []
        for label in labels:
            series = [row["fate_counts"][label] for row in history]
            ax.plot(ticks, series, label=label)
        ax.set_title("Fate Timeline")
        ax.set_ylabel("Cell count")
    else:
        ax.plot(ticks, [row["development_diversity"] for row in history], label="development_diversity")
        ax.plot(ticks, [row["repair_activation"] for row in history], label="repair_activation")
        ax.plot(ticks, [row["division_pressure"] for row in history], label="division_pressure")
        ax.plot(ticks, [row["risk_exposure"] for row in history], label="risk_exposure")
        ax.plot(ticks, [row.get("contact_inhibition", 0.0) for row in history], label="contact_inhibition")
        ax.plot(ticks, [row.get("communities", 0) for row in history], label="communities")
        attractor_labels = sorted({label for row in history for label in row.get("attractor_occupancy", {}).keys()})
        for label in attractor_labels:
            series = [row.get("attractor_occupancy", {}).get(label, 0.0) for row in history]
            ax.plot(ticks, series, label=f"attr:{label}")
        probe_labels = sorted({probe for row in history for probe in row.get("probe_counts", {}).keys()})
        for label in probe_labels:
            series = [row.get("probe_counts", {}).get(label, 0) for row in history]
            ax.plot(ticks, series, label=label)
        ax.set_title("Development Timeline")
        ax.set_ylabel("Metric / Count")
    ax.set_xlabel("Tick")
    ax.legend(frameon=False, ncol=2)
    path = output_path / "fate_timeline.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_lineage(runtime, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    graph = nx.DiGraph()
    graph.add_edges_from(runtime.lineage_edges)
    fig, ax = plt.subplots(figsize=(6, 5))
    if graph.number_of_nodes() > 0:
        positions = nx.spring_layout(graph, seed=runtime.config.seed)
        nx.draw_networkx(graph, pos=positions, node_size=60, with_labels=False, arrows=False, ax=ax)
    ax.set_title("Lineage Graph")
    ax.set_xticks([])
    ax.set_yticks([])
    path = output_path / "lineage.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path

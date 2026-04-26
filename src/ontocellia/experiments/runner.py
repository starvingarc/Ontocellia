from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from ontocellia.config import OntocelliaConfig
from ontocellia.observation import export_summary, plot_fate_timeline, plot_fields, plot_interaction_graph, plot_lineage
from ontocellia.scheduler.runtime import OntocelliaRuntime
from ontocellia.specs import EnvironmentSpec, ExperimentSpec, GenomeSpec, load_experiment_spec
from ontocellia.specs.loader import _load_mapping


@dataclass(slots=True)
class ExperimentRunResult:
    name: str
    output_dir: Path
    summary_path: Path
    metrics_path: Path | None
    final_metrics: dict[str, Any]
    phenotypes: dict[str, int]


@dataclass(slots=True)
class ExperimentResult:
    name: str
    output_dir: Path
    runs: list[ExperimentRunResult]
    comparison_path: Path
    comparison_csv_path: Path
    report_path: Path | None


class ExperimentRunner:
    def __init__(self, spec: ExperimentSpec):
        self.spec = spec

    @classmethod
    def from_spec_file(cls, path: str | Path) -> "ExperimentRunner":
        return cls(load_experiment_spec(path))

    def run(self, output_dir: str | Path) -> ExperimentResult:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        runs_dir = output_path / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        run_results: list[ExperimentRunResult] = []
        for variant in self.spec.variants:
            runtime = self._build_runtime(variant.config_patch, variant.genome_patch, variant.environment_patch)
            runtime.step(self.spec.base.steps)
            run_dir = runs_dir / variant.name
            run_dir.mkdir(parents=True, exist_ok=True)
            summary_path = export_summary(runtime, run_dir)
            if self.spec.outputs.plots:
                plot_fields(runtime, run_dir)
                plot_interaction_graph(runtime, run_dir)
                plot_fate_timeline(runtime, run_dir)
                plot_lineage(runtime, run_dir)
            metrics_path = self._write_metrics_csv(runtime, run_dir) if self.spec.outputs.metrics_csv else None
            run_results.append(
                ExperimentRunResult(
                    name=variant.name,
                    output_dir=run_dir,
                    summary_path=summary_path,
                    metrics_path=metrics_path,
                    final_metrics=dict(runtime.metrics.history[-1]),
                    phenotypes=runtime.phenotype_counts(),
                )
            )

        comparison_rows = [self._comparison_row(result) for result in run_results]
        comparison_path = output_path / "comparison.json"
        comparison_path.write_text(
            json.dumps({"experiment": self.spec.metadata.name, "runs": comparison_rows}, indent=2),
            encoding="utf-8",
        )
        comparison_csv_path = self._write_comparison_csv(comparison_rows, output_path)
        self._write_comparison_plot(comparison_rows, output_path)
        report_path = self._write_report(comparison_rows, output_path) if self.spec.outputs.report else None
        return ExperimentResult(
            name=self.spec.metadata.name,
            output_dir=output_path,
            runs=run_results,
            comparison_path=comparison_path,
            comparison_csv_path=comparison_csv_path,
            report_path=report_path,
        )

    def _build_runtime(
        self,
        config_patch: dict[str, Any],
        genome_patch: dict[str, Any],
        environment_patch: dict[str, Any],
    ) -> OntocelliaRuntime:
        genome_data = self._patched_mapping(self.spec.resolved_genome_path, genome_patch)
        environment_data = self._patched_mapping(self.spec.resolved_environment_path, environment_patch)
        genome = GenomeSpec.from_dict(genome_data, source_path=self.spec.resolved_genome_path)
        environment = EnvironmentSpec.from_dict(environment_data, source_path=self.spec.resolved_environment_path)
        config = OntocelliaConfig(seed=self.spec.base.seed)
        allowed_config_keys = set(config.as_dict()) | {
            "hidden_dim",
            "local_memory_dim",
            "initial_cells",
            "communication_radius",
            "division_threshold",
            "death_threshold",
            "energy_floor",
            "gene_evolution_period",
        }
        for key, value in config_patch.items():
            if key not in allowed_config_keys or not hasattr(config, key):
                raise ValueError(f"config_patch.{key} is not a supported OntocelliaConfig field")
            setattr(config, key, value)
        return OntocelliaRuntime.from_specs(genome, environment, sim_config=config)

    def _patched_mapping(self, path: Path, patch: dict[str, Any]) -> dict[str, Any]:
        data = deepcopy(_load_mapping(path))
        for key, value in patch.items():
            data[key] = value
        return data

    def _write_metrics_csv(self, runtime: OntocelliaRuntime, output_dir: Path) -> Path:
        path = output_dir / "metrics.csv"
        fieldnames = sorted({key for row in runtime.metrics.history for key in row if isinstance(row.get(key), (int, float, str))})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in runtime.metrics.history:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
        return path

    def _comparison_row(self, result: ExperimentRunResult) -> dict[str, Any]:
        metrics = result.final_metrics
        return {
            "variant": result.name,
            "population": metrics.get("population", 0),
            "development_diversity": metrics.get("development_diversity", metrics.get("heterogeneity", 0.0)),
            "repair_activation": metrics.get("repair_activation", 0.0),
            "risk_exposure": metrics.get("risk_exposure", 0.0),
            "division_pressure": metrics.get("division_pressure", 0.0),
            "graph_density": metrics.get("graph_density", 0.0),
            "communities": metrics.get("communities", 0),
            "phenotypes": result.phenotypes,
        }

    def _write_comparison_csv(self, rows: list[dict[str, Any]], output_dir: Path) -> Path:
        path = output_dir / "comparison.csv"
        scalar_keys = [key for key in rows[0] if key != "phenotypes"] if rows else ["variant"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=scalar_keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in scalar_keys})
        return path

    def _write_comparison_plot(self, rows: list[dict[str, Any]], output_dir: Path) -> Path:
        path = output_dir / "comparison.png"
        labels = [str(row["variant"]) for row in rows]
        metrics = ["population", "development_diversity", "repair_activation", "risk_exposure"]
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        for ax, metric in zip(axes.ravel(), metrics, strict=True):
            values = [float(row.get(metric, 0.0)) for row in rows]
            ax.bar(labels, values)
            ax.set_title(metric)
            ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return path

    def _write_report(self, rows: list[dict[str, Any]], output_dir: Path) -> Path:
        path = output_dir / "report.md"
        lines = [
            f"# Experiment Report: {self.spec.metadata.name}",
            "",
            f"- Steps: {self.spec.base.steps}",
            f"- Seed: {self.spec.base.seed}",
            f"- Variants: {', '.join(str(row['variant']) for row in rows)}",
            "",
            "| Variant | Population | Diversity | Repair | Risk | Communities |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                "| {variant} | {population} | {development_diversity:.3f} | {repair_activation:.3f} | {risk_exposure:.3f} | {communities} |".format(
                    **row
                )
            )
        lines.append("")
        lines.append("Phenotypes are recorded in `comparison.json` for exact downstream analysis.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

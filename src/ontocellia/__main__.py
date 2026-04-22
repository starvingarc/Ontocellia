from __future__ import annotations

import argparse
from pathlib import Path

from ontocellia.config import GeneAsset, GeneKind, OntocelliaConfig
from ontocellia.observation import export_summary, plot_fate_timeline, plot_fields, plot_interaction_graph, plot_lineage
from ontocellia.scheduler.runtime import OntocelliaRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an Ontocellia developmental simulation.")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--genome-spec", type=Path)
    parser.add_argument("--environment-spec", type=Path)
    parser.add_argument("--damage-step", type=int, default=30)
    parser.add_argument("--damage-radius", type=float, default=3.5)
    parser.add_argument("--damage-intensity", type=float, default=0.85)
    parser.add_argument("--with-warning-gene", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = OntocelliaConfig(seed=args.seed)
    if args.genome_spec or args.environment_spec:
        if not args.genome_spec or not args.environment_spec:
            raise SystemExit("--genome-spec and --environment-spec must be provided together")
        runtime = OntocelliaRuntime.from_spec_files(args.genome_spec, args.environment_spec, sim_config=config)
    else:
        runtime = OntocelliaRuntime(config)
        runtime.inject_goal((config.width * 0.72, config.height * 0.48), radius=4.0, intensity=0.18)

    if args.with_warning_gene:
        runtime.add_gene(
            GeneAsset(
                kind=GeneKind.WARNING,
                name="avoid-damage-division",
                signals=["damage", "crowding"],
                summary="Suppress risky replication and bias cells into local repair when damage spikes.",
                avoid=["replicate inside damaged zones", "overcrowd stressed regions"],
                validation_hooks=["deaths decrease", "repair fraction rises briefly then stabilizes"],
                magnitude=0.25,
            )
        )

    for step in range(args.steps):
        if runtime.mode == "legacy" and step == args.damage_step:
            runtime.inject_damage(
                center=(config.width / 2, config.height / 2),
                radius=args.damage_radius,
                intensity=args.damage_intensity,
            )
        if runtime.mode == "legacy" and step % 15 == 0 and step > 0:
            runtime.inject_resource_pulse((config.width * 0.2, config.height * 0.2), radius=2.8, intensity=0.2)
        runtime.step()

    output_dir = args.output
    summary_path = export_summary(runtime, output_dir)
    plot_fields(runtime, output_dir)
    plot_interaction_graph(runtime, output_dir)
    plot_fate_timeline(runtime, output_dir)
    plot_lineage(runtime, output_dir)
    print(f"Summary written to {summary_path}")
    print(f"Mode: {runtime.mode}")
    print(f"Final population: {len(runtime.cells)}")
    print(f"Phenotypes: {runtime.phenotype_counts()}")


if __name__ == "__main__":
    main()

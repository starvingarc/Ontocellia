from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ontocellia.config import GeneAsset, GeneKind, OntocelliaConfig
from ontocellia.experiments import ExperimentRunner
from ontocellia.observation import export_summary, plot_fate_timeline, plot_fields, plot_interaction_graph, plot_lineage
from ontocellia.scheduler.runtime import ReferenceRuntime
from ontocellia.specs import export_schema_docs, validate_experiment_spec, validate_model_specs


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--genome-spec", type=Path)
    parser.add_argument("--environment-spec", type=Path)
    parser.add_argument("--damage-step", type=int, default=30)
    parser.add_argument("--damage-radius", type=float, default=3.5)
    parser.add_argument("--damage-intensity", type=float, default=0.85)
    parser.add_argument("--with-warning-gene", action="store_true")


def build_legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an Ontocellia developmental simulation.")
    add_run_arguments(parser)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ontocellia developmental agent framework.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a single Ontocellia simulation.")
    add_run_arguments(run_parser)

    experiment_parser = subparsers.add_parser("experiment", help="Run an Ontocellia experiment spec.")
    experiment_parser.add_argument("--experiment-spec", type=Path, required=True)
    experiment_parser.add_argument("--output", type=Path, default=Path("artifacts/experiment"))

    validate_parser = subparsers.add_parser("validate", help="Validate model or experiment specs.")
    validate_parser.add_argument("--genome-spec", type=Path)
    validate_parser.add_argument("--environment-spec", type=Path)
    validate_parser.add_argument("--experiment-spec", type=Path)

    schema_parser = subparsers.add_parser("schema-docs", help="Export Markdown schema reference docs.")
    schema_parser.add_argument("--output", type=Path, default=Path("docs/schema"))
    return parser


def run_simulation(args: argparse.Namespace) -> None:
    config = OntocelliaConfig(seed=args.seed)
    if args.genome_spec or args.environment_spec:
        if not args.genome_spec or not args.environment_spec:
            raise SystemExit("--genome-spec and --environment-spec must be provided together")
        runtime = ReferenceRuntime.from_spec_files(args.genome_spec, args.environment_spec, sim_config=config)
    else:
        runtime = ReferenceRuntime(config)
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
    print("Framework: Ontocellia developmental agent architecture")
    print(f"Mode: {runtime.mode}")
    print(f"Final population: {len(runtime.cells)}")
    print(f"Phenotypes: {runtime.phenotype_counts()}")


def run_experiment(args: argparse.Namespace) -> None:
    result = ExperimentRunner.from_spec_file(args.experiment_spec).run(args.output)
    print(f"Experiment written to {result.output_dir}")
    print(f"Comparison written to {result.comparison_path}")
    if result.report_path is not None:
        print(f"Report written to {result.report_path}")


def run_validate(args: argparse.Namespace) -> None:
    errors: list[str] = []
    if args.experiment_spec:
        errors.extend(validate_experiment_spec(args.experiment_spec))
    else:
        if not args.genome_spec or not args.environment_spec:
            raise SystemExit("validate requires --experiment-spec or both --genome-spec and --environment-spec")
        errors.extend(validate_model_specs(args.genome_spec, args.environment_spec))
    if errors:
        for error in errors:
            print(f"Validation error: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("Validation passed")


def run_schema_docs(args: argparse.Namespace) -> None:
    paths = export_schema_docs(args.output)
    print(f"Schema docs written to {args.output}")
    for path in paths:
        print(path)


def main(argv: list[str] | None = None) -> None:
    args_list = list(sys.argv[1:] if argv is None else argv)
    commands = {"run", "experiment", "validate", "schema-docs"}
    if args_list and args_list[0] in commands:
        args = build_parser().parse_args(args_list)
        if args.command == "run":
            run_simulation(args)
        elif args.command == "experiment":
            run_experiment(args)
        elif args.command == "validate":
            run_validate(args)
        elif args.command == "schema-docs":
            run_schema_docs(args)
        return
    run_simulation(build_legacy_parser().parse_args(args_list))


if __name__ == "__main__":
    main()

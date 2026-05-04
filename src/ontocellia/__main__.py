from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

from ontocellia.cli_ui import (
    prompt,
    render_banner,
    render_config_status,
    render_error,
    render_help,
    render_models,
    render_choice_list,
    render_provider_catalog,
    render_tissue_summary,
)
from ontocellia.config import GeneAsset, GeneKind, OntocelliaConfig
from ontocellia.experiments import ExperimentRunner
from ontocellia.framework import (
    EffectorRuntime,
    ExecutionPolicy,
    ExecutionRuntime,
    ExtracellularToolRuntime,
    InductionRequest,
    MockLLMProvider,
    MutationCandidateGenerator,
    MutationSelectionRuntime,
    OfficialBenchmarkRunner,
    OrganValidationResult,
    StructureSearchRunner,
    TemplateInductionCompiler,
    TissueRuntime,
    BenchmarkSuite,
    TissueBenchmarkRunner,
    ToolPolicy,
    ValidationHookPolicy,
    ValidationHookRequest,
    ValidationHookRunner,
    load_agent_genome,
    load_task_microenvironment,
    resolve_effector_provider,
    run_repo_repair_demo,
    write_mutation_outputs,
)
from ontocellia.framework.llm import CellPrompt
from ontocellia.framework.model_config import (
    PROVIDER_DEFAULTS,
    ModelProfile,
    config_path,
    get_config_value,
    load_secret_env,
    load_user_config,
    save_secret,
    save_user_config,
    secrets_path,
    set_config_value,
    unset_config_value,
)
from ontocellia.tui import run_tui
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

    tissue_parser = subparsers.add_parser("tissue", help="Run an agent tissue from AgentGenome and TaskMicroenvironment specs.")
    tissue_parser.add_argument("--genome-spec", type=Path, required=True)
    tissue_parser.add_argument("--environment-spec", type=Path, required=True)
    tissue_parser.add_argument("--steps", type=int, default=8)
    tissue_parser.add_argument("--seed", type=int, default=7)
    tissue_parser.add_argument("--stem-cells", type=int, default=1)
    tissue_parser.add_argument("--effector", choices=["rule", "mock-llm", "llm", "deepseek", "kimi", "minimax"], default="rule")
    tissue_parser.add_argument("--model-profile")
    tissue_parser.add_argument("--llm-model")
    tissue_parser.add_argument("--llm-base-url")
    tissue_parser.add_argument("--validation-result", type=Path)
    tissue_parser.add_argument("--run-validation-hooks", action="store_true")
    tissue_parser.add_argument("--allow-validation-hook", action="append", default=[])
    tissue_parser.add_argument("--validation-timeout", type=float, default=60.0)
    tissue_parser.add_argument("--execute-actions", action="store_true")
    tissue_parser.add_argument("--execution-dry-run", dest="execution_dry_run", action="store_true", default=True)
    tissue_parser.add_argument("--no-execution-dry-run", dest="execution_dry_run", action="store_false")
    tissue_parser.add_argument("--allow-interface", action="append", default=[])
    tissue_parser.add_argument("--allow-command", action="append", default=[])
    tissue_parser.add_argument("--allow-write", action="append", default=[])
    tissue_parser.add_argument("--allow-network-host", action="append", default=[])
    tissue_parser.add_argument("--allow-mcp-tool", action="append", default=[])
    tissue_parser.add_argument("--allow-git-command", action="append", default=[])
    tissue_parser.add_argument("--enable-http-tools", action="store_true")
    tissue_parser.add_argument("--enable-browser-tools", action="store_true")
    tissue_parser.add_argument("--execution-timeout", type=float, default=60.0)
    tissue_parser.add_argument("--output", type=Path, default=Path("artifacts/tissue"))

    induce_parser = subparsers.add_parser("induce", help="Compile a natural language task into agent tissue specs.")
    induce_parser.add_argument("--task", required=True)
    induce_parser.add_argument("--domain", default="repo_repair")
    induce_parser.add_argument("--interface", action="append", dest="interfaces", default=[])
    induce_parser.add_argument("--seed", type=int, default=7)
    induce_parser.add_argument("--output", type=Path, default=Path("artifacts/induced"))

    mutate_parser = subparsers.add_parser("mutate", help="Generate and select genome mutations from validation evidence.")
    mutate_parser.add_argument("--genome-spec", type=Path, required=True)
    mutate_parser.add_argument("--environment-spec", type=Path, required=True)
    mutate_parser.add_argument("--baseline-validation", type=Path, required=True)
    mutate_parser.add_argument("--candidate-validation", type=Path, required=True)
    mutate_parser.add_argument("--output", type=Path, default=Path("artifacts/mutation_selection"))

    demo_parser = subparsers.add_parser("demo", help="Run a complete deterministic repo repair tissue demo.")
    demo_parser.add_argument("--task", default="Fix failing tests while preserving existing behavior.")
    demo_parser.add_argument("--steps", type=int, default=4)
    demo_parser.add_argument("--seed", type=int, default=7)
    demo_parser.add_argument("--output", type=Path, default=Path("artifacts/complete_repo_repair_demo"))

    benchmark_parser = subparsers.add_parser("benchmark", help="Run Ontocellia tissue benchmark suites.")
    benchmark_parser.add_argument("--suite", default="ontocellia_minibench_v1")
    benchmark_parser.add_argument("--effector", choices=["mock-llm", "llm"], default="mock-llm")
    benchmark_parser.add_argument("--model-profile")
    benchmark_parser.add_argument("--steps", type=int, default=6)
    benchmark_parser.add_argument("--execute-actions", action="store_true")
    benchmark_parser.add_argument("--execution-dry-run", dest="execution_dry_run", action="store_true", default=True)
    benchmark_parser.add_argument("--no-execution-dry-run", dest="execution_dry_run", action="store_false")
    benchmark_parser.add_argument("--allow-interface", action="append", default=[])
    benchmark_parser.add_argument("--allow-command", action="append", default=[])
    benchmark_parser.add_argument("--output", type=Path, default=Path("artifacts/benchmarks/minibench"))

    official_parser = subparsers.add_parser("official-benchmark", help="Run official benchmark data through Ontocellia.")
    official_subparsers = official_parser.add_subparsers(dest="official_command", required=True)
    official_prepare = official_subparsers.add_parser("prepare", help="Prepare official benchmark data or write a dry-run plan.")
    official_prepare.add_argument("--benchmark", choices=["bfcl", "tau-bench", "terminal-bench", "multiagentbench", "swe-bench-lite"], required=True)
    official_prepare.add_argument("--output", type=Path, default=Path("artifacts/official_benchmarks/prepare"))
    official_prepare.add_argument("--dry-run", action="store_true")
    official_run = official_subparsers.add_parser("run", help="Run official benchmark data with a configured model profile.")
    official_run.add_argument("--benchmark", choices=["bfcl", "tau-bench", "terminal-bench", "multiagentbench", "swe-bench-lite"], required=True)
    official_run.add_argument("--model-profile")
    official_run.add_argument("--limit", type=int)
    official_run.add_argument("--task-id")
    official_run.add_argument("--full", action="store_true")
    official_run.add_argument("--dry-run", action="store_true")
    official_run.add_argument("--mode", choices=["adaptive-tissue", "provider-baseline"], default=None)
    official_run.add_argument("--category", default="BFCL_v3_simple")
    official_run.add_argument("--split", default="test")
    official_run.add_argument("--source-dir", type=Path)
    official_run.add_argument("--tau-domain", choices=["airline", "retail"], default="airline")
    official_run.add_argument("--structure-search", action="store_true")
    official_run.add_argument("--run-official-scorer", action="store_true")
    official_run.add_argument("--output", type=Path, default=Path("artifacts/official_benchmarks/bfcl/run"))

    structure_parser = subparsers.add_parser("structure-search", help="Run deterministic tissue structure variant search.")
    structure_parser.add_argument("--task", required=True)
    structure_parser.add_argument("--domain", default="repo_repair")
    structure_parser.add_argument("--effector", choices=["mock-llm", "llm"], default="mock-llm")
    structure_parser.add_argument("--model-profile")
    structure_parser.add_argument("--steps", type=int, default=6)
    structure_parser.add_argument("--seed", type=int, default=7)
    structure_parser.add_argument("--output", type=Path, default=Path("artifacts/structure_search"))

    subparsers.add_parser("tui", help="Start the interactive Ontocellia TUI.")

    server_parser = subparsers.add_parser("server", help="Start the Ontocellia HTTP/WebSocket app server.")
    server_parser.add_argument("--host", default="127.0.0.1")
    server_parser.add_argument("--port", type=int, default=8765)
    server_parser.add_argument("--output", type=Path, default=Path("artifacts/server_sessions"))
    server_parser.add_argument("--real-provider", action="store_true", help="Use configured provider profiles instead of mock provider by default.")

    config_parser = subparsers.add_parser("config", help="Inspect or edit user configuration.")
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)
    config_subparsers.add_parser("file", help="Print the active config file path.")
    config_subparsers.add_parser("setup", help="Run first-time model provider setup.")
    config_subparsers.add_parser("validate", help="Validate the active config file.")
    config_get = config_subparsers.add_parser("get", help="Read a config value.")
    config_get.add_argument("path")
    config_set = config_subparsers.add_parser("set", help="Set a config value.")
    config_set.add_argument("path")
    config_set.add_argument("value")
    config_unset = config_subparsers.add_parser("unset", help="Unset a config value.")
    config_unset.add_argument("path")

    config_models = config_subparsers.add_parser("models", help="Inspect, select, or test model profiles.")
    models_subparsers = config_models.add_subparsers(dest="models_command", required=True)
    models_subparsers.add_parser("add", help="Add a model profile interactively.")
    models_subparsers.add_parser("list", help="List configured model profiles.")
    models_subparsers.add_parser("status", help="Show model configuration status.")
    models_set = models_subparsers.add_parser("set", help="Set the default model profile.")
    models_set.add_argument("profile")
    models_test = models_subparsers.add_parser("test", help="Send a minimal provider smoke request.")
    models_test.add_argument("profile", nargs="?")
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


def run_tissue(args: argparse.Namespace) -> None:
    genome = load_agent_genome(args.genome_spec)
    environment = load_task_microenvironment(args.environment_spec)
    validation_results = _load_validation_results(args.validation_result) or []
    tissue = TissueRuntime.seeded(genome=genome, environment=environment, stem_cells=args.stem_cells, seed=args.seed)
    tissue.develop(ticks=args.steps, validation_results=validation_results)
    provider = resolve_effector_provider(
        args.effector,
        model=args.llm_model,
        base_url=args.llm_base_url,
        model_profile=args.model_profile,
    )
    effectors = EffectorRuntime(provider) if provider is not None else None
    actions = tissue.execute(effectors=effectors)
    tool_results = []
    tool_invocations = []
    execution_results = []
    if args.execute_actions:
        tool_results = tissue.execute_actions(
            actions,
            ExtracellularToolRuntime(),
            ToolPolicy(
                workspace_root=Path.cwd(),
                allowed_interfaces=[str(item) for item in args.allow_interface],
                allowed_commands=[str(item) for item in args.allow_command],
                allowed_write_globs=[str(item) for item in args.allow_write],
                allowed_network_hosts=[str(item) for item in args.allow_network_host],
                allowed_mcp_tools=[str(item) for item in args.allow_mcp_tool],
                allowed_git_commands=[str(item) for item in args.allow_git_command],
                enable_http_tools=bool(args.enable_http_tools),
                enable_browser_tools=bool(args.enable_browser_tools),
                timeout_seconds=float(args.execution_timeout),
                artifact_root=args.output,
                dry_run=bool(args.execution_dry_run),
            ),
        )
        tool_invocations = [result.invocation for result in tool_results]
        execution_results = [result.to_execution_result() for result in tool_results]
        validation_results = [*validation_results, *[result.to_validation_result() for result in tool_results]]
        tissue.develop(ticks=1, validation_results=validation_results)
    runner_results: list[OrganValidationResult] = []
    if args.run_validation_hooks:
        requests = _collect_validation_hook_requests(genome, actions)
        runner_results = ValidationHookRunner().run(
            requests,
            ValidationHookPolicy(
                allowed_commands=[str(command) for command in args.allow_validation_hook],
                timeout_seconds=float(args.validation_timeout),
                artifact_root=args.output,
            ),
            trace=tissue.trace,
        )
        validation_results = [*validation_results, *runner_results]
        tissue.develop(ticks=1, validation_results=validation_results)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "objective": environment.objective,
        "ticks": tissue.tick_count,
        "population": len(tissue.cells),
        "fate_counts": tissue.fate_counts(),
        "stage_counts": tissue.stage_counts(),
        "development_stage": tissue.development_stage,
        "origin_cell_id": tissue.origin_cell_id,
        "proliferation_events": sum(1 for event in tissue.trace.events if event["type"] == "proliferation"),
        "niche_occupancy": tissue.niche_occupancy(),
        "organ_selection": tissue.last_organ_selection_report.as_dict() if tissue.last_organ_selection_report is not None else {},
        "validation_results": len(validation_results),
        "mcp_interfaces": sum(1 for interface in environment.interfaces if interface.id.startswith("mcp:")),
        "messages": sum(1 for event in tissue.trace.events if event["type"] == "message_emitted"),
        "matrix_records": len(tissue.environment.matrix.records),
        "handoffs": sum(1 for event in tissue.trace.events if event["type"] == "handoff_completed"),
        "actions": actions,
        "execution_results": len(execution_results),
        "executed_actions": sum(1 for result in execution_results if result.status == "passed"),
        "skipped_actions": sum(1 for result in execution_results if result.status in {"skipped", "dry_run"}),
        "changed_files": sorted({path for result in execution_results for path in result.changed_files}),
        "tool_invocations": len(tool_invocations),
        "tool_results": len(tool_results),
        "blocked_tool_invocations": sum(1 for result in tool_results if result.status in {"skipped", "dry_run"}),
        "tool_adapters": sorted({result.invocation.adapter for result in tool_results}),
        "mcp_tool_calls": sum(1 for result in tool_results if result.invocation.adapter == "mcp"),
        "network_tool_calls": sum(1 for result in tool_results if result.invocation.adapter == "http"),
        "browser_tool_calls": sum(1 for result in tool_results if result.invocation.adapter == "browser"),
    }
    output_stats = _output_digest_stats([*tool_results, *execution_results, *validation_results])
    summary.update(output_stats)
    summary_path = args.output / "tissue_summary.json"
    trace_path = args.output / "tissue_trace.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    trace_path.write_text(json.dumps(tissue.trace.events, indent=2, sort_keys=True), encoding="utf-8")
    if args.run_validation_hooks:
        (args.output / "validation_results.json").write_text(
            json.dumps([result.as_dict() for result in validation_results], indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.execute_actions:
        (args.output / "tool_invocations.json").write_text(
            json.dumps([invocation.as_dict() for invocation in tool_invocations], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (args.output / "tool_results.json").write_text(
            json.dumps([result.as_dict() for result in tool_results], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (args.output / "execution_results.json").write_text(
            json.dumps([result.as_dict() for result in execution_results], indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if provider is not None:
        (args.output / "action_intents.json").write_text(json.dumps(actions, indent=2, sort_keys=True), encoding="utf-8")
        llm_trace = [event for event in tissue.trace.events if event["type"] == "llm_effector"]
        (args.output / "llm_trace.json").write_text(json.dumps(llm_trace, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Tissue summary written to {summary_path}")
    print(f"Tissue trace written to {trace_path}")


def run_induce(args: argparse.Namespace) -> None:
    request = InductionRequest(task=args.task, domain=args.domain, available_interfaces=args.interfaces, seed=args.seed)
    draft = TemplateInductionCompiler().compile(request)
    paths = draft.write(args.output)
    print(f"Genome written to {paths['genome']}")
    print(f"Environment written to {paths['environment']}")
    print(f"Induction report written to {paths['report']}")


def run_mutate(args: argparse.Namespace) -> None:
    genome = load_agent_genome(args.genome_spec)
    environment = load_task_microenvironment(args.environment_spec)
    baseline = _load_validation_results(args.baseline_validation) or []
    candidate_validation = _load_validation_results(args.candidate_validation) or []
    candidates = MutationCandidateGenerator().generate(genome, baseline, environment=environment)
    report = MutationSelectionRuntime().select(genome, candidates, baseline, candidate_validation)
    paths = write_mutation_outputs(report, args.output)
    print(f"Mutation candidates written to {paths['candidates']}")
    print(f"Mutation report written to {paths['report']}")
    print(f"Solidified genome written to {paths['genome']}")


def run_demo(args: argparse.Namespace) -> None:
    result = run_repo_repair_demo(output=args.output, task=args.task, steps=args.steps, seed=args.seed)
    print(f"Demo summary written to {result.summary_path}")
    print(f"Demo report written to {result.report_path}")


def run_benchmark(args: argparse.Namespace) -> None:
    result = TissueBenchmarkRunner(
        suite=BenchmarkSuite.builtin(args.suite),
        effector=args.effector,
        model_profile=args.model_profile,
        steps=args.steps,
        execute_actions=args.execute_actions,
        execution_dry_run=args.execution_dry_run,
        allowed_interfaces=[str(item) for item in args.allow_interface],
        allowed_commands=[str(item) for item in args.allow_command],
    ).run(args.output)
    print(f"Benchmark summary written to {result.summary_path}")
    print(f"Benchmark report written to {result.report_path}")


def run_official_benchmark(args: argparse.Namespace) -> None:
    runner = OfficialBenchmarkRunner()
    if args.official_command == "prepare":
        plan = runner.prepare(args.benchmark, output=args.output, dry_run=args.dry_run)
        print(f"Official benchmark prepare plan written to {args.output / 'prepare_plan.json'}")
        print(f"Benchmark: {plan['benchmark']}")
        return
    try:
        result = runner.run(
            benchmark=args.benchmark,
            output=args.output,
            model_profile=args.model_profile,
            limit=args.limit,
            task_id=args.task_id,
            full=args.full,
            dry_run=args.dry_run,
            category=args.category,
            mode=args.mode,
            split=args.split,
            source_dir=args.source_dir,
            tau_domain=args.tau_domain,
            structure_search=args.structure_search,
            run_official_scorer=args.run_official_scorer,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(f"Official benchmark summary written to {result.summary_path}")
    print(f"Official benchmark report written to {result.report_path}")


def run_structure_search(args: argparse.Namespace) -> None:
    report = StructureSearchRunner(
        task=args.task,
        domain=args.domain,
        effector=args.effector,
        model_profile=args.model_profile,
        steps=args.steps,
        seed=args.seed,
    ).run(args.output)
    print(f"Structure search written to {report.output_dir}")
    print(f"Selected variant: {report.selected_variant}")


def run_server(args: argparse.Namespace) -> None:
    import uvicorn

    from ontocellia.server import create_app

    app = create_app(output_root=args.output, use_mock=not bool(args.real_provider))
    print(f"Ontocellia server listening on http://{args.host}:{args.port}")
    uvicorn.run(app, host=str(args.host), port=int(args.port))


def run_configure(args: argparse.Namespace) -> None:
    config = load_user_config()
    print("Configure Ontocellia model provider")
    provider = _choose_provider()
    defaults = PROVIDER_DEFAULTS[provider]
    profile_name = provider
    model = _choose_model(provider, defaults)
    base_url = _setup_base_url(provider, defaults)
    api_key_env = _setup_api_key_env(provider, defaults)
    api_key = ""
    if api_key_env:
        api_key = getpass.getpass(f"API key for {api_key_env} (leave blank to skip): ")
    config.profiles[profile_name] = ModelProfile(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
    )
    if not config.default_model or _prompt("Set as default? yes/no", "yes").lower().startswith("y"):
        config.default_model = profile_name
    path = save_user_config(config)
    if api_key and api_key_env:
        save_secret(api_key_env, api_key)
        print(f"Secret saved to {secrets_path()}")
    print(f"Config saved to {path}")


def _choose_provider() -> str:
    provider_names = _ordered_provider_names()
    print(render_provider_catalog(PROVIDER_DEFAULTS, title="Choose Provider"))
    while True:
        value = _prompt("provider", "2")
        provider = _choice_value(value, provider_names)
        if provider is not None:
            return provider
        print(render_error(f"Unknown provider selection: {value}", "Choose a number from the provider list."))


def _choose_model(provider: str, defaults: dict[str, object]) -> str:
    choices = [str(item) for item in defaults.get("models", [])]
    if provider == "custom-openai-compatible":
        return _prompt("model", str(defaults["model"]))
    if not choices:
        return str(defaults["model"])
    print(render_choice_list("Choose Model", choices))
    while True:
        value = _prompt("model", "1")
        model = _choice_value(value, choices)
        if model is not None:
            return model
        print(render_error(f"Unknown model selection: {value}", "Choose a number from the model list."))


def _setup_base_url(provider: str, defaults: dict[str, object]) -> str:
    if provider == "custom-openai-compatible":
        return _prompt("base url", str(defaults["base_url"]))
    return str(defaults["base_url"])


def _setup_api_key_env(provider: str, defaults: dict[str, object]) -> str:
    if provider == "custom-openai-compatible":
        return _prompt("api key env", str(defaults["api_key_env"]))
    return str(defaults["api_key_env"])


def run_config_command(args: argparse.Namespace) -> None:
    config = load_user_config()
    if args.config_command == "file":
        print(config_path())
    elif args.config_command == "setup":
        run_configure(args)
    elif args.config_command == "validate":
        _validate_user_config(config)
        print(render_config_status(config_path(), secrets_path()))
    elif args.config_command == "models":
        run_models_command(args)
    elif args.config_command == "get":
        value = get_config_value(config, args.path)
        print(json.dumps(value, sort_keys=True))
    elif args.config_command == "set":
        set_config_value(config, args.path, args.value)
        save_user_config(config)
        print(f"Set {args.path}")
    elif args.config_command == "unset":
        unset_config_value(config, args.path)
        save_user_config(config)
        print(f"Unset {args.path}")


def run_models_command(args: argparse.Namespace) -> None:
    config = load_user_config()
    if args.models_command == "add":
        run_configure(args)
    elif args.models_command == "list":
        _print_model_profiles(config)
    elif args.models_command == "status":
        print(render_config_status(config_path(), secrets_path()))
        print(render_models(config, include_key_status=True, secrets=load_secret_env()))
        print(render_provider_catalog(PROVIDER_DEFAULTS))
    elif args.models_command == "set":
        if args.profile not in config.profiles:
            raise SystemExit(f"Unknown model profile: {args.profile}")
        config.default_model = args.profile
        save_user_config(config)
        print(f"Default model profile set to {args.profile}")
    elif args.models_command == "test":
        provider = resolve_effector_provider("llm", model_profile=args.profile)
        if isinstance(provider, MockLLMProvider):
            print("Model profile test passed: mock-llm")
            return
        prompt = CellPrompt(
            system="Return exactly one compact JSON object for an Ontocellia ActionIntent.",
            context={
                "cell_id": 1,
                "fate": "repair",
                "position": {"node_id": "repair-niche"},
                "expressed_gene_ids": ["gene_repair_from_test_failures"],
                "allowed_interfaces": ["workspace", "pytest"],
                "validation_hooks": [],
            },
            output_schema={"type": "ActionIntent"},
        )
        response = provider.complete(prompt)
        print(f"Model profile test passed: {getattr(provider, 'name', response.model)} / {response.model}")


def run_interactive(input_stream: object | None = None, output_stream: object | None = None) -> None:
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    print(render_banner(load_user_config()), file=output_stream)
    while True:
        print(prompt(), end="", file=output_stream, flush=True)
        line = input_stream.readline()
        if line == "":
            print("", file=output_stream)
            return
        command = line.strip()
        if not command:
            continue
        if command.startswith("/"):
            command = command[1:]
        if command in {"exit", "quit"}:
            print("culture closed. see you next induction.", file=output_stream)
            return
        if command == "help":
            print(render_help(), file=output_stream)
        elif command == "setup":
            run_configure(argparse.Namespace(section="models"))
        elif command == "config":
            _validate_user_config(load_user_config())
            print(render_config_status(config_path(), secrets_path()), file=output_stream)
        elif command == "config models":
            print(render_models(load_user_config()), file=output_stream)
        elif command == "config models add":
            run_configure(argparse.Namespace(section="models"))
        elif command.startswith("config models test"):
            parts = command.split()
            run_models_command(argparse.Namespace(models_command="test", profile=parts[3] if len(parts) > 3 else None))
        elif command.startswith("use "):
            run_models_command(argparse.Namespace(models_command="set", profile=command.split(maxsplit=1)[1]))
        elif command == "run tissue":
            main(
                [
                    "tissue",
                    "--genome-spec",
                    "examples/framework/repo_repair_genome.yaml",
                    "--environment-spec",
                    "examples/framework/failing_tests_environment.yaml",
                    "--effector",
                    "llm",
                    "--output",
                    "artifacts/interactive_tissue",
                ]
            )
            summary_path = Path("artifacts/interactive_tissue/tissue_summary.json")
            if summary_path.exists():
                print(render_tissue_summary(summary_path, json.loads(summary_path.read_text(encoding="utf-8"))), file=output_stream)
        else:
            print(render_error(f"Unknown command: {line.strip()}", "Type /help to see available commands."), file=output_stream)


def _load_validation_results(path: Path | None) -> list[OrganValidationResult] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("--validation-result must contain a JSON list")
    return [
        OrganValidationResult(
            name=str(item["name"]),
            passed=bool(item.get("passed", False)),
            score=float(item.get("score", 0.0)),
            target=str(item.get("target", "")),
            evidence=str(item.get("evidence", "")),
            cost=float(item.get("cost", 0.0)),
            risk=float(item.get("risk", 0.0)),
            latency=float(item.get("latency", 0.0)),
            output_digest=dict(item.get("output_digest", {})),
        )
        for item in data
    ]


def _output_digest_stats(results: list[Any]) -> dict[str, Any]:
    digests = [
        dict(getattr(result, "output_digest", {}))
        for result in results
        if isinstance(getattr(result, "output_digest", {}), dict) and getattr(result, "output_digest", {})
    ]
    raw_paths = {str(digest.get("raw_output_path")) for digest in digests if digest.get("raw_output_path")}
    return {
        "raw_outputs": len(raw_paths),
        "truncated_outputs": sum(1 for digest in digests if digest.get("truncated")),
        "output_digest_chars": sum(int(digest.get("inline_chars", 0)) for digest in digests),
    }


def _collect_validation_hook_requests(genome: object, actions: list[dict[str, object]]) -> list[ValidationHookRequest]:
    requests: list[ValidationHookRequest] = []
    for gene in getattr(genome, "genes", []):
        for hook in getattr(gene, "validation_hooks", []):
            requests.append(
                ValidationHookRequest(
                    name=str(getattr(gene, "id", hook)),
                    command=str(hook),
                    source_gene_id=str(getattr(gene, "id", "")),
                )
            )
    for index, action in enumerate(actions):
        hooks = action.get("validation_hooks", [])
        if not isinstance(hooks, list):
            continue
        for hook in hooks:
            requests.append(
                ValidationHookRequest(
                    name=str(action.get("intent_type") or action.get("gene_id") or f"action-{index}"),
                    command=str(hook),
                    source_action_id=str(action.get("intent_type") or action.get("gene_id") or f"action-{index}"),
                    source_cell_id=int(action["cell_id"]) if "cell_id" in action else None,
                )
            )
    return requests


def _prompt(label: str, default: str) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or default


def _choice_value(value: str, choices: list[str]) -> str | None:
    if value.isdigit():
        index = int(value)
        if 1 <= index <= len(choices):
            return choices[index - 1]
        return None
    normalized = value.strip().lower()
    for choice in choices:
        if choice.lower() == normalized:
            return choice
    return None


def _ordered_provider_names() -> list[str]:
    preferred = [
        "mock-llm",
        "deepseek",
        "minimax",
        "kimi",
        "openai",
        "openrouter",
        "ollama",
        "custom-openai-compatible",
    ]
    return [name for name in preferred if name in PROVIDER_DEFAULTS] + [
        name for name in sorted(PROVIDER_DEFAULTS) if name not in preferred
    ]


def _validate_user_config(config: object) -> None:
    if not isinstance(config, object):
        raise ValueError("invalid config")


def _print_model_profiles(config: object, *, include_key_status: bool = False) -> None:
    print(render_models(config, include_key_status=include_key_status, secrets=load_secret_env()))


def main(argv: list[str] | None = None) -> None:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        if sys.stdin.isatty() and sys.stdout.isatty() and not os.environ.get("ONTOCELLIA_NO_TUI"):
            run_tui()
            return
        run_interactive()
        return
    commands = {
        "run",
        "experiment",
        "validate",
        "schema-docs",
        "tissue",
        "induce",
        "mutate",
        "demo",
        "benchmark",
        "official-benchmark",
        "structure-search",
        "tui",
        "server",
        "config",
    }
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
        elif args.command == "tissue":
            run_tissue(args)
        elif args.command == "induce":
            run_induce(args)
        elif args.command == "mutate":
            run_mutate(args)
        elif args.command == "demo":
            run_demo(args)
        elif args.command == "benchmark":
            run_benchmark(args)
        elif args.command == "official-benchmark":
            run_official_benchmark(args)
        elif args.command == "structure-search":
            run_structure_search(args)
        elif args.command == "tui":
            run_tui()
        elif args.command == "server":
            run_server(args)
        elif args.command == "config":
            run_config_command(args)
        return
    if args_list and not args_list[0].startswith("-"):
        build_parser().parse_args(args_list)
    run_simulation(build_legacy_parser().parse_args(args_list))


if __name__ == "__main__":
    main()

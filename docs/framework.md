# Framework Overview

Ontocellia's framework layer models a task-induced agent tissue. It is separate from the older reference simulation runtime, but both share the same developmental vocabulary.

## Layers

- `AgentGenome`: shared heritable program made of genes, regulatory elements, epigenetic defaults, and mutation history.
- `AgentCell`: autonomous cell-agent with stage, fate, position, lineage, receptor profile, adhesion profile, competence, energy, stress, and local history.
- `MorphogenField`: global and local task signals that bias expression, migration, regeneration, and selection.
- `TissueTopology`: graph-first tissue position model with semantic nodes, regions, neighbors, and 3D embedding fallback.
- `FateLandscape`: fate attractors, thresholds, niche bias, competence, and hysteresis.
- `TaskMicroenvironment`: objective, morphogens, topology, niches, extracellular interfaces, MCP adapter metadata, organ selection targets, communication policy, and shared matrix.
- `TissueRuntime`: deterministic harness for development, regeneration, effectors, execution, communication, and trace recording.

## Runtime Flow

```text
seed one stem-origin cell
refresh niche occupancy
resolve vacancies and regeneration signals
proliferate into a stem/progenitor pool
differentiate through local morphogens and fate landscape
move cells over tissue topology
apply organ selection feedback
execute rule or LLM effectors
optionally execute actions through an allowlisted extracellular execution policy
optionally run explicitly allowlisted validation hooks
route messages, handoffs, and matrix deposits
record trace and summary artifacts
```

By default, framework tissues begin from a single `stem-origin` cell. The runtime first enters a `proliferating` stage, expands until the minimum differentiation pool is available, then commits cells into task-induced niches. Explicit `--stem-cells N` remains available for experiments that need a larger initial reserve.

## Organ Selection

The organ selection layer consumes structured validation results and tissue state. It emits bounded feedback signals such as `selection_pressure`, `validation_pressure`, `risk_pressure`, `resource_pressure`, and `reward_signal`.

Validation hooks remain metadata until the opt-in Validation Hook Runner is enabled. The runner executes only exact allowlisted commands, converts outcomes into `OrganValidationResult` records, and feeds those records back into organ selection.

## Communication

Cells communicate with `TissueMessage` objects and durable `MatrixRecord` entries. Direct, local, fate-scoped, and broadcast messages support short-term coordination. The shared extracellular matrix stores evidence, memory, hypotheses, validation notes, and handoff context.

## Context Homeostasis

Context is managed through the shared extracellular matrix rather than a single prompt history. Matrix records carry lifecycle state, validation state, references, salience, decay, lineage, and correction links. Cells receive a bounded `ContextPacket` assembled by deterministic retrieval over tags, fate, graph locality, confidence, freshness, validation status, and receptor/interface relevance.

`CellPromptBuilder` includes the packet as `relevant_matrix` and records `context_record_ids` in every LLM effector trace and emitted `ActionIntent`. Execution and validation results deposit evidence back into the matrix, where stale weak hypotheses decay and contradicted records can be corrected or suppressed.

## Context Metabolism

Context metabolism is the tissue's matrix-remodeling system. Instead of repeatedly summarizing the whole history, the runtime deterministically digests recent matrix records, trace events, execution evidence, validation feedback, constraints, and contradicted records into higher-order metabolite `MatrixRecord` entries.

The first metabolites are `failure_signature`, `episode_summary`, `causal_chain`, `constraint_digest`, and `toxic_context`. They preserve source record IDs, source trace event IDs, compression level, lossiness, source count, and scope in record metadata. Source records are not deleted; their salience is weakly reduced so retrieval prefers compact metabolites while keeping raw evidence inspectable.

`TissueRuntime.develop()` runs metabolism after organ selection and validation feedback, then before matrix decay. `CellPromptBuilder` keeps the legacy `relevant_matrix` section and also exposes `context_metabolites` plus `raw_context_record_ids`, so LLM effectors can distinguish synthesized context from raw evidence.

## Output Metabolism

Output metabolism keeps large tool and validation outputs out of prompts and result JSON while preserving full auditability. Tool execution and validation hooks write long raw stdout, stderr, and evidence to `raw_outputs/` artifacts, then expose deterministic digests inline.

Digests preserve head, tail, error/failure/traceback lines, original character count, truncation status, and the raw artifact path. Matrix records receive the same output metadata, so context metabolism can work from compact evidence while raw logs remain available for debugging.

## MCP Adapter

MCP is modeled as an extracellular interface implementation detail. MCP tools become membrane-channel interfaces, MCP resources seed extracellular matrix records, MCP prompts become induction-factor interfaces, and tool results can deposit matrix evidence plus returned morphogen signals.

## Mutation Selection

Mutation selection turns failed validation and matrix evidence into deterministic genome mutation candidates. Candidates are shallow gene-field changes and are only solidified when candidate validation improves over the baseline. Solidified genomes retain `LineageMutation` history for auditability.

## Reference End-To-End Demo

The repo repair demo connects induction, tissue development, mock LLM effectors, communication, validation evidence, mutation selection, and report artifacts in one deterministic workflow.

## Benchmark Harness

The tissue benchmark harness evaluates framework-native agent capabilities with deterministic MiniBench tasks. It scores structured intents, interface policy compliance, matrix memory, handoff completion, self-repair recovery, lineage traceability, and decentralized coordination. The first built-in suite is `ontocellia_minibench_v1`.

## Official Benchmark Integration

The official benchmark integration layer prepares and runs selected official-compatible flows while keeping Ontocellia's adaptive tissue metrics separate from external scorer results. BFCL uses official data and answer scoring as a provider baseline. SWE-bench Lite loads the upstream Hugging Face split, Terminal-Bench loads upstream `task.yaml` files, and tau-bench parses upstream airline/retail task files without importing their runtime dependencies.

Non-BFCL runs report Ontocellia structure metrics separately from official scorer status. When an official scorer is not executed, artifacts state `official_score_status: not_run`; pass/fail is only reported after an actual official scorer command runs. Adaptive runs can compare structure variants and record selected structure, repair presence, expected fate coverage, matrix reuse, provider calls, and provider-call errors.

The scorer adapter layer writes benchmark-specific command plans without changing cell behavior. SWE-bench Lite writes a SWE-style prediction file and an official harness command plan; the scorer runs only when the harness dependency is installed. The generic `--official-scorer-command` escape hatch remains available for local experiments.

## Official Agent Adapters

Ontocellia also exposes harness-facing agent boundaries without adding benchmark-specific cell fates or effectors.

Terminal-Bench can import `ontocellia.official_terminal_agent:OntocelliaTerminalAgent` through its custom `BaseAgent` path. The adapter turns a terminal task into normal Ontocellia induction, single-stem development, mock or configured intent generation, and a bounded set of terminal commands through the official session object. Artifacts are written under the harness logging directory.

tau-bench-style tool-calling harnesses can point at the app server's OpenAI-compatible `/v1/chat/completions` bridge. The bridge accepts chat messages and tool schemas, runs the request through an Ontocellia tissue, and returns a standard assistant message or tool call. Tool names are drawn from the provided tool schema and receptor/interface constraints; airline or retail domain behavior is not hard-coded into the effector.

Official task induction applies repair pressure to repo-like tasks without changing cell effectors. SWE-bench Lite always uses repo-repair induction; Terminal-Bench coding, debugging, software-engineering, compatibility, pytest, failing, regression, fix, or bug tasks receive repair morphogens and a repair niche, while generic data-processing/file-operation tasks can remain generic even when their instructions mention testing.

BFCL remains available as a provider/tool-call baseline. Ontocellia's main benchmark path targets multi-step tool-calling, terminal, collaboration, and repo-repair settings such as tau-bench, Terminal-Bench, MultiAgentBench/MARBLE-style tasks, and SWE-bench-style repo repair.

## Structure Search

Structure search grows several deterministic tissue variants from the same task induction and compares their organization. The first built-in variants are baseline, repair-heavy, review-heavy, memory-heavy, and lean. Each trial runs the normal single-stem tissue development, mock or configured effectors, communication, matrix deposition, context metabolism, and organ feedback loop.

The selection score combines validation score, fate match, matrix reuse, handoff completion, regeneration recovery, cost efficiency, and fate diversity. The selected variant is the highest-scoring tissue structure under the same seed, with stable tie-breaking by variant name.

## Contribution Attribution

Contribution attribution turns tissue artifacts into an inspectable causal graph. It reads live tissue state or saved `tissue_trace.json`, `action_intents.json`, tool results, execution results, validation results, and matrix references, then scores which cells, genes, matrix records, handoffs, tools, and validation signals contributed positively or negatively.

The first attribution runtime is deterministic. `llm_effector` events connect cell and gene expression to emitted action intents. `context_record_ids` and matrix references connect retrieved context to later actions. Completed handoffs reward source and recipient cells, while failed or skipped tools and validations assign negative contribution to the related action, cell, and evidence path. Passed validation and low-risk tool results provide positive contribution.

Attribution writes `contribution_graph.json`, `contribution_summary.json`, contribution CSVs, and a Markdown report. Tissue, structure-search, and official benchmark runs can opt in with `--with-attribution`; Web Lab can later use the graph for causal-chain visualization.

## Resource Competition

Resource competition turns contribution and cost into weak survival pressure. Every development tick applies maintenance cost to cells, emits resource morphogens, and records a `resource_competition` event. Action intents and tool results can add additional cost, while contribution attribution can reward helpful cells or penalize failed paths.

The runtime tracks average cell energy, population pressure, low-energy pressure, resource pressure, and resource efficiency. Optional policy fields can cap population, allow quiescence for low-energy differentiated cells, or allow non-origin low-energy cells to be removed under over-cap pressure. The default policy is conservative and weak, so existing tissues remain stable unless a task or experiment configures stronger pressure.

Structure search and official benchmark metrics include resource efficiency, average cell energy, and population pressure, allowing tissue variants to be compared by organization and cost rather than fate coverage alone.

## Extracellular Tool Runtime

The tool runtime sits between structured `ActionIntent` records and real local effects. It normalizes intents into `ToolInvocation` records, routes them through adapter-specific executors, and returns `ToolResult` records. The older `ExecutionRuntime` API remains as a compatibility wrapper.

Supported adapter surfaces include workspace read/search/list/patch, git read commands, validation commands, exact allowlisted shell commands, declared MCP tools, allowlisted HTTP/API requests, and an optional browser adapter boundary. Every tool result records trace events, deposits evidence into the extracellular matrix, and can be converted into organ-selection validation feedback.

Execution is opt-in. Dry-run is the default; writes, shell commands, MCP calls, HTTP/API requests, and browser actions require explicit policy allowlists.

## Effectors

Effectors translate expressed gene programs into structured actions. The default rule-based path is deterministic. The mock LLM provider is deterministic for tests. Real providers are optional and use OpenAI-compatible chat completion APIs.

## Model Configuration

The model configuration layer keeps provider selection outside the genome. `ontocellia` without arguments starts a Textual/Rich Soft Lab Console TUI for configuring model profiles, inducing task tissues, observing agents, and inspecting intents, matrix records, handoffs, and reports. User config lives under `~/.ontocellia/`; traces record provider/profile/model metadata without recording API keys.

The TUI is an observation and orchestration surface. Cells emit structured `ActionIntent` records and communicate through the shared matrix; planned tool invocations are visible, while real execution remains behind the explicit extracellular tool policy.

## Living Tissue App Server

The app server exposes live tissue sessions over local HTTP and WebSocket APIs. It wraps `InteractiveTissueSession`, stores artifacts under `artifacts/server_sessions`, and streams session snapshots plus trace-derived events for induction, development, intents, messages, matrix deposits, handoffs, tool invocations, and organ feedback.

The first server version is a local development surface. It binds to `127.0.0.1` by default, uses the mock provider unless configured otherwise, and does not execute tools unless the existing explicit tool policy is used by a caller.

## Web Lab Design Target

The Web Lab is the intended browser surface for the living tissue server. It keeps the biological product model visible: a local project contains many petri-dish sessions, and each session is one task-induced tissue culture. The home view should show a petri-dish wall with development stage, population, fate mix, and validation/risk-oriented life status.

Inside a session, natural language input acts as culture-medium exchange. The server records the medium change, emits morphogen pressure, deposits the change into the extracellular matrix, and advances the same tissue lineage. The central dish should use a hybrid visual model: soft 2D cells, graph/topology overlays, message arcs, optional morphogen and matrix layers, plus a timeline for replay-oriented observation.

The Web Lab also needs controlled research interventions. Users should be able to inject morphogens, clear or freeze cells, pause/resume a session, inspect a cell profile, and review pending tool invocations. Tool approval remains a membrane-channel operation backed by the extracellular tool runtime and project policy; the UI does not bypass receptor, environment, or policy gates.

The detailed visual direction is captured in [web-lab-design.md](web-lab-design.md).

# Architecture

Ontocellia has two runtime paths. The current project direction is the framework-first agent tissue path; the older scheduler/spec simulation path remains as a reference substrate for deterministic developmental experiments.

## Framework Agent Tissue Path

This is the main path used by the TUI, app server, benchmark harness, official agent adapters, and end-to-end demos.

```text
natural language task
-> induction compiler
-> AgentGenome + TaskMicroenvironment
-> single stem-origin cell
-> proliferation, fate commitment, communication, matrix memory
-> LLM or rule effectors emit ActionIntent
-> optional extracellular tool runtime
-> validation and organ feedback
-> trace, artifacts, reports, benchmark metrics
```

Key modules:

- `framework.genome`: genes, regulatory elements, expression, epigenetic marks, mutation history.
- `framework.cell`: cell stage, fate, lineage, receptor, adhesion, competence, graph position, local history.
- `framework.core`: `TissueRuntime`, task microenvironment, niches, morphogen field, trace.
- `framework.communication`: messages, handoffs, extracellular matrix, context homeostasis, context metabolism.
- `framework.llm`: prompt building, provider abstraction, mock provider, structured `ActionIntent`.
- `framework.execution`: extracellular tool runtime, policy gates, execution results, output metabolism.
- `framework.selection`: organ-level validation and weak feedback.
- `framework.structure_search`: deterministic tissue variant trials and structure scoring.
- `framework.official_benchmark` and `framework.official_agents`: official data adapters, scorer plans, and harness-facing agent boundaries.

## Reference Simulation Path

The legacy/spec simulation path is used by the older `run`, `experiment`, `validate`, and schema-doc workflows.

```text
GenomeSpec / EnvironmentSpec / ExperimentSpec
-> compiler layer
-> scheduler runtime and step pipeline
-> observation plots and comparison artifacts
```

This path preserves the original developmental simulation vocabulary: shared genome, local divergence, fate commitment, microenvironment fields, interaction graph, community state, and weak global feedback. It is useful for reproducible ablation experiments and schema-driven mechanism comparison.

## Product Surfaces

- CLI: quick commands, model setup, tissue runs, validation hooks, benchmark runs, structure search.
- TUI: local Soft Lab Console for task entry, tissue observation, model selection, and mock benchmark smoke tests.
- App server: local HTTP/WebSocket API for live sessions, culture-medium changes, interventions, tool approval, and OpenAI-compatible tool-calling bridge.
- Official adapters: Terminal-Bench custom agent import path and tau-bench-style OpenAI-compatible bridge.
- Artifacts: JSON traces, summaries, matrix records, action intents, tool results, raw output digests, reports.

## Boundary

Ontocellia is a research framework for adaptive agent organization. It prioritizes deterministic traces, bounded execution, inspectable context, and benchmark compatibility. Real model providers and tools are optional execution substrates behind explicit configuration and policy gates.

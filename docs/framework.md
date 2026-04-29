# Framework Overview

Ontocellia's framework layer models a task-induced agent tissue. It is separate from the older reference simulation runtime, but both share the same developmental vocabulary.

## Layers

- `AgentGenome`: shared heritable program made of genes, regulatory elements, epigenetic defaults, and mutation history.
- `AgentCell`: autonomous cell-agent with stage, fate, position, lineage, receptor profile, adhesion profile, competence, energy, stress, and local history.
- `MorphogenField`: global and local task signals that bias expression, migration, regeneration, and selection.
- `TissueTopology`: graph-first tissue position model with semantic nodes, regions, neighbors, and 3D embedding fallback.
- `FateLandscape`: fate attractors, thresholds, niche bias, competence, and hysteresis.
- `TaskMicroenvironment`: objective, morphogens, topology, niches, extracellular interfaces, MCP adapter metadata, organ selection targets, communication policy, and shared matrix.
- `TissueRuntime`: deterministic harness for development, regeneration, effectors, communication, and trace recording.

## Runtime Flow

```text
seed stem cells
refresh niche occupancy
resolve vacancies and regeneration
fill open niches
differentiate through local morphogens and fate landscape
move cells over tissue topology
apply organ selection feedback
execute rule or LLM effectors
optionally run explicitly allowlisted validation hooks
route messages, handoffs, and matrix deposits
record trace and summary artifacts
```

## Organ Selection

The organ selection layer consumes structured validation results and tissue state. It emits bounded feedback signals such as `selection_pressure`, `validation_pressure`, `risk_pressure`, `resource_pressure`, and `reward_signal`.

Validation hooks remain metadata until the opt-in Validation Hook Runner is enabled. The runner executes only exact allowlisted commands, converts outcomes into `OrganValidationResult` records, and feeds those records back into organ selection.

## Communication

Cells communicate with `TissueMessage` objects and durable `MatrixRecord` entries. Direct, local, fate-scoped, and broadcast messages support short-term coordination. The shared extracellular matrix stores evidence, memory, hypotheses, validation notes, and handoff context.

## MCP Adapter

MCP is modeled as an extracellular interface implementation detail. MCP tools become membrane-channel interfaces, MCP resources seed extracellular matrix records, MCP prompts become induction-factor interfaces, and tool results can deposit matrix evidence plus returned morphogen signals. Phase8 does not start external MCP processes or call network tools.

## Effectors

Effectors translate expressed gene programs into structured actions. The default rule-based path is deterministic. The mock LLM provider is deterministic for tests. Real providers are optional and use OpenAI-compatible chat completion APIs.

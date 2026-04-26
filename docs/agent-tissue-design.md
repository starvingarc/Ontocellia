# Ontocellia Agent Tissue Design

## 1. Positioning

Ontocellia is a decentralized, developmental multi-agent framework where agents act as cells and the whole system forms a task-induced functional tissue.

The biological vocabulary is the core design language for a developmental agent harness:

- undifferentiated agents begin as stem-like cells
- a shared genome defines the heritable base program of the tissue
- the task creates a microenvironment that releases signals
- agents express genes according to local signals and history
- agents differentiate into functional cell types through gene expression, morphogen response, and fate commitment
- differentiated agents communicate, divide, retire, and regenerate tissue structure
- the whole tissue performs the target function and maintains itself under disturbance

The central product claim is:

> Given a task microenvironment, Ontocellia induces a population of initially similar agents to self-organize into a functional agent tissue.

## 2. Design Principles

### Keep the Developmental Language

Ontocellia should keep biologically meaningful terms such as genome, cell, tissue, morphogen, fate, lineage, niche, organ, apoptosis, and regeneration. These terms encode how the framework thinks about coordination.

Engineering-facing documentation can explain the mapping while preserving the public developmental model.

### Decentralized Before Supervisor-Driven

The default architecture is decentralized. Global feedback acts like selection pressure or environmental conditioning.

Agents should mainly react to:

- their local state
- nearby signals
- shared tissue memory
- task pressure
- feedback from tests, tools, users, or external systems
- spatial position and nearby tissue state

### Task-Induced Differentiation

Roles emerge through induction. The same stem-agent population should be able to form different tissues under different tasks.

For example:

- a bug-fixing task may induce planner, explorer, repair, reviewer, and memory cells
- a research task may induce scout, verifier, synthesizer, critic, and citation cells
- a maintenance task may induce monitor, triager, dependency, test, and release cells

The framework should describe how induction, specialization, and tissue formation happen.

### Regenerative Self-Repair Is a First-Class Process

Self-repair means preserving tissue function when cells are damaged, removed, exhausted, or manually cleared. A failing cell emits damage signals, enters apoptosis, or is cleared by the tissue. Stem and progenitor cells respond by dividing, differentiating, and occupying the missing function.

Regeneration is driven by local tissue signals:

- cell failure or manual removal -> damage morphogen rises
- missing function -> niche vacancy signal rises
- vacancy location -> replacement cells migrate toward the damaged niche
- nearby mature cells -> damage and vacancy signals are amplified
- stem/progenitor pool -> division pressure rises
- daughter cells -> differentiation pressure toward the missing fate rises
- repeated local failure -> high-cost reprogramming may be induced in selected differentiated cells

Task failure remains one kind of tissue damage signal, but the central mechanism is stem/progenitor-mediated replacement and functional restoration.

## 3. Core Biological-Engineering Mapping

| Biological term | Ontocellia meaning |
| --- | --- |
| Cell | One autonomous agent instance with local state, tools, memory, role commitment, and energy budget |
| Stem cell | An undifferentiated agent that can become different functional cell types |
| Genome | The heritable base program shared by all cells in a tissue |
| Gene | The lowest-level heritable functional unit in the genome; it encodes a compact strategy, constraint, capability tendency, or regulatory response |
| Gene expression | Context-dependent activation of genes into cellular behavior, fate bias, signaling, constraints, or validation tendencies |
| Morphogen | A task signal that biases differentiation and behavior |
| Microenvironment | Local task context: files, logs, tests, user goals, artifacts, constraints, risk |
| Position | A cell's location in the tissue topology, such as a coordinate, graph region, work zone, or artifact neighborhood |
| Morphogen gradient | Directional signal difference that guides migration, clustering, and fate induction |
| Chemotaxis | Movement toward or away from signal gradients |
| Adhesion | Preference for cells with compatible fate or shared function to remain near each other |
| Boundary | A maintained separation between functional domains of the tissue |
| Extracellular matrix | Structured external material that cells can sense, read, bind to, and modify |
| Membrane receptor | A compatibility surface that lets a cell recognize whether an external channel or signal is usable |
| Membrane channel | A bounded passage through which a cell exchanges actions, resources, or signals with its environment |
| Induction factor | A reusable environmental pattern that biases gene expression or fate commitment |
| Niche | A local region of work where certain cell types are useful |
| Fate | A committed functional role, such as explorer, repair, reviewer, memory, planner |
| Fate landscape | The role-attractor space that guides differentiation |
| Competence window | Conditions under which a cell is ready to adopt or switch a role |
| Stem division | Spawning a related stem/progenitor cell with inherited genome and slight variation |
| Transit-amplifying cell | A short-lived proliferative intermediate that expands replacement capacity before differentiation |
| Differentiation | A cell committing to a functional role |
| Dedifferentiation | A committed cell returning to a more plastic state under high regeneration pressure |
| Reprogramming | A high-cost conversion of a differentiated cell toward a stem-like or alternative fate |
| Apoptosis | Graceful retirement of an unneeded, harmful, or exhausted agent |
| Lineage | Trace of spawn, specialization, handoff, and role transitions |
| Tissue | A coordinated population of differentiated cells performing a task |
| Organ function | The externally observable outcome the tissue is formed to deliver |
| Organ selection | Weak system-level feedback from tests, user approval, quality, cost, latency, or safety |
| Damage signal | A local signal emitted when a cell fails, degrades, is removed, or leaves a functional niche vacant |
| Niche vacancy | A missing functional position in the tissue that nearby or reserve cells can refill |
| Regeneration | Replacement of damaged or removed cells through stem/progenitor division, differentiation, and functional takeover |

## 4. Framework Architecture

```text
Ontocellia
├── Genome Layer
│   ├── AgentGenome
│   ├── Genes
│   ├── Regulatory elements
│   ├── Gene expression programs
│   ├── Epigenetic marks
│   ├── Lineage mutations
│   ├── Tool protocol
│   ├── Memory protocol
│   └── Communication protocol
│
├── Cell Layer
│   ├── AgentCell
│   ├── StemCellState
│   ├── ProgenitorCellState
│   ├── TransitAmplifyingCellState
│   ├── DifferentiatedCellState
│   ├── Position / tissue coordinates
│   ├── Adhesion profile
│   ├── Receptor profile
│   ├── Energy / budget
│   ├── Competence profile
│   └── Lineage history
│
├── Task Microenvironment
│   ├── User goal
│   ├── Work substrate
│   ├── Spatial topology
│   ├── Functional niches
│   ├── Extracellular niches
│   ├── Membrane channels
│   ├── Extracellular matrix
│   ├── Tool outputs
│   ├── Test / validation signals
│   ├── Risk and uncertainty
│   └── Resource constraints
│
├── Morphogen Field
│   ├── ambiguity
│   ├── exploration_pressure
│   ├── implementation_pressure
│   ├── verification_pressure
│   ├── repair_pressure
│   ├── coordination_pressure
│   ├── quiescence_pressure
│   ├── niche_vacancy
│   └── boundary_signal
│
├── Tissue Topology
│   ├── Cell positions
│   ├── Functional domains
│   ├── Adhesion rules
│   ├── Migration rules
│   ├── Boundary maintenance
│   └── Niche occupancy
│
├── Fate Landscape
│   ├── Role attractors
│   ├── Commitment thresholds
│   ├── Reprogramming cost
│   ├── Competence windows
│   └── Hysteresis
│
├── Tissue Runtime
│   ├── Sensing
│   ├── Migration
│   ├── Adhesion / clustering
│   ├── Differentiation
│   ├── Division
│   ├── Communication
│   ├── Handoff
│   ├── Apoptosis
│   ├── Niche refilling
│   └── Regeneration
│
├── Organ Selection
│   ├── Tests
│   ├── Review
│   ├── User feedback
│   ├── Cost / latency
│   ├── Safety constraints
│   └── Outcome quality
│
└── Observation Layer
    ├── Tissue trace
    ├── Lineage graph
    ├── Fate timeline
    ├── Signal timeline
    ├── Repair events
    └── Organ function report
```

## 5. Main Runtime Loop

The tissue runtime should proceed through repeated developmental ticks.

```text
1. Sense the task microenvironment
2. Update morphogen fields from local and global signals
3. Let each cell sample local signals and neighbor state
4. Express genes according to signals, cell state, and regulatory context
5. Update fate commitment through the fate landscape
6. Update position through chemotaxis, adhesion, boundary signals, and niche occupancy
7. Resolve life processes: stem division, progenitor amplification, differentiation, migration, quiescence, apoptosis, niche refilling, regeneration, reprogramming
8. Execute role-specific actions through membrane channels, tools, or LLM calls
9. Emit signals and artifacts back into the environment
10. Apply organ selection feedback from validation and user goals
11. Record lineage, tissue state, position, and organ function metrics
```

This loop should be deterministic where possible. LLM calls may introduce nondeterminism, so the harness should capture prompts, model settings, outputs, tool calls, and decisions as part of the tissue trace.

### Current Framework Implementation

The current implementation introduces a small programmable framework layer in `ontocellia.framework`.

It contains:

- `Gene`: the lowest-level endogenous unit with morphogen affinity, expression constraints, encoded response, and validation hooks
- `AgentGenome`: a shared heritable program that selects expressed genes under morphogen pressure
- `MorphogenField`: task and tissue signals used for induction
- `Niche`: a positioned functional region with required fate and demand
- `ExtracellularInterface`: a biological interface boundary that can be implemented by MCP, shell commands, LLM calls, or local code
- `TaskMicroenvironment`: objective, morphogens, niches, interfaces, and extracellular matrix
- `AgentCell`: a stem, progenitor, transit-amplifying, or differentiated cell-agent
- `TissueRuntime`: deterministic development, niche filling, regeneration, position updates, and effector action emission
- `TissueTrace`: lineage and tissue events

The implementation is framework-first. The direct API and YAML specs define an agent tissue; the older simulation and experiment runtime can be used as a reference substrate for dynamics, metrics, and ablation studies.

### Phase 2 Cell Layer Implementation

The Cell Layer now uses graph topology as the primary position model. A cell position is a `CellPosition` with `node_id`, `region`, `neighbors`, and a three-dimensional `embedding`. The graph node carries the semantic tissue location, such as a repair niche, review boundary, repository subsystem, document section, or resource niche. The three-dimensional embedding is used for visualization and fallback distance estimation.

The framework cell object now carries:

- stage state: stem, progenitor, transit-amplifying, or differentiated
- lineage record: parent, root, generation, and local events
- receptor profile: signal sensitivities and accepted extracellular interfaces
- adhesion profile: compatible fates and local cohesion strength
- competence profile: fate scores and plasticity
- epigenetic marks and local history

`AgentCell` can now construct a genome `ExpressionContext`, spawn lineage children, commit to a fate, record local events, and gate extracellular interface usage through receptor compatibility. `TissueRuntime` uses these methods for differentiation, division, regeneration, reprogramming, and action emission.

### Phase 3 LLM Integration Architecture

LLM integration is split into two layers with different biological meanings.

```text
LLM Integration Architecture
├── Induction Compiler
│   └── natural language task -> culture condition / tissue specs
└── Cell Effector Layer
    └── expressed gene program -> structured action intent
```

The Induction Compiler is a compile-time design assistant. It translates a user task into an induced culture condition: genome, morphogens, niches, extracellular interfaces, candidate gene assets, and experiment drafts. It does not execute task actions.

The Cell Effector Layer is runtime cellular translation machinery. It consumes expressed gene programs, cell state, local microenvironment, and receptor-allowed interfaces, then emits a structured `ActionIntent`. It does not mutate genome, call tools directly, or bypass membrane/receptor gates.

Real LLM providers can be added as provider adapters. The first implementation uses deterministic templates and a mock LLM provider so traces, tests, and reproducibility stay stable.

## 6. Genes as the Lowest-Level Unit

Ontocellia should treat genes as the lowest-level endogenous units of the system.

In this design, a gene is the minimal inheritable unit that biases what a cell senses, how it responds, what it suppresses, how it validates, how it signals, and which fate it becomes competent to enter.

The paper *From Procedural Skills to Strategy Genes: Towards Experience-Driven Test-Time Evolution* is useful here because it frames a gene as a compact control-oriented object with task-matching signals, strategy, failure salience, constraints, and validation hooks. Ontocellia adopts that compactness and control orientation while placing gene at the base of the biology-inspired stack as an endogenous genomic unit.

The biological mapping for the paper's terminology is:

| Paper term | Biological mapping | Ontocellia interpretation |
| --- | --- | --- |
| Procedural skill | Culture protocol / external experience substrate | Exogenous knowledge that can condition evolution or provide selection evidence |
| Strategy gene | Gene locus / regulatory cassette | Endogenous compact unit encoded in the genome |
| Selected strategy gene at runtime | Expressed gene program | Active cellular control tendency under morphogen and local state |
| Failure history | Damage trace / lineage stress memory | Compressed pressure that can mutate suppression cues or expression windows |
| Validation hook | Organ-selection assay | External fitness signal that determines whether a mutation stabilizes |
| Gene evolution protocol | Mutation-selection-solidification cycle | Heritable genome update governed by tissue-level validation |

In this framing, external experience influences Ontocellia through assimilation pressure, mutation, and organ selection. The resulting stable object is a gene in the genome.

```text
Genome
  stable inheritable program
        |
        | gene expression under morphogen and local state
        v
Expressed gene program
  compact cellular control tendency
        |
        | translation through cell machinery
        v
Effector behavior
  tool action, signal emission, regeneration, validation, division gating, or fate bias
```

### Gene Object Shape

A gene should be compact, structured, and locally expressible. It should have enough internal structure to be selected by signals, inherited by daughter cells, mutated through lineage, and validated through organ feedback.

Recommended fields:

```yaml
type: Gene
id: gene_repo_repair_from_test_failures
category: regeneration
morphogen_affinity:
  - test_failure
  - traceback
  - regression
expression_window:
  - failing validation signal is present
encoded_response:
  - Extract structured failure signals from logs and user goal.
  - Estimate blast radius before editing.
  - Apply the smallest reversible patch.
  - Validate with declared checks.
  - Emit a repair event with evidence.
inhibitors:
  - low_confidence_missing_logs
  - excessive_blast_radius
suppression_cues:
  - evidence-light broad rewrites
  - adding unrelated abstractions
constraints:
  max_files: 5
  forbidden_paths: [.git, node_modules]
validation_hooks:
  - python -m pytest -q
heritability:
  mutation_rate: 0.03
  lineage_bias: regeneration
```

This field naming is intentionally biological. It compiles down to operational fields such as signal matching, strategy steps, constraints, and validation checks, while the conceptual model remains gene-first.

### Gene Expression

Gene expression is how a cell turns genomic potential into local behavior.

Expression depends on:

- morphogen fields
- local microenvironment
- cell state and energy
- competence windows
- epigenetic marks
- lineage history
- organ selection feedback

A cell should usually express a small set of dominant genes. Expressing too many genes at once should be treated as biological noise: it dilutes control signal and creates conflicting responses.

Expression also depends on developmental maturity. Stem and progenitor cells can express division and differentiation genes. Mature differentiated cells primarily express maintenance, signaling, validation, and effector genes. Mature cells can enter reprogramming under strong regeneration pressure, but this path carries high cost and stricter validation.

### Regulatory Elements

Not all genomic material should be behavior genes. Some parts of the genome should regulate expression:

- promoters: increase expression under certain morphogens
- inhibitors: suppress expression under unsafe or irrelevant conditions
- enhancers: amplify a response once a cell is committed
- silencers: keep genes inactive in mature fates
- epigenetic locks: make a fate stable unless regeneration pressure is high

### Tissue Topology and Positional Patterning

Ontocellia should treat position as part of cell state. A cell's position may be a literal coordinate, a graph neighborhood, a workspace region, a repository subsystem, a document section, or a tool/resource niche. The key idea is that cells act locally and similar fates tend to form functional domains.

Developmental patterning mechanisms:

- morphogen gradients guide cells toward regions where their fate is useful
- chemotaxis moves cells toward signals such as failure, ambiguity, review pressure, or niche vacancy
- adhesion keeps compatible cells near one another so they can coordinate with low communication cost
- boundary signals keep different functional domains from collapsing into one undifferentiated cluster
- niche occupancy tracks which functions are already covered in each region

This creates tissue-like organization:

```text
repair niche
  repair cells + repair progenitors + nearby memory support

review boundary
  reviewer cells positioned between repair output and organ selection

exploration front
  explorer cells migrating along ambiguity gradients

quiescent reserve
  stem/progenitor cells held near high-risk or high-demand regions
```

Functional clustering is therefore a product of local adhesion and morphogen gradients, not a static team assignment. Cells with the same or complementary fates accumulate around the same niche because they sense the same signals, express compatible adhesion profiles, and reduce coordination cost by staying close.

### Extracellular Interfaces and Effector Programs

An expressed gene biases effector programs. In an implementation that uses MCP, MCP is the engineering substrate for several biological interface concepts: extracellular matrix, membrane channels, receptors, induction factors, and returned signals.

The mapping should start from the biological interface and then identify the MCP implementation:

| Biological interface | MCP implementation | Ontocellia meaning |
| --- | --- | --- |
| Extracellular niche | MCP server | A bounded capability region in the microenvironment |
| Membrane channel / effector port | MCP tool | A callable action channel used by a cell |
| Extracellular matrix | MCP resource | Structured environmental material a cell can sense or read |
| Induction factor | MCP prompt/template | Reusable environmental pattern that can bias expression |
| Receptor binding surface | MCP tool schema | The contract that determines how a cell can recognize and use a channel |
| Returned metabolite / signal | MCP response | New environmental evidence that updates morphogen fields |

MCP is therefore placed inside the microenvironment and effector interface. Genes determine when and how a cell becomes competent to use those biological interfaces, and MCP provides one concrete protocol for implementing them.

Effector outputs may include:

- tool call preference
- MCP tool invocation
- MCP resource read
- prompt/control fragment
- message emission
- patch proposal
- validation command
- memory write
- stem division signal
- progenitor amplification signal
- apoptosis or quiescence signal
- niche refilling signal
- fate transition bias

This separation keeps gene as the lowest-level inherited unit while allowing many concrete runtimes.

### Mutation and Selection

Genes may mutate across lineage or tissue generations, but mutation must be constrained by organ selection. A mutation is retained only if it improves validated tissue function or regenerative capability.

Mutation records should include:

- source gene
- mutation objective
- changed fields
- validation result
- affected cell lineage
- organ-level outcome

This makes self-evolution auditable and keeps the biological core coherent.

## 7. Cell Types

The first practical tissue should support a small role landscape.

| Fate | Function | Inducing signals |
| --- | --- | --- |
| Stem | Preserve plasticity, divide, and generate replacement lineages | niche vacancy, regeneration pressure, early task phase |
| Progenitor | Expand a committed lineage before final specialization | local demand, missing function, moderate fate bias |
| Explorer | Search, inspect, gather context | ambiguity, missing information, large unknown surface |
| Planner | Form task decomposition and hypotheses | high ambiguity, broad context, weak execution confidence |
| Builder | Modify artifacts or produce implementation output | clear plan, implementation pressure, stable local context |
| Reviewer | Critique plans, code, claims, and risk | high risk, large diff, safety pressure, completion signal |
| Repair | Restore missing function after stem/progenitor replacement and local regeneration | cell damage, vacant niche, test failure, broken invariant, user correction |
| Memory | Preserve assumptions, decisions, evidence, and lineage | long task, repeated failures, coordination pressure |
| Quiescent | Stay inactive but available | low demand, high cost pressure, sufficient tissue coverage |

These are fate attractors. A cell may dedifferentiate and recommit when the microenvironment changes.

## 8. Specs

The framework should evolve toward three main spec families.

### AgentGenomeSpec

Defines the shared developmental program.

```yaml
metadata:
  name: repo-repair-genome

genes:
  - type: Gene
    id: gene_inspect_context
    category: exploration
    morphogen_affinity: [ambiguity, missing_context]
    expression_window:
      - task context is incomplete
    encoded_response:
      - Locate failing or high-signal artifacts first.
      - Read nearby context before broad search.
      - Emit findings as structured evidence.
    constraints:
      max_files: 8
    validation_hooks: []

  - type: Gene
    id: gene_repair_from_test_failures
    category: regeneration
    morphogen_affinity: [test_failure, regression, traceback]
    expression_window:
      - validation failure or niche vacancy is present
    encoded_response:
      - Extract failure signature.
      - Estimate blast radius.
      - Patch the narrowest cause.
      - Re-run validation.
    suppression_cues:
      - speculative broad rewrites
      - suppressing tests
    validation_hooks:
      - python -m pytest -q

fate_landscape:
  attractors:
    - name: explorer
      promoters: [ambiguity, missing_context]
      inhibitors: [verification_pressure]
    - name: repair
      promoters: [test_failure, damaged_artifact]
      inhibitors: [low_confidence]
```

### TaskMicroenvironmentSpec

Defines the task substrate and signal sources.

```yaml
metadata:
  name: failing-test-repair-environment

task:
  objective: Fix failing tests while preserving existing behavior.

signals:
  test_failure:
    source: pytest
    maps_to: repair_pressure
  large_diff:
    source: git
    maps_to: review_pressure
```

### TissueExperimentSpec

Defines reproducible experiments for the harness.

```yaml
metadata:
  name: repair-tissue-ablation

base:
  genome: genomes/repo_repair.yaml
  environment: environments/failing_tests.yaml

variants:
  - name: full_tissue
  - name: no_memory_cells
    genome_patch:
      fate_landscape:
        disabled_attractors: [memory]
```

## 9. Example Flow: Failing Test Repair Tissue with Cell Regeneration

User task:

```text
Fix the failing tests in this repository and preserve existing behavior.
```

Initial state:

```text
12 stem cells
shared AgentGenome
plastic fate landscape
```

The microenvironment emits:

```text
test_failure: high
ambiguity: high
implementation_pressure: medium
review_pressure: medium
cost_pressure: low
```

Developmental response:

```text
2 cells commit to planner fate
3 cells commit to explorer fate
3 cells commit to repair fate
1 cell commits to reviewer fate
1 cell commits to memory fate
2 cells remain quiescent stem cells
```

Positional patterning:

```text
explorer cells migrate toward the ambiguity gradient around failing logs
repair cells adhere around the damaged test/code niche
reviewer cells settle near the boundary between repair output and validation
memory cells remain adjacent to planner and repair domains
quiescent stem cells stay near the repair niche as reserve capacity
```

Execution:

```text
explorer cells inspect failing logs and related files
memory cell records hypotheses and failed attempts
repair cells express gene_repair_from_test_failures
repair cells propose candidate patches under gene constraints
reviewer cell checks risk and unintended changes
one repair cell stalls and emits a damage signal
the stalled repair cell enters apoptosis or is manually cleared
the local repair niche becomes vacant
nearby mature cells amplify vacancy and repair morphogens
one reserve stem cell divides into a stem cell and a progenitor cell
the progenitor cell enters a transit-amplifying repair lineage
the progenitor migrates into the vacant repair niche along the vacancy gradient
the repair progenitor differentiates into a replacement repair cell
an explorer cell contributes context signals while preserving its mature fate
new context is found
builder/repair cells produce a smaller patch
tests pass
review_pressure rises
reviewer validates behavior and risk
extra cells enter apoptosis or quiescence
```

Final tissue after regeneration:

```text
planner + explorers + regenerated repair cells + reviewer + memory cell
```

Organ function:

```text
repository returns to passing tests with a traceable repair lineage and evidence.
```

Regenerative lineage:

```text
the repair gene's suppression cues mutate from compressed lineage evidence
organ selection retains the mutation only if validation improves
lineage records the cleared cell, replacement cell, and fate transition
future repair tissues inherit the revised gene through the genome
```

Reprogramming path:

```text
if the stem/progenitor pool is exhausted, regeneration pressure can induce reprogramming
a selected differentiated cell pays reprogramming cost and returns toward a plastic state
organ selection validates the regenerated function before stabilizing the new fate
```

## 10. Near-Term Product Direction

The existing simulation runtime can remain as a reference developmental sandbox, but the product direction should shift toward agent tissue orchestration.

Near-term milestones:

1. Rename public positioning from "developmental simulation framework" to "developmental agent tissue framework".
2. Add an `AgentGenomeSpec` draft that treats genes as the lowest-level units, with morphogen affinity, expression windows, encoded responses, suppression cues, constraints, validation hooks, heritability, and regulatory elements.
3. Add a minimal non-LLM harness demo using rule-based cells to validate tissue dynamics.
4. Add an LLM-backed cell adapter after traces, budgets, prompts, and reproducibility hooks are defined.
5. Build the first task tissue: failing-test repair or research synthesis.

LLMs enter as one kind of cell execution substrate. The developmental harness remains model-agnostic enough to support LLM cells, rule cells, tool cells, evaluator cells, and memory cells.

## 11. Open Design Questions

- Should a tissue have one shared memory substrate, local cell memories, or both?
- How much global visibility should organ selection have before the system stops being decentralized?
- Should fate commitment be explicit and inspectable, or implicit in prompt/tool selection?
- Should cells communicate through direct messages, shared extracellular matrix, or both?
- What is the minimal first tissue that proves the framework: repo repair, research synthesis, or long-running project maintenance?
- What is the right limit on simultaneous gene expression before control signal becomes diluted?
- How should external experience be assimilated into the genome while preserving gene as the lowest-level endogenous unit?

## References

- [From Procedural Skills to Strategy Genes: Towards Experience-Driven Test-Time Evolution](https://arxiv.org/abs/2604.15097)
- [EvoMap/evolver](https://github.com/EvoMap/evolver)
- [EvoMap/critpt-openclaw-reproducible-70](https://github.com/EvoMap/critpt-openclaw-reproducible-70)

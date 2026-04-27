# Ontocellia

Ontocellia is a developmental agent-tissue framework for building decentralized, self-organizing, self-repairing multi-agent systems.

Agents are modeled as cells. A shared genome defines inheritable genes, a task microenvironment releases morphogens, and initially plastic cells differentiate into a functional tissue around positioned niches. The bundled simulation and experiment tools are reference substrates for studying this framework.

## Features

- `ontocellia.framework` agent tissue primitives: `Gene`, `AgentGenome`, `TaskMicroenvironment`, `Niche`, `ExtracellularInterface`, `TissueRuntime`
- Task-induced differentiation from stem-like cells into explorer, planner, builder, repair, reviewer, memory, and quiescent fates
- Spatial niche occupancy, chemotaxis-like positioning, and stem/progenitor-mediated regeneration
- Extracellular interfaces for membrane channels, extracellular matrix, and future MCP/LLM/tool adapters
- Shared `GenomeProgram` with per-agent state divergence
- YAML/JSON `GenomeSpec`, `EnvironmentSpec`, and `ExperimentSpec`
- Continuous developmental dynamics with attractor commitment
- Long-range fields, short-range contact signaling, and background context
- Life processes: divide, differentiate, migrate, apoptose, dedifferentiate, quiesce
- Lightweight community formation and weak organ-level feedback
- Single-run outputs plus comparison artifacts for ablation studies

## Quick Start

```bash
conda env create -f environment.yml
conda activate ontocellia
python -m ontocellia tissue --genome-spec examples/framework/repo_repair_genome.yaml --environment-spec examples/framework/failing_tests_environment.yaml --steps 4 --output artifacts/repo_repair_tissue
python -m ontocellia induce --task "Fix failing tests" --domain repo_repair --output artifacts/induced
python -m ontocellia tissue --genome-spec artifacts/induced/genome.yaml --environment-spec artifacts/induced/environment.yaml --effector mock-llm --output artifacts/induced_tissue
python -m ontocellia run --steps 20 --output artifacts/demo
python -m ontocellia run --genome-spec examples/specs/minimal_genome.yaml --environment-spec examples/specs/minimal_environment.yaml --steps 40 --output artifacts/spec_demo
python -m ontocellia experiment --experiment-spec examples/experiments/contact_ablation.yaml --output artifacts/contact_ablation
python -m ontocellia validate --genome-spec examples/specs/minimal_genome.yaml --environment-spec examples/specs/minimal_environment.yaml
pytest
```

The legacy command form still works:

```bash
python -m ontocellia --steps 20 --output artifacts/legacy_demo
```

## Core Model

- `GenomeSpec`: shared developmental rule set
- `CellState`: local hidden state, development state, energy, stress, competence, history, receptor profile
- `Microenvironment`: diffusive fields, contact context, mechanical/resource context
- `FateLandscape`: attractors, competence windows, hysteresis, reprogramming cost
- `LifeProcessModel`: division, differentiation, migration, apoptosis, quiescence, repair response
- `OrganSelectionField`: weak global feedback through environment pressure, not central control

## Agent Tissue Framework

The first agent-framework layer lives in `ontocellia.framework`. It models a task-induced tissue directly:

- `Gene`: lowest-level endogenous genomic unit
- `AgentGenome`: heritable program shared by the tissue
- `MorphogenField`: global signals plus local graph-positioned morphogen sources
- `TissueTopology`: graph-first tissue position model with 3D embedding fallback
- `FateLandscape`: inspectable fate attractors, thresholds, niche bias, and hysteresis
- `TaskMicroenvironment`: objective, morphogens, topology, functional niches, and extracellular interfaces
- `Niche`: positioned functional region such as a repair niche or review boundary
- `ExtracellularInterface`: biological interface that can later be backed by MCP, shell tools, LLM calls, or local code
- `TissueRuntime`: deterministic harness for local sensing, graph migration, differentiation, regeneration, and effector action emission

Try the minimal tissue example:

```bash
python examples/framework/repo_repair_tissue.py
```

Or run the same idea from YAML:

```bash
python -m ontocellia tissue \
  --genome-spec examples/framework/repo_repair_genome.yaml \
  --environment-spec examples/framework/failing_tests_environment.yaml \
  --steps 4 \
  --output artifacts/repo_repair_tissue
```

This writes `tissue_summary.json` and `tissue_trace.json`.

## Induction And LLM Effectors

`ontocellia induce` is the compile-time induction layer: it turns a natural language task into genome and microenvironment specs.

```bash
python -m ontocellia induce \
  --task "Fix failing tests while preserving behavior" \
  --domain repo_repair \
  --output artifacts/induced
```

`--effector mock-llm` enables the runtime cell-effector layer. It translates expressed gene programs into structured action intents without calling external APIs.

```bash
python -m ontocellia tissue \
  --genome-spec artifacts/induced/genome.yaml \
  --environment-spec artifacts/induced/environment.yaml \
  --effector mock-llm \
  --output artifacts/induced_tissue
```

Real LLM provider adapters are intentionally separate from the deterministic mock provider.

Real provider calls use OpenAI-compatible chat completion endpoints. The current adapters are:

| Provider | Environment variable | Default base URL | Default model |
| --- | --- | --- | --- |
| `deepseek` | `DEEPSEEK_API_KEY` | `https://api.deepseek.com` | `deepseek-v4-flash` |
| `kimi` | `MOONSHOT_API_KEY` or `KIMI_API_KEY` | `https://api.moonshot.ai/v1` | `kimi-k2.6` |
| `minimax` | `MINIMAX_API_KEY` | `https://api.minimax.io/v1` | `MiniMax-M2.7` |

Example:

```bash
export DEEPSEEK_API_KEY=...
python -m ontocellia tissue \
  --genome-spec artifacts/induced/genome.yaml \
  --environment-spec artifacts/induced/environment.yaml \
  --effector deepseek \
  --llm-model deepseek-v4-flash \
  --output artifacts/deepseek_tissue
```

Provider calls write `action_intents.json` and `llm_trace.json`. API keys are read from environment variables and are not written into trace artifacts.

Some MiniMax token-plan keys are issued for a regional host. In that case, pass the matching base URL explicitly:

```bash
python -m ontocellia tissue \
  --genome-spec examples/framework/repo_repair_genome.yaml \
  --environment-spec examples/framework/failing_tests_environment.yaml \
  --effector minimax \
  --llm-base-url https://api.minimax.chat/v1 \
  --output artifacts/minimax_tissue
```

Live provider E2E tests are opt-in so regular test runs remain deterministic:

```bash
set -a
source .env.local
set +a
ONTOCELLIA_LIVE_LLM=1 conda run -n ontocellia python -m pytest -q tests/test_llm_live_e2e.py
```

## Experiments

Experiments compare a base model against named variants. v0.2 supports top-level patches for config, genome, and environment data.

```yaml
metadata:
  name: contact-signaling-ablation
base:
  genome: ../specs/minimal_genome.yaml
  environment: ../specs/minimal_environment.yaml
  steps: 40
  seed: 7
variants:
  - name: baseline
  - name: no_contact
    genome_patch:
      contact_programs: []
outputs:
  summary: true
  plots: true
  metrics_csv: true
  report: true
```

Running an experiment writes per-variant run directories plus `comparison.json`, `comparison.csv`, `comparison.png`, and `report.md`.

## Documentation

- [Architecture](docs/architecture.md)
- [GenomeSpec reference](docs/schema/genome-spec.md)
- [EnvironmentSpec reference](docs/schema/environment-spec.md)
- [ExperimentSpec reference](docs/schema/experiment-spec.md)

See [examples/specs/minimal_genome.yaml](examples/specs/minimal_genome.yaml), [examples/specs/minimal_environment.yaml](examples/specs/minimal_environment.yaml), and [examples/experiments](examples/experiments) for working setups.

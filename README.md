# Ontocellia

Ontocellia is a developmental multi-agent experiment framework for building decentralized, cell-like systems.

Agents share one genome-like program, but diverge through local state, microenvironment, contact signals, fate commitment, and weak system-level selection pressure. The bundled runtime is a reference implementation for reproducible developmental experiments.

## Features

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
- `TaskMicroenvironment`: objective, morphogens, functional niches, and extracellular interfaces
- `Niche`: positioned functional region such as a repair niche or review boundary
- `ExtracellularInterface`: biological interface that can later be backed by MCP, shell tools, LLM calls, or local code
- `TissueRuntime`: deterministic harness for development, regeneration, and effector action emission

Try the minimal tissue example:

```bash
python examples/framework/repo_repair_tissue.py
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

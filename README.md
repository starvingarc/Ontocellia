# Ontocellia

Ontocellia is a developmental agent framework for building decentralized, cell-like multi-agent systems.

Agents share one genome-like program, but diverge through local state, microenvironment, contact signals, and weak system-level selection pressure. The bundled runtime is a reference implementation of that architecture.

## Features

- Shared `GenomeProgram` with per-agent state divergence
- Continuous developmental dynamics with attractor commitment
- Global environment + local microenvironment separation
- Long-range fields, short-range contact signaling, and background context
- Life processes: divide, differentiate, migrate, apoptose, dedifferentiate, quiesce
- Lightweight community formation for higher-order organization
- Reference runtime, specs, plots, and tests included

## Quick Start

```bash
python -m pip install -e .
python -m ontocellia --steps 80 --output artifacts
python -m ontocellia --genome-spec examples/specs/minimal_genome.yaml --environment-spec examples/specs/minimal_environment.yaml --output spec_artifacts
pytest
```

## Core Model

- `Genome`: shared developmental rule set
- `Cell State`: internal state, fate state, energy, age, competence, history, receptor profile
- `Microenvironment`: diffusive fields, contact context, mechanical/resource context
- `Fate Landscape`: attractors, competence windows, hysteresis, reprogramming cost
- `Life Processes`: division, differentiation, migration, apoptosis, community formation
- `Organ Selection`: weak global feedback through environment pressure, not central control

## Specs

- `GenomeSpec` defines the shared developmental program
- `EnvironmentSpec` defines global environment, spatial environment, and task translation
- `GeneAsset` is an optional runtime modulation layer, not the genome itself

See [examples/specs/minimal_genome.yaml](examples/specs/minimal_genome.yaml) and [examples/specs/minimal_environment.yaml](examples/specs/minimal_environment.yaml) for a minimal setup.

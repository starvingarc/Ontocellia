# Ontocellia

Ontocellia is a biologically grounded Python runtime for decentralized developmental agents.
Cells share one genome-like rule set and specialize through morphogen gradients,
contact inhibition, competence windows, microenvironmental context, and epigenetic locking.

The core abstraction is:

- `Genome layer`: shared developmental rule set
- `Cell state layer`: per-cell internal state and attractor dynamics
- `Microenvironment layer`: long-range morphogens, short-range contact signals, slow background context
- `Fate engine layer`: continuous landscape with multi-step commitment
- `Communication layer`: diffusive secretion, neighbor-only contact signals, and behavior actions

## What v1 demonstrates

- Self-organization from initially homogeneous cells
- Resource-driven division and bounded growth
- Differentiation through commitment dynamics
- Self-repair after local damage
- Hybrid substrate behavior through 2D fields plus a dynamic interaction graph
- Strategy and warning genes as structured control objects

## Quick start

```bash
python -m pip install -e .
python -m ontocellia --steps 80 --output artifacts
python -m ontocellia --genome-spec examples/specs/minimal_genome.yaml --environment-spec examples/specs/minimal_environment.yaml --output spec_artifacts
pytest
```

The demo writes a JSON summary and several plots into the chosen output folder.

## Biologically grounded spec mode

- `GenomeSpec` defines the shared developmental rule set
- `EnvironmentSpec` defines diffusive fields, contact context, background context, sources, events, and task translation
- `GeneAsset` remains a runtime strategy/warning modulation layer and is not the same thing as the genome

The default v3 examples include:

- long-range morphogens `M1`, `M2`, `M3`
- `NotchLike` short-range contact inhibition
- slow background context via `ECM`, `mechanical_stress`, `nutrient`, and `damage`
- competence windows and hysteresis / epigenetic lock
- attractor-based fate landscape with multi-step commitment

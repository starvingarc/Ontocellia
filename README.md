# Ontocellia

Ontocellia is a developmental agent-tissue framework for decentralized, self-organizing, self-repairing multi-agent systems.

Agents are modeled as cells. A single stem-origin cell proliferates and differentiates under a task microenvironment; the resulting tissue uses a shared genome, morphogen field, topology, fate landscape, organ selection layer, and communication matrix to coordinate around the task.

## Core Ideas

- **Genome first:** genes are endogenous control units for expression, validation tendencies, and fate bias.
- **Single-stem origin:** a task tissue starts from one stem-like cell, expands, then differentiates.
- **Cells as agents:** each cell carries stage, fate, lineage, receptors, adhesion, energy, and local history.
- **Task-induced tissue:** morphogens, niches, and topology guide differentiation and clustering.
- **Weak organ selection:** validation, risk, cost, and coverage become bounded feedback, not central control.
- **Shared matrix memory:** cells exchange messages, handoffs, and evidence through an extracellular matrix.
- **Optional effectors and execution:** providers emit structured intents; explicit policies can execute safe local actions.

## Quick Start

```bash
conda env create -f environment.yml
conda activate ontocellia

python -m ontocellia tissue \
  --genome-spec examples/framework/repo_repair_genome.yaml \
  --environment-spec examples/framework/failing_tests_environment.yaml \
  --steps 4 \
  --output artifacts/repo_repair_tissue
```

This writes `tissue_summary.json` and `tissue_trace.json`.

For the interactive tissue TUI:

```bash
python -m ontocellia
```

Type a task in the TUI to induce and run an agent tissue, or use `/setup` to configure model providers first.

For a deterministic tissue benchmark:

```bash
python -m ontocellia benchmark
```

Adaptive benchmark runs and provider baselines are documented in [docs/usage.md](docs/usage.md).

## Documentation

- [Framework overview](docs/framework.md)
- [Usage guide](docs/usage.md)
- [Communication layer](docs/communication.md)
- [Architecture](docs/architecture.md)
- [Design document](docs/agent-tissue-design.md)
- [Roadmap](docs/roadmap.md)
- [Schema reference](docs/schema)

## Examples

- Framework tissue specs: [examples/framework](examples/framework)
- Ablation experiments: [examples/experiments](examples/experiments)
- Minimal legacy specs: [examples/specs](examples/specs)

## Development

```bash
conda activate ontocellia
python -m pytest -q
```

Live LLM tests are opt-in. See [docs/usage.md](docs/usage.md).

# Ontocellia

Ontocellia is a developmental agent-tissue framework for decentralized, self-organizing, self-repairing multi-agent systems.

Agents are modeled as cells. A shared genome, task microenvironment, morphogen field, tissue topology, fate landscape, organ selection layer, and communication matrix together induce a functional tissue around a task.

## Core Ideas

- **Genome first:** genes are endogenous control units for expression, validation tendencies, and fate bias.
- **Cells as agents:** each cell carries stage, fate, lineage, receptors, adhesion, energy, and local history.
- **Task-induced tissue:** morphogens, niches, and topology guide differentiation and clustering.
- **Weak organ selection:** validation, risk, cost, and coverage become bounded feedback, not central control.
- **Shared matrix memory:** cells exchange messages, handoffs, and evidence through an extracellular matrix.
- **Optional effectors:** rule-based, mock LLM, and OpenAI-compatible providers emit structured action intents.

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

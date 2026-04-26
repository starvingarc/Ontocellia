from __future__ import annotations

from pathlib import Path


DOCS = {
    "genome-spec.md": """# GenomeSpec

`GenomeSpec` defines the shared developmental program used by cells in spec mode.

Required top-level sections:

- `metadata`
- `state_dims`
- `sensing`
- `behavior_biases`
- `morphogens`
- `contact_programs`
- `background_context`
- `secretion_programs`
- `competence_windows`
- `epigenetic_lock`
- `lineage_rules`
- `task_coupling`
- `fate_landscape`
- `reporting_probes`

Vector lengths must match `state_dims.development_dim` unless documented otherwise.
""",
    "environment-spec.md": """# EnvironmentSpec

`EnvironmentSpec` defines the global task, spatial environment, fields, contact
context, sources, scheduled events, and task translation zones.

Required top-level sections:

- `metadata`
- `global_environment`
- `spatial_environment`
- `task_translation`

Field initial patterns currently support `constant`, `x_gradient`, `y_gradient`,
and `radial`.
""",
    "experiment-spec.md": """# ExperimentSpec

`ExperimentSpec` defines one reproducible experiment with a base model and one
or more variants.

Required top-level sections:

- `metadata`
- `base`

Optional sections:

- `variants`
- `outputs`

Each variant may provide `config_patch`, `genome_patch`, and
`environment_patch`. v0.2 patches are top-level mapping updates.
""",
}


def export_schema_docs(output_dir: str | Path) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename, text in DOCS.items():
        path = output_path / filename
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths

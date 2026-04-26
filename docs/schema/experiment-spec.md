# ExperimentSpec

`ExperimentSpec` defines one reproducible experiment with a base model and one or more variants.

Required top-level sections:

- `metadata`
- `base`

Optional sections:

- `variants`
- `outputs`

Each variant may provide `config_patch`, `genome_patch`, and `environment_patch`. v0.2 patches are top-level mapping updates.

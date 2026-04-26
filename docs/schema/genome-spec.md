# GenomeSpec

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

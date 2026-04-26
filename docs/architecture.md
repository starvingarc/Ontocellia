# Ontocellia Architecture

Ontocellia is organized as a developmental experiment framework rather than a general-purpose agent platform.

## Layers

- **Spec layer:** `GenomeSpec`, `EnvironmentSpec`, and `ExperimentSpec` describe the shared developmental program, spatial task environment, and reproducible experiment variants.
- **Compile layer:** genome and environment compilers turn declarative specs into matrices, field maps, event lists, and runtime-ready structures.
- **Runtime layer:** `OntocelliaRuntime` remains the public facade. `StepPipeline` owns deterministic step orchestration and delegates mature legacy/spec transition behavior to the runtime.
- **Developmental model layer:** `GenomeProgram`, `FateLandscape`, `LifeProcessModel`, `Microenvironment`, `InteractionGraph`, `CommunityState`, and `OrganSelectionField` preserve the project's original abstraction: shared genome, local divergence, fate commitment, and weak global feedback.
- **Experiment layer:** `ExperimentRunner` expands variants, applies top-level patches, runs simulations, and writes comparison artifacts.
- **Observation layer:** summary JSON, metrics CSV, plots, comparison JSON/CSV/PNG, and Markdown reports make mechanisms easier to inspect and compare.

## v0.2 Design Principle

The v0.2 boundary is intentionally conservative. Experiments can compare mechanisms without turning Ontocellia into a clone of Mesa, Morpheus, or PhysiCell. Future changes should add new mechanisms through specs, pipeline stages, or experiment tooling instead of expanding the runtime facade directly.

# Repository Review Findings

This note records the current expert-review synthesis so future phases stay aligned with the project's real risks.

## Positioning

Ontocellia is not best described as another fixed multi-agent framework. Its strongest positioning is a developmental harness for growing task-induced agent tissues, comparing tissue structures, explaining their contributions, and solidifying reusable developmental tendencies.

The core claim is therefore empirical: adaptive tissue organization should eventually reduce cost, improve recovery, improve context reuse, or improve validation outcomes compared with a direct agent loop or a fixed agent graph.

## Findings

- The conceptual layers are now coherent: genome, cell, morphogen field, fate landscape, matrix memory, context/output metabolism, tool gates, structure search, attribution, resource pressure, annealing, and solidification all exist as code.
- The official benchmark layer must keep external scorer results separate from Ontocellia structure metrics. Internal structure scores should not be presented as official pass/fail.
- The execution layer has good default safety boundaries, but real-workspace use needs stronger server-side policy ownership, per-invocation approval provenance, read/write glob controls, and secret redaction before context is exported to providers.
- The codebase has accumulated large orchestration files. `__main__.py`, `official_benchmark.py`, `execution.py`, and `communication.py` should be split as behavior-preserving refactors rather than extended indefinitely.
- The docs and first-run experience should explain what problem Ontocellia solves in practical engineering terms: bounded context, generated agent organization, policy-gated tools, and structure-level evaluation.

## Immediate Direction

Phase33 adds longitudinal replay and controlled baselines. It is the first step toward proving that adaptive tissue structure has measurable value beyond a single provider's raw ability.

The next phases should prioritize:

- cost-normalized comparisons against direct-agent and fixed-graph baselines;
- anti-gaming metrics where matrix/handoff/fate diversity count only when reused by successful downstream actions;
- secret-safe tool execution and provider egress controls;
- official scorer loops that feed real pass/fail back into organ selection.

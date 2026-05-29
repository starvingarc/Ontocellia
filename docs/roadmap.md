# Roadmap

Ontocellia is being built as a framework-first developmental agent tissue system. The repository already contains the core runtime layers for genome expression, cell state, induction, LLM effectors, morphogen fields, organ feedback, communication, context metabolism, output metabolism, policy-gated tools, app-server sessions, benchmark adapters, and structure search.

Phase numbers are historical implementation markers. Web Lab and official harness expansion were intentionally deferred while the adaptive-structure and benchmark-adapter layers were built first.

## Review Findings Integrated

The latest repository review found that Ontocellia's strongest differentiator is now clear: it is a developmental harness for growing, comparing, explaining, and solidifying task-induced multi-agent tissues. The next risk is not lack of concepts, but lack of proof that adaptive tissue organization improves real task outcomes over simpler agent loops.

The review also identified four engineering pressure points:

- official benchmark outputs must distinguish Ontocellia tissue metrics from actual official scorer pass/fail;
- execution and app-server approval paths need stricter policy ownership, secret-safe output handling, and per-invocation provenance before broader real-task use;
- `__main__.py`, `official_benchmark.py`, and the execution/communication modules are large enough that future phases should include behavior-preserving refactors;
- product docs should keep README concise while adding clearer first-run and concept guides for new users.

## Implemented Capabilities

- Phase 1: Genome layer
- Phase 2: Cell layer
- Phase 3A: Induction compiler
- Phase 3B: LLM effector layer
- Phase 4: Morphogen field, tissue topology, and fate landscape
- Phase 5: Organ selection layer
- Phase 6: Communication, handoff, and shared matrix memory
- Phase 7: Validation hook runner
- Phase 8: MCP / extracellular interface adapter
- Phase 9: Mutation-selection-solidification
- Phase 10: End-to-end task tissue demo
- Phase 11: Single-stem developmental runtime and interactive TUI
- Phase 12: Tissue benchmark harness
- Phase 13: Extracellular execution layer
- Phase 14: Adaptive tissue benchmark protocol, with BFCL retained as provider baseline
- Phase 15: Context homeostasis layer
- Phase 16: Extracellular tool runtime
- Phase 17: Living tissue app server with HTTP and WebSocket APIs
- Phase 18: Context metabolism and matrix remodeling layer
- Phase 19: Output metabolism for tool and validation output
- Phase 22: Structure search and tissue variant trials
- Phase 23: Official benchmark source fidelity and scoring-status reporting
- Phase 24: Official benchmark structure-search integration
- Phase 25: Official scorer execution wrapper
- Phase 26: Benchmark-induced repair specialization
- Phase 27: Official scorer adapters
- Phase 28: Official custom agent adapters for Terminal-Bench and tau-bench bridge flows
- Phase 29: Contribution attribution and causal trace layer
- Phase 30: Resource competition and population pressure
- Phase 31: Developmental annealing and reprogramming control
- Phase 32: Selection solidification v2
- Phase 33: Longitudinal replay and controlled baseline comparison

## Upcoming Work

- Phase 34: Controlled baseline expansion and anti-gaming metrics
  Promote direct-agent, single-cell, fixed-tissue, and adaptive-tissue comparisons to first-class evaluation reports. Keep official success, validation pass rate, and cost-normalized success as primary metrics; matrix, handoff, and fate diversity remain explanatory secondary metrics.
- Phase 35: Policy-governed tool approval and secret-safe execution
  Move server approval to fixed server-side policy profiles, add per-invocation approval provenance, read-glob controls, secret redaction, and provider egress flags before running broader real workspace tasks.
- Phase 36: Official scorer first-class loop
  Close the SWE-bench, Terminal-Bench, and tau-bench scorer paths so official pass/fail can flow back into organ selection and longitudinal replay without benchmark-specific cell behavior.
- Phase 37: Runtime and CLI consolidation
  Split large command, benchmark, execution, and artifact-writing modules while preserving current behavior and tests.

## Deferred Product Surfaces

- Phase 20: Web Lab petri-dish frontend implementation based on the committed design concept.
- Phase 21: Broader official harness expansion for tau-bench, SWE-bench Lite, Terminal-Bench, and multi-agent collaboration suites.

## Project Direction

The reference simulation runtime can remain a developmental sandbox. The main framework direction is an agent tissue orchestration harness with decentralized coordination, traceable self-repair, bounded tool execution, and adaptive structure formation.

Ontocellia grows candidate structures from task pressure, metabolizes evidence, applies weak selection, compares variants, and gradually converges toward task-fit tissue organizations.

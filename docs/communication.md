# Communication Layer

The communication layer lets cells coordinate without turning the tissue into a centrally planned agent runner. It combines short-lived messages with durable extracellular matrix records.

This document is a developer reference for matrix memory, handoffs, context retrieval, and context/output metabolism. For command recipes, see [usage.md](usage.md).

## Concepts

- `TissueMessage`: short-lived communication emitted by a cell.
- `MessageScope`: `direct`, `local`, `fate`, or `broadcast`.
- `MatrixRecord`: durable evidence or memory deposited into the extracellular matrix.
- `ExtracellularMatrix`: shared queryable substrate for observations, hypotheses, validation notes, and memories.
- `ContextPacket`: bounded fate-aware context view retrieved from the matrix for one cell.
- `ContextMetabolismRuntime`: deterministic matrix-remodeling pass that condenses raw evidence into traceable metabolites.
- `HandoffRequest`: explicit transfer of work from one cell to a target fate.
- `HandoffReceipt`: traceable acceptance of a handoff by a recipient cell.
- `CommunicationPolicy`: routing and promotion limits.

## Common Tuning Tasks

- Limit prompt context size with `communication.context_budget_chars`.
- Limit matrix retrieval breadth with `communication.matrix_query_limit`.
- Turn matrix remodeling on or off with `communication.context_metabolism.enabled`.
- Inspect cell context provenance through `llm_effector` trace events and `ActionIntent.payload.context_record_ids`.
- Inspect condensed records through `context_metabolite_deposited` and `context_metabolism` events.

## Routing

- Direct messages go only to `recipient_cell_id`.
- Local messages go to cells on the same graph node, graph neighbors, or the same region.
- Fate messages go to cells with the requested fate.
- Broadcast messages are capped by `broadcast_limit`.

Routing records `message_emitted` and `message_delivered` events in `tissue_trace.json`.

## Matrix Promotion

Messages become matrix records when they are high-confidence observations or explicit memory records. Matrix records can be queried by tags, fate, and graph position.

Records may expire through `expires_tick`. Memory records are durable by default.

## Context Homeostasis

The matrix also controls context lifecycle. Each record can carry:

- `status`: `active`, `suppressed`, or `corrected`
- `validation_status`: `unverified`, `validated`, `failed`, or `contradicted`
- `references`: source action, execution, validation, or corrected record IDs
- `lineage_id`: optional cell lineage anchor
- `salience` and `decay_rate`: deterministic freshness controls
- `corrects_record_id`: link from a correction to the earlier record

Cells do not receive the full matrix. `query_context` builds a `ContextPacket` scored by tag overlap, fate, graph locality, confidence, salience, freshness, validation status, and accepted interfaces. `CellPromptBuilder` writes the selected records into `relevant_matrix` and traces their IDs as `context_record_ids`.

Execution and validation feedback become matrix evidence. Passed results validate related records; failed results can suppress weak or contradicted records and deposit a validation record that references the original evidence.

## Context Metabolism

Context metabolism treats context compression as matrix remodeling rather than one-shot summarization. During development, the tissue digests recent matrix records and trace events into metabolite `MatrixRecord` entries. Raw records remain available, but their salience is weakly lowered after they have been condensed.

The deterministic metabolites are:

- `failure_signature`: failed validation, failed execution, regression, and test-failure evidence.
- `episode_summary`: recent active observations and memory from the current tick window.
- `causal_chain`: LLM effector, message, handoff, execution, and validation chains.
- `constraint_digest`: task, policy, constraint, and culture-medium changes.
- `toxic_context`: contradicted, failed, corrected, or suppressed records that should influence retrieval cautiously.

Metabolite records use `kind: context_metabolite`, carry tags such as `metabolite` and the metabolite name, and store source metadata:

```json
{
  "metabolite_kind": "failure_signature",
  "source_record_ids": ["validation-1", "execution-2"],
  "source_trace_event_ids": ["trace:12"],
  "compression_level": "metabolite",
  "lossiness": "bounded",
  "source_count": 2,
  "scope": "validation"
}
```

`CellPromptBuilder` still includes the compatible `relevant_matrix` field. It also groups selected metabolite records into `context_metabolites` and lists non-metabolite records as `raw_context_record_ids`.

## Handoffs

An action intent can request a handoff through payload fields:

```json
{
  "message": "Patch ready for review.",
  "handoff_to_fate": "reviewer",
  "matrix_tags": ["patch", "review"]
}
```

The communication runtime emits:

- `handoff_requested`
- `handoff_completed`

## YAML

```yaml
communication:
  matrix_query_limit: 5
  context_budget_chars: 1600
  default_ttl: 3
  promote_confidence_threshold: 0.6
  allow_broadcast: true
  broadcast_limit: 8
  context_metabolism:
    enabled: true
    window_ticks: 3
    max_metabolites_per_tick: 4
    max_metabolite_chars: 700
    min_source_records: 2
    source_salience_decay: 0.15

matrix:
  records:
    - kind: observation
      content: Existing failing test evidence.
      tags: [test_failure, repo]
      confidence: 0.8
      status: active
      validation_status: unverified
      salience: 0.8
      position:
        node_id: repair-niche
        region: repo/tests
```

Existing `matrix: {}` input remains valid.

## Summary Output

`tissue_summary.json` includes:

- `messages`
- `matrix_records`
- `handoffs`

Detailed delivery, matrix, and handoff events are in `tissue_trace.json`.
LLM effector events additionally include `context_record_ids` so an intent can be traced back to the exact matrix records that shaped it.
Context metabolism adds `context_metabolite_deposited` and `context_metabolism` events to the same trace.

## Output Metabolism Metadata

Tool execution and validation hooks can produce long stdout, stderr, or evidence. Output metabolism keeps result JSON and prompt context bounded while preserving raw artifacts. When output exceeds the inline budget, raw text is written under `raw_outputs/` and inline evidence becomes a deterministic digest.

Matrix records deposited from tool and validation results include metadata such as:

```json
{
  "raw_output_path": "raw_outputs/tool-0-evidence.txt",
  "raw_output_chars": 20480,
  "digest_kind": "execution",
  "truncated": true,
  "source_result_id": "tool-0"
}
```

`tissue_summary.json` also reports `raw_outputs`, `truncated_outputs`, and `output_digest_chars` so users can see whether a run produced large external output.

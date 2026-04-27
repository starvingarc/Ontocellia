# Communication Layer

The communication layer lets cells coordinate without turning the tissue into a centrally planned agent runner. It combines short-lived messages with durable extracellular matrix records.

## Concepts

- `TissueMessage`: short-lived communication emitted by a cell.
- `MessageScope`: `direct`, `local`, `fate`, or `broadcast`.
- `MatrixRecord`: durable evidence or memory deposited into the extracellular matrix.
- `ExtracellularMatrix`: shared queryable substrate for observations, hypotheses, validation notes, and memories.
- `HandoffRequest`: explicit transfer of work from one cell to a target fate.
- `HandoffReceipt`: traceable acceptance of a handoff by a recipient cell.
- `CommunicationPolicy`: routing and promotion limits.

## Routing

- Direct messages go only to `recipient_cell_id`.
- Local messages go to cells on the same graph node, graph neighbors, or the same region.
- Fate messages go to cells with the requested fate.
- Broadcast messages are capped by `broadcast_limit`.

Routing records `message_emitted` and `message_delivered` events in `tissue_trace.json`.

## Matrix Promotion

Messages become matrix records when they are high-confidence observations or explicit memory records. Matrix records can be queried by tags, fate, and graph position.

Records may expire through `expires_tick`. Memory records are durable by default.

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
  default_ttl: 3
  promote_confidence_threshold: 0.6
  allow_broadcast: true
  broadcast_limit: 8

matrix:
  records:
    - kind: observation
      content: Existing failing test evidence.
      tags: [test_failure, repo]
      confidence: 0.8
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

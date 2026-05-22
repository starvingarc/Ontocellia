# Ontocellia Web Lab Design Concept

This document records the accepted product direction for the future Ontocellia browser interface. It is a design target, not the current implementation.

![Ontocellia Web Lab concept](assets/ontocellia-web-lab-concept.png)

## Product Metaphor

Ontocellia should feel like a living laboratory for task-grown agent tissues. A session is a culture dish. A task is the culture medium. Cells are decentralized agents that proliferate, differentiate, communicate, repair, and leave evidence in the extracellular matrix.

The Web Lab should make that biological model visible as more than an agent roster. Users should see how one task-induced tissue forms, how cells specialize, what they communicate, which evidence becomes shared memory, and where tool requests are waiting for approval.

## Information Architecture

The home surface is a petri dish wall. It should show multiple sessions as culture cards with recognizable dish thumbnails, current development stage, cell count, fate distribution, validation confidence, and risk. The wall is for comparison and triage: which tissue is alive, mature, risky, blocked, or ready for review.

The session surface is the live culture console. It should put the active dish at the center, with a lightweight project/session rail on the left and a focused cell inspector on the right. The lower area holds timeline replay, tool approval, matrix records, handoffs, and reports.

The app should keep three levels distinct:

- Project: a local workspace containing many tissue sessions.
- Session: one task-induced culture dish with lineage and artifacts.
- Cell: one local agent with fate, receptors, expressed genes, energy, memory, and traceable actions.

## Visual Direction

The style is Soft Lab Console: clean, bright, glassy, calm, and warm while remaining instrument-grade. The interface should feel precise enough for research and inviting enough that the user wants to watch the tissue grow.

Preferred visual traits:

- white and pale mint backgrounds with translucent glass panels;
- soft teal, blue, peach, violet, and yellow fate colors;
- circular dish forms as the dominant motif;
- lightly anthropomorphic cells with clear stage/fate identity;
- thin graph topology lines and message arcs inside the dish;
- subtle morphogen gradients as visible field pressure;
- rounded but disciplined panels, closer to a lab instrument than a marketing dashboard.

The central dish must be the visual anchor. Sidebars and inspector panels should support it, not compete with it.

## Core Interaction Model

Natural language input inside a session is culture-medium exchange. It changes the environment of the current tissue rather than starting a normal chat thread. The system should translate that input into morphogen pressure, matrix records, and development ticks while preserving lineage continuity.

Expected user operations:

- create a new culture from a task;
- change the medium with natural language;
- run or step tissue development;
- pause and scrub the timeline;
- toggle topology, morphogen, message, matrix, and tool layers;
- inspect a cell and follow its causal chain;
- clear or freeze a cell and watch self-repair;
- approve or reject pending tool invocations;
- open a visual report and raw artifacts.

## Causal Chain

The frontend should expose why the system did something. A user should be able to click from:

```text
morphogen signal
-> gene expression
-> cell fate / competence
-> ActionIntent
-> message / handoff / tool invocation
-> matrix record
-> validation or organ feedback
```

This causal chain differentiates Ontocellia from transcript-first chat UIs.

## Safety Model

The UI must not bypass runtime policy. Tool approval is a membrane-channel operation:

- the cell receptor must accept the interface;
- the environment must declare the interface;
- the project policy must allow the action;
- write operations and shell commands need explicit approval;
- dry-run remains the default.

The approval queue should show pending invocations, requested interface, target, scope, and requesting cell. A later UI pass can join `ToolInvocation` records with originating `ActionIntent` and `ToolResult` records to show rationale and risk estimates.

## Implementation Notes

The current committed backend already exposes the live-session surfaces needed by a future Web Lab: project/session listing, medium change, interventions, tool approval, cell details, intents, matrix records, handoffs, tools, artifacts, and WebSocket events.

The first browser implementation attempt is intentionally not part of this design document. The next implementation should treat the concept image above as the fidelity target and rebuild the frontend around the product model described here.

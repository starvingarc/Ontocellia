# Resource Competition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Phase30 resource competition so cell energy, population pressure, contribution feedback, and structure-search cost efficiency form a deterministic survival loop.

**Architecture:** Introduce a focused `ontocellia.framework.resources` module with policy, report, and runtime classes. `TissueRuntime.develop()` applies weak resource competition every tick, and optional attribution reports can reward or penalize cells without changing the LLM/tool execution model.

**Tech Stack:** Python dataclasses, existing `TissueRuntime`, `ContributionReport`, YAML loader, pytest.

---

### Task 1: Resource Competition Core

**Files:**
- Create: `src/ontocellia/framework/resources.py`
- Modify: `src/ontocellia/framework/__init__.py`
- Test: `tests/test_resource_competition.py`

- [ ] **Step 1: Write failing tests for policy, cell energy, and contribution feedback**

Tests should import `ResourceCompetitionPolicy` and `ResourceCompetitionRuntime`, apply them to a repo-repair tissue, and assert that maintenance costs lower energy, positive attribution rewards a cell, and negative attribution increases stress.

- [ ] **Step 2: Run the tests and confirm failure**

Run: `conda run -n ontocellia python -m pytest -q tests/test_resource_competition.py`

Expected: import failure for `ontocellia.framework.resources`.

- [ ] **Step 3: Implement resource dataclasses and runtime**

Implement:
- `ResourceCompetitionPolicy`
- `CellResourceDelta`
- `ResourceCompetitionReport`
- `ResourceCompetitionRuntime.apply(tissue, contribution_report=None, actions=None, tool_results=None, validation_results=None)`

The runtime adjusts cell energy, stress, resource morphogens, and records `resource_competition` trace events.

- [ ] **Step 4: Run targeted tests**

Run: `conda run -n ontocellia python -m pytest -q tests/test_resource_competition.py`

Expected: all tests pass.

### Task 2: Runtime And Spec Integration

**Files:**
- Modify: `src/ontocellia/framework/core.py`
- Modify: `src/ontocellia/framework/specs.py`
- Modify: `src/ontocellia/__main__.py`

- [ ] **Step 1: Add failing tests for runtime summaries and YAML policy loading**

Tests should assert `tissue.last_resource_report` exists after development, summary JSON includes resource fields, and YAML `resources:` config loads into `TaskMicroenvironment`.

- [ ] **Step 2: Implement integration**

Add `resource_policy` to `TaskMicroenvironment`, `resource_runtime` and `last_resource_report` to `TissueRuntime`, call resource competition in `develop()`, parse YAML `resources`, and include resource summary in CLI output.

- [ ] **Step 3: Run targeted tests**

Run: `conda run -n ontocellia python -m pytest -q tests/test_resource_competition.py tests/test_single_stem_developmental_runtime.py`

Expected: all tests pass.

### Task 3: Structure Search And Docs

**Files:**
- Modify: `src/ontocellia/framework/structure_search.py`
- Modify: `docs/framework.md`
- Modify: `docs/usage.md`
- Modify: `docs/roadmap.md`

- [ ] **Step 1: Add tests for resource-aware metrics**

Structure-search tests should assert metrics include `resource_efficiency`, `average_cell_energy`, and `population_pressure`.

- [ ] **Step 2: Implement metrics and docs**

Fold resource report data into structure-search metrics and mark Phase30 implemented in roadmap.

- [ ] **Step 3: Run regression**

Run:
```bash
conda run -n ontocellia python -m pytest -q tests/test_resource_competition.py tests/test_structure_search.py tests/test_contribution_attribution.py
conda run -n ontocellia python -m pytest -q
git diff --check
```

Expected: all tests pass and diff check is clean.

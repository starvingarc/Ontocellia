# Usage Guide

This guide covers the current command-line workflows. Commands assume the `ontocellia` conda environment is active.

## Install

```bash
conda env create -f environment.yml
conda activate ontocellia
```

## Run A Framework Tissue

```bash
python -m ontocellia tissue \
  --genome-spec examples/framework/repo_repair_genome.yaml \
  --environment-spec examples/framework/failing_tests_environment.yaml \
  --steps 4 \
  --output artifacts/repo_repair_tissue
```

Outputs:

- `tissue_summary.json`
- `tissue_trace.json`

## Induce Specs From A Task

```bash
python -m ontocellia induce \
  --task "Fix failing tests while preserving behavior" \
  --domain repo_repair \
  --output artifacts/induced
```

Run the induced tissue:

```bash
python -m ontocellia tissue \
  --genome-spec artifacts/induced/genome.yaml \
  --environment-spec artifacts/induced/environment.yaml \
  --effector mock-llm \
  --output artifacts/induced_tissue
```

## Use Organ Selection Results

Organ selection consumes structured validation results. By default, validation hooks remain metadata and are not executed.

```bash
python -m ontocellia tissue \
  --genome-spec examples/framework/repo_repair_genome.yaml \
  --environment-spec examples/framework/failing_tests_environment.yaml \
  --validation-result examples/framework/validation_failed.json \
  --steps 4 \
  --output artifacts/organ_selection_tissue
```

To execute hooks, use the opt-in Validation Hook Runner and explicitly allow each command:

```bash
python -m ontocellia tissue \
  --genome-spec examples/framework/repo_repair_genome.yaml \
  --environment-spec examples/framework/failing_tests_environment.yaml \
  --steps 4 \
  --run-validation-hooks \
  --allow-validation-hook "python -m pytest -q" \
  --output artifacts/validation_runner_tissue
```

The runner uses exact command allowlisting and does not execute through a shell. It writes `validation_results.json`, records hook events in `tissue_trace.json`, and feeds the resulting `OrganValidationResult` records back into organ selection.

## MCP Adapter Specs

MCP entries live inside the environment spec. Ontocellia maps them into biological interfaces without starting external MCP servers.

```yaml
mcp:
  servers:
    - id: repo
      tools:
        - name: read_file
          description: Read a workspace file.
          accepts_fates: [explorer, repair]
          input_schema:
            type: object
      resources:
        - id: failing-log
          uri: file://pytest.log
          content: 3 failing tests
          tags: [test_failure, repo]
          position:
            node_id: repair-niche
      prompts:
        - id: repair-protocol
          template: Inspect failure, patch narrow, validate.
          tags: [repair]
```

Loaded tools appear as `mcp:<server>:tool:<name>` membrane channels, resources become matrix records, and prompts become induction-factor interfaces. The tissue summary includes `mcp_interfaces`.

## Mutation Selection

Mutation selection compares baseline and candidate validation results. It writes mutation candidates, a decision report, and a solidified genome.

```bash
python -m ontocellia mutate \
  --genome-spec examples/framework/repo_repair_genome.yaml \
  --environment-spec examples/framework/failing_tests_environment.yaml \
  --baseline-validation examples/framework/validation_failed.json \
  --candidate-validation examples/framework/validation_passed.json \
  --output artifacts/mutation_selection
```

The input genome is never overwritten. If candidate validation does not improve, `solidified_genome.yaml` contains the original genome and the report marks the decision as `not_selected`.

## Complete Demo

The deterministic repo repair demo writes induced specs, tissue trace, mock LLM intents, validation evidence, mutation outputs, and a final report.

```bash
python -m ontocellia demo \
  --task "Fix failing tests while preserving behavior" \
  --steps 4 \
  --output artifacts/complete_repo_repair_demo
```

## LLM Effectors

Mock LLM mode is deterministic:

```bash
python -m ontocellia tissue \
  --genome-spec examples/framework/repo_repair_genome.yaml \
  --environment-spec examples/framework/failing_tests_environment.yaml \
  --effector mock-llm \
  --output artifacts/mock_llm_tissue
```

Real providers are optional. API keys are read from environment variables and are not written into trace artifacts.

| Provider | Environment variable | Default base URL |
| --- | --- | --- |
| `deepseek` | `DEEPSEEK_API_KEY` | `https://api.deepseek.com` |
| `kimi` | `MOONSHOT_API_KEY` or `KIMI_API_KEY` | `https://api.moonshot.ai/v1` |
| `minimax` | `MINIMAX_API_KEY` | `https://api.minimax.io/v1` |

MiniMax token-plan keys may require a regional host:

```bash
python -m ontocellia tissue \
  --genome-spec examples/framework/repo_repair_genome.yaml \
  --environment-spec examples/framework/failing_tests_environment.yaml \
  --effector minimax \
  --llm-base-url https://api.minimax.chat/v1 \
  --output artifacts/minimax_tissue
```

Live provider tests are opt-in:

```bash
set -a
source .env.local
set +a
ONTOCELLIA_LIVE_LLM=1 conda run -n ontocellia python -m pytest -q tests/test_llm_live_e2e.py
```

## Experiments

```bash
python -m ontocellia experiment \
  --experiment-spec examples/experiments/contact_ablation.yaml \
  --output artifacts/contact_ablation
```

Experiments write per-variant run directories plus comparison artifacts.

## Validation And Schema Docs

```bash
python -m ontocellia validate \
  --genome-spec examples/specs/minimal_genome.yaml \
  --environment-spec examples/specs/minimal_environment.yaml

python -m ontocellia schema-docs --output docs/schema
```

## Legacy Reference Runtime

```bash
python -m ontocellia run --steps 20 --output artifacts/demo
python -m ontocellia --steps 20 --output artifacts/legacy_demo
```

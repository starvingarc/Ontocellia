# Usage Guide

This guide covers the current command-line workflows. Commands assume the `ontocellia` conda environment is active.

## Install

```bash
conda env create -f environment.yml
conda activate ontocellia
```

## Interactive TUI

Start Ontocellia without arguments to enter the Soft Lab Console TUI:

```bash
python -m ontocellia
```

You can type a natural language task directly:

```text
Fix failing tests while preserving behavior.
```

The TUI induces specs, seeds a tissue, runs an initial collaboration pass, and shows agents, action intents, matrix records, handoffs, and a session report. TUI sessions write artifacts to `artifacts/tui_sessions/<session-id>/`.

Framework tissues start from one stem-origin cell by default. The runtime proliferates first, then differentiates cells into task-specific fates before emitting action intents.

Useful TUI commands:

```text
/setup
/models
/new <task>
/run [ticks]
/step
/agents
/intents
/matrix
/handoffs
/report
/mock
/exit
```

TTY launches use the Textual/Rich TUI. Non-TTY input falls back to the lightweight boxed shell for script compatibility. You can also force the TUI with:

```bash
python -m ontocellia tui
```

The setup flow uses numbered provider and model choices. It stores local model configuration under `~/.ontocellia/`. API keys are stored in `~/.ontocellia/secrets.env` with user-only file permissions when entered through `/setup`.

Non-interactive equivalents are available:

```bash
python -m ontocellia config setup
python -m ontocellia config models list
python -m ontocellia config models set deepseek
python -m ontocellia config models test deepseek
python -m ontocellia config get models.default
python -m ontocellia config file
```

## Run A Framework Tissue

```bash
python -m ontocellia tissue \
  --genome-spec examples/framework/repo_repair_genome.yaml \
  --environment-spec examples/framework/failing_tests_environment.yaml \
  --steps 4 \
  --output artifacts/repo_repair_tissue
```

Use `--stem-cells N` only when you want an experiment to start with a larger initial stem pool.

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

## Benchmark A Tissue

The built-in MiniBench suite measures Ontocellia-native agent tissue capabilities with mock LLM effectors.

```bash
python -m ontocellia benchmark \
  --suite ontocellia_minibench_v1 \
  --effector mock-llm \
  --output artifacts/benchmarks/minibench
```

Outputs:

- `benchmark_summary.json`
- `benchmark_results.csv`
- `benchmark_report.md`
- per-task tissue traces, summaries, and action intents

The TUI also supports `/benchmark`, which runs the mock MiniBench and prints a score summary.

## Run Adaptive Benchmark Data

Official benchmark runs use upstream task shapes and evaluate Ontocellia as a tissue. The default mode for non-BFCL benchmarks is `adaptive-tissue`.

```bash
python -m ontocellia official-benchmark run \
  --benchmark tau-bench \
  --model-profile deepseek \
  --limit 1 \
  --mode adaptive-tissue \
  --output artifacts/official_benchmarks/tau_bench/deepseek_smoke
```

Outputs include:

- `official_tasks.jsonl`
- `ontocellia_predictions.jsonl`
- `official_results.json`
- `structure_report.json`
- `adaptation_report.md`
- `ontocellia_summary.json`
- per-task tissue traces under `tissue_traces/`

Use `--task-id` for one specific official task, or `--full` only when you intend to run the full selected benchmark. API keys are read from the configured model profile and are not written into artifacts.

BFCL is kept as a provider/tool-call baseline:

```bash
python -m ontocellia official-benchmark run \
  --benchmark bfcl \
  --model-profile deepseek \
  --limit 50 \
  --mode provider-baseline \
  --output artifacts/official_benchmarks/bfcl/provider_baseline
```

## Execute Action Intents

By default, `tissue` emits intents and communication artifacts without performing local effects. Add `--execute-actions` to route intents through the extracellular execution policy. Dry-run is enabled by default.

```bash
python -m ontocellia tissue \
  --genome-spec examples/framework/repo_repair_genome.yaml \
  --environment-spec examples/framework/failing_tests_environment.yaml \
  --steps 4 \
  --effector mock-llm \
  --execute-actions \
  --execution-dry-run \
  --allow-interface workspace.search \
  --allow-interface git.diff \
  --output artifacts/execution_dry_run
```

This writes `execution_results.json` and deposits execution evidence into the matrix. To allow real local execution, keep the allowlist exact:

```bash
python -m ontocellia tissue \
  --genome-spec examples/framework/repo_repair_genome.yaml \
  --environment-spec examples/framework/failing_tests_environment.yaml \
  --effector mock-llm \
  --execute-actions \
  --no-execution-dry-run \
  --allow-interface pytest.run \
  --allow-command "python -m pytest -q" \
  --allow-write "src/**/*.py" \
  --output artifacts/execution_allowed
```

Ontocellia does not commit, push, install dependencies, or download benchmark data from the execution layer.

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
| `openai` | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| `openrouter` | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` |
| `ollama` | `OLLAMA_API_KEY` | `http://localhost:11434/v1` |
| `custom-openai-compatible` | `ONTOCELLIA_CUSTOM_API_KEY` | configured in setup |

DeepSeek setup offers the current official model IDs `deepseek-v4-flash` and `deepseek-v4-pro`.

After configuring a default model profile, use the simplified LLM effector:

```bash
python -m ontocellia tissue \
  --genome-spec examples/framework/repo_repair_genome.yaml \
  --environment-spec examples/framework/failing_tests_environment.yaml \
  --effector llm \
  --output artifacts/configured_llm_tissue
```

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

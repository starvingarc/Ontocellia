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
/config
/new <task>
/run [ticks]
/step
/agents
/intents
/matrix
/handoffs
/tools
/report
/benchmark
/mock
/clear
/exit
```

`/agents`, `/intents`, `/matrix`, `/handoffs`, `/tools`, `/report`, and `/config` refresh the corresponding panels and record the selected view in the event feed. Non-TTY input uses a lightweight boxed shell for script compatibility.

You can also force the Textual/Rich TUI with:

```bash
python -m ontocellia tui
```

The setup flow uses numbered provider and model choices. It stores local model configuration under `~/.ontocellia/`. API keys are stored in `~/.ontocellia/secrets.env` with user-only file permissions when entered through `/setup`.

## App Server

Start the local HTTP/WebSocket server:

```bash
python -m ontocellia server --host 127.0.0.1 --port 8765
```

Core endpoints:

```text
GET  /health
GET  /projects
GET  /projects/{project}/sessions
POST /v1/chat/completions
POST /sessions
GET  /sessions
GET  /sessions/{id}
POST /sessions/{id}/task
POST /sessions/{id}/change-medium
POST /sessions/{id}/run
POST /sessions/{id}/step
POST /sessions/{id}/interventions
GET  /sessions/{id}/agents
GET  /sessions/{id}/intents
GET  /sessions/{id}/matrix
GET  /sessions/{id}/handoffs
GET  /sessions/{id}/tools
GET  /sessions/{id}/tool-approvals
POST /sessions/{id}/tool-approvals
GET  /sessions/{id}/artifacts/{name}
WS   /sessions/{id}/events
```

The server writes artifacts under `artifacts/server_sessions/<session-id>/`. It uses the mock provider by default; pass `--real-provider` only when you want it to use configured model profiles. WebSocket clients receive an initial snapshot followed by live session and trace events. The browser Web Lab product direction is documented in [web-lab-design.md](web-lab-design.md).

Natural language entered into an existing session is treated as a culture-medium change. It deposits a matrix record, emits task morphogens, advances the tissue, and preserves the session's lineage rather than replacing the tissue. The first supported interventions are morphogen injection, cell clearing, cell freezing, pause, and resume.

Tool approvals remain safe by default. Pending `ToolInvocation` records can be approved through the API, but approval uses a dry-run policy unless a caller explicitly supplies a stricter project policy with write/command allowlists.

Non-interactive equivalents are available:

```bash
python -m ontocellia config setup
python -m ontocellia config models list
python -m ontocellia config models status
python -m ontocellia config models add
python -m ontocellia config models set deepseek
python -m ontocellia config models test deepseek
python -m ontocellia config validate
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

## Reference End-To-End Demo

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

## Search Tissue Structures

Structure search compares deterministic tissue variants induced from the same task. It is used to ask which organization fits the current environment better, not which provider is strongest.

```bash
python -m ontocellia structure-search \
  --task "Fix failing tests while preserving behavior." \
  --domain repo_repair \
  --effector mock-llm \
  --steps 6 \
  --seed 7 \
  --output artifacts/structure_search
```

Outputs:

- `structure_search_summary.json`
- `structure_trials.csv`
- `structure_search_report.md`
- `selected_variant.json`
- per-variant tissue summaries, traces, and action intents under `variants/`

## Run Official Benchmark Data

Official benchmark runs use upstream task shapes and report Ontocellia tissue metrics separately from external scorer status. The default mode for non-BFCL benchmarks is `adaptive-tissue`.

```bash
python -m ontocellia official-benchmark run \
  --benchmark tau-bench \
  --model-profile deepseek \
  --limit 1 \
  --mode adaptive-tissue \
  --tau-domain airline \
  --structure-search \
  --output artifacts/official_benchmarks/tau_bench/deepseek_smoke
```

Outputs include:

- `official_tasks.jsonl`
- `official_task_manifest.json`
- `scoring_status.json`
- `ontocellia_predictions.jsonl`
- `official_results.json`
- `structure_report.json`
- `adaptation_report.md`
- `ontocellia_summary.json`
- per-task tissue traces under `tissue_traces/`

Use `--task-id` for one specific official task, or `--full` only when you intend to run the full selected benchmark. Use `--run-official-scorer` when external scorer execution is intentional. See [official-benchmarks.md](official-benchmarks.md) for Terminal-Bench custom agent, tau-bench bridge, SWE-bench scorer, and custom scorer details.

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

By default, `tissue` emits intents and communication artifacts without performing local effects. Add `--execute-actions` to route intents through the extracellular tool policy. Dry-run is enabled by default.

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

The execution layer only performs work allowed by the active policy. It does not commit, push, install dependencies, or download benchmark data as part of action execution. It writes:

- `tool_invocations.json`
- `tool_results.json`
- `execution_results.json` for compatibility
- matrix records containing execution evidence

Long tool and validation output is handled by output metabolism. Full raw text goes under `raw_outputs/`; result JSON and matrix records keep bounded digests plus artifact references. See [communication.md](communication.md) for context and output metadata details.

Additional adapter gates are explicit:

```bash
python -m ontocellia tissue \
  --genome-spec examples/framework/repo_repair_genome.yaml \
  --environment-spec examples/framework/failing_tests_environment.yaml \
  --effector mock-llm \
  --execute-actions \
  --allow-interface mcp:repo:tool:read_file \
  --allow-mcp-tool mcp:repo:tool:read_file \
  --allow-interface http.request \
  --allow-network-host api.example.com \
  --enable-http-tools \
  --output artifacts/tool_runtime
```

MCP, HTTP, and browser adapters are disabled until their specific policy flags are present. Browser support is currently an adapter boundary for future richer automation.

## Inspect Context

Use `communication.context_budget_chars` and `communication.context_metabolism` in an environment spec to tune approximate context size and matrix remodeling:

```yaml
communication:
  matrix_query_limit: 5
  context_budget_chars: 1600
  context_metabolism:
    enabled: true
    window_ticks: 3
    max_metabolites_per_tick: 4
    max_metabolite_chars: 700
    min_source_records: 2
    source_salience_decay: 0.15
```

Cells receive bounded matrix context instead of the full tissue history. Inspect `tissue_trace.json` for `llm_effector`, `context_metabolite_deposited`, and `context_metabolism` events. See [communication.md](communication.md) for record lifecycle fields, context packets, and output digest metadata.

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

DeepSeek setup includes default profile choices such as `deepseek-v4-flash` and `deepseek-v4-pro`.

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

## Reference Simulation Runtime

```bash
python -m ontocellia run --steps 20 --output artifacts/demo
python -m ontocellia --steps 20 --output artifacts/legacy_demo
```

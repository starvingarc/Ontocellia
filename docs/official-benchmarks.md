# Official Benchmark Integration

Ontocellia treats official benchmarks as external environments that can drive or score the same adaptive tissue runtime. The benchmark layer keeps three statuses separate:

- official task source loaded;
- Ontocellia adaptive tissue metrics produced;
- external official scorer executed.

`official_score_status: not_run` means official task data was used, but no external scorer command was executed.

## Bounded Adaptive Run

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

Use `--task-id` for a single task, `--limit` for bounded runs, and `--full` only for an intentional full run.

## Scorer Status

With `--run-official-scorer` and no explicit command, Ontocellia uses benchmark-aware scorer adapters:

- SWE-bench Lite writes `official_scorer_predictions.jsonl` and `official_scorer_plan.json`, then runs the official harness when the `swebench` package is installed.
- Terminal-Bench writes a command plan using `--agent-import-path ontocellia.official_terminal_agent:OntocelliaTerminalAgent`; when the official package is installed the plan is `ready`.
- tau-bench writes `bridge_required` until a local Ontocellia OpenAI-compatible bridge URL is supplied.

To run a local scorer command directly, pass `--official-scorer-command`. The command is split with `shlex`, does not use a shell, and writes `official_stdout.log`, `official_stderr.log`, and `scoring_status.json`.

## Terminal-Bench Custom Agent

Terminal-Bench can drive Ontocellia through the official custom agent import path:

```bash
tb run \
  --dataset-path artifacts/official_sources/terminal-bench/original-tasks \
  --agent-import-path ontocellia.official_terminal_agent:OntocelliaTerminalAgent
```

The adapter turns a terminal task into the standard Ontocellia flow: induction, single-stem development, cell intent emission, bounded terminal commands, and artifacts under the harness logging directory.

## tau-bench Tool-Calling Bridge

Start the local app server:

```bash
python -m ontocellia server --host 127.0.0.1 --port 8765
```

Then pass the OpenAI-compatible base URL to an official benchmark run:

```bash
python -m ontocellia official-benchmark run \
  --benchmark tau-bench \
  --model-profile deepseek \
  --limit 1 \
  --mode adaptive-tissue \
  --tau-domain airline \
  --run-official-scorer \
  --bridge-url http://127.0.0.1:8765/v1 \
  --output artifacts/official_benchmarks/tau_bridge
```

The bridge accepts chat messages and tool schemas at `POST /v1/chat/completions`, runs the request through an Ontocellia tissue, and returns a standard assistant message or tool call.

## Repo-Like Tasks

SWE-bench Lite uses repo-repair induction by default. Terminal-Bench coding, debugging, software-engineering, compatibility, pytest, failing, regression, fix, or bug tasks receive repair pressure and a repair niche. Other Terminal-Bench task categories can remain generic.

## BFCL Provider Baseline

BFCL is kept as a provider/tool-call baseline:

```bash
python -m ontocellia official-benchmark run \
  --benchmark bfcl \
  --model-profile deepseek \
  --limit 50 \
  --mode provider-baseline \
  --output artifacts/official_benchmarks/bfcl/provider_baseline
```

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ontocellia.__main__ import main


@pytest.mark.skipif(os.environ.get("ONTOCELLIA_LIVE_LLM") != "1", reason="set ONTOCELLIA_LIVE_LLM=1 to run live provider E2E")
def test_live_minimax_tissue_e2e_writes_structured_intents(tmp_path: Path) -> None:
    api_key = os.environ.get("MINIMAX_API_KEY", "")
    if not api_key:
        pytest.skip("MINIMAX_API_KEY is required for live MiniMax E2E")

    output = tmp_path / "live_minimax_tissue"
    main(
        [
            "tissue",
            "--genome-spec",
            "examples/framework/repo_repair_genome.yaml",
            "--environment-spec",
            "examples/framework/failing_tests_environment.yaml",
            "--steps",
            "4",
            "--effector",
            "minimax",
            "--llm-base-url",
            os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.chat/v1"),
            "--output",
            str(output),
        ]
    )

    intents = json.loads((output / "action_intents.json").read_text(encoding="utf-8"))
    trace_raw = (output / "llm_trace.json").read_text(encoding="utf-8")
    trace = json.loads(trace_raw)

    assert intents
    assert trace
    assert all(intent["cell_id"] is not None for intent in intents)
    assert all(intent["intent_type"] for intent in intents)
    assert all(event["provider"] == "minimax" for event in trace)
    assert all(event["model"] for event in trace)
    assert api_key not in trace_raw

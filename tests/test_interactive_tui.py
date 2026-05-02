from __future__ import annotations

import asyncio

from ontocellia.framework.llm import MockLLMProvider
from ontocellia.framework.model_config import ModelProfile, OntocelliaUserConfig, save_user_config
from ontocellia.tui import OntocelliaTUI


def test_tui_starts_with_soft_lab_panels(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ONTOCELLIA_CONFIG_DIR", str(tmp_path / "config"))
    app = OntocelliaTUI(output_root=tmp_path / "sessions", provider=MockLLMProvider())

    async def run() -> None:
        async with app.run_test() as pilot:
            assert "Ontocellia" in str(app.query_one("#status").render())
            assert app.query_one("#agents").border_title == "Cell Culture"
            assert "Matrix" in app.query_one("#bottom").border_title
            await pilot.press("ctrl+c")

    asyncio.run(run())


def test_tui_help_and_task_run_update_panels(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ONTOCELLIA_CONFIG_DIR", str(tmp_path / "config"))
    save_user_config(
        OntocelliaUserConfig(
            models={
                "default": "mock",
                "profiles": {"mock": ModelProfile(provider="mock-llm", model="mock-llm")},
            }
        )
    )
    app = OntocelliaTUI(output_root=tmp_path / "sessions", provider=MockLLMProvider())

    async def run() -> None:
        async with app.run_test() as pilot:
            await pilot.click("#command-input")
            await pilot.press(*list("/help"), "enter")
            assert "/new <task>" in str(app.query_one("#events").render())

            await pilot.press(*list("Fix failing tests while preserving behavior."), "enter")
            assert "Fix failing tests" in str(app.query_one("#summary").render())
            assert "propose_patch" in str(app.query_one("#intent-view").render())
            assert "collaboration" in str(app.query_one("#events").render())

            await pilot.press(*list("/run 2"), "enter")
            assert "propose_patch" in str(app.query_one("#intent-view").render())
            assert app.query_one("#agents").row_count > 0

    asyncio.run(run())


def test_tui_uses_fresh_soft_lab_copy(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ONTOCELLIA_CONFIG_DIR", str(tmp_path / "config"))
    app = OntocelliaTUI(output_root=tmp_path / "sessions", provider=MockLLMProvider())

    async def run() -> None:
        async with app.run_test() as pilot:
            status = str(app.query_one("#status").render())
            events = str(app.query_one("#events").render())
            assert "Soft Lab" in status
            assert "task to culture" in events
            assert app.query_one("#agents").border_title == "Cell Culture"
            await pilot.press("ctrl+c")

    asyncio.run(run())


def test_tui_benchmark_command_displays_score_summary(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ONTOCELLIA_CONFIG_DIR", str(tmp_path / "config"))
    app = OntocelliaTUI(output_root=tmp_path / "sessions", provider=MockLLMProvider())

    async def run() -> None:
        async with app.run_test() as pilot:
            await pilot.click("#command-input")
            await pilot.press(*list("/benchmark"), "enter")
            events = str(app.query_one("#events").render())
            assert "benchmark" in events
            assert "ontocellia_minibench_v1" in events
            assert "average" in events
            await pilot.press("ctrl+c")

    asyncio.run(run())

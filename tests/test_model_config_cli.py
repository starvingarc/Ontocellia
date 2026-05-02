from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ontocellia.__main__ import main
from ontocellia.framework import MockLLMProvider, resolve_effector_provider
from ontocellia.framework.model_config import ModelProfile, OntocelliaUserConfig, load_user_config, save_user_config


def test_main_without_arguments_enters_interactive_cli_and_exits(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("/exit\n"))

    main([])

    output = capsys.readouterr().out
    assert "╭" in output
    assert "Ontocellia" in output
    assert "developmental agent tissue CLI" in output
    assert "ontocellia ▸" in output
    assert "culture:" in output
    assert "/config" in output


def test_interactive_help_uses_boxed_commands(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("/help\n/exit\n"))

    main([])

    output = capsys.readouterr().out
    assert "╭─ Commands" in output
    assert "/setup" in output
    assert "/config models" in output
    assert "/run tissue" in output


def test_user_config_round_trips_model_profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONTOCELLIA_CONFIG_DIR", str(tmp_path))
    config = OntocelliaUserConfig(
        models={
            "default": "deepseek",
            "profiles": {
                "deepseek": ModelProfile(
                    provider="deepseek",
                    model="deepseek-v4-flash",
                    base_url="https://api.deepseek.com",
                    api_key_env="DEEPSEEK_API_KEY",
                )
            },
        }
    )

    save_user_config(config)
    loaded = load_user_config()

    assert loaded.default_model == "deepseek"
    assert loaded.profile("deepseek").model == "deepseek-v4-flash"
    assert (tmp_path / "config.yaml").exists()


def test_models_set_default_updates_user_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONTOCELLIA_CONFIG_DIR", str(tmp_path))
    save_user_config(
        OntocelliaUserConfig(
            models={
                "default": "mock",
                "profiles": {
                    "mock": ModelProfile(provider="mock-llm", model="mock-llm"),
                    "deepseek": ModelProfile(
                        provider="deepseek",
                        model="deepseek-v4-flash",
                        api_key_env="DEEPSEEK_API_KEY",
                    ),
                },
            }
        )
    )

    main(["config", "models", "set", "deepseek"])

    assert load_user_config().default_model == "deepseek"


def test_config_get_set_unset_and_file_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("ONTOCELLIA_CONFIG_DIR", str(tmp_path))

    main(["config", "set", "runtime.trace_prompts", "false"])
    main(["config", "get", "runtime.trace_prompts"])
    main(["config", "file"])
    main(["config", "unset", "runtime.trace_prompts"])

    output = capsys.readouterr().out
    assert "false" in output
    assert str(tmp_path / "config.yaml") in output
    assert "trace_prompts" not in load_user_config().runtime


def test_resolve_effector_provider_uses_configured_mock_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONTOCELLIA_CONFIG_DIR", str(tmp_path))
    save_user_config(
        OntocelliaUserConfig(
            models={
                "default": "mock",
                "profiles": {"mock": ModelProfile(provider="mock-llm", model="mock-llm")},
            }
        )
    )

    provider = resolve_effector_provider("llm")

    assert isinstance(provider, MockLLMProvider)


def test_tissue_cli_llm_effector_uses_default_model_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONTOCELLIA_CONFIG_DIR", str(tmp_path / "config"))
    save_user_config(
        OntocelliaUserConfig(
            models={
                "default": "mock",
                "profiles": {"mock": ModelProfile(provider="mock-llm", model="mock-llm")},
            }
        )
    )
    output = tmp_path / "llm_tissue"

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
            "llm",
            "--output",
            str(output),
        ]
    )

    intents = json.loads((output / "action_intents.json").read_text(encoding="utf-8"))
    trace = json.loads((output / "llm_trace.json").read_text(encoding="utf-8"))
    assert intents
    assert trace[0]["provider"] == "mock-llm"


def test_models_list_marks_default_and_non_default_profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("ONTOCELLIA_CONFIG_DIR", str(tmp_path))
    save_user_config(
        OntocelliaUserConfig(
            models={
                "default": "deepseek",
                "profiles": {
                    "deepseek": ModelProfile(
                        provider="deepseek",
                        model="deepseek-v4-flash",
                        api_key_env="DEEPSEEK_API_KEY",
                    ),
                    "mock": ModelProfile(provider="mock-llm", model="mock-llm"),
                },
            }
        )
    )

    main(["config", "models", "list"])

    output = capsys.readouterr().out
    assert "╭─ Model Profiles" in output
    assert "● deepseek" in output
    assert "○ mock" in output


def test_models_list_without_profiles_shows_configure_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("ONTOCELLIA_CONFIG_DIR", str(tmp_path))

    main(["config", "models", "list"])

    output = capsys.readouterr().out
    assert "No model profiles found" in output
    assert "Run /setup" in output


def test_interactive_run_tissue_prints_tissue_summary_box(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("ONTOCELLIA_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.chdir(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    (tmp_path / "examples" / "framework").mkdir(parents=True)
    (tmp_path / "examples" / "framework" / "repo_repair_genome.yaml").write_text(
        (repo_root / "examples" / "framework" / "repo_repair_genome.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "examples" / "framework" / "failing_tests_environment.yaml").write_text(
        (repo_root / "examples" / "framework" / "failing_tests_environment.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    save_user_config(
        OntocelliaUserConfig(
            models={
                "default": "mock",
                "profiles": {"mock": ModelProfile(provider="mock-llm", model="mock-llm")},
            }
        )
    )
    monkeypatch.setattr("sys.stdin", io.StringIO("/run tissue\n/exit\n"))

    main([])

    output = capsys.readouterr().out
    assert "╭─ Tissue Summary" in output
    assert "cells" in output
    assert "trace" in output


def test_top_level_configure_and_models_commands_are_removed() -> None:
    for argv in (["configure", "--help"], ["models", "list"]):
        with pytest.raises(SystemExit):
            main(argv)


def test_config_models_status_shows_provider_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("ONTOCELLIA_CONFIG_DIR", str(tmp_path))

    main(["config", "models", "status"])

    output = capsys.readouterr().out
    assert "Available Providers" in output
    assert "deepseek" in output
    assert "openai" in output
    assert "openrouter" in output
    assert "ollama" in output


def test_config_setup_uses_numbered_provider_and_model_choices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ONTOCELLIA_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr("sys.stdin", io.StringIO("2\n1\n\n"))
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "")

    main(["config", "setup"])

    output = capsys.readouterr().out
    config = load_user_config()
    assert "Choose Provider" in output
    assert "Choose Model" in output
    assert config.default_model == "deepseek"
    assert config.profile("deepseek").provider == "deepseek"
    assert config.profile("deepseek").model == "deepseek-v4-flash"


def test_config_setup_custom_provider_allows_custom_base_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ONTOCELLIA_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr("sys.stdin", io.StringIO("8\nmy-model\nhttps://llm.example/v1\nCUSTOM_KEY\n\n"))
    monkeypatch.setattr("getpass.getpass", lambda _prompt: "")

    main(["config", "setup"])

    output = capsys.readouterr().out
    config = load_user_config()
    assert "Custom OpenAI-compatible" in output
    assert config.default_model == "custom-openai-compatible"
    assert config.profile().model == "my-model"
    assert config.profile().base_url == "https://llm.example/v1"

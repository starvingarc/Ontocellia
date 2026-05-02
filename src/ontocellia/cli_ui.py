from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PROMPT = "ontocellia ▸ "


def prompt() -> str:
    return PROMPT


def render_banner(config: Any | None = None) -> str:
    model = getattr(config, "default_model", "") if config is not None else ""
    return "\n".join(
        [
            _box(
                "Ontocellia",
                [
                    "developmental agent tissue CLI",
                    "",
                    "culture: repo-repair        tissue: inactive",
                    f"model:   {model or 'unconfigured':<16} genome: repo_repair_genome.yaml",
                    "matrix:  0 records          trace:  artifacts/interactive_tissue",
                ],
            ),
            "",
            "Type /help for commands, /setup for providers, or /config for status.",
        ]
    )


def render_help() -> str:
    return _box(
        "Commands",
        [
            "/setup            run first-time provider setup",
            "/config models    show model profiles",
            "/config models test NAME",
            "/use NAME         set default model profile",
            "/config           show config status",
            "/run tissue       run default tissue demo",
            "/exit             leave the culture",
        ],
    )


def render_config_status(config_path: Path, secrets_path: Path) -> str:
    return _box(
        "Config",
        [
            f"config:  {config_path}",
            f"secrets: {secrets_path}",
            "status:  valid",
        ],
    )


def render_models(config: Any, *, include_key_status: bool = False, secrets: dict[str, str] | None = None) -> str:
    profiles = getattr(config, "profiles", {})
    default = getattr(config, "default_model", "")
    if not profiles:
        return render_error(
            "No model profiles found.",
            "Run /setup to prepare an effector provider, or `ontocellia config setup` outside the shell.",
        )
    secrets = secrets or {}
    lines = [f"default: {default or '-'}", ""]
    for name, profile in sorted(profiles.items()):
        marker = "●" if name == default else "○"
        lines.append(f"{marker} {name}")
        lines.append(f"  provider: {profile.provider}")
        lines.append(f"  model:    {profile.model or '-'}")
        if include_key_status and getattr(profile, "api_key_env", ""):
            lines.append(f"  key:      {_key_status(profile.api_key_env, secrets)}")
        lines.append("")
    if lines[-1] == "":
        lines.pop()
    return _box("Model Profiles", lines)


def render_provider_catalog(defaults: dict[str, dict[str, str]], *, title: str = "Available Providers") -> str:
    lines = []
    for index, (name, values) in enumerate(_ordered_providers(defaults), start=1):
        lines.append(f"{index}. {_display_provider(name):<24} {values.get('model', '-')}")
    return _box(title, lines)


def render_choice_list(title: str, choices: list[str]) -> str:
    return _box(title, [f"{index}. {choice}" for index, choice in enumerate(choices, start=1)])


def render_tissue_summary(summary_path: Path, summary: dict[str, Any]) -> str:
    return _box(
        "Tissue Summary",
        [
            f"cells      {summary.get('population', 0)}",
            f"actions    {len(summary.get('actions', []))}",
            f"messages   {summary.get('messages', 0)}",
            f"matrix     {summary.get('matrix_records', 0)} records",
            f"trace      {summary_path.parent}",
        ],
    )


def render_error(message: str, hint: str | None = None) -> str:
    lines = [message]
    if hint:
        lines.extend(["", hint])
    return _box("Notice", lines)


def _key_status(api_key_env: str, secrets: dict[str, str]) -> str:
    if api_key_env in secrets or api_key_env in os.environ:
        return "set"
    return f"missing ({api_key_env})"


def _box(title: str, lines: list[str], *, width: int = 78) -> str:
    top = f"╭─ {title} " + "─" * max(0, width - len(title) - 5) + "╮"
    bottom = "╰" + "─" * (width - 2) + "╯"
    body = [f"│ {_fit(line, width - 3):<{width - 3}}│" for line in lines]
    return "\n".join([top, *body, bottom])


def _fit(line: str, width: int) -> str:
    if len(line) <= width:
        return line
    if width <= 1:
        return line[:width]
    return line[: width - 1] + "…"


def _ordered_providers(defaults: dict[str, dict[str, str]]) -> list[tuple[str, dict[str, str]]]:
    preferred = [
        "mock-llm",
        "deepseek",
        "minimax",
        "kimi",
        "openai",
        "openrouter",
        "ollama",
        "custom-openai-compatible",
    ]
    ordered = [(name, defaults[name]) for name in preferred if name in defaults]
    ordered.extend((name, defaults[name]) for name in sorted(defaults) if name not in preferred)
    return ordered


def _display_provider(name: str) -> str:
    labels = {
        "custom-openai-compatible": "Custom OpenAI-compatible",
        "mock-llm": "Mock LLM",
    }
    return labels.get(name, name)

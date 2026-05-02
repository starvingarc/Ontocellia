from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ontocellia.framework.llm import MockLLMProvider, OpenAICompatibleProvider


PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "deepseek": {
        "model": "deepseek-v4-flash",
        "models": ["deepseek-v4-flash", "deepseek-v4-pro"],
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "kimi": {
        "model": "kimi-k2.6",
        "models": ["kimi-k2.6", "kimi-latest"],
        "base_url": "https://api.moonshot.ai/v1",
        "api_key_env": "MOONSHOT_API_KEY",
    },
    "minimax": {
        "model": "MiniMax-M2.7",
        "models": ["MiniMax-M2.7"],
        "base_url": "https://api.minimax.io/v1",
        "api_key_env": "MINIMAX_API_KEY",
    },
    "openai": {
        "model": "gpt-4.1-mini",
        "models": ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini"],
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "openrouter": {
        "model": "openai/gpt-4.1-mini",
        "models": ["openai/gpt-4.1-mini", "anthropic/claude-3.5-sonnet", "google/gemini-flash-1.5"],
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "ollama": {
        "model": "llama3.1",
        "models": ["llama3.1", "qwen2.5-coder", "mistral"],
        "base_url": "http://localhost:11434/v1",
        "api_key_env": "OLLAMA_API_KEY",
    },
    "custom-openai-compatible": {
        "model": "custom-model",
        "models": [],
        "base_url": "https://example.com/v1",
        "api_key_env": "ONTOCELLIA_CUSTOM_API_KEY",
    },
    "mock-llm": {
        "model": "mock-llm",
        "models": ["mock-llm"],
        "base_url": "",
        "api_key_env": "",
    },
}


@dataclass(slots=True)
class ModelProfile:
    provider: str
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelProfile":
        return cls(
            provider=str(data.get("provider", "")),
            model=str(data.get("model", "")),
            base_url=str(data.get("base_url", "")),
            api_key_env=str(data.get("api_key_env", "")),
        )

    def as_dict(self) -> dict[str, str]:
        data = {"provider": self.provider}
        if self.model:
            data["model"] = self.model
        if self.base_url:
            data["base_url"] = self.base_url
        if self.api_key_env:
            data["api_key_env"] = self.api_key_env
        return data


@dataclass(slots=True)
class OntocelliaUserConfig:
    models: dict[str, Any] = field(default_factory=lambda: {"default": "", "profiles": {}})
    runtime: dict[str, Any] = field(
        default_factory=lambda: {"trace_prompts": True, "trace_raw_response": False, "redact_secrets": True}
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OntocelliaUserConfig":
        models = dict(data.get("models", {}) or {})
        raw_profiles = dict(models.get("profiles", {}) or {})
        profiles = {
            str(name): (profile if isinstance(profile, ModelProfile) else ModelProfile.from_dict(dict(profile or {})))
            for name, profile in raw_profiles.items()
        }
        models["profiles"] = profiles
        return cls(models=models, runtime=dict(data.get("runtime", {}) or {}))

    @property
    def default_model(self) -> str:
        return str(self.models.get("default", ""))

    @default_model.setter
    def default_model(self, value: str) -> None:
        self.models["default"] = value

    @property
    def profiles(self) -> dict[str, ModelProfile]:
        profiles = self.models.setdefault("profiles", {})
        return profiles

    def profile(self, name: str | None = None) -> ModelProfile:
        profile_name = name or self.default_model
        if not profile_name:
            raise ValueError("No model profile configured. Run `ontocellia config setup` first.")
        if profile_name not in self.profiles:
            raise ValueError(f"Unknown model profile: {profile_name}")
        return self.profiles[profile_name]

    def as_dict(self) -> dict[str, Any]:
        return {
            "models": {
                "default": self.default_model,
                "profiles": {name: profile.as_dict() for name, profile in sorted(self.profiles.items())},
            },
            "runtime": dict(self.runtime),
        }


def config_dir() -> Path:
    configured = os.environ.get("ONTOCELLIA_CONFIG_DIR", "")
    return Path(configured).expanduser() if configured else Path.home() / ".ontocellia"


def config_path() -> Path:
    return config_dir() / "config.yaml"


def secrets_path() -> Path:
    return config_dir() / "secrets.env"


def load_user_config(path: Path | None = None) -> OntocelliaUserConfig:
    target = path or config_path()
    if not target.exists():
        return OntocelliaUserConfig()
    data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid Ontocellia config file: {target}")
    return OntocelliaUserConfig.from_dict(data)


def save_user_config(config: OntocelliaUserConfig, path: Path | None = None) -> Path:
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config.as_dict(), sort_keys=True), encoding="utf-8")
    return target


def load_secret_env(path: Path | None = None) -> dict[str, str]:
    target = path or secrets_path()
    if not target.exists():
        return {}
    result: dict[str, str] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        result[key.strip()] = value.strip().strip("'\"")
    return result


def save_secret(name: str, value: str, path: Path | None = None) -> Path:
    target = path or secrets_path()
    secrets = load_secret_env(target)
    secrets[name] = value
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(f"{key}={value!r}\n" for key, value in sorted(secrets.items())), encoding="utf-8")
    target.chmod(0o600)
    return target


def set_config_value(config: OntocelliaUserConfig, dotted_path: str, value: str) -> None:
    parent, key = _resolve_parent(config, dotted_path, create=True)
    parent[key] = _parse_scalar(value)


def get_config_value(config: OntocelliaUserConfig, dotted_path: str) -> Any:
    current: Any = config.as_dict()
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_path)
        current = current[part]
    return current


def unset_config_value(config: OntocelliaUserConfig, dotted_path: str) -> None:
    parent, key = _resolve_parent(config, dotted_path, create=False)
    parent.pop(key, None)


def resolve_effector_provider(
    effector: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    model_profile: str | None = None,
) -> object | None:
    if effector == "rule":
        return None
    if effector == "mock-llm":
        return MockLLMProvider()
    if effector == "llm":
        config = load_user_config()
        profile = config.profile(model_profile)
        return _provider_from_profile(profile)
    return OpenAICompatibleProvider.from_name(effector, model=model, base_url=base_url)


def _provider_from_profile(profile: ModelProfile) -> object:
    if profile.provider == "mock-llm":
        return MockLLMProvider()
    env = dict(os.environ)
    env.update(load_secret_env())
    if profile.api_key_env and profile.api_key_env in env:
        default_key_env = PROVIDER_DEFAULTS.get(profile.provider, {}).get("api_key_env", profile.api_key_env)
        env[default_key_env] = env[profile.api_key_env]
    return OpenAICompatibleProvider.from_name(
        profile.provider,
        model=profile.model or None,
        base_url=profile.base_url or None,
        env=env,
    )


def _resolve_parent(config: OntocelliaUserConfig, dotted_path: str, *, create: bool) -> tuple[dict[str, Any], str]:
    if not dotted_path or "." not in dotted_path:
        raise ValueError("config path must be dotted, for example runtime.trace_prompts")
    data = config.as_dict()
    current: dict[str, Any] = data
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        if part not in current:
            if not create:
                raise KeyError(dotted_path)
            current[part] = {}
        if not isinstance(current[part], dict):
            raise ValueError(f"config path is not a mapping: {part}")
        current = current[part]
    updated = OntocelliaUserConfig.from_dict(data)
    config.models = updated.models
    config.runtime = updated.runtime
    current = config.models if parts[0] == "models" else config.runtime
    for part in parts[1:-1]:
        current = current.setdefault(part, {})
    return current, parts[-1]


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .experiment import ExperimentSpec
from .schema import EnvironmentSpec, GenomeSpec


def _load_mapping(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported spec format: {file_path.suffix}")
    if not isinstance(data, dict):
        raise ValueError(f"Spec file must contain a top-level mapping: {file_path}")
    return data


def load_genome_spec(path: str | Path) -> GenomeSpec:
    file_path = Path(path)
    return GenomeSpec.from_dict(_load_mapping(file_path), source_path=file_path)


def load_environment_spec(path: str | Path) -> EnvironmentSpec:
    file_path = Path(path)
    return EnvironmentSpec.from_dict(_load_mapping(file_path), source_path=file_path)


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    file_path = Path(path)
    return ExperimentSpec.from_dict(_load_mapping(file_path), source_path=file_path)

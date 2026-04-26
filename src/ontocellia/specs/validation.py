from __future__ import annotations

from pathlib import Path

from .experiment import ExperimentSpec
from .loader import _load_mapping
from .schema import EnvironmentSpec, GenomeSpec


def validate_genome_spec(path: str | Path) -> list[str]:
    try:
        GenomeSpec.from_dict(_load_mapping(path), source_path=Path(path))
    except Exception as exc:
        return [str(exc)]
    return []


def validate_environment_spec(path: str | Path) -> list[str]:
    try:
        EnvironmentSpec.from_dict(_load_mapping(path), source_path=Path(path))
    except Exception as exc:
        return [str(exc)]
    return []


def validate_experiment_spec(path: str | Path) -> list[str]:
    try:
        ExperimentSpec.from_dict(_load_mapping(path), source_path=Path(path))
    except Exception as exc:
        return [str(exc)]
    return []


def validate_model_specs(genome_path: str | Path, environment_path: str | Path) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_genome_spec(genome_path))
    errors.extend(validate_environment_spec(environment_path))
    return errors

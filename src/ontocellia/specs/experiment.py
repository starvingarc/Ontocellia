from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import MetadataSpec


def _mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _optional_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


@dataclass(slots=True)
class ExperimentBaseSpec:
    genome: Path
    environment: Path
    steps: int = 80
    seed: int = 7

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentBaseSpec":
        if "genome" not in data:
            raise ValueError("base.genome is required")
        if "environment" not in data:
            raise ValueError("base.environment is required")
        steps = int(data.get("steps", 80))
        if steps < 0:
            raise ValueError("base.steps must be non-negative")
        return cls(
            genome=Path(str(data["genome"])),
            environment=Path(str(data["environment"])),
            steps=steps,
            seed=int(data.get("seed", 7)),
        )


@dataclass(slots=True)
class ExperimentVariantSpec:
    name: str
    config_patch: dict[str, Any] = field(default_factory=dict)
    genome_patch: dict[str, Any] = field(default_factory=dict)
    environment_patch: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int) -> "ExperimentVariantSpec":
        if not isinstance(data, dict):
            raise ValueError(f"variants[{index}] must be a mapping")
        name = data.get("name")
        if not name:
            raise ValueError(f"variants[{index}].name is required")
        for key in ("config_patch", "genome_patch", "environment_patch"):
            if key in data and not isinstance(data[key], dict):
                raise ValueError(f"variants[{index}].{key} must be a mapping")
        return cls(
            name=str(name),
            config_patch=dict(data.get("config_patch", {})),
            genome_patch=dict(data.get("genome_patch", {})),
            environment_patch=dict(data.get("environment_patch", {})),
        )


@dataclass(slots=True)
class ExperimentOutputsSpec:
    summary: bool = True
    plots: bool = True
    metrics_csv: bool = True
    report: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentOutputsSpec":
        return cls(
            summary=bool(data.get("summary", True)),
            plots=bool(data.get("plots", True)),
            metrics_csv=bool(data.get("metrics_csv", True)),
            report=bool(data.get("report", True)),
        )


@dataclass(slots=True)
class ExperimentSpec:
    metadata: MetadataSpec
    base: ExperimentBaseSpec
    variants: list[ExperimentVariantSpec]
    outputs: ExperimentOutputsSpec = field(default_factory=ExperimentOutputsSpec)
    source_path: Path | None = None

    @property
    def source_dir(self) -> Path:
        return self.source_path.parent if self.source_path is not None else Path.cwd()

    @property
    def resolved_genome_path(self) -> Path:
        return self._resolve(self.base.genome)

    @property
    def resolved_environment_path(self) -> Path:
        return self._resolve(self.base.environment)

    def _resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else self.source_dir / path

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source_path: Path | None = None) -> "ExperimentSpec":
        metadata = MetadataSpec.from_dict(_mapping(data, "metadata"))
        base = ExperimentBaseSpec.from_dict(_mapping(data, "base"))
        raw_variants = data.get("variants", [{"name": "baseline"}])
        if not isinstance(raw_variants, list):
            raise ValueError("variants must be a list")
        variants = [ExperimentVariantSpec.from_dict(variant, index) for index, variant in enumerate(raw_variants)]
        if not variants:
            raise ValueError("variants must include at least one variant")
        outputs = ExperimentOutputsSpec.from_dict(_optional_mapping(data, "outputs"))
        return cls(metadata=metadata, base=base, variants=variants, outputs=outputs, source_path=source_path)


def validate_experiment_data(data: dict[str, Any], *, source_path: Path | None = None) -> list[str]:
    try:
        ExperimentSpec.from_dict(data, source_path=source_path)
    except Exception as exc:
        return [str(exc)]
    return []

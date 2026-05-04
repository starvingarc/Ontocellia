from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ERROR_KEYWORDS = ("error", "failed", "failure", "traceback", "exception", "stderr")


@dataclass(slots=True)
class OutputMetabolismPolicy:
    max_inline_chars: int = 12000
    head_ratio: float = 0.45
    max_error_lines: int = 8


@dataclass(slots=True)
class OutputDigest:
    kind: str
    inline: str
    raw_chars: int
    inline_chars: int
    truncated: bool
    raw_output_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OutputMetabolismRuntime:
    def digest_output(
        self,
        kind: str,
        text: str,
        policy: OutputMetabolismPolicy | None = None,
        artifact_root: Path | str | None = None,
        source_id: str = "output",
    ) -> OutputDigest:
        active_policy = policy or OutputMetabolismPolicy()
        max_chars = max(0, int(active_policy.max_inline_chars))
        value = str(text or "")
        raw_path = None
        if len(value) <= max_chars:
            return OutputDigest(kind=kind, inline=value, raw_chars=len(value), inline_chars=len(value), truncated=False)
        if artifact_root is not None:
            raw_path = _write_raw_output(Path(artifact_root), source_id, value)
        inline = _digest_inline(kind, value, max_chars, active_policy, raw_path)
        return OutputDigest(
            kind=kind,
            inline=inline,
            raw_chars=len(value),
            inline_chars=len(inline),
            truncated=True,
            raw_output_path=raw_path,
        )


def digest_output(
    kind: str,
    text: str,
    policy: OutputMetabolismPolicy | None = None,
    artifact_root: Path | str | None = None,
    source_id: str = "output",
) -> OutputDigest:
    return OutputMetabolismRuntime().digest_output(kind, text, policy, artifact_root, source_id)


def _write_raw_output(artifact_root: Path, source_id: str, value: str) -> str:
    directory = artifact_root / "raw_outputs"
    directory.mkdir(parents=True, exist_ok=True)
    relative = Path("raw_outputs") / f"{_safe_source_id(source_id)}.txt"
    (artifact_root / relative).write_text(value, encoding="utf-8")
    return relative.as_posix()


def _safe_source_id(source_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(source_id)).strip(".-")
    return safe or "output"


def _digest_inline(
    kind: str,
    value: str,
    max_chars: int,
    policy: OutputMetabolismPolicy,
    raw_path: str | None,
) -> str:
    header = [
        "[output digest]",
        f"kind: {kind}",
        f"raw_chars: {len(value)}",
        "truncated: true",
    ]
    if raw_path:
        header.append(f"raw_output_path: {raw_path}")
    error_lines = _error_lines(value, policy.max_error_lines)
    error_section = "\n".join(["", "[error lines]", *error_lines]) if error_lines else ""
    section_overhead = "\n\n[head]\n\n\n[tail]\n"
    remaining = max_chars - len("\n".join(header)) - len(error_section) - len(section_overhead)
    if remaining <= 0:
        return _fit("\n".join(header) + error_section, max_chars)
    head_chars = int(remaining * max(0.0, min(1.0, policy.head_ratio)))
    tail_chars = max(0, remaining - head_chars)
    inline = (
        "\n".join(header)
        + "\n\n[head]\n"
        + value[:head_chars].rstrip()
        + error_section
        + "\n\n[tail]\n"
        + value[-tail_chars:].lstrip()
    )
    return _fit(inline, max_chars)


def _error_lines(value: str, limit: int) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for line in value.splitlines():
        normalized = line.strip()
        if not normalized:
            continue
        if not any(keyword in normalized.lower() for keyword in ERROR_KEYWORDS):
            continue
        clipped = normalized[:180]
        if clipped in seen:
            continue
        seen.add(clipped)
        lines.append(clipped)
        if len(lines) >= limit:
            break
    return lines


def _fit(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    marker = "\n[digest truncated]"
    if max_chars <= len(marker):
        return value[:max_chars]
    return value[: max_chars - len(marker)] + marker

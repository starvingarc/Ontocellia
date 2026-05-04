from __future__ import annotations

from pathlib import Path

from ontocellia.framework import OutputMetabolismPolicy, OutputMetabolismRuntime, digest_output


def test_long_output_generates_bounded_digest_and_raw_artifact(tmp_path: Path) -> None:
    text = "\n".join([f"head-{index}" for index in range(30)] + ["Traceback: important failure"] + [f"tail-{index}" for index in range(30)])

    digest = digest_output(
        "execution",
        text,
        OutputMetabolismPolicy(max_inline_chars=260),
        artifact_root=tmp_path,
        source_id="tool-1",
    )

    assert digest.truncated is True
    assert digest.raw_chars == len(text)
    assert digest.raw_output_path == "raw_outputs/tool-1.txt"
    assert len(digest.inline) <= 260
    assert "head-0" in digest.inline
    assert "tail-29" in digest.inline
    assert "Traceback: important failure" in digest.inline
    assert (tmp_path / digest.raw_output_path).read_text(encoding="utf-8") == text


def test_short_output_is_not_truncated_or_written(tmp_path: Path) -> None:
    digest = digest_output(
        "validation",
        "short output",
        OutputMetabolismPolicy(max_inline_chars=120),
        artifact_root=tmp_path,
        source_id="validation-1",
    )

    assert digest.truncated is False
    assert digest.inline == "short output"
    assert digest.raw_output_path is None
    assert not (tmp_path / "raw_outputs").exists()


def test_digest_is_deterministic_for_same_input(tmp_path: Path) -> None:
    text = "\n".join(f"line-{index}" for index in range(100))
    policy = OutputMetabolismPolicy(max_inline_chars=180)

    first = OutputMetabolismRuntime().digest_output("tool", text, policy, tmp_path, "same-source")
    second = OutputMetabolismRuntime().digest_output("tool", text, policy, tmp_path, "same-source")

    assert first.as_dict() == second.as_dict()
    assert (tmp_path / "raw_outputs" / "same-source.txt").read_text(encoding="utf-8") == text


def test_digest_sanitizes_source_id_for_artifact_path(tmp_path: Path) -> None:
    digest = digest_output(
        "execution",
        "x" * 200,
        OutputMetabolismPolicy(max_inline_chars=80),
        artifact_root=tmp_path,
        source_id="../unsafe/source",
    )

    assert digest.raw_output_path == "raw_outputs/unsafe-source.txt"
    assert (tmp_path / "raw_outputs" / "unsafe-source.txt").exists()

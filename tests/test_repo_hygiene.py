"""Focused tests for repository hygiene guardrails."""

from __future__ import annotations

from pathlib import Path

import scripts.check_repo_hygiene as hygiene


def _write_minimal_readme(root: Path) -> None:
    root.joinpath("README.md").write_text(
        "\n".join(
            [
                "# NeuroSight",
                "",
                "Not for clinical use.",
                "Synthetic demo data only.",
                "Implemented scope and planned scope are separated.",
            ]
        ),
        encoding="utf-8",
    )


def test_hygiene_rejects_duplicate_and_build_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Copied files and TypeScript build caches must fail the hygiene gate."""
    _write_minimal_readme(tmp_path)
    tmp_path.joinpath("MODEL_CARD 2.md").write_text("duplicate", encoding="utf-8")
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    frontend.joinpath("tsconfig.tsbuildinfo").write_text("cache", encoding="utf-8")
    monkeypatch.setattr(hygiene, "ROOT", tmp_path)

    dirs, files = hygiene.iter_paths()
    errors = hygiene.check_forbidden_paths(dirs, files)

    assert any("copied/duplicate artifact" in error for error in errors)
    assert any("TypeScript incremental build cache" in error for error in errors)


def test_hygiene_rejects_tiny_placeholder_images(tmp_path: Path, monkeypatch) -> None:
    """A 1x1 PNG is not acceptable visual evidence."""
    _write_minimal_readme(tmp_path)
    image_path = tmp_path / "docs" / "screenshots" / "placeholder.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00"
    )
    monkeypatch.setattr(hygiene, "ROOT", tmp_path)

    _, files = hygiene.iter_paths()
    errors = hygiene.check_images(files)

    assert any("placeholder-sized image" in error for error in errors)

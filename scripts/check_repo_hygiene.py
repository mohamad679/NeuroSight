#!/usr/bin/env python3
"""Repository hygiene checks for public NeuroSight releases."""

from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
}

FORBIDDEN_DIR_NAMES = {
    "__pycache__": "Python bytecode cache",
    ".venv": "local virtual environment",
    "venv": "local virtual environment",
    "env": "local virtual environment",
    "ENV": "local virtual environment",
    ".ruff_cache": "Ruff cache",
    ".pytest_cache": "pytest cache",
    ".mypy_cache": "mypy cache",
    ".cache": "tool cache",
    ".turbo": "frontend tool cache",
    ".parcel-cache": "frontend tool cache",
    "node_modules": "frontend dependency install",
    ".next": "Next.js build output",
    "out": "frontend static export",
    "coverage": "coverage output",
    "htmlcov": "coverage output",
    "logs": "generated logs/reports",
    "outputs": "Hydra or training output",
    ".deploy": "local deployment staging output",
    "mlruns": "local MLflow runs",
    "mlartifacts": "local MLflow artifacts",
    "__MACOSX": "macOS archive metadata",
}

FORBIDDEN_FILE_NAMES = {
    ".DS_Store": "OS metadata",
    "Thumbs.db": "OS metadata",
}

FORBIDDEN_SUFFIXES = {
    ".pyc": "Python bytecode",
    ".pyo": "Python bytecode",
    ".log": "local log file",
    ".pt": "model checkpoint artifact",
    ".pth": "model checkpoint artifact",
    ".ckpt": "model checkpoint artifact",
    ".onnx": "model export artifact",
    ".ort": "model export artifact",
    ".safetensors": "model checkpoint artifact",
    ".tsbuildinfo": "TypeScript incremental build cache",
    ".tmp": "temporary file",
    ".bak": "backup file",
    ".orig": "merge backup file",
}

ALLOWED_ENV_FILES = {
    ".env.example",
    ".env.local.example",
    "frontend/.env.example",
    "frontend/.env.local.example",
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif"}

README_FORBIDDEN_PATTERNS = {
    r"\bclinically validated\b": "unsupported clinical-validation claim",
    r"\bvalidated clinical product\b": "unsupported clinical-product claim",
    r"\bstate[- ]of[- ]the[- ]art\b": "unsupported benchmark superiority claim",
    r"\bdiagnostic reliability\b": "unsupported reliability claim",
    r"\bclinical performance\b(?!\s+(?:is not|not|is \*\*not measured\*\*))": (
        "clinical performance wording must be explicitly negated"
    ),
    r"\barxiv\b": "arXiv reference or badge is not supported by a real paper artifact",
    r"img\.shields\.io/badge/.{0,80}(?:live|deployed|production)": (
        "status badge implies deployment or production state"
    ),
}

COPY_FILE_PATTERNS = (
    re.compile(r"(^|[\s._-])copy($|[\s._-])", flags=re.IGNORECASE),
    re.compile(r"\s+\d+(?=\.[^.]+$|$)"),
)

MARKDOWN_IMAGE_PATTERNS = (
    re.compile(r"!\[[^\]]*\]\(([^)]+)\)"),
    re.compile(r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"']", flags=re.IGNORECASE),
)


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def iter_paths(allow_dev_caches: bool = False) -> tuple[list[Path], list[Path]]:
    dirs: list[Path] = []
    files: list[Path] = []

    def walk(path: Path) -> None:
        for child in sorted(path.iterdir(), key=lambda item: item.name):
            rel_parts = child.relative_to(ROOT).parts
            if any(part in SKIP_DIRS for part in rel_parts):
                continue
            if child.is_dir():
                if allow_dev_caches and child.name in FORBIDDEN_DIR_NAMES and child.name != "__MACOSX":
                    continue
                dirs.append(child)
                if child.name in FORBIDDEN_DIR_NAMES:
                    continue
                walk(child)
            else:
                files.append(child)

    walk(ROOT)
    return dirs, files


def image_size(path: Path) -> tuple[int, int] | None:
    data = path.read_bytes()
    if path.suffix.lower() == ".png":
        if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
            return struct.unpack(">II", data[16:24])
        return None
    if path.suffix.lower() == ".gif":
        if data[:6] in {b"GIF87a", b"GIF89a"} and len(data) >= 10:
            return struct.unpack("<HH", data[6:10])
        return None
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        return jpeg_size(data)
    return None


def jpeg_size(data: bytes) -> tuple[int, int] | None:
    if not data.startswith(b"\xff\xd8"):
        return None
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        index += 2
        if marker in {0xD8, 0xD9}:
            continue
        if index + 2 > len(data):
            return None
        segment_length = struct.unpack(">H", data[index : index + 2])[0]
        if marker in range(0xC0, 0xC4):
            if index + 7 > len(data):
                return None
            height, width = struct.unpack(">HH", data[index + 3 : index + 7])
            return width, height
        index += segment_length
    return None


def check_forbidden_paths(dirs: list[Path], files: list[Path], allow_dev_caches: bool = False) -> list[str]:
    errors: list[str] = []
    for directory in dirs:
        if directory.name in FORBIDDEN_DIR_NAMES:
            if allow_dev_caches and directory.name != "__MACOSX":
                continue
            errors.append(
                f"{relative(directory)}: remove {FORBIDDEN_DIR_NAMES[directory.name]}"
            )

    for path in files:
        rel = relative(path)
        if path.name in FORBIDDEN_FILE_NAMES:
            errors.append(f"{rel}: remove {FORBIDDEN_FILE_NAMES[path.name]}")
        if path.suffix in FORBIDDEN_SUFFIXES:
            if allow_dev_caches:
                continue
            errors.append(f"{rel}: remove {FORBIDDEN_SUFFIXES[path.suffix]}")
        if any(pattern.search(path.name) for pattern in COPY_FILE_PATTERNS):
            errors.append(f"{rel}: remove copied/duplicate artifact")
        if path.name.startswith(".env") and rel not in ALLOWED_ENV_FILES:
            if allow_dev_caches and (path.name == ".env" or path.name.startswith(".env.")):
                continue
            errors.append(f"{rel}: remove local environment/secrets file")
        if "/.env" in rel and rel not in ALLOWED_ENV_FILES:
            if allow_dev_caches:
                continue
            errors.append(f"{rel}: remove local environment/secrets file")
    return errors


def check_images(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        size = image_size(path)
        rel = relative(path)
        if size is None:
            errors.append(f"{rel}: unsupported or corrupt image header")
            continue
        width, height = size
        if width <= 1 or height <= 1:
            errors.append(f"{rel}: placeholder-sized image ({width}x{height})")
        elif width * height < 10_000:
            errors.append(f"{rel}: suspiciously tiny image ({width}x{height})")
    return errors


def check_readme() -> list[str]:
    readme = ROOT / "README.md"
    if not readme.exists():
        return ["README.md: missing"]
    text = readme.read_text(encoding="utf-8")
    lowered = text.lower()
    errors: list[str] = []
    for pattern, reason in README_FORBIDDEN_PATTERNS.items():
        if re.search(pattern, lowered, flags=re.IGNORECASE | re.DOTALL):
            errors.append(f"README.md: {reason}")
    if "not for clinical use" not in lowered:
        errors.append("README.md: missing explicit 'Not for clinical use' warning")
    if "synthetic" not in lowered:
        errors.append("README.md: missing synthetic-data disclosure")
    if "implemented" not in lowered or "planned" not in lowered:
        errors.append("README.md: must separate implemented and planned scope")
    return errors


def _local_artifact_path(markdown_path: Path, raw_target: str) -> Path | None:
    target = raw_target.strip().split("#", 1)[0].split("?", 1)[0]
    if not target or re.match(r"^[a-z][a-z0-9+.-]*://", target, flags=re.IGNORECASE):
        return None
    if target.startswith("mailto:") or target.startswith("#"):
        return None
    if target.startswith("/"):
        return ROOT / target.lstrip("/")
    if target.startswith(("docs/", "frontend/", "data/", "evaluation/", "scripts/", "tests/")):
        return ROOT / target
    return markdown_path.parent / target


def check_markdown_image_references(files: list[Path]) -> list[str]:
    """Ensure Markdown image artifacts point to real local files."""
    errors: list[str] = []
    markdown_files = [path for path in files if path.suffix.lower() == ".md"]
    for markdown_path in markdown_files:
        text = markdown_path.read_text(encoding="utf-8")
        for pattern in MARKDOWN_IMAGE_PATTERNS:
            for match in pattern.finditer(text):
                artifact = _local_artifact_path(markdown_path, match.group(1))
                if artifact is None:
                    continue
                if not artifact.exists():
                    errors.append(
                        f"{relative(markdown_path)}: image reference is missing: {match.group(1)}"
                    )
    return errors


def main() -> int:
    allow_dev_caches = "--allow-dev-caches" in sys.argv
    dirs, files = iter_paths(allow_dev_caches)
    errors = []
    errors.extend(check_forbidden_paths(dirs, files, allow_dev_caches))
    errors.extend(check_images(files))
    errors.extend(check_readme())
    errors.extend(check_markdown_image_references(files))

    if errors:
        print("Repository hygiene check failed:\n")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Repository hygiene check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

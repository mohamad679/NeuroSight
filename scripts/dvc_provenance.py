#!/usr/bin/env python3
"""Generate a Git-safe data/model provenance manifest.

The manifest records hashes, sizes, and recommended DVC commands for local
data/model artifacts without committing the artifacts themselves. It is designed
to be run directly or through the `provenance_manifest` stage in `dvc.yaml`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List


DEFAULT_PATTERNS = [
    "data/ADNIMERGE_synthetic.csv",
    "data/neurosight_kg.json",
    "data/raw/*.npy",
    "checkpoints/*.pt",
    "checkpoints/*.pth",
    "checkpoints/*.ckpt",
    "logs/model_registry.json",
    "logs/eval_results.json",
    "logs/modality_ablation.json",
]

IGNORED_NAMES = {".DS_Store"}


@dataclass
class ArtifactRecord:
    path: str
    artifact_type: str
    exists: bool
    size_bytes: int | None
    sha256: str | None
    git_policy: str
    dvc_command: str | None


def artifact_type(path: Path) -> str:
    text = path.as_posix()
    if text.startswith("checkpoints/") or path.suffix in {".pt", ".pth", ".ckpt"}:
        return "model_checkpoint"
    if text == "logs/model_registry.json":
        return "model_registry"
    if text.startswith("logs/"):
        return "experiment_output"
    if text.startswith("data/raw/"):
        return "raw_demo_modality"
    if text.startswith("data/"):
        return "demo_dataset"
    return "artifact"


def git_policy(path: Path) -> str:
    text = path.as_posix()
    if text.startswith(("data/", "checkpoints/", "logs/")):
        return "ignored_by_git_policy_track_with_dvc_or_regenerate"
    return "git_trackable"


def dvc_command(path: Path) -> str | None:
    text = path.as_posix()
    if text.startswith(("data/", "checkpoints/")):
        return f"dvc add {text}"
    if text.startswith("logs/"):
        return f"dvc add {text} # optional: track only stable experiment artifacts"
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_artifacts(patterns: Iterable[str]) -> List[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        matches = list(Path(".").glob(pattern))
        if not matches and not any(char in pattern for char in "*?[]"):
            paths.add(Path(pattern))
        for match in matches:
            if match.is_file() and match.name not in IGNORED_NAMES:
                paths.add(match)
    return sorted(paths, key=lambda item: item.as_posix())


def build_record(path: Path) -> ArtifactRecord:
    relative = Path(path.as_posix())
    exists = relative.exists()
    return ArtifactRecord(
        path=relative.as_posix(),
        artifact_type=artifact_type(relative),
        exists=exists,
        size_bytes=relative.stat().st_size if exists else None,
        sha256=sha256_file(relative) if exists else None,
        git_policy=git_policy(relative),
        dvc_command=dvc_command(relative),
    )


def load_registry_summary(registry_path: Path) -> dict[str, Any]:
    if not registry_path.exists():
        return {"exists": False, "runs": 0, "production_run_id": None}

    try:
        parsed = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"exists": True, "runs": 0, "production_run_id": None, "error": "unreadable"}

    entries = parsed if isinstance(parsed, list) else parsed.get("runs", []) if isinstance(parsed, dict) else []
    if not isinstance(entries, list):
        entries = []
    production = next(
        (
            str(entry.get("run_id"))
            for entry in entries
            if isinstance(entry, dict) and entry.get("status") == "production"
        ),
        None,
    )
    return {
        "exists": True,
        "runs": len([entry for entry in entries if isinstance(entry, dict)]),
        "production_run_id": production,
    }


def build_manifest(patterns: Iterable[str]) -> dict[str, Any]:
    records = [build_record(path) for path in discover_artifacts(patterns)]
    existing = [record for record in records if record.exists]
    total_bytes = sum(record.size_bytes or 0 for record in existing)
    return {
        "schema": "neurosight.provenance.v1",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "summary": {
            "artifact_count": len(records),
            "existing_artifact_count": len(existing),
            "total_existing_bytes": total_bytes,
            "git_safe": True,
            "note": "Manifest contains hashes and metadata only; data/checkpoint bytes remain outside Git.",
        },
        "registry": load_registry_summary(Path("logs/model_registry.json")),
        "artifacts": [asdict(record) for record in records],
    }


def print_summary(manifest: dict[str, Any], output: Path) -> None:
    summary = manifest["summary"]
    print("")
    print("NeuroSight DVC provenance manifest")
    print(f"Output: {output.as_posix()}")
    print(f"Artifacts discovered: {summary['artifact_count']}")
    print(f"Existing artifacts: {summary['existing_artifact_count']}")
    print(f"Total existing bytes: {summary['total_existing_bytes']}")
    print("")
    for artifact in manifest["artifacts"]:
        marker = "FOUND" if artifact["exists"] else "MISS"
        size = artifact["size_bytes"] if artifact["size_bytes"] is not None else "-"
        print(f"[{marker}] {artifact['path']} ({artifact['artifact_type']}, size={size})")
        if artifact["dvc_command"]:
            print(f"        {artifact['dvc_command']}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a hash-based provenance manifest for DVC-tracked data/model artifacts."
    )
    parser.add_argument(
        "--output",
        default="logs/dvc_provenance_manifest.json",
        help="Manifest output path.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        help="Artifact glob to scan. Can be repeated. Defaults to NeuroSight data/checkpoint patterns.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Do not print the human-readable summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.patterns or DEFAULT_PATTERNS)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not args.json_only:
        print_summary(manifest, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

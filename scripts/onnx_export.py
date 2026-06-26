#!/usr/bin/env python3
"""Export the NeuroSight cognitive classifier to ONNX.

If optional ONNX dependencies are missing, the script writes an honest manifest
showing exactly what is missing. Install the optional Poetry group and rerun the
same command to produce the `.onnx` artifact and ONNX Runtime validation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_ONNX_PATH = "logs/onnx/neurosight_cognitive_classifier.onnx"
DEFAULT_MANIFEST_PATH = "logs/onnx/neurosight_onnx_export_manifest.json"
DEFAULT_CHECKPOINT_PATH = "checkpoints/best_fusion.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export NeuroSight's cognitive classifier to ONNX and validate with ONNX Runtime."
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_ONNX_PATH,
        help=f"Output ONNX path. Default: {DEFAULT_ONNX_PATH}",
    )
    parser.add_argument(
        "--manifest-out",
        default=DEFAULT_MANIFEST_PATH,
        help=f"Output manifest path. Default: {DEFAULT_MANIFEST_PATH}",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Optional NeuroSight checkpoint path. Default: {DEFAULT_CHECKPOINT_PATH}",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Sample batch size used for tracing and validation.",
    )
    parser.add_argument(
        "--skip-runtime-validation",
        action="store_true",
        help="Export ONNX without running ONNX Runtime validation.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if ONNX dependencies are missing.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print manifest JSON to stdout instead of writing a file.",
    )
    return parser.parse_args()


def summarize_manifest(
    manifest: dict[str, Any],
    output_path: Path | None,
    *,
    stream: Any = sys.stdout,
) -> None:
    artifact = manifest.get("onnx_artifact", {})
    runtime = manifest.get("onnxruntime_validation", {})
    print("ONNX EXPORT PASSED" if manifest.get("status") == "exported" else "ONNX EXPORT NOT READY", file=stream)
    print(f"Status: {manifest.get('status')}", file=stream)
    print(f"Model: {manifest.get('model', {}).get('name')}", file=stream)
    print(f"ONNX path: {artifact.get('path')} exists={artifact.get('exists')}", file=stream)
    print(f"Runtime validation: {runtime.get('status')}", file=stream)
    if manifest.get("missing_dependencies"):
        print(f"Missing dependencies: {', '.join(manifest['missing_dependencies'])}", file=stream)
        print("Install: poetry install --with onnx", file=stream)
    print("Clinical boundary: export mechanics only, not clinical validation.", file=stream)
    if output_path is not None:
        print(f"Wrote: {output_path}", file=stream)


def main() -> int:
    from neurosight.deployment import onnx_export

    args = parse_args()
    config = onnx_export.ONNXExportConfig(
        output_path=args.out,
        checkpoint_path=args.checkpoint,
        opset_version=max(11, int(args.opset)),
        batch_size=max(1, int(args.batch_size)),
        validate_runtime=not args.skip_runtime_validation,
    )

    exit_code = 0
    try:
        manifest = onnx_export.export_cognitive_classifier(config)
    except onnx_export.ONNXExportDependencyError as exc:
        manifest = onnx_export.build_missing_dependency_manifest(config, exc.missing)
        exit_code = 1 if args.strict else 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ONNX EXPORT FAILED: {exc}", file=sys.stderr)
        return 1

    if args.stdout:
        print(onnx_export.manifest_to_json(manifest))
        summarize_manifest(manifest, None, stream=sys.stderr)
    else:
        manifest_path = onnx_export.write_manifest(manifest, args.manifest_out)
        summarize_manifest(manifest, manifest_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

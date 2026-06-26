#!/usr/bin/env python3
"""Generate a DICOM/DICOMweb awareness manifest for NeuroSight.

The script can run without input to document current capabilities and future
DICOMweb route planning. With `--input`, it scans a local DICOM file, folder, or
ZIP using pydicom and writes PHI-minimized metadata only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_OUTPUT = "logs/dicomweb/neurosight_dicomweb_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a NeuroSight DICOM/DICOMweb awareness manifest."
    )
    parser.add_argument(
        "--input",
        help="Optional DICOM file, directory, or ZIP to inspect.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=50,
        help="Maximum files to inspect from a directory or ZIP.",
    )
    parser.add_argument(
        "--include-uids",
        action="store_true",
        help="Include raw Study/Series/SOP Instance UIDs. PatientID is still hashed.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print JSON to stdout instead of writing a file.",
    )
    return parser.parse_args()


def summarize_manifest(manifest: dict[str, Any], output_path: Path | None, *, stream: Any = sys.stdout) -> None:
    """Print a short manifest summary."""
    scan = manifest.get("input_scan", {})
    summary = scan.get("summary", {}) if isinstance(scan, dict) else {}
    current = manifest.get("current_support", {})
    print("DICOMWEB MANIFEST PASSED", file=stream)
    print(f"Status: {manifest.get('status')}", file=stream)
    print(f"Current endpoint: {current.get('neurosight_endpoint')}", file=stream)
    print("DICOMweb services: QIDO-RS, WADO-RS, STOW-RS, RS Capabilities", file=stream)
    if summary:
        print(
            "Input scan: "
            f"{summary.get('instances', 0)} instances, "
            f"{summary.get('studies', 0)} studies, "
            f"{summary.get('series', 0)} series",
            file=stream,
        )
    else:
        print("Input scan: no local DICOM input provided", file=stream)
    if output_path is not None:
        print(f"Wrote: {output_path}", file=stream)


def main() -> int:
    from neurosight.interop import dicomweb

    args = parse_args()
    try:
        input_summary = None
        if args.input:
            input_summary = dicomweb.inspect_dicom_input(
                args.input,
                max_instances=max(1, int(args.max_instances)),
                include_uids=bool(args.include_uids),
            )
        manifest = dicomweb.build_dicomweb_awareness_manifest(input_summary=input_summary)
        if args.stdout:
            print(dicomweb.manifest_to_json(manifest))
            summarize_manifest(manifest, None, stream=sys.stderr)
        else:
            output_path = dicomweb.write_manifest(manifest, args.out)
            summarize_manifest(manifest, output_path)
    except (dicomweb.DicomInspectionError, OSError, ValueError) as exc:
        print(f"DICOMWEB MANIFEST FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

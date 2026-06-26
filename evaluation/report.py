"""Evaluation report generation for NeuroSight benchmarks.

Produces JSON and Markdown reports with mandatory provenance metadata
including ``synthetic_data``, ``clinical_validity``, methodology,
dependency versions, seed, and warning text.

No report produced by this module claims clinical validity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.benchmark_table import render_report_markdown


def save_json_report(results: dict[str, Any], output_path: str | Path) -> Path:
    """Serialise benchmark results to a JSON file with provenance metadata.

    The output JSON always contains ``synthetic_data``, ``clinical_validity``,
    ``warning``, ``methodology``, ``seed``, and ``dependency_versions``.
    Any missing fields are inserted with safe defaults so the file is always
    machine-readable and auditable.

    Args:
        results: Benchmark results dictionary returned by ``run_benchmark``.
        output_path: Destination file path (directories created automatically).

    Returns:
        Absolute ``Path`` to the written file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Ensure mandatory provenance fields are present
    safe_results = dict(results)
    safe_results.setdefault("synthetic_data", True)
    safe_results.setdefault("clinical_validity", False)
    safe_results.setdefault(
        "warning",
        "SYNTHETIC BENCHMARK — NOT CLINICAL PERFORMANCE. "
        "Results are from generated data and do not estimate real-world accuracy.",
    )
    safe_results.setdefault("methodology", "Not documented.")
    safe_results.setdefault("dependency_versions", {})

    out.write_text(json.dumps(safe_results, indent=2), encoding="utf-8")
    return out.resolve()


def save_markdown_report(results: dict[str, Any], output_path: str | Path) -> Path:
    """Render benchmark results to a full Markdown report file.

    Args:
        results: Benchmark results dictionary returned by ``run_benchmark``.
        output_path: Destination ``.md`` file path.

    Returns:
        Absolute ``Path`` to the written file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    markdown_content = render_report_markdown(results)
    out.write_text(markdown_content, encoding="utf-8")
    return out.resolve()


def generate_full_report(
    results: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "benchmark_report",
) -> dict[str, Path]:
    """Save both JSON and Markdown reports to an output directory.

    Args:
        results: Benchmark results dictionary returned by ``run_benchmark``.
        output_dir: Directory for output files (created if absent).
        prefix: Filename prefix for generated files.

    Returns:
        Dictionary with keys ``"json"`` and ``"markdown"`` mapping to the
        absolute paths of the written files.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = save_json_report(results, out_dir / f"{prefix}.json")
    md_path = save_markdown_report(results, out_dir / f"{prefix}.md")

    return {"json": json_path, "markdown": md_path}

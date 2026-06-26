"""Checkpoint, registry, and evaluation reporting contract helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    """Convert NaN/inf values into JSON-safe nulls recursively."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _read_json(path: Path) -> Any:
    """Read JSON-ish artifacts, accepting Python's default NaN handling."""
    if not path.exists():
        return None
    return _json_safe(json.loads(path.read_text(encoding="utf-8")))


def _load_registry_entries(registry_path: Path) -> list[dict[str, Any]]:
    """Load model registry entries from supported registry shapes."""
    parsed = _read_json(registry_path)
    if parsed is None:
        return []
    if isinstance(parsed, list):
        return [entry for entry in parsed if isinstance(entry, dict)]
    if isinstance(parsed, dict) and isinstance(parsed.get("runs"), list):
        return [entry for entry in parsed["runs"] if isinstance(entry, dict)]
    return []


def _safe_metric(metrics: dict[str, Any], key: str) -> float | None:
    """Extract a finite metric value when present."""
    value = metrics.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _registry_summary(registry_path: Path) -> dict[str, Any]:
    """Summarize model registry readiness."""
    entries = _load_registry_entries(registry_path)
    production = next((entry for entry in entries if entry.get("status") == "production"), None)

    def _score(entry: dict[str, Any]) -> float:
        metrics = entry.get("metrics", {})
        if not isinstance(metrics, dict):
            return float("-inf")
        value = _safe_metric(metrics, "val_auc")
        return value if value is not None else float("-inf")

    best_run = max(entries, key=_score) if entries else None
    return {
        "path": str(registry_path),
        "exists": registry_path.exists(),
        "run_count": len(entries),
        "production_run_id": production.get("run_id") if production else None,
        "best_run": {
            "run_id": best_run.get("run_id"),
            "status": best_run.get("status"),
            "checkpoint_path": best_run.get("checkpoint_path"),
            "metrics": _json_safe(best_run.get("metrics", {})),
        }
        if best_run
        else None,
    }


def _evaluation_summary(eval_results_path: Path) -> dict[str, Any]:
    """Summarize persisted evaluation results."""
    parsed = _read_json(eval_results_path)
    if not isinstance(parsed, dict):
        return {
            "path": str(eval_results_path),
            "exists": eval_results_path.exists(),
            "available": False,
            "metrics": {},
        }

    return {
        "path": str(eval_results_path),
        "exists": True,
        "available": True,
        "checkpoint_path": parsed.get("checkpoint_path"),
        "epoch": parsed.get("epoch"),
        "metrics": {
            "macro_auc": parsed.get("macro_auc"),
            "ece": parsed.get("ece"),
            "per_class_f1": parsed.get("per_class_f1", {}),
        },
        "confusion_matrix": parsed.get("confusion_matrix", []),
        "modality_ablation": parsed.get("modality_ablation", {}),
        "data_scope": "synthetic_adni_like_only",
    }


def _model_card_summary(model_card_path: Path) -> dict[str, Any]:
    """Return lightweight model-card artifact metadata."""
    return {
        "path": str(model_card_path),
        "exists": model_card_path.exists(),
        "size_bytes": int(model_card_path.stat().st_size) if model_card_path.exists() else 0,
    }


def build_checkpoint_contract(
    checkpoint_path: Path,
    registry_path: Path,
    eval_results_path: Path,
    model_card_path: Path,
    *,
    load_enabled: bool,
    loaded: bool,
    load_error: str | None = None,
) -> dict[str, Any]:
    """Build an honest checkpoint/evaluation reporting payload."""
    checkpoint_path = Path(checkpoint_path)
    registry_path = Path(registry_path)
    eval_results_path = Path(eval_results_path)
    model_card_path = Path(model_card_path)
    exists = checkpoint_path.exists()
    size_bytes = int(checkpoint_path.stat().st_size) if exists else 0

    return {
        "status": "loaded" if loaded else "available" if exists else "missing",
        "checkpoint": {
            "path": str(checkpoint_path),
            "exists": exists,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 2) if size_bytes else 0.0,
        },
        "loading": {
            "enabled": bool(load_enabled),
            "loaded": bool(loaded),
            "error": load_error,
            "env_var": "NEUROSIGHT_LOAD_CHECKPOINT",
        },
        "registry": _registry_summary(registry_path),
        "evaluation": _evaluation_summary(eval_results_path),
        "model_card": _model_card_summary(model_card_path),
        "scientific_claims": {
            "clinical_validation": False,
            "clinical_use_allowed": False,
            "public_metrics_are_clinical_claims": False,
            "training_data": "synthetic_adni_like",
            "real_private_adni_included": False,
            "notice": (
                "Checkpoint/evaluation artifacts are useful for portfolio demonstration only. "
                "They are synthetic-data results and are not clinical performance evidence."
            ),
        },
    }

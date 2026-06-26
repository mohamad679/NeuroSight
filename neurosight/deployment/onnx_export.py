"""ONNX export helpers for NeuroSight models.

The current export target is the cognitive classifier because it is the
smallest model on the active diagnosis path and has a stable tensor contract:
`(batch, 8)` cognitive features in, logits/probabilities/embedding out.
"""

from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from neurosight.contracts import Diagnosis
from neurosight.models.cognitive import CognitiveClassifier
from neurosight.schemas.cognitive import COGNITIVE_FEATURES

COGNITIVE_FEATURE_ORDER: tuple[str, ...] = COGNITIVE_FEATURES


class ONNXExportDependencyError(RuntimeError):
    """Raised when optional ONNX export dependencies are unavailable."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__(f"Missing optional ONNX dependencies: {', '.join(missing)}")


@dataclass(frozen=True)
class ONNXExportConfig:
    """Configuration for exporting the cognitive classifier."""

    output_path: str
    checkpoint_path: str
    opset_version: int = 17
    batch_size: int = 1
    validate_runtime: bool = True


class CognitiveONNXWrapper(nn.Module):
    """Torch module wrapper with ONNX-friendly tensor outputs."""

    def __init__(self, model: CognitiveClassifier) -> None:
        super().__init__()
        self.model = model.eval()

    def forward(self, cognitive_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, embedding = self.model(cognitive_features)
        probabilities = torch.softmax(logits, dim=-1)
        return logits, probabilities, embedding


def utc_now() -> str:
    """Return current UTC time as an ISO-8601 instant."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def dependency_status() -> dict[str, dict[str, Any]]:
    """Return installed/missing status for ONNX export dependencies."""
    packages = {
        "torch": "torch",
        "onnx": "onnx",
        "onnxruntime": "onnxruntime",
    }
    status: dict[str, dict[str, Any]] = {}
    for import_name, package_name in packages.items():
        installed = importlib.util.find_spec(import_name) is not None
        try:
            package_version = version(package_name) if installed else None
        except PackageNotFoundError:
            package_version = None
        status[package_name] = {
            "installed": installed,
            "version": package_version,
        }
    return status


def missing_dependencies(*, validate_runtime: bool = True) -> list[str]:
    """List dependencies required for the requested export mode."""
    status = dependency_status()
    required = ["onnx"]
    if validate_runtime:
        required.append("onnxruntime")
    return [package for package in required if not status[package]["installed"]]


def demo_cognitive_input(batch_size: int = 1) -> torch.Tensor:
    """Build a deterministic cognitive input tensor for export validation."""
    row = torch.tensor(
        [24.0, 20.0, 1.0, 60.0, 150.0, 12.0, 70.0, 1.0],
        dtype=torch.float32,
    )
    return row.unsqueeze(0).repeat(max(1, int(batch_size)), 1)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def load_cognitive_classifier(
    checkpoint_path: str | Path,
    *,
    num_classes: int = len(Diagnosis),
) -> tuple[CognitiveClassifier, dict[str, Any]]:
    """Load a cognitive classifier, optionally from a NeuroSight checkpoint."""
    path = Path(checkpoint_path)
    model = CognitiveClassifier(num_classes=num_classes).eval()
    metadata: dict[str, Any] = {
        "checkpoint_path": str(path),
        "checkpoint_exists": path.exists(),
        "loaded_from_checkpoint": False,
        "checkpoint_keys": [],
        "load_warning": None,
    }
    if not path.exists():
        metadata["load_warning"] = "Checkpoint missing; exported freshly initialized cognitive classifier."
        return model, metadata

    checkpoint = torch.load(str(path), map_location="cpu")
    if isinstance(checkpoint, dict):
        metadata["checkpoint_keys"] = sorted(str(key) for key in checkpoint.keys())
        state = checkpoint.get("cog_state")
        if isinstance(state, dict):
            model.load_state_dict(state)
            metadata["loaded_from_checkpoint"] = True
            metadata["epoch"] = checkpoint.get("epoch")
            metadata["val_auc"] = _json_safe(checkpoint.get("val_auc"))
            return model, metadata

    metadata["load_warning"] = "Checkpoint did not contain `cog_state`; exported freshly initialized cognitive classifier."
    return model, metadata


def _runtime_validation(
    onnx_path: Path,
    sample_input: torch.Tensor,
    reference_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(
        None,
        {"cognitive_features": sample_input.detach().cpu().numpy().astype(np.float32)},
    )
    output_names = ("logits", "probabilities", "embedding")
    checks: dict[str, Any] = {}
    for name, expected, actual in zip(output_names, reference_outputs, ort_outputs):
        expected_np = expected.detach().cpu().numpy()
        actual_np = np.asarray(actual)
        max_abs_diff = float(np.max(np.abs(expected_np - actual_np)))
        checks[name] = {
            "shape": list(actual_np.shape),
            "max_abs_diff": round(max_abs_diff, 8),
            "passed": max_abs_diff <= 1e-4,
        }
    return {
        "status": "passed" if all(item["passed"] for item in checks.values()) else "failed",
        "provider": "CPUExecutionProvider",
        "outputs": checks,
    }


def _onnx_checker(onnx_path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    return {
        "status": "passed",
        "ir_version": int(model.ir_version),
        "producer_name": str(model.producer_name),
    }


def export_cognitive_classifier(config: ONNXExportConfig) -> dict[str, Any]:
    """Export the cognitive classifier to ONNX and optionally validate with ORT."""
    missing = missing_dependencies(validate_runtime=config.validate_runtime)
    if missing:
        raise ONNXExportDependencyError(missing)

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model, checkpoint_metadata = load_cognitive_classifier(config.checkpoint_path)
    wrapper = CognitiveONNXWrapper(model).eval()
    sample_input = demo_cognitive_input(config.batch_size)

    with torch.no_grad():
        reference_outputs = wrapper(sample_input)

    torch.onnx.export(
        wrapper,
        sample_input,
        str(output_path),
        input_names=["cognitive_features"],
        output_names=["logits", "probabilities", "embedding"],
        dynamic_axes={
            "cognitive_features": {0: "batch"},
            "logits": {0: "batch"},
            "probabilities": {0: "batch"},
            "embedding": {0: "batch"},
        },
        opset_version=int(config.opset_version),
        do_constant_folding=True,
    )

    checker = _onnx_checker(output_path)
    runtime = (
        _runtime_validation(output_path, sample_input, reference_outputs)
        if config.validate_runtime
        else {"status": "skipped"}
    )

    return build_manifest(
        status="exported",
        config=config,
        checkpoint_metadata=checkpoint_metadata,
        onnx_path=output_path,
        checker=checker,
        runtime_validation=runtime,
    )


def build_missing_dependency_manifest(config: ONNXExportConfig, missing: list[str]) -> dict[str, Any]:
    """Build a JSON artifact when export cannot run in the current environment."""
    return build_manifest(
        status="missing_dependencies",
        config=config,
        checkpoint_metadata={
            "checkpoint_path": config.checkpoint_path,
            "checkpoint_exists": Path(config.checkpoint_path).exists(),
            "loaded_from_checkpoint": False,
        },
        onnx_path=Path(config.output_path),
        checker={"status": "not_run"},
        runtime_validation={"status": "not_run"},
        extra={
            "missing_dependencies": missing,
            "install": {
                "poetry": "poetry install --with onnx",
                "pip": "pip install onnx onnxruntime",
            },
        },
    )


def build_manifest(
    *,
    status: str,
    config: ONNXExportConfig,
    checkpoint_metadata: dict[str, Any],
    onnx_path: Path,
    checker: dict[str, Any],
    runtime_validation: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe ONNX export manifest."""
    onnx_exists = onnx_path.exists()
    manifest = {
        "project": "NeuroSight",
        "generated_at": utc_now(),
        "task": "onnx_runtime_export",
        "status": status,
        "model": {
            "name": "cognitive_classifier",
            "class": "neurosight.models.cognitive.CognitiveClassifier",
            "classes": [diagnosis.value for diagnosis in Diagnosis],
            "source": "checkpoint_cog_state" if checkpoint_metadata.get("loaded_from_checkpoint") else "fresh_initialization",
        },
        "config": asdict(config),
        "dependencies": dependency_status(),
        "input_contract": {
            "input_name": "cognitive_features",
            "dtype": "float32",
            "shape": ["batch", len(COGNITIVE_FEATURE_ORDER)],
            "feature_order": list(COGNITIVE_FEATURE_ORDER),
        },
        "output_contract": {
            "logits": ["batch", len(Diagnosis)],
            "probabilities": ["batch", len(Diagnosis)],
            "embedding": ["batch", 64],
        },
        "checkpoint": _json_safe(checkpoint_metadata),
        "onnx_artifact": {
            "path": str(onnx_path),
            "exists": onnx_exists,
            "size_bytes": int(onnx_path.stat().st_size) if onnx_exists else 0,
        },
        "onnx_checker": checker,
        "onnxruntime_validation": runtime_validation,
        "clinical_boundary": (
            "ONNX export validates deployment mechanics only. It does not make "
            "the model clinically validated or suitable for medical use."
        ),
    }
    if extra:
        manifest.update(extra)
    return _json_safe(manifest)


def manifest_to_json(manifest: dict[str, Any]) -> str:
    """Serialize an ONNX export manifest with stable formatting."""
    return json.dumps(manifest, indent=2, sort_keys=True)


def write_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
    """Write an ONNX export manifest to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest_to_json(manifest) + "\n", encoding="utf-8")
    return path

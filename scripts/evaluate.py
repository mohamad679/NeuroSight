"""Load checkpoint and compute evaluation metrics on the test split.

Usage:
    python scripts/evaluate.py checkpoints/best_fusion.pt
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure project root is on sys.path for standalone execution.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from evaluation.metrics import (
    compute_auc_roc,
    compute_ece,
    compute_modality_ablation,
    compute_per_class_metrics,
)
from neurosight.contracts import Diagnosis
from neurosight.data.multimodal_dataloader import get_dataloaders
from neurosight.models.cognitive import CognitiveClassifier
from neurosight.models.eeg import EEGClassifier
from neurosight.models.fusion import CrossModalAttentionFusion
from neurosight.models.mri import MRIClassifier


def _resolve_repo_root() -> Path:
    """Resolve project root directory from script location.

    Returns:
        Absolute path to repository root.
    """
    return Path(__file__).resolve().parents[1]


def _resolve_data_path(path_value: Optional[str], root: Path) -> Optional[str]:
    """Resolve configured relative path against repository root.

    Args:
        path_value: Optional path string from config.
        root: Repository root.

    Returns:
        Absolute path string or None.
    """
    if path_value is None:
        return None
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = root / candidate
    return str(candidate.resolve())


def _prepare_data_config(cfg: DictConfig, root: Path) -> dict[str, Any]:
    """Convert config object into dataloader-ready dictionary.

    Args:
        cfg: Configuration with `data` section.
        root: Repository root.

    Returns:
        Resolved plain dictionary for `get_dataloaders`.
    """
    resolved_data = OmegaConf.to_container(cfg.data, resolve=True)
    if not isinstance(resolved_data, dict):
        raise ValueError("Configuration data section must resolve to a dictionary.")
    data_cfg: dict[str, Any] = dict(resolved_data)
    data_cfg["csv_path"] = _resolve_data_path(str(data_cfg["csv_path"]), root)
    data_cfg["mri_dir"] = _resolve_data_path(
        None if data_cfg.get("mri_dir") is None else str(data_cfg["mri_dir"]),
        root,
    )
    data_cfg["eeg_dir"] = _resolve_data_path(
        None if data_cfg.get("eeg_dir") is None else str(data_cfg["eeg_dir"]),
        root,
    )
    return data_cfg


def _encode_with_classifier(
    classifier: torch.nn.Module,
    input_tensor: Optional[torch.Tensor],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Extract embedding from a unimodal classifier.

    Args:
        classifier: Classifier returning `(logits, embedding)`.
        input_tensor: Optional modality tensor.
        device: Active torch device.

    Returns:
        Optional embedding tensor.
    """
    if input_tensor is None:
        return None
    batch_input = input_tensor.to(device=device, dtype=torch.float32)
    _, embedding = classifier(batch_input)
    return embedding


def _extract_batch_embeddings(
    batch: dict[str, Any],
    mri_model: MRIClassifier,
    eeg_model: EEGClassifier,
    cog_model: CognitiveClassifier,
    device: torch.device,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extract MRI/EEG/cognitive embeddings from one batch.

    Args:
        batch: Batch dictionary from multimodal dataloader.
        mri_model: MRI classifier.
        eeg_model: EEG classifier.
        cog_model: Cognitive classifier.
        device: Active torch device.

    Returns:
        Tuple of modality embeddings with optional `None` values.
    """
    mri_embedding = _encode_with_classifier(mri_model, batch.get("mri"), device)
    eeg_embedding = _encode_with_classifier(eeg_model, batch.get("eeg"), device)
    cognitive_embedding = _encode_with_classifier(cog_model, batch.get("cog"), device)
    return mri_embedding, eeg_embedding, cognitive_embedding


def _compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Compute confusion matrix without external dependencies.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.
        n_classes: Total number of classes.

    Returns:
        Confusion matrix of shape `(n_classes, n_classes)`.
    """
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, prediction in zip(y_true.tolist(), y_pred.tolist()):
        matrix[int(truth), int(prediction)] += 1
    return matrix


def _format_confusion_matrix(matrix: np.ndarray, class_names: list[str]) -> str:
    """Create readable confusion matrix text table.

    Args:
        matrix: Confusion matrix.
        class_names: Ordered class name labels.

    Returns:
        Text rendering of confusion matrix.
    """
    header = ["true\\pred"] + class_names
    widths = [max(len(item), 10) for item in header]
    for class_idx, class_name in enumerate(class_names):
        widths[0] = max(widths[0], len(class_name))
        for pred_idx in range(len(class_names)):
            widths[pred_idx + 1] = max(widths[pred_idx + 1], len(str(int(matrix[class_idx, pred_idx]))))

    def _join_row(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    lines = [_join_row(header), "-+-".join("-" * width for width in widths)]
    for row_idx, class_name in enumerate(class_names):
        row_values = [class_name] + [str(int(matrix[row_idx, col_idx])) for col_idx in range(len(class_names))]
        lines.append(_join_row(row_values))
    return "\n".join(lines)


def _safe_macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute macro AUC with dependency-safe fallback.

    Args:
        y_true: Ground-truth labels.
        y_prob: Predicted class probabilities.

    Returns:
        Macro AUC value or NaN.
    """
    try:
        return float(compute_auc_roc(y_true, y_prob).get("macro", float("nan")))
    except (ModuleNotFoundError, ValueError):
        return float("nan")


def _safe_per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float]:
    """Compute per-class F1 with fallback behavior.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        n_classes: Number of classes.

    Returns:
        Mapping from class index keys to F1 scores.
    """
    try:
        per_class_metrics = compute_per_class_metrics(y_true, y_pred)["per_class"]
        return {
            f"class_{class_idx}": float(per_class_metrics[f"class_{class_idx}"]["f1"])
            for class_idx in range(n_classes)
        }
    except (ModuleNotFoundError, ValueError, KeyError):
        return {f"class_{class_idx}": float("nan") for class_idx in range(n_classes)}


def _collect_ablation_samples(
    loader: DataLoader[dict[str, Any]],
    mri_model: MRIClassifier,
    eeg_model: EEGClassifier,
    cog_model: CognitiveClassifier,
    device: torch.device,
    max_samples: int = 256,
) -> list[dict[str, Any]]:
    """Collect embedding-level samples for modality ablation.

    Args:
        loader: Evaluation dataloader.
        mri_model: MRI classifier.
        eeg_model: EEG classifier.
        cog_model: Cognitive classifier.
        device: Active torch device.
        max_samples: Maximum number of samples to collect.

    Returns:
        List of sample dictionaries compatible with `compute_modality_ablation`.
    """
    samples: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device=device, dtype=torch.long)
            mri_embedding, eeg_embedding, cognitive_embedding = _extract_batch_embeddings(
                batch=batch,
                mri_model=mri_model,
                eeg_model=eeg_model,
                cog_model=cog_model,
                device=device,
            )
            batch_size = labels.shape[0]
            for idx in range(batch_size):
                samples.append(
                    {
                        "mri": None
                        if mri_embedding is None
                        else mri_embedding[idx : idx + 1].detach().to(device),
                        "eeg": None
                        if eeg_embedding is None
                        else eeg_embedding[idx : idx + 1].detach().to(device),
                        "cog": None
                        if cognitive_embedding is None
                        else cognitive_embedding[idx : idx + 1].detach().to(device),
                        "label": int(labels[idx].item()),
                    }
                )
                if len(samples) >= max_samples:
                    return samples
    return samples


def evaluate(checkpoint_path: str) -> dict[str, Any]:
    """Evaluate a saved fusion checkpoint on test data.

    Args:
        checkpoint_path: Path to model checkpoint file.

    Returns:
        Dictionary with confusion matrix, AUC, ECE, F1, and ablation metrics.
    """
    root = _resolve_repo_root()
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_absolute():
        checkpoint_file = (root / checkpoint_file).resolve()
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(str(checkpoint_file), map_location=device)

    if "config" in checkpoint:
        cfg = OmegaConf.create(checkpoint["config"])
    else:
        cfg = OmegaConf.load(root / "neurosight" / "configs" / "default.yaml")
    data_cfg = _prepare_data_config(cfg, root)
    dataloaders = get_dataloaders(data_cfg)

    n_classes = int(cfg.model.n_classes)
    diagnoses = [diagnosis.value for diagnosis in Diagnosis]
    if len(diagnoses) != n_classes:
        diagnoses = [f"class_{idx}" for idx in range(n_classes)]

    mri_model = MRIClassifier(num_classes=n_classes).to(device).eval()
    eeg_model = EEGClassifier(num_classes=n_classes).to(device).eval()
    cog_model = CognitiveClassifier(num_classes=n_classes).to(device).eval()
    fusion_model = CrossModalAttentionFusion(num_classes=n_classes).to(device).eval()

    if "mri_state" in checkpoint:
        mri_model.load_state_dict(checkpoint["mri_state"])
    if "eeg_state" in checkpoint:
        eeg_model.load_state_dict(checkpoint["eeg_state"])
    if "cog_state" in checkpoint:
        cog_model.load_state_dict(checkpoint["cog_state"])
    fusion_model.load_state_dict(checkpoint["model_state"])

    y_prob_parts: list[np.ndarray] = []
    y_true_parts: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloaders["test"]:
            labels = batch["label"].to(device=device, dtype=torch.long)
            mri_embedding, eeg_embedding, cognitive_embedding = _extract_batch_embeddings(
                batch=batch,
                mri_model=mri_model,
                eeg_model=eeg_model,
                cog_model=cog_model,
                device=device,
            )
            output = fusion_model(mri=mri_embedding, eeg=eeg_embedding, cog=cognitive_embedding)
            y_prob_parts.append(output["probs"].detach().cpu().numpy())
            y_true_parts.append(labels.detach().cpu().numpy())

    if not y_prob_parts:
        raise ValueError("Test dataloader returned zero batches; evaluation cannot proceed.")

    y_prob = np.vstack(y_prob_parts)
    y_true = np.concatenate(y_true_parts)
    y_pred = np.argmax(y_prob, axis=1)

    confusion = _compute_confusion_matrix(y_true, y_pred, n_classes=n_classes)
    macro_auc = _safe_macro_auc(y_true=y_true, y_prob=y_prob)
    try:
        ece = float(compute_ece(y_true, y_prob))
    except (ModuleNotFoundError, ValueError):
        ece = float("nan")
    per_class_f1 = _safe_per_class_f1(y_true=y_true, y_pred=y_pred, n_classes=n_classes)

    ablation_samples = _collect_ablation_samples(
        loader=dataloaders["test"],
        mri_model=mri_model,
        eeg_model=eeg_model,
        cog_model=cog_model,
        device=device,
    )
    modality_ablation = (
        compute_modality_ablation(fusion_model, ablation_samples) if ablation_samples else {}
    )

    print("Confusion Matrix:")
    print(_format_confusion_matrix(confusion, diagnoses))
    print(f"Macro AUC: {macro_auc:.4f}" if not math.isnan(macro_auc) else "Macro AUC: nan")
    print(f"ECE: {ece:.4f}" if not math.isnan(ece) else "ECE: nan")
    print("Per-class F1:")
    for class_idx in range(n_classes):
        key = f"class_{class_idx}"
        value = per_class_f1.get(key, float("nan"))
        label = diagnoses[class_idx] if class_idx < len(diagnoses) else key
        line = f"  - {label}: {value:.4f}" if not math.isnan(value) else f"  - {label}: nan"
        print(line)
    print("Modality Ablation:")
    if modality_ablation:
        for key, value in modality_ablation.items():
            if isinstance(value, (int, float)) and not math.isnan(float(value)):
                print(f"  - {key}: {float(value):.4f}")
            else:
                print(f"  - {key}: {value}")
    else:
        print("  - unavailable")

    results: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_file),
        "epoch": int(checkpoint.get("epoch", -1)),
        "macro_auc": macro_auc,
        "ece": ece,
        "per_class_f1": per_class_f1,
        "confusion_matrix": confusion.tolist(),
        "modality_ablation": modality_ablation,
    }
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_file = logs_dir / "eval_results.json"
    output_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def _main() -> None:
    """Command-line entry point for checkpoint evaluation."""
    if len(sys.argv) != 2:
        raise ValueError("Usage: python scripts/evaluate.py checkpoints/best_fusion.pt")
    evaluate(sys.argv[1])


if __name__ == "__main__":
    _main()


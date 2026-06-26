"""Cross-validation utilities for NeuroSight model evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation.metrics import compute_auc_roc, compute_ece, compute_per_class_metrics
from neurosight.data.adni_dataset import ADNIDataset
from neurosight.models.cognitive import CognitiveClassifier
from neurosight.models.fusion import CrossModalAttentionFusion


def _set_seeds(seed: int) -> None:
    """Set deterministic random seeds for NumPy and PyTorch.

    Args:
        seed: Seed value used across frameworks.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collect_train_val_cognitive(
    csv_path: str,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load cognitive vectors from train and validation dataset splits.

    Args:
        csv_path: Path to ADNI-compatible CSV file.
        split_seed: Seed controlling deterministic split generation.

    Returns:
        Tuple of feature matrix `(N, 8)` and label vector `(N,)`.
    """
    train_dataset = ADNIDataset(csv_path=csv_path, split="train", split_seed=split_seed)
    val_dataset = ADNIDataset(csv_path=csv_path, split="val", split_seed=split_seed)

    features: list[np.ndarray] = []
    labels: list[int] = []

    for dataset in (train_dataset, val_dataset):
        for index in range(len(dataset)):
            sample = dataset[index]
            cognitive_tensor = sample["cog"]
            label_tensor = sample["label"]
            features.append(cognitive_tensor.detach().cpu().numpy().astype(np.float32))
            labels.append(int(label_tensor.item()))

    if not features:
        raise ValueError("Train/validation splits produced zero samples for cross-validation.")

    return np.stack(features, axis=0), np.array(labels, dtype=np.int64)


def _build_class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for cross-entropy loss.

    Args:
        labels: Integer class labels for training fold.
        n_classes: Total number of diagnosis classes.

    Returns:
        Normalized class weight tensor with shape `(n_classes,)`.
    """
    class_counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    safe_counts = np.where(class_counts > 0, class_counts, 1.0)
    inverse = 1.0 / safe_counts
    normalized = inverse / float(inverse.mean())
    return torch.tensor(normalized, dtype=torch.float32)


def _train_fold(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    n_classes: int,
    cv_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> dict[str, float]:
    """Train one cross-validation fold and compute validation metrics.

    Args:
        train_x: Training feature matrix.
        train_y: Training labels.
        val_x: Validation feature matrix.
        val_y: Validation labels.
        n_classes: Number of output classes.
        cv_epochs: Number of epochs for fold training.
        batch_size: Batch size for fold training.
        lr: Optimizer learning rate.
        weight_decay: Optimizer weight decay.
        seed: Seed for deterministic fold behavior.

    Returns:
        Dictionary containing `val_auc`, `val_f1`, and `ece`.
    """
    if cv_epochs <= 0:
        raise ValueError("cv_epochs must be a positive integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    _set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(val_x, dtype=torch.float32),
        torch.tensor(val_y, dtype=torch.long),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    cognitive_model = CognitiveClassifier(num_classes=n_classes).to(device)
    fusion_model = CrossModalAttentionFusion(num_classes=n_classes).to(device)

    class_weights = _build_class_weights(train_y, n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        list(cognitive_model.parameters()) + list(fusion_model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    for _ in range(cv_epochs):
        cognitive_model.train()
        fusion_model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            _, cognitive_embedding = cognitive_model(x_batch)
            output = fusion_model(mri=None, eeg=None, cog=cognitive_embedding)
            loss = criterion(output["logits"], y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(cognitive_model.parameters()) + list(fusion_model.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

    cognitive_model.eval()
    fusion_model.eval()

    probability_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            _, cognitive_embedding = cognitive_model(x_batch)
            output = fusion_model(mri=None, eeg=None, cog=cognitive_embedding)
            probability_parts.append(output["probs"].detach().cpu().numpy())
            label_parts.append(y_batch.detach().cpu().numpy())

    if not probability_parts:
        raise ValueError("Validation split produced no batches in fold training.")

    y_prob = np.vstack(probability_parts)
    y_true = np.concatenate(label_parts)
    y_pred = np.argmax(y_prob, axis=1)

    try:
        val_auc = float(compute_auc_roc(y_true, y_prob)["macro"])
    except (ModuleNotFoundError, ValueError, KeyError):
        val_auc = float("nan")

    try:
        val_f1 = float(compute_per_class_metrics(y_true, y_pred)["macro_f1"])
    except (ModuleNotFoundError, ValueError, KeyError):
        val_f1 = float("nan")

    try:
        ece = float(compute_ece(y_true, y_prob))
    except (ModuleNotFoundError, ValueError):
        ece = float("nan")

    return {"val_auc": val_auc, "val_f1": val_f1, "ece": ece}


def run_kfold_cv(config: dict[str, Any], k: int = 5) -> dict[str, Any]:
    """Run stratified K-fold cross-validation on train+val data.

    Args:
        config: Runtime configuration containing at least `seed`, `csv_path`,
            and `cv_epochs`. Optional keys include `batch_size`, `lr`,
            `weight_decay`, `n_classes`, and `split_seed`.
        k: Number of folds for stratified cross-validation.

    Returns:
        Dictionary with per-fold metrics and aggregate mean/std statistics.
    """
    if k < 2:
        raise ValueError("k must be at least 2 for cross-validation.")

    try:
        from sklearn.model_selection import StratifiedKFold
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-learn is required for StratifiedKFold cross-validation."
        ) from exc

    seed = int(config.get("seed", 42))
    _set_seeds(seed)

    csv_path = str(config.get("csv_path", "data/ADNIMERGE_synthetic.csv"))
    split_seed = int(config.get("split_seed", seed))
    n_classes = int(config.get("n_classes", 6))
    cv_epochs = int(config.get("cv_epochs", 20))
    batch_size = int(config.get("batch_size", 16))
    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 0.01))

    features, labels = _collect_train_val_cognitive(csv_path=csv_path, split_seed=split_seed)

    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    per_fold: list[dict[str, float | int]] = []

    for fold_index, (train_indices, val_indices) in enumerate(splitter.split(features, labels), start=1):
        fold_metrics = _train_fold(
            train_x=features[train_indices],
            train_y=labels[train_indices],
            val_x=features[val_indices],
            val_y=labels[val_indices],
            n_classes=n_classes,
            cv_epochs=cv_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed + fold_index,
        )
        per_fold.append(
            {
                "fold": fold_index,
                "val_auc": float(fold_metrics["val_auc"]),
                "val_f1": float(fold_metrics["val_f1"]),
                "ece": float(fold_metrics["ece"]),
            }
        )

    auc_values = np.array([float(entry["val_auc"]) for entry in per_fold], dtype=np.float64)
    f1_values = np.array([float(entry["val_f1"]) for entry in per_fold], dtype=np.float64)

    return {
        "k": k,
        "per_fold": per_fold,
        "mean_val_auc": float(np.nanmean(auc_values)),
        "std_val_auc": float(np.nanstd(auc_values)),
        "mean_val_f1": float(np.nanmean(f1_values)),
        "std_val_f1": float(np.nanstd(f1_values)),
    }

"""Training script for NeuroSight multimodal fusion model.

Usage:
    python scripts/train.py
    python scripts/train.py training.epochs=100
    python scripts/train.py data.csv_path=my.csv
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

# Ensure project root is on sys.path so Hydra's working-directory change
# does not break relative imports (e.g. evaluation.metrics, neurosight.*).
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import hydra
import numpy as np
import torch
import torch.nn as nn
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from evaluation.metrics import (
    compute_auc_roc,
    compute_modality_ablation,
    compute_per_class_metrics,
)
from neurosight.data.multimodal_dataloader import get_dataloaders
from neurosight.models.cognitive import CognitiveClassifier
from neurosight.models.eeg import EEGClassifier
from neurosight.models.fusion import CrossModalAttentionFusion
from neurosight.models.mri import MRIClassifier
from neurosight.tracking.experiment_logger import ExperimentLogger
from neurosight.tracking.model_registry import ModelRegistry


def _set_global_seeds(seed: int) -> None:
    """Set deterministic seeds for NumPy and PyTorch.

    Args:
        seed: Integer random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _flatten_dict(values: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
    """Flatten nested dictionaries using dotted keys.

    Args:
        values: Dictionary to flatten.
        parent_key: Prefix for recursive flattening.

    Returns:
        Flat dictionary with dotted keys.
    """
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            flattened.update(_flatten_dict(value, parent_key=new_key))
        else:
            flattened[new_key] = value
    return flattened


@dataclass
class _EpochStats:
    """Container with training and validation metrics for one epoch."""

    train_loss: float
    train_grad_norm: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    val_macro_auc: float


class _TrackingBackend:
    """MLflow logger with automatic JSONL fallback."""

    def __init__(self, experiment_name: str, tracking_uri: Optional[str]) -> None:
        """Initialize tracking backend.

        Args:
            experiment_name: MLflow experiment name.
            tracking_uri: Optional MLflow server URI.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._mlflow: Any = None
        self._run_active = False
        self._fallback_path = Path(to_absolute_path("logs/mlflow_fallback.jsonl"))
        self._fallback_run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    def start_run(self, run_name: str, params: dict[str, Any]) -> None:
        """Start tracking run and log initial parameters.

        Args:
            run_name: Human-readable run name.
            params: Flattened training configuration.
        """
        try:
            import mlflow
            from mlflow.exceptions import MlflowException

            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run(run_name=run_name)
            safe_params = {key: str(value) for key, value in params.items()}
            mlflow.log_params(safe_params)
            self._mlflow = mlflow
            self._run_active = True
        except ModuleNotFoundError:
            self._log_fallback(
                {
                    "event": "run_start",
                    "run_name": run_name,
                    "params": params,
                }
            )
        except (OSError, RuntimeError, ValueError, ConnectionError, MlflowException):
            self._log_fallback(
                {
                    "event": "run_start",
                    "run_name": run_name,
                    "params": params,
                }
            )

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log epoch metrics to active backend.

        Args:
            metrics: Scalar metric mapping.
            step: Epoch index.
        """
        finite_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
        if self._mlflow is not None and self._run_active and finite_metrics:
            self._mlflow.log_metrics(finite_metrics, step=step)
            return
        self._log_fallback({"event": "metrics", "step": step, "metrics": metrics})

    def log_dict(self, name: str, payload: dict[str, Any]) -> None:
        """Log dictionary artifact to backend.

        Args:
            name: Artifact name identifier.
            payload: JSON-serializable payload.
        """
        if self._mlflow is not None and self._run_active:
            temp_path = Path(to_absolute_path("logs")) / f"{name}.json"
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._mlflow.log_artifact(str(temp_path))
            return
        self._log_fallback({"event": name, "payload": payload})

    def end_run(self) -> None:
        """Close active tracking run."""
        if self._mlflow is not None and self._run_active:
            self._mlflow.end_run()
            self._run_active = False

    def _log_fallback(self, entry: dict[str, Any]) -> None:
        """Append event entry to local JSONL fallback log.

        Args:
            entry: Event payload dictionary.
        """
        self._fallback_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "run_id": self._fallback_run_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **entry,
        }
        with self._fallback_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")


def _prepare_data_config(cfg: DictConfig) -> dict[str, Any]:
    """Resolve data paths for dataloader creation.

    Args:
        cfg: Full Hydra config.

    Returns:
        Resolved plain dictionary for `get_dataloaders`.
    """
    raw_data = OmegaConf.to_container(cfg.data, resolve=True)
    if not isinstance(raw_data, dict):
        raise ValueError("cfg.data must resolve to a dictionary.")
    data_cfg: dict[str, Any] = dict(raw_data)

    if data_cfg.get("csv_path") is not None:
        data_cfg["csv_path"] = to_absolute_path(str(data_cfg["csv_path"]))
    if data_cfg.get("mri_dir") is not None:
        data_cfg["mri_dir"] = to_absolute_path(str(data_cfg["mri_dir"]))
    if data_cfg.get("eeg_dir") is not None:
        data_cfg["eeg_dir"] = to_absolute_path(str(data_cfg["eeg_dir"]))
    return data_cfg


def _freeze_module(module: nn.Module) -> None:
    """Disable gradient updates for all module parameters.

    Args:
        module: Module to freeze.
    """
    for parameter in module.parameters():
        parameter.requires_grad = False


def _unfreeze_module(module: nn.Module) -> None:
    """Enable gradient updates for all module parameters.

    Args:
        module: Module to unfreeze.
    """
    for parameter in module.parameters():
        parameter.requires_grad = True


def _configure_warmup_phase(
    mri_model: MRIClassifier,
    eeg_model: EEGClassifier,
    cog_model: CognitiveClassifier,
    fusion_model: CrossModalAttentionFusion,
) -> None:
    """Configure warmup phase by freezing unimodal encoders.

    Args:
        mri_model: MRI classifier.
        eeg_model: EEG classifier.
        cog_model: Cognitive classifier.
        fusion_model: Fusion model.
    """
    _freeze_module(mri_model)
    _freeze_module(eeg_model)
    _freeze_module(cog_model)
    _freeze_module(fusion_model)

    _unfreeze_module(fusion_model.mri_proj)
    _unfreeze_module(fusion_model.eeg_proj)
    _unfreeze_module(fusion_model.cog_proj)
    _unfreeze_module(fusion_model.head)
    fusion_model.temperature.requires_grad = True

    mri_model.eval()
    eeg_model.eval()
    cog_model.eval()
    fusion_model.train()


def _configure_finetune_phase(
    mri_model: MRIClassifier,
    eeg_model: EEGClassifier,
    cog_model: CognitiveClassifier,
    fusion_model: CrossModalAttentionFusion,
) -> None:
    """Configure fine-tuning phase by unfreezing all parameters.

    Args:
        mri_model: MRI classifier.
        eeg_model: EEG classifier.
        cog_model: Cognitive classifier.
        fusion_model: Fusion model.
    """
    _unfreeze_module(mri_model)
    _unfreeze_module(eeg_model)
    _unfreeze_module(cog_model)
    _unfreeze_module(fusion_model)

    mri_model.train()
    eeg_model.train()
    cog_model.train()
    fusion_model.train()


def _has_trainable_params(module: nn.Module) -> bool:
    """Check whether module has any trainable parameters.

    Args:
        module: Module to inspect.

    Returns:
        True when at least one parameter requires gradients.
    """
    return any(parameter.requires_grad for parameter in module.parameters())


def _encode_with_classifier(
    classifier: nn.Module,
    input_tensor: Optional[torch.Tensor],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Run classifier to obtain embedding when modality tensor is available.

    Args:
        classifier: Unimodal classifier returning `(logits, embedding)`.
        input_tensor: Optional batch input tensor for the modality.
        device: Device where inference/training occurs.

    Returns:
        Optional embedding tensor with batch dimension.
    """
    if input_tensor is None:
        return None
    batch_input = input_tensor.to(device=device, dtype=torch.float32)
    if _has_trainable_params(classifier):
        _, embedding = classifier(batch_input)
    else:
        with torch.no_grad():
            _, embedding = classifier(batch_input)
    return embedding


def _extract_batch_embeddings(
    batch: dict[str, Any],
    mri_model: MRIClassifier,
    eeg_model: EEGClassifier,
    cog_model: CognitiveClassifier,
    device: torch.device,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extract modality embeddings for fusion model input.

    Args:
        batch: DataLoader batch dictionary.
        mri_model: MRI classifier.
        eeg_model: EEG classifier.
        cog_model: Cognitive classifier.
        device: Active torch device.

    Returns:
        Tuple of MRI, EEG, and cognitive embedding tensors or `None`.
    """
    mri_embedding = _encode_with_classifier(mri_model, batch.get("mri"), device)
    eeg_embedding = _encode_with_classifier(eeg_model, batch.get("eeg"), device)
    cognitive_embedding = _encode_with_classifier(cog_model, batch.get("cog"), device)
    return mri_embedding, eeg_embedding, cognitive_embedding


def _run_train_epoch(
    train_loader: DataLoader[dict[str, Any]],
    mri_model: MRIClassifier,
    eeg_model: EEGClassifier,
    cog_model: CognitiveClassifier,
    fusion_model: CrossModalAttentionFusion,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch.

    Args:
        train_loader: Training data loader.
        mri_model: MRI classifier.
        eeg_model: EEG classifier.
        cog_model: Cognitive classifier.
        fusion_model: Fusion model.
        criterion: Loss function.
        optimizer: Optimizer instance.
        device: Active torch device.

    Returns:
        Tuple with mean training loss and mean pre-clipping gradient norm.
    """
    total_loss = 0.0
    total_grad_norm = 0.0
    num_batches = 0

    for batch in train_loader:
        labels = batch["label"].to(device=device, dtype=torch.long)
        mri_embedding, eeg_embedding, cognitive_embedding = _extract_batch_embeddings(
            batch=batch,
            mri_model=mri_model,
            eeg_model=eeg_model,
            cog_model=cog_model,
            device=device,
        )

        optimizer.zero_grad()
        output = fusion_model(mri=mri_embedding, eeg=eeg_embedding, cog=cognitive_embedding)
        loss = criterion(output["logits"], labels)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ],
            max_norm=1.0,
        )
        optimizer.step()

        total_loss += float(loss.item())
        grad_norm_value = float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
        if math.isfinite(grad_norm_value):
            total_grad_norm += grad_norm_value
        num_batches += 1

    if num_batches == 0:
        raise ValueError("Training DataLoader must contain at least one batch.")
    return total_loss / float(num_batches), total_grad_norm / float(num_batches)


def _safe_macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute macro AUC while handling optional dependency failures.

    Args:
        y_true: Ground-truth class labels.
        y_prob: Predicted class probabilities.

    Returns:
        Macro AUC value or NaN when unavailable.
    """
    try:
        return float(compute_auc_roc(y_true, y_prob).get("macro", float("nan")))
    except (ModuleNotFoundError, ValueError):
        return float("nan")


def _safe_accuracy_and_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Compute accuracy and macro-F1 safely.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.

    Returns:
        Tuple `(accuracy, macro_f1)` where values may be NaN if unavailable.
    """
    try:
        metrics = compute_per_class_metrics(y_true, y_pred)
        return float(metrics["accuracy"]), float(metrics["macro_f1"])
    except (ModuleNotFoundError, ValueError, KeyError):
        return float("nan"), float("nan")


def _run_validation_epoch(
    val_loader: DataLoader[dict[str, Any]],
    mri_model: MRIClassifier,
    eeg_model: EEGClassifier,
    cog_model: CognitiveClassifier,
    fusion_model: CrossModalAttentionFusion,
    criterion: nn.Module,
    device: torch.device,
) -> _EpochStats:
    """Run one validation epoch and compute classification metrics.

    Args:
        val_loader: Validation data loader.
        mri_model: MRI classifier.
        eeg_model: EEG classifier.
        cog_model: Cognitive classifier.
        fusion_model: Fusion model.
        criterion: Loss function.
        device: Active torch device.

    Returns:
        Epoch stats containing loss and validation metrics.
    """
    mri_model.eval()
    eeg_model.eval()
    cog_model.eval()
    fusion_model.eval()

    total_loss = 0.0
    num_batches = 0
    all_probabilities: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in val_loader:
            labels = batch["label"].to(device=device, dtype=torch.long)
            mri_embedding, eeg_embedding, cognitive_embedding = _extract_batch_embeddings(
                batch=batch,
                mri_model=mri_model,
                eeg_model=eeg_model,
                cog_model=cog_model,
                device=device,
            )
            output = fusion_model(mri=mri_embedding, eeg=eeg_embedding, cog=cognitive_embedding)
            loss = criterion(output["logits"], labels)

            probabilities = output["probs"].detach().cpu().numpy()
            predictions = np.argmax(probabilities, axis=1)

            total_loss += float(loss.item())
            num_batches += 1
            all_probabilities.append(probabilities)
            all_predictions.append(predictions)
            all_labels.append(labels.detach().cpu().numpy())

    if num_batches == 0:
        raise ValueError("Validation DataLoader must contain at least one batch.")

    y_prob = np.vstack(all_probabilities)
    y_pred = np.concatenate(all_predictions)
    y_true = np.concatenate(all_labels)

    val_macro_auc = _safe_macro_auc(y_true=y_true, y_prob=y_prob)
    val_accuracy, val_macro_f1 = _safe_accuracy_and_f1(y_true=y_true, y_pred=y_pred)
    val_loss = total_loss / float(num_batches)

    return _EpochStats(
        train_loss=float("nan"),
        train_grad_norm=float("nan"),
        val_loss=val_loss,
        val_accuracy=val_accuracy,
        val_macro_f1=val_macro_f1,
        val_macro_auc=val_macro_auc,
    )


def _is_better_auc(candidate_auc: float, best_auc: float) -> bool:
    """Compare macro AUC values with NaN handling.

    Args:
        candidate_auc: Candidate score.
        best_auc: Current best score.

    Returns:
        True when `candidate_auc` improves over `best_auc`.
    """
    if math.isnan(candidate_auc):
        return False
    if math.isnan(best_auc):
        return True
    return candidate_auc > best_auc


def _collect_ablation_samples(
    loader: DataLoader[dict[str, Any]],
    mri_model: MRIClassifier,
    eeg_model: EEGClassifier,
    cog_model: CognitiveClassifier,
    device: torch.device,
    max_samples: int = 256,
) -> list[dict[str, Any]]:
    """Collect embedding-level samples for modality ablation analysis.

    Args:
        loader: Source DataLoader (typically test split).
        mri_model: MRI classifier.
        eeg_model: EEG classifier.
        cog_model: Cognitive classifier.
        device: Active torch device.
        max_samples: Upper bound on collected samples.

    Returns:
        List of dictionaries compatible with `compute_modality_ablation`.
    """
    mri_model.eval()
    eeg_model.eval()
    cog_model.eval()
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
                sample = {
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
                samples.append(sample)
                if len(samples) >= max_samples:
                    return samples
    return samples


def _format_summary(best_auc: float, best_epoch: int, checkpoint_path: Path) -> str:
    """Build unicode summary table for terminal output.

    Args:
        best_auc: Best macro validation AUC.
        best_epoch: Epoch index where best AUC was obtained.
        checkpoint_path: Path to saved best checkpoint.

    Returns:
        Rendered box summary string.
    """
    lines = [
        "NeuroSight Training Complete",
        f"Best Val AUC (macro): {best_auc:.4f}" if not math.isnan(best_auc) else "Best Val AUC (macro): nan",
        f"Best epoch: {best_epoch}",
        f"Checkpoint: {checkpoint_path}",
    ]
    width = max(len(line) for line in lines) + 4
    top = "╔" + ("═" * width) + "╗"
    body = "\n".join(f"║  {line.ljust(width - 2)}║" for line in lines)
    bottom = "╚" + ("═" * width) + "╝"
    return "\n".join([top, body, bottom])


def train(cfg: DictConfig) -> None:
    """Run two-phase multimodal fusion training with tracking and checkpointing.

    Args:
        cfg: Hydra configuration for data, model, and optimization settings.
    """
    _set_global_seeds(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = _prepare_data_config(cfg)
    dataloaders = get_dataloaders(data_cfg)

    mri_enc = MRIClassifier(num_classes=int(cfg.model.n_classes)).to(device)
    eeg_enc = EEGClassifier(num_classes=int(cfg.model.n_classes)).to(device)
    cog_enc = CognitiveClassifier(num_classes=int(cfg.model.n_classes)).to(device)
    fusion = CrossModalAttentionFusion(num_classes=int(cfg.model.n_classes)).to(device)

    train_dataset = dataloaders["train"].dataset
    class_weights = (
        train_dataset.get_class_weights()
        if hasattr(train_dataset, "get_class_weights")
        else torch.ones(int(cfg.model.n_classes), dtype=torch.float32)
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    checkpoint_dir = Path(to_absolute_path(str(cfg.training.checkpoint_dir)))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_fusion.pt"

    experiment_logger = ExperimentLogger(
        experiment_name=str(cfg.mlflow.experiment_name),
        tracking_uri=None if cfg.mlflow.tracking_uri is None else str(cfg.mlflow.tracking_uri),
    )
    model_registry = ModelRegistry(registry_path=str(Path(to_absolute_path("logs/model_registry.json"))))
    params = _flatten_dict(OmegaConf.to_container(cfg, resolve=True))
    run_name = f"train_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_tags = {
        "project": str(cfg.project.name),
        "version": str(cfg.project.version),
        "stage": "training",
    }
    experiment_logger.start_run(run_name=run_name, tags=run_tags)
    experiment_logger.log_params(params)

    best_auc = float("nan")
    best_f1 = float("nan")
    best_epoch = 0
    epochs_without_improvement = 0
    global_epoch = 0
    patience = int(cfg.training.patience)
    stop_training = False

    warmup_epochs = int(cfg.training.warmup_epochs)
    finetune_epochs = int(cfg.training.finetune_epochs)
    phase_specs = [("warmup", warmup_epochs), ("finetune", finetune_epochs)]

    for phase_name, phase_epochs in phase_specs:
        if phase_epochs <= 0:
            continue

        if phase_name == "warmup":
            _configure_warmup_phase(mri_enc, eeg_enc, cog_enc, fusion)
            optimizer = torch.optim.AdamW(
                [parameter for parameter in fusion.parameters() if parameter.requires_grad],
                lr=float(cfg.training.lr_warmup),
                weight_decay=float(cfg.training.weight_decay),
            )
            scheduler: Optional[CosineAnnealingWarmRestarts] = None
        else:
            _configure_finetune_phase(mri_enc, eeg_enc, cog_enc, fusion)
            optimizer = torch.optim.AdamW(
                [
                    parameter
                    for model in (mri_enc, eeg_enc, cog_enc, fusion)
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                lr=float(cfg.training.lr_finetune),
                weight_decay=float(cfg.training.weight_decay),
            )
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)

        for _ in range(phase_epochs):
            global_epoch += 1
            if phase_name == "warmup":
                fusion.train()
            else:
                mri_enc.train()
                eeg_enc.train()
                cog_enc.train()
                fusion.train()

            train_loss, train_grad_norm = _run_train_epoch(
                train_loader=dataloaders["train"],
                mri_model=mri_enc,
                eeg_model=eeg_enc,
                cog_model=cog_enc,
                fusion_model=fusion,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )

            val_stats = _run_validation_epoch(
                val_loader=dataloaders["val"],
                mri_model=mri_enc,
                eeg_model=eeg_enc,
                cog_model=cog_enc,
                fusion_model=fusion,
                criterion=criterion,
                device=device,
            )
            epoch_stats = _EpochStats(
                train_loss=train_loss,
                train_grad_norm=train_grad_norm,
                val_loss=val_stats.val_loss,
                val_accuracy=val_stats.val_accuracy,
                val_macro_f1=val_stats.val_macro_f1,
                val_macro_auc=val_stats.val_macro_auc,
            )

            experiment_logger.log_metrics(
                {
                    "train_loss": epoch_stats.train_loss,
                    "train_grad_norm": epoch_stats.train_grad_norm,
                    "val_loss": epoch_stats.val_loss,
                    "val_accuracy": epoch_stats.val_accuracy,
                    "val_macro_f1": epoch_stats.val_macro_f1,
                    "val_macro_auc": epoch_stats.val_macro_auc,
                },
                step=global_epoch,
            )

            if _is_better_auc(epoch_stats.val_macro_auc, best_auc):
                best_auc = epoch_stats.val_macro_auc
                best_f1 = epoch_stats.val_macro_f1
                best_epoch = global_epoch
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": global_epoch,
                        "model_state": fusion.state_dict(),
                        "val_auc": best_auc,
                        "config": OmegaConf.to_container(cfg, resolve=True),
                        "mri_state": mri_enc.state_dict(),
                        "eeg_state": eeg_enc.state_dict(),
                        "cog_state": cog_enc.state_dict(),
                    },
                    str(checkpoint_path),
                )
            else:
                epochs_without_improvement += 1

            if scheduler is not None:
                scheduler.step(global_epoch)

            if epochs_without_improvement >= patience:
                stop_training = True
                break

        if stop_training:
            break

    if not checkpoint_path.exists():
        if best_epoch == 0:
            best_epoch = global_epoch
        torch.save(
            {
                "epoch": best_epoch,
                "model_state": fusion.state_dict(),
                "val_auc": best_auc,
                "config": OmegaConf.to_container(cfg, resolve=True),
                "mri_state": mri_enc.state_dict(),
                "eeg_state": eeg_enc.state_dict(),
                "cog_state": cog_enc.state_dict(),
            },
            str(checkpoint_path),
        )

    ablation_metrics: dict[str, float] = {}
    ablation_samples = _collect_ablation_samples(
        loader=dataloaders["test"],
        mri_model=mri_enc,
        eeg_model=eeg_enc,
        cog_model=cog_enc,
        device=device,
    )
    if ablation_samples:
        ablation_metrics = compute_modality_ablation(fusion, ablation_samples)
        ablation_path = Path(to_absolute_path("logs/modality_ablation.json"))
        ablation_path.parent.mkdir(parents=True, exist_ok=True)
        ablation_path.write_text(json.dumps(ablation_metrics, indent=2), encoding="utf-8")
        experiment_logger.log_artifact(str(ablation_path), artifact_name="reports")

    summary_metrics = {
        "val_auc": float(best_auc),
        "val_f1": float(best_f1),
    }
    run_id = experiment_logger.register_model(
        model_path=str(checkpoint_path),
        model_name="neurosight_fusion_v1",
        metrics=summary_metrics,
    )
    model_registry.register_model(
        run_id=run_id,
        model_name="neurosight_fusion_v1",
        checkpoint_path=str(checkpoint_path),
        metrics=summary_metrics,
        config=OmegaConf.to_container(cfg, resolve=True),
        status="staging",
    )

    experiment_logger.end_run(status="FINISHED")
    print(_format_summary(best_auc=best_auc, best_epoch=best_epoch, checkpoint_path=checkpoint_path))


@hydra.main(version_base=None, config_path="../neurosight/configs", config_name="default")
def _main(cfg: DictConfig) -> None:
    """Hydra entry point for training.

    Args:
        cfg: Hydra-loaded configuration.
    """
    train(cfg)


if __name__ == "__main__":
    _main()

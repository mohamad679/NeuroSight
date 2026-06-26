"""DataLoader construction for NeuroSight multimodal training."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from neurosight.data.adni_dataset import ADNIDataset


def _stack_or_none(tensors: list[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    """Stack tensors, filling missing elements with NaN tensors of the same shape.

    Args:
        tensors: Optional tensor list for one modality across the batch.

    Returns:
        Stacked tensor with NaNs for missing items, or None if all are None.
    """
    if not tensors:
        return None
    first_tensor = next((t for t in tensors if t is not None), None)
    if first_tensor is None:
        return None
    shape = first_tensor.shape
    dtype = first_tensor.dtype
    device = first_tensor.device

    filled_tensors = []
    for t in tensors:
        if t is None:
            filled_tensors.append(torch.full(shape, float("nan"), dtype=dtype, device=device))
        else:
            filled_tensors.append(t)
    return torch.stack(filled_tensors, dim=0)


def multimodal_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate multimodal samples while preserving missing modalities.

    Args:
        batch: List of dataset item dictionaries.

    Returns:
        Batched dictionary where missing modalities are represented by stacked tensors containing NaN.
    """
    mri_tensors = [sample.get("mri") for sample in batch]
    eeg_tensors = [sample.get("eeg") for sample in batch]
    cog_tensors = [sample.get("cog") for sample in batch]
    labels = [sample["label"] for sample in batch]
    patient_ids = [str(sample["patient_id"]) for sample in batch]

    return {
        "mri": _stack_or_none(mri_tensors),
        "eeg": _stack_or_none(eeg_tensors),
        "cog": _stack_or_none(cog_tensors),
        "label": torch.stack(labels, dim=0),
        "patient_id": patient_ids,
    }


def get_dataloaders(config: dict[str, Any]) -> dict[str, DataLoader[dict[str, Any]]]:
    """Create train/val/test DataLoaders for ADNI-compatible multimodal data.

    Args:
        config: Data configuration containing paths and loader parameters.

    Returns:
        Dictionary with `train`, `val`, and `test` DataLoader objects.
    """
    split_seed = int(config.get("split_seed", 42))
    np.random.seed(split_seed)
    torch.manual_seed(split_seed)

    csv_path = str(config["csv_path"])
    mri_dir = config.get("mri_dir")
    eeg_dir = config.get("eeg_dir")
    batch_size = int(config.get("batch_size", 16))
    num_workers = int(config.get("num_workers", 0))

    train_dataset = ADNIDataset(
        csv_path=csv_path,
        mri_dir=mri_dir,
        eeg_dir=eeg_dir,
        split="train",
        split_seed=split_seed,
    )
    val_dataset = ADNIDataset(
        csv_path=csv_path,
        mri_dir=mri_dir,
        eeg_dir=eeg_dir,
        split="val",
        split_seed=split_seed,
    )
    test_dataset = ADNIDataset(
        csv_path=csv_path,
        mri_dir=mri_dir,
        eeg_dir=eeg_dir,
        split="test",
        split_seed=split_seed,
    )

    class_weights = train_dataset.get_class_weights()
    train_labels = torch.tensor(train_dataset.get_labels(), dtype=torch.long)
    sample_weights = class_weights[train_labels].to(dtype=torch.double)
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(split_seed)
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=int(sample_weights.shape[0]),
        replacement=True,
        generator=sampler_generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}


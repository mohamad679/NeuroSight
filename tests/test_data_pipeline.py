"""Data pipeline tests for ADNI-compatible loading and batching."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from neurosight.data.adni_dataset import ADNIDataset
from neurosight.data.multimodal_dataloader import multimodal_collate_fn
from neurosight.data.synthetic import ADNI_COLUMNS, generate_structured_synthetic


def _row_count_and_columns(table: Any) -> tuple[int, list[str]]:
    """Extract row count and columns from pandas or list-backed table."""
    if hasattr(table, "columns"):
        return int(len(table)), [str(column) for column in list(table.columns)]
    if isinstance(table, list):
        if not table:
            return 0, []
        first_row = table[0]
        return len(table), [str(column) for column in first_row.keys()]
    raise TypeError(f"Unsupported synthetic table type: {type(table)}")


def test_synthetic_generation_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Synthetic dataset produces correct column names and row count."""
    np.random.seed(7)
    torch.manual_seed(7)
    monkeypatch.chdir(tmp_path)

    synthetic = generate_structured_synthetic(n_per_class=5, seed=7)
    row_count, columns = _row_count_and_columns(synthetic)

    assert row_count == 30, "Synthetic generator must create 6 classes * n_per_class rows."
    assert columns == list(ADNI_COLUMNS), "Synthetic CSV/DataFrame columns must match ADNI schema."
    assert (
        tmp_path / "data" / "ADNIMERGE_synthetic.csv"
    ).exists(), "Synthetic generator must write data/ADNIMERGE_synthetic.csv."


def test_adni_dataset_splits_are_disjoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Train/val/test patient IDs do not overlap."""
    np.random.seed(19)
    torch.manual_seed(19)
    monkeypatch.chdir(tmp_path)

    generate_structured_synthetic(n_per_class=15, seed=19)
    csv_path = tmp_path / "data" / "ADNIMERGE_synthetic.csv"

    train_dataset = ADNIDataset(csv_path=str(csv_path), split="train", split_seed=19)
    val_dataset = ADNIDataset(csv_path=str(csv_path), split="val", split_seed=19)
    test_dataset = ADNIDataset(csv_path=str(csv_path), split="test", split_seed=19)

    train_ids = set(train_dataset.patient_ids)
    val_ids = set(val_dataset.patient_ids)
    test_ids = set(test_dataset.patient_ids)

    assert train_ids.isdisjoint(val_ids), "Train and validation patient IDs must be disjoint."
    assert train_ids.isdisjoint(test_ids), "Train and test patient IDs must be disjoint."
    assert val_ids.isdisjoint(test_ids), "Validation and test patient IDs must be disjoint."


def test_adni_dataset_class_weights_sum(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Class weights are positive and len matches n_classes."""
    np.random.seed(23)
    torch.manual_seed(23)
    monkeypatch.chdir(tmp_path)

    generate_structured_synthetic(n_per_class=12, seed=23)
    csv_path = tmp_path / "data" / "ADNIMERGE_synthetic.csv"

    train_dataset = ADNIDataset(csv_path=str(csv_path), split="train", split_seed=23)
    class_weights = train_dataset.get_class_weights()

    assert (
        class_weights.shape == (train_dataset.n_classes,)
    ), "Class weight tensor shape must equal (n_classes,)."
    assert bool(torch.all(class_weights > 0).item()), "Every class weight must be strictly positive."
    assert float(class_weights.sum().item()) > 0.0, "Class weights must have a positive total sum."


def test_collate_fn_handles_missing_mri(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Batch with None MRI does not raise and returns None for mri key."""
    np.random.seed(29)
    torch.manual_seed(29)
    monkeypatch.chdir(tmp_path)

    generate_structured_synthetic(n_per_class=10, seed=29)
    csv_path = tmp_path / "data" / "ADNIMERGE_synthetic.csv"
    missing_mri_dir = tmp_path / "data" / "mri"
    missing_mri_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = ADNIDataset(
        csv_path=str(csv_path),
        mri_dir=str(missing_mri_dir),
        split="train",
        split_seed=29,
    )
    batch_samples = [train_dataset[index] for index in range(min(4, len(train_dataset)))]
    batch = multimodal_collate_fn(batch_samples)

    assert batch["mri"] is None, "Collate function must keep MRI batch as None when any item is missing."
    assert isinstance(batch["cog"], torch.Tensor), "Cognitive modality must still be stacked as a tensor."
    assert batch["label"].shape[0] == len(
        batch_samples
    ), "Label batch dimension must match number of collated samples."


def test_cognitive_normalization_uses_train_stats(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Val set uses train-split mean/std, not its own stats."""
    np.random.seed(31)
    torch.manual_seed(31)
    monkeypatch.chdir(tmp_path)

    generate_structured_synthetic(n_per_class=20, seed=31)
    csv_path = tmp_path / "data" / "ADNIMERGE_synthetic.csv"

    train_dataset = ADNIDataset(csv_path=str(csv_path), split="train", split_seed=31)
    val_dataset = ADNIDataset(csv_path=str(csv_path), split="val", split_seed=31)

    train_mean = train_dataset.scaler_state["mean"]
    train_std = train_dataset.scaler_state["std"]
    val_mean = val_dataset.scaler_state["mean"]
    val_std = val_dataset.scaler_state["std"]

    assert np.allclose(
        train_mean, val_mean
    ), "Validation dataset must reuse train-split means for normalization."
    assert np.allclose(
        train_std, val_std
    ), "Validation dataset must reuse train-split standard deviations for normalization."

    first_patient_id = val_dataset.patient_ids[0]
    first_index = val_dataset.patient_ids.index(first_patient_id)
    val_item = val_dataset[first_index]
    raw_features = val_dataset.raw_cognitive_by_patient[first_patient_id]
    expected_normalized = (raw_features - train_mean) / train_std

    np.testing.assert_allclose(
        val_item["cog"].numpy(),
        expected_normalized,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Validation cognitive features must be normalized with train split statistics.",
    )


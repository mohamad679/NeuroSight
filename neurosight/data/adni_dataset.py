"""ADNI-compatible multimodal dataset for NeuroSight."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, cast

import numpy as np
import torch
from torch.utils.data import Dataset

from neurosight.contracts import Diagnosis
from neurosight.data.synthetic import ADNI_COLUMNS, generate_structured_synthetic

try:
    from sklearn.model_selection import StratifiedShuffleSplit
except ModuleNotFoundError:
    StratifiedShuffleSplit = None

COGNITIVE_FEATURE_ORDER: tuple[str, ...] = (
    "MMSE",
    "MOCA",
    "CDRSB",
    "ADAS11",
    "RAVLT_immediate",
    "RAVLT_learning",
    "FAQ",
    "AGE",
)


@dataclass(frozen=True)
class _PatientRow:
    """Container for one patient record before split-specific normalization."""

    patient_id: str
    label: int
    cognitive_raw: np.ndarray
    mri_path: Optional[Path]
    eeg_path: Optional[Path]


def _to_float(value: Any) -> float:
    """Parse possibly missing tabular values to float."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    parsed = str(value).strip()
    if not parsed or parsed.upper() in {"NAN", "NA", "NULL"}:
        return float("nan")
    try:
        return float(parsed)
    except ValueError:
        return float("nan")


def _normalize_patient_id(rid_value: Any) -> str:
    """Normalize RID to a canonical string used for filename lookup."""
    patient_id = str(rid_value).strip()
    if patient_id.endswith(".0"):
        trimmed = patient_id[:-2]
        if trimmed.isdigit():
            patient_id = trimmed
    return patient_id


def _map_dx_to_diagnosis(dx_raw: Any) -> Optional[Diagnosis]:
    """Map ADNI `DX_bl` labels to NeuroSight diagnosis enum."""
    normalized = str(dx_raw).strip().upper().replace(" ", "")
    if normalized in {"CN", "NORMAL", "NC"}:
        return Diagnosis.NORMAL
    if "MCI" in normalized:
        return Diagnosis.MCI
    if normalized in {"DEMENTIA", "AD", "ALZHEIMER", "ALZHEIMERSDISEASE"}:
        return Diagnosis.AD
    if "FTD" in normalized:
        return Diagnosis.FTD
    if "LBD" in normalized or "LEWYBODY" in normalized:
        return Diagnosis.LBD
    if normalized == "VD" or "VASCULAR" in normalized:
        return Diagnosis.VD
    return None


def _read_csv_rows(csv_path: Path) -> list[dict[str, Any]]:
    """Read ADNI CSV and validate expected columns."""
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {csv_path} is missing a header row.")

        missing_columns = sorted(set(ADNI_COLUMNS) - set(reader.fieldnames))
        if missing_columns:
            missing_csv = ", ".join(missing_columns)
            raise ValueError(f"CSV file {csv_path} is missing required columns: {missing_csv}.")

        return list(reader)


def _fallback_stratified_split(
    labels: np.ndarray,
    train_frac: float,
    val_frac: float,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback stratified split when scikit-learn is unavailable."""
    rng = np.random.default_rng(split_seed)
    label_to_indices: dict[int, list[int]] = {}
    for idx, label in enumerate(labels.tolist()):
        label_to_indices.setdefault(int(label), []).append(idx)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for class_indices in label_to_indices.values():
        shuffled = np.array(class_indices, dtype=np.int64)
        rng.shuffle(shuffled)
        class_count = int(shuffled.size)
        if class_count < 3:
            raise ValueError(
                "Each diagnosis class must contain at least 3 samples for train/val/test split."
            )

        n_train = int(np.floor(class_count * train_frac))
        n_val = int(np.floor(class_count * val_frac))
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        n_test = class_count - n_train - n_val

        if n_test <= 0:
            n_test = 1
            if n_val > 1:
                n_val -= 1
            else:
                n_train -= 1

        train_indices.extend(shuffled[:n_train].tolist())
        val_indices.extend(shuffled[n_train : n_train + n_val].tolist())
        test_indices.extend(shuffled[n_train + n_val : n_train + n_val + n_test].tolist())

    return (
        np.array(train_indices, dtype=np.int64),
        np.array(val_indices, dtype=np.int64),
        np.array(test_indices, dtype=np.int64),
    )


def _compute_split_indices(
    labels: np.ndarray,
    train_frac: float,
    val_frac: float,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute stratified train/val/test indices with deterministic randomness."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in the open interval (0, 1).")
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be in the open interval (0, 1).")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be less than 1.0.")

    if StratifiedShuffleSplit is None:
        return _fallback_stratified_split(labels, train_frac, val_frac, split_seed)

    all_indices = np.arange(labels.shape[0], dtype=np.int64)
    try:
        first_stage = StratifiedShuffleSplit(
            n_splits=1,
            train_size=train_frac,
            random_state=split_seed,
        )
        train_idx, remaining_idx = next(first_stage.split(all_indices, labels))
    except ValueError:
        return _fallback_stratified_split(labels, train_frac, val_frac, split_seed)

    remaining_labels = labels[remaining_idx]
    remaining_fraction = 1.0 - train_frac
    val_share = val_frac / remaining_fraction

    try:
        second_stage = StratifiedShuffleSplit(
            n_splits=1,
            train_size=val_share,
            random_state=split_seed,
        )
        val_relative, test_relative = next(
            second_stage.split(np.arange(remaining_idx.shape[0]), remaining_labels)
        )
        val_idx = remaining_idx[val_relative]
        test_idx = remaining_idx[test_relative]
    except ValueError:
        return _fallback_stratified_split(labels, train_frac, val_frac, split_seed)

    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)


def _default_mri_transform(mri_array: np.ndarray) -> torch.Tensor:
    """Fallback MRI transform used when MONAI transforms are unavailable."""
    if mri_array.ndim == 3:
        return torch.from_numpy(mri_array).unsqueeze(0).to(dtype=torch.float32)
    if mri_array.ndim == 4 and mri_array.shape[0] == 1:
        return torch.from_numpy(mri_array).to(dtype=torch.float32)
    raise ValueError(f"Expected MRI shape (D,H,W) or (1,D,H,W), got {mri_array.shape}.")


class ADNIDataset(Dataset[dict[str, Any]]):
    """ADNI-compatible multimodal dataset with deterministic split normalization.

    Args:
        csv_path: Path to ADNI `ADNIMERGE.csv`.
        mri_dir: Optional path to MRI `.npy` volumes named by RID.
        eeg_dir: Optional path to EEG `.npy` tensors named by RID.
        split: Dataset split to load (`train`, `val`, or `test`).
        split_seed: Seed controlling deterministic stratified splitting.
        train_frac: Fraction of data used for train split.
        val_frac: Fraction of data used for validation split.
    """

    def __init__(
        self,
        csv_path: str,
        mri_dir: Optional[str] = None,
        eeg_dir: Optional[str] = None,
        split: str = "train",
        split_seed: int = 42,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test.")

        np.random.seed(split_seed)
        torch.manual_seed(split_seed)

        self.split = split
        self.split_seed = split_seed
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.csv_path = Path(csv_path)
        self.mri_dir = Path(mri_dir) if mri_dir is not None else None
        self.eeg_dir = Path(eeg_dir) if eeg_dir is not None else None
        self.n_classes = len(Diagnosis)

        rows = self._load_rows()
        patient_rows = self._build_patient_rows(rows)
        if not patient_rows:
            raise ValueError("No valid patient rows found after diagnosis mapping and filtering.")

        labels = np.array([row.label for row in patient_rows], dtype=np.int64)
        train_idx, val_idx, test_idx = _compute_split_indices(
            labels=labels,
            train_frac=train_frac,
            val_frac=val_frac,
            split_seed=split_seed,
        )

        if split == "train":
            selected_indices = train_idx
        elif split == "val":
            selected_indices = val_idx
        else:
            selected_indices = test_idx

        train_cognitive = np.stack(
            [patient_rows[i].cognitive_raw for i in train_idx], axis=0
        ).astype(np.float32)
        train_means = np.nanmean(train_cognitive, axis=0)
        train_means = np.where(np.isnan(train_means), 0.0, train_means).astype(np.float32)
        train_stds = np.nanstd(train_cognitive, axis=0)
        train_stds = np.where(
            np.logical_or(np.isnan(train_stds), train_stds < 1e-6),
            1.0,
            train_stds,
        ).astype(np.float32)

        self.scaler_state: dict[str, np.ndarray] = {
            "mean": train_means,
            "std": train_stds,
            "feature_order": np.array(COGNITIVE_FEATURE_ORDER, dtype=object),
        }

        self._samples: list[dict[str, Any]] = []
        self.patient_ids: list[str] = []
        self.raw_cognitive_by_patient: dict[str, np.ndarray] = {}

        for idx in selected_indices.tolist():
            source_row = patient_rows[idx]
            raw_filled = np.where(
                np.isnan(source_row.cognitive_raw),
                train_means,
                source_row.cognitive_raw,
            ).astype(np.float32)
            normalized = ((raw_filled - train_means) / train_stds).astype(np.float32)

            self.raw_cognitive_by_patient[source_row.patient_id] = raw_filled.copy()
            self.patient_ids.append(source_row.patient_id)
            self._samples.append(
                {
                    "patient_id": source_row.patient_id,
                    "label": int(source_row.label),
                    "cog": normalized,
                    "mri_path": source_row.mri_path,
                    "eeg_path": source_row.eeg_path,
                }
            )

        self._mri_transform = self._resolve_mri_transform()

    def _load_rows(self) -> list[dict[str, Any]]:
        """Load rows from CSV or synthetic fallback."""
        if self.csv_path.exists():
            return _read_csv_rows(self.csv_path)

        synthetic_data = generate_structured_synthetic(seed=self.split_seed)
        if isinstance(synthetic_data, list):
            return synthetic_data
        return cast(list[dict[str, Any]], synthetic_data.to_dict(orient="records"))

    def _build_patient_rows(self, rows: list[dict[str, Any]]) -> list[_PatientRow]:
        """Create one canonical row per patient ID."""
        deduplicated: dict[str, _PatientRow] = {}
        for row in rows:
            diagnosis = _map_dx_to_diagnosis(row.get("DX_bl"))
            if diagnosis is None:
                continue

            patient_id = _normalize_patient_id(row.get("RID"))
            if not patient_id:
                continue

            cognitive_values = np.array(
                [
                    _to_float(row.get("MMSE")),
                    _to_float(row.get("MOCA")),
                    _to_float(row.get("CDRSB")),
                    _to_float(row.get("ADAS11")),
                    _to_float(row.get("RAVLT_immediate")),
                    _to_float(row.get("RAVLT_learning")),
                    _to_float(row.get("FAQ")),
                    _to_float(row.get("AGE")),
                ],
                dtype=np.float32,
            )

            mri_path = self.mri_dir / f"{patient_id}.npy" if self.mri_dir is not None else None
            eeg_path = self.eeg_dir / f"{patient_id}.npy" if self.eeg_dir is not None else None

            if patient_id not in deduplicated:
                deduplicated[patient_id] = _PatientRow(
                    patient_id=patient_id,
                    label=int(list(Diagnosis).index(diagnosis)),
                    cognitive_raw=cognitive_values,
                    mri_path=mri_path,
                    eeg_path=eeg_path,
                )

        return list(deduplicated.values())

    def _resolve_mri_transform(self) -> Callable[[np.ndarray], torch.Tensor]:
        """Resolve MRI transform callable with MONAI fallback behavior."""
        try:
            from neurosight.models.mri import get_mri_transforms
        except ModuleNotFoundError:
            return _default_mri_transform
        return cast(Callable[[np.ndarray], torch.Tensor], get_mri_transforms())

    def _load_mri(self, mri_path: Optional[Path]) -> Optional[torch.Tensor]:
        """Load and transform MRI array if file exists."""
        if mri_path is None or not mri_path.exists():
            return None
        mri_array = np.load(mri_path, allow_pickle=False).astype(np.float32)
        transformed = self._mri_transform(mri_array)
        return transformed.to(dtype=torch.float32)

    @staticmethod
    def _load_optional_eeg(eeg_path: Optional[Path]) -> Optional[torch.Tensor]:
        """Load EEG tensor if `.npy` path exists."""
        if eeg_path is None or not eeg_path.exists():
            return None
        eeg_array = np.load(eeg_path, allow_pickle=False).astype(np.float32)
        return torch.from_numpy(eeg_array).to(dtype=torch.float32)

    def __len__(self) -> int:
        """Return number of samples in the current split."""
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one multimodal sample for the requested index.

        Args:
            index: Integer sample index.

        Returns:
            Dictionary with `mri`, `eeg`, `cog`, `label`, and `patient_id`.
        """
        sample = self._samples[index]
        return {
            "mri": self._load_mri(sample["mri_path"]),
            "eeg": self._load_optional_eeg(sample["eeg_path"]),
            "cog": torch.from_numpy(sample["cog"]).to(dtype=torch.float32),
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "patient_id": sample["patient_id"],
        }

    def get_labels(self) -> list[int]:
        """Return integer labels for all samples in the current split."""
        return [int(sample["label"]) for sample in self._samples]

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for loss reweighting.

        Returns:
            Tensor of shape `(n_classes,)` with positive normalized class weights.
        """
        labels = np.array(self.get_labels(), dtype=np.int64)
        counts = np.bincount(labels, minlength=self.n_classes).astype(np.float32)
        safe_counts = np.where(counts > 0, counts, 1.0)
        inverse_frequency = 1.0 / safe_counts
        normalized = inverse_frequency / float(inverse_frequency.mean())
        return torch.tensor(normalized, dtype=torch.float32)


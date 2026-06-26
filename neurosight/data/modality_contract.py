"""MRI/EEG modality preprocessing contract helpers."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np


MRI_MODEL_TENSOR_SHAPE: tuple[int, int, int, int, int] = (1, 1, 96, 96, 96)
EEG_MODEL_TENSOR_SHAPE: tuple[int, int, int] = (1, 19, 1024)
MRI_SUPPORTED_FORMATS: tuple[str, ...] = (".npy", ".nii", ".nii.gz", "DICOM .zip")
EEG_SUPPORTED_FORMATS: tuple[str, ...] = (".npy", ".edf")


def _module_available(module_name: str) -> bool:
    """Return whether an optional preprocessing dependency can be imported."""
    return find_spec(module_name) is not None


def infer_mri_format(filename: str | None) -> str:
    """Infer MRI source format from filename."""
    normalized = str(filename or "").lower()
    if normalized.endswith(".nii.gz"):
        return "nifti_gz"
    if normalized.endswith(".nii"):
        return "nifti"
    if normalized.endswith(".zip"):
        return "dicom_zip"
    if normalized.endswith(".npy"):
        return "numpy"
    return "unknown"


def infer_eeg_format(filename: str | None) -> str:
    """Infer EEG source format from filename."""
    suffix = Path(str(filename or "").lower()).suffix
    if suffix == ".npy":
        return "numpy"
    if suffix == ".edf":
        return "edf"
    return "unknown"


def _shape(array: np.ndarray) -> list[int]:
    """Return a JSON-safe array shape."""
    return [int(value) for value in array.shape]


def _value_count(array: np.ndarray) -> int:
    """Return a JSON-safe value count."""
    return int(array.size)


def inspect_mri_array(array: np.ndarray, filename: str | None = None) -> dict[str, Any]:
    """Describe MRI preprocessing that will be applied before encoder inference."""
    source_shape = _shape(array)
    input_layout = "unsupported"
    accepted = False

    if array.ndim == 3:
        input_layout = "volume_dhw"
        accepted = True
    elif array.ndim == 4 and array.shape[0] == 1:
        input_layout = "channel_volume_cd_hw"
        accepted = True
    elif array.ndim == 4 and array.shape[-1] == 1:
        input_layout = "volume_dhw_channel"
        accepted = True

    return {
        "modality": "mri",
        "source_format": infer_mri_format(filename),
        "source_shape": source_shape,
        "source_dtype": str(array.dtype),
        "value_count": _value_count(array),
        "accepted_layout": accepted,
        "input_layout": input_layout,
        "model_tensor_shape": list(MRI_MODEL_TENSOR_SHAPE),
        "preprocessing_steps": [
            "parse_source_file",
            "validate_3d_volume",
            "ensure_channel_first",
            "scale_intensity",
            "resize_to_96x96x96",
            "convert_to_float32_tensor",
        ],
        "quality_notes": [
            "Synthetic/public demo mode does not perform clinical-grade MRI QC.",
            "Real ADNI-style use should add orientation, spacing, skull-stripping, and scanner protocol QC.",
        ],
    }


def inspect_eeg_array(array: np.ndarray, filename: str | None = None) -> dict[str, Any]:
    """Describe EEG preprocessing that will be applied before encoder inference."""
    source_shape = _shape(array)
    input_layout = "unsupported"
    accepted = False
    operation = "reject"

    if array.ndim == 3:
        input_layout = "epochs_channels_time"
        accepted = array.shape[1] == EEG_MODEL_TENSOR_SHAPE[1] or array.shape[2] == EEG_MODEL_TENSOR_SHAPE[1]
        operation = "average_epochs_then_pad_or_truncate_time"
    elif array.ndim == 2:
        if array.shape[0] == EEG_MODEL_TENSOR_SHAPE[1]:
            input_layout = "channels_time"
            accepted = True
            operation = "pad_or_truncate_time"
        elif array.shape[1] == EEG_MODEL_TENSOR_SHAPE[1]:
            input_layout = "time_channels_transposed"
            accepted = True
            operation = "transpose_then_pad_or_truncate_time"

    return {
        "modality": "eeg",
        "source_format": infer_eeg_format(filename),
        "source_shape": source_shape,
        "source_dtype": str(array.dtype),
        "value_count": _value_count(array),
        "accepted_layout": accepted,
        "input_layout": input_layout,
        "normalization_operation": operation,
        "model_tensor_shape": list(EEG_MODEL_TENSOR_SHAPE),
        "preprocessing_steps": [
            "parse_source_file",
            "validate_19_channels",
            "average_epochs_when_present",
            "transpose_if_time_by_channels",
            "pad_or_truncate_to_1024_timepoints",
            "convert_to_float32_tensor",
        ],
        "edf_steps": [
            "load_edf_with_mne",
            "bandpass_1_to_40_hz",
            "notch_filter_50_and_60_hz",
            "average_reference",
            "fixed_length_4_second_epochs",
            "epochwise_z_score",
        ],
        "quality_notes": [
            "Synthetic/public demo mode does not perform clinical-grade EEG artifact rejection.",
            "Real EEG use should add channel montage checks, bad-channel detection, ICA/artifact review, and sampling-rate QC.",
        ],
    }


def build_modality_contract(limits: dict[str, int]) -> dict[str, Any]:
    """Build public MRI/EEG preprocessing readiness metadata."""
    return {
        "status": "ready",
        "mri": {
            "model_tensor_shape": list(MRI_MODEL_TENSOR_SHAPE),
            "embedding_dim": 768,
            "supported_formats": list(MRI_SUPPORTED_FORMATS),
            "accepted_array_shapes": ["(D,H,W)", "(1,D,H,W)", "(D,H,W,1)"],
            "preprocessing_steps": inspect_mri_array(np.zeros((2, 2, 2), dtype=np.float32))[
                "preprocessing_steps"
            ],
            "limits": {
                "max_values": int(limits["max_mri_values"]),
                "max_upload_bytes": int(limits["max_upload_bytes"]),
                "max_dicom_zip_members": int(limits["max_dicom_zip_members"]),
            },
            "optional_dependencies": {
                "monai": _module_available("monai"),
                "nibabel": _module_available("nibabel"),
                "pydicom": _module_available("pydicom"),
            },
        },
        "eeg": {
            "model_tensor_shape": list(EEG_MODEL_TENSOR_SHAPE),
            "embedding_dim": 256,
            "supported_formats": list(EEG_SUPPORTED_FORMATS),
            "accepted_array_shapes": ["(19,time)", "(time,19)", "(epochs,19,time)"],
            "preprocessing_steps": inspect_eeg_array(np.zeros((19, 1024), dtype=np.float32))[
                "preprocessing_steps"
            ],
            "edf_steps": inspect_eeg_array(np.zeros((19, 1024), dtype=np.float32))["edf_steps"],
            "limits": {
                "max_values": int(limits["max_eeg_values"]),
                "max_upload_bytes": int(limits["max_upload_bytes"]),
            },
            "optional_dependencies": {
                "mne": _module_available("mne"),
            },
        },
        "cognitive": {
            "model_tensor_shape": [1, 8],
            "embedding_dim": 64,
            "required_or_defaulted_features": [
                "MMSE",
                "MOCA",
                "CDRSB",
                "ADAS11",
                "RAVLT_immediate",
                "RAVLT_learning",
                "FAQ",
                "AGE",
            ],
        },
        "scientific_notice": (
            "MRI and EEG ingestion is real-data-shaped, but public demo checkpoints are unvalidated. "
            "Clinical-grade preprocessing and external validation are future work."
        ),
    }

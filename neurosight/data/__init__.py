"""Data utilities for ADNI-compatible NeuroSight training pipelines."""

from neurosight.data.adni_dataset import ADNIDataset
from neurosight.data.demo_contract import build_data_contract
from neurosight.data.modality_contract import build_modality_contract
from neurosight.data.multimodal_dataloader import get_dataloaders, multimodal_collate_fn
from neurosight.data.synthetic import generate_structured_synthetic

__all__ = [
    "ADNIDataset",
    "build_data_contract",
    "build_modality_contract",
    "get_dataloaders",
    "multimodal_collate_fn",
    "generate_structured_synthetic",
]

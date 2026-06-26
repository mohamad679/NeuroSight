"""Unit tests for EEG encoder architecture constraints."""

from __future__ import annotations

import torch

from neurosight.models.eeg import EEGEncoder


def test_eeg_encoder_output_shape_and_parameter_budget() -> None:
    """EEGEncoder outputs 256-d embeddings and stays under parameter budget."""
    torch.manual_seed(42)
    encoder = EEGEncoder()
    inputs = torch.randn(4, 19, 1024)

    with torch.no_grad():
        outputs = encoder(inputs)

    assert outputs.shape == (4, 256), "EEGEncoder output must have shape (B, 256)."
    total_trainable = sum(
        parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad
    )
    assert (
        total_trainable < 500_000
    ), "EEGEncoder trainable parameter count must remain below 500,000."


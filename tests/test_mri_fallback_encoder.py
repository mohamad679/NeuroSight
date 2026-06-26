from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import neurosight.models.mri as mri_module
from neurosight.models.mri import (
    MRI_EMBEDDING_DIM,
    OLD_FLATTEN_LINEAR_PARAMETER_COUNT,
    LightweightMRIEncoder,
    get_mri_transforms,
)


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_lightweight_mri_encoder_outputs_expected_embedding_shape() -> None:
    encoder = LightweightMRIEncoder().eval()
    x = torch.randn(2, 1, 96, 96, 96)

    with torch.no_grad():
        embedding = encoder(x)

    assert tuple(embedding.shape) == (2, MRI_EMBEDDING_DIM)
    assert torch.isfinite(embedding).all()


def test_lightweight_mri_encoder_parameter_count_is_reasonable() -> None:
    encoder = LightweightMRIEncoder()
    params = _parameter_count(encoder)

    assert OLD_FLATTEN_LINEAR_PARAMETER_COUNT == 679_478_016
    assert params < 250_000
    assert params < OLD_FLATTEN_LINEAR_PARAMETER_COUNT // 2_000
    assert not any(
        isinstance(module, nn.Linear) and module.in_features == 96 * 96 * 96
        for module in encoder.modules()
    )


def test_lightweight_mri_encoder_rejects_nan_or_inf_inputs() -> None:
    encoder = LightweightMRIEncoder().eval()
    x = torch.zeros(1, 1, 96, 96, 96)
    x[0, 0, 0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match="NaN or Inf"):
        encoder(x)

    x = torch.zeros(1, 1, 96, 96, 96)
    x[0, 0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="NaN or Inf"):
        encoder(x)


def test_lightweight_mri_encoder_rejects_wrong_shape_clearly() -> None:
    encoder = LightweightMRIEncoder().eval()

    with pytest.raises(ValueError, match=r"shape \(batch, 1, 96, 96, 96\)"):
        encoder(torch.randn(1, 96, 96, 96))

    with pytest.raises(ValueError, match=r"shape \(batch, 1, 96, 96, 96\)"):
        encoder(torch.randn(1, 1, 64, 64, 64))


def test_fallback_transform_resizes_and_rejects_nonfinite_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mri_module, "_MONAI_AVAILABLE", False)
    transform = get_mri_transforms(spatial_size=(96, 96, 96))
    volume = torch.randn(32, 40, 48)

    transformed = transform(volume)

    assert tuple(transformed.shape) == (1, 96, 96, 96)
    assert torch.isfinite(transformed).all()

    bad = torch.zeros(32, 40, 48)
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        transform(bad)

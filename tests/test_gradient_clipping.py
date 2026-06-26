"""Gradient clipping regression tests for training loops."""

from __future__ import annotations

import torch

from scripts.train import _run_train_epoch


class _TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(8, 64)
        self.head = torch.nn.Linear(64, 3)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.proj(inputs)
        return self.head(embedding), embedding


class _TinyFusion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(64, 3)

    def forward(
        self,
        *,
        mri: torch.Tensor | None,
        eeg: torch.Tensor | None,
        cog: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        del mri, eeg
        if cog is None:
            raise AssertionError("Test fixture expects cognitive embeddings.")
        logits = self.head(cog)
        return {"logits": logits, "probs": torch.softmax(logits, dim=-1)}


class _RecordingSGD(torch.optim.SGD):
    def __init__(self, params, calls: list[str]) -> None:  # type: ignore[no-untyped-def]
        super().__init__(params, lr=0.01)
        self.calls = calls

    def step(self, closure=None):  # type: ignore[no-untyped-def]
        self.calls.append("step")
        return super().step(closure)


def test_train_epoch_clips_gradients_before_optimizer_step(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[str] = []
    cog_model = _TinyClassifier()
    fusion_model = _TinyFusion()
    optimizer = _RecordingSGD(
        list(cog_model.parameters()) + list(fusion_model.parameters()),
        calls=calls,
    )

    def _fake_clip(parameters, max_norm: float):  # type: ignore[no-untyped-def]
        params = list(parameters)
        assert max_norm == 1.0
        assert params, "clip_grad_norm_ should receive trainable parameters."
        assert any(param.grad is not None for param in params)
        calls.append("clip")
        return torch.tensor(2.5)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _fake_clip)
    batch = {
        "cog": torch.randn(4, 8),
        "label": torch.tensor([0, 1, 2, 1]),
    }

    train_loss, grad_norm = _run_train_epoch(
        train_loader=[batch],
        mri_model=_TinyClassifier(),
        eeg_model=_TinyClassifier(),
        cog_model=cog_model,
        fusion_model=fusion_model,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert train_loss > 0.0
    assert grad_norm == 2.5
    assert calls == ["clip", "step"]

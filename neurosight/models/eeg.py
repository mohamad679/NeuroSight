"""EEG preprocessing and modeling components for NeuroSight."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn


def preprocess_eeg(raw_path: str, output_path: str) -> np.ndarray:
    """Preprocess an EDF EEG file and persist normalized epochs as `.npy`.

    Args:
        raw_path: Path to the source `.edf` file.
        output_path: Destination path for the saved preprocessed `.npy`.

    Returns:
        EEG array with shape `(epochs, channels, timepoints)`.

    Raises:
        ValueError: If EEG loading or preprocessing fails.
    """
    import mne

    try:
        raw = mne.io.read_raw_edf(raw_path, preload=True, verbose=False)
        raw.filter(1.0, 40.0, fir_design="firwin", verbose=False)
        raw.notch_filter(freqs=[50, 60], verbose=False)
        raw.set_eeg_reference("average", projection=True, verbose=False)
        raw.apply_proj()
        epochs = mne.make_fixed_length_epochs(raw, duration=4.0, overlap=0.5, verbose=False)
        data = epochs.get_data()
        data = (data - data.mean(axis=-1, keepdims=True)) / (data.std(axis=-1, keepdims=True) + 1e-8)
        np.save(output_path, data)
        return data
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Failed to process EEG file '{raw_path}': {exc}") from exc


def extract_spectral_features(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Extract logarithmic band-power spectral features for EEG epochs.

    Args:
        data: EEG tensor with shape `(n_epochs, n_channels, n_times)`.
        sfreq: Sampling frequency of EEG data.

    Returns:
        Spectral feature tensor with shape `(n_epochs, n_channels, 5)`.

    Raises:
        ValueError: If `data` does not have 3 dimensions or `sfreq` is invalid.
    """
    if data.ndim != 3:
        raise ValueError(f"EEG data must be 3D (epochs, channels, times), got shape {data.shape}.")
    if sfreq <= 0:
        raise ValueError("Sampling frequency must be positive.")

    bands: dict[str, tuple[float, float]] = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 40.0),
    }
    n_epochs, n_channels, n_times = data.shape
    freqs = np.fft.rfftfreq(n_times, 1.0 / sfreq)
    psd = np.abs(np.fft.rfft(data, axis=-1)) ** 2
    features = np.zeros((n_epochs, n_channels, len(bands)), dtype=np.float32)
    for band_idx, (fmin, fmax) in enumerate(bands.values()):
        mask = (freqs >= fmin) & (freqs <= fmax)
        features[:, :, band_idx] = psd[:, :, mask].mean(axis=-1)
    return np.log1p(features)


class EEGEncoder(nn.Module):
    """EEG encoder with temporal CNN and attention-pooled Transformer.

    Replaces the flattened Linear projection with a learnable query-based
    attention pooling to reduce from 64*1024 to 256 without an explosion
    in parameter count.
    """

    def __init__(
        self,
        n_channels: int = 19,
        d_model: int = 64,
        n_heads: int = 8,
        n_layers: int = 4,
        out_dim: int = 256,
    ) -> None:
        """Initialize EEG encoder layers.

        Args:
            n_channels: Number of EEG channels.
            d_model: Transformer hidden dimension.
            n_heads: Number of attention heads.
            n_layers: Number of Transformer encoder layers.
            out_dim: Output embedding dimension.
        """
        super().__init__()
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(n_channels, d_model, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=256,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attn_pool = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode EEG tensor to compact representation.

        Args:
            x: EEG tensor of shape `(B, 19, 1024)`.

        Returns:
            Tensor of shape `(B, 256)`.
        """
        x = self.temporal_cnn(x)
        x = x.transpose(1, 2)
        x = self.transformer(x)
        q = self.query.expand(x.size(0), -1, -1)
        pooled, _ = self.attn_pool(q, x, x)
        return self.proj(pooled.squeeze(1))


class EEGClassifier(nn.Module):
    """Unimodal EEG classifier with temperature-scaled logits."""

    def __init__(self, num_classes: int = 6) -> None:
        """Initialize EEG classifier.

        Args:
            num_classes: Number of diagnosis classes.
        """
        super().__init__()
        self.encoder = EEGEncoder()
        self.head = nn.Linear(256, num_classes)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return temperature-scaled logits and latent embedding.

        Args:
            x: EEG tensor of shape `(B, 19, 1024)`.

        Returns:
            Tuple of:
            - scaled logits tensor `(B, num_classes)`
            - embedding tensor `(B, 256)`
        """
        embedding = self.encoder(x)
        logits = self.head(embedding)
        scaled_logits = logits / self.temperature
        return scaled_logits, embedding


class EEGTrainer:
    """Trainer utility for unimodal EEG classifier workflows."""

    def __init__(
        self,
        model: EEGClassifier,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
    ) -> None:
        """Initialize trainer state.

        Args:
            model: EEG classifier model instance.
            optimizer: Optional optimizer; Adam is used by default.
            criterion: Optional loss module; cross-entropy by default.
        """
        self.model = model
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=1e-4)
        self.criterion = criterion or nn.CrossEntropyLoss()

    def train_epoch(
        self,
        dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
    ) -> float:
        """Run one training epoch.

        Args:
            dataloader: Iterable returning `(inputs, labels)` tensors.
            device: Device where computation runs.

        Returns:
            Mean loss across epoch batches.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            self.optimizer.zero_grad()
            logits, _ = self.model(x_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += float(loss.item())
            num_batches += 1
        if num_batches == 0:
            raise ValueError("Training dataloader must contain at least one batch.")
        return total_loss / float(num_batches)

    def val_epoch(
        self,
        dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
    ) -> float:
        """Run one validation epoch.

        Args:
            dataloader: Iterable returning `(inputs, labels)` tensors.
            device: Device where computation runs.

        Returns:
            Mean validation loss across batches.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits, _ = self.model(x_batch)
                loss = self.criterion(logits, y_batch)
                total_loss += float(loss.item())
                num_batches += 1
        if num_batches == 0:
            raise ValueError("Validation dataloader must contain at least one batch.")
        return total_loss / float(num_batches)

    def calibrate_temperature(
        self,
        val_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
    ) -> float:
        """Calibrate model temperature using LBFGS on validation logits.

        Args:
            val_loader: Validation batches with inputs and labels.
            device: Device where calibration runs.

        Returns:
            Calibrated scalar temperature value.
        """
        self.model.eval()
        logits_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                logits, _ = self.model(x_batch)
                logits_list.append(logits)
                labels_list.append(y_batch.to(device))
        if not logits_list:
            raise ValueError("Validation loader must contain at least one batch for calibration.")
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)

        temp_opt = torch.optim.LBFGS([self.model.temperature], lr=0.01, max_iter=50)

        def _closure() -> torch.Tensor:
            temp_opt.zero_grad()
            loss = nn.CrossEntropyLoss()(logits / self.model.temperature, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.model.temperature], max_norm=1.0)
            return loss

        temp_opt.step(_closure)
        return float(self.model.temperature.item())

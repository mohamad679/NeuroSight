from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from monai.networks.nets import ViT
    from monai.transforms import Compose, EnsureChannelFirst, Resize, ScaleIntensity, ToTensor
    _MONAI_AVAILABLE = True
except ImportError:
    _MONAI_AVAILABLE = False

MRI_INPUT_SHAPE: tuple[int, int, int, int] = (1, 96, 96, 96)
MRI_EMBEDDING_DIM = 768
OLD_FLATTEN_LINEAR_PARAMETER_COUNT = 96 * 96 * 96 * MRI_EMBEDDING_DIM + MRI_EMBEDDING_DIM


def _validate_mri_tensor(x: torch.Tensor) -> torch.Tensor:
    """Validate the MRI tensor contract before encoder inference."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"MRI input must be a torch.Tensor, got {type(x).__name__}.")
    if x.ndim != 5:
        raise ValueError(
            f"MRI input must be 5D with shape (batch, 1, 96, 96, 96), got {tuple(x.shape)}."
        )
    if tuple(x.shape[1:]) != MRI_INPUT_SHAPE:
        raise ValueError(
            f"MRI input must have shape (batch, 1, 96, 96, 96), got {tuple(x.shape)}."
        )
    if not torch.isfinite(x).all():
        raise ValueError("MRI input contains NaN or Inf values; refusing encoder inference.")
    return x.to(dtype=torch.float32)


class _ConvNormAct(nn.Module):
    """Small 3D CNN block used by the CPU-safe fallback encoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.block(x))


class LightweightMRIEncoder(nn.Module):
    """CPU-safe 3D CNN fallback that preserves the 768-d MRI embedding contract.

    This path is for demo and CI environments where MONAI is unavailable. It is
    intentionally compact and should not be interpreted as a validated clinical
    imaging model.
    """

    def __init__(self, embedding_dim: int = MRI_EMBEDDING_DIM) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.features = nn.Sequential(
            _ConvNormAct(1, 8),
            _ConvNormAct(8, 16),
            _ConvNormAct(16, 24),
            _ConvNormAct(24, 48),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(48),
            nn.Linear(48, self.embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _validate_mri_tensor(x)
        x = self.features(x)
        x = self.pool(x)
        return cast(torch.Tensor, self.projection(x))


class MRIEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        if _MONAI_AVAILABLE:
            self.vit = ViT(
                in_channels=1,
                img_size=(96, 96, 96),
                patch_size=(16, 16, 16),
                pos_embed="conv",
                classification=True,
                num_classes=MRI_EMBEDDING_DIM
            )
            self._use_vit = True
        else:
            # Compact fallback for CI/HF Spaces when MONAI is unavailable.
            self._use_vit = False
            self.fallback = LightweightMRIEncoder(MRI_EMBEDDING_DIM)

    def forward(self, x):
        x = _validate_mri_tensor(x)
        if self._use_vit:
            emb, hidden = self.vit(x)
            return emb
        return self.fallback(x)

class MRIClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.encoder = MRIEncoder()
        self.head = nn.Linear(MRI_EMBEDDING_DIM, num_classes)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    def forward(self, x):
        emb = self.encoder(x)
        logits = self.head(emb)
        scaled_logits = logits / self.temperature
        return scaled_logits, emb

class MRITrainer:
    def __init__(self, model, optimizer=None, criterion=None):
        self.model = model
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=1e-4)
        self.criterion = criterion or nn.CrossEntropyLoss()
        
    def train_epoch(self, dataloader, device):
        self.model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            self.optimizer.zero_grad()
            logits, _ = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)
        
    def val_epoch(self, dataloader, device):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                logits, _ = self.model(x)
                loss = self.criterion(logits, y)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def calibrate_temperature(self, val_loader, device):
        self.model.eval()
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                l, _ = self.model(x)
                logits_list.append(l)
                labels_list.append(y)
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)
        
        temp_opt = torch.optim.LBFGS([self.model.temperature], lr=0.01, max_iter=50)
        def eval():
            temp_opt.zero_grad()
            loss = nn.CrossEntropyLoss()(logits / self.model.temperature, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.model.temperature], max_norm=1.0)
            return loss
        temp_opt.step(eval)
        return self.model.temperature.item()

def get_mri_transforms(spatial_size=(96, 96, 96)):
    if _MONAI_AVAILABLE:
        return Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            ScaleIntensity(),
            Resize(spatial_size),
            ToTensor()
        ])
    else:
        # CPU-safe fallback: tensor conversion, finite check, intensity scaling,
        # and trilinear resize to the encoder contract.
        import numpy as _np
        class _BasicTransform:
            def __call__(self, x):
                if isinstance(x, _np.ndarray):
                    tensor = torch.as_tensor(x, dtype=torch.float32)
                elif isinstance(x, torch.Tensor):
                    tensor = x.to(dtype=torch.float32)
                else:
                    raise TypeError(f"MRI transform expects numpy array or tensor, got {type(x).__name__}.")
                if tensor.ndim != 3:
                    raise ValueError(f"MRI transform expects 3D volume (D,H,W), got {tuple(tensor.shape)}.")
                if not torch.isfinite(tensor).all():
                    raise ValueError("MRI volume contains NaN or Inf values; refusing preprocessing.")
                min_value = tensor.min()
                max_value = tensor.max()
                if torch.abs(max_value - min_value) > 1e-8:
                    tensor = (tensor - min_value) / (max_value - min_value)
                tensor = tensor.unsqueeze(0).unsqueeze(0)
                if tuple(tensor.shape[-3:]) != tuple(spatial_size):
                    tensor = F.interpolate(
                        tensor,
                        size=tuple(spatial_size),
                        mode="trilinear",
                        align_corners=False,
                    )
                return tensor.squeeze(0)
        return _BasicTransform()

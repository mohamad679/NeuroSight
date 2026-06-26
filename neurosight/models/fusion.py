import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Any

class ModalityProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
    def forward(self, x):
        return self.proj(x)


class AttentionEncoderLayer(nn.Module):
    """Transformer-like encoder layer that exposes attention weights."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        """Initialize attention and feed-forward sublayers.

        Args:
            d_model: Hidden feature dimension.
            nhead: Number of attention heads.
            dim_feedforward: Feed-forward hidden size.
            dropout: Dropout probability.

        Returns:
            None.
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one encoder layer and return attention maps.

        Args:
            x: Input token sequence of shape `(B, T, D)`.

        Returns:
            Tuple of:
            - updated token sequence `(B, T, D)`
            - attention weights `(B, H, T, T)`
        """
        attn_out, attn_weights = self.self_attn(
            x,
            x,
            x,
            need_weights=True,
            average_attn_weights=False,
        )
        x = self.norm1(x + self.dropout1(attn_out))
        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff))
        return x, attn_weights


class CrossModalAttentionFusion(nn.Module):
    def __init__(self, num_classes: int = 6, d_model: int = 512):
        super().__init__()
        self.mri_proj = ModalityProjection(768, d_model)
        self.eeg_proj = ModalityProjection(256, d_model)
        self.cog_proj = ModalityProjection(64, d_model)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.missing_mri = nn.Parameter(torch.randn(1, 1, d_model))
        self.missing_eeg = nn.Parameter(torch.randn(1, 1, d_model))
        self.missing_cog = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.encoder_layers = nn.ModuleList(
            [
                AttentionEncoderLayer(
                    d_model=d_model,
                    nhead=8,
                    dim_feedforward=2048,
                    dropout=0.1,
                )
                for _ in range(6)
            ]
        )
        
        self.head = nn.Linear(d_model, num_classes)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    @staticmethod
    def _compute_modality_attention(
        attention_weights: Optional[torch.Tensor],
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """Convert CLS-to-modality attention to normalized output artifacts.

        Args:
            attention_weights: Attention map tensor from final encoder layer with
                shape `(B, H, T, T)` where token order is `[CLS, mri, eeg, cog]`.

        Returns:
            Tuple of:
            - modality weight dictionary with `mri/eeg/cog` float values
            - attention map ndarray of shape `(3,)` with matching values
        """
        if attention_weights is None:
            uniform = np.full((3,), 1.0 / 3.0, dtype=np.float32)
            return (
                {"mri": float(uniform[0]), "eeg": float(uniform[1]), "cog": float(uniform[2])},
                uniform,
            )

        mean_attn = attention_weights.mean(dim=1)
        cls_to_modalities = mean_attn[:, 0, 1:4]
        cls_to_modalities = torch.clamp(cls_to_modalities, min=0.0)
        denom = cls_to_modalities.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        normalized = cls_to_modalities / denom
        averaged = normalized.mean(dim=0).detach().cpu().numpy().astype(np.float32)

        # Guard against numerical drift so outputs always sum to one.
        total = float(averaged.sum())
        if total <= 0.0:
            averaged = np.full((3,), 1.0 / 3.0, dtype=np.float32)
        else:
            averaged = averaged / total

        return (
            {"mri": float(averaged[0]), "eeg": float(averaged[1]), "cog": float(averaged[2])},
            averaged,
        )

    def forward(
        self,
        mri: Optional[torch.Tensor],
        eeg: Optional[torch.Tensor],
        cog: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        if mri is None and eeg is None and cog is None:
            raise ValueError("At least one modality tensor must be provided.")

        if mri is not None:
            batch_size = mri.size(0)
        elif eeg is not None:
            batch_size = eeg.size(0)
        elif cog is not None:
            batch_size = cog.size(0)
        else:
            raise ValueError("At least one modality tensor must be provided.")
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        if mri is not None:
            is_missing_mri = torch.isnan(mri).any(dim=-1)
            mri_clean = mri.clone()
            mri_clean[is_missing_mri] = 0.0
            mri_tokens = self.mri_proj(mri_clean).unsqueeze(1)
            if is_missing_mri.any():
                mri_tokens[is_missing_mri] = self.missing_mri[0]
        else:
            mri_tokens = self.missing_mri.expand(batch_size, -1, -1)

        if eeg is not None:
            is_missing_eeg = torch.isnan(eeg).any(dim=-1)
            eeg_clean = eeg.clone()
            eeg_clean[is_missing_eeg] = 0.0
            eeg_tokens = self.eeg_proj(eeg_clean).unsqueeze(1)
            if is_missing_eeg.any():
                eeg_tokens[is_missing_eeg] = self.missing_eeg[0]
        else:
            eeg_tokens = self.missing_eeg.expand(batch_size, -1, -1)

        if cog is not None:
            is_missing_cog = torch.isnan(cog).any(dim=-1)
            cog_clean = cog.clone()
            cog_clean[is_missing_cog] = 0.0
            cog_tokens = self.cog_proj(cog_clean).unsqueeze(1)
            if is_missing_cog.any():
                cog_tokens[is_missing_cog] = self.missing_cog[0]
        else:
            cog_tokens = self.missing_cog.expand(batch_size, -1, -1)
        
        # [CLS, mri, eeg, cog]
        tokens = torch.cat((cls_tokens, mri_tokens, eeg_tokens, cog_tokens), dim=1)
        fused = tokens
        final_attention: Optional[torch.Tensor] = None
        for layer in self.encoder_layers:
            fused, layer_attention = layer(fused)
            final_attention = layer_attention

        cls_out = fused[:, 0, :]
        
        logits = self.head(cls_out)
        scaled_logits = logits / self.temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        modality_weights, attention_map = self._compute_modality_attention(final_attention)
        
        return {
            "logits": scaled_logits,
            "probs": probs,
            "embedding": cls_out,
            "modality_weights": modality_weights,
            "attention_map": attention_map,
        }

class FusionTrainer:
    def __init__(self, model):
        self.model = model
        
    def train_phase_A(self, train_loader, device, epochs=10):
        for param in self.model.parameters():
            param.requires_grad = False
        trainable_modules = [self.model.mri_proj, self.model.eeg_proj, self.model.cog_proj, self.model.encoder_layers, self.model.head]
        for module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True
        self.model.cls_token.requires_grad = True
        self.model.missing_mri.requires_grad = True
        self.model.missing_eeg.requires_grad = True
        self.model.missing_cog.requires_grad = True
        
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        epoch_losses = []
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                mri, eeg, cog, labels = batch["mri"], batch["eeg"], batch["cog"], batch["label"]
                mri = mri.to(device) if mri is not None else None
                eeg = eeg.to(device) if eeg is not None else None
                cog = cog.to(device) if cog is not None else None
                labels = labels.to(device)
                
                optimizer.zero_grad()
                out = self.model(mri, eeg, cog)
                loss = criterion(out["logits"], labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                total_loss += loss.item()
            epoch_losses.append(total_loss / len(train_loader))
        return epoch_losses

    def train_phase_B(self, train_loader, device, epochs=40):
        for param in self.model.parameters():
            param.requires_grad = True
            
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
        criterion = nn.CrossEntropyLoss()
        
        epoch_losses = []
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                mri, eeg, cog, labels = batch["mri"], batch["eeg"], batch["cog"], batch["label"]
                mri = mri.to(device) if mri is not None else None
                eeg = eeg.to(device) if eeg is not None else None
                cog = cog.to(device) if cog is not None else None
                labels = labels.to(device)
                
                optimizer.zero_grad()
                out = self.model(mri, eeg, cog)
                loss = criterion(out["logits"], labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            epoch_losses.append(total_loss / len(train_loader))
        return epoch_losses

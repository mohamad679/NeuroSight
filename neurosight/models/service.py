import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch

from neurosight.contracts import Diagnosis
from neurosight.schemas.cognitive import CognitiveSchema
from neurosight.models.cognitive import CognitiveClassifier
from neurosight.models.mri import MRIClassifier
from neurosight.models.eeg import EEGClassifier
from neurosight.models.fusion import CrossModalAttentionFusion

class ModelService:
    """Manages model loading, device placement, feature preprocessing, and status metadata."""

    def __init__(self, checkpoint_path: Optional[Path] = None, device: Optional[str] = None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.num_classes = len(Diagnosis)

        # Initialize model instances
        self.mri_model = MRIClassifier(num_classes=self.num_classes).to(self.device).eval()
        self.eeg_model = EEGClassifier(num_classes=self.num_classes).to(self.device).eval()
        self.cognitive_model = CognitiveClassifier(num_classes=self.num_classes).to(self.device).eval()
        self.fusion_model = CrossModalAttentionFusion(num_classes=self.num_classes).to(self.device).eval()

        self.checkpoint_loaded = False
        self.checkpoint_error = None
        self.checkpoint_metadata = {}
        self.checkpoint_path = checkpoint_path

        # Preprocessing scaling stats computed from synthetic CN/MCI/AD datasets
        # Corresponding to: ["MMSE", "MOCA", "CDRSB", "ADAS11", "RAVLT_immediate", "RAVLT_learning", "FAQ", "AGE"]
        self.cog_means = np.array([23.2, 21.0, 1.4, 15.0, 25.0, 3.0, 8.0, 70.0], dtype=np.float32)
        self.cog_stds = np.array([4.0, 5.0, 1.5, 10.0, 15.0, 3.0, 8.0, 10.0], dtype=np.float32)

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, path: Path):
        """Loads a weights checkpoint file into the model instances."""
        self.checkpoint_path = path
        if not path.exists():
            self.checkpoint_loaded = False
            self.checkpoint_error = f"Checkpoint file not found: {path}"
            return

        try:
            checkpoint = torch.load(str(path), map_location="cpu")
            if not isinstance(checkpoint, dict):
                raise ValueError("Checkpoint must be a dictionary.")

            # Load weights
            if "mri_state" in checkpoint:
                self.mri_model.load_state_dict(checkpoint["mri_state"])
            if "eeg_state" in checkpoint:
                self.eeg_model.load_state_dict(checkpoint["eeg_state"])
            if "cog_state" in checkpoint:
                self.cognitive_model.load_state_dict(checkpoint["cog_state"])
            if "model_state" in checkpoint:
                self.fusion_model.load_state_dict(checkpoint["model_state"])

            # Load cognitive scaler state if present
            if "scaler_state" in checkpoint:
                self.cog_means = np.array(checkpoint["scaler_state"]["mean"], dtype=np.float32)
                self.cog_stds = np.array(checkpoint["scaler_state"]["std"], dtype=np.float32)

            self.checkpoint_metadata = {
                "path": str(path),
                "epoch": checkpoint.get("epoch"),
                "val_auc": checkpoint.get("val_auc"),
            }
            self.checkpoint_loaded = True
            self.checkpoint_error = None
        except Exception as exc:
            self.checkpoint_loaded = False
            self.checkpoint_error = str(exc)

    def preprocess_cognitive(self, schema: CognitiveSchema) -> torch.Tensor:
        """Preprocesses inputs, normalizing them using scaler stats."""
        feature_dict = schema.to_features_dict()
        values = np.array([
            feature_dict["MMSE"],
            feature_dict["MOCA"],
            feature_dict["CDRSB"],
            feature_dict["ADAS11"],
            feature_dict["RAVLT_immediate"],
            feature_dict["RAVLT_learning"],
            feature_dict["FAQ"],
            feature_dict["AGE"]
        ], dtype=np.float32)

        # Standardize
        normalized = (values - self.cog_means) / (self.cog_stds + 1e-8)
        return torch.tensor([normalized], dtype=torch.float32).to(self.device)

    def predict_cognitive_unimodal(self, schema: CognitiveSchema) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs unimodal cognitive prediction and returns (logits, embedding)."""
        x = self.preprocess_cognitive(schema)
        with torch.no_grad():
            logits, emb = self.cognitive_model(x)
        return logits, emb

    def get_status_metadata(self) -> Dict[str, Any]:
        """Returns standard metadata for model status reporting."""
        checkpoint_id = "none"
        if self.checkpoint_loaded:
            epoch = self.checkpoint_metadata.get("epoch")
            val_auc = self.checkpoint_metadata.get("val_auc")
            checkpoint_id = f"epoch_{epoch}_auc_{val_auc:.3f}" if epoch is not None and val_auc is not None else "custom"

        mode = "demo_untrained"
        if self.checkpoint_loaded:
            mode = "trained_checkpoint"

        return {
            "model_mode": mode,
            "checkpoint_id": checkpoint_id,
            "trained_on_real_data": False,
            "clinical_validated": False,
            "disclaimer": "This is a machine learning portfolio prototype. It is NOT a clinical product and must not be used for diagnosis or medical decision making."
        }

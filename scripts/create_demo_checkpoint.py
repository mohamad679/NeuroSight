#!/usr/bin/env python3
import os
import torch
import numpy as np
from pathlib import Path

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

def main():
    set_seed(42)
    
    from neurosight.models.cognitive import CognitiveClassifier
    from neurosight.models.mri import MRIClassifier
    from neurosight.models.eeg import EEGClassifier
    from neurosight.models.fusion import CrossModalAttentionFusion
    
    num_classes = 6
    
    mri_model = MRIClassifier(num_classes=num_classes)
    eeg_model = EEGClassifier(num_classes=num_classes)
    cognitive_model = CognitiveClassifier(num_classes=num_classes)
    fusion_model = CrossModalAttentionFusion(num_classes=num_classes)
    
    checkpoint = {
        "epoch": 0,
        "val_auc": 0.5,
        "config": {
            "seed": 42,
            "num_classes": num_classes,
        },
        "mri_state": mri_model.state_dict(),
        "eeg_state": eeg_model.state_dict(),
        "cog_state": cognitive_model.state_dict(),
        "model_state": fusion_model.state_dict(),
        "scaler_state": {
            "mean": [23.2, 21.0, 1.4, 15.0, 25.0, 3.0, 8.0, 70.0],
            "std": [4.0, 5.0, 1.5, 10.0, 15.0, 3.0, 8.0, 10.0]
        }
    }
    
    out_dir = Path("checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "best_fusion.pt"
    
    torch.save(checkpoint, str(out_path))
    print(f"Created demo checkpoint at {out_path} with seed 42.")

if __name__ == "__main__":
    main()

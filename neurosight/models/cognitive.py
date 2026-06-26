from typing import cast
import torch
import torch.nn as nn
import numpy as np

class CognitiveFeatureExtractor:
    def __init__(self):
        self.means = np.zeros(8)
        self.stds = np.ones(8)
        
    def __call__(self, scores: list) -> np.ndarray:
        arr = np.array(scores, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=self.means)
        return cast(np.ndarray, (arr - self.means) / self.stds)

class CognitiveEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(8, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )
        
    def forward(self, x):
        return self.mlp(x)

class CognitiveClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.encoder = CognitiveEncoder()
        self.head = nn.Linear(64, num_classes)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    def forward(self, x):
        emb = self.encoder(x)
        logits = self.head(emb)
        scaled_logits = logits / self.temperature
        return scaled_logits, emb

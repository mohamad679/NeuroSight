import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurosight.models.eeg import EEGClassifier
from neurosight.models.cognitive import CognitiveClassifier

def test_phase2():
    print("Running Phase 2 Test...")
    eeg_model = EEGClassifier(num_classes=6).eval()
    cog_model = CognitiveClassifier(num_classes=6).eval()
    
    batch_size = 2
    x_eeg = torch.randn(batch_size, 19, 1024)
    x_cog = torch.randn(batch_size, 8)
    
    with torch.no_grad():
        _, emb_eeg = eeg_model(x_eeg)
        _, emb_cog = cog_model(x_cog)
        
    assert list(emb_eeg.shape) == [batch_size, 256]
    assert list(emb_cog.shape) == [batch_size, 64]
    
    print("PHASE 2 COMPLETE ✓")

if __name__ == "__main__":
    test_phase2()

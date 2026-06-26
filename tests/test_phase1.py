import sys
import os
import torch
import torch.nn.functional as F

# Ensure we can import neurosight
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurosight.models.mri import MRIClassifier
from neurosight.scripts.download_data import generate_synthetic_data

def test_phase1():
    print("Running Phase 1 Test...")
    generate_synthetic_data()
    model = MRIClassifier(num_classes=6)
    model.eval()
    
    batch_size = 2
    x = torch.randn(batch_size, 1, 96, 96, 96)
    
    with torch.no_grad():
        logits, emb = model(x)
        probs = F.softmax(logits, dim=-1)
        
    assert list(logits.shape) == [batch_size, 6]
    assert list(emb.shape) == [batch_size, 768]
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size))
    print("PHASE 1 COMPLETE ✓")

if __name__ == "__main__":
    test_phase1()

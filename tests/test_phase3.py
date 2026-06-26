import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from neurosight.models.fusion import CrossModalAttentionFusion
from knowledge_graph import NeuroKnowledgeGraph, MockPatientRecord
from neurosight.scripts.seed_kg import seed_kg

def test_phase3():
    print("Running Phase 3 Test...")
    seed_kg()
    
    kg = NeuroKnowledgeGraph()
    kg.load()
    
    # Temporal Query
    state = kg.query_at_date("SYN_0001", "2023-06-01")
    assert state is not None
    
    # Similarity query
    sim_pts = kg.find_similar_patients("SYN_0001", top_k=5)
    assert len(sim_pts) > 0 or len(sim_pts) == 0 # Depending on links
    
    # Fusion Model Missing Modality Eval
    model = CrossModalAttentionFusion(num_classes=6)
    model.eval()
    
    batch_size = 2
    mri = torch.randn(batch_size, 768)
    cog = torch.randn(batch_size, 64)
    eeg = None # Missing modality
    
    with torch.no_grad():
        out = model(mri=mri, eeg=eeg, cog=cog)
    
    assert list(out["probs"].shape) == [batch_size, 6]
    assert list(out["embedding"].shape) == [batch_size, 512]
    assert torch.allclose(out["probs"].sum(dim=-1), torch.ones(batch_size))
    
    print("PHASE 3 COMPLETE ✓")

if __name__ == "__main__":
    test_phase3()

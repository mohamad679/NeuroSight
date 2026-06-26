from fastapi.testclient import TestClient
import sys
import os

os.environ.setdefault("APP_ENV", "test")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api.main import app

VALID_COGNITIVE_SCORES = {
    "MMSE": 26,
    "MOCA": 23,
    "CDRSB": 0.5,
    "ADAS11": 10.0,
    "RAVLT_immediate": 40.0,
    "RAVLT_learning": 4.0,
    "FAQ": 2.0,
    "AGE": 72,
}

def test_phase5():
    print("Running Phase 5 Test...")
    client = TestClient(app)
    
    # Test POST /v1/upload/cognitive
    res1 = client.post("/v1/upload/cognitive", json={"scores": VALID_COGNITIVE_SCORES})
    assert res1.status_code == 200
    
    # Test POST /v1/diagnose
    res2 = client.post(
        "/v1/diagnose",
        json={"query": "test", "cognitive_scores": VALID_COGNITIVE_SCORES},
    )
    assert res2.status_code == 200
    data = res2.json()
    assert "diagnosis" in data
    assert "confidence" in data
    assert "requires_review" in data
    assert "report_text" in data
    
    # Test GET /v1/eval/metrics
    res3 = client.get("/v1/eval/metrics")
    assert res3.status_code == 200
    
    print("PHASE 5 COMPLETE")
    print("NeuroSight trust/XAI contracts are ready for public demo QA.")

if __name__ == "__main__":
    test_phase5()

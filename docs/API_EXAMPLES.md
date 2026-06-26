# API Examples

Protected `/v1/*` endpoints require an API key outside test mode:

```bash
export NEUROSIGHT_API_KEY=change-me-in-production
```

Uploads default to a 100 MB request-body limit. For larger local experiments, set
`NEUROSIGHT_MAX_UPLOAD_BYTES` on the API process.

Current API risk profiling responses run in demo/untrained mode unless you add a
validated checkpoint-loading path. Treat returned labels and confidence values as
non-clinical research demo output, not clinical evidence.

Preferred naming uses `risk-profile`. The older `/v1/diagnose` routes remain
available only for backward compatibility and are legacy/deprecated API names.

## Demo Readiness
```bash
curl http://localhost:8000/v1/demo/readiness \
  -H "X-API-Key: ${NEUROSIGHT_API_KEY}"
```

## Quick Risk Profiling (Preferred)
```bash
curl -X POST http://localhost:8000/v1/risk-profile \
  -H "X-API-Key: ${NEUROSIGHT_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"cognitive_scores": {"MMSE": 20, "MOCA": 16, "CDRSB": 2.0,
        "ADAS11": 24.0, "RAVLT_immediate": 28.0,
        "RAVLT_learning": 1.0, "FAQ": 8.0, "AGE": 76}}'
```

Legacy route retained for backward compatibility:

```bash
curl -X POST http://localhost:8000/v1/diagnose \
  -H "X-API-Key: ${NEUROSIGHT_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"cognitive_scores": {"MMSE": 20, "MOCA": 16, "CDRSB": 2.0,
        "ADAS11": 24.0, "RAVLT_immediate": 28.0,
        "RAVLT_learning": 1.0, "FAQ": 8.0, "AGE": 76}}'
```

## MRI Upload
```bash
# Generate test MRI
python -c "import numpy as np; np.save('/tmp/test_mri.npy', np.random.randn(96,96,96))"

curl -X POST http://localhost:8000/v1/upload/mri \
  -H "X-API-Key: ${NEUROSIGHT_API_KEY}" \
  -F "file=@/tmp/test_mri.npy"
```

## Streaming Risk Profiling
```python
import httpx

with httpx.stream(
    "POST",
    "http://localhost:8000/v1/risk-profile/stream",
    headers={"X-API-Key": "change-me-in-production"},
    json={
        "cognitive_scores": {
            "MMSE": 20,
            "MOCA": 16,
            "CDRSB": 2.0,
            "ADAS11": 24.0,
            "RAVLT_immediate": 28.0,
            "RAVLT_learning": 1.0,
            "FAQ": 8.0,
            "AGE": 76,
        }
    },
) as response:
    for line in response.iter_lines():
        if line.startswith("data:") and line != "data: [DONE]":
            print(line[6:])
```

## Knowledge Graph Query
```bash
# Seed demo data first
python neurosight/scripts/seed_kg.py

# Query patient history
curl -X POST http://localhost:8000/v1/kg/query \
  -H "X-API-Key: ${NEUROSIGHT_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "SYN_0001", "query_type": "history"}'
```

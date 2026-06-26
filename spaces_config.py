"""HuggingFace Spaces compatibility configuration.

Disables MONAI ViT warm-up (too slow on CPU), uses pre-computed
embeddings for MRI demo, and limits batch sizes.
"""

from __future__ import annotations

import os

SPACES_MODE: bool = os.environ.get("SPACE_ID") is not None
DISABLE_MRI_WARMUP: bool = (
    SPACES_MODE
    or os.environ.get("APP_ENV", "").strip().lower() == "test"
    or os.environ.get("DISABLE_MRI_WARMUP", "0") == "1"
)
MAX_DEMO_PATIENTS: int = int(os.environ.get("MAX_DEMO_PATIENTS", "50"))

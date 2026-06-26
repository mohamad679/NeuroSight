import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosight.models.cognitive import CognitiveClassifier
from neurosight.models.eeg import EEGClassifier
from neurosight.models.mri import MRIClassifier
from neurosight.models.xai import AttentionRollout, SHAPExplainer, XAIEngine


def test_gradcamplusplus_on_mri_classifier() -> None:
    """GradCAM++ should return a non-empty saliency map for MRI classifier."""
    model = MRIClassifier(num_classes=6).eval()
    engine = XAIEngine(mri_model=model)
    input_tensor = torch.randn(1, 1, 96, 96, 96, dtype=torch.float32)

    explanation = engine.explain_mri(input_tensor, target_class=0)
    saliency = explanation.saliency

    assert isinstance(saliency, np.ndarray), "GradCAM saliency must be a NumPy array."
    assert saliency.size > 0, "GradCAM saliency map should not be empty."
    assert saliency.ndim >= 1, "GradCAM saliency must have at least 1 dimension."


def test_attention_rollout_on_eeg_classifier() -> None:
    """AttentionRollout should produce non-empty EEG importance output."""
    eeg_model = EEGClassifier(num_classes=6).eval()
    rollout = AttentionRollout(eeg_model)
    eeg_tensor = torch.randn(1, 19, 1024, dtype=torch.float32)

    importance = rollout(eeg_tensor)

    assert isinstance(importance, np.ndarray), "Rollout output must be a NumPy array."
    assert importance.size > 0, "Rollout output should not be empty."
    assert np.isfinite(importance).all(), "Rollout output should contain finite values."


def test_shap_fallback_gradient_x_input_returns_8_features() -> None:
    """Gradient×input fallback should output all 8 named cognitive features."""
    cognitive_model = CognitiveClassifier(num_classes=6).eval()
    engine = XAIEngine(cognitive_model=cognitive_model)
    cog_tensor = torch.randn(1, 8, dtype=torch.float32)

    explanation = engine.explain_cognitive(cog_tensor, cognitive_model=cognitive_model)
    saliency = explanation.saliency

    assert isinstance(saliency, dict), "Cognitive explanation saliency should be a dictionary."
    assert set(saliency.keys()) == set(SHAPExplainer.FEATURE_NAMES), (
        "Fallback saliency should include exactly the expected 8 cognitive features."
    )
    assert all(isinstance(v, float) for v in saliency.values()), "All feature scores must be floats."


def test_generate_nl_explanation_without_llm_uses_modality_weights() -> None:
    """NL explanation fallback should be non-empty and mention dominant modality evidence."""
    engine = XAIEngine(llm_client=None)
    report = SimpleNamespace(
        final_diagnosis="ad",
        confidence=0.62,
        requires_review=True,
        modality_weights={"mri": 0.75, "eeg": 0.15, "cog": 0.10},
    )

    text = engine.generate_nl_explanation(report, explanations=[], llm_client=None)

    assert isinstance(text, str) and text.strip(), "Generated NL explanation should be non-empty."
    assert "Primary evidence" in text, "Fallback explanation should mention primary evidence."
    assert "structural MRI abnormalities" in text, "Dominant MRI modality should be reflected in text."

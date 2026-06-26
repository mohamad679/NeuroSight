"""NeuroSight — Gradio web interface for multimodal synthetic risk profiling demo.

Launch locally:
    python app.py

Deploy to HuggingFace Spaces:
    gradio deploy
"""

from __future__ import annotations

import os
import socket
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

_MPLCONFIG_DIR = Path(tempfile.gettempdir()) / "neurosight_matplotlib"
_MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIG_DIR))

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

import spaces_config
from evaluation.calibration import CalibrationAnalyzer
from evaluation.metrics import compute_ece, compute_modality_ablation
from knowledge_graph import MockPatientRecord, NeuroKnowledgeGraph
from neurosight.agents.orchestrator import build_diagnosis_graph, run_diagnosis
from neurosight.contracts import Diagnosis
from neurosight.schemas.cognitive import CognitiveSchema
from neurosight.models.cognitive import CognitiveClassifier, CognitiveEncoder
from neurosight.models.eeg import EEGClassifier, preprocess_eeg
from neurosight.models.fusion import CrossModalAttentionFusion
from neurosight.models.mri import MRIClassifier, get_mri_transforms
from neurosight.models.xai import XAIEngine
from neurosight.scripts.seed_kg import seed_kg

try:
    import nibabel as nib
except ModuleNotFoundError:
    nib = None

_MRI_MODEL: Optional[MRIClassifier] = None
_EEG_MODEL: Optional[EEGClassifier] = None
_FUSION_MODEL: Optional[CrossModalAttentionFusion] = None
_COG_MODEL: Optional[CognitiveClassifier] = None
_KG: Optional[NeuroKnowledgeGraph] = None

_MODEL_LOCK = threading.Lock()
_KG_LOCK = threading.Lock()
_CLASS_ORDER: list[Diagnosis] = list(Diagnosis)
_DEFAULT_SEED = 42
DEMO_MODEL_NOTICE = (
    "Demo mode: this interface uses freshly initialized model weights and "
    "synthetic/demo scaffolding. Outputs are not clinically meaningful."
)


_MODEL_SERVICE: Optional[ModelService] = None


def _load_models() -> None:
    """Load NeuroSight model instances once at module level using ModelService."""
    global _MODEL_SERVICE, _MRI_MODEL, _EEG_MODEL, _FUSION_MODEL, _COG_MODEL
    with _MODEL_LOCK:
        if _MODEL_SERVICE is None:
            from neurosight.models.service import ModelService
            checkpoint_path = Path("checkpoints/best_fusion.pt")
            is_test = os.environ.get("APP_ENV", "").strip().lower() == "test"
            _MODEL_SERVICE = ModelService(checkpoint_path=checkpoint_path if (checkpoint_path.exists() and not is_test) else None)
            
            _MRI_MODEL = _MODEL_SERVICE.mri_model
            _EEG_MODEL = _MODEL_SERVICE.eeg_model
            _COG_MODEL = _MODEL_SERVICE.cognitive_model
            _FUSION_MODEL = _MODEL_SERVICE.fusion_model

            # Warm up models on correct device
            with torch.no_grad():
                if not spaces_config.DISABLE_MRI_WARMUP and _MRI_MODEL is not None:
                    _MRI_MODEL(torch.randn(1, 1, 96, 96, 96).to(_MODEL_SERVICE.device))
                if _EEG_MODEL is not None:
                    _EEG_MODEL(torch.randn(1, 19, 1024).to(_MODEL_SERVICE.device))
                if _COG_MODEL is not None:
                    _COG_MODEL(torch.randn(1, 8).to(_MODEL_SERVICE.device))


def _get_status_notice() -> str:
    """Retrieve model status notice dynamically based on the current model service state."""
    if _MODEL_SERVICE is None:
        return DEMO_MODEL_NOTICE
    meta = _MODEL_SERVICE.get_status_metadata()
    if meta["model_mode"] == "trained_checkpoint":
        return (
            f"Checkpoint Mode: A trained model checkpoint has been loaded. "
            f"Checkpoint ID: {meta['checkpoint_id']}. "
            f"Disclaimer: {meta['disclaimer']}"
        )
    else:
        return (
            f"Demo Mode: This interface uses freshly initialized model weights. "
            f"Disclaimer: {meta['disclaimer']}"
        )


def _load_kg() -> NeuroKnowledgeGraph:
    """Load or initialize knowledge graph instance.

    Returns:
        Shared knowledge graph instance.
    """
    global _KG
    with _KG_LOCK:
        if _KG is None:
            _KG = NeuroKnowledgeGraph()
            kg_path = Path("data/neurosight_kg.json")
            if kg_path.exists():
                _KG.load(str(kg_path))
        return _KG


def _load_mri_embedding(mri_file: Optional[str]) -> Optional[torch.Tensor]:
    """Load MRI file and return encoder embedding.

    Args:
        mri_file: Path to MRI file (`.npy`, `.nii`, `.nii.gz`) or `None`.

    Returns:
        Optional MRI embedding tensor with shape `(1, 768)`.
    """
    if mri_file is None:
        return None
    if _MRI_MODEL is None:
        raise RuntimeError("MRI model is not loaded.")

    path = Path(mri_file)
    suffix_name = path.name.lower()

    if suffix_name.endswith(".npy"):
        mri_array = np.load(path, allow_pickle=False).astype(np.float32)
    elif suffix_name.endswith(".nii") or suffix_name.endswith(".nii.gz"):
        if nib is None:
            raise ModuleNotFoundError("nibabel is required to load NIfTI files.")
        mri_array = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32), dtype=np.float32)
    else:
        raise ValueError("MRI upload must be .npy, .nii, or .nii.gz.")

    if mri_array.ndim == 4 and mri_array.shape[0] == 1:
        mri_array = np.squeeze(mri_array, axis=0)
    if mri_array.ndim == 4 and mri_array.shape[-1] == 1:
        mri_array = np.squeeze(mri_array, axis=-1)
    if mri_array.ndim != 3:
        raise ValueError(f"Invalid MRI shape {mri_array.shape}; expected 3D volume.")

    transforms = get_mri_transforms()
    transformed = transforms(mri_array).unsqueeze(0).to(dtype=torch.float32, device=_MODEL_SERVICE.device)
    with torch.no_grad():
        embedding = _MRI_MODEL.encoder(transformed)
    return embedding


def _load_eeg_embedding(eeg_file: Optional[str]) -> Optional[torch.Tensor]:
    """Load EEG file and return encoder embedding.

    Args:
        eeg_file: Path to EEG file (`.npy` or `.edf`) or `None`.

    Returns:
        Optional EEG embedding tensor with shape `(1, 256)`.
    """
    if eeg_file is None:
        return None
    if _EEG_MODEL is None:
        raise RuntimeError("EEG model is not loaded.")

    path = Path(eeg_file)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        eeg_array = np.load(path, allow_pickle=False).astype(np.float32)
    elif suffix == ".edf":
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as out_file:
            out_path = out_file.name
        try:
            eeg_array = preprocess_eeg(str(path), out_path)
        finally:
            try:
                os.remove(out_path)
            except OSError as remove_error:
                ignored_cleanup_error = remove_error
                del ignored_cleanup_error
    else:
        raise ValueError("EEG upload must be .npy or .edf.")

    if eeg_array.ndim == 3:
        eeg_array = eeg_array.mean(axis=0)
    if eeg_array.ndim != 2:
        raise ValueError(f"Invalid EEG shape {eeg_array.shape}; expected channels x time.")
    if eeg_array.shape[0] != 19 and eeg_array.shape[1] == 19:
        eeg_array = eeg_array.T
    if eeg_array.shape[0] != 19:
        raise ValueError("EEG input must have 19 channels.")

    n_time = eeg_array.shape[1]
    if n_time < 1024:
        eeg_array = np.pad(eeg_array, ((0, 0), (0, 1024 - n_time)), mode="constant")
    elif n_time > 1024:
        eeg_array = eeg_array[:, :1024]

    eeg_tensor = torch.tensor(eeg_array, dtype=torch.float32).unsqueeze(0).to(device=_MODEL_SERVICE.device)
    with torch.no_grad():
        embedding = _EEG_MODEL.encoder(eeg_tensor)
    return embedding


def _plot_feature_importance(importance_dict: dict[str, float]) -> Figure:
    """Plot horizontal feature-importance chart.

    Args:
        importance_dict: Cognitive feature importance values.

    Returns:
        Matplotlib figure object.
    """
    names = list(importance_dict.keys())
    values = [float(importance_dict[name]) for name in names]
    colors = ["#4F46E5" if abs(value) > 0.05 else "#94A3B8" for value in values]

    figure, axis = plt.subplots(figsize=(6, 4))
    positions = np.arange(len(names))
    axis.barh(positions, values, color=colors)
    axis.set_yticks(positions, labels=names)
    axis.set_xlabel("Importance Score")
    axis.set_title("Cognitive Feature Importance (gradient × input)")
    axis.invert_yaxis()
    axis.grid(False)
    axis.set_facecolor("white")
    figure.tight_layout()
    return figure


def _plot_modality_weights(weights_dict: dict[str, float]) -> Figure:
    """Plot horizontal modality-weight bars.

    Args:
        weights_dict: Modality contribution weights.

    Returns:
        Matplotlib figure object.
    """
    labels = ["MRI", "EEG", "Cognitive"]
    values = [
        float(weights_dict.get("mri", 0.0)),
        float(weights_dict.get("eeg", 0.0)),
        float(weights_dict.get("cog", weights_dict.get("cognitive", 0.0))),
    ]
    colors = ["#EF4444", "#3B82F6", "#10B981"]

    figure, axis = plt.subplots(figsize=(6, 2.8))
    positions = np.arange(len(labels))
    axis.barh(positions, values, color=colors)
    axis.set_yticks(positions, labels=labels)
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Attention Weight")
    axis.set_title("Cross-Modal Attention Weights")
    figure.tight_layout()
    return figure


def _make_diagnosis_badge(diagnosis_label: str) -> str:
    """Render risk profile badge HTML.

    Args:
        diagnosis_label: Predicted risk profile class.

    Returns:
        HTML string for diagnosis badge.
    """
    normalized = diagnosis_label.lower()
    if normalized == "normal":
        color = "#10B981"
    elif normalized == "mci":
        color = "#F59E0B"
    else:
        color = "#EF4444"
    return (
        "<div style='padding:12px;border-radius:10px;"
        f"background:{color};color:white;font-size:26px;font-weight:700;text-align:center;'>"
        f"{diagnosis_label.upper()}</div>"
    )


def _make_confidence_meter(confidence: float) -> str:
    """Render confidence meter HTML.

    Args:
        confidence: Confidence value in [0, 1].

    Returns:
        HTML string with visual confidence meter.
    """
    clipped = float(max(0.0, min(1.0, confidence)))
    percent = clipped * 100.0
    if percent < 50.0:
        color = "#EF4444"
    elif percent < 75.0:
        color = "#F59E0B"
    else:
        color = "#10B981"
    return (
        "<div>"
        f"<div style='font-weight:600;margin-bottom:6px;'>Confidence: {percent:.1f}%</div>"
        "<div style='background:#1F2937;border-radius:8px;height:16px;width:100%;overflow:hidden;'>"
        f"<div style='background:{color};width:{percent:.1f}%;height:16px;'></div>"
        "</div></div>"
    )


def run_diagnosis_ui(
    mmse: float,
    moca: float,
    cdrsb: float,
    adas11: float,
    ravlt_immediate: float,
    ravlt_learning: float,
    faq: float,
    age: float,
    mri_file: Optional[str],
    eeg_file: Optional[str],
    query: str,
) -> tuple[str, float, bool, dict[str, float], dict[str, float], str, object]:
    """Run multimodal risk profiling for Gradio interface inputs.

    Args:
        mmse: MMSE score.
        moca: MoCA score.
        cdrsb: CDRSB score.
        adas11: ADAS-11 score.
        ravlt_immediate: RAVLT Immediate score.
        ravlt_learning: RAVLT Learning score.
        faq: FAQ score.
        age: Patient age.
        mri_file: Optional MRI filepath from uploader.
        eeg_file: Optional EEG filepath from uploader.
        query: Optional research query.

    Returns:
        Tuple containing diagnosis label, confidence, review flag,
        modality weights, feature importance, report text, and raw attention map.
    """
    _load_models()

    if _COG_MODEL is None or _FUSION_MODEL is None or _MODEL_SERVICE is None:
        raise RuntimeError("Core models are unavailable for diagnosis.")

    schema = CognitiveSchema(
        MMSE=mmse,
        MOCA=moca,
        CDRSB=cdrsb,
        ADAS11=adas11,
        RAVLT_immediate=ravlt_immediate,
        RAVLT_learning=ravlt_learning,
        FAQ=faq,
        AGE=age,
    )
    cognitive_tensor = _MODEL_SERVICE.preprocess_cognitive(schema)

    with torch.no_grad():
        _, cognitive_embedding = _COG_MODEL(cognitive_tensor)

    mri_embedding = _load_mri_embedding(mri_file)
    eeg_embedding = _load_eeg_embedding(eeg_file)

    with torch.no_grad():
        fusion_out = _FUSION_MODEL(mri=mri_embedding, eeg=eeg_embedding, cog=cognitive_embedding)
    probabilities = fusion_out["probs"][0].detach().cpu().numpy()
    pred_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[pred_idx])
    diagnosis_label = _CLASS_ORDER[pred_idx].value
    requires_review = True
    modality_weights = {
        "mri": float(fusion_out["modality_weights"]["mri"]),
        "eeg": float(fusion_out["modality_weights"]["eeg"]),
        "cog": float(fusion_out["modality_weights"]["cog"]),
    }

    xai_engine = XAIEngine(cognitive_model=_COG_MODEL)
    cognitive_explanation = xai_engine.explain_cognitive(
        cognitive_tensor,
        target_class=pred_idx,
        cognitive_model=_COG_MODEL,
    )
    feature_importance = {
        str(key): float(value)
        for key, value in dict(cognitive_explanation.saliency).items()
    }

    patient = MockPatientRecord(patient_id="UI_DEMO_PATIENT")
    patient.age = float(age)
    patient.cognitive = {
        "MMSE": float(mmse),
        "MOCA": float(moca),
        "CDRSB": float(cdrsb),
        "ADAS11": float(adas11),
        "RAVLT_immediate": float(ravlt_immediate),
        "RAVLT_learning": float(ravlt_learning),
        "FAQ": float(faq),
        "AGE": float(age),
    }
    graph = build_diagnosis_graph(None, _load_kg())
    report = run_diagnosis(patient, query or "What should the research/demo workflow inspect?", graph)

    return (
        diagnosis_label,
        confidence,
        requires_review,
        modality_weights,
        feature_importance,
        str(report.report_text),
        fusion_out["attention_map"],
    )


def _run_diagnosis_and_render(
    mmse: float,
    moca: float,
    cdrsb: float,
    adas11: float,
    ravlt_immediate: float,
    ravlt_learning: float,
    faq: float,
    age: float,
    mri_file: Optional[str],
    eeg_file: Optional[str],
    query: str,
) -> tuple[str, str, str, Figure, Figure, str]:
    """Run evaluation and map outputs into Gradio-renderable UI artifacts."""
    try:
        diagnosis, confidence, requires_review, modality_weights, feature_importance, report_text, _ = run_diagnosis_ui(
            mmse=mmse,
            moca=moca,
            cdrsb=cdrsb,
            adas11=adas11,
            ravlt_immediate=ravlt_immediate,
            ravlt_learning=ravlt_learning,
            faq=faq,
            age=age,
            mri_file=mri_file,
            eeg_file=eeg_file,
            query=query,
        )
        badge = _make_diagnosis_badge(diagnosis)
        meter = _make_confidence_meter(confidence)
        warning = f"Human review required. {_get_status_notice()}"
        modality_plot = _plot_modality_weights(modality_weights)
        feature_plot = _plot_feature_importance(feature_importance)
        report_md = f"### Research Output Summary\n\n> {_get_status_notice()}\n\n{report_text}"
        return badge, meter, warning, modality_plot, feature_plot, report_md
    except (
        FileNotFoundError,
        ModuleNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        error_badge = _make_diagnosis_badge("error")
        error_meter = _make_confidence_meter(0.0)
        warning = "⚠️ Human review recommended"
        error_plot = _plot_modality_weights({"mri": 0.0, "eeg": 0.0, "cog": 1.0})
        error_feature_plot = _plot_feature_importance(
            {
                "MMSE": 0.0,
                "MOCA": 0.0,
                "CDRSB": 0.0,
                "ADAS11": 0.0,
                "RAVLT_immediate": 0.0,
                "RAVLT_learning": 0.0,
                "FAQ": 0.0,
                "AGE": 0.0,
            }
        )
        report_md = f"### Research Output Summary\n\n> {_get_status_notice()}\n\nDemo inference execution failed: `{exc}`"
        return error_badge, error_meter, warning, error_plot, error_feature_plot, report_md


def _build_ablation_samples(n_per_class: int = 12, seed: int = 42) -> list[dict[str, Any]]:
    """Build balanced multimodal sample list for ablation analysis.

    Args:
        n_per_class: Number of synthetic examples per class.
        seed: Random seed.

    Returns:
        List of dicts compatible with `compute_modality_ablation`.
    """
    if _COG_MODEL is None or _MODEL_SERVICE is None:
        raise RuntimeError("Cognitive model or service is not loaded.")

    rng = np.random.default_rng(seed)
    samples: list[dict[str, Any]] = []
    with torch.no_grad():
        for class_index, diagnosis in enumerate(_CLASS_ORDER):
            for _ in range(n_per_class):
                if diagnosis == Diagnosis.NORMAL:
                    cog_values = [29, 28, 0.0, 8.0, 50.0, 6.0, 0.0, 65.0]
                elif diagnosis == Diagnosis.MCI:
                    cog_values = [26, 23, 0.5, 15.0, 35.0, 2.0, 2.0, 72.0]
                elif diagnosis == Diagnosis.AD:
                    cog_values = [18, 14, 2.0, 28.0, 20.0, -2.0, 8.0, 78.0]
                elif diagnosis == Diagnosis.FTD:
                    cog_values = [22, 19, 1.5, 22.0, 25.0, 1.0, 6.0, 68.0]
                elif diagnosis == Diagnosis.LBD:
                    cog_values = [21, 18, 1.5, 24.0, 24.0, 1.0, 7.0, 74.0]
                else:
                    cog_values = [23, 20, 1.5, 21.0, 26.0, 1.5, 5.0, 73.0]

                noise = rng.normal(0.0, 0.5, size=8).astype(np.float32)
                val = np.array(cog_values, dtype=np.float32) + noise
                val[0] = np.clip(val[0], 0.0, 30.0)
                val[1] = np.clip(val[1], 0.0, 30.0)
                val[2] = np.clip(val[2], 0.0, 18.0)
                val[3] = np.clip(val[3], 0.0, 70.0)
                val[4] = np.clip(val[4], 0.0, 75.0)
                val[5] = np.clip(val[5], -15.0, 15.0)
                val[6] = np.clip(val[6], 0.0, 30.0)
                val[7] = np.clip(val[7], 0.0, 120.0)

                schema = CognitiveSchema(
                    MMSE=float(val[0]),
                    MOCA=float(val[1]),
                    CDRSB=float(val[2]),
                    ADAS11=float(val[3]),
                    RAVLT_immediate=float(val[4]),
                    RAVLT_learning=float(val[5]),
                    FAQ=float(val[6]),
                    AGE=float(val[7]),
                )
                cognitive_tensor = _MODEL_SERVICE.preprocess_cognitive(schema)
                _, cog_embedding = _COG_MODEL(cognitive_tensor)

                mri_embedding = torch.randn(1, 768, generator=torch.Generator().manual_seed(seed + class_index)).to(_MODEL_SERVICE.device)
                eeg_embedding = torch.randn(1, 256, generator=torch.Generator().manual_seed(seed + class_index + 100)).to(_MODEL_SERVICE.device)
                samples.append(
                    {
                        "mri": mri_embedding,
                        "eeg": eeg_embedding,
                        "cog": cog_embedding,
                        "label": int(class_index),
                    }
                )
    return samples


def run_ablation_ui() -> tuple[pd.DataFrame, Figure]:
    """Run modality ablation on synthetic balanced data.

    Returns:
        Tuple of ablation dataframe and horizontal bar figure.
    """
    _load_models()
    if _FUSION_MODEL is None:
        raise RuntimeError("Fusion model is not loaded.")

    samples = _build_ablation_samples()
    ablation = compute_modality_ablation(_FUSION_MODEL, samples)
    full_auc = float(ablation.get("all", 0.0))
    rows = [
        {"Configuration": "All modalities", "AUC (macro)": full_auc, "Δ vs Full": "baseline"},
        {"Configuration": "No MRI", "AUC (macro)": float(ablation.get("no_mri", 0.0)),
         "Δ vs Full": f"{float(ablation.get('no_mri', 0.0)) - full_auc:+.4f}"},
        {"Configuration": "No EEG", "AUC (macro)": float(ablation.get("no_eeg", 0.0)),
         "Δ vs Full": f"{float(ablation.get('no_eeg', 0.0)) - full_auc:+.4f}"},
        {"Configuration": "No Cognitive", "AUC (macro)": float(ablation.get("no_cognitive", 0.0)),
         "Δ vs Full": f"{float(ablation.get('no_cognitive', 0.0)) - full_auc:+.4f}"},
    ]
    dataframe = pd.DataFrame(rows)

    figure, axis = plt.subplots(figsize=(7, 3.6))
    axis.barh(
        dataframe["Configuration"],
        dataframe["AUC (macro)"],
        color=["#4F46E5", "#EF4444", "#3B82F6", "#10B981"],
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Macro AUC")
    axis.set_title("Modality Ablation Performance")
    figure.tight_layout()
    return dataframe, figure


def run_calibration_ui() -> tuple[float, Figure]:
    """Run calibration analysis. Returns ECE and reliability diagram figure."""
    _load_models()
    if _FUSION_MODEL is None:
        raise RuntimeError("Fusion model is not loaded.")

    samples = _build_ablation_samples(n_per_class=10, seed=73)
    all_probs: list[np.ndarray] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for sample in samples:
            output = _FUSION_MODEL(mri=sample["mri"], eeg=sample["eeg"], cog=sample["cog"])
            all_probs.append(output["probs"][0].detach().cpu().numpy())
            all_labels.append(int(sample["label"]))

    y_prob = np.array(all_probs, dtype=np.float64)
    y_true = np.array(all_labels, dtype=np.int64)
    ece_value = float(compute_ece(y_true, y_prob))

    analyzer = CalibrationAnalyzer(y_true=y_true, y_prob=y_prob, n_bins=10)
    temp_file = Path(tempfile.gettempdir()) / "neurosight_reliability.png"
    analyzer.reliability_diagram(str(temp_file))
    image_array = plt.imread(temp_file)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.imshow(image_array)
    axis.axis("off")
    axis.set_title("Reliability Diagram")
    figure.tight_layout()
    return ece_value, figure


def query_kg_ui(patient_id: str, query_type: str, target_date: str) -> pd.DataFrame:
    """Query knowledge graph and return tabular results.

    Args:
        patient_id: Patient ID to query.
        query_type: One of `history`, `similar`, `snapshot`, `progression`.
        target_date: Optional query date for snapshot.

    Returns:
        DataFrame of query results.
    """
    kg = _load_kg()
    normalized_type = query_type.strip().lower()
    patient_key = patient_id.strip()
    rows: list[dict[str, Any]] = []

    if not patient_key:
        return pd.DataFrame(columns=["patient_id", "query_type", "result"])

    if normalized_type == "history":
        history = kg.get_patient_history(patient_key)
        for item in history:
            rows.append(
                {
                    "patient_id": patient_key,
                    "query_type": "history",
                    "result": str(item),
                }
            )
    elif normalized_type == "similar":
        similar = kg.find_similar_patients(patient_key, top_k=5)
        for item in similar:
            rows.append(
                {
                    "patient_id": patient_key,
                    "query_type": "similar",
                    "result": f"{item.patient_id} | score={item.similarity_score:.3f}",
                }
            )
    elif normalized_type == "snapshot":
        if target_date.strip():
            snapshot = kg.query_at_date(patient_key, target_date.strip())
            rows.append(
                {
                    "patient_id": patient_key,
                    "query_type": "snapshot",
                    "result": str(snapshot),
                }
            )
    elif normalized_type == "progression":
        progression = kg.get_disease_progression(patient_key)
        for item in progression:
            rows.append(
                {
                    "patient_id": patient_key,
                    "query_type": "progression",
                    "result": str(item),
                }
            )
    else:
        rows.append(
            {
                "patient_id": patient_key,
                "query_type": normalized_type,
                "result": "Unsupported query type",
            }
        )
    return pd.DataFrame(rows)


def seed_demo_kg_ui() -> str:
    """Run seed_kg.seed_kg() and return status message."""
    seed_kg()
    with _KG_LOCK:
        global _KG
        _KG = None
    kg = _load_kg()
    nodes = len(kg.graph.nodes)
    edges = len(kg.graph.edges)
    return f"Seeded demo KG successfully. Nodes={nodes}, Edges={edges}."


def _kg_stats_payload() -> dict[str, int]:
    """Get knowledge graph node/edge stats for UI JSON panel."""
    kg = _load_kg()
    return {"nodes": int(len(kg.graph.nodes)), "edges": int(len(kg.graph.edges))}


def _kg_stats_text() -> str:
    """Render KG stats in a Gradio-safe text format."""
    stats = _kg_stats_payload()
    return f"nodes: {stats['nodes']}\nedges: {stats['edges']}"


def build_interface() -> gr.Blocks:
    """Build Gradio Blocks application for NeuroSight synthetic evaluation and risk profiling demo.

    Returns:
        Configured Gradio Blocks app.
    """
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="cyan",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    ).set(
        body_background_fill="#0F172A",
        body_text_color="#E2E8F0",
        block_background_fill="#111827",
        block_border_color="#334155",
        button_primary_background_fill="#4F46E5",
        button_primary_text_color="#FFFFFF",
        button_secondary_background_fill="#06B6D4",
        button_secondary_text_color="#082F49",
    )
    css = """
    .gradio-container {background:#0F172A !important; color:#E2E8F0 !important;}
    .ns-header {display:flex;justify-content:space-between;align-items:center;padding:12px 16px;
                border:1px solid #334155;border-radius:12px;background:#111827;margin-bottom:12px;}
    .ns-logo {font-size:30px;font-weight:800;color:#E2E8F0;}
    .ns-badge {font-size:14px;font-weight:600;color:#93C5FD;background:#1E293B;padding:6px 10px;border-radius:999px;}
    .ns-footer {margin-top:16px;font-size:13px;color:#FCA5A5;font-style:italic;}
    """

    with gr.Blocks(theme=theme, css=css, title="NeuroSight") as demo:
        gr.HTML(
            "<div class='ns-header'>"
            "<div class='ns-logo'>🧠 NeuroSight</div>"
            "<div class='ns-badge'>v0.2.0 · demo weights</div>"
            "</div>"
        )

        with gr.Tabs():
            with gr.Tab("Synthetic Risk Profiling"):
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.Markdown("## Research/demo risk profiling")
                        gr.Markdown("### Cognitive Scores")
                        mmse = gr.Slider(0, 30, value=26, step=1, label="MMSE (0–30)")
                        moca = gr.Slider(0, 30, value=24, step=1, label="MoCA (0–30)")
                        cdrsb = gr.Slider(0.0, 18.0, value=0.5, step=0.5, label="CDRSB (0–18)")
                        adas11 = gr.Slider(0.0, 70.0, value=10.0, step=0.5, label="ADAS11 (0–70)")
                        ravlt_immediate = gr.Slider(0.0, 75.0, value=40.0, step=1.0, label="RAVLT Immediate (0–75)")
                        ravlt_learning = gr.Slider(-15.0, 15.0, value=4.0, step=0.5, label="RAVLT Learning (-15–15)")
                        faq = gr.Slider(0.0, 30.0, value=2.0, step=1.0, label="FAQ (0–30)")
                        age = gr.Slider(0.0, 120.0, value=70.0, step=1.0, label="Age (0–120)")

                        gr.Markdown("### Optional Modalities")
                        mri_file = gr.File(label="MRI Upload (.npy / .nii / .nii.gz)", type="filepath")
                        eeg_file = gr.File(label="EEG Upload (.npy / .edf)", type="filepath")

                        gr.Markdown("### Research/demo query (optional)")
                        query = gr.Textbox(
                            value="",
                            placeholder="What is the model-generated risk profile?",
                            label="Query",
                        )

                        analyze_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                        gr.Examples(
                            examples=[
                                [26, 23, 0.5, 15.0, 35.0, 2.0, 2.0, 72.0, ""],
                                [18, 14, 2.0, 28.0, 20.0, -2.0, 8.0, 78.0, ""],
                                [29, 28, 0.0, 8.0, 50.0, 6.0, 0.0, 65.0, ""],
                            ],
                            inputs=[
                                mmse,
                                moca,
                                cdrsb,
                                adas11,
                                ravlt_immediate,
                                ravlt_learning,
                                faq,
                                age,
                                query,
                            ],
                        )

                    with gr.Column(scale=5):
                        diagnosis_badge = gr.HTML(label="Risk Profile Badge")
                        confidence_meter = gr.HTML(label="Confidence Meter")
                        warning_banner = gr.Markdown("")
                        modality_weights_plot = gr.Plot(label="Modality Weights")
                        feature_plot = gr.Plot(label="Feature Importance")
                        report_text = gr.Markdown(
                            f"### Research Output Summary\n\n> {DEMO_MODEL_NOTICE}\n\nNo analysis yet."
                        )
                        gr.Markdown(
                            "*Research prototype only. Demo outputs are not for clinical use.*",
                            elem_classes=["ns-footer"],
                        )

                analyze_btn.click(
                    fn=_run_diagnosis_and_render,
                    inputs=[
                        mmse,
                        moca,
                        cdrsb,
                        adas11,
                        ravlt_immediate,
                        ravlt_learning,
                        faq,
                        age,
                        mri_file,
                        eeg_file,
                        query,
                    ],
                    outputs=[
                        diagnosis_badge,
                        confidence_meter,
                        warning_banner,
                        modality_weights_plot,
                        feature_plot,
                        report_text,
                    ],
                    api_name="diagnose",
                )

            with gr.Tab("📊 Benchmark & Ablation"):
                gr.Markdown("## Modality Ablation Study")
                gr.Markdown(
                    "Ablation removes one modality at a time (MRI, EEG, or cognitive) "
                    "to quantify each modality's contribution to evaluation performance."
                )
                run_ablation_button = gr.Button("▶ Run Ablation Study", variant="secondary")
                ablation_table = gr.Dataframe(
                    headers=["Configuration", "AUC (macro)", "Δ vs Full"],
                    datatype=["str", "number", "str"],
                    label="Ablation Table",
                )
                ablation_plot = gr.Plot(label="Ablation Bar Chart")

                gr.Markdown("## Calibration Analysis")
                run_calibration_button = gr.Button("▶ Run Calibration Check")
                ece_value = gr.Number(label="Expected Calibration Error")
                reliability_plot = gr.Plot(label="Reliability Plot")

                run_ablation_button.click(
                    fn=run_ablation_ui,
                    inputs=[],
                    outputs=[ablation_table, ablation_plot],
                )
                run_calibration_button.click(
                    fn=run_calibration_ui,
                    inputs=[],
                    outputs=[ece_value, reliability_plot],
                )

            with gr.Tab("🔬 Knowledge Graph Explorer"):
                gr.Markdown("## Patient Knowledge Graph")
                with gr.Row():
                    patient_id = gr.Textbox(label="Patient ID", placeholder="e.g. SYN_0001")
                    query_type = gr.Dropdown(
                        choices=["history", "similar", "snapshot", "progression"],
                        value="history",
                        label="Query Type",
                    )
                    target_date = gr.Textbox(label="Date (optional)", placeholder="YYYY-MM-DD (for snapshot)")
                    query_button = gr.Button("Query KG", variant="primary")

                timeline_table = gr.Dataframe(label="Timeline Table")
                kg_stats = gr.Textbox(
                    value=_kg_stats_text,
                    label="KG Stats",
                    lines=2,
                    interactive=False,
                )
                seed_button = gr.Button("🌱 Seed Demo Patients")
                seed_status = gr.Markdown("")

                query_button.click(
                    fn=query_kg_ui,
                    inputs=[patient_id, query_type, target_date],
                    outputs=[timeline_table],
                )
                seed_button.click(
                    fn=seed_demo_kg_ui,
                    inputs=[],
                    outputs=[seed_status],
                ).then(fn=_kg_stats_text, inputs=[], outputs=[kg_stats])

        gr.Markdown(
            "NeuroSight is a research prototype. Current demo weights are not validated "
            "for clinical use. Outputs must be reviewed by qualified specialists.",
            elem_classes=["ns-footer"],
        )
    return demo


def _resolve_server_name() -> str:
    """Return the local/deployment host for Gradio."""
    configured_host = os.environ.get("GRADIO_SERVER_NAME") or os.environ.get("HOST")
    if configured_host:
        return configured_host
    return "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"


def _resolve_server_port(server_name: str) -> int:
    """Return configured PORT or the first available local Gradio port."""
    configured_port = os.environ.get("PORT")
    if configured_port:
        return int(configured_port)

    # On HuggingFace Spaces, always use 7860
    if os.environ.get("SPACE_ID"):
        return 7860

    for candidate in range(7860, 9000):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((server_name, candidate))
            except OSError:
                continue
            return candidate
    raise OSError("No available port found in range 7860-8999.")


if __name__ == "__main__":
    _load_models()
    interface = build_interface()
    server_name = _resolve_server_name()
    server_port = _resolve_server_port(server_name)
    interface.launch(
        server_name=server_name,
        server_port=server_port,
        show_error=True,
    )

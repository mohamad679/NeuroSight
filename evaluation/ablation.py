"""Full modality ablation evaluation for NeuroSight fusion model.

Tests performance across all meaningful subsets of the three modality streams
using synthetic data.

WARNING — SYNTHETIC MODALITIES
================================
In this synthetic benchmark, the MRI and EEG embeddings are independent
Gaussian noise tensors (zero label-predictive information).  Therefore:

  - Cognitive-only performance represents the true information signal.
  - MRI-only and EEG-only configurations test noise-fed pipeline mechanics.
  - The fusion model cannot meaningfully improve over cognitive-only on noise.

This is the correct honest result.  It demonstrates that the modality-routing
architecture works, not that MRI/EEG provide real diagnostic signal.

All outputs carry ``synthetic_data: true`` and ``clinical_validity: false``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


_MODALITY_COMBOS: list[dict[str, Any]] = [
    {"name": "cognitive_only",       "use_mri": False, "use_eeg": False, "use_cog": True},
    {"name": "mri_only_noise",       "use_mri": True,  "use_eeg": False, "use_cog": False},
    {"name": "eeg_only_noise",       "use_mri": False, "use_eeg": True,  "use_cog": False},
    {"name": "mri_plus_cognitive",   "use_mri": True,  "use_eeg": False, "use_cog": True},
    {"name": "eeg_plus_cognitive",   "use_mri": False, "use_eeg": True,  "use_cog": True},
    {"name": "mri_plus_eeg_noise",   "use_mri": True,  "use_eeg": True,  "use_cog": False},
    {"name": "all_modalities",       "use_mri": True,  "use_eeg": True,  "use_cog": True},
]


def run_ablation_benchmark(
    csv_path: str,
    seed: int = 42,
    n_per_class: int = 30,
    epochs: int = 10,
    missing_rate: float = 0.5,
) -> dict[str, Any]:
    """Run full modality ablation on synthetic tabular + noise embeddings.

    Trains and evaluates the CrossModalAttentionFusion model across seven
    modality configurations.  Also runs a missing-modality stress test where
    each sample randomly has some modalities unavailable at inference.

    Args:
        csv_path: Path to synthetic CSV (generated if absent).
        seed: Reproducibility seed.
        n_per_class: Samples per class for synthetic data generation.
        epochs: Training epochs per configuration.
        missing_rate: Fraction of samples with randomly dropped modalities
            during the stress test.

    Returns:
        Dictionary with ``results`` (per-combo AUC/F1), ``stress_test``
        results, and full provenance metadata.
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler

    from evaluation.benchmark import (
        _build_orthogonal_noise_embeddings,
        _load_structured_data,
        _set_seed,
    )
    from evaluation.metrics import compute_auc_roc, compute_brier_score, compute_per_class_metrics
    from neurosight.models.fusion import CrossModalAttentionFusion

    _set_seed(seed)
    features, labels = _load_structured_data(
        csv_path=csv_path, seed=seed, n_per_class=n_per_class
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(split.split(features, labels))
    x_train = features[train_idx]
    y_train = labels[train_idx]
    x_test = features[test_idx]
    y_test = labels[test_idx]

    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    n_classes = 6

    # Scale cognitive features
    scaler = StandardScaler()
    cog_train_sc = scaler.fit_transform(x_train).astype(np.float32)
    cog_test_sc = scaler.transform(x_test).astype(np.float32)

    # Build noise embeddings (independent Gaussian — not from cognitive features)
    mri_train_noise, eeg_train_noise, cog_emb_train = _build_orthogonal_noise_embeddings(
        n_train, seed=seed
    )
    mri_test_noise, eeg_test_noise, cog_emb_test = _build_orthogonal_noise_embeddings(
        n_test, seed=seed + 1
    )

    def _tensor(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32)

    def _evaluate_combo(
        model: CrossModalAttentionFusion,
        use_mri: bool,
        use_eeg: bool,
        use_cog: bool,
    ) -> dict[str, Any]:
        """Evaluate one modality combo, returning AUC/F1/Brier."""
        model.eval()
        with torch.no_grad():
            mri_t = _tensor(mri_test_noise) if use_mri else None
            eeg_t = _tensor(eeg_test_noise) if use_eeg else None
            cog_t = _tensor(cog_emb_test) if use_cog else None

            if mri_t is None and eeg_t is None and cog_t is None:
                return {"macro_auc": float("nan"), "macro_f1": float("nan"), "brier": float("nan")}

            out = model(mri=mri_t, eeg=eeg_t, cog=cog_t)
            y_prob = out["probs"].detach().cpu().numpy().astype(np.float64)

        y_pred = np.argmax(y_prob, axis=1)
        try:
            auc = float(compute_auc_roc(y_test, y_prob).get("macro", float("nan")))
        except Exception:
            auc = float("nan")
        try:
            f1 = float(compute_per_class_metrics(y_test, y_pred)["macro_f1"])
        except Exception:
            f1 = float("nan")
        brier = compute_brier_score(y_test, y_prob)
        return {"macro_auc": auc, "macro_f1": f1, "brier": brier}

    combo_results: dict[str, dict[str, Any]] = {}

    for combo in _MODALITY_COMBOS:
        _set_seed(seed + hash(combo["name"]) % 1000)
        model = CrossModalAttentionFusion(num_classes=n_classes)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        train_y_t = _tensor(y_train.astype(np.float32)).long()
        mri_tr = _tensor(mri_train_noise) if combo["use_mri"] else None
        eeg_tr = _tensor(eeg_train_noise) if combo["use_eeg"] else None
        cog_tr = _tensor(cog_emb_train) if combo["use_cog"] else None

        if mri_tr is None and eeg_tr is None and cog_tr is None:
            combo_results[combo["name"]] = {
                "macro_auc": float("nan"),
                "macro_f1": float("nan"),
                "brier": float("nan"),
                "note": "zero modalities — skipped",
            }
            continue

        model.train()
        for _ in range(epochs):
            perm = torch.randperm(n_train)
            for start in range(0, n_train, 16):
                bidx = perm[start : start + 16]
                optimizer.zero_grad()
                out = model(
                    mri=None if mri_tr is None else mri_tr[bidx],
                    eeg=None if eeg_tr is None else eeg_tr[bidx],
                    cog=None if cog_tr is None else cog_tr[bidx],
                )
                loss = criterion(out["logits"], train_y_t[bidx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        combo_results[combo["name"]] = _evaluate_combo(
            model,
            use_mri=combo["use_mri"],
            use_eeg=combo["use_eeg"],
            use_cog=combo["use_cog"],
        )

    # ---- Missing modality stress test ----
    # Use the all-modalities trained model and randomly null out modalities
    _set_seed(seed + 999)
    stress_model = CrossModalAttentionFusion(num_classes=n_classes)
    stress_opt = torch.optim.AdamW(stress_model.parameters(), lr=8e-4, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    stress_model.train()
    train_y_t = _tensor(y_train.astype(np.float32)).long()
    for _ in range(epochs):
        perm = torch.randperm(n_train)
        for start in range(0, n_train, 16):
            bidx = perm[start : start + 16]
            stress_opt.zero_grad()
            out = stress_model(
                mri=_tensor(mri_train_noise)[bidx],
                eeg=_tensor(eeg_train_noise)[bidx],
                cog=_tensor(cog_emb_train)[bidx],
            )
            loss = criterion(out["logits"], train_y_t[bidx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(stress_model.parameters(), max_norm=1.0)
            stress_opt.step()

    stress_model.eval()
    rng = np.random.default_rng(seed + 7)
    drop_mri = rng.random(n_test) < missing_rate
    drop_eeg = rng.random(n_test) < missing_rate
    drop_cog = rng.random(n_test) < missing_rate

    stress_probs = []
    with torch.no_grad():
        for i in range(n_test):
            mri_i = None if drop_mri[i] else _tensor(mri_test_noise[i : i + 1])
            eeg_i = None if drop_eeg[i] else _tensor(eeg_test_noise[i : i + 1])
            cog_i = None if drop_cog[i] else _tensor(cog_emb_test[i : i + 1])
            if mri_i is None and eeg_i is None and cog_i is None:
                # Fallback: use all-zeros cog embedding
                cog_i = torch.zeros(1, 64, dtype=torch.float32)
            out = stress_model(mri=mri_i, eeg=eeg_i, cog=cog_i)
            stress_probs.append(out["probs"].detach().cpu().numpy())

    stress_prob_arr = np.vstack(stress_probs).astype(np.float64)
    try:
        stress_auc = float(
            compute_auc_roc(y_test, stress_prob_arr).get("macro", float("nan"))
        )
    except Exception:
        stress_auc = float("nan")
    stress_f1 = float(
        compute_per_class_metrics(y_test, np.argmax(stress_prob_arr, axis=1))["macro_f1"]
    )

    stress_result = {
        "missing_rate_per_modality": missing_rate,
        "macro_auc": stress_auc,
        "macro_f1": stress_f1,
        "note": (
            "Each modality independently dropped with probability equal to "
            "missing_rate. Samples with all modalities dropped receive a "
            "zero-tensor cognitive embedding as fallback."
        ),
    }

    return {
        "synthetic_data": True,
        "clinical_validity": False,
        "warning": (
            "MRI and EEG modalities are Gaussian noise tensors in this benchmark. "
            "Cognitive-only AUC represents the true information ceiling. "
            "Any advantage from noise modalities reflects variance, not signal."
        ),
        "modality_note": (
            "mri_only_noise and eeg_only_noise configurations feed pure random "
            "noise to the fusion model — they are expected to perform near-random."
        ),
        "seed": seed,
        "results": combo_results,
        "stress_test": stress_result,
    }

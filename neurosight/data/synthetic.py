"""Synthetic ADNI-like tabular data generation utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

ADNI_COLUMNS: tuple[str, ...] = (
    "RID",
    "DX_bl",
    "AGE",
    "PTGENDER",
    "MMSE",
    "CDRSB",
    "ADAS11",
    "RAVLT_immediate",
    "RAVLT_learning",
    "FAQ",
    "MOCA",
)


def _sample_with_noise(
    rng: np.random.Generator,
    low: float,
    high: float,
    noise_std: float = 1.5,
) -> float:
    """Sample a bounded clinical score with Gaussian perturbation.

    Args:
        rng: Random generator for reproducible sampling.
        low: Lower bound of the class-specific score range.
        high: Upper bound of the class-specific score range.
        noise_std: Standard deviation for Gaussian noise.

    Returns:
        Noisy sample clipped to the provided score range.
    """
    base_value = float(rng.uniform(low, high))
    noisy_value = float(base_value + rng.normal(0.0, noise_std))
    return float(np.clip(noisy_value, low, high))


def generate_structured_synthetic(
    n_per_class: int = 30,
    seed: int = 42,
    output_path: Path | str | None = None,
) -> Any:
    """Generate ADNI-compatible structured synthetic cognitive tabular data.

    Args:
        n_per_class: Number of patients to generate for each diagnosis class.
        seed: Random seed used for NumPy and PyTorch reproducibility.
        output_path: Optional custom path to save the generated CSV.

    Returns:
        Pandas DataFrame when pandas is installed, otherwise list of row dictionaries.
    """
    if n_per_class <= 0:
        raise ValueError("n_per_class must be a positive integer.")

    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    class_profiles: dict[str, dict[str, tuple[float, float]]] = {
        "CN": {
            "MMSE": (28.0, 30.0),
            "CDRSB": (0.0, 0.4),
            "MOCA": (26.0, 30.0),
            "ADAS11": (3.0, 10.0),
            "RAVLT_immediate": (35.0, 55.0),
            "RAVLT_learning": (4.0, 8.0),
            "FAQ": (0.0, 3.0),
            "AGE": (58.0, 85.0),
        },
        "MCI": {
            "MMSE": (24.0, 27.0),
            "CDRSB": (0.5, 1.5),
            "MOCA": (22.0, 25.0),
            "ADAS11": (8.0, 18.0),
            "RAVLT_immediate": (22.0, 40.0),
            "RAVLT_learning": (2.0, 6.0),
            "FAQ": (2.0, 10.0),
            "AGE": (60.0, 88.0),
        },
        "Dementia": {
            "MMSE": (18.0, 23.0),
            "CDRSB": (2.0, 5.0),
            "MOCA": (15.0, 21.0),
            "ADAS11": (18.0, 35.0),
            "RAVLT_immediate": (8.0, 24.0),
            "RAVLT_learning": (0.0, 3.0),
            "FAQ": (10.0, 28.0),
            "AGE": (62.0, 92.0),
        },
        "FTD": {
            "MMSE": (20.0, 25.0),
            "CDRSB": (1.0, 3.0),
            "MOCA": (16.0, 22.0),
            "ADAS11": (14.0, 30.0),
            "RAVLT_immediate": (10.0, 30.0),
            "RAVLT_learning": (0.0, 4.0),
            "FAQ": (7.0, 22.0),
            "AGE": (50.0, 78.0),
        },
        "LBD": {
            "MMSE": (20.0, 26.0),
            "CDRSB": (1.0, 4.0),
            "MOCA": (17.0, 23.0),
            "ADAS11": (12.0, 30.0),
            "RAVLT_immediate": (10.0, 32.0),
            "RAVLT_learning": (1.0, 5.0),
            "FAQ": (6.0, 24.0),
            "AGE": (58.0, 86.0),
        },
        "VD": {
            "MMSE": (20.0, 27.0),
            "CDRSB": (1.0, 4.0),
            "MOCA": (18.0, 24.0),
            "ADAS11": (10.0, 28.0),
            "RAVLT_immediate": (12.0, 35.0),
            "RAVLT_learning": (1.0, 5.0),
            "FAQ": (5.0, 20.0),
            "AGE": (60.0, 90.0),
        },
    }

    rows: list[dict[str, Any]] = []
    rid_counter = 100000
    for dx_label, profile in class_profiles.items():
        for _ in range(n_per_class):
            mmse = _sample_with_noise(rng, *profile["MMSE"])
            cdrsb = _sample_with_noise(rng, *profile["CDRSB"])
            moca = _sample_with_noise(rng, *profile["MOCA"])
            adas11 = _sample_with_noise(rng, *profile["ADAS11"])
            ravlt_immediate = _sample_with_noise(rng, *profile["RAVLT_immediate"])
            ravlt_learning = _sample_with_noise(rng, *profile["RAVLT_learning"])
            faq = _sample_with_noise(rng, *profile["FAQ"])
            age = _sample_with_noise(rng, *profile["AGE"], noise_std=2.0)
            sex = str(rng.choice(["Male", "Female"]))

            rows.append(
                {
                    "RID": str(rid_counter),
                    "DX_bl": dx_label,
                    "AGE": round(age, 3),
                    "PTGENDER": sex,
                    "MMSE": round(mmse, 3),
                    "CDRSB": round(cdrsb, 3),
                    "ADAS11": round(adas11, 3),
                    "RAVLT_immediate": round(ravlt_immediate, 3),
                    "RAVLT_learning": round(ravlt_learning, 3),
                    "FAQ": round(faq, 3),
                    "MOCA": round(moca, 3),
                }
            )
            rid_counter += 1

    if output_path is None:
        target_path = Path("data") / "ADNIMERGE_synthetic.csv"
    else:
        target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with target_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ADNI_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    if pd is not None:
        return pd.DataFrame(rows, columns=list(ADNI_COLUMNS))
    return rows


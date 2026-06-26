"""Generate all report figures for NeuroSight documentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _ensure_output_dir() -> Path:
    """Ensure docs figure directory exists.

    Returns:
        Path to `docs/figures`.
    """
    output_dir = Path("docs") / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _softmax(values: np.ndarray) -> np.ndarray:
    """Compute stable softmax.

    Args:
        values: Input logits array.

    Returns:
        Softmax probabilities.
    """
    centered = values - np.max(values)
    exp_values = np.exp(centered)
    return exp_values / np.sum(exp_values)


def _plot_modality_probability_distributions(output_dir: Path) -> Path:
    """Generate modality probability distribution figure.

    Args:
        output_dir: Destination directory.

    Returns:
        Output image path.
    """
    import matplotlib.pyplot as plt

    diagnoses = ["normal", "mci", "ad", "ftd", "lbd", "vd"]
    prototypes: list[tuple[str, np.ndarray]] = [
        ("Normal prototype", np.array([2.8, 1.2, -0.3, -0.6, -0.5, -0.7], dtype=np.float64)),
        ("MCI prototype", np.array([0.7, 2.1, 0.4, -0.4, -0.5, -0.6], dtype=np.float64)),
        ("AD prototype", np.array([-0.2, 0.5, 2.5, 0.4, 0.2, 0.1], dtype=np.float64)),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for axis, (title, logits) in zip(axes, prototypes):
        probs = _softmax(logits)
        axis.bar(diagnoses, probs, color=["#10B981", "#F59E0B", "#EF4444", "#EF4444", "#EF4444", "#EF4444"])
        axis.set_ylim(0.0, 1.0)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Probability")
    figure.suptitle("Per-Modality Probability Distributions")
    figure.tight_layout()

    output_path = output_dir / "modality_probs.png"
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return output_path


def _plot_xai_cognitive_importance(output_dir: Path) -> Path:
    """Generate cognitive feature-importance chart.

    Args:
        output_dir: Destination directory.

    Returns:
        Output image path.
    """
    import matplotlib.pyplot as plt

    features = [
        "cdrsb",
        "adas11",
        "mmse",
        "moca",
        "faq",
        "ravlt_immediate",
        "ravlt_learning",
        "age",
    ]
    importance = np.array([0.92, 0.86, 0.71, 0.54, 0.41, 0.30, 0.18, 0.06], dtype=np.float64)

    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.barh(features, importance, color="#4F46E5")
    axis.invert_yaxis()
    axis.set_xlabel("Importance Score")
    axis.set_title("Cognitive Feature Importance")
    figure.tight_layout()

    output_path = output_dir / "xai_cognitive.png"
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return output_path


def _plot_calibration_reliability(output_dir: Path) -> Path:
    """Generate reliability diagram using calibration analyzer.

    Args:
        output_dir: Destination directory.

    Returns:
        Output image path.
    """
    from evaluation.calibration import CalibrationAnalyzer

    rng = np.random.default_rng(42)
    n_classes = 6
    n_samples = 300
    y_true = np.repeat(np.arange(n_classes), n_samples // n_classes)
    base_probs = np.full((n_samples, n_classes), 0.05, dtype=np.float64)

    for index, target in enumerate(y_true):
        draw = rng.dirichlet(alpha=np.ones(n_classes) * 0.8)
        draw[target] += 0.55
        draw = draw / draw.sum()
        base_probs[index] = draw

    analyzer = CalibrationAnalyzer(y_true=y_true.astype(np.int64), y_prob=base_probs)
    output_path = output_dir / "calibration_reliability.png"
    analyzer.reliability_diagram(str(output_path))
    return output_path


def _plot_modality_attention_heatmap(output_dir: Path) -> Path:
    """Generate modality attention heatmap figure.

    Args:
        output_dir: Destination directory.

    Returns:
        Output image path.
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(7)
    weights = rng.uniform(0.1, 1.0, size=(3, 6))
    weights = weights / weights.sum(axis=0, keepdims=True)

    figure, axis = plt.subplots(figsize=(10, 3.8))
    heatmap = axis.imshow(weights, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    axis.set_yticks([0, 1, 2], labels=["MRI", "EEG", "Cognitive"])
    axis.set_xticks(np.arange(6), labels=[f"Patient {index + 1}" for index in range(6)])
    axis.set_title("Cross-Modal Attention Weights")
    plt.colorbar(heatmap, ax=axis, fraction=0.03, pad=0.02, label="Weight")
    figure.tight_layout()

    output_path = output_dir / "modality_attention_heatmap.png"
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return output_path


def generate_all_figures() -> list[Path]:
    """Generate all documentation figures.

    Returns:
        List of generated figure paths.
    """
    np.random.seed(42)
    output_dir = _ensure_output_dir()

    generated: list[Path] = []
    generated.append(_plot_modality_probability_distributions(output_dir))
    generated.append(_plot_xai_cognitive_importance(output_dir))
    generated.append(_plot_calibration_reliability(output_dir))
    generated.append(_plot_modality_attention_heatmap(output_dir))
    return generated


def main() -> None:
    """CLI entry point for figure generation."""
    generated_paths = generate_all_figures()
    print(f"Generated {len(generated_paths)} figures to docs/figures/")


if __name__ == "__main__":
    main()

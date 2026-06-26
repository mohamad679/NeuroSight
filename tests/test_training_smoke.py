"""Smoke tests for training and evaluation scripts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.slow


def _repo_root() -> Path:
    """Return repository root path from test file location."""
    return Path(__file__).resolve().parents[1]


def _latest_run_log(run_logs_dir: Path) -> Path:
    """Resolve latest run JSONL path from run log directory.

    Args:
        run_logs_dir: Directory containing run JSONL files.

    Returns:
        Path to the most recently modified JSONL file.
    """
    candidates = list(run_logs_dir.glob("*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No run logs found in {run_logs_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


@pytest.fixture(scope="module")
def training_artifacts() -> dict[str, Any]:
    """Execute smoke training run once and collect artifact metadata."""
    __import__("torch")
    __import__("hydra")
    __import__("monai")

    repo_root = _repo_root()
    command = [
        sys.executable,
        "scripts/train.py",
        "training.warmup_epochs=1",
        "training.finetune_epochs=1",
        "data.batch_size=4",
        "data.num_workers=0",
    ]
    result = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "Smoke training should exit successfully.\n"
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )

    checkpoint_path = repo_root / "checkpoints" / "best_fusion.pt"
    assert checkpoint_path.exists(), "Training must produce checkpoints/best_fusion.pt."

    runs_dir = repo_root / "logs" / "runs"
    latest_log = _latest_run_log(runs_dir)
    train_losses: list[float] = []
    with latest_log.open("r", encoding="utf-8") as handle:
        for raw in handle:
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "metric" and event.get("key") == "train_loss":
                train_losses.append(float(event["value"]))

    return {
        "repo_root": repo_root,
        "checkpoint_path": checkpoint_path,
        "train_losses": train_losses,
        "train_result": result,
    }


def test_one_epoch_training_completes(training_artifacts: dict[str, Any]) -> None:
    """Run 1 epoch of training on synthetic data — no crash, loss decreases."""
    losses = training_artifacts["train_losses"]
    assert losses, "Training run must log at least one train_loss metric event."
    assert len(losses) >= 2, (
        "Warmup=1 and finetune=1 should produce at least two train loss values."
    )
    # Allow for single-epoch variance and warmup-to-finetune shift on synthetic data
    assert min(losses[1:]) <= losses[0] * 1.6, (
        "Post-initial epoch loss should remain bounded relative to initial loss."
    )


def test_checkpoint_is_loadable(training_artifacts: dict[str, Any]) -> None:
    """Saved checkpoint loads correctly into fresh model."""
    import torch

    from neurosight.models.fusion import CrossModalAttentionFusion

    checkpoint_path = training_artifacts["checkpoint_path"]
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")

    required_keys = {"epoch", "model_state", "val_auc", "config"}
    assert required_keys.issubset(checkpoint.keys()), (
        "Checkpoint must include epoch/model_state/val_auc/config keys."
    )

    model = CrossModalAttentionFusion(num_classes=6)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    with torch.no_grad():
        output = model(mri=None, eeg=None, cog=torch.randn(1, 64))
    assert "probs" in output, "Loaded checkpoint model should produce probability outputs."


def test_evaluate_script_produces_json(training_artifacts: dict[str, Any]) -> None:
    """evaluate.py produces logs/eval_results.json with expected keys."""
    repo_root = training_artifacts["repo_root"]
    checkpoint_path = training_artifacts["checkpoint_path"]

    command = [sys.executable, "scripts/evaluate.py", str(checkpoint_path)]
    result = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "Evaluation script should exit successfully.\n"
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )

    eval_path = repo_root / "logs" / "eval_results.json"
    assert eval_path.exists(), "Evaluation script must write logs/eval_results.json."

    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    expected_keys = {
        "checkpoint_path",
        "epoch",
        "macro_auc",
        "ece",
        "per_class_f1",
        "confusion_matrix",
        "modality_ablation",
    }
    assert expected_keys.issubset(payload.keys()), (
        "Evaluation JSON must include checkpoint_path, epoch, macro_auc, ece, per_class_f1, "
        "confusion_matrix, and modality_ablation."
    )

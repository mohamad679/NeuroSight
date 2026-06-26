"""Unified experiment tracking for NeuroSight."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentLogger:
    """Unified experiment tracking with MLflow + JSONL fallback.

    Supports logging parameters, metrics, artifacts, and model checkpoints.
    Automatically falls back to JSONL if MLflow server is unavailable.
    """

    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None) -> None:
        """Initialize an experiment logger.

        Args:
            experiment_name: Name of the logical experiment namespace.
            tracking_uri: Optional MLflow tracking server URI.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_id: Optional[str] = None
        self._run_active = False
        self._mlflow: Optional[Any] = None
        self._mlflow_available = False
        self._jsonl_path: Optional[Path] = None
        self._artifacts_dir: Optional[Path] = None

        try:
            import mlflow

            self._mlflow = mlflow
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._mlflow_available = True
        except ModuleNotFoundError:
            self._mlflow_available = False

    def start_run(self, run_name: str, tags: Dict[str, str]) -> None:
        """Begin a new tracked run.

        Args:
            run_name: Human-readable run name.
            tags: Metadata tags associated with the run.
        """
        if self._run_active:
            raise RuntimeError("A run is already active. Call end_run() before starting a new run.")

        generated_run_id = uuid.uuid4().hex
        self.run_id = generated_run_id

        if self._mlflow_available and self._mlflow is not None:
            try:
                active_run = self._mlflow.start_run(run_name=run_name, tags=tags)
                self.run_id = str(active_run.info.run_id)
            except (OSError, ValueError, RuntimeError, ConnectionError):
                self._mlflow_available = False

        if self.run_id is None:
            raise RuntimeError("Unable to initialize run identifier.")

        runs_dir = Path("logs") / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = runs_dir / f"{self.run_id}.jsonl"
        self._artifacts_dir = Path("logs") / "artifacts" / self.run_id
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._run_active = True

        self._append_event(
            {
                "event": "run_start",
                "run_name": run_name,
                "tags": tags,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }
        )

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters (training config).

        Args:
            params: Mapping from parameter names to values.
        """
        self._ensure_active_run()

        if self._mlflow_available and self._mlflow is not None:
            safe_params = {key: str(value) for key, value in params.items()}
            try:
                self._mlflow.log_params(safe_params)
            except (OSError, ValueError, RuntimeError, ConnectionError):
                self._mlflow_available = False

        for key, value in params.items():
            self._append_event(
                {
                    "event": "param",
                    "key": str(key),
                    "value": value,
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                }
            )

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log scalar metrics at a given training step.

        Args:
            metrics: Mapping from metric names to scalar values.
            step: Training step or epoch index.
        """
        self._ensure_active_run()

        if self._mlflow_available and self._mlflow is not None:
            try:
                numeric_metrics = {
                    key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))
                }
                self._mlflow.log_metrics(numeric_metrics, step=step)
            except (OSError, ValueError, RuntimeError, ConnectionError):
                self._mlflow_available = False

        for key, value in metrics.items():
            self._append_event(
                {
                    "event": "metric",
                    "key": str(key),
                    "value": float(value),
                    "step": int(step),
                    "ts": datetime.now(tz=timezone.utc).isoformat(),
                }
            )

    def log_artifact(self, local_path: str, artifact_name: str) -> None:
        """Log a file artifact (e.g., checkpoint, confusion matrix).

        Args:
            local_path: Source file path on local filesystem.
            artifact_name: Artifact namespace or folder name.
        """
        self._ensure_active_run()
        artifact_path = Path(local_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact path does not exist: {local_path}")

        if self._mlflow_available and self._mlflow is not None:
            try:
                self._mlflow.log_artifact(str(artifact_path), artifact_path=artifact_name)
            except (OSError, ValueError, RuntimeError, ConnectionError):
                self._mlflow_available = False

        if self._artifacts_dir is None:
            raise RuntimeError("Artifact directory is not initialized.")
        destination_dir = self._artifacts_dir / artifact_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / artifact_path.name
        shutil.copy2(artifact_path, destination_path)

        self._append_event(
            {
                "event": "artifact",
                "artifact_name": artifact_name,
                "local_path": str(artifact_path),
                "stored_path": str(destination_path),
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }
        )

    def register_model(self, model_path: str, model_name: str, metrics: Dict[str, float]) -> str:
        """Register a trained model in the registry. Returns run_id.

        Args:
            model_path: Local checkpoint path to register.
            model_name: Human-readable model name.
            metrics: Summary metrics for the registered model.

        Returns:
            Active run ID associated with this registration.
        """
        self._ensure_active_run()
        if self.run_id is None:
            raise RuntimeError("Run ID is not available.")

        self.log_artifact(model_path, artifact_name="checkpoints")

        if self._mlflow_available and self._mlflow is not None:
            model_uri = f"runs:/{self.run_id}/checkpoints/{Path(model_path).name}"
            try:
                self._mlflow.register_model(model_uri=model_uri, name=model_name)
            except (OSError, ValueError, RuntimeError, ConnectionError):
                self._mlflow_available = False

        self._append_event(
            {
                "event": "model_registration",
                "model_name": model_name,
                "model_path": model_path,
                "metrics": metrics,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }
        )
        return self.run_id

    def end_run(self, status: str = "FINISHED") -> None:
        """Close the current run.

        Args:
            status: Terminal run status string.
        """
        self._ensure_active_run()
        if self._mlflow_available and self._mlflow is not None:
            try:
                self._mlflow.end_run(status=status)
            except (OSError, ValueError, RuntimeError, ConnectionError):
                self._mlflow_available = False

        self._append_event(
            {
                "event": "run_end",
                "status": status,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }
        )
        self._run_active = False

    def _ensure_active_run(self) -> None:
        """Validate that an active run exists before logging calls."""
        if not self._run_active:
            raise RuntimeError("No active run. Call start_run() before logging.")

    def _append_event(self, payload: Dict[str, Any]) -> None:
        """Append one JSON event to the run-local JSONL log.

        Args:
            payload: Event dictionary to serialize.
        """
        if self._jsonl_path is None:
            raise RuntimeError("JSONL run path is not initialized.")
        with self._jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")


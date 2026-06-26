"""MLflow bridge for the NeuroSight model registry.

The project keeps a lightweight JSON model registry for local/demo use. This
module maps that registry into MLflow tracking concepts so reviewers can inspect
run lineage, metrics, checkpoint artifacts, and promotion status with standard
MLOps tooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from neurosight.tracking.model_registry import ModelRegistry


DEFAULT_EXPERIMENT_NAME = "neurosight_model_registry"


@dataclass
class MLflowRegistryResult:
    """Result for one model-registry sync or dry-run plan row."""

    source_run_id: str
    model_name: str
    status: str
    alias: str
    checkpoint_path: str
    checkpoint_exists: bool
    metrics: Dict[str, float]
    synced: bool
    detail: str
    mlflow_run_id: Optional[str] = None
    mlflow_model_version: Optional[str] = None


class MLflowRegistryBridge:
    """Synchronize the local NeuroSight registry into MLflow tracking."""

    def __init__(
        self,
        registry_path: str = "logs/model_registry.json",
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
        tracking_uri: Optional[str] = None,
    ) -> None:
        self.registry = ModelRegistry(registry_path=registry_path)
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

    def plan(self) -> List[MLflowRegistryResult]:
        """Return a dry-run plan without importing or writing to MLflow."""
        return [self._plan_entry(entry) for entry in self.registry.list_runs()]

    def sync(
        self,
        *,
        register_model: bool = False,
    ) -> List[MLflowRegistryResult]:
        """Sync local registry entries into MLflow tracking.

        Args:
            register_model: Also attempt MLflow model registration from the
                checkpoint artifact. This is optional because the current
                NeuroSight checkpoint is a PyTorch `.pt` artifact, not a full
                MLflow model directory with an `MLmodel` file.

        Returns:
            One result per local registry entry.
        """
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
        except ModuleNotFoundError as exc:
            raise RuntimeError("MLflow is not installed. Install project dependencies first.") from exc

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        client = MlflowClient()

        results: List[MLflowRegistryResult] = []
        for entry in self.registry.list_runs():
            result = self._plan_entry(entry)
            checkpoint = Path(result.checkpoint_path)
            tags = {
                "neurosight.source": "local_json_registry",
                "neurosight.source_run_id": result.source_run_id,
                "neurosight.status": result.status,
                "neurosight.alias": result.alias,
                "neurosight.checkpoint_exists": str(result.checkpoint_exists).lower(),
            }

            with mlflow.start_run(run_name=f"registry-{result.source_run_id}", tags=tags) as run:
                result.mlflow_run_id = str(run.info.run_id)
                mlflow.log_params(flatten_config(entry.get("config", {})))
                mlflow.log_metrics(result.metrics)

                if checkpoint.exists():
                    mlflow.log_artifact(str(checkpoint), artifact_path="checkpoints")
                else:
                    mlflow.set_tag("neurosight.checkpoint_missing_reason", "path_not_found")

                if register_model and checkpoint.exists():
                    model_uri = f"runs:/{result.mlflow_run_id}/checkpoints/{checkpoint.name}"
                    try:
                        registered = mlflow.register_model(
                            model_uri=model_uri,
                            name=result.model_name,
                        )
                        result.mlflow_model_version = str(registered.version)
                        self._set_alias(client, result.model_name, result.alias, registered.version)
                    except (OSError, ValueError, RuntimeError) as exc:
                        result.synced = True
                        result.detail = (
                            "synced run and artifact, but MLflow model registration failed: "
                            f"{exc}"
                        )
                        results.append(result)
                        continue

                result.synced = True
                result.detail = "synced to MLflow tracking"
                if register_model and not checkpoint.exists():
                    result.detail = "synced metadata; skipped registration because checkpoint is missing"

            results.append(result)

        return results

    @staticmethod
    def _set_alias(client: Any, model_name: str, alias: str, version: Any) -> None:
        """Set an MLflow registered-model alias when supported by the client."""
        setter = getattr(client, "set_registered_model_alias", None)
        if callable(setter):
            setter(model_name, alias, str(version))

    def _plan_entry(self, entry: Dict[str, Any]) -> MLflowRegistryResult:
        metrics = numeric_metrics(entry.get("metrics", {}))
        checkpoint_path = str(entry.get("checkpoint_path", ""))
        status = str(entry.get("status", "staging"))
        return MLflowRegistryResult(
            source_run_id=str(entry.get("run_id", "unknown")),
            model_name=str(entry.get("model_name", "neurosight_fusion")),
            status=status,
            alias=status_to_alias(status),
            checkpoint_path=checkpoint_path,
            checkpoint_exists=Path(checkpoint_path).exists() if checkpoint_path else False,
            metrics=metrics,
            synced=False,
            detail="dry run",
        )


def status_to_alias(status: str) -> str:
    """Map NeuroSight registry status to MLflow-style aliases."""
    normalized = status.strip().lower()
    if normalized == "production":
        return "champion"
    if normalized == "archived":
        return "archived"
    return "candidate"


def numeric_metrics(value: Any) -> Dict[str, float]:
    """Return only numeric metrics in MLflow-compatible form."""
    if not isinstance(value, dict):
        return {}
    metrics: Dict[str, float] = {}
    for key, item in value.items():
        if isinstance(item, (int, float)):
            metrics[str(key)] = float(item)
    return metrics


def flatten_config(value: Any, prefix: str = "") -> Dict[str, str]:
    """Flatten nested config dictionaries for MLflow params."""
    if not isinstance(value, dict):
        return {}

    flattened: Dict[str, str] = {}
    for key, item in value.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            flattened.update(flatten_config(item, full_key))
        elif isinstance(item, (str, int, float, bool)) or item is None:
            flattened[full_key] = "null" if item is None else str(item)
        else:
            flattened[full_key] = str(item)
    return flattened

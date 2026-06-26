"""Model registry persistence helpers for NeuroSight."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class ModelRegistry:
    """Local JSON-backed model registry with environment promotion support."""

    def __init__(self, registry_path: str = "logs/model_registry.json") -> None:
        """Initialize model registry store.

        Args:
            registry_path: Path to JSON registry file.
        """
        self.registry_path = Path(registry_path)

    def register_model(
        self,
        run_id: str,
        model_name: str,
        checkpoint_path: str,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        status: str = "staging",
    ) -> Dict[str, Any]:
        """Save one model run metadata entry into registry.

        Args:
            run_id: Unique run identifier.
            model_name: Name of model version.
            checkpoint_path: Checkpoint file path.
            metrics: Validation metrics for this run.
            config: Training configuration dictionary.
            status: Run status (`staging`, `production`, or `archived`).

        Returns:
            The newly written metadata entry.
        """
        if status not in {"staging", "production", "archived"}:
            raise ValueError("status must be one of: staging, production, archived.")

        entries = self._load_entries()
        entries = [entry for entry in entries if str(entry.get("run_id")) != run_id]

        if status == "production":
            for entry in entries:
                entry["status"] = "archived"

        new_entry: Dict[str, Any] = {
            "run_id": run_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "model_name": model_name,
            "checkpoint_path": checkpoint_path,
            "metrics": metrics,
            "config": config,
            "status": status,
        }
        entries.append(new_entry)
        self._save_entries(entries)
        return new_entry

    def promote_to_production(self, run_id: str) -> Dict[str, Any]:
        """Mark a run as production and archive all other runs.

        Args:
            run_id: Target run ID to promote.

        Returns:
            Metadata entry of promoted run.
        """
        entries = self._load_entries()
        promoted: Optional[Dict[str, Any]] = None

        for entry in entries:
            if str(entry.get("run_id")) == run_id:
                entry["status"] = "production"
                promoted = entry
            else:
                entry["status"] = "archived"

        if promoted is None:
            raise ValueError(f"Run '{run_id}' not found in registry.")

        self._save_entries(entries)
        return promoted

    def get_production_model(self) -> Dict[str, Any]:
        """Return metadata of current production model.

        Returns:
            Production model metadata, or an empty dictionary if unavailable.
        """
        entries = self._load_entries()
        production_entries = [entry for entry in entries if entry.get("status") == "production"]
        if not production_entries:
            return {}

        production_entries.sort(key=lambda item: str(item.get("timestamp", "")), reverse=True)
        return production_entries[0]

    def list_runs(self) -> List[Dict[str, Any]]:
        """List all registered model runs sorted by val_auc descending.

        Returns:
            Sorted list of model metadata entries.
        """
        entries = self._load_entries()

        def _score(entry: Dict[str, Any]) -> float:
            metrics = entry.get("metrics", {})
            if not isinstance(metrics, dict):
                return float("-inf")
            try:
                return float(metrics.get("val_auc", float("-inf")))
            except (TypeError, ValueError):
                return float("-inf")

        return sorted(entries, key=_score, reverse=True)

    def _load_entries(self) -> List[Dict[str, Any]]:
        """Load model registry entries from disk."""
        if not self.registry_path.exists():
            return []

        try:
            content = self.registry_path.read_text(encoding="utf-8").strip()
            if not content:
                return []
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid registry JSON format: {self.registry_path}") from exc

        if isinstance(parsed, list):
            entries = [entry for entry in parsed if isinstance(entry, dict)]
        elif isinstance(parsed, dict) and isinstance(parsed.get("runs"), list):
            entries = [entry for entry in parsed["runs"] if isinstance(entry, dict)]
        else:
            raise ValueError("Registry JSON must be a list or an object containing a 'runs' list.")
        return entries

    def _save_entries(self, entries: List[Dict[str, Any]]) -> None:
        """Persist registry entries to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

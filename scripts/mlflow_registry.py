#!/usr/bin/env python3
"""Inspect or sync the NeuroSight model registry with MLflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neurosight.tracking.mlflow_registry import (
    DEFAULT_EXPERIMENT_NAME,
    MLflowRegistryBridge,
    MLflowRegistryResult,
)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an MLflow sync plan or sync local NeuroSight model-registry entries."
    )
    parser.add_argument(
        "--registry-path",
        default="logs/model_registry.json",
        help="Path to the local JSON model registry.",
    )
    parser.add_argument(
        "--experiment-name",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME),
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help="Optional MLflow tracking URI. Defaults to local ./mlruns behavior.",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Write registry entries to MLflow tracking. Default is dry-run only.",
    )
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Also attempt MLflow Model Registry registration for checkpoint artifacts.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON.",
    )
    return parser.parse_args(argv)


def print_table(results: List[MLflowRegistryResult], *, dry_run: bool) -> None:
    title = "MLflow registry sync plan" if dry_run else "MLflow registry sync result"
    print("")
    print(title)
    print("")
    if not results:
        print("No local model-registry entries found.")
        return

    for item in results:
        marker = "SYNCED" if item.synced else "PLAN"
        checkpoint = "yes" if item.checkpoint_exists else "no"
        metrics = ", ".join(f"{key}={value:.4f}" for key, value in item.metrics.items())
        print(f"[{marker}] {item.source_run_id}")
        print(f"        model: {item.model_name}")
        print(f"       status: {item.status} -> alias={item.alias}")
        print(f"   checkpoint: {item.checkpoint_path or '-'} exists={checkpoint}")
        print(f"      metrics: {metrics or '-'}")
        if item.mlflow_run_id:
            print(f"   mlflow_run: {item.mlflow_run_id}")
        if item.mlflow_model_version:
            print(f" model_version: {item.mlflow_model_version}")
        print(f"       detail: {item.detail}")
        print("")


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    bridge = MLflowRegistryBridge(
        registry_path=args.registry_path,
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
    )

    try:
        results = bridge.sync(register_model=args.register_model) if args.sync else bridge.plan()
    except RuntimeError as exc:
        print(f"MLflow registry sync failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        print_table(results, dry_run=not args.sync)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

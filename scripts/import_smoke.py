#!/usr/bin/env python3
"""Import smoke test for clean CI environments."""

from __future__ import annotations

import importlib
import sys

MODULES = (
    "api.main",
    "evaluation.benchmark",
    "neurosight.data.synthetic",
    "neurosight.models.cognitive",
    "neurosight.models.eeg",
    "neurosight.models.fusion",
    "neurosight.models.mri",
    "neurosight.tracking.model_registry",
)


def main() -> int:
    failed: list[str] = []
    for module_name in MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - smoke test reports import failures.
            failed.append(f"{module_name}: {exc}")

    if failed:
        print("Import smoke failed:")
        for item in failed:
            print(f"- {item}")
        return 1

    print("Import smoke passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

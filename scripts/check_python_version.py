#!/usr/bin/env python3
"""Validate the supported Python runtime before installing dependencies."""

from __future__ import annotations

import sys


def main() -> int:
    version = sys.version_info
    if version.major == 3 and version.minor == 11:
        print("Python version check passed: 3.11")
        return 0

    print(
        "NeuroSight requires Python 3.11.x. "
        f"Current interpreter is {version.major}.{version.minor}.{version.micro}.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

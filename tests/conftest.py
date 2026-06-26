"""Shared pytest fixtures for CPU-only synthetic test data."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import pytest
import torch

from neurosight.utils.seed import set_global_seed

matplotlib.use("Agg", force=True)


def pytest_configure(config: pytest.Config) -> None:
    """Register test markers used by CI and local commands."""
    config.addinivalue_line("markers", "unit: fast unit tests")
    config.addinivalue_line("markers", "integration: essential integration tests")
    config.addinivalue_line("markers", "slow: CPU-heavy or long-running tests")
    config.addinivalue_line("markers", "benchmark: benchmark and baseline execution tests")


@pytest.fixture(autouse=True)
def deterministic_test_seed() -> None:
    """Reset deterministic seeds before each test."""
    set_global_seed(1234)


@pytest.fixture(scope="session")
def cpu_device() -> torch.device:
    """Return the CPU device required by all tests."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def device(cpu_device: torch.device) -> torch.device:
    """Compatibility alias for tests that request a generic device fixture."""
    return cpu_device


@pytest.fixture(scope="session")
def batch_size() -> int:
    """Return the standard synthetic batch size."""
    return 2


@pytest.fixture()
def synthetic_multimodal_batch(
    cpu_device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Return synthetic ADNI-like multimodal tensors for CPU tests."""
    return {
        "mri": torch.randn(batch_size, 1, 96, 96, 96, device=cpu_device),
        "eeg": torch.randn(batch_size, 19, 1024, device=cpu_device),
        "cog": torch.randn(batch_size, 8, device=cpu_device),
        "label": torch.arange(batch_size, device=cpu_device) % 6,
    }


@pytest.fixture()
def synthetic_batch(
    synthetic_multimodal_batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compatibility alias for the shared synthetic multimodal batch."""
    return synthetic_multimodal_batch

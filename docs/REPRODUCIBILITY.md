# Reproducibility And Test Guide

NeuroSight targets Python 3.11 only. The CI and reviewer path uses
`requirements-dev.txt` with `requirements.lock` so a reviewer can install from
a clean clone without Poetry or hidden local files.

## Setup

```bash
python3.11 -m venv /tmp/neurosight-venv
source /tmp/neurosight-venv/bin/activate
python scripts/check_python_version.py
make install
```

Equivalent manual install:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt -c requirements.lock
python -m pip install flwr==1.8.0 --no-deps
```

Expected install notes:

- `requirements.lock` pins the reviewed Python 3.11 dependency set used by CI.
- `constraints.txt` is kept only as a bounded compatibility file for
  exploratory local installs.
- `torch==2.2.0` and `monai==1.3.0` are pinned because model tensor behavior
  and MONAI APIs are central to the scaffold.
- `numpy<2.0` keeps compatibility with Torch 2.2, SHAP 0.44, and numba 0.59.
- `scikit-learn>=1.4,<1.8` covers current stable releases while protecting
  against silent baseline API changes.
- `fastapi==0.111.0` and `starlette>=0.37.2,<0.38` keep TestClient behavior
  stable.
- `gradio>=4.40,<4.45` avoids unreviewed major UI/runtime changes.

Refresh the lock only after dependency review:

```bash
python -m pip install pip-tools
pip-compile requirements-dev.txt \
  --constraint constraints.txt \
  --output-file requirements.lock
```

Do not commit credentials, private data, local virtual environments, generated
logs, benchmark outputs, or real-cohort processed files.

## One-Command Verification

The reviewer command is:

```bash
make verify
```

It runs repository hygiene, lint/type checks, model-card and quality gates, the
local supply-chain audit, import smoke, fast unit/integration tests, safety
red-team tests, and the synthetic benchmark smoke CLI. Frontend checks run
automatically when `frontend/node_modules` is present; otherwise they skip with
a clear message.

## Test Commands

Fast CPU-safe suite, expected to run in a few minutes on a laptop CPU:

```bash
make test
```

Benchmark smoke test with tiny synthetic data:

```bash
make benchmark-smoke
```

Explicit fast test target:

```bash
make test-fast
```

Optional slow/release checks:

```bash
make test-slow
```

Repository hygiene and imports:

```bash
make hygiene
make import-smoke
```

API smoke path:

```bash
make smoke-api
```

Safety and quality:

```bash
make safety
make quality
```

All pytest Make targets pass a timeout through `pytest-timeout`. The default is
300 seconds per test and can be changed locally:

```bash
make test-fast PYTEST_TIMEOUT=120
```

## Docker

Build and run the API container:

```bash
docker build -t neurosight:local .
docker run --rm -p 8000:8000 -e APP_ENV=production neurosight:local
```

The Docker image uses Python 3.11 slim, installs from `requirements.lock`, sets
explicit runtime environment variables, and runs `uvicorn` as a non-root
`neurosight` user. Secrets are not baked into the image; pass environment
variables at runtime only when needed.

## Pytest Markers

- `unit`: fast unit-level checks.
- `integration`: essential in-process integration checks.
- `benchmark`: benchmark and baseline execution checks.
- `slow`: CPU-heavy or long-running checks kept out of default CI.

Default CI runs fast non-slow/non-benchmark/non-safety tests, plus explicit
safety and benchmark smoke jobs.

## Determinism

Tests use `neurosight.utils.seed.set_global_seed`, which seeds Python `random`,
NumPy, Torch CPU/CUDA generators, and Torch deterministic mode. New tests should
use this helper instead of adding ad hoc seed code.

## Troubleshooting

- If `pytest` resolves to a broken interpreter, run it through the active
  environment: `python -m pytest ...`.
- Keep virtual environments outside the repository tree when running the
  hygiene gate; `.venv/` inside the repo is intentionally rejected.
- If sklearn baseline tests fail after an upgrade, inspect
  `evaluation.benchmark._build_logistic_regression`; it must not pass removed
  multiclass parameters.
- If hygiene fails after linting, remove generated local caches:
  `rm -rf .ruff_cache .pytest_cache .mypy_cache`.
- If dependency resolution reports a Torch conflict on Python 3.12 or 3.13,
  switch to Python 3.11; Torch 2.2 is intentionally pinned for this project.
- If Torch wheels are unavailable for your platform, use the official PyTorch
  CPU wheel index for Python 3.11 and keep `torch==2.2.0`.

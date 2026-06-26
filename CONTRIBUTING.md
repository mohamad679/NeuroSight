# Contributing to NeuroSight

## Development Setup
1. Install Python 3.11 and Poetry.
2. Install project dependencies:
   ```bash
   poetry install --with dev
   ```
3. Create local environment variables from the repository root:
   ```bash
   cp .env.example .env
   ```
4. Launch the API and UI locally:
   ```bash
   poetry run uvicorn api.main:app --reload
   poetry run python app.py
   ```

## Running Tests
Run the complete suite with test-mode behavior:
```bash
APP_ENV=test poetry run pytest tests/ -v --tb=short --ignore=tests/test_phase1.py
```
Run focused suites:
```bash
poetry run pytest tests/test_data_pipeline.py -v
poetry run pytest tests/test_api_complete.py -v
poetry run pytest tests/test_benchmark.py -v
```

## Code Style (ruff + mypy)
Lint and type-check before opening a PR:
```bash
poetry run ruff check . --ignore E501
poetry run mypy neurosight/ api/ evaluation/ --ignore-missing-imports
```
Format code with:
```bash
poetry run ruff format .
```

## Public Repository Hygiene
- Keep local assistant files such as `CLAUDE.md`, `.claude/`, `.codex/`, `.cursor/`, and `.windsurf/` out of Git.
- Keep `.env`, generated `logs/`, local `data/`, and `checkpoints/` out of Git.
- Commit only safe examples such as `.env.example`, docs, source code, tests, synthetic-data contracts, and generated public figures.
- Before publishing, run:
  ```bash
  python3 scripts/github_readiness.py --strict
  python3 scripts/portfolio_check.py --strict
  ```
- See `docs/PUBLIC_REPOSITORY_GUIDE.md` for the public/private file policy and AI wording standard.

## Medical AI Guidelines for Contributors
- Do not hardcode diagnosis values in endpoint/business logic.
- Safety guardian rules cannot be weakened without explicit reviewer approval.
- Every new endpoint returning medical outputs must include `requires_review` in the payload.
- New features must not break existing `tests/test_phase*.py` compatibility tests.
- Keep medical disclaimers intact in README, UI, and API-facing documentation.
- Describe outputs as research/demo model behavior, not clinical diagnosis or medical certainty.

## How to Add a New Modality
1. Add modality contracts in `neurosight/contracts.py`.
2. Implement encoder/classifier under `neurosight/models/` with calibrated output path.
3. Extend `CrossModalAttentionFusion` projection and missing-modality handling.
4. Update upload endpoint(s) in `api/main.py` for preprocessing + embedding extraction.
5. Add modality-level XAI integration in `neurosight/models/xai.py`.
6. Add unit/integration tests under `tests/` for upload, diagnosis, and failure handling.

## How to Add a New Evaluation Metric
1. Implement metric function under `evaluation/metrics.py` or a dedicated module.
2. Add metric logging to `scripts/train.py` and `scripts/evaluate.py`.
3. Add API exposure if needed (`/v1/eval/*` endpoint updates).
4. Add benchmark/test assertions so metric appears in regression checks.
5. Document interpretation limits in README or `docs/TECHNICAL_REPORT.md`.

## Pull Request Process
1. Create a branch from `main`.
2. Keep PR scope tight; separate refactors from feature changes.
3. Include tests for all behavior changes.
4. Fill out `.github/PULL_REQUEST_TEMPLATE.md`.
5. Request review from at least one maintainer before merge.
6. Squash or rebase only when branch history is noisy; preserve meaningful commit history.

## Issue Labels
- `bug`: Reproducible incorrect behavior.
- `enhancement`: Feature request or capability extension.
- `documentation`: README/report/notebook/docs updates.
- `good first issue`: Safe onboarding task for new contributors.
- `medical-safety`: Changes that affect triage text, confidence thresholds, or safety guardrails.
- `performance`: Latency, memory, throughput, or training-time optimization tasks.

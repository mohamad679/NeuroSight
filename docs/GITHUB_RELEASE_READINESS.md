# GitHub Release Readiness

Phase 5 adds a final portfolio-release audit for NeuroSight. This is the last
layer before publishing the repository for scholarship, internship, or job
review. It checks whether the project is easy to evaluate, honest about its
limits, and protected from common GitHub portfolio mistakes.

## Runnable Command

```bash
python3 scripts/github_readiness.py --strict
```

With Poetry:

```bash
make github-readiness
```

Default report:

```text
logs/github_readiness/neurosight_github_readiness_report.json
```

The report is generated under `logs/`, which is intentionally ignored by Git.

## What It Checks

| Check | Why it matters |
|-------|----------------|
| Required public files | Ensures README, model card, security policy, license, dependencies, and frontend manifest exist |
| Reviewer evidence files | Ensures the architecture, API, portfolio, safety, and runnable proof artifacts are present |
| README structure | Ensures reviewers can quickly find demo, architecture, limitations, API, dataset, and license sections |
| Positioning disclosures | Ensures synthetic-data and non-clinical warnings are visible |
| `.gitignore` hygiene | Ensures secrets, generated logs, data, checkpoints, and frontend build artifacts are protected |
| Private working-file hygiene | Ensures assistant and IDE planning files such as `CLAUDE.md` are ignored |
| Public secret scan | Searches non-ignored public candidate files for obvious token formats |
| Visual evidence | Ensures generated figures exist and rejects placeholder screenshots |
| GitHub workflows | Ensures CI, quality, security, and deploy workflows are present |
| Frontend scripts | Ensures the dashboard exposes dev, build, and type-check commands |
| License alignment | Ensures README and `LICENSE` agree on MIT licensing |

## Why This Is Different From The Quality Gate

The quality gate checks repository engineering readiness. The GitHub readiness
audit checks portfolio presentation and publishability.

| Script | Main question answered |
|--------|------------------------|
| `scripts/quality_gate.py` | Does the repo still contain required implementation, governance, and CI artifacts? |
| `scripts/api_contract_check.py` | Does the backend route surface behave as documented? |
| `scripts/github_readiness.py` | Is the public GitHub package clean, honest, and reviewer-friendly? |
| `scripts/portfolio_check.py` | Do the main reviewer-proof checks pass together? |

## Before Publishing

Run:

```bash
python3 scripts/github_readiness.py --strict
python3 scripts/portfolio_check.py --strict
```

Then inspect:

```text
logs/github_readiness/neurosight_github_readiness_report.json
logs/portfolio/neurosight_portfolio_check_report.json
```

## What This Does Not Prove

This audit does not prove clinical performance, real deployment uptime, real
MRI/EEG diagnostic accuracy, or regulatory readiness. It proves GitHub portfolio
hygiene and reviewer evidence quality.

## Defensible Phase 5 Outcome

After Phase 5, NeuroSight has:

- a real license file,
- a runnable GitHub readiness audit,
- secret/artifact guardrail checks,
- local assistant and IDE working-file guardrails,
- portfolio evidence discoverability checks,
- public-positioning checks,
- and integration into the one-command portfolio proof.

That makes the project much easier to defend as a serious student AI systems
portfolio rather than a UI-only prototype.

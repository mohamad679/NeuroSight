# CI/CD Quality Gate

This item adds an offline quality gate that can run locally or in GitHub Actions
before heavier dependency installation and test jobs. It is designed to answer a
reviewer's practical question: "Does this repository still contain the evidence,
disclosures, safety checks, and CI wiring claimed by the README?"

## Files

| File | Purpose |
|------|---------|
| `neurosight/governance/quality_gate.py` | Quality-gate logic and JSON report builder |
| `scripts/quality_gate.py` | CLI entrypoint for local and CI runs |
| `scripts/api_contract_check.py` | In-process FastAPI contract check for routes, SSE, auth, and headers |
| `scripts/github_readiness.py` | GitHub portfolio publishability audit |
| `scripts/portfolio_check.py` | Reviewer-facing aggregate proof script that includes the quality gate |
| `.github/workflows/quality_gate.yml` | GitHub Actions workflow that runs the gate |
| `logs/quality/neurosight_quality_gate_report.json` | Default generated report path; ignored by Git through `logs/` |

## What The Gate Checks

| Gate | Blocks CI? | What it proves |
|------|------------|----------------|
| Required repo files | Yes | README, model card, security policy, dependency manifests, and frontend manifest exist |
| Roadmap artifacts | Yes | The senior-roadmap docs and runnable scripts are present |
| Python syntax | Yes | `neurosight/` and `scripts/` Python files compile |
| Evaluation artifact | Yes | `evaluation/results.json` is valid and contains core metrics |
| Model card | Yes | Required sections, disclosures, metrics, and evidence links pass |
| AI safety | Yes | OWASP GenAI regression suite passes |
| Supply-chain criticals | Yes | No critical secret/supply-chain finding is detected by the local audit |
| Workflow baseline | Yes | CI, security, deploy, and quality-gate workflows exist with minimal permissions |
| Make targets | Warning | Developer commands expose the roadmap scripts |

Warnings are recorded in the report but do not fail strict mode. Blockers fail
strict mode.

## Run Locally

```bash
python3 scripts/quality_gate.py
```

With the Makefile:

```bash
make quality-gate
```

Reviewer-facing aggregate proof:

```bash
python3 scripts/api_contract_check.py --strict
python3 scripts/github_readiness.py --strict
python3 scripts/portfolio_check.py --strict
make api-contract-check
make github-readiness
make portfolio-check
```

Strict mode:

```bash
python3 scripts/quality_gate.py --strict
```

Print the JSON:

```bash
python3 scripts/quality_gate.py --stdout
```

## GitHub Actions

The workflow is:

```text
.github/workflows/quality_gate.yml
```

It runs on pushes to `main` and `develop`, pull requests to `main`, and manual
dispatch. It uploads the JSON report as an artifact:

```text
logs/quality/neurosight_quality_gate_report.json
```

The workflow installs the locked reviewer dependency set before running
`make quality`, which includes lint/type checks, repository hygiene,
model-card consistency, the quality gate, and the local supply-chain audit.

## Relationship To Existing CI

The quality gate complements, but does not replace:

- `ci.yml`: locked pip install, ruff, mypy, pytest, safety tests, benchmark smoke,
- `security_supply_chain.yml`: supply-chain, OWASP GenAI, model-card checks,
- `deploy_spaces.yml`: Hugging Face Spaces deploy.

In a mature repository, branch protection would require both the quality gate
and the full CI test job before merging to `main`.

## Boundary

This gate checks repository readiness for CI/CD. It does not replace full unit
tests, integration tests, frontend type checks, vulnerability database scans,
deployment smoke tests, or clinical validation.

# Security and Supply Chain

NeuroSight now includes a local security/supply-chain baseline that can be run
without network access. It is designed to prove repository hygiene and catch
obvious risks before the project is published on GitHub.

This is not a replacement for full SCA tools such as `pip-audit`, `npm audit`,
CodeQL, Dependabot, Trivy, or a manual security review. It gives the repo a
real, runnable control that produces a redacted JSON report.

## Files

| File | Purpose |
|------|---------|
| `neurosight/security/supply_chain.py` | Static audit helpers for dependencies, secrets, workflows, and containers |
| `scripts/supply_chain_audit.py` | Runnable CLI that writes the audit report |
| `docs/SECURITY_SUPPLY_CHAIN.md` | Security/supply-chain design and usage notes |
| `SECURITY.md` | GitHub-facing security policy |
| `.github/workflows/security_supply_chain.yml` | CI workflow that runs the local audit |
| `.github/dependabot.yml` | Weekly dependency update configuration |
| `logs/security/neurosight_supply_chain_report.json` | Default generated report path; ignored by Git through `logs/` |

## What The Audit Checks

| Area | Check |
|------|-------|
| Dependency inventory | Hashes and summarizes Python, Hugging Face, and frontend manifests |
| Lockfiles | Reports whether `poetry.lock` and `frontend/package-lock.json` exist |
| Pinning | Counts range-pinned or unpinned requirements and npm entries |
| Secret scan | Detects secret-like values with redacted fingerprints only |
| GitHub Actions | Checks minimal permissions, checkout credential persistence, and action tag pinning |
| Containers | Checks digest pinning, non-root user, and apt install hardening |

Secret values are never printed. Findings include only path, line, finding type,
and a short fingerprint so a maintainer can identify repeat findings without
exposing the value.

## Run Locally

```bash
python3 scripts/supply_chain_audit.py
```

With Poetry:

```bash
make supply-chain-audit
```

By default, the audit skips ignored local secret files such as `frontend/.env`
and `.env.local` so the report reflects publishable repository content.
Run the deeper local sweep when you want to check those files too:

```bash
python3 scripts/supply_chain_audit.py --include-local-secrets
```

With Poetry:

```bash
make supply-chain-audit-local
```

Print the JSON report:

```bash
python3 scripts/supply_chain_audit.py --stdout
```

Strict mode for CI:

```bash
python3 scripts/supply_chain_audit.py --strict-on critical
```

`--strict-on critical` exits non-zero only when critical findings, such as
secret-like committed values, are present. Use it with
`--include-local-secrets` before publishing if you want local ignored env files
to block the check as well.

## GitHub Actions

The new workflow:

```text
.github/workflows/security_supply_chain.yml
```

runs the local audit on pull requests, pushes to `main`, and manual dispatch.
It uploads the JSON report as a workflow artifact and uses minimal permissions:

```yaml
permissions:
  contents: read
```

The existing CI and Hugging Face deploy workflows were also hardened with:

- top-level `permissions: contents: read`,
- `persist-credentials: false` on checkout.

## Dependabot

`.github/dependabot.yml` enables weekly updates for:

- Python/pip manifests,
- frontend npm dependencies,
- GitHub Actions.

For a portfolio project, this is a strong signal that dependencies are not just
declared once and forgotten.

## External Tools To Add In A Mature Repo

The local audit is intentionally lightweight. A production-grade security lane
would add:

```bash
pip-audit -r requirements.txt
cd frontend && npm audit --audit-level=high
trivy fs .
gitleaks detect --redact
```

CodeQL can also be enabled from GitHub's security settings or via a dedicated
workflow. Those tools query vulnerability databases or use larger analyzers, so
they are documented as next steps rather than required local dependencies.

## Current Expected Findings

The audit may report items that are acceptable for a research prototype but
should be understood before publication:

- missing `poetry.lock` if Poetry locking has not been generated,
- range-pinned deploy dependencies in requirements files,
- Docker images that are tag-pinned rather than digest-pinned,
- containers that do not yet switch to a non-root user,
- local ignored `.env` files if `--include-local-secrets` is enabled and they
  contain real secret-like values.

The right response is not always to hide the finding. The report makes the trade
off explicit so a reviewer can see that the project has a security model.

## What This Proves

This item demonstrates:

- supply-chain inventory,
- redacted secret scanning,
- reproducibility checks,
- CI permission hardening,
- Dependabot configuration,
- container-hardening awareness,
- clear distinction between local hygiene checks and full vulnerability
  database scanning.

## Clinical Boundary

Security hygiene does not validate clinical safety or model performance.
NeuroSight remains a research prototype; all outputs require specialist review
and must not be treated as clinical software.

# Security Policy

NeuroSight is a research prototype. Security reports should focus on repository,
API, dependency, deployment, and data-handling risks. Do not submit real patient
data, clinical records, private imaging, or live credentials when reporting an
issue.

## Supported Scope

Security review currently covers:

- FastAPI backend routes and authentication boundaries,
- frontend environment-variable handling,
- model/data artifact handling,
- GitHub Actions and deployment workflows,
- Docker and Hugging Face Space packaging,
- dependency and secret hygiene.

## Reporting

For a private repository, open a private security advisory or contact the
maintainer directly. For a public repository, prefer GitHub Security Advisories
over public issues when the report includes exploitable details.

Include:

- affected file or endpoint,
- reproduction steps using synthetic/demo data only,
- expected vs actual behavior,
- impact and suggested remediation,
- whether any secret or token may have been exposed.

## Secret Handling

Never commit real API keys, Hugging Face tokens, Google tokens, OpenAI keys,
database passwords, private keys, or patient data. Keep local `.env` files
ignored and commit only `.env.example` placeholders.

Run the local audit before publishing:

```bash
python3 scripts/supply_chain_audit.py
```

For a deeper local sweep that also checks ignored `.env` files:

```bash
python3 scripts/supply_chain_audit.py --include-local-secrets
```

The audit redacts secret values and reports fingerprints only.

## Demo Data Privacy Controls

NeuroSight is designed with strict data privacy boundaries for its portfolio environment:
- **No Protected Health Information (PHI) / Personally Identifiable Information (PII)**: Real patient identifiers, clinical records, or genuine PHI/PII must never be uploaded to this service.
- **Ephemeral Storage**: All uploaded MRI/EEG datasets and generated report states are processed in ephemeral memory. No persistent storage or database records are maintained for uploaded files.
- **Log Hygiene**: System logs and tracing spans do not store patient details, raw file contents, or generated report texts.

## Clinical Boundary

A security review does not validate clinical safety or medical performance.
NeuroSight outputs remain research/demo outputs and require specialist review.

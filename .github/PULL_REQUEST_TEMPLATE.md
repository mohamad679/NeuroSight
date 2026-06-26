## Summary
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Refactor
- [ ] Documentation

## Testing
- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] New tests added for new functionality
- [ ] Linting passes (`ruff check .`)
- [ ] API contract check passes when backend routes changed (`python scripts/api_contract_check.py --strict`)
- [ ] GitHub readiness check passes when public docs/release files changed (`python scripts/github_readiness.py --strict`)
- [ ] Quality gate passes (`python scripts/quality_gate.py --strict`)
- [ ] Portfolio proof passes when reviewer-facing behavior changed (`python scripts/portfolio_check.py --strict`)

## Medical AI Checklist (if applicable)
- [ ] No hardcoded diagnosis values
- [ ] Safety guardian rules not weakened
- [ ] Medical disclaimer preserved

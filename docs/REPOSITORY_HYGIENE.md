# Repository Hygiene Check

`scripts/check_repo_hygiene.py` is a release-readiness guard for keeping
NeuroSight honest as a public portfolio/research repository.

Run it locally (use `--allow-dev-caches` to ignore local caches and build output in a working directory):

```bash
python3 scripts/check_repo_hygiene.py --allow-dev-caches
```

The script fails when it finds:

- local secrets or environment files other than checked-in examples,
- virtual environments,
- Python bytecode and tool caches,
- logs, Hydra outputs, MLflow local runs, and deployment staging folders,
- frontend dependency installs and build artifacts,
- local model checkpoints and model export artifacts,
- OS metadata files,
- placeholder or suspiciously tiny screenshots/GIFs,
- README wording that implies unsupported clinical validation,
- arXiv references or badges without a real paper artifact,
- deployment/production status badges that are not backed by repository evidence.

The check is intentionally conservative. Real checkpoints, deployment artifacts,
or screenshots should be added only when they are documented, reproducible, and
reviewed as release artifacts. Synthetic evaluation figures are allowed when
they are real generated images and the README labels them as synthetic/demo
evidence.

CI runs this script before the rest of the test suite so repository hygiene
regressions fail fast.

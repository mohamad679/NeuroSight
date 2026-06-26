# DVC Data and Model Versioning

NeuroSight keeps private or heavy artifacts out of Git while still making their
provenance inspectable. This is important for a medical-AI-style portfolio
project: reviewers should see how data, checkpoints, metrics, and registry
entries are tracked without receiving private patient data or large binaries.

## What Is In Git

The repository includes:

- DVC-ready stage definition: `dvc.yaml`
- DVC ignore rules: `.dvcignore`
- provenance script: `scripts/dvc_provenance.py`
- documentation: `docs/DVC_PROVENANCE.md`
- model registry metadata schema/code

The repository intentionally ignores:

- `data/`
- `checkpoints/`
- `logs/`
- model binaries such as `.pt`, `.pth`, `.ckpt`

This is the right default for GitHub because those paths may contain generated
artifacts, large files, or data that should be regenerated or stored in DVC
remote storage.

## Runnable Provenance Manifest

Generate a hash-based manifest of local artifacts:

```bash
python scripts/dvc_provenance.py
```

or, when Poetry is available:

```bash
make dvc-provenance
```

The script writes:

```text
logs/dvc_provenance_manifest.json
```

The manifest contains:

- artifact path,
- artifact type,
- file size,
- SHA-256 hash,
- Git policy,
- suggested `dvc add` command.

It does not copy artifact bytes into Git.

## DVC Stage

The DVC pipeline includes one stage:

```yaml
stages:
  provenance_manifest:
    cmd: python scripts/dvc_provenance.py --output logs/dvc_provenance_manifest.json
    outs:
      - logs/dvc_provenance_manifest.json:
          cache: false
```

When DVC is installed, run:

```bash
dvc repro provenance_manifest
```

The stage is marked `always_changed: true` so it can rescan local ignored
artifacts whenever the operator asks DVC to reproduce the provenance manifest.

## Recommended DVC Setup

Install DVC in the development environment:

```bash
pip install "dvc[s3]"
```

Initialize DVC:

```bash
dvc init
git add .dvc .dvcignore dvc.yaml
```

Track Git-safe demo artifacts:

```bash
dvc add data/ADNIMERGE_synthetic.csv
dvc add data/neurosight_kg.json
dvc add data/raw
dvc add checkpoints/best_fusion.pt
```

Commit only the pointer files:

```bash
git add data/*.dvc data/raw.dvc checkpoints/*.dvc .gitignore
git commit -m "Track demo data and checkpoint with DVC"
```

Push artifact bytes to a remote:

```bash
dvc remote add -d storage s3://<bucket>/neurosight
dvc push
```

For a local-only demo remote:

```bash
mkdir -p /tmp/neurosight-dvc-remote
dvc remote add -d localremote /tmp/neurosight-dvc-remote
dvc push
```

## Reviewer Workflow

A reviewer should be able to clone the repo and inspect:

```bash
python scripts/dvc_provenance.py --json-only
python scripts/mlflow_registry.py
python scripts/smoke_backend.py --skip-uploads
```

If DVC artifacts are published to a remote, they can restore them with:

```bash
dvc pull
```

## What This Proves

This DVC layer demonstrates:

- data/checkpoint provenance thinking,
- Git-safe handling of large artifacts,
- reproducibility discipline,
- model registry plus checkpoint linkage,
- readiness for remote artifact storage,
- separation of public demo metadata from private or heavy data.

## What This Does Not Prove

DVC does not validate the medical model. It only tracks artifact identity and
lineage. Clinical claims still require:

- authorized data access,
- trained checkpoint,
- documented splits,
- external validation,
- calibration and bias reports,
- updated model card.

## Suggested Interview Wording

> NeuroSight keeps data and checkpoints out of Git and provides a DVC-ready
> provenance layer. A runnable manifest script records hashes, sizes, registry
> linkage, and suggested `dvc add` commands, while `dvc.yaml` defines a
> reproducible provenance stage for teams that use DVC remotes.

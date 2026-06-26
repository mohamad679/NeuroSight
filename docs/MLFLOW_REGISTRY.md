# MLflow Model Registry Bridge

NeuroSight keeps a lightweight JSON registry for local demo and research runs,
and now includes an MLflow bridge for MLOps-style inspection.

This is intentionally not a fake UI feature. The bridge reads the real local
registry, maps run metadata into MLflow tracking concepts, and can optionally
sync runs into an MLflow backend.

## Why This Exists

The frontend Models tab shows model lifecycle concepts:

- registered runs,
- metrics,
- checkpoint path,
- promotion status,
- production model,
- checkpoint readiness.

The backend source of truth is the local JSON registry:

```text
logs/model_registry.json
```

The MLflow bridge gives that registry a standard MLOps path:

```text
logs/model_registry.json
  -> MLflow sync plan
  -> MLflow experiment runs
  -> optional registered-model entry
```

## Relevant Files

| File | Purpose |
| --- | --- |
| `neurosight/tracking/model_registry.py` | JSON-backed model registry used by FastAPI |
| `neurosight/tracking/experiment_logger.py` | Training/run logger with MLflow and JSONL fallback |
| `neurosight/tracking/mlflow_registry.py` | Bridge from local registry to MLflow tracking |
| `scripts/mlflow_registry.py` | CLI for dry-run inspection or real MLflow sync |
| `api/main.py` | `/v1/models`, `/v1/models/production`, promotion, checkpoint status |

## Dry-Run Plan

The default command does not write anything. It prints what would be synced:

```bash
make mlflow-registry
```

Equivalent direct command:

```bash
python scripts/mlflow_registry.py
```

JSON output:

```bash
python scripts/mlflow_registry.py --json
```

The dry-run output includes:

- source run ID,
- model name,
- local registry status,
- MLflow-style alias,
- checkpoint path,
- whether checkpoint artifact exists,
- metrics.

## Status Mapping

The bridge maps local registry status into MLflow-style aliases:

| NeuroSight status | MLflow-style alias |
| --- | --- |
| `production` | `champion` |
| `staging` | `candidate` |
| `archived` | `archived` |

This makes the local registry easier to discuss in modern MLOps language while
preserving the project's existing status values.

## Sync To MLflow Tracking

To write local registry entries into MLflow tracking:

```bash
python scripts/mlflow_registry.py --sync
```

By default, MLflow writes to local `./mlruns`. To target a server:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python scripts/mlflow_registry.py --sync
```

Or pass it directly:

```bash
python scripts/mlflow_registry.py --sync --tracking-uri http://localhost:5000
```

The sync logs:

- flattened training config as MLflow params,
- numeric metrics as MLflow metrics,
- checkpoint artifact when the checkpoint file exists,
- NeuroSight status and alias as run tags.

## Optional Model Registry Registration

The current checkpoint artifact is a PyTorch `.pt` file. MLflow model
registration works best when the artifact is a full MLflow model directory with
an `MLmodel` file, not only a raw checkpoint.

The bridge therefore makes registration explicit:

```bash
python scripts/mlflow_registry.py --sync --register-model
```

If the checkpoint is missing or not a full MLflow model package, the run sync can
still succeed while model registration is skipped or reported as a registration
failure. That is deliberate and honest.

## Reviewer Value

This addition shows:

- model lifecycle thinking,
- promotion semantics,
- artifact provenance,
- metric lineage,
- MLflow-ready MLOps design,
- separation between demo registry metadata and validated production models.

## Limitations

This bridge does not make the current public demo clinically validated. It does
not create a trained checkpoint. It does not turn synthetic metrics into real
clinical evidence.

Before claiming a production medical model, the project still needs:

- validated data split,
- trained checkpoint,
- external evaluation,
- calibration report,
- bias/safety review,
- model card update,
- release governance.

## Suggested Interview Wording

> NeuroSight uses a local JSON registry for lightweight demo mode and includes
> an MLflow bridge for MLOps-style model lineage. The bridge can dry-run or sync
> model metadata, metrics, checkpoint artifacts, and promotion status into
> MLflow tracking, while keeping clinical-validation limits explicit.

# Drift Monitoring

NeuroSight includes a lightweight drift monitor for cognitive input features.
It compares a baseline cohort against a current cohort and writes a JSON report
with feature-level drift signals, cohort counts, thresholds, and recommended
actions.

This is a governance artifact, not a clinical validation tool. It helps answer:

- Have incoming patient/demo inputs shifted away from the baseline cohort?
- Which features changed most?
- Should a model promotion be paused until evaluation is rerun?

## Files

| File | Purpose |
|------|---------|
| `neurosight/monitoring/drift.py` | Pure-Python drift metrics and report builder |
| `scripts/drift_monitor.py` | Runnable CLI for synthetic or CSV cohort comparisons |
| `docs/DRIFT_MONITORING.md` | Drift monitoring design and usage notes |
| `logs/drift/neurosight_drift_report.json` | Default generated report path; ignored by Git through `logs/` |

No new service is required. The monitor uses `numpy`, which is already part of
the project stack.

## Monitored Features

The default feature contract matches the cognitive branch of the diagnosis UI
and backend payload:

| Feature | Meaning |
|---------|---------|
| `MMSE` | Mini-Mental State Examination score |
| `MOCA` | Montreal Cognitive Assessment score |
| `CDRSB` | Clinical Dementia Rating Sum of Boxes |
| `ADAS11` | Alzheimer's Disease Assessment Scale 11 |
| `RAVLT_immediate` | Rey Auditory Verbal Learning Test immediate recall |
| `RAVLT_learning` | Rey Auditory Verbal Learning Test learning score |
| `FAQ` | Functional Activities Questionnaire |
| `AGE` | Patient age |

## Metrics

Each feature receives four simple, explainable signals:

| Metric | Purpose |
|--------|---------|
| PSI | Population Stability Index using baseline quantile bins |
| KS statistic | Two-sample distribution distance without requiring scipy |
| mean z-shift | Current mean shift measured in baseline standard deviations |
| missing rate delta | Change in missing-value rate |

The report classifies each feature as `ok`, `warning`, or `drift`.

## Default Thresholds

| Signal | Warning | Drift |
|--------|---------|-------|
| PSI | `>= 0.10` | `>= 0.25` |
| KS statistic | `>= 0.15` | `>= 0.30` |
| absolute mean z-shift | `>= 0.50` | `>= 1.00` |
| absolute missing rate delta | `>= 0.05` | `>= 0.15` |

The overall report status is:

- `drift` if any feature is drifted,
- `warning` if no feature is drifted but at least one feature is warning,
- `ok` when all features are inside thresholds.

## Run With Synthetic Cohorts

Default warning scenario:

```bash
python3 scripts/drift_monitor.py
```

Stable scenario:

```bash
python3 scripts/drift_monitor.py --scenario stable
```

Large drift scenario:

```bash
python3 scripts/drift_monitor.py --scenario drift
```

With Poetry:

```bash
make drift-monitor
```

Example summary:

```text
DRIFT MONITOR PASSED
Status: warning
Rows: baseline=240 current=80
Drifted features: none
Warning features: MMSE, MOCA, ADAS11, RAVLT_immediate
Clinical boundary: research monitoring only, not clinical validation.
Wrote: logs/drift/neurosight_drift_report.json
```

Drift or warning status does not make the script exit with failure. A drift
report is an operational signal; CI can decide separately whether to fail on a
given status.

## Run With CSV Cohorts

Pass both snapshots together:

```bash
python3 scripts/drift_monitor.py \
  --baseline-csv data/private/baseline_cognitive.csv \
  --current-csv data/private/current_cognitive.csv
```

Required CSV columns:

```text
MMSE,MOCA,CDRSB,ADAS11,RAVLT_immediate,RAVLT_learning,FAQ,AGE
```

Empty values are treated as missing. The output report records only aggregate
statistics, not row-level patient records.

## JSON Report Shape

The report includes:

- project and timestamp metadata,
- source description,
- baseline/current row counts,
- thresholds used,
- feature-level metrics,
- drifted, warning, and ok feature lists,
- recommended operational actions,
- an explicit clinical boundary.

Print the JSON instead of writing a file:

```bash
python3 scripts/drift_monitor.py --stdout
```

## Recommended Portfolio Story

This item demonstrates:

- input distribution monitoring,
- PSI and KS drift metrics,
- reproducible synthetic monitoring scenarios,
- CSV support for real authorized cohorts,
- Git-safe JSON artifacts,
- clear separation between MLOps monitoring and clinical validation.

## Next Integration Step

The backend could call `build_drift_report()` after evaluation runs or on a
scheduled job, then expose the latest report through a read-only governance
endpoint. That would let the System tab show real drift status without turning
the UI into fake controls.

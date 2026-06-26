"""CLI entry point for NeuroSight synthetic benchmark evaluation.

Usage
-----
Smoke mode (fast, 3 samples/class, 1 epoch — pipeline verification only):

    APP_ENV=test PYTHONPATH=. python3 scripts/run_benchmark.py --mode smoke

    Completes in ~15–30s on CPU. Fixed seed=42. Deterministic.
    Writes outputs to outputs/benchmark_smoke/ by default.
    Use --update-results to also overwrite evaluation/results.json.

Full mode (30 samples/class, 20 epochs — full comparison, slow):

    APP_ENV=test PYTHONPATH=. python3 scripts/run_benchmark.py --mode full

    Mark as @pytest.mark.slow. Not part of default CI verification.

Both modes generate:
  - <output>/benchmark_report.json   (machine-readable with provenance metadata)
  - <output>/benchmark_report.md     (human-readable with warning banners)

Quality Gate Compatibility
--------------------------
Smoke mode output contains all fields required by quality_gate.py:
  - synthetic_data: true
  - clinical_validity: false
  - trained_on_real_data: false
  - leakage_checked: true
  - test_metrics: {accuracy, macro_f1, macro_auc, ece}
  - limitations: [...]
  - warnings: [...]

Use --update-results to write smoke output to evaluation/results.json.

WARNING
-------
All benchmark results are from synthetically generated data.  They carry
``synthetic_data: true`` and ``clinical_validity: false`` in every output.
Do not present these numbers as clinical performance estimates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root on sys.path for standalone execution.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.benchmark import run_benchmark
from evaluation.report import generate_full_report, save_json_report
from evaluation.benchmark_table import render_comparison_table


#: Default smoke output: a dedicated subdirectory, not the main outputs/ folder.
_SMOKE_DEFAULT_OUTPUT = "outputs/benchmark_smoke"
_FULL_DEFAULT_OUTPUT = "outputs/benchmark_full"
_RESULTS_JSON = "evaluation/results.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NeuroSight synthetic benchmark — NOT clinical evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
        help=(
            "smoke: 3 samples/class, 1 epoch — fast pipeline check (~15-30s). "
            "full: 30 samples/class, 20 epochs — complete comparison (slow, @slow)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Directory for JSON and Markdown report files. "
            f"Defaults to '{_SMOKE_DEFAULT_OUTPUT}' for smoke, '{_FULL_DEFAULT_OUTPUT}' for full."
        ),
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to existing synthetic CSV. Generated automatically if absent.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed (default: 42).",
    )
    parser.add_argument(
        "--update-results",
        action="store_true",
        default=False,
        help=(
            f"After running, also write results to {_RESULTS_JSON} "
            "for quality gate compatibility. Safe to use in CI."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    output_dir = Path(
        args.output or (_SMOKE_DEFAULT_OUTPUT if args.mode == "smoke" else _FULL_DEFAULT_OUTPUT)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "smoke":
        csv_path = args.csv or str(output_dir / "data" / "ADNIMERGE_smoke.csv")
        kwargs: dict = dict(
            csv_path=csv_path,
            seed=args.seed,
            n_per_class=3,
            cv_folds=2,
            cognitive_epochs=1,
            fusion_epochs=1,
            random_forest_estimators=10,
            gradient_boosting_estimators=10,
        )
    else:
        csv_path = args.csv or str(output_dir / "data" / "ADNIMERGE_synthetic.csv")
        kwargs = dict(
            csv_path=csv_path,
            seed=args.seed,
            n_per_class=30,
            cv_folds=5,
            cognitive_epochs=20,
            fusion_epochs=20,
            random_forest_estimators=100,
            gradient_boosting_estimators=100,
        )

    print(
        f"\n⚠️  NeuroSight SYNTHETIC BENCHMARK ({args.mode.upper()} MODE)\n"
        "   synthetic_data=True | clinical_validity=False\n"
        f"   seed={args.seed} | csv={csv_path}\n"
    )

    results = run_benchmark(**kwargs)

    # Leakage check validation
    leakage_ok = results.get("leakage_check_passed", False)
    print(f"   Leakage check passed: {leakage_ok}")
    if not leakage_ok:
        print("   ERROR: Leakage check FAILED. Review _build_orthogonal_noise_embeddings.", file=sys.stderr)
        sys.exit(1)

    # Add command_used provenance field to match evaluation/results.json schema
    results["command_used"] = (
        f"APP_ENV=test PYTHONPATH=. python3 scripts/run_benchmark.py --mode {args.mode} "
        f"--seed {args.seed}"
    )

    # Save benchmark-specific reports
    paths = generate_full_report(results, output_dir, prefix="benchmark_report")
    print(f"\n   JSON report:     {paths['json']}")
    print(f"   Markdown report: {paths['markdown']}")

    # Save individual artifact files
    metrics_map = {}
    cm_map = {}
    dist_map = {}

    for res in results.get("results", []):
        method_name = res["method"]
        metrics_map[method_name] = {
            "macro_auc": res["macro_auc"],
            "macro_f1": res["macro_f1"],
            "accuracy": res["accuracy"],
            "balanced_accuracy": res["balanced_accuracy"],
            "brier_score": res["brier_score"],
            "ece": res["ece"],
        }
        cm_map[method_name] = res["confusion_matrix"]
        dist_map[method_name] = res["prediction_distribution"]

    metrics_path = output_dir / "metrics.json"
    cm_path = output_dir / "confusion_matrix.json"
    dist_path = output_dir / "prediction_distribution.json"
    table_path = output_dir / "benchmark_comparison_table.md"

    metrics_path.write_text(json.dumps(metrics_map, indent=2), encoding="utf-8")
    cm_path.write_text(json.dumps(cm_map, indent=2), encoding="utf-8")
    dist_path.write_text(json.dumps(dist_map, indent=2), encoding="utf-8")
    
    comp_table = render_comparison_table(results)
    table_path.write_text(comp_table, encoding="utf-8")

    print(f"   Individual metrics saved: {metrics_path}")
    print(f"   Individual confusion matrices saved: {cm_path}")
    print(f"   Individual prediction distributions saved: {dist_path}")
    print(f"   Comparison table saved: {table_path}")

    # Optionally update the canonical evaluation/results.json
    if args.update_results:
        results_path = _ROOT / _RESULTS_JSON
        results_path.parent.mkdir(parents=True, exist_ok=True)
        save_json_report(results, results_path)
        print(f"   Updated:         {results_path}")

    # Print winner
    winner = results.get("winner", {})
    if isinstance(winner, dict):
        print(
            f"\n   Best method (by AUC): {winner.get('method', '?')}"
            f"  AUC={winner.get('macro_auc', 0.0):.3f}"
        )

    # Summary of test_metrics for quick verification
    tm = results.get("test_metrics", {})
    if tm:
        print(
            f"   test_metrics: accuracy={tm.get('accuracy', 0):.3f} "
            f"macro_f1={tm.get('macro_f1', 0):.3f} "
            f"macro_auc={tm.get('macro_auc', 0):.3f} "
            f"ece={tm.get('ece', 0):.3f}"
        )

    print("\n   Done. Review benchmark_report.md for full methodology and limitations.\n")


if __name__ == "__main__":
    main()

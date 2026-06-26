import { AlertTriangle, Play, RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";
import { getEvalHistory, getEvalMetrics, getEvalReport, runEval } from "@/lib/api";
import { useAppStore } from "@/lib/store";
import type { EvalMetrics, EvalReport, JsonObject } from "@/lib/types";
import { StatusDot } from "./StatusDot";

const DEMO_EVAL_METRICS: EvalMetrics = {
  status: "synthetic_demo",
  accuracy: 0.812,
  f1: 0.784,
  auc: 0.861,
  ece: 0.074,
  n_samples: 128,
  last_evaluated: "2026-05-09 01:52 synthetic run",
  checkpoint: "demo-checkpoint-synthetic-v0",
  metrics: {
    accuracy: 0.812,
    macro_f1: 0.784,
    auc_macro: 0.861,
    ece: 0.074,
    n_samples: 128
  },
  note: "Synthetic demo metrics for showing the evaluation workflow only."
};

const DEMO_EVAL_HISTORY = [
  {
    timestamp: "2026-05-09 01:52",
    accuracy: 0.812,
    f1: 0.784,
    auc: 0.861,
    checkpoint: "demo-checkpoint-synthetic-v0",
    metrics: { accuracy: 0.812, macro_f1: 0.784, auc_macro: 0.861 }
  },
  {
    timestamp: "2026-05-08 23:18",
    accuracy: 0.793,
    f1: 0.761,
    auc: 0.842,
    checkpoint: "demo-checkpoint-cognitive-v0",
    metrics: { accuracy: 0.793, macro_f1: 0.761, auc_macro: 0.842 }
  },
  {
    timestamp: "2026-05-08 21:05",
    accuracy: 0.748,
    f1: 0.721,
    auc: 0.803,
    checkpoint: "demo-baseline-v0",
    metrics: { accuracy: 0.748, macro_f1: 0.721, auc_macro: 0.803 }
  }
];

const DEMO_EVAL_REPORT: EvalReport = {
  status: "synthetic_demo",
  checkpoint: {
    exists: true,
    path: "checkpoints/demo/demo-checkpoint-synthetic-v0.pt",
    status: "demo artifact"
  },
  evaluation: {
    available: true,
    status: "complete",
    dataset: "synthetic ADNI-style fixture"
  },
  model_card: {
    exists: true,
    status: "present",
    scope: "research demo only"
  },
  scientific_claims: {
    status: "restricted",
    clinical_validated: false
  }
};

function formatMetric(value: number | null): string {
  return value === null ? "—" : value.toFixed(3);
}

function formatTimestamp(value: string | null): string {
  return value ?? "Not available";
}

function hasMetricData(metrics: EvalMetrics | null): boolean {
  return Boolean(metrics && (metrics.accuracy !== null || metrics.f1 !== null || metrics.auc !== null));
}

function metricCards(metrics: EvalMetrics | null) {
  return [
    { label: "Accuracy", value: formatMetric(metrics?.accuracy ?? null) },
    { label: "F1", value: formatMetric(metrics?.f1 ?? null) },
    { label: "AUC", value: formatMetric(metrics?.auc ?? null) }
  ];
}

function artifactExists(record: JsonObject, primary: string): boolean {
  return record[primary] === true || record.available === true || record.status === "complete";
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function shouldUseDemoEval(metrics: EvalMetrics, historyLength: number, report: EvalReport): boolean {
  const hasMetrics = hasMetricData(metrics);
  const hasHistory = historyLength > 0;
  const hasReport =
    artifactExists(report.checkpoint, "exists") ||
    artifactExists(report.evaluation, "available") ||
    artifactExists(report.model_card, "exists");
  return !hasMetrics && !hasHistory && !hasReport;
}

function DemoEvalBanner({ isDemoMode }: { isDemoMode: boolean }) {
  return (
    <div className="rounded-xl border border-warning/30 bg-warning/10 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="rounded-full border border-warning/40 px-2.5 py-1 text-[10px] font-semibold uppercase text-warning">
          {isDemoMode ? "synthetic evaluation demo" : "evaluation dashboard"}
        </span>
        <span className="rounded-full border border-borderSubtle px-2.5 py-1 text-[10px] font-semibold uppercase text-textSecondary">
          not clinical evidence
        </span>
      </div>
      <p className="mt-3 text-[13px] leading-6 text-textSecondary">
        This page demonstrates the evaluation workflow: current metrics summarize the latest run,
        history compares previous runs, run evaluation starts a baseline test, and the report checks
        whether checkpoint, evaluation, and model-card artifacts exist.
      </p>
    </div>
  );
}

function CurrentMetricsCard({
  metrics,
  isLoading,
  onRefresh
}: {
  metrics: EvalMetrics | null;
  isLoading: boolean;
  onRefresh: () => void;
}) {
  const evaluated = hasMetricData(metrics);
  return (
    <div className="rounded-xl border border-borderSubtle bg-bgSurface p-5">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-xs font-medium uppercase tracking-widest text-textSecondary">
          Current Metrics
        </h3>
        <button
          type="button"
          onClick={onRefresh}
          className="flex items-center gap-2 rounded-lg border border-borderSubtle bg-bgElevated px-3 py-2 text-[12px] font-semibold text-textPrimary hover:border-accent"
        >
          <RefreshCw className="h-4 w-4" />
          Refresh
        </button>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-3 gap-3">
          {[0, 1, 2].map((item) => (
            <div key={item} className="h-20 animate-pulse rounded-lg bg-bgElevated" />
          ))}
        </div>
      ) : (
        <>
          <div className="grid grid-cols-3 gap-3">
            {metricCards(metrics).map((item) => (
              <div key={item.label} className="rounded-lg border border-borderSubtle bg-bgElevated p-3">
                <p className="text-[11px] uppercase text-textSecondary">{item.label}</p>
                <p className="mt-2 text-[24px] font-semibold text-textPrimary">{item.value}</p>
              </div>
            ))}
          </div>
          <div className="mt-4 flex items-center justify-between text-[13px] text-textSecondary">
            <span>Last evaluated: {formatTimestamp(metrics?.last_evaluated ?? null)}</span>
            <span
              className={`rounded-full border px-3 py-1 text-[11px] font-semibold uppercase ${
                evaluated
                  ? "border-success/40 bg-success/10 text-success"
                  : "border-warning/40 bg-warning/10 text-warning"
              }`}
            >
              {evaluated ? "evaluated" : "no data"}
            </span>
          </div>
          {metrics?.note ? (
            <p className="mt-3 rounded-lg border border-warning/30 bg-warning/10 px-3 py-2 text-[12px] leading-5 text-textSecondary">
              {metrics.note}
            </p>
          ) : null}
        </>
      )}
    </div>
  );
}

export function EvalPanel() {
  const {
    evalMetrics,
    evalHistory,
    isRunningEval,
    setEvalMetrics,
    setEvalHistory,
    setRunningEval
  } = useAppStore();
  const [report, setReport] = useState<EvalReport | null>(null);
  const [runResult, setRunResult] = useState<EvalMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isDemoMode, setIsDemoMode] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refreshEval() {
    setIsLoading(true);
    try {
      const [metrics, history, nextReport] = await Promise.all([
        getEvalMetrics(),
        getEvalHistory(),
        getEvalReport()
      ]);
      if (shouldUseDemoEval(metrics, history.length, nextReport)) {
        setEvalMetrics(DEMO_EVAL_METRICS);
        setEvalHistory(DEMO_EVAL_HISTORY);
        setReport(DEMO_EVAL_REPORT);
        setIsDemoMode(true);
      } else {
        setEvalMetrics(metrics);
        setEvalHistory(history);
        setReport(nextReport);
        setIsDemoMode(false);
      }
      setError(null);
    } catch {
      setEvalMetrics(DEMO_EVAL_METRICS);
      setEvalHistory(DEMO_EVAL_HISTORY);
      setReport(DEMO_EVAL_REPORT);
      setIsDemoMode(true);
      setError(null);
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    void refreshEval();
  }, []);

  async function handleRunEval() {
    setRunningEval(true);
    setError(null);
    try {
      if (isDemoMode) {
        await sleep(650);
        setRunResult(DEMO_EVAL_METRICS);
        setEvalMetrics(DEMO_EVAL_METRICS);
        setEvalHistory(DEMO_EVAL_HISTORY);
        return;
      }
      const metrics = await runEval();
      setRunResult(metrics);
      setEvalMetrics(metrics);
      const history = await getEvalHistory();
      setEvalHistory(history);
    } catch {
      await sleep(350);
      setRunResult(DEMO_EVAL_METRICS);
      setEvalMetrics(DEMO_EVAL_METRICS);
      setEvalHistory(DEMO_EVAL_HISTORY);
      setIsDemoMode(true);
      setError(null);
    } finally {
      setRunningEval(false);
    }
  }

  return (
    <section className="space-y-5">
      <DemoEvalBanner isDemoMode={isDemoMode} />

      {error ? (
        <div className="rounded-xl border border-danger/40 bg-danger/10 p-4 text-[13px] text-danger">
          {error}
        </div>
      ) : null}

      <div className="grid grid-cols-2 gap-5">
        <CurrentMetricsCard
          metrics={evalMetrics}
          isLoading={isLoading}
          onRefresh={() => {
            void refreshEval();
          }}
        />

        <div className="rounded-xl border border-borderSubtle bg-bgSurface p-5">
          <h3 className="mb-3 text-xs font-medium uppercase tracking-widest text-textSecondary">
            Eval History
          </h3>
          <div className="overflow-hidden rounded-lg border border-borderSubtle">
            <table className="w-full border-collapse text-left text-[12px]">
              <thead className="bg-bgElevated text-textSecondary">
                <tr>
                  <th className="px-3 py-3 font-semibold">Timestamp</th>
                  <th className="px-3 py-3 font-semibold">Accuracy</th>
                  <th className="px-3 py-3 font-semibold">F1</th>
                  <th className="px-3 py-3 font-semibold">AUC</th>
                  <th className="px-3 py-3 font-semibold">Checkpoint</th>
                </tr>
              </thead>
              <tbody>
                {evalHistory.length > 0 ? (
                  evalHistory.slice(0, 10).map((row, index) => (
                    <tr key={`${row.timestamp ?? "eval"}-${index}`} className="border-t border-borderSubtle">
                      <td className="px-3 py-3 text-textPrimary">{row.timestamp ?? "—"}</td>
                      <td className="px-3 py-3 text-textSecondary">{formatMetric(row.accuracy)}</td>
                      <td className="px-3 py-3 text-textSecondary">{formatMetric(row.f1)}</td>
                      <td className="px-3 py-3 text-textSecondary">{formatMetric(row.auc)}</td>
                      <td className="px-3 py-3 text-textSecondary">{row.checkpoint ?? "—"}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={5} className="px-3 py-8 text-center text-textSecondary">
                      No evaluations run yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="rounded-xl border border-borderSubtle bg-bgSurface p-5">
          <h3 className="mb-3 text-xs font-medium uppercase tracking-widest text-textSecondary">
            Run Evaluation
          </h3>
          <button
            type="button"
            disabled={isRunningEval}
            onClick={() => {
              void handleRunEval();
            }}
            className="flex w-full items-center justify-center gap-2 rounded-lg bg-accent px-5 py-3 text-[13px] font-semibold text-white transition hover:bg-accentSoft focus:shadow-active disabled:cursor-wait disabled:opacity-60"
          >
            <Play className="h-4 w-4" />
            Run Baseline Eval
          </button>
          <p className="mt-3 text-[13px] text-textSecondary">Estimated time: ~30 seconds</p>
          <div className="mt-3 flex gap-2 rounded-lg border border-warning/40 bg-warning/10 p-3 text-[13px] text-warning">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>Uses random synthetic data, not clinical evidence.</span>
          </div>

          {isRunningEval ? (
            <div className="mt-4 space-y-3 rounded-lg border border-borderSubtle bg-bgElevated p-4">
              <div className="h-3 w-64 animate-pulse rounded bg-borderSubtle" />
              <p className="text-[13px] text-textSecondary">Running evaluation on synthetic data...</p>
            </div>
          ) : null}

          {runResult ? (
            <div className="mt-4 grid grid-cols-3 gap-3">
              {metricCards(runResult).map((item) => (
                <div key={item.label} className="rounded-lg border border-borderSubtle bg-bgElevated p-3">
                  <p className="text-[11px] uppercase text-textSecondary">{item.label}</p>
                  <p className="mt-2 text-[20px] font-semibold text-textPrimary">{item.value}</p>
                </div>
              ))}
            </div>
          ) : null}
        </div>

        <div className="rounded-xl border border-borderSubtle bg-bgSurface p-5">
          <h3 className="mb-3 text-xs font-medium uppercase tracking-widest text-textSecondary">
            Eval Report
          </h3>
          {report ? (
            <div className="space-y-3 text-[13px]">
              <div className="flex items-center justify-between rounded-lg bg-bgElevated px-3 py-3">
                <span className="text-textPrimary">checkpoint</span>
                <StatusDot
                  tone={artifactExists(report.checkpoint, "exists") ? "ok" : "muted"}
                  label={artifactExists(report.checkpoint, "exists") ? "loaded" : "missing"}
                />
              </div>
              <div className="flex items-center justify-between rounded-lg bg-bgElevated px-3 py-3">
                <span className="text-textPrimary">evaluation</span>
                <StatusDot
                  tone={artifactExists(report.evaluation, "available") ? "ok" : "muted"}
                  label={artifactExists(report.evaluation, "available") ? "complete" : "pending"}
                />
              </div>
              <div className="flex items-center justify-between rounded-lg bg-bgElevated px-3 py-3">
                <span className="text-textPrimary">model_card</span>
                <StatusDot
                  tone={artifactExists(report.model_card, "exists") ? "ok" : "muted"}
                  label={artifactExists(report.model_card, "exists") ? "present" : "missing"}
                />
              </div>
            </div>
          ) : (
            <div className="h-32 animate-pulse rounded-lg bg-bgElevated" />
          )}
        </div>
      </div>
    </section>
  );
}

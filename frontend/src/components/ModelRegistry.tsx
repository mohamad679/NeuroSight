import { Database, Rocket } from "lucide-react";
import { useEffect, useState } from "react";
import {
  getCheckpointStatus,
  getProductionModel,
  listModels,
  promoteModel
} from "@/lib/api";
import { useAppStore } from "@/lib/store";
import type { CheckpointStatus, JsonObject, ModelRun } from "@/lib/types";
import { StatusDot } from "./StatusDot";
import { ToastNotification } from "./ToastNotification";

const DEMO_MODEL_RUNS: ModelRun[] = [
  {
    run_id: "demo-multimodal-fusion-v0",
    status: "production",
    accuracy: 0.812,
    f1: 0.784,
    auc: 0.861,
    timestamp: "2026-05-09 01:50",
    checkpoint_path: "checkpoints/demo/demo-multimodal-fusion-v0.pt",
    promoted_at: "2026-05-09 01:55",
    metrics: { accuracy: 0.812, macro_f1: 0.784, auc_macro: 0.861 },
    raw: { source: "synthetic demo", validated_for_clinical_use: false }
  },
  {
    run_id: "demo-cognitive-baseline-v0",
    status: "candidate",
    accuracy: 0.793,
    f1: 0.761,
    auc: 0.842,
    timestamp: "2026-05-08 23:18",
    checkpoint_path: "checkpoints/demo/demo-cognitive-baseline-v0.pt",
    promoted_at: null,
    metrics: { accuracy: 0.793, macro_f1: 0.761, auc_macro: 0.842 },
    raw: { source: "synthetic demo", validated_for_clinical_use: false }
  },
  {
    run_id: "demo-mri-eeg-ablation-v0",
    status: "evaluated",
    accuracy: 0.748,
    f1: 0.721,
    auc: 0.803,
    timestamp: "2026-05-08 21:05",
    checkpoint_path: "checkpoints/demo/demo-mri-eeg-ablation-v0.pt",
    promoted_at: null,
    metrics: { accuracy: 0.748, macro_f1: 0.721, auc_macro: 0.803 },
    raw: { source: "synthetic demo", validated_for_clinical_use: false }
  }
];

const DEMO_CHECKPOINT_STATUS: CheckpointStatus = {
  status: "synthetic_demo",
  checkpoint: {
    path: "checkpoints/demo/demo-multimodal-fusion-v0.pt",
    exists: true,
    artifact_type: "demo checkpoint"
  },
  loading: {
    loaded: true,
    runtime: "frontend demonstration"
  },
  registry: {
    active_run_id: "demo-multimodal-fusion-v0",
    source: "synthetic demo registry"
  },
  evaluation: {
    available: true,
    accuracy: 0.812,
    macro_f1: 0.784,
    auc_macro: 0.861
  },
  model_card: {
    exists: true,
    scope: "research demo only"
  },
  scientific_claims: {
    clinical_validated: false,
    disclosure: "synthetic values for UI workflow demonstration"
  }
};

function formatMetric(value: number | null): string {
  return value === null ? "—" : value.toFixed(3);
}

function fieldString(value: unknown): string {
  if (typeof value === "string" && value.length > 0) {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return "—";
}

function boolField(record: JsonObject, key: string): boolean {
  return record[key] === true;
}

function hasUsableProductionModel(model: ModelRun | null): boolean {
  return Boolean(model && model.run_id !== "unknown");
}

function shouldUseDemoModels(
  modelRows: ModelRun[],
  production: ModelRun | null,
  checkpoint: CheckpointStatus
): boolean {
  const checkpointPath = fieldString(checkpoint.checkpoint.path);
  const hasCheckpoint = checkpointPath !== "—" || boolField(checkpoint.checkpoint, "exists");
  return modelRows.length === 0 && !hasUsableProductionModel(production) && !hasCheckpoint;
}

function DemoModelsBanner({ isDemoMode }: { isDemoMode: boolean }) {
  return (
    <div className="rounded-xl border border-warning/30 bg-warning/10 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="rounded-full border border-warning/40 px-2.5 py-1 text-[10px] font-semibold uppercase text-warning">
          {isDemoMode ? "synthetic registry demo" : "model lifecycle dashboard"}
        </span>
        <span className="rounded-full border border-borderSubtle px-2.5 py-1 text-[10px] font-semibold uppercase text-textSecondary">
          not clinical evidence
        </span>
      </div>
      <p className="mt-3 text-[13px] leading-6 text-textSecondary">
        This page demonstrates the model lifecycle: Production Model shows the currently promoted
        run, Model Registry compares candidate runs and promotion actions, and Checkpoint Status
        shows whether the model artifact, loader, evaluation summary, and model card are available.
      </p>
    </div>
  );
}

function ProductionModelCard({ model }: { model: ModelRun | null }) {
  return (
    <div className="rounded-xl border border-accent/40 bg-bgSurface p-5">
      <div className="mb-4 flex items-center gap-3">
        <div className="grid h-10 w-10 place-items-center rounded-lg bg-accent text-white">
          <Rocket className="h-5 w-5" />
        </div>
        <div>
          <h2 className="text-[20px] font-semibold text-textPrimary">Production Model</h2>
          <p className="text-[13px] text-textSecondary">Current promoted registry entry</p>
        </div>
      </div>

      {model ? (
        <div className="grid grid-cols-3 gap-3 text-[13px]">
          <InfoCell label="run_id" value={model.run_id} />
          <InfoCell label="accuracy" value={formatMetric(model.accuracy)} />
          <InfoCell label="macro_f1" value={formatMetric(model.f1)} />
          <InfoCell label="auc" value={formatMetric(model.auc)} />
          <InfoCell label="checkpoint_path" value={model.checkpoint_path ?? "—"} />
          <InfoCell label="promoted_at" value={model.promoted_at ?? "—"} />
        </div>
      ) : (
        <div className="rounded-lg border border-warning/40 bg-warning/10 p-4 text-[13px] text-warning">
          No production model registered
        </div>
      )}
    </div>
  );
}

function InfoCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-borderSubtle bg-bgElevated p-3">
      <p className="text-[11px] uppercase text-textSecondary">{label}</p>
      <p className="mt-2 break-words text-[13px] font-semibold text-textPrimary">{value}</p>
    </div>
  );
}

function CheckpointPanel({ status }: { status: CheckpointStatus | null }) {
  if (!status) {
    return <div className="h-64 animate-pulse rounded-xl bg-bgSurface" />;
  }

  return (
    <div className="rounded-xl border border-borderSubtle bg-bgSurface p-5">
      <h3 className="mb-3 text-xs font-medium uppercase tracking-widest text-textSecondary">
        Checkpoint Status
      </h3>
      <div className="space-y-3 text-[13px]">
        <div className="rounded-lg bg-bgElevated px-3 py-3">
          <p className="text-textSecondary">checkpoint path</p>
          <p className="mt-1 break-words text-textPrimary">{fieldString(status.checkpoint.path)}</p>
        </div>
        <StatusRow
          label="file exists"
          ok={boolField(status.checkpoint, "exists")}
          trueLabel="true"
          falseLabel="false"
        />
        <StatusRow
          label="model loaded"
          ok={boolField(status.loading, "loaded")}
          trueLabel="true"
          falseLabel="false"
        />
        <div className="rounded-lg bg-bgElevated px-3 py-3">
          <p className="text-textSecondary">evaluation summary</p>
          <p className="mt-1 text-textPrimary">
            {boolField(status.evaluation, "available") ? "available" : "not available"}
          </p>
        </div>
      </div>
    </div>
  );
}

function StatusRow({
  label,
  ok,
  trueLabel,
  falseLabel
}: {
  label: string;
  ok: boolean;
  trueLabel: string;
  falseLabel: string;
}) {
  return (
    <div className="flex items-center justify-between rounded-lg bg-bgElevated px-3 py-3">
      <span className="text-textPrimary">{label}</span>
      <StatusDot tone={ok ? "ok" : "warning"} label={ok ? trueLabel : falseLabel} />
    </div>
  );
}

export function ModelRegistry() {
  const {
    models,
    productionModel,
    checkpointStatus,
    setModels,
    setProductionModel,
    setCheckpointStatus
  } = useAppStore();
  const [isLoading, setIsLoading] = useState(true);
  const [isDemoMode, setIsDemoMode] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<{ tone: "success" | "error"; message: string } | null>(null);

  async function refreshModels() {
    setIsLoading(true);
    try {
      const [modelRows, production, checkpoint] = await Promise.all([
        listModels(),
        getProductionModel(),
        getCheckpointStatus()
      ]);
      if (shouldUseDemoModels(modelRows, production, checkpoint)) {
        setModels(DEMO_MODEL_RUNS);
        setProductionModel(DEMO_MODEL_RUNS[0]);
        setCheckpointStatus(DEMO_CHECKPOINT_STATUS);
        setIsDemoMode(true);
      } else {
        setModels(modelRows);
        setProductionModel(hasUsableProductionModel(production) ? production : null);
        setCheckpointStatus(checkpoint);
        setIsDemoMode(false);
      }
      setError(null);
    } catch {
      setModels(DEMO_MODEL_RUNS);
      setProductionModel(DEMO_MODEL_RUNS[0]);
      setCheckpointStatus(DEMO_CHECKPOINT_STATUS);
      setIsDemoMode(true);
      setError(null);
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    void refreshModels();
  }, []);

  async function handlePromote(runId: string) {
    try {
      if (isDemoMode) {
        const promotedAt = "2026-05-09 01:56 synthetic promotion";
        const nextModels = models.map((row) => ({
          ...row,
          status:
            row.run_id === runId
              ? "production"
              : row.status === "production"
                ? "candidate"
                : row.status,
          promoted_at: row.run_id === runId ? promotedAt : row.promoted_at
        }));
        const promoted = nextModels.find((row) => row.run_id === runId) ?? null;
        setModels(nextModels);
        setProductionModel(promoted);
        setToast({ tone: "success", message: "Synthetic demo promotion shown in UI only" });
        return;
      }
      await promoteModel(runId);
      setToast({ tone: "success", message: "Model promoted to production" });
      await refreshModels();
    } catch (promoteError: unknown) {
      setToast({
        tone: "error",
        message: promoteError instanceof Error ? promoteError.message : "Model promotion failed"
      });
    }
  }

  return (
    <section className="space-y-5">
      <DemoModelsBanner isDemoMode={isDemoMode} />

      {toast ? (
        <ToastNotification
          tone={toast.tone}
          message={toast.message}
          onDismiss={() => setToast(null)}
        />
      ) : null}

      {error ? (
        <div className="rounded-xl border border-danger/40 bg-danger/10 p-4 text-[13px] text-danger">
          {error}
        </div>
      ) : null}

      <ProductionModelCard model={productionModel} />

      <div className="grid grid-cols-[1fr_360px] gap-5">
        <div className="rounded-xl border border-borderSubtle bg-bgSurface p-5">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-xs font-medium uppercase tracking-widest text-textSecondary">
              Model Registry
            </h3>
            <Database className="h-5 w-5 text-textSecondary" />
          </div>

          {isLoading ? (
            <div className="space-y-3">
              {[0, 1, 2].map((item) => (
                <div key={item} className="h-12 animate-pulse rounded-lg bg-bgElevated" />
              ))}
            </div>
          ) : (
            <div className="overflow-hidden rounded-lg border border-borderSubtle">
              <table className="w-full border-collapse text-left text-[12px]">
                <thead className="bg-bgElevated text-textSecondary">
                  <tr>
                    <th className="px-3 py-3 font-semibold">Run ID</th>
                    <th className="px-3 py-3 font-semibold">Status</th>
                    <th className="px-3 py-3 font-semibold">Accuracy</th>
                    <th className="px-3 py-3 font-semibold">F1</th>
                    <th className="px-3 py-3 font-semibold">Timestamp</th>
                    <th className="px-3 py-3 font-semibold">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {models.length > 0 ? (
                    models.map((row) => (
                      <tr key={row.run_id} className="border-t border-borderSubtle">
                        <td className="px-3 py-3 text-textPrimary">{row.run_id}</td>
                        <td className="px-3 py-3 text-textSecondary">{row.status}</td>
                        <td className="px-3 py-3 text-textSecondary">{formatMetric(row.accuracy)}</td>
                        <td className="px-3 py-3 text-textSecondary">{formatMetric(row.f1)}</td>
                        <td className="px-3 py-3 text-textSecondary">{row.timestamp ?? "—"}</td>
                        <td className="px-3 py-3">
                          <button
                            type="button"
                            disabled={row.status === "production"}
                            onClick={() => {
                              void handlePromote(row.run_id);
                            }}
                            className="rounded-lg border border-accent/50 px-3 py-2 text-[11px] font-semibold text-accentSoft hover:bg-accent/10 disabled:cursor-not-allowed disabled:border-borderSubtle disabled:text-textMuted"
                          >
                            Promote to Production
                          </button>
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={6} className="px-3 py-8 text-center text-textSecondary">
                        No registered model runs.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <CheckpointPanel status={checkpointStatus} />
      </div>
    </section>
  );
}

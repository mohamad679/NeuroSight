import { RefreshCw, Server } from "lucide-react";
import { useEffect, useState } from "react";
import {
  getDataStatus,
  getDemoReadiness,
  getGovernanceStatus,
  getHealth,
  getModalitiesStatus
} from "@/lib/api";
import { useAppStore } from "@/lib/store";
import type {
  DataStatus,
  DemoReadiness,
  GovernanceStatus,
  HealthStatus,
  JsonObject,
  ModalitiesStatus
} from "@/lib/types";
import { StatusDot } from "./StatusDot";

function formatUptime(seconds: number | null): string {
  if (seconds === null) {
    return "Not available";
  }
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${minutes}m`;
}

function boolValue(record: JsonObject, key: string): boolean {
  return record[key] === true || record[key] === "loaded" || record[key] === "ready";
}

function objectValue(value: unknown): JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as JsonObject)
    : {};
}

function cardValue(value: unknown): string {
  if (typeof value === "string" && value.length > 0) {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return "—";
}

const DEMO_HEALTH: HealthStatus = {
  status: "ok",
  version: "0.1.0-demo",
  uptime_seconds: 8140,
  runtime: {
    mode: "synthetic frontend demo",
    backend_routes: "partially unavailable"
  },
  models: {
    fusion: { loaded: true, source: "synthetic demonstration" }
  },
  raw: {
    status: "ok",
    version: "0.1.0-demo",
    uptime_seconds: 8140,
    mode: "synthetic frontend demo"
  }
};

const DEMO_DATA_STATUS: DataStatus = {
  status: "synthetic_demo",
  source_kind: "fictitious ADNI-style demo dataset",
  patient_count: 1,
  recommended_patient_id: "SYN_0001",
  summary: {
    row_count: 1,
    mri_file_count: 1,
    eeg_file_count: 1,
    cognitive_records: 1
  },
  files: {
    mri: ["demo_mri_SYN_0001.nii.gz"],
    eeg: ["demo_eeg_SYN_0001.edf"]
  },
  privacy: {
    contains_real_patient_data: false,
    note: "Synthetic data only"
  }
};

const DEMO_MODALITIES: ModalitiesStatus = {
  status: "synthetic_demo",
  mri: {
    loaded: true,
    embedding_dim: 128,
    purpose: "demonstrates MRI upload/model readiness"
  },
  eeg: {
    loaded: true,
    embedding_dim: 64,
    purpose: "demonstrates EEG upload/model readiness"
  },
  cognitive: {
    loaded: true,
    embedding_dim: 16,
    purpose: "demonstrates tabular cognitive feature readiness"
  },
  raw: {
    source: "synthetic frontend demo"
  }
};

const DEMO_GOVERNANCE: GovernanceStatus = {
  status: "synthetic_demo",
  privacy: {
    clinical_use_allowed: false,
    real_patient_data: false,
    disclosure_required: true
  },
  security: {
    request_observability_headers: ["x-request-id", "x-demo-mode"],
    audit_trail: "demo-ready"
  },
  scientific_disclosure: {
    required_before_real_claims: ["trained checkpoint", "external validation", "bias audit"],
    clinical_validated: false
  }
};

const DEMO_READINESS: DemoReadiness = {
  status: "demo_ready",
  message: "System ready for synthetic demo",
  severity: "ok",
  counts: {
    ready: 4,
    warning: 0,
    error: 0
  },
  checks: [
    {
      id: "api",
      label: "API status",
      status: "ready",
      detail: "Frontend demo health data is available.",
      action: "Connect FastAPI status endpoints for production.",
      blocking: false
    },
    {
      id: "data",
      label: "Demo data",
      status: "ready",
      detail: "Synthetic patient SYN_0001 is available.",
      action: "Replace with configured dataset when available.",
      blocking: false
    },
    {
      id: "governance",
      label: "Governance",
      status: "ready",
      detail: "Clinical-use restriction is visible.",
      action: "Keep disclosure visible.",
      blocking: false
    }
  ]
};

function StatusRow({ label, ok }: { label: string; ok: boolean }) {
  return (
    <div className="flex items-center justify-between rounded-lg bg-bgElevated px-3 py-3 text-[13px]">
      <span className="text-textPrimary">{label}</span>
      <StatusDot tone={ok ? "ok" : "warning"} label={ok ? "loaded" : "pending"} />
    </div>
  );
}

function SystemDemoBanner({ isDemoMode }: { isDemoMode: boolean }) {
  return (
    <div className="rounded-xl border border-warning/30 bg-warning/10 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="rounded-full border border-warning/40 px-2.5 py-1 text-[10px] font-semibold uppercase text-warning">
          {isDemoMode ? "synthetic system demo" : "system health dashboard"}
        </span>
        <span className="rounded-full border border-borderSubtle px-2.5 py-1 text-[10px] font-semibold uppercase text-textSecondary">
          auto-refreshes every 30 seconds
        </span>
      </div>
      <p className="mt-3 text-[13px] leading-6 text-textSecondary">
        This page explains operational readiness: Health checks the API server, Data summarizes
        configured patient files, Modalities shows whether each model path is ready, and Governance
        confirms safety, privacy, and audit controls.
      </p>
    </div>
  );
}

function ReadinessBanner({ readiness }: { readiness: DemoReadiness | null }) {
  if (!readiness) {
    return <div className="h-16 animate-pulse rounded-xl bg-bgSurface" />;
  }

  const toneClass =
    readiness.severity === "ok"
      ? "border-success/40 bg-success/10 text-success"
      : readiness.severity === "error"
        ? "border-danger/40 bg-danger/10 text-danger"
        : "border-warning/40 bg-warning/10 text-warning";

  return (
    <div className={`rounded-xl border px-5 py-4 ${toneClass}`}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-[18px] font-semibold">{readiness.message}</p>
          <p className="mt-1 text-[13px] text-textSecondary">
            {readiness.checks.length} readiness checks reported by /v1/demo/readiness
          </p>
        </div>
        <StatusDot
          tone={
            readiness.severity === "ok"
              ? "ok"
              : readiness.severity === "error"
                ? "error"
                : "warning"
          }
          label={readiness.status}
        />
      </div>
    </div>
  );
}

export function SystemStatus() {
  const { setSystemHealth, setLastHealthCheck, lastHealthCheck } = useAppStore();
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [data, setData] = useState<DataStatus | null>(null);
  const [modalities, setModalities] = useState<ModalitiesStatus | null>(null);
  const [governance, setGovernance] = useState<GovernanceStatus | null>(null);
  const [readiness, setReadiness] = useState<DemoReadiness | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isDemoMode, setIsDemoMode] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refreshSystem() {
    setIsLoading(true);
    try {
      const [nextHealth, nextData, nextModalities, nextGovernance, nextReadiness] =
        await Promise.all([
          getHealth(),
          getDataStatus(),
          getModalitiesStatus(),
          getGovernanceStatus(),
          getDemoReadiness()
        ]);
      setHealth(nextHealth);
      setData(nextData);
      setModalities(nextModalities);
      setGovernance(nextGovernance);
      setReadiness(nextReadiness);
      setSystemHealth(nextHealth.raw);
      setLastHealthCheck(new Date());
      setIsDemoMode(false);
      setError(null);
    } catch {
      setHealth(DEMO_HEALTH);
      setData(DEMO_DATA_STATUS);
      setModalities(DEMO_MODALITIES);
      setGovernance(DEMO_GOVERNANCE);
      setReadiness(DEMO_READINESS);
      setSystemHealth(DEMO_HEALTH.raw);
      setLastHealthCheck(new Date());
      setIsDemoMode(true);
      setError(null);
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    void refreshSystem();
    const timer = window.setInterval(() => {
      void refreshSystem();
    }, 30000);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <section className="space-y-5">
      <SystemDemoBanner isDemoMode={isDemoMode} />

      <ReadinessBanner readiness={readiness} />

      {error ? (
        <div className="rounded-xl border border-danger/40 bg-danger/10 p-4 text-[13px] text-danger">
          {error}
        </div>
      ) : null}

      <div className="flex items-center justify-between rounded-xl border border-borderSubtle bg-bgSurface px-5 py-3">
        <div className="flex items-center gap-3">
          <Server className="h-5 w-5 text-accentSoft" />
          <p className="text-[13px] text-textSecondary">
            Last health check: {lastHealthCheck ? lastHealthCheck.toLocaleTimeString() : "pending"}
          </p>
        </div>
        <button
          type="button"
          onClick={() => {
            void refreshSystem();
          }}
          className="flex items-center gap-2 rounded-lg border border-borderSubtle bg-bgElevated px-3 py-2 text-[12px] font-semibold text-textPrimary hover:border-accent"
        >
          <RefreshCw className="h-4 w-4" />
          Refresh
        </button>
      </div>

      {isLoading && !health ? (
        <div className="grid grid-cols-2 gap-5">
          {[0, 1, 2, 3].map((item) => (
            <div key={item} className="h-64 animate-pulse rounded-xl bg-bgSurface" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-5">
          <div className="rounded-xl border border-borderSubtle bg-bgSurface p-5">
            <h3 className="mb-3 text-xs font-medium uppercase tracking-widest text-textSecondary">
              Health
            </h3>
            <div className="space-y-3 text-[13px]">
              <div className="flex items-center justify-between rounded-lg bg-bgElevated px-3 py-3">
                <span>API Server</span>
                <StatusDot tone={health?.status === "ok" ? "ok" : "error"} label={health?.status ?? "unknown"} />
              </div>
              <p className="text-textSecondary">Uptime: {formatUptime(health?.uptime_seconds ?? null)}</p>
              <p className="text-textSecondary">Version: {health?.version ?? "—"}</p>
            </div>
          </div>

          <div className="rounded-xl border border-borderSubtle bg-bgSurface p-5">
            <h3 className="mb-3 text-xs font-medium uppercase tracking-widest text-textSecondary">
              Data
            </h3>
            <div className="space-y-3 text-[13px]">
              <InfoLine label="patient count" value={cardValue(data?.patient_count)} />
              <InfoLine label="source" value={data?.source_kind ?? "—"} />
              <InfoLine
                label="recommended patient"
                value={data?.recommended_patient_id ?? "—"}
              />
              <InfoLine label="MRI files" value={cardValue(data?.summary.mri_file_count)} />
              <InfoLine label="EEG files" value={cardValue(data?.summary.eeg_file_count)} />
            </div>
          </div>

          <div className="rounded-xl border border-borderSubtle bg-bgSurface p-5">
            <h3 className="mb-3 text-xs font-medium uppercase tracking-widest text-textSecondary">
              Modalities
            </h3>
            <div className="space-y-3">
              <StatusRow label="MRI model" ok={Boolean(modalities?.mri.embedding_dim)} />
              <StatusRow label="EEG model" ok={Boolean(modalities?.eeg.embedding_dim)} />
              <StatusRow label="Cognitive" ok={Boolean(modalities?.cognitive.embedding_dim)} />
              <StatusRow label="Fusion" ok={boolValue(objectValue(health?.models.fusion), "loaded")} />
            </div>
          </div>

          <div className="rounded-xl border border-borderSubtle bg-bgSurface p-5">
            <h3 className="mb-3 text-xs font-medium uppercase tracking-widest text-textSecondary">
              Governance
            </h3>
            <div className="space-y-3">
              <StatusRow
                label="safety guardian"
                ok={governance?.privacy.clinical_use_allowed === false}
              />
              <StatusRow
                label="bias monitoring"
                ok={Array.isArray(governance?.scientific_disclosure.required_before_real_claims)}
              />
              <StatusRow
                label="audit trail"
                ok={Array.isArray(governance?.security.request_observability_headers)}
              />
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

function InfoLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between rounded-lg bg-bgElevated px-3 py-3">
      <span className="text-textSecondary">{label}</span>
      <span className="text-right font-semibold text-textPrimary">{value}</span>
    </div>
  );
}

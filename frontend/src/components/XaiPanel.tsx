import * as Select from "@radix-ui/react-select";
import { ChevronDown, Eye, Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { getPatientXai, getXaiStatus } from "@/lib/api";
import type { XaiResult, XaiStatus } from "@/lib/types";
import { CollapsibleJson } from "./CollapsibleJson";
import { StatusDot } from "./StatusDot";

type XaiModality = "cognitive" | "mri" | "eeg";

const DEMO_XAI_STATUS: XaiStatus = {
  status: "synthetic_demo",
  runtime_mode: "frontend demonstration",
  class_mode: "research demo",
  methods: [
    {
      modality: "cognitive",
      status: "implemented",
      method: "SHAP",
      source: "synthetic demo explanation",
      validated_for_clinical_use: false,
      limitations: ["Synthetic values; demonstrates feature-attribution workflow only."]
    },
    {
      modality: "mri",
      status: "requires_checkpoint",
      method: "GradCAM",
      artifact: "trained imaging checkpoint",
      requires_uploaded_data: true,
      validated_for_clinical_use: false,
      limitations: ["Requires a trained MRI model checkpoint and image preprocessing pipeline."]
    },
    {
      modality: "eeg",
      status: "requires_checkpoint",
      method: "Integrated Gradients",
      artifact: "trained EEG checkpoint",
      requires_uploaded_data: true,
      validated_for_clinical_use: false,
      limitations: ["Requires a trained EEG model checkpoint and signal preprocessing pipeline."]
    }
  ],
  interpretation_policy: {
    primary_notice:
      "XAI describes which inputs influenced the demonstration model output. It is not clinical evidence.",
    privacy_notice: "Patient data persisted by XAI: No",
    intended_use: "Research workflow demonstration"
  }
};

const modalityLabels: Record<XaiModality, { title: string; method: string }> = {
  cognitive: { title: "Cognitive", method: "SHAP" },
  mri: { title: "MRI", method: "GradCAM" },
  eeg: { title: "EEG", method: "IntGrad" }
};

function demoXaiResult(patientId: string, modality: XaiModality): XaiResult {
  const isCognitive = modality === "cognitive";
  return {
    patient_id: patientId,
    modality,
    method: modalityLabels[modality].method,
    feature_importance: isCognitive
      ? {
          MMSE: 0.22,
          MOCA: 0.2,
          CDRSB: 0.16,
          ADAS11: 0.14,
          RAVLT_immediate: 0.12,
          RAVLT_learning: 0.07,
          FAQ: 0.06,
          AGE: 0.03
        }
      : {},
    text_summary: isCognitive
      ? "For the synthetic cognitive explanation, MMSE and MOCA have the strongest influence, followed by CDRSB and ADAS11. This demonstrates how a research-facing XAI panel would summarize feature attribution."
      : "",
    xai_available: isCognitive,
    note: isCognitive
      ? null
      : `${modalityLabels[modality].title} XAI is shown as unavailable in this demo until a trained ${modality.toUpperCase()} checkpoint is connected.`,
    target_label: "MCI",
    method_contract: {
      source: "synthetic frontend demo",
      patient_id: patientId,
      modality,
      method: modalityLabels[modality].method,
      output_type: isCognitive ? "feature_importance_bar_chart" : "availability_guidance",
      clinical_validated: false
    },
    interpretation_policy: DEMO_XAI_STATUS.interpretation_policy,
    privacy: {
      persisted: false,
      note: "No patient data is stored by this demo XAI view."
    }
  };
}

function shouldUseDemoXaiStatus(payload: XaiStatus): boolean {
  return payload.methods.length === 0 || payload.status === "unknown";
}

function XaiTooltip({
  active,
  payload,
  label
}: {
  active?: boolean;
  payload?: Array<{ value?: number; name?: string }>;
  label?: string;
}) {
  if (!active || !payload || payload.length === 0) {
    return null;
  }
  return (
    <div className="rounded-lg border border-borderSubtle bg-bgSurface px-3 py-2 text-[11px] text-textPrimary shadow-xl">
      <p className="font-semibold">{label}</p>
      {payload.map((item) => (
        <p key={item.name ?? "importance"} className="text-textSecondary">
          {item.name}: {typeof item.value === "number" ? item.value.toFixed(4) : "—"}
        </p>
      ))}
    </div>
  );
}

export function XaiPanel() {
  const [status, setStatus] = useState<XaiStatus | null>(null);
  const [patientId, setPatientId] = useState("REACT_FRONTEND_DEMO");
  const [modality, setModality] = useState<XaiModality>("cognitive");
  const [result, setResult] = useState<XaiResult | null>(null);
  const [isLoadingStatus, setIsLoadingStatus] = useState(true);
  const [isLoadingResult, setIsLoadingResult] = useState(false);
  const [isDemoMode, setIsDemoMode] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    async function loadStatus() {
      setIsLoadingStatus(true);
      try {
        const payload = await getXaiStatus();
        if (isMounted) {
          if (shouldUseDemoXaiStatus(payload)) {
            setStatus(DEMO_XAI_STATUS);
            setIsDemoMode(true);
          } else {
            setStatus(payload);
            setIsDemoMode(false);
          }
          setError(null);
        }
      } catch {
        if (isMounted) {
          setStatus(DEMO_XAI_STATUS);
          setIsDemoMode(true);
          setError(null);
        }
      } finally {
        if (isMounted) {
          setIsLoadingStatus(false);
        }
      }
    }
    void loadStatus();
    return () => {
      isMounted = false;
    };
  }, []);

  const chartData = useMemo(
    () =>
      result
        ? Object.entries(result.feature_importance)
            .map(([feature, importance]) => ({
              feature,
              importance: Number(importance.toFixed(4))
            }))
            .sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance))
        : [],
    [result]
  );

  async function loadExplanation() {
    const trimmed = patientId.trim();
    if (!trimmed) {
      setError("Patient ID is required.");
      return;
    }

    setIsLoadingResult(true);
    setError(null);
    try {
      if (isDemoMode) {
        setResult(demoXaiResult(trimmed, modality));
        return;
      }
      const payload = await getPatientXai(trimmed, modality);
      setResult(payload);
    } catch {
      setResult(demoXaiResult(trimmed, modality));
      setIsDemoMode(true);
      setError(null);
    } finally {
      setIsLoadingResult(false);
    }
  }

  return (
    <section className="space-y-5">
      <div className="rounded-xl border border-warning/30 bg-warning/10 p-4">
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded-full border border-warning/40 px-2.5 py-1 text-[10px] font-semibold uppercase text-warning">
            {isDemoMode ? "synthetic xai demo" : "explainability dashboard"}
          </span>
          <span className="rounded-full border border-borderSubtle px-2.5 py-1 text-[10px] font-semibold uppercase text-textSecondary">
            not clinical evidence
          </span>
        </div>
        <p className="mt-3 text-[13px] leading-6 text-textSecondary">
          This page explains model behavior: XAI Status lists which explanation methods are
          available per modality, Patient XAI requests an explanation for one patient, the chart
          ranks influential features, and method_contract records what the explanation means.
        </p>
      </div>

      {error ? (
        <div className="rounded-xl border border-danger/40 bg-danger/10 p-4 text-[13px] text-danger">
          {error}
        </div>
      ) : null}

      <div className="grid grid-cols-2 gap-5">
        <div className="rounded-xl border border-borderSubtle bg-bgSurface p-5">
          <div className="mb-5 flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-lg bg-accent text-white">
              <Eye className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-[20px] font-semibold text-textPrimary">XAI Status</h2>
              <p className="text-[13px] text-textSecondary">Explainability capability contract</p>
            </div>
          </div>

          {isLoadingStatus ? (
            <div className="grid grid-cols-3 gap-3">
              {[0, 1, 2].map((item) => (
                <div key={item} className="h-24 animate-pulse rounded-lg bg-bgElevated" />
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-3 gap-3">
              {(["cognitive", "mri", "eeg"] as XaiModality[]).map((item) => {
                const method = status?.methods.find((entry) => entry.modality === item);
                const available = method?.status === "implemented";
                return (
                  <div key={item} className="rounded-lg border border-borderSubtle bg-bgElevated p-3">
                    <p className="text-[15px] font-semibold text-textPrimary">
                      {modalityLabels[item].title}
                    </p>
                    <p className="mt-2 text-[13px] text-textSecondary">
                      {modalityLabels[item].method}
                    </p>
                    <div className="mt-3 text-[12px] text-textSecondary">
                      <StatusDot
                        tone={available ? "ok" : "warning"}
                        label={available ? "available" : "requires checkpoint"}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          <div className="mt-5 rounded-lg border border-borderSubtle bg-bgElevated p-4">
            <p className="text-[11px] font-semibold uppercase tracking-wide text-textSecondary">
              Interpretation policy
            </p>
            <p className="mt-2 text-[13px] leading-6 text-textSecondary">
              {typeof status?.interpretation_policy.primary_notice === "string"
                ? status.interpretation_policy.primary_notice
                : "XAI payloads describe model behavior and are not clinical evidence."}
            </p>
          </div>
        </div>

        <div className="rounded-xl border border-borderSubtle bg-bgSurface p-5">
          <h2 className="text-[20px] font-semibold text-textPrimary">Patient XAI</h2>
          <div className="mt-5 grid grid-cols-[1fr_180px] gap-3">
            <label className="space-y-2">
              <span className="text-[13px] font-medium text-textPrimary">Patient ID</span>
              <input
                value={patientId}
                onChange={(event) => setPatientId(event.target.value)}
                className="h-11 w-full rounded-lg border border-borderSubtle bg-bgElevated px-3 text-[13px] text-textPrimary outline-none focus:shadow-active"
              />
            </label>

            <div className="space-y-2">
              <span className="text-[13px] font-medium text-textPrimary">Modality</span>
              <Select.Root value={modality} onValueChange={(value) => setModality(value as XaiModality)}>
                <Select.Trigger className="flex h-11 w-full items-center justify-between rounded-lg border border-borderSubtle bg-bgElevated px-3 text-[13px] text-textPrimary outline-none focus:shadow-active">
                  <Select.Value />
                  <Select.Icon>
                    <ChevronDown className="h-4 w-4 text-textSecondary" />
                  </Select.Icon>
                </Select.Trigger>
                <Select.Portal>
                  <Select.Content className="overflow-hidden rounded-lg border border-borderSubtle bg-bgSurface shadow-xl">
                    <Select.Viewport className="p-1">
                      {(["cognitive", "mri", "eeg"] as XaiModality[]).map((item) => (
                        <Select.Item
                          key={item}
                          value={item}
                          className="cursor-pointer rounded-md px-3 py-2 text-[13px] text-textPrimary outline-none hover:bg-bgElevated"
                        >
                          <Select.ItemText>{item}</Select.ItemText>
                        </Select.Item>
                      ))}
                    </Select.Viewport>
                  </Select.Content>
                </Select.Portal>
              </Select.Root>
            </div>
          </div>

          <button
            type="button"
            disabled={isLoadingResult}
            onClick={() => {
              void loadExplanation();
            }}
            className="mt-4 flex w-full items-center justify-center gap-2 rounded-lg bg-accent px-5 py-3 text-[13px] font-semibold text-white transition hover:bg-accentSoft focus:shadow-active disabled:cursor-wait disabled:opacity-60"
          >
            <Search className="h-4 w-4" />
            Get Explanation
          </button>

          <div className="mt-5 rounded-lg border border-borderSubtle bg-bgElevated p-3 text-[13px] text-textSecondary">
            Patient data persisted by XAI: No
          </div>
        </div>
      </div>

      <div className="rounded-xl border border-borderSubtle bg-bgSurface p-5">
        <h3 className="mb-3 text-xs font-medium uppercase tracking-widest text-textSecondary">
          Explanation Result
        </h3>
        {isLoadingResult ? (
          <div className="h-72 animate-pulse rounded-lg bg-bgElevated" />
        ) : result ? (
          <div className="grid grid-cols-[1fr_360px] gap-5">
            <div>
              {result.xai_available ? (
                <div className="h-80 rounded-lg border border-borderSubtle bg-bgElevated p-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} layout="vertical" margin={{ left: 14, right: 20 }}>
                      <CartesianGrid stroke="rgba(223,241,232,0.10)" horizontal={false} />
                      <XAxis type="number" stroke="rgba(167,178,169,0.55)" tick={{ fill: "var(--color-text-muted)", fontSize: 11 }} />
                      <YAxis
                        dataKey="feature"
                        type="category"
                        width={116}
                        stroke="rgba(167,178,169,0.55)"
                        tick={{ fill: "var(--color-text-muted)", fontSize: 11 }}
                      />
                      <Tooltip content={<XaiTooltip />} cursor={{ fill: "rgba(244,184,96,0.08)" }} />
                      <Bar dataKey="importance" fill="var(--color-accent)" radius={[0, 6, 6, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="rounded-lg border border-warning/40 bg-warning/10 p-4 text-[13px] leading-6 text-warning">
                  {result.note ?? "Explanation is unavailable for this modality in the current runtime."}
                </div>
              )}
            </div>
            <div className="space-y-3">
              <div className="rounded-lg border border-borderSubtle bg-bgElevated p-4">
                <p className="text-[11px] font-semibold uppercase text-textSecondary">Method</p>
                <p className="mt-2 text-[15px] font-semibold text-textPrimary">{result.method}</p>
                {result.text_summary ? (
                  <p className="mt-3 text-[13px] leading-6 text-textSecondary">{result.text_summary}</p>
                ) : null}
              </div>
              <CollapsibleJson title="method_contract" value={result.method_contract} defaultOpen />
            </div>
          </div>
        ) : (
          <div className="grid min-h-64 place-items-center rounded-lg border border-borderSubtle bg-bgElevated text-center text-[13px] text-textSecondary">
            Request a patient explanation to render feature importance or availability guidance.
          </div>
        )}
      </div>
    </section>
  );
}

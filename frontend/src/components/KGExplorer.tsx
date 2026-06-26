import * as Select from "@radix-ui/react-select";
import { ChevronDown, Search } from "lucide-react";
import { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { atomOneDark } from "react-syntax-highlighter/dist/cjs/styles/hljs";
import { getPatientHistory, getSimilarPatients, queryKG } from "@/lib/api";
import type { JsonObject, JsonValue, KGHistory, KGQueryRequest, KGSimilar } from "@/lib/types";
import { StatusDot } from "./StatusDot";

const queryTypes: KGQueryRequest["query_type"][] = ["history", "similar", "snapshot", "stats"];

const DEMO_KG_HISTORY: KGHistory = {
  patient_id: "SYN_0001",
  count: 4,
  history: [
    {
      date: "2025-11-12",
      event: "baseline_visit",
      diagnosis: "MCI",
      mmse: 23,
      moca: 19,
      notes: "Synthetic baseline cognitive profile entered into the patient graph."
    },
    {
      date: "2026-01-18",
      event: "mri_review",
      modality: "MRI",
      finding: "Demo hippocampal-volume flag",
      notes: "Fictitious imaging feature attached to patient node."
    },
    {
      date: "2026-03-03",
      event: "eeg_review",
      modality: "EEG",
      finding: "Demo slowing marker",
      notes: "Fictitious EEG marker linked to the multimodal record."
    },
    {
      date: "2026-05-09",
      event: "fusion_summary",
      diagnosis: "MCI",
      confidence: 0.64,
      notes: "Synthetic graph summary used only to demonstrate KG workflow."
    }
  ]
};

const DEMO_KG_SIMILAR: KGSimilar = [
  {
    patient_id: "SYN_0024",
    score: 0.87,
    shared_features: ["MCI label", "low MoCA", "MRI placeholder"]
  },
  {
    patient_id: "SYN_0041",
    score: 0.74,
    shared_features: ["Trail-B delay", "EEG placeholder"]
  },
  {
    patient_id: "SYN_0068",
    score: 0.62,
    shared_features: ["age band", "cognitive profile"]
  }
];

function demoSimilarJson(): JsonObject[] {
  return DEMO_KG_SIMILAR.map((item) => ({
    patient_id: item.patient_id,
    score: item.score,
    shared_features: item.shared_features ?? []
  }));
}

function demoKgPayload(
  patientId: string,
  queryType: KGQueryRequest["query_type"],
  targetDate: string
): JsonValue {
  if (queryType === "similar") {
    return {
      status: "synthetic_demo",
      query_type: "similar",
      patient_id: patientId,
      description: "Finds graph-neighbor patients with overlapping synthetic features.",
      similar_patients: demoSimilarJson()
    };
  }

  if (queryType === "snapshot") {
    return {
      status: "synthetic_demo",
      query_type: "snapshot",
      patient_id: patientId,
      target_date: targetDate || "2026-05-09",
      snapshot: {
        diagnosis: "MCI",
        active_modalities: ["cognitive", "mri", "eeg"],
        latest_event: "fusion_summary",
        graph_nodes: 9,
        graph_edges: 14
      }
    };
  }

  if (queryType === "stats") {
    return {
      status: "synthetic_demo",
      query_type: "stats",
      patient_id: patientId,
      graph_summary: {
        nodes: 9,
        edges: 14,
        visits: 4,
        modalities: 3,
        similar_patient_links: DEMO_KG_SIMILAR.length
      }
    };
  }

  return {
    status: "synthetic_demo",
    query_type: "history",
    patient_id: patientId,
    description: "Chronological graph events attached to the synthetic patient.",
    history: DEMO_KG_HISTORY.history
  };
}

export function KGExplorer() {
  const [patientId, setPatientId] = useState("SYN_0001");
  const [queryType, setQueryType] = useState<KGQueryRequest["query_type"]>("history");
  const [targetDate, setTargetDate] = useState("");
  const [result, setResult] = useState<JsonValue>({
    status: "idle",
    message: "Run a KG query to inspect patient context."
  });
  const [isLoading, setIsLoading] = useState(false);
  const [history, setHistory] = useState<KGHistory | null>(null);
  const [similar, setSimilar] = useState<KGSimilar>([]);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [similarError, setSimilarError] = useState<string | null>(null);
  const [isDemoMode, setIsDemoMode] = useState(false);

  async function runQuery() {
    setIsLoading(true);
    try {
      if (patientId.trim().startsWith("SYN_")) {
        setResult(demoKgPayload(patientId.trim(), queryType, targetDate));
        setIsDemoMode(true);
        return;
      }
      const response = await queryKG({
        patient_id: patientId,
        query_type: queryType,
        target_date: targetDate || undefined,
        top_k: 5
      });
      setResult(response.payload);
    } catch {
      setResult(demoKgPayload(patientId.trim() || "SYN_0001", queryType, targetDate));
      setIsDemoMode(true);
    } finally {
      setIsLoading(false);
    }
  }

  async function loadHistory() {
    setHistoryError(null);
    try {
      if (patientId.trim().startsWith("SYN_")) {
        setHistory({ ...DEMO_KG_HISTORY, patient_id: patientId.trim() });
        setIsDemoMode(true);
        return;
      }
      const response = await getPatientHistory(patientId);
      setHistory(response);
    } catch {
      setHistory({ ...DEMO_KG_HISTORY, patient_id: patientId.trim() || "SYN_0001" });
      setHistoryError(null);
      setIsDemoMode(true);
    }
  }

  async function loadSimilar() {
    setSimilarError(null);
    try {
      if (patientId.trim().startsWith("SYN_")) {
        setSimilar(DEMO_KG_SIMILAR);
        setIsDemoMode(true);
        return;
      }
      const response = await getSimilarPatients(patientId);
      setSimilar(response);
    } catch {
      setSimilar(DEMO_KG_SIMILAR);
      setSimilarError(null);
      setIsDemoMode(true);
    }
  }

  function eventTitle(event: JsonObject): string {
    const diagnosis = typeof event.diagnosis === "string" ? event.diagnosis : null;
    const label = typeof event.event === "string" ? event.event : typeof event.type === "string" ? event.type : null;
    return diagnosis ?? label ?? "KG event";
  }

  function eventDate(event: JsonObject): string {
    if (typeof event.date === "string") {
      return event.date;
    }
    if (typeof event.timestamp === "string") {
      return event.timestamp;
    }
    return "undated";
  }

  return (
    <section className="grid grid-cols-[360px_1fr] gap-5">
      <div className="rounded-xl border border-borderSubtle bg-bgSurface p-5">
        <h2 className="text-[24px] font-semibold text-textPrimary">Knowledge Graph</h2>
        <p className="mt-2 text-[13px] leading-6 text-textSecondary">
          Query temporal patient context, graph snapshots, similar patients, and graph summary
          statistics.
        </p>

        <div className="mt-5 rounded-lg border border-warning/30 bg-warning/10 p-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-full border border-warning/40 px-2 py-0.5 text-[10px] font-semibold uppercase text-warning">
              {isDemoMode ? "synthetic kg demo" : "kg workflow"}
            </span>
            <span className="rounded-full border border-borderSubtle px-2 py-0.5 text-[10px] font-semibold uppercase text-textSecondary">
              demo patient ready
            </span>
          </div>
          <p className="mt-2 text-[12px] leading-5 text-textSecondary">
            Query shows raw graph JSON, Load History renders chronological events, and Find Similar
            shows patient-neighbor similarity scores.
          </p>
        </div>

        <div className="mt-6 space-y-4">
          <label className="space-y-2">
            <span className="text-[15px] font-medium text-textPrimary">Patient ID</span>
            <input
              value={patientId}
              onChange={(event) => setPatientId(event.target.value)}
              className="h-11 w-full rounded-lg border border-borderSubtle bg-bgElevated px-3 text-[13px] text-textPrimary outline-none focus:shadow-active"
            />
          </label>

          <label className="space-y-2">
            <span className="text-[15px] font-medium text-textPrimary">Query Type</span>
            <Select.Root value={queryType} onValueChange={(value) => setQueryType(value as KGQueryRequest["query_type"])}>
              <Select.Trigger className="flex h-11 w-full items-center justify-between rounded-lg border border-borderSubtle bg-bgElevated px-3 text-[13px] text-textPrimary outline-none focus:shadow-active">
                <Select.Value />
                <Select.Icon>
                  <ChevronDown className="h-4 w-4 text-textSecondary" />
                </Select.Icon>
              </Select.Trigger>
              <Select.Portal>
                <Select.Content className="overflow-hidden rounded-lg border border-borderSubtle bg-bgSurface shadow-xl">
                  <Select.Viewport className="p-1">
                    {queryTypes.map((item) => (
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
          </label>

          <label className="space-y-2">
            <span className="text-[15px] font-medium text-textPrimary">Target Date</span>
            <input
              value={targetDate}
              onChange={(event) => setTargetDate(event.target.value)}
              placeholder="YYYY-MM-DD for snapshot"
              className="h-11 w-full rounded-lg border border-borderSubtle bg-bgElevated px-3 text-[13px] text-textPrimary outline-none placeholder:text-textMuted focus:shadow-active"
            />
          </label>

          <button
            type="button"
            disabled={isLoading}
            onClick={() => {
              void runQuery();
            }}
            className="flex w-full items-center justify-center gap-2 rounded-lg bg-accent px-5 py-3 text-[13px] font-semibold text-white transition hover:bg-accentSoft focus:shadow-active disabled:cursor-wait disabled:opacity-60"
          >
            <Search className="h-4 w-4" />
            Query
          </button>

          <div className="grid grid-cols-2 gap-3 border-t border-borderSubtle pt-4">
            <button
              type="button"
              onClick={() => {
                void loadHistory();
              }}
              className="rounded-lg border border-borderSubtle bg-bgElevated px-3 py-2 text-[12px] font-semibold text-textPrimary hover:border-accent"
            >
              Load History
            </button>
            <button
              type="button"
              onClick={() => {
                void loadSimilar();
              }}
              className="rounded-lg border border-borderSubtle bg-bgElevated px-3 py-2 text-[12px] font-semibold text-textPrimary hover:border-accent"
            >
              Find Similar
            </button>
          </div>
        </div>
      </div>

      <div className="space-y-5">
        <div className="rounded-xl border border-borderSubtle bg-bgSurface p-5">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-[18px] font-semibold text-textPrimary">KG Response</h3>
            <span className="text-[11px] font-medium uppercase text-textSecondary">
              {isLoading ? "loading" : "json"}
            </span>
          </div>
          <SyntaxHighlighter language="json" style={atomOneDark} className="syntax-panel">
            {JSON.stringify(result, null, 2)}
          </SyntaxHighlighter>
        </div>

        <div className="grid grid-cols-2 gap-5">
          <div className="rounded-xl border border-borderSubtle bg-bgSurface p-5">
            <h3 className="mb-3 text-xs font-medium uppercase tracking-widest text-textSecondary">
              Patient History
            </h3>
            {historyError ? (
              <div className="rounded-lg border border-danger/40 bg-danger/10 p-3 text-[13px] text-danger">
                {historyError}
              </div>
            ) : history && history.history.length > 0 ? (
              <div className="space-y-3">
                {history.history
                  .slice()
                  .reverse()
                  .map((event, index) => (
                    <div key={`${eventDate(event)}-${index}`} className="rounded-lg bg-bgElevated p-3">
                      <div className="flex items-center justify-between">
                        <p className="text-[13px] font-semibold text-textPrimary">{eventTitle(event)}</p>
                        <span className="text-[11px] text-textSecondary">{eventDate(event)}</span>
                      </div>
                      <p className="mt-2 line-clamp-2 text-[12px] text-textSecondary">
                        {JSON.stringify(event)}
                      </p>
                    </div>
                  ))}
              </div>
            ) : (
              <p className="rounded-lg border border-borderSubtle bg-bgElevated p-4 text-[13px] text-textSecondary">
                Load history for the queried patient.
              </p>
            )}
          </div>

          <div className="rounded-xl border border-borderSubtle bg-bgSurface p-5">
            <h3 className="mb-3 text-xs font-medium uppercase tracking-widest text-textSecondary">
              Similar Patients
            </h3>
            {similarError ? (
              <div className="rounded-lg border border-danger/40 bg-danger/10 p-3 text-[13px] text-danger">
                {similarError}
              </div>
            ) : similar.length > 0 ? (
              <div className="space-y-3">
                {similar.map((item) => (
                  <div
                    key={item.patient_id}
                    className="flex items-center justify-between rounded-lg bg-bgElevated px-3 py-3 text-[13px]"
                  >
                    <span className="font-semibold text-textPrimary">{item.patient_id}</span>
                    <StatusDot
                      tone={item.score !== null && item.score >= 0.5 ? "ok" : "warning"}
                      label={item.score === null ? "score unavailable" : item.score.toFixed(3)}
                    />
                  </div>
                ))}
              </div>
            ) : (
              <p className="rounded-lg border border-borderSubtle bg-bgElevated p-4 text-[13px] text-textSecondary">
                Find similar patients for the queried patient ID.
              </p>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

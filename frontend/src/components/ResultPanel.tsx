import ReactMarkdown from "react-markdown";
import { Activity, AlertTriangle, ArrowUpRight, FileText, ShieldCheck } from "lucide-react";
import { AgentTimeline } from "./AgentTimeline";
import { AttentionHeatmap } from "./AttentionHeatmap";
import { ConfidenceMeter } from "./ConfidenceMeter";
import { ModalityBadge } from "./ModalityBadge";
import type { DiagnoseResponse } from "@/lib/types";

interface ResultPanelProps {
  result: DiagnoseResponse | null;
  isLoading: boolean;
  isStreaming?: boolean;
}

function SkeletonBlock({ className }: { className: string }) {
  return <div className={`skeleton ${className}`} />;
}

function ReliabilityBanner({ result }: { result: DiagnoseResponse }) {
  if (!result.reliability_note) {
    return null;
  }
  const sourceLabel =
    result.backend_source === "gradio"
      ? "Gradio Space"
      : result.backend_source === "synthetic"
        ? "Synthetic patient"
        : "FastAPI";
  return (
    <div className="luxury-panel-muted flex flex-col gap-3 p-4 md:flex-row md:items-start md:justify-between">
      <div className="flex gap-3">
        <div className="grid h-10 w-10 shrink-0 place-items-center rounded-[8px] border border-warning/40 bg-warning/10 text-warning">
          <AlertTriangle className="h-4 w-4" />
        </div>
        <div>
          <div className="flex flex-wrap gap-2">
            <span className="rounded-full border border-warning/40 px-2.5 py-1 text-[0.65rem] font-extrabold uppercase tracking-[0.16em] text-warning">
              {sourceLabel}
            </span>
            <span className="rounded-full border border-danger/30 px-2.5 py-1 text-[0.65rem] font-extrabold uppercase tracking-[0.16em] text-danger">
              not validated
            </span>
          </div>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-textSecondary">
            {result.reliability_note}
          </p>
        </div>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <section className="luxury-panel relative grid min-h-[720px] overflow-hidden p-8">
      <div className="absolute right-10 top-10 hidden h-52 w-52 rotate-12 border border-accent/20 md:block" />
      <div className="absolute right-24 top-28 hidden h-24 w-24 rotate-12 border border-teal/20 md:block" />
      <div className="relative z-10 grid content-between gap-10">
        <div>
          <p className="luxury-kicker">awaiting signal</p>
          <h2 className="luxury-heading mt-5 max-w-3xl text-[clamp(3rem,8vw,7rem)]">
            Build the case. Read the system.
          </h2>
          <p className="mt-6 max-w-xl text-base leading-8 text-textSecondary">
            Run a direct or streamed demo analysis. The result surface will show class output,
            model score, modality attention, XAI diagnostics, and the research report.
          </p>
        </div>
        <div className="grid max-w-3xl gap-3 md:grid-cols-3">
          {["Review required", "Synthetic public data", "No clinical use"].map((item) => (
            <div key={item} className="luxury-panel-muted p-4">
              <p className="text-sm font-extrabold text-textPrimary">{item}</p>
              <p className="mt-2 text-xs leading-5 text-textMuted">
                Explicitly surfaced before any output can be interpreted.
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export function ResultPanel({ result, isLoading, isStreaming = false }: ResultPanelProps) {
  if (isStreaming) {
    return (
      <section className="grid min-h-[720px] gap-4 xl:grid-cols-[minmax(0,0.92fr)_minmax(320px,0.55fr)]">
        <div className="luxury-panel reveal p-7">
          <p className="luxury-kicker">live stream</p>
          <h2 className="luxury-heading mt-4 max-w-3xl text-[clamp(2.6rem,6vw,6.4rem)]">
            Agents are moving.
          </h2>
          <p className="mt-5 max-w-2xl text-base leading-8 text-textSecondary">
            The static report view returns when the final completion event arrives from the backend.
          </p>
          <div className="mt-10 luxury-panel-muted p-6">
            <AgentTimeline />
          </div>
        </div>
        <div className="luxury-panel reveal reveal-delay-2 flex flex-col justify-between p-6">
          <Activity className="h-7 w-7 text-accent" />
          <div>
            <p className="font-display text-4xl font-extrabold tracking-[-0.04em] text-textPrimary">
              Streaming
            </p>
            <p className="mt-3 text-sm leading-6 text-textSecondary">
              Every stage emits observable state. No hidden autonomous diagnosis is implied.
            </p>
          </div>
        </div>
      </section>
    );
  }

  if (isLoading) {
    return (
      <section className="grid min-h-[720px] gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="luxury-panel p-7">
          <SkeletonBlock className="h-7 w-36" />
          <SkeletonBlock className="mt-7 h-24 w-full max-w-3xl" />
          <div className="mt-12 grid gap-4 md:grid-cols-2">
            <SkeletonBlock className="h-44" />
            <SkeletonBlock className="h-44" />
          </div>
          <SkeletonBlock className="mt-4 h-72" />
        </div>
        <div className="luxury-panel p-6">
          <SkeletonBlock className="h-12 w-full" />
          <SkeletonBlock className="mt-6 h-64 w-full" />
          <SkeletonBlock className="mt-4 h-28 w-full" />
        </div>
      </section>
    );
  }

  if (!result) {
    return <EmptyState />;
  }

  return (
    <section className="grid min-h-[720px] gap-4">
      <div className="grid gap-4 xl:grid-cols-[minmax(280px,0.42fr)_minmax(360px,0.58fr)_minmax(260px,0.36fr)]">
        <div className="luxury-panel reveal p-5">
          <p className="mb-4 text-xs font-extrabold uppercase tracking-[0.18em] text-textMuted">
            demo model assigned class
          </p>
          <ModalityBadge
            diagnosis={result.diagnosis}
            requiresReview={result.requires_review}
            blockedBySafety={result.blocked_by_safety}
          />
        </div>
        <ConfidenceMeter
          confidence={result.confidence}
          label={result.clinical_validated ? "validated model score" : "demo model score"}
        />
        <div className="luxury-panel reveal reveal-delay-2 flex flex-col justify-between p-5">
          <div className="grid h-11 w-11 place-items-center rounded-[8px] border border-success/30 bg-success/10 text-success">
            <ShieldCheck className="h-5 w-5" />
          </div>
          <div>
            <p className="mt-8 font-display text-3xl font-extrabold tracking-[-0.04em]">
              Review locked
            </p>
            <p className="mt-3 text-sm leading-6 text-textSecondary">
              Output is explicitly framed as research-demo evidence only.
            </p>
          </div>
        </div>
      </div>

      <ReliabilityBanner result={result} />

      <div className="luxury-panel-muted reveal reveal-delay-2 grid gap-3 p-4 md:grid-cols-3">
        <div>
          <p className="text-[0.65rem] font-extrabold uppercase tracking-[0.16em] text-textMuted">
            model mode
          </p>
          <p className="mt-2 text-sm font-bold text-textPrimary">{result.model_mode ?? "unknown"}</p>
        </div>
        <div>
          <p className="text-[0.65rem] font-extrabold uppercase tracking-[0.16em] text-textMuted">
            checkpoint
          </p>
          <p className="mt-2 break-all text-sm font-bold text-textPrimary">
            {result.checkpoint_id ?? "none"}
          </p>
        </div>
        <div>
          <p className="text-[0.65rem] font-extrabold uppercase tracking-[0.16em] text-textMuted">
            clinical status
          </p>
          <p className="mt-2 text-sm font-bold text-danger">
            not clinical software
          </p>
        </div>
        <div className="md:col-span-3">
          <p className="text-sm leading-6 text-textSecondary">
            {result.disclaimer}
          </p>
          {result.warnings.length > 0 ? (
            <ul className="mt-3 grid gap-2 text-xs font-bold text-warning md:grid-cols-3">
              {result.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          ) : null}
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,0.98fr)_minmax(360px,0.58fr)]">
        <div className="reveal reveal-delay-2 min-h-[440px]">
          <AttentionHeatmap
            weights={result.modality_weights}
            featureImportance={result.feature_importance}
            weightsSource={result.modality_weights_source}
            featureSource={result.feature_importance_source}
          />
        </div>

        <div className="luxury-panel reveal reveal-delay-3 flex min-h-[440px] flex-col overflow-hidden">
          <div className="flex items-start justify-between gap-4 border-b border-borderSubtle p-5">
            <div>
              <p className="luxury-kicker">report</p>
              <h3 className="font-display mt-2 text-3xl font-extrabold tracking-[-0.04em]">
                Research Output Summary
              </h3>
            </div>
            <FileText className="h-5 w-5 text-accent" />
          </div>
          <div className="report-markdown min-h-0 flex-1 overflow-y-auto p-5 text-sm leading-7 text-textSecondary">
            <ReactMarkdown>{result.report_text}</ReactMarkdown>
            <div className="mt-5 rounded-[8px] border border-warning/30 bg-warning/10 p-4 text-sm leading-6">
              This report is generated from NeuroSight demo model outputs. It is not medical
              evidence or medical advice.
            </div>
          </div>
          <div className="border-t border-borderSubtle p-4">
            <button type="button" className="btn btn-secondary w-full" aria-label="Open report context">
              Inspect context
              <ArrowUpRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}

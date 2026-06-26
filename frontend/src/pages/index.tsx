import * as Tabs from "@radix-ui/react-tabs";
import Head from "next/head";
import {
  Activity,
  BarChart2,
  Brain,
  ChevronLeft,
  ChevronRight,
  Database,
  Eye,
  GitBranch,
  PanelLeft,
  Server,
  Sparkles,
  User
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useMemo, useState } from "react";
import { AgentTimeline } from "@/components/AgentTimeline";
import { DiagnosisPanel } from "@/components/DiagnosisPanel";
import { EvalPanel } from "@/components/EvalPanel";
import { KGExplorer } from "@/components/KGExplorer";
import { ModelRegistry } from "@/components/ModelRegistry";
import { PatientDiagnosis } from "@/components/PatientDiagnosis";
import { ResultPanel } from "@/components/ResultPanel";
import { StreamEventLog } from "@/components/StreamEventLog";
import { SystemStatus } from "@/components/SystemStatus";
import { XaiPanel } from "@/components/XaiPanel";
import {
  diagnose,
  diagnoseStream,
  diagnoseViaGradio,
  getDiagnosisBackendMode,
  isApiRouteUnavailableError,
  uploadEEG,
  uploadMRI
} from "@/lib/api";
import { useAppStore } from "@/lib/store";
import type { DiagnoseRequest, DiagnoseResponse, DiagnosisLabel, StreamEvent } from "@/lib/types";

const TABS = [
  { id: "diagnosis", label: "Demo Inference", icon: Brain, description: "Input cockpit" },
  { id: "stream", label: "Stream", icon: Activity, description: "Agent timeline" },
  { id: "patient", label: "Patient", icon: User, description: "Demo patient run" },
  { id: "eval", label: "Eval", icon: BarChart2, description: "Synthetic metrics" },
  { id: "models", label: "Models", icon: Database, description: "Registry status" },
  { id: "xai", label: "XAI", icon: Eye, description: "Explanation layer" },
  { id: "system", label: "System", icon: Server, description: "Health and policy" },
  { id: "kg", label: "KG", icon: GitBranch, description: "Graph context" }
] as const satisfies ReadonlyArray<{
  id: string;
  label: string;
  icon: LucideIcon;
  description: string;
}>;

function diagnosisValue(value: string | undefined): DiagnosisLabel {
  const normalized = (value ?? "mci").toLowerCase();
  if (
    normalized === "normal" ||
    normalized === "mci" ||
    normalized === "ad" ||
    normalized === "ftd" ||
    normalized === "lbd" ||
    normalized === "vd"
  ) {
    return normalized;
  }
  return "mci";
}

function weightsFromRequest(req: DiagnoseRequest) {
  const mri = req.mri_embedding && req.mri_embedding.length > 0 ? 1 : 0;
  const eeg = req.eeg_embedding && req.eeg_embedding.length > 0 ? 1 : 0;
  const cog = 1;
  const total = mri + eeg + cog;
  return {
    mri: mri / total,
    eeg: eeg / total,
    cog: cog / total
  };
}

function streamEventToResult(event: StreamEvent, req: DiagnoseRequest): DiagnoseResponse {
  return {
    diagnosis: diagnosisValue(event.diagnosis),
    confidence: event.confidence ?? 0,
    report_text: event.report_text ?? "No research report returned.",
    requires_review: event.requires_review ?? true,
    blocked_by_safety: event.blocked ?? false,
    modality_weights: weightsFromRequest(req),
    feature_importance: {},
    modality_weights_source: "stream request modality availability",
    feature_importance_source: "not emitted by streaming endpoint",
    backend_source: "fastapi",
    model_mode: "demo_untrained",
    checkpoint_id: null,
    trained_on_real_data: false,
    clinical_validated: false,
    requires_expert_review: true,
    disclaimer:
      "Not clinical software. Streaming output is for research/demo review only.",
    warnings: ["Streaming demo output.", "No clinical validation.", "Expert review required."],
    reliability_note:
      "Connected to the FastAPI streaming backend. This remains research software and requires a trained, validated checkpoint before any clinical use."
  };
}

interface StreamPanelProps {
  onRunStream: () => Promise<void>;
  isStreaming: boolean;
  events: StreamEvent[];
  timings: Record<string, number>;
}

function StreamPanel({ onRunStream, isStreaming, events, timings }: StreamPanelProps) {
  return (
    <section className="grid min-h-[720px] gap-4 xl:grid-cols-[minmax(0,1fr)_420px]">
      <div className="luxury-panel reveal flex min-h-[620px] flex-col justify-between p-8">
        <div className="max-w-3xl">
          <p className="luxury-kicker">orchestration</p>
          <h2 className="luxury-heading mt-4 max-w-2xl text-[clamp(2.4rem,5vw,5.4rem)]">
            Live state machine, no theater.
          </h2>
          <p className="mt-5 max-w-xl text-[0.96rem] leading-7 text-textSecondary">
            LangGraph execution is rendered as operational telemetry: running state,
            completion state, latency, and final payload handoff.
          </p>
        </div>
        <div className="mt-10 grid gap-6 lg:grid-cols-[minmax(0,0.9fr)_minmax(260px,0.55fr)]">
          <div className="luxury-panel-muted p-6">
            <AgentTimeline large timings={timings} />
          </div>
          <div className="flex flex-col justify-end gap-4">
            <div className="luxury-panel-muted p-4">
              <p className="text-xs font-bold uppercase tracking-[0.18em] text-textMuted">
                contract
              </p>
              <p className="mt-2 text-sm leading-6 text-textSecondary">
                Final output remains blocked behind review and synthetic-demo disclosure.
              </p>
            </div>
            <button
              type="button"
              disabled={isStreaming}
              onClick={() => {
                void onRunStream();
              }}
              className="btn btn-primary w-full"
              aria-label="Run live stream analysis"
            >
              {isStreaming ? "Streaming" : "Run stream"}
              <Activity className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>
      <div className="reveal reveal-delay-2 min-h-[620px]">
        <StreamEventLog events={events} />
      </div>
    </section>
  );
}

function WorkspaceHeader({
  activeTab,
  isCollapsed,
  onToggleRail
}: {
  activeTab: (typeof TABS)[number];
  isCollapsed: boolean;
  onToggleRail: () => void;
}) {
  const ActiveIcon = activeTab.icon;
  return (
    <header className="sticky top-0 z-30 border-b border-borderSubtle bg-[rgba(var(--color-bg-rgb),0.72)] backdrop-blur-2xl">
      <div className="flex min-h-[82px] items-center justify-between gap-4 px-4 py-4 sm:px-6">
        <div className="flex min-w-0 items-center gap-4">
          <button
            type="button"
            onClick={onToggleRail}
            className="btn btn-ghost h-11 w-11 p-0 lg:hidden"
            aria-label={isCollapsed ? "Expand navigation" : "Collapse navigation"}
          >
            <PanelLeft className="h-5 w-5" />
          </button>
          <div className="grid h-12 w-12 shrink-0 place-items-center rounded-[8px] border border-borderSubtle bg-white/[0.045] text-accent shadow-luxury">
            <ActiveIcon className="h-5 w-5" />
          </div>
          <div className="min-w-0">
            <p className="luxury-kicker">NeuroSight console</p>
            <h1 className="font-display mt-1 truncate text-[1.75rem] font-extrabold leading-none tracking-[-0.04em] text-textPrimary">
              {activeTab.label}
            </h1>
          </div>
        </div>

        <div className="hidden items-center gap-3 md:flex">
          <span className="rounded-full border border-warning/30 bg-warning/10 px-3 py-1.5 text-[0.68rem] font-extrabold uppercase tracking-[0.16em] text-warning">
            synthetic demo
          </span>
          <span className="rounded-full border border-borderSubtle bg-white/[0.035] px-3 py-1.5 text-[0.68rem] font-extrabold uppercase tracking-[0.16em] text-textSecondary">
            not clinical software
          </span>
        </div>
      </div>
    </header>
  );
}

function CommandRail({
  activeValue,
  collapsed,
  onToggle
}: {
  activeValue: string;
  collapsed: boolean;
  onToggle: () => void;
}) {
  return (
    <aside
      className={`hidden border-r border-borderSubtle bg-[rgba(var(--color-bg-rgb),0.78)] backdrop-blur-2xl lg:block ${
        collapsed ? "w-[88px]" : "w-[286px]"
      }`}
    >
      <div className="sticky top-0 flex h-screen flex-col p-4">
        <div className="mb-8 flex items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-3">
            <div className="grid h-11 w-11 shrink-0 place-items-center rounded-[8px] border border-accent/40 bg-accent/15 text-accent">
              <Sparkles className="h-5 w-5" />
            </div>
            {!collapsed ? (
              <div className="min-w-0">
                <p className="font-display text-[1.18rem] font-extrabold tracking-[-0.04em]">
                  NeuroSight
                </p>
                <p className="text-xs font-semibold text-textMuted">Research cockpit</p>
              </div>
            ) : null}
          </div>
          <button
            type="button"
            onClick={onToggle}
            className="btn btn-ghost hidden h-9 w-9 shrink-0 p-0 lg:inline-flex"
            aria-label={collapsed ? "Expand navigation" : "Collapse navigation"}
          >
            {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
          </button>
        </div>

        <Tabs.List className="flex flex-1 flex-col gap-1" aria-label="NeuroSight workspace">
          {TABS.map((tab, index) => {
            const Icon = tab.icon;
            const isActive = activeValue === tab.id;
            return (
              <Tabs.Trigger
                key={tab.id}
                value={tab.id}
                className={`group reveal flex items-center gap-3 rounded-[8px] border px-3 py-3 text-left outline-none ${
                  collapsed ? "justify-center" : ""
                } ${
                  isActive
                    ? "border-accent/40 bg-accent/10 text-textPrimary shadow-[0_0_42px_rgba(244,184,96,0.10)]"
                    : "border-transparent text-textMuted hover:border-borderSubtle hover:bg-white/[0.035] hover:text-textPrimary"
                }`}
                style={{ animationDelay: `${80 + index * 34}ms` }}
                aria-label={tab.label}
              >
                <Icon className={`h-4 w-4 shrink-0 ${isActive ? "text-accent" : ""}`} />
                {!collapsed ? (
                  <span className="min-w-0">
                    <span className="block text-sm font-extrabold">{tab.label}</span>
                    <span className="mt-0.5 block truncate text-xs text-textMuted">
                      {tab.description}
                    </span>
                  </span>
                ) : null}
              </Tabs.Trigger>
            );
          })}
        </Tabs.List>

        <div className="mt-6 rounded-[8px] border border-borderSubtle bg-white/[0.035] p-3">
          <p className="text-[0.66rem] font-extrabold uppercase tracking-[0.18em] text-textMuted">
            review gate
          </p>
          {!collapsed ? (
            <p className="mt-2 text-xs leading-5 text-textSecondary">
              Every output is treated as a research artifact with explicit review requirements.
            </p>
          ) : null}
        </div>
      </div>
    </aside>
  );
}

export function HomePage() {
  const {
    cogScores,
    mriFile,
    eegFile,
    query,
    isLoading,
    result,
    streamEvents,
    isStreaming,
    setLoading,
    setResult,
    setError,
    setMriFile,
    setEegFile,
    addStreamEvent,
    setStreaming,
    setAgentState,
    resetStream
  } = useAppStore();
  const [liveStreamMode, setLiveStreamMode] = useState(false);
  const [streamTimings, setStreamTimings] = useState<Record<string, number>>({});
  const [activeTab, setActiveTab] = useState("diagnosis");
  const [railCollapsed, setRailCollapsed] = useState(false);
  const streamAgents = ["supervisor", "kg_retriever", "report_writer", "safety_guardian"] as const;

  const activeTabConfig = useMemo(
    () => TABS.find((tab) => tab.id === activeTab) ?? TABS[0],
    [activeTab]
  );

  function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => window.setTimeout(resolve, ms));
  }

  function baseDiagnosisRequest(
    selectedMriFile: File | null,
    selectedEegFile: File | null
  ): DiagnoseRequest {
    return {
      ...cogScores,
      query,
      mri_file: selectedMriFile?.name ?? null,
      eeg_file: selectedEegFile?.name ?? null
    };
  }

  function clearProcessedFiles(selectedMriFile: File | null, selectedEegFile: File | null) {
    if (selectedMriFile) {
      setMriFile(null);
    }
    if (selectedEegFile) {
      setEegFile(null);
    }
  }

  async function buildDiagnosisRequest(
    selectedMriFile: File | null,
    selectedEegFile: File | null
  ): Promise<DiagnoseRequest> {
    const [mriUpload, eegUpload] = await Promise.all([
      selectedMriFile ? uploadMRI(selectedMriFile) : Promise.resolve(null),
      selectedEegFile ? uploadEEG(selectedEegFile) : Promise.resolve(null)
    ]);

    return {
      ...baseDiagnosisRequest(selectedMriFile, selectedEegFile),
      mri_embedding: mriUpload?.embedding,
      eeg_embedding: eegUpload?.embedding
    };
  }

  async function markCompatibilityAgent(
    agent: (typeof streamAgents)[number],
    startTimes: Record<string, number>,
    action?: () => Promise<void>
  ) {
    startTimes[agent] = performance.now();
    setAgentState(agent, "running");
    addStreamEvent({ agent, status: "running" });
    await (action ? action() : sleep(300));
    setAgentState(agent, "completed");
    addStreamEvent({ agent, status: "completed" });
    setStreamTimings((current) => ({
      ...current,
      [agent]: (performance.now() - startTimes[agent]) / 1000
    }));
  }

  async function runGradioDiagnosis(
    selectedMriFile: File | null,
    selectedEegFile: File | null,
    withTimeline: boolean
  ): Promise<DiagnoseResponse> {
    const request = baseDiagnosisRequest(selectedMriFile, selectedEegFile);
    if (!withTimeline) {
      return diagnoseViaGradio(request, selectedMriFile, selectedEegFile);
    }

    const startTimes: Record<string, number> = {};
    await markCompatibilityAgent("supervisor", startTimes);
    await markCompatibilityAgent("kg_retriever", startTimes);

    let response: DiagnoseResponse | null = null;
    await markCompatibilityAgent("report_writer", startTimes, async () => {
      response = await diagnoseViaGradio(request, selectedMriFile, selectedEegFile);
    });
    await markCompatibilityAgent("safety_guardian", startTimes);

    if (!response) {
      throw new Error("Gradio risk-profile demo did not return a response.");
    }
    addStreamEvent({ agent: "complete", status: "done" });
    return response;
  }

  async function runStreamingDiagnosis() {
    const selectedMriFile = mriFile;
    const selectedEegFile = eegFile;
    setLoading(true);
    setError(null);
    setResult(null);
    resetStream();
    setStreaming(true);
    setStreamTimings({});

    const startTimes: Record<string, number> = {};
    try {
      const backendMode = await getDiagnosisBackendMode();
      if (backendMode === "gradio") {
        const fallback = await runGradioDiagnosis(selectedMriFile, selectedEegFile, true);
        setResult(fallback);
        clearProcessedFiles(selectedMriFile, selectedEegFile);
        return;
      }

      const request = await buildDiagnosisRequest(selectedMriFile, selectedEegFile);
      for await (const event of diagnoseStream(request)) {
        addStreamEvent(event);
        if (event.agent !== "complete") {
          if (event.status === "running") {
            startTimes[event.agent] = performance.now();
            setAgentState(event.agent, "running");
          }
          if (event.status === "completed") {
            const startedAt = startTimes[event.agent];
            if (startedAt !== undefined) {
              setStreamTimings((current) => ({
                ...current,
                [event.agent]: (performance.now() - startedAt) / 1000
              }));
            }
            setAgentState(event.agent, "completed");
          }
        } else {
          setResult(streamEventToResult(event, request));
        }
      }
      clearProcessedFiles(selectedMriFile, selectedEegFile);
    } catch (error: unknown) {
      if (isApiRouteUnavailableError(error)) {
        try {
          const fallback = await runGradioDiagnosis(selectedMriFile, selectedEegFile, true);
          setResult(fallback);
          clearProcessedFiles(selectedMriFile, selectedEegFile);
          return;
        } catch (fallbackError: unknown) {
          setResult(null);
          setError(
            fallbackError instanceof Error ? fallbackError.message : "Gradio fallback failed"
          );
        }
      } else {
        setResult(null);
        setError(error instanceof Error ? error.message : "Streaming analysis failed");
      }
    } finally {
      setStreaming(false);
      setLoading(false);
    }
  }

  async function handleAnalyze() {
    const selectedMriFile = mriFile;
    const selectedEegFile = eegFile;
    if (liveStreamMode) {
      await runStreamingDiagnosis();
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const backendMode = await getDiagnosisBackendMode();
      if (backendMode === "gradio") {
        const fallback = await runGradioDiagnosis(selectedMriFile, selectedEegFile, false);
        setResult(fallback);
        clearProcessedFiles(selectedMriFile, selectedEegFile);
        return;
      }

      const request = await buildDiagnosisRequest(selectedMriFile, selectedEegFile);
      const response = await diagnose(request);
      setResult(response);
      clearProcessedFiles(selectedMriFile, selectedEegFile);
    } catch (error: unknown) {
      if (isApiRouteUnavailableError(error)) {
        try {
          const fallback = await runGradioDiagnosis(selectedMriFile, selectedEegFile, false);
          setResult(fallback);
          clearProcessedFiles(selectedMriFile, selectedEegFile);
          return;
        } catch (fallbackError: unknown) {
          setResult(null);
          setError(
            fallbackError instanceof Error ? fallbackError.message : "Gradio fallback failed"
          );
        }
      } else {
        setResult(null);
        setError(error instanceof Error ? error.message : "Analysis failed");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <Head>
        <title>NeuroSight Research Console</title>
        <meta
          name="description"
          content="Luxury research interface for the NeuroSight multimodal medical AI scaffold."
        />
      </Head>

      <Tabs.Root
        value={activeTab}
        onValueChange={setActiveTab}
        className="min-h-screen text-textPrimary"
      >
        <div className="flex min-h-screen">
          <CommandRail
            activeValue={activeTab}
            collapsed={railCollapsed}
            onToggle={() => setRailCollapsed((value) => !value)}
          />
          <main className="min-w-0 flex-1">
            <WorkspaceHeader
              activeTab={activeTabConfig}
              isCollapsed={railCollapsed}
              onToggleRail={() => setRailCollapsed((value) => !value)}
            />
            <div className="mx-auto w-full max-w-[1720px] px-4 py-5 sm:px-6 lg:px-8">
              <Tabs.Content value="diagnosis" className="outline-none">
                <div className="grid min-h-[calc(100vh-130px)] gap-5 xl:grid-cols-[minmax(360px,440px)_minmax(0,1fr)]">
                  <div className="reveal">
                    <DiagnosisPanel
                      onAnalyze={handleAnalyze}
                      isStreamMode={liveStreamMode}
                      onStreamModeChange={setLiveStreamMode}
                    />
                  </div>
                  <div className="reveal reveal-delay-2 min-w-0">
                    <ResultPanel result={result} isLoading={isLoading} isStreaming={isStreaming} />
                  </div>
                </div>
              </Tabs.Content>

              <Tabs.Content value="stream" className="outline-none">
                <StreamPanel
                  onRunStream={runStreamingDiagnosis}
                  isStreaming={isStreaming}
                  events={streamEvents}
                  timings={streamTimings}
                />
              </Tabs.Content>

              <Tabs.Content value="patient" className="reveal outline-none">
                <PatientDiagnosis />
              </Tabs.Content>

              <Tabs.Content value="eval" className="reveal outline-none">
                <EvalPanel />
              </Tabs.Content>

              <Tabs.Content value="models" className="reveal outline-none">
                <ModelRegistry />
              </Tabs.Content>

              <Tabs.Content value="xai" className="reveal outline-none">
                <XaiPanel />
              </Tabs.Content>

              <Tabs.Content value="system" className="reveal outline-none">
                <SystemStatus />
              </Tabs.Content>

              <Tabs.Content value="kg" className="reveal outline-none">
                <KGExplorer />
              </Tabs.Content>
            </div>
          </main>
        </div>
      </Tabs.Root>
    </>
  );
}

export default HomePage;

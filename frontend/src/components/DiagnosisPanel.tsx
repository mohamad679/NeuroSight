import * as Slider from "@radix-ui/react-slider";
import {
  Brain,
  Check,
  FileUp,
  Loader2,
  RadioTower,
  Send,
  ShieldAlert,
  X
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import type { DragEvent } from "react";
import { useAppStore, type CogKey } from "@/lib/store";
import type { CognitiveScores } from "@/lib/types";

interface DiagnosisPanelProps {
  onAnalyze: () => Promise<void>;
  isStreamMode: boolean;
  onStreamModeChange: (enabled: boolean) => void;
}

interface SliderConfig {
  key: CogKey;
  label: string;
  min: number;
  max: number;
  step: number;
  unit?: string;
}

const sliders: SliderConfig[] = [
  { key: "MMSE", label: "MMSE (0-30)", min: 0, max: 30, step: 1 },
  { key: "MOCA", label: "MOCA (0-30)", min: 0, max: 30, step: 1 },
  { key: "CDRSB", label: "CDRSB (0-18)", min: 0, max: 18, step: 0.5 },
  { key: "ADAS11", label: "ADAS11 (0-70)", min: 0, max: 70, step: 0.5 },
  { key: "RAVLT_immediate", label: "RAVLT immediate (0-75)", min: 0, max: 75, step: 1 },
  { key: "RAVLT_learning", label: "RAVLT learning (-15-15)", min: -15, max: 15, step: 0.5 },
  { key: "FAQ", label: "FAQ (0-30)", min: 0, max: 30, step: 1 },
  { key: "AGE", label: "AGE (0-120)", min: 0, max: 120, step: 1 }
];

interface DropZoneProps {
  label: string;
  accept: string;
  file: File | null;
  onFile: (file: File | null) => void;
}

function DropZone({ label, accept, file, onFile }: DropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!file && inputRef.current) {
      inputRef.current.value = "";
    }
  }, [file]);

  const onDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setIsDragging(false);
      const dropped = event.dataTransfer.files.item(0);
      if (dropped) {
        onFile(dropped);
      }
    },
    [onFile]
  );

  return (
    <div
      onDragOver={(event) => {
        event.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={onDrop}
      className={`group relative rounded-[8px] border border-dashed p-3 ${
        isDragging
          ? "border-accent bg-accent/10 shadow-[0_0_0_4px_rgba(244,184,96,0.12)]"
          : "border-borderSubtle bg-white/[0.035] hover:border-accent/50 hover:bg-accent/5"
      }`}
    >
      <label className="flex min-h-[104px] cursor-pointer flex-col justify-between">
        <span className="flex items-start justify-between gap-3">
          <span className="grid h-10 w-10 place-items-center rounded-[8px] border border-borderSubtle bg-bgBase/70 text-accent">
            {file ? <Check className="h-4 w-4" /> : <FileUp className="h-4 w-4" />}
          </span>
          <span className="text-right text-[0.62rem] font-extrabold uppercase tracking-[0.18em] text-textMuted">
            optional
          </span>
        </span>
        <span>
          <span className="block text-sm font-extrabold text-textPrimary">{label}</span>
          <span className="mt-1 block max-w-full truncate text-xs leading-5 text-textSecondary">
            {file ? file.name : accept}
          </span>
        </span>
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          className="sr-only"
          aria-label={`${label} file input`}
          onChange={(event) => onFile(event.target.files?.item(0) ?? null)}
        />
      </label>
      {file ? (
        <button
          type="button"
          aria-label={`Clear ${label}`}
          onClick={() => {
            onFile(null);
            if (inputRef.current) {
              inputRef.current.value = "";
            }
          }}
          className="btn btn-ghost absolute right-2 top-2 h-8 w-8 p-0"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      ) : null}
    </div>
  );
}

function AssessmentSlider({
  config,
  scores,
  onChange
}: {
  config: SliderConfig;
  scores: CognitiveScores;
  onChange: (key: CogKey, value: number) => void;
}) {
  const value = scores[config.key];
  const pct = ((Number(value) - config.min) / (config.max - config.min)) * 100;
  return (
    <div className="luxury-panel-muted p-3">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <label className="text-xs font-extrabold uppercase tracking-[0.14em] text-textMuted">
            {config.label}
          </label>
          <div className="mt-1 h-px w-10 bg-accent/50" />
        </div>
        <span className="rounded-[6px] border border-borderSubtle bg-bgBase/80 px-2 py-1 text-xs font-extrabold text-textPrimary">
          {value}
          {config.unit ?? ""}
        </span>
      </div>
      <Slider.Root
        min={config.min}
        max={config.max}
        step={config.step}
        value={[Number(value)]}
        onValueChange={([next]) => onChange(config.key, next)}
        className="relative flex h-5 w-full touch-none select-none items-center"
        aria-label={config.label}
      >
        <Slider.Track className="relative h-1.5 grow overflow-hidden rounded-full bg-white/[0.07]">
          <Slider.Range className="absolute h-full rounded-full bg-accent shadow-[0_0_24px_rgba(244,184,96,0.35)]" />
        </Slider.Track>
        <Slider.Thumb className="block h-4 w-4 rounded-full border border-accent bg-bgBase shadow-[0_0_0_5px_rgba(244,184,96,0.12)] outline-none transition hover:scale-110 focus-visible:scale-110" />
      </Slider.Root>
      <div className="mt-2 flex justify-between text-[0.65rem] font-bold text-textMuted">
        <span>{config.min}</span>
        <span>{Math.round(pct)}%</span>
        <span>{config.max}</span>
      </div>
    </div>
  );
}

export function DiagnosisPanel({
  onAnalyze,
  isStreamMode,
  onStreamModeChange
}: DiagnosisPanelProps) {
  const {
    cogScores,
    mriFile,
    eegFile,
    query,
    isLoading,
    error,
    setCog,
    setMriFile,
    setEegFile,
    setQuery
  } = useAppStore();

  return (
    <aside className="luxury-panel flex h-full min-h-[720px] flex-col overflow-hidden">
      <div className="border-b border-borderSubtle p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="luxury-kicker">input cockpit</p>
            <h2 className="font-display mt-3 text-[2rem] font-extrabold leading-none tracking-[-0.045em] text-textPrimary">
              Case builder
            </h2>
            <p className="mt-3 max-w-sm text-sm leading-6 text-textSecondary">
              Synthetic cognitive values, optional modality uploads, and explicit review guardrails.
            </p>
          </div>
          <div className="grid h-11 w-11 place-items-center rounded-[8px] border border-accent/40 bg-accent/10 text-accent">
            <Brain className="h-5 w-5" />
          </div>
        </div>
      </div>

      <div className="min-h-0 flex-1 space-y-5 overflow-y-auto p-5">
        <section className="luxury-panel-muted p-2">
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => onStreamModeChange(false)}
              className={`btn h-11 ${!isStreamMode ? "btn-primary" : "btn-ghost"}`}
              aria-pressed={!isStreamMode}
            >
              Direct
              <Send className="h-4 w-4" />
            </button>
            <button
              type="button"
              onClick={() => onStreamModeChange(true)}
              className={`btn h-11 ${isStreamMode ? "btn-primary" : "btn-ghost"}`}
              aria-pressed={isStreamMode}
            >
              Stream
              <RadioTower className="h-4 w-4" />
            </button>
          </div>
        </section>

        <section>
          <div className="mb-3 flex items-center justify-between gap-3">
            <h3 className="text-xs font-extrabold uppercase tracking-[0.18em] text-textMuted">
              Cognitive assessment
            </h3>
            <span className="text-xs font-bold text-textSecondary">8 canonical signals</span>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">
            {sliders.map((config) => (
              <AssessmentSlider
                key={config.key}
                config={config}
                scores={cogScores}
                onChange={(key, value) => setCog(key, value)}
              />
            ))}
          </div>
        </section>

        <section>
          <h3 className="mb-3 text-xs font-extrabold uppercase tracking-[0.18em] text-textMuted">
            Optional modalities
          </h3>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">
            <DropZone label="MRI upload" accept=".npy,.nii,.nii.gz" file={mriFile} onFile={setMriFile} />
            <DropZone label="EEG upload" accept=".npy,.edf" file={eegFile} onFile={setEegFile} />
          </div>
        </section>

        <section>
          <label className="text-xs font-extrabold uppercase tracking-[0.18em] text-textMuted">
            Research/demo query
          </label>
          <textarea
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="What should the research workflow inspect?"
            className="field mt-3 min-h-[104px] w-full resize-none px-4 py-3 text-sm leading-6 placeholder:text-textMuted"
            aria-label="Research/demo query"
          />
        </section>

        {error ? (
          <div className="flex gap-3 rounded-[8px] border border-danger/30 bg-danger/10 px-4 py-3 text-sm leading-6 text-danger">
            <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" />
            <p>{error}</p>
          </div>
        ) : null}
      </div>

      <div className="border-t border-borderSubtle bg-bgBase/50 p-5">
        <button
          type="button"
          disabled={isLoading}
          onClick={() => {
            void onAnalyze();
          }}
          className="btn btn-primary w-full"
          aria-label={isStreamMode ? "Run streamed research analysis" : "Run research analysis"}
        >
          {isLoading ? (
            <>
              Processing
              <Loader2 className="h-4 w-4 animate-spin" />
            </>
          ) : (
            <>
              Analyze
              <Send className="h-4 w-4" />
            </>
          )}
        </button>
      </div>
    </aside>
  );
}

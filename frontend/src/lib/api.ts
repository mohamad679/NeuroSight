import { useAppStore } from "./store";
import type {
  CheckpointStatus,
  DataStatus,
  DemoReadiness,
  DemoReadinessCheck,
  DiagnoseRequest,
  DiagnoseResponse,
  DiagnosisLabel,
  EvalHistoryEntry,
  EvalMetrics,
  EvalReport,
  GovernanceStatus,
  HealthStatus,
  JsonObject,
  JsonValue,
  KGHistory,
  KGQueryRequest,
  KGQueryResponse,
  KGSimilar,
  ModalityWeights,
  ModalitiesStatus,
  ModelRun,
  StreamEvent,
  UploadEmbeddingResponse,
  XaiMethod,
  XaiResult,
  XaiStatus
} from "./types";

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ??
  (typeof window === "undefined" ? "http://localhost:8000" : window.location.origin);
const API_KEY = process.env.NEXT_PUBLIC_API_KEY ?? "dev-key";
const XAI_PATIENT_ID = "REACT_FRONTEND_DEMO";

interface BackendDiagnosePayload {
  query?: string;
  cognitive_scores: Record<string, number>;
  mri_embedding?: number[];
  eeg_embedding?: number[];
  cog_embedding?: number[];
}

interface GradioFileData {
  path: string;
  url?: string | null;
  size?: number | null;
  orig_name?: string | null;
  mime_type?: string | null;
  is_stream?: boolean;
  meta: {
    _type: "gradio.FileData";
  };
}

interface GradioCallResponse {
  event_id?: string;
}

export type DiagnosisBackendMode = "fastapi" | "gradio";

let diagnosisBackendModePromise: Promise<DiagnosisBackendMode> | null = null;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asJsonValue(value: unknown): JsonValue {
  if (
    value === null ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map(asJsonValue);
  }
  if (isRecord(value)) {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [key, asJsonValue(item)])
    ) as JsonObject;
  }
  return String(value);
}

function numberValue(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function booleanValue(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function stringValue(value: unknown, fallback: string): string {
  return typeof value === "string" ? value : fallback;
}

function diagnosisValue(value: unknown): DiagnosisLabel {
  const normalized = stringValue(value, "mci").toLowerCase();
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

function numberArray(value: unknown): number[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter((item): item is number => typeof item === "number" && Number.isFinite(item))
    .map((item) => item);
}

function numberRecord(value: unknown): Record<string, number> {
  if (!isRecord(value)) {
    return {};
  }
  return Object.fromEntries(
    Object.entries(value)
      .filter((entry): entry is [string, number] => typeof entry[1] === "number")
      .map(([key, item]) => [key, Number(item)])
  );
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function optionalString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function toJsonObject(value: unknown): JsonObject {
  const json = asJsonValue(value);
  return isRecord(json) && !Array.isArray(json) ? json : {};
}

function toJsonObjectArray(value: unknown): JsonObject[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map(toJsonObject).filter((item) => Object.keys(item).length > 0);
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === "string");
}

function metricValue(metrics: Record<string, number>, keys: string[]): number | null {
  for (const key of keys) {
    const value = metrics[key];
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
  }
  return null;
}

function requestErrorMessage(error: unknown): string {
  if (!(error instanceof Error)) {
    return "Network request failed";
  }
  if (
    error.message === "Load failed" ||
    error.message === "Failed to fetch" ||
    error.message.includes("NetworkError")
  ) {
    return `Cannot reach NeuroSight backend at ${BACKEND_URL}. Start FastAPI on port 8000, or check CORS if the frontend is not on localhost:3000.`;
  }
  return error.message;
}

export function isApiRouteUnavailableError(error: unknown): boolean {
  return (
    error instanceof Error &&
    (error.message.includes("Not Found (404)") ||
      error.message.includes("Method Not Allowed (405)"))
  );
}

function isApiRouteUnavailableMessage(message: string): boolean {
  return message.includes("Not Found (404)") || message.includes("Method Not Allowed (405)");
}

export async function getDiagnosisBackendMode(): Promise<DiagnosisBackendMode> {
  if (!diagnosisBackendModePromise) {
    diagnosisBackendModePromise = (async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/config`, { method: "GET" });
        const payload = await parseResponse(res);
        if (
          res.ok &&
          isRecord(payload) &&
          (payload.api_prefix === "/gradio_api" || payload.mode === "blocks")
        ) {
          return "gradio";
        }
      } catch {
        return "fastapi";
      }
      return "fastapi";
    })();
  }
  return diagnosisBackendModePromise;
}

function normalizeWeights(weights: ModalityWeights): ModalityWeights {
  const total = weights.mri + weights.eeg + weights.cog;
  if (total <= 0) {
    return { mri: 0, eeg: 0, cog: 1 };
  }
  return {
    mri: weights.mri / total,
    eeg: weights.eeg / total,
    cog: weights.cog / total
  };
}

function inferWeights(req: DiagnoseRequest): ModalityWeights {
  return normalizeWeights({
    mri: req.mri_embedding && req.mri_embedding.length > 0 ? 1 : 0,
    eeg: req.eeg_embedding && req.eeg_embedding.length > 0 ? 1 : 0,
    cog: req.cog_embedding && req.cog_embedding.length > 0 ? 1 : 1
  });
}

async function parseResponse(res: Response): Promise<unknown> {
  const text = await res.text();
  if (!text) {
    return {};
  }
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return { detail: text };
  }
}

export async function requestJson(
  path: string,
  init: RequestInit,
  options: { suppressRouteUnavailableError?: boolean } = {}
): Promise<unknown> {
  try {
    const res = await fetch(`${BACKEND_URL}${path}`, init);
    const payload = await parseResponse(res);
    if (!res.ok) {
      const detail = isRecord(payload) ? stringValue(payload.detail, res.statusText) : res.statusText;
      throw new Error(`${detail} (${res.status})`);
    }
    useAppStore.getState().setError(null);
    return payload;
  } catch (error: unknown) {
    const message = requestErrorMessage(error);
    if (!(options.suppressRouteUnavailableError && isApiRouteUnavailableMessage(message))) {
      useAppStore.getState().setError(message);
    }
    throw new Error(message);
  }
}

async function uploadGradioFile(file: File): Promise<GradioFileData> {
  const form = new FormData();
  form.append("files", file);
  const res = await fetch(`${BACKEND_URL}/gradio_api/upload`, {
    method: "POST",
    body: form
  });
  const payload = await parseResponse(res);
  if (!res.ok) {
    const detail = isRecord(payload) ? stringValue(payload.detail, res.statusText) : res.statusText;
    throw new Error(`${detail} (${res.status})`);
  }
  const uploadedPath = Array.isArray(payload) && typeof payload[0] === "string" ? payload[0] : null;
  if (!uploadedPath) {
    throw new Error("Gradio upload did not return a file path.");
  }
  return {
    path: uploadedPath,
    orig_name: file.name,
    size: file.size,
    mime_type: file.type || null,
    is_stream: false,
    meta: { _type: "gradio.FileData" }
  };
}

function gradioInputValue(fileData: GradioFileData | null): JsonValue {
  if (!fileData) {
    return null;
  }
  return {
    path: fileData.path,
    url: fileData.url ?? null,
    size: fileData.size ?? null,
    orig_name: fileData.orig_name ?? null,
    mime_type: fileData.mime_type ?? null,
    is_stream: fileData.is_stream ?? false,
    meta: fileData.meta
  };
}

async function readGradioResult(eventId: string): Promise<unknown[]> {
  const res = await fetch(`${BACKEND_URL}/gradio_api/call/diagnose/${encodeURIComponent(eventId)}`, {
    method: "GET"
  });
  const text = await res.text();
  if (!res.ok) {
    throw new Error(`${text || res.statusText} (${res.status})`);
  }
  let latest: unknown[] = [];
  for (const line of text.split("\n")) {
    if (line.startsWith("data: ")) {
      const raw = line.slice(6).trim();
      if (raw.length === 0) {
        continue;
      }
      const parsed = JSON.parse(raw) as unknown;
      if (Array.isArray(parsed)) {
        latest = parsed;
      }
    }
  }
  return latest;
}

function parseGradioDiagnosis(value: unknown): DiagnosisLabel {
  if (typeof value !== "string") {
    return "mci";
  }
  const text = value.replace(/<[^>]*>/g, " ").toLowerCase();
  for (const label of ["normal", "mci", "ad", "ftd", "lbd", "vd"] as const) {
    if (text.includes(label)) {
      return label;
    }
  }
  return "mci";
}

function parseGradioConfidence(value: unknown): number {
  if (typeof value !== "string") {
    return 0;
  }
  const match = value.match(/Confidence:\s*([0-9]+(?:\.[0-9]+)?)%/i) ??
    value.match(/([0-9]+(?:\.[0-9]+)?)\s*%/);
  if (!match) {
    return 0;
  }
  return Math.max(0, Math.min(1, Number(match[1]) / 100));
}

function gradioFallbackWeights(req: DiagnoseRequest): ModalityWeights {
  return normalizeWeights({
    mri: req.mri_file ? 1 : 0,
    eeg: req.eeg_file ? 1 : 0,
    cog: 1
  });
}

function diagnosisDisplay(label: DiagnosisLabel): string {
  const labels: Record<DiagnosisLabel, string> = {
    normal: "Normal",
    mci: "MCI",
    ad: "Alzheimer's disease",
    ftd: "Frontotemporal dementia",
    lbd: "Lewy body dementia",
    vd: "Vascular dementia"
  };
  return labels[label];
}

function percentLabel(value: number): string {
  return `${Math.round(value * 1000) / 10}%`;
}

function uploadedFileLabel(file: File | null): string {
  return file ? `uploaded (${file.name})` : "not provided";
}

function stripDemoHeading(text: string): string {
  const cleaned = text.replace(/^#+\s*Demo Report\s*/i, "").trim();
  return cleaned.length > 0 ? cleaned : "The backend did not return additional narrative detail.";
}

function buildGradioResearchReport({
  req,
  mriFile,
  eegFile,
  diagnosis,
  confidence,
  warning,
  backendReport,
  weights,
  reliabilityNote
}: {
  req: DiagnoseRequest;
  mriFile: File | null;
  eegFile: File | null;
  diagnosis: DiagnosisLabel;
  confidence: number;
  warning: string;
  backendReport: string;
  weights: ModalityWeights;
  reliabilityNote: string;
}): string {
  const query = req.query?.trim() || "What should the research/demo workflow inspect?";
  const reviewStatus = warning.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
  const narrative = stripDemoHeading(backendReport);

  return `### Research Output Summary

#### Input Summary
- Cognitive profile: MMSE ${req.MMSE}, MOCA ${req.MOCA}, CDRSB ${req.CDRSB}, ADAS11 ${req.ADAS11}, RAVLT immediate ${req.RAVLT_immediate}, RAVLT learning ${req.RAVLT_learning}, FAQ ${req.FAQ}, AGE ${req.AGE}.
- MRI input: ${uploadedFileLabel(mriFile)}.
- EEG input: ${uploadedFileLabel(eegFile)}.
- Research/demo query: ${query}

#### Model Output
- Demo model assigned class: ${diagnosisDisplay(diagnosis)} (${diagnosis.toUpperCase()}).
- Model score: ${percentLabel(confidence)}.
- Review status: ${reviewStatus || "Human review required."}

#### Modality Context
- MRI contribution shown in the interface: ${percentLabel(weights.mri)}.
- EEG contribution shown in the interface: ${percentLabel(weights.eeg)}.
- Cognitive contribution shown in the interface: ${percentLabel(weights.cog)}.
- The deployed Gradio Space does not expose structured numeric attention values through the compatibility API. The bars are therefore rendered from the available modality mix, while the backend narrative below is preserved as returned by the model service.

#### Backend Report
${narrative}

#### Test Project Disclaimer
${reliabilityNote} This NeuroSight screen is a test project and research console only. It is not medical software, not a diagnostic device, and must not be used for patient-care decisions.`;
}

export async function diagnoseViaGradio(
  req: DiagnoseRequest,
  mriFile: File | null,
  eegFile: File | null
): Promise<DiagnoseResponse> {
  const [mriUpload, eegUpload] = await Promise.all([
    mriFile ? uploadGradioFile(mriFile) : Promise.resolve(null),
    eegFile ? uploadGradioFile(eegFile) : Promise.resolve(null)
  ]);

  const callPayload: JsonObject = {
    data: [
      req.MMSE,
      req.MOCA,
      req.CDRSB,
      req.ADAS11,
      req.RAVLT_immediate,
      req.RAVLT_learning,
      req.FAQ,
      req.AGE,
      gradioInputValue(mriUpload),
      gradioInputValue(eegUpload),
      req.query ?? ""
    ]
  };

  const callRes = await fetch(`${BACKEND_URL}/gradio_api/call/diagnose`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(callPayload)
  });
  const callJson = (await parseResponse(callRes)) as GradioCallResponse;
  if (!callRes.ok || !callJson.event_id) {
    const detail = isRecord(callJson) ? stringValue(callJson.detail, callRes.statusText) : callRes.statusText;
    throw new Error(`${detail} (${callRes.status})`);
  }

  const output = await readGradioResult(callJson.event_id);
  const warning = typeof output[2] === "string" ? output[2] : "Human review required.";
  const reportText = typeof output[5] === "string" ? output[5] : "No report text returned.";
  const diagnosis = parseGradioDiagnosis(output[0]);
  const confidence = parseGradioConfidence(output[1]);
  const weights = gradioFallbackWeights(req);
  const reliabilityNote =
    "Connected to the deployed Hugging Face Gradio Space. Uploaded files are processed by the demo model, but the current Space reports freshly initialized/demo weights, so this is not a clinically reliable prediction.";
  useAppStore.getState().setError(null);
  return {
    diagnosis,
    confidence,
    requires_review: warning.toLowerCase().includes("review"),
    blocked_by_safety: false,
    modality_weights: weights,
    feature_importance: {},
    modality_weights_source: "available modality mix from Gradio compatibility path",
    feature_importance_source: "structured feature importance unavailable from Gradio response",
    backend_source: "gradio",
    reliability_note: reliabilityNote,
    model_mode: "demo_untrained",
    checkpoint_id: null,
    trained_on_real_data: false,
    clinical_validated: false,
    requires_expert_review: true,
    disclaimer:
      "Not clinical software. This research/demo output is not validated for diagnosis, treatment, triage, or emergency use.",
    warnings: [
      "Demo/synthetic compatibility path.",
      "Outputs require expert review.",
      "The deployed Gradio response does not provide clinical validation evidence."
    ],
    report_text: buildGradioResearchReport({
      req,
      mriFile,
      eegFile,
      diagnosis,
      confidence,
      warning,
      backendReport: reportText,
      weights,
      reliabilityNote
    })
  };
}

function authHeaders(contentType: "json" | "form"): HeadersInit {
  if (contentType === "json") {
    return {
      "Content-Type": "application/json",
      "X-API-Key": API_KEY
    };
  }
  return {
    "X-API-Key": API_KEY
  };
}

function toBackendPayload(req: DiagnoseRequest): BackendDiagnosePayload {
  const payload: BackendDiagnosePayload = {
    query: req.query,
    cognitive_scores: {
      MMSE: req.MMSE,
      MOCA: req.MOCA,
      CDRSB: req.CDRSB,
      ADAS11: req.ADAS11,
      RAVLT_immediate: req.RAVLT_immediate,
      RAVLT_learning: req.RAVLT_learning,
      FAQ: req.FAQ,
      AGE: req.AGE
    }
  };
  if (req.mri_embedding && req.mri_embedding.length > 0) {
    payload.mri_embedding = req.mri_embedding;
  }
  if (req.eeg_embedding && req.eeg_embedding.length > 0) {
    payload.eeg_embedding = req.eeg_embedding;
  }
  if (req.cog_embedding && req.cog_embedding.length > 0) {
    payload.cog_embedding = req.cog_embedding;
  }
  return payload;
}

async function getCognitiveXai(): Promise<Record<string, number>> {
  const payload = await requestJson(`/v1/xai/${XAI_PATIENT_ID}?modality=cognitive`, {
    method: "GET",
    headers: authHeaders("json")
  });
  return isRecord(payload) ? numberRecord(payload.feature_importance) : {};
}

function normalizeDiagnosePayload(
  raw: unknown,
  fallbackWeights: ModalityWeights,
  featureImportance: Record<string, number>
): DiagnoseResponse {
  const payload = isRecord(raw) ? raw : {};
  const rawWeights = isRecord(payload.modality_weights)
    ? {
        mri: numberValue(payload.modality_weights.mri, 0),
        eeg: numberValue(payload.modality_weights.eeg, 0),
        cog: numberValue(payload.modality_weights.cog, 0)
      }
    : fallbackWeights;

  return {
    diagnosis: diagnosisValue(payload.diagnosis),
    confidence: numberValue(payload.confidence, 0),
    report_text: stringValue(payload.report_text, "No research output returned."),
    requires_review: booleanValue(payload.requires_review, true),
    blocked_by_safety: booleanValue(payload.blocked_by_safety, false),
    modality_weights: normalizeWeights(rawWeights),
    feature_importance: featureImportance,
    modality_weights_source: isRecord(payload.modality_weights)
      ? "backend"
      : "demo default from available modalities",
    feature_importance_source:
      Object.keys(featureImportance).length > 0 ? "backend cognitive XAI" : "unavailable",
    backend_source: "fastapi",
    model_mode: optionalString(payload.model_mode),
    checkpoint_id: optionalString(payload.checkpoint_id),
    trained_on_real_data: booleanValue(payload.trained_on_real_data, false),
    clinical_validated: booleanValue(payload.clinical_validated, false),
    requires_expert_review: booleanValue(payload.requires_expert_review, true),
    disclaimer: stringValue(
      payload.disclaimer,
      "Not clinical software. This output is for research/demo review only."
    ),
    warnings: stringArray(payload.warnings),
    reliability_note:
      "Connected to the FastAPI risk-profile backend. This project still needs subject-disjoint validation before outputs can be treated as real-world evidence."
  };
}

function normalizeDiagnoseResponse(
  raw: unknown,
  req: DiagnoseRequest,
  featureImportance: Record<string, number>
): DiagnoseResponse {
  return normalizeDiagnosePayload(raw, inferWeights(req), featureImportance);
}

export async function diagnose(req: DiagnoseRequest): Promise<DiagnoseResponse> {
  const raw = await requestJson("/v1/risk-profile", {
    method: "POST",
    headers: authHeaders("json"),
    body: JSON.stringify(toBackendPayload(req))
  }, { suppressRouteUnavailableError: true });

  let featureImportance: Record<string, number> = {};
  try {
    featureImportance = await getCognitiveXai();
  } catch {
    featureImportance = {};
  }

  return normalizeDiagnoseResponse(raw, req, featureImportance);
}

function streamStatusValue(value: unknown): StreamEvent["status"] {
  if (value === "completed" || value === "done") {
    return value;
  }
  return "running";
}

function normalizeStreamEvent(raw: unknown): StreamEvent | null {
  if (!isRecord(raw)) {
    return null;
  }
  const agent = optionalString(raw.agent);
  if (!agent) {
    return null;
  }
  return {
    agent,
    status: streamStatusValue(raw.status),
    diagnosis: optionalString(raw.diagnosis) ?? undefined,
    confidence: finiteNumber(raw.confidence) ?? undefined,
    requires_review:
      typeof raw.requires_review === "boolean" ? raw.requires_review : undefined,
    blocked: typeof raw.blocked === "boolean" ? raw.blocked : undefined,
    report_text: optionalString(raw.report_text) ?? undefined
  };
}

export async function* diagnoseStream(req: DiagnoseRequest): AsyncGenerator<StreamEvent> {
  try {
    const response = await fetch(`${BACKEND_URL}/v1/risk-profile/stream`, {
      method: "POST",
      headers: authHeaders("json"),
      body: JSON.stringify(toBackendPayload(req))
    });

    if (!response.ok) {
      const payload = await parseResponse(response);
      const detail = isRecord(payload)
        ? stringValue(payload.detail, response.statusText)
        : response.statusText;
      throw new Error(`${detail} (${response.status})`);
    }

    if (!response.body) {
      throw new Error("Streaming response body is unavailable");
    }

    useAppStore.getState().setError(null);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let isComplete = false;

    while (!isComplete) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      const chunks = buffer.split("\n\n");
      buffer = chunks.pop() ?? "";

      for (const chunk of chunks) {
        const lines = chunk.split("\n");
        for (const line of lines) {
          if (!line.startsWith("data: ")) {
            continue;
          }
          const raw = line.slice(6).trim();
          if (raw === "[DONE]") {
            isComplete = true;
            break;
          }
          const parsed = JSON.parse(raw) as unknown;
          const event = normalizeStreamEvent(parsed);
          if (event) {
            yield event;
          }
        }
        if (isComplete) {
          break;
        }
      }
    }
  } catch (error: unknown) {
    const message = requestErrorMessage(error);
    if (!isApiRouteUnavailableMessage(message)) {
      useAppStore.getState().setError(message);
    }
    throw new Error(message);
  }
}

export async function diagnosePatient(patientId: string): Promise<DiagnoseResponse> {
  const raw = await requestJson(`/v1/risk-profile/patient/${encodeURIComponent(patientId)}`, {
    method: "POST",
    headers: authHeaders("json"),
    body: JSON.stringify({})
  }, { suppressRouteUnavailableError: true });

  let featureImportance: Record<string, number> = {};
  try {
    const xai = await getPatientXai(patientId, "cognitive");
    featureImportance = xai.feature_importance;
  } catch {
    featureImportance = {};
  }

  return normalizeDiagnosePayload(raw, { mri: 0, eeg: 0, cog: 1 }, featureImportance);
}

export async function getDemoPatients(): Promise<string[]> {
  const payload = await requestJson("/v1/data/demo-patients", {
    method: "GET",
    headers: authHeaders("json")
  }, { suppressRouteUnavailableError: true });

  if (Array.isArray(payload)) {
    return payload
      .map((item) => (typeof item === "string" ? item : isRecord(item) ? item.patient_id : null))
      .filter((item): item is string => typeof item === "string" && item.length > 0);
  }

  if (!isRecord(payload)) {
    return [];
  }

  const fromPatients = Array.isArray(payload.patients)
    ? payload.patients
        .map((item) => (isRecord(item) ? optionalString(item.patient_id) : null))
        .filter((item): item is string => typeof item === "string" && item.length > 0)
    : [];
  const recommended = optionalString(payload.recommended_patient_id);
  return recommended && !fromPatients.includes(recommended)
    ? [recommended, ...fromPatients]
    : fromPatients;
}

export async function uploadMRI(file: File): Promise<UploadEmbeddingResponse> {
  const form = new FormData();
  form.append("file", file);
  const payload = await requestJson("/v1/upload/mri", {
    method: "POST",
    headers: authHeaders("form"),
    body: form
  }, { suppressRouteUnavailableError: true });
  const record = isRecord(payload) ? payload : {};
  return {
    status: stringValue(record.status, "ok"),
    embedding_dim: numberValue(record.embedding_dim, 0),
    embedding: numberArray(record.embedding),
    file_id: stringValue(record.file_id, file.name)
  };
}

export async function uploadEEG(file: File): Promise<UploadEmbeddingResponse> {
  const form = new FormData();
  form.append("file", file);
  const payload = await requestJson("/v1/upload/eeg", {
    method: "POST",
    headers: authHeaders("form"),
    body: form
  }, { suppressRouteUnavailableError: true });
  const record = isRecord(payload) ? payload : {};
  return {
    status: stringValue(record.status, "ok"),
    embedding_dim: numberValue(record.embedding_dim, 0),
    embedding: numberArray(record.embedding),
    file_id: stringValue(record.file_id, file.name)
  };
}

export async function queryKG(req: KGQueryRequest): Promise<KGQueryResponse> {
  const queryType = req.query_type === "stats" ? "history" : req.query_type;
  const payload = await requestJson("/v1/kg/query", {
    method: "POST",
    headers: authHeaders("json"),
    body: JSON.stringify({
      patient_id: req.patient_id,
      query_type: queryType,
      target_date: req.target_date,
      top_k: req.top_k
    })
  }, { suppressRouteUnavailableError: true });
  const jsonPayload = asJsonValue(payload);
  if (req.query_type === "stats" && isRecord(jsonPayload)) {
    return {
      payload: {
        ...jsonPayload,
        frontend_note: "The current backend has no stats query type; history is used as a safe KG summary."
      }
    };
  }
  return { payload: jsonPayload };
}

function normalizeEvalMetrics(raw: unknown): EvalMetrics {
  const record = isRecord(raw) ? raw : {};
  const nestedMetrics = isRecord(record.metrics) ? numberRecord(record.metrics) : {};
  const directMetrics = numberRecord(record);
  const metrics = { ...directMetrics, ...nestedMetrics };
  const status = stringValue(record.status, Object.keys(metrics).length > 0 ? "evaluated" : "no data");
  return {
    status,
    accuracy: metricValue(metrics, ["accuracy", "val_accuracy"]),
    f1: metricValue(metrics, ["macro_f1", "f1", "val_f1"]),
    auc: metricValue(metrics, ["auc_macro", "macro_auc", "auc", "val_auc"]),
    ece: metricValue(metrics, ["ece"]),
    n_samples: metricValue(metrics, ["n_samples"]),
    last_evaluated: optionalString(record.timestamp),
    checkpoint: optionalString(record.checkpoint) ?? optionalString(record.model_checkpoint),
    metrics,
    note: optionalString(record.note) ?? undefined
  };
}

function normalizeEvalHistoryEntry(raw: unknown): EvalHistoryEntry {
  const record = isRecord(raw) ? raw : {};
  const metrics = isRecord(record.metrics) ? numberRecord(record.metrics) : numberRecord(record);
  return {
    timestamp: optionalString(record.timestamp),
    accuracy: metricValue(metrics, ["accuracy", "val_accuracy"]),
    f1: metricValue(metrics, ["macro_f1", "f1", "val_f1"]),
    auc: metricValue(metrics, ["auc_macro", "macro_auc", "auc", "val_auc"]),
    checkpoint: optionalString(record.checkpoint) ?? optionalString(record.model_checkpoint),
    metrics
  };
}

export async function getEvalMetrics(): Promise<EvalMetrics> {
  const payload = await requestJson("/v1/eval/metrics", {
    method: "GET",
    headers: authHeaders("json")
  }, { suppressRouteUnavailableError: true });
  return normalizeEvalMetrics(payload);
}

export async function getEvalHistory(): Promise<EvalHistoryEntry[]> {
  const payload = await requestJson("/v1/eval/history", {
    method: "GET",
    headers: authHeaders("json")
  }, { suppressRouteUnavailableError: true });
  return Array.isArray(payload) ? payload.map(normalizeEvalHistoryEntry).slice(0, 10) : [];
}

export async function runEval(): Promise<EvalMetrics> {
  const payload = await requestJson("/v1/eval/run", {
    method: "POST",
    headers: authHeaders("json")
  }, { suppressRouteUnavailableError: true });
  return normalizeEvalMetrics(payload);
}

export async function getEvalReport(): Promise<EvalReport> {
  const payload = await requestJson("/v1/eval/report", {
    method: "GET",
    headers: authHeaders("json")
  }, { suppressRouteUnavailableError: true });
  const record = isRecord(payload) ? payload : {};
  return {
    status: stringValue(record.status, "unknown"),
    checkpoint: toJsonObject(record.checkpoint),
    evaluation: toJsonObject(record.evaluation),
    model_card: toJsonObject(record.model_card),
    scientific_claims: toJsonObject(record.scientific_claims)
  };
}

function normalizeModelRun(raw: unknown): ModelRun {
  const record = toJsonObject(raw);
  const metrics = isRecord(record.metrics) ? numberRecord(record.metrics) : {};
  return {
    run_id: stringValue(record.run_id, "unknown"),
    status: stringValue(record.status, "unknown"),
    accuracy: metricValue(metrics, ["accuracy", "val_accuracy"]),
    f1: metricValue(metrics, ["macro_f1", "f1", "val_f1"]),
    auc: metricValue(metrics, ["val_auc", "auc_macro", "macro_auc", "auc"]),
    timestamp: optionalString(record.timestamp),
    checkpoint_path: optionalString(record.checkpoint_path),
    promoted_at: optionalString(record.promoted_at) ?? optionalString(record.timestamp),
    metrics,
    raw: record
  };
}

export async function listModels(): Promise<ModelRun[]> {
  const payload = await requestJson("/v1/models", {
    method: "GET",
    headers: authHeaders("json")
  }, { suppressRouteUnavailableError: true });
  return Array.isArray(payload) ? payload.map(normalizeModelRun) : [];
}

export async function getProductionModel(): Promise<ModelRun | null> {
  const payload = await requestJson("/v1/models/production", {
    method: "GET",
    headers: authHeaders("json")
  }, { suppressRouteUnavailableError: true });
  if (isRecord(payload) && payload.status === "no_production_model") {
    return null;
  }
  return isRecord(payload) && Object.keys(payload).length > 0 ? normalizeModelRun(payload) : null;
}

export async function getCheckpointStatus(): Promise<CheckpointStatus> {
  const payload = await requestJson("/v1/models/checkpoint/status", {
    method: "GET",
    headers: authHeaders("json")
  }, { suppressRouteUnavailableError: true });
  const record = isRecord(payload) ? payload : {};
  return {
    status: stringValue(record.status, "unknown"),
    checkpoint: toJsonObject(record.checkpoint),
    loading: toJsonObject(record.loading),
    registry: toJsonObject(record.registry),
    evaluation: toJsonObject(record.evaluation),
    model_card: toJsonObject(record.model_card),
    scientific_claims: toJsonObject(record.scientific_claims)
  };
}

export async function promoteModel(runId: string): Promise<{ status: string }> {
  const payload = await requestJson(`/v1/models/${encodeURIComponent(runId)}/promote`, {
    method: "POST",
    headers: authHeaders("json")
  }, { suppressRouteUnavailableError: true });
  const record = isRecord(payload) ? payload : {};
  return { status: stringValue(record.status, "unknown") };
}

function normalizeXaiMethod(raw: unknown): XaiMethod {
  const record = isRecord(raw) ? raw : {};
  return {
    modality: stringValue(record.modality, "unknown"),
    status: stringValue(record.status, "unknown"),
    method: stringValue(record.method, "unknown"),
    artifact: optionalString(record.artifact) ?? undefined,
    source: optionalString(record.source) ?? undefined,
    requires_uploaded_data:
      typeof record.requires_uploaded_data === "boolean" ? record.requires_uploaded_data : undefined,
    validated_for_clinical_use:
      typeof record.validated_for_clinical_use === "boolean"
        ? record.validated_for_clinical_use
        : undefined,
    limitations: stringArray(record.limitations)
  };
}

export async function getXaiStatus(): Promise<XaiStatus> {
  const payload = await requestJson("/v1/xai/status", {
    method: "GET",
    headers: authHeaders("json")
  }, { suppressRouteUnavailableError: true });
  const record = isRecord(payload) ? payload : {};
  return {
    status: stringValue(record.status, "unknown"),
    runtime_mode: optionalString(record.runtime_mode),
    class_mode: optionalString(record.class_mode),
    methods: Array.isArray(record.methods) ? record.methods.map(normalizeXaiMethod) : [],
    interpretation_policy: toJsonObject(record.interpretation_policy)
  };
}

function modalityValue(value: unknown): "cognitive" | "mri" | "eeg" {
  return value === "mri" || value === "eeg" ? value : "cognitive";
}

export async function getPatientXai(
  patientId: string,
  modality: "cognitive" | "mri" | "eeg"
): Promise<XaiResult> {
  const payload = await requestJson(
    `/v1/xai/${encodeURIComponent(patientId)}?modality=${encodeURIComponent(modality)}`,
    {
      method: "GET",
      headers: authHeaders("json")
    },
    { suppressRouteUnavailableError: true }
  );
  const record = isRecord(payload) ? payload : {};
  return {
    patient_id: stringValue(record.patient_id, patientId),
    modality: modalityValue(record.modality),
    method: stringValue(record.method, "unknown"),
    feature_importance: numberRecord(record.feature_importance),
    text_summary: stringValue(record.text_summary, ""),
    xai_available: booleanValue(record.xai_available, false),
    note: optionalString(record.note),
    target_label: optionalString(record.target_label),
    method_contract: toJsonObject(record.method_contract),
    interpretation_policy: toJsonObject(record.interpretation_policy),
    privacy: toJsonObject(record.privacy)
  };
}

export async function getHealth(): Promise<HealthStatus> {
  const payload = await requestJson("/healthz", {
    method: "GET",
    headers: authHeaders("json")
  }, { suppressRouteUnavailableError: true });
  const record = toJsonObject(payload);
  return {
    status: stringValue(record.status, "unknown"),
    version: optionalString(record.version),
    uptime_seconds: finiteNumber(record.uptime_seconds),
    runtime: toJsonObject(record.runtime),
    models: toJsonObject(record.models),
    raw: record
  };
}

export async function getDataStatus(): Promise<DataStatus> {
  const payload = await requestJson("/v1/data/status", {
    method: "GET",
    headers: authHeaders("json")
  }, { suppressRouteUnavailableError: true });
  const record = isRecord(payload) ? payload : {};
  const summary = toJsonObject(record.summary);
  return {
    status: stringValue(record.status, "unknown"),
    source_kind: optionalString(record.source_kind),
    patient_count: finiteNumber(summary.row_count),
    recommended_patient_id: optionalString(record.recommended_patient_id),
    summary,
    files: toJsonObject(record.files),
    privacy: toJsonObject(record.privacy)
  };
}

export async function getModalitiesStatus(): Promise<ModalitiesStatus> {
  const payload = await requestJson("/v1/modalities/status", {
    method: "GET",
    headers: authHeaders("json")
  }, { suppressRouteUnavailableError: true });
  const record = toJsonObject(payload);
  return {
    status: stringValue(record.status, "unknown"),
    mri: toJsonObject(record.mri),
    eeg: toJsonObject(record.eeg),
    cognitive: toJsonObject(record.cognitive),
    raw: record
  };
}

export async function getGovernanceStatus(): Promise<GovernanceStatus> {
  const payload = await requestJson("/v1/governance/status", {
    method: "GET",
    headers: authHeaders("json")
  }, { suppressRouteUnavailableError: true });
  const record = isRecord(payload) ? payload : {};
  return {
    status: stringValue(record.status, "unknown"),
    privacy: toJsonObject(record.privacy),
    security: toJsonObject(record.security),
    scientific_disclosure: toJsonObject(record.scientific_disclosure)
  };
}

function readinessSeverity(status: string): DemoReadiness["severity"] {
  if (status === "demo_ready") {
    return "ok";
  }
  if (status === "needs_attention") {
    return "error";
  }
  return "warning";
}

function readinessMessage(status: string): string {
  if (status === "demo_ready") {
    return "System ready for demo";
  }
  if (status === "needs_attention") {
    return "System not ready";
  }
  return "System partially ready";
}

function normalizeReadinessCheck(raw: unknown): DemoReadinessCheck {
  const record = isRecord(raw) ? raw : {};
  return {
    id: stringValue(record.id, "check"),
    label: stringValue(record.label, "Readiness check"),
    status: stringValue(record.status, "unknown"),
    detail: stringValue(record.detail, ""),
    action: stringValue(record.action, ""),
    blocking: booleanValue(record.blocking, false)
  };
}

export async function getDemoReadiness(): Promise<DemoReadiness> {
  const payload = await requestJson("/v1/demo/readiness", {
    method: "GET",
    headers: authHeaders("json")
  }, { suppressRouteUnavailableError: true });
  const record = isRecord(payload) ? payload : {};
  const status = stringValue(record.status, "unknown");
  return {
    status,
    message: readinessMessage(status),
    severity: readinessSeverity(status),
    counts: toJsonObject(record.counts),
    checks: Array.isArray(record.checks) ? record.checks.map(normalizeReadinessCheck) : []
  };
}

export async function getPatientHistory(patientId: string): Promise<KGHistory> {
  const payload = await requestJson(`/v1/kg/patient/${encodeURIComponent(patientId)}/history`, {
    method: "GET",
    headers: authHeaders("json")
  }, { suppressRouteUnavailableError: true });
  const record = isRecord(payload) ? payload : {};
  return {
    patient_id: stringValue(record.patient_id, patientId),
    history: toJsonObjectArray(record.history),
    count: numberValue(record.count, 0)
  };
}

export async function getSimilarPatients(patientId: string): Promise<KGSimilar> {
  const payload = await requestJson(`/v1/kg/patient/${encodeURIComponent(patientId)}/similar`, {
    method: "GET",
    headers: authHeaders("json")
  }, { suppressRouteUnavailableError: true });
  if (!Array.isArray(payload)) {
    return [];
  }
  return payload
    .map((item) => {
      const record = isRecord(item) ? item : {};
      return {
        patient_id: stringValue(record.patient_id, "unknown"),
        score: finiteNumber(record.score) ?? finiteNumber(record.similarity_score),
        shared_features: stringArray(record.shared_features)
      };
    })
    .filter((item) => item.patient_id !== "unknown");
}

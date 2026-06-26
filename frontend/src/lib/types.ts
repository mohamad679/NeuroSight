export type DiagnosisLabel = "normal" | "mci" | "ad" | "ftd" | "lbd" | "vd";

export interface CognitiveScores {
  MMSE: number;
  MOCA: number;
  CDRSB: number;
  ADAS11: number;
  RAVLT_immediate: number;
  RAVLT_learning: number;
  FAQ: number;
  AGE: number;
}

export interface DiagnoseRequest extends CognitiveScores {
  query?: string;
  mri_file?: string | null;
  eeg_file?: string | null;
  mri_embedding?: number[];
  eeg_embedding?: number[];
  cog_embedding?: number[];
}

export interface ModalityWeights {
  mri: number;
  eeg: number;
  cog: number;
}

export interface DiagnoseResponse {
  diagnosis: DiagnosisLabel;
  confidence: number;
  report_text: string;
  requires_review: boolean;
  blocked_by_safety: boolean;
  modality_weights: ModalityWeights;
  feature_importance: Record<string, number>;
  modality_weights_source?: string;
  feature_importance_source?: string;
  backend_source?: "fastapi" | "gradio" | "synthetic";
  reliability_note?: string;
  model_mode?: string | null;
  checkpoint_id?: string | null;
  trained_on_real_data: boolean;
  clinical_validated: boolean;
  requires_expert_review: boolean;
  disclaimer: string;
  warnings: string[];
}

export interface UploadEmbeddingResponse {
  status: string;
  embedding_dim: number;
  embedding: number[];
  file_id?: string;
}

export interface KGQueryRequest {
  patient_id: string;
  query_type: "history" | "similar" | "snapshot" | "stats";
  target_date?: string;
  top_k?: number;
}

export type JsonValue = string | number | boolean | null | JsonObject | JsonValue[];

export interface JsonObject {
  [key: string]: JsonValue;
}

export interface KGQueryResponse {
  payload: JsonValue;
}

export interface BenchmarkClassMetric {
  className: DiagnosisLabel;
  f1: number;
  auc: number;
}

export type AgentState = "pending" | "running" | "completed";

export interface StreamEvent {
  agent: string;
  status: "running" | "completed" | "done";
  diagnosis?: string;
  confidence?: number;
  requires_review?: boolean;
  blocked?: boolean;
  report_text?: string;
}

export interface EvalMetrics {
  status: string;
  accuracy: number | null;
  f1: number | null;
  auc: number | null;
  ece: number | null;
  n_samples: number | null;
  last_evaluated: string | null;
  checkpoint: string | null;
  metrics: Record<string, number>;
  note?: string;
}

export interface EvalHistoryEntry {
  timestamp: string | null;
  accuracy: number | null;
  f1: number | null;
  auc: number | null;
  checkpoint: string | null;
  metrics: Record<string, number>;
}

export interface EvalReport {
  status: string;
  checkpoint: JsonObject;
  evaluation: JsonObject;
  model_card: JsonObject;
  scientific_claims: JsonObject;
}

export interface ModelRun {
  run_id: string;
  status: string;
  accuracy: number | null;
  f1: number | null;
  auc: number | null;
  timestamp: string | null;
  checkpoint_path: string | null;
  promoted_at: string | null;
  metrics: Record<string, number>;
  raw: JsonObject;
}

export interface CheckpointStatus {
  status: string;
  checkpoint: JsonObject;
  loading: JsonObject;
  registry: JsonObject;
  evaluation: JsonObject;
  model_card: JsonObject;
  scientific_claims: JsonObject;
}

export interface XaiMethod {
  modality: string;
  status: string;
  method: string;
  artifact?: string;
  source?: string;
  requires_uploaded_data?: boolean;
  validated_for_clinical_use?: boolean;
  limitations: string[];
}

export interface XaiStatus {
  status: string;
  runtime_mode: string | null;
  class_mode: string | null;
  methods: XaiMethod[];
  interpretation_policy: JsonObject;
}

export interface XaiResult {
  patient_id: string;
  modality: "cognitive" | "mri" | "eeg";
  method: string;
  feature_importance: Record<string, number>;
  text_summary: string;
  xai_available: boolean;
  note: string | null;
  target_label: string | null;
  method_contract: JsonObject;
  interpretation_policy: JsonObject;
  privacy: JsonObject;
}

export interface HealthStatus {
  status: string;
  version: string | null;
  uptime_seconds: number | null;
  runtime: JsonObject;
  models: JsonObject;
  raw: JsonObject;
}

export interface DataStatus {
  status: string;
  source_kind: string | null;
  patient_count: number | null;
  recommended_patient_id: string | null;
  summary: JsonObject;
  files: JsonObject;
  privacy: JsonObject;
}

export interface ModalitiesStatus {
  status: string;
  mri: JsonObject;
  eeg: JsonObject;
  cognitive: JsonObject;
  raw: JsonObject;
}

export interface GovernanceStatus {
  status: string;
  privacy: JsonObject;
  security: JsonObject;
  scientific_disclosure: JsonObject;
}

export interface DemoReadinessCheck {
  id: string;
  label: string;
  status: string;
  detail: string;
  action: string;
  blocking: boolean;
}

export interface DemoReadiness {
  status: string;
  message: string;
  severity: "ok" | "warning" | "error";
  counts: JsonObject;
  checks: DemoReadinessCheck[];
}

export interface KGHistory {
  patient_id: string;
  history: JsonObject[];
  count: number;
}

export interface KGSimilarEntry {
  patient_id: string;
  score: number | null;
  shared_features?: string[];
}

export type KGSimilar = KGSimilarEntry[];

const BACKEND = window.NEUROSIGHT_BACKEND || "";

const views = {
  overview: "Overview",
  demo: "Demo",
  data: "Data",
  diagnosis: "Risk Profiling",
  uploads: "Uploads",
  streaming: "Streaming",
  kg: "Knowledge Graph",
  xai: "XAI",
  trust: "Trust",
  evaluation: "Evaluation",
  models: "Models",
};

const embeddings = {
  mri: null,
  eeg: null,
  cog: null,
};

const embeddingLabels = {
  mri: "MRI",
  eeg: "EEG",
  cog: "Cognitive",
};

const streamAgents = ["supervisor", "kg_retriever", "report_writer", "safety_guardian"];

function $(selector) {
  return document.querySelector(selector);
}

function formatNumber(value) {
  const number = Number(value || 0);
  return new Intl.NumberFormat("en-US").format(number);
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${bytes} B`;
}

function formatUptime(seconds) {
  const total = Math.max(0, Number(seconds || 0));
  if (total < 60) return `${total.toFixed(0)}s`;
  if (total < 3600) return `${Math.floor(total / 60)}m ${Math.floor(total % 60)}s`;
  return `${Math.floor(total / 3600)}h ${Math.floor((total % 3600) / 60)}m`;
}

function formatLabel(value) {
  return String(value || "--")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function number(form, name) {
  return Number(form.get(name));
}

function collectCognitiveScores() {
  const form = new FormData($("#diagnosisForm"));
  return {
    mmse: number(form, "mmse"),
    moca: number(form, "moca"),
    cdrsb: number(form, "cdrsb"),
    adas11: number(form, "adas11"),
    ravlt_immediate: number(form, "ravlt_immediate"),
    ravlt_learning: number(form, "ravlt_learning"),
    faq: number(form, "faq"),
    age: number(form, "age"),
  };
}

function setView(name) {
  const viewName = views[name] ? name : "overview";
  document.querySelectorAll(".view").forEach((view) => {
    view.classList.toggle("is-active", view.id === viewName);
  });
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.classList.toggle("is-active", item.dataset.view === viewName);
  });
  $("#viewTitle").textContent = views[viewName];
}

function setStatus(selector, text, kind = "") {
  const element = $(selector);
  element.textContent = text;
  element.classList.toggle("is-ready", kind === "ready");
  element.classList.toggle("is-error", kind === "error");
}

function renderRuntimeContract(runtime = {}) {
  const container = $("#runtimeContract");
  if (!runtime || Object.keys(runtime).length === 0) {
    container.innerHTML = '<div class="empty-line">Runtime metadata was not returned by this backend.</div>';
    return;
  }

  const classes = Array.isArray(runtime.classes) ? runtime.classes : [];
  const adniClasses = Array.isArray(runtime.adni_style_classes) ? runtime.adni_style_classes : [];
  const syntheticClasses = Array.isArray(runtime.synthetic_demo_only_classes)
    ? runtime.synthetic_demo_only_classes
    : [];

  container.innerHTML = `
    <div class="runtime-summary">
      <strong>${escapeHtml(runtime.runtime_label || formatLabel(runtime.runtime_mode))}</strong>
      <span>${escapeHtml(runtime.runtime_description || "No runtime description returned.")}</span>
    </div>
    <div class="scope-block">
      <span class="scope-label">Class Mode</span>
      <strong>${escapeHtml(runtime.class_label || formatLabel(runtime.class_mode))}</strong>
      <p>${escapeHtml(runtime.class_description || "")}</p>
    </div>
    <div class="scope-chips">
      ${classes.map((item) => `<span class="scope-chip">${escapeHtml(item)}</span>`).join("")}
    </div>
    <div class="scope-note">
      ADNI-style: ${escapeHtml(adniClasses.join(", ") || "not specified")}. 
      Synthetic/demo only: ${escapeHtml(syntheticClasses.join(", ") || "none")}.
    </div>
  `;
}

function renderCapabilityCoverage(capabilities = {}) {
  const summary = capabilities.summary || {};
  const items = Array.isArray(capabilities.items) ? capabilities.items : [];
  const implemented = Number(summary.implemented || 0);
  const total = Number(summary.total || items.length || 0);
  const uiCoverage = Number(summary.ui_coverage_percent || 0);

  $("#capabilitySummary").textContent = total
    ? `${implemented}/${total} implemented · ${uiCoverage.toFixed(1)}% exposed in this local console.`
    : "Capability metadata was not returned by this backend.";

  $("#capabilityList").innerHTML = items.length
    ? items.map((item) => `
      <div class="capability-row">
        <div>
          <strong>${escapeHtml(item.label || item.id || "Capability")}</strong>
          <span>${escapeHtml(item.endpoint || "")}</span>
        </div>
        <div class="capability-badges">
          <span>${escapeHtml(item.ui_view || "API only")}</span>
          <span class="${item.status === "implemented" ? "loaded" : "not-loaded"}">
            ${escapeHtml(item.status || "unknown")}
          </span>
        </div>
      </div>
    `).join("")
    : '<div class="empty-line">No capability rows returned.</div>';
}

function sourceLabel(sourceKind) {
  return String(sourceKind || "--")
    .replace("synthetic_adni_like_demo", "Synthetic Demo")
    .replace("operator_supplied_adni_style", "ADNI Style");
}

function renderDataStatus(payload = {}, writeRaw = true) {
  const summary = payload.summary || {};
  const schema = payload.schema || {};
  const files = payload.files || {};
  const paths = payload.paths || {};
  const labels = summary.label_distribution || {};
  const missingColumns = Array.isArray(schema.missing_columns) ? schema.missing_columns : [];

  $("#dataMetricStatus").textContent = String(payload.status || "--").toUpperCase();
  $("#dataMetricRows").textContent = formatNumber(summary.row_count);
  $("#dataMetricSource").textContent = sourceLabel(payload.source_kind);
  $("#dataMetricPatient").textContent = payload.recommended_patient_id || "--";

  const labelRows = Object.entries(labels).map(([label, count]) => `
    <span class="scope-chip">${escapeHtml(label)} ${formatNumber(count)}</span>
  `).join("");
  const schemaStatus = missingColumns.length
    ? `Missing columns: ${missingColumns.join(", ")}`
    : "Schema ready";

  $("#dataSummary").innerHTML = `
    <div class="data-card">
      <strong>${escapeHtml(sourceLabel(payload.source_kind))}</strong>
      <span>${escapeHtml(payload.privacy?.notice || "No data privacy notice returned.")}</span>
    </div>
    <div class="data-card">
      <strong>${escapeHtml(schemaStatus)}</strong>
      <span>CSV: ${escapeHtml(paths.csv || "not configured")}</span>
    </div>
    <div class="data-card">
      <strong>Modalities</strong>
      <span>
        Cognitive rows: ${formatNumber(summary.row_count)} ·
        MRI files: ${formatNumber(summary.mri_file_count)} ·
        EEG files: ${formatNumber(summary.eeg_file_count)}
      </span>
    </div>
    <div class="data-card">
      <strong>Class Scope</strong>
      <span>
        ADNI-style rows: ${formatNumber(summary.adni_style_count)} ·
        Synthetic-only rows: ${formatNumber(summary.synthetic_demo_only_count)} ·
        Unknown labels: ${formatNumber(summary.unknown_label_count)}
      </span>
      <div class="scope-chips compact">${labelRows || '<span class="scope-chip">No labels</span>'}</div>
    </div>
    <div class="data-card">
      <strong>Files</strong>
      <span>
        CSV ${files.csv_exists ? "found" : "missing"} ·
        MRI directory ${files.mri_dir_exists ? "found" : "missing"} ·
        EEG directory ${files.eeg_dir_exists ? "found" : "missing"}
      </span>
    </div>
  `;

  if (writeRaw) {
    $("#dataRaw").textContent = JSON.stringify(payload, null, 2);
  }
}

function demoPatientRows(patients) {
  return patients.map((patient) => ({
    patient_id: patient.patient_id,
    diagnosis: patient.diagnosis_label,
    scope: patient.class_scope,
    age: patient.age,
    sex: patient.sex,
    mmse: patient.scores?.mmse,
    moca: patient.scores?.moca,
    cdrsb: patient.scores?.cdrsb,
    adas11: patient.scores?.adas11,
    ravlt_immediate: patient.scores?.ravlt_immediate,
    ravlt_learning: patient.scores?.ravlt_learning,
    faq: patient.scores?.faq,
    mri: patient.modalities?.mri_npy ? "available" : "missing",
    eeg: patient.modalities?.eeg_npy ? "available" : "missing",
  }));
}

function renderDemoPatients(payload = {}) {
  const patients = Array.isArray(payload.patients) ? payload.patients : [];
  $("#dataRaw").textContent = JSON.stringify(payload, null, 2);
  if (payload.recommended_patient_id) {
    $("#dataMetricPatient").textContent = payload.recommended_patient_id;
  }
  renderTable("#demoPatientsTable", demoPatientRows(patients));
}

function readinessBadge(value) {
  return value ? "ready" : "missing";
}

function renderDependencyBadges(dependencies = {}) {
  return Object.entries(dependencies).map(([name, available]) => `
    <span class="${available ? "loaded" : "not-loaded"}">${escapeHtml(name)} ${readinessBadge(available)}</span>
  `).join("");
}

function renderModalityStatus(payload = {}) {
  const mri = payload.mri || {};
  const eeg = payload.eeg || {};
  const cognitive = payload.cognitive || {};

  $("#modalityContract").innerHTML = `
    <div class="modality-card">
      <div>
        <span class="scope-label">MRI</span>
        <strong>${escapeHtml((mri.supported_formats || []).join(", ") || "No MRI formats returned")}</strong>
        <p>Encoder input ${escapeHtml((mri.model_tensor_shape || []).join(" x ") || "--")} · embedding ${escapeHtml(mri.embedding_dim || "--")}d.</p>
        <p>${escapeHtml((mri.accepted_array_shapes || []).join(", "))}</p>
      </div>
      <div class="capability-badges">${renderDependencyBadges(mri.optional_dependencies)}</div>
    </div>
    <div class="modality-card">
      <div>
        <span class="scope-label">EEG</span>
        <strong>${escapeHtml((eeg.supported_formats || []).join(", ") || "No EEG formats returned")}</strong>
        <p>Encoder input ${escapeHtml((eeg.model_tensor_shape || []).join(" x ") || "--")} · embedding ${escapeHtml(eeg.embedding_dim || "--")}d.</p>
        <p>${escapeHtml((eeg.accepted_array_shapes || []).join(", "))}</p>
      </div>
      <div class="capability-badges">${renderDependencyBadges(eeg.optional_dependencies)}</div>
    </div>
    <div class="modality-card">
      <div>
        <span class="scope-label">Cognitive</span>
        <strong>${escapeHtml((cognitive.required_or_defaulted_features || []).join(", ") || "No cognitive schema returned")}</strong>
        <p>Encoder input ${escapeHtml((cognitive.model_tensor_shape || []).join(" x ") || "--")} · embedding ${escapeHtml(cognitive.embedding_dim || "--")}d.</p>
      </div>
    </div>
    <div class="data-card">
      <strong>Scientific Notice</strong>
      <span>${escapeHtml(payload.scientific_notice || "No modality notice returned.")}</span>
    </div>
  `;
}

function summarizeCheckpoint(payload = {}) {
  const checkpoint = payload.checkpoint || {};
  const loading = payload.loading || {};
  if (!checkpoint.exists) {
    return `Checkpoint missing at ${checkpoint.path || "unconfigured path"}.`;
  }
  const loaded = loading.loaded ? "loaded into runtime" : "available on disk";
  const size = checkpoint.size_mb ? `${checkpoint.size_mb} MB` : "unknown size";
  return `${formatLabel(payload.status)} · ${loaded} · ${size}.`;
}

function renderCheckpointStatus(payload = {}) {
  const checkpoint = payload.checkpoint || {};
  const loading = payload.loading || {};
  const registry = payload.registry || {};
  const evaluation = payload.evaluation || {};
  const claims = payload.scientific_claims || {};
  const bestRun = registry.best_run || {};
  const bestMetrics = bestRun.metrics || {};

  $("#checkpointSummary").textContent = summarizeCheckpoint(payload);
  $("#checkpointContract").innerHTML = `
    <div class="data-card">
      <strong>Artifact</strong>
      <span>${escapeHtml(checkpoint.path || "not configured")} · ${checkpoint.exists ? "found" : "missing"} · ${escapeHtml(checkpoint.size_mb || 0)} MB</span>
    </div>
    <div class="data-card">
      <strong>Runtime Loading</strong>
      <span>
        ${loading.enabled ? "Enabled" : "Disabled"} via ${escapeHtml(loading.env_var || "env")} ·
        ${loading.loaded ? "Loaded" : "Not loaded"}${loading.error ? ` · ${escapeHtml(loading.error)}` : ""}
      </span>
    </div>
    <div class="data-card">
      <strong>Registry</strong>
      <span>
        ${formatNumber(registry.run_count)} run${Number(registry.run_count || 0) === 1 ? "" : "s"} ·
        production ${escapeHtml(registry.production_run_id || "none")} ·
        best ${escapeHtml(bestRun.run_id || "none")} ${bestMetrics.val_auc ? `· val_auc ${Number(bestMetrics.val_auc).toFixed(4)}` : ""}
      </span>
    </div>
    <div class="data-card">
      <strong>Evaluation</strong>
      <span>
        ${evaluation.available ? "Available" : "Missing"} ·
        macro_auc ${evaluation.metrics?.macro_auc ?? "--"} ·
        ece ${evaluation.metrics?.ece ?? "--"}
      </span>
    </div>
    <div class="data-card">
      <strong>Scientific Claims</strong>
      <span>${escapeHtml(claims.notice || "No scientific-claims notice returned.")}</span>
    </div>
  `;
}

function renderXaiStatus(payload = {}, writeRaw = true) {
  const methods = Array.isArray(payload.methods) ? payload.methods : [];
  const policy = payload.interpretation_policy || {};

  $("#xaiMethodGrid").innerHTML = methods.length
    ? methods.map((method) => `
      <div class="data-card">
        <strong>${escapeHtml(formatLabel(method.modality))}</strong>
        <span>
          ${escapeHtml(method.method || "method unavailable")} ·
          ${escapeHtml(formatLabel(method.status || "unknown"))} ·
          ${method.validated_for_clinical_use ? "clinical validation claimed" : "no clinical validation claimed"}
        </span>
        <div class="scope-chips compact">
          <span class="scope-chip">${escapeHtml(method.artifact || "artifact")}</span>
          <span class="scope-chip">${method.requires_uploaded_data ? "upload required" : "no upload required"}</span>
        </div>
      </div>
    `).join("")
    : '<div class="empty-line">No XAI methods returned.</div>';

  $("#xaiPolicySummary").innerHTML = `
    <div class="policy-stack">
      <strong>${policy.clinical_use_allowed ? "Clinical use allowed" : "Clinical use blocked"}</strong>
      <span>${escapeHtml(policy.primary_notice || "No interpretation policy returned.")}</span>
      <span>Human review: ${policy.requires_human_review ? "required" : "not specified"} · Causal claims: ${policy.causal_claims_allowed ? "allowed" : "blocked"}</span>
    </div>
  `;

  if (writeRaw) {
    $("#xaiRaw").textContent = JSON.stringify(payload, null, 2);
  }
}

function renderTrustStatus(payload = {}, writeRaw = true) {
  const privacy = payload.privacy || {};
  const security = payload.security || {};
  const disclosure = payload.scientific_disclosure || {};
  const controls = security.upload_controls || {};
  const requirements = Array.isArray(disclosure.required_before_real_claims)
    ? disclosure.required_before_real_claims
    : [];

  $("#trustMetricAdni").textContent = privacy.private_adni_in_repository ? "Present" : "Absent";
  $("#trustMetricClinical").textContent = privacy.clinical_use_allowed ? "Allowed" : "Blocked";
  $("#trustMetricAuth").textContent = security.api_key_header || "--";
  $("#trustMetricUpload").textContent = formatBytes(controls.max_upload_bytes);

  $("#trustPrivacy").innerHTML = `
    <div class="data-card">
      <strong>Repository Data</strong>
      <span>${escapeHtml(privacy.demo_data_policy || "not specified")} · private ADNI ${privacy.private_adni_in_repository ? "present" : "absent"}</span>
    </div>
    <div class="data-card">
      <strong>Operator Data</strong>
      <span>${escapeHtml(privacy.operator_supplied_data_policy || "not specified")}</span>
    </div>
    <div class="data-card">
      <strong>Retention</strong>
      <span>${escapeHtml(privacy.upload_retention || "not specified")}</span>
    </div>
    <div class="data-card">
      <strong>PHI Policy</strong>
      <span>${escapeHtml(privacy.phi_policy || "not specified")}</span>
    </div>
  `;

  $("#trustSecurity").innerHTML = `
    <div class="data-card">
      <strong>Protected Scope</strong>
      <span>${escapeHtml(security.protected_endpoint_scope || "not specified")}</span>
    </div>
    <div class="data-card">
      <strong>CORS</strong>
      <span>${escapeHtml((security.cors_allowed_origins || []).join(", ") || "not configured")}</span>
    </div>
    <div class="data-card">
      <strong>Upload Guards</strong>
      <span>
        ${formatBytes(controls.max_upload_bytes)} upload cap ·
        ${formatNumber(controls.max_dicom_zip_members)} DICOM members ·
        pickle ${controls.numpy_pickle_disabled ? "disabled" : "not specified"}
      </span>
    </div>
    <div class="data-card">
      <strong>Observability</strong>
      <span>${escapeHtml((security.request_observability_headers || []).join(", ") || "not specified")} · rate limiting ${security.rate_limiting_available ? "available" : "not installed"}</span>
    </div>
  `;

  $("#trustDisclosure").innerHTML = `
    <div class="data-card">
      <strong>Validation</strong>
      <span>${disclosure.validated_clinically ? "Clinical validation claimed" : "No clinical validation claimed"}</span>
    </div>
    <div class="data-card">
      <strong>Class Policy</strong>
      <span>${escapeHtml(disclosure.label_policy || "not specified")}</span>
    </div>
    <div class="scope-chips">
      ${requirements.map((item) => `<span class="scope-chip">${escapeHtml(item)}</span>`).join("")}
    </div>
  `;

  if (writeRaw) {
    $("#trustRaw").textContent = JSON.stringify(payload, null, 2);
  }
}

function statusLabel(status) {
  return String(status || "--")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function renderDemoReadiness(payload = {}, writeRaw = true) {
  const counts = payload.counts || {};
  const checks = Array.isArray(payload.checks) ? payload.checks : [];
  const flow = Array.isArray(payload.recommended_ui_flow) ? payload.recommended_ui_flow : [];
  const adniRun = payload.adni_style_private_run || {};
  const requiredEnv = Array.isArray(adniRun.required_env) ? adniRun.required_env : [];
  const optionalEnv = Array.isArray(adniRun.optional_checkpoint_env) ? adniRun.optional_checkpoint_env : [];

  $("#demoMetricStatus").textContent = statusLabel(payload.status);
  $("#demoMetricReady").textContent = `${formatNumber(counts.ready)} / ${formatNumber(counts.total)}`;
  $("#demoMetricWarnings").textContent = formatNumber(Number(counts.warning || 0) + Number(counts.action_required || 0));
  $("#demoMetricPatient").textContent = payload.recommended_patient_id || "--";

  $("#demoChecklist").innerHTML = checks.length
    ? checks.map((item) => `
      <div class="readiness-row">
        <div>
          <strong>${escapeHtml(item.label || item.id || "Check")}</strong>
          <span>${escapeHtml(item.detail || "")}</span>
          ${item.action ? `<span class="readiness-action">${escapeHtml(item.action)}</span>` : ""}
        </div>
        <span class="check-status ${escapeHtml(item.status || "unknown")}">${escapeHtml(statusLabel(item.status))}</span>
      </div>
    `).join("")
    : '<div class="empty-line">No readiness checks returned.</div>';

  $("#demoFlow").innerHTML = flow.length
    ? flow.map((step) => `
      <div class="data-card">
        <strong>${formatNumber(step.order)}. ${escapeHtml(step.view || "View")} · ${escapeHtml(step.action || "Action")}</strong>
        <span>${escapeHtml(step.expected || "")}</span>
      </div>
    `).join("")
    : '<div class="empty-line">No recommended demo flow returned.</div>';

  $("#demoAdniRun").innerHTML = `
    <div class="data-card">
      <strong>Recommended Mode</strong>
      <span>${escapeHtml(adniRun.recommended_class_mode || "three_class_adni")}</span>
    </div>
    <div class="data-card">
      <strong>Required Environment</strong>
      <span>${escapeHtml(requiredEnv.join(" · ") || "No required environment returned.")}</span>
    </div>
    <div class="data-card">
      <strong>Optional Checkpoint Environment</strong>
      <span>${escapeHtml(optionalEnv.join(" · ") || "No optional checkpoint environment returned.")}</span>
    </div>
    <div class="data-card">
      <strong>Warning</strong>
      <span>${escapeHtml(adniRun.warning || "Keep private data outside public repositories.")}</span>
    </div>
  `;

  if (writeRaw) {
    $("#demoRaw").textContent = JSON.stringify(payload, null, 2);
  }
}

function renderHealth(payload) {
  const status = String(payload.status || "unknown");
  const ok = status.toLowerCase() === "ok";
  const pill = $("#statusPill");

  pill.textContent = ok ? "Online" : "Attention";
  pill.classList.toggle("is-ok", ok);
  pill.classList.toggle("is-bad", !ok);

  $("#metricStatus").textContent = status.toUpperCase();
  $("#metricVersion").textContent = payload.version || "--";
  $("#metricUptime").textContent = formatUptime(payload.uptime_seconds);

  const kg = payload.kg || {};
  $("#metricKg").textContent = `${formatNumber(kg.nodes)} / ${formatNumber(kg.edges)}`;
  const runtime = payload.runtime || {};
  const capabilities = payload.capabilities || {};
  const coverage = capabilities.summary || {};
  $("#metricRuntime").textContent = formatLabel(runtime.runtime_mode || "--");
  $("#metricCoverage").textContent = typeof coverage.ui_coverage_percent === "number"
    ? `${coverage.ui_coverage_percent.toFixed(0)}%`
    : "--";
  renderRuntimeContract(runtime);
  renderCapabilityCoverage(capabilities);
  if (payload.data) {
    renderDataStatus(payload.data, false);
  }
  if (payload.modalities) {
    renderModalityStatus(payload.modalities);
  }
  if (payload.checkpoint) {
    renderCheckpointStatus(payload.checkpoint);
  }
  if (payload.xai) {
    renderXaiStatus(payload.xai, false);
  }
  if (payload.governance) {
    renderTrustStatus(payload.governance, false);
  }
  if (payload.demo_readiness) {
    renderDemoReadiness(payload.demo_readiness, false);
  }

  const models = payload.models || {};
  const modelRows = Object.entries(models).map(([name, model]) => {
    const loaded = Boolean(model && model.loaded);
    const params = model && typeof model.params !== "undefined" ? formatNumber(model.params) : "0";
    return `
      <div class="model-row">
        <div>
          <strong>${name.replaceAll("_", " ")}</strong>
          <span>${params} parameters</span>
        </div>
        <span class="${loaded ? "loaded" : "not-loaded"}">${loaded ? "Loaded" : "Missing"}</span>
      </div>
    `;
  });
  $("#modelList").innerHTML = modelRows.length
    ? modelRows.join("")
    : '<div class="empty-line">No model status returned.</div>';

  const production = payload.production_model || {};
  if (production.run_id) {
    $("#productionSummary").textContent = `Run ${production.run_id} · val_auc ${production.val_auc ?? "--"}`;
    $("#productionModelSummary").textContent = `Run ${production.run_id} · val_auc ${production.val_auc ?? "--"}`;
  } else {
    $("#productionSummary").textContent = "No production model metadata returned.";
    $("#productionModelSummary").textContent = "No production model metadata returned.";
  }

  $("#healthRaw").textContent = JSON.stringify(payload, null, 2);
}

function renderHealthError(error) {
  const pill = $("#statusPill");
  pill.textContent = "Offline";
  pill.classList.remove("is-ok");
  pill.classList.add("is-bad");
  $("#metricStatus").textContent = "ERROR";
  $("#metricRuntime").textContent = "--";
  $("#metricCoverage").textContent = "--";
  renderRuntimeContract({});
  renderCapabilityCoverage({});
  $("#healthRaw").textContent = String(error);
}

async function checkHealth() {
  const button = $("#healthBtn");
  button.disabled = true;
  button.textContent = "Checking...";
  $("#healthRaw").textContent = "Checking backend...";
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    renderHealth(data);
  } catch (error) {
    renderHealthError(error);
  } finally {
    button.disabled = false;
    button.textContent = "Check Backend";
  }
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  const text = await response.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = {detail: text};
  }
  if (!response.ok) throw new Error(data.detail || text || response.statusText);
  return data;
}

async function getJson(url) {
  const response = await fetch(url);
  const text = await response.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = {detail: text};
  }
  if (!response.ok) throw new Error(data.detail || text || response.statusText);
  return data;
}

async function postFormData(url, formData) {
  const response = await fetch(url, {
    method: "POST",
    body: formData,
  });
  const text = await response.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = {detail: text};
  }
  if (!response.ok) throw new Error(data.detail || text || response.statusText);
  return data;
}

async function withButton(button, loadingText, task) {
  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = loadingText;
  try {
    return await task();
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
}

function displayValue(value) {
  if (value === null || typeof value === "undefined") return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function normalizeRows(data) {
  if (Array.isArray(data)) return data;
  if (data && Array.isArray(data.results)) return data.results;
  if (data && Array.isArray(data.history)) return data.history;
  if (data && data.results && typeof data.results === "object") {
    return Object.entries(data.results).map(([key, value]) => ({key, value}));
  }
  if (data && typeof data === "object") {
    return Object.entries(data).map(([key, value]) => ({key, value}));
  }
  return [{value: data}];
}

function renderTable(selector, data) {
  const container = $(selector);
  const rows = normalizeRows(data);
  if (rows.length === 0) {
    container.innerHTML = '<div class="empty-line">No rows returned.</div>';
    return;
  }

  const keys = Array.from(rows.reduce((set, row) => {
    if (row && typeof row === "object" && !Array.isArray(row)) {
      Object.keys(row).forEach((key) => set.add(key));
    } else {
      set.add("value");
    }
    return set;
  }, new Set()));

  container.innerHTML = `
    <table class="data-table">
      <thead>
        <tr>${keys.map((key) => `<th>${escapeHtml(key)}</th>`).join("")}</tr>
      </thead>
      <tbody>
        ${rows.map((row) => `
          <tr>
            ${keys.map((key) => `<td>${escapeHtml(displayValue(row && typeof row === "object" ? row[key] : row))}</td>`).join("")}
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function renderModelsTable(runs) {
  const rows = Array.isArray(runs) ? runs : normalizeRows(runs);
  const container = $("#modelsTable");
  if (rows.length === 0) {
    container.innerHTML = '<div class="empty-line">No model runs returned.</div>';
    return;
  }

  const preferred = ["run_id", "status", "timestamp", "metrics", "model_checkpoint"];
  const discovered = Array.from(rows.reduce((set, row) => {
    if (row && typeof row === "object" && !Array.isArray(row)) {
      Object.keys(row).forEach((key) => set.add(key));
    }
    return set;
  }, new Set()));
  const keys = preferred.filter((key) => discovered.includes(key));
  discovered.forEach((key) => {
    if (!keys.includes(key) && keys.length < 6) keys.push(key);
  });

  container.innerHTML = `
    <table class="data-table">
      <thead>
        <tr>${keys.map((key) => `<th>${escapeHtml(key)}</th>`).join("")}<th>action</th></tr>
      </thead>
      <tbody>
        ${rows.map((row) => {
          const runId = row && typeof row === "object" ? String(row.run_id || "") : "";
          return `
            <tr>
              ${keys.map((key) => `<td>${escapeHtml(displayValue(row && typeof row === "object" ? row[key] : row))}</td>`).join("")}
              <td>${runId ? `<button class="table-action" type="button" data-promote-run="${escapeHtml(runId)}">Promote</button>` : ""}</td>
            </tr>
          `;
        }).join("")}
      </tbody>
    </table>
  `;
}

function setResult(payload) {
  const diagnosis = String(payload.diagnosis || "UNKNOWN").toUpperCase();
  const confidence = Math.max(0, Math.min(1, Number(payload.confidence || 0)));
  const badge = $("#badge");
  const bar = $("#bar");

  badge.textContent = diagnosis;
  badge.style.background = diagnosis === "NORMAL"
    ? "var(--green)"
    : diagnosis === "MCI"
      ? "var(--amber)"
      : "var(--red)";
  bar.style.width = `${(confidence * 100).toFixed(1)}%`;
  bar.style.background = confidence < 0.5
    ? "var(--red)"
    : confidence < 0.75
      ? "var(--amber)"
      : "var(--green)";
  $("#confidence").textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
  $("#report").textContent = payload.report_text || JSON.stringify(payload, null, 2);
}

function setEmbedding(kind, payload) {
  embeddings[kind] = {
    embedding_dim: Number(payload.embedding_dim || 0),
    embedding: Array.isArray(payload.embedding) ? payload.embedding : [],
    source_status: payload.status || "ok",
    top_prediction: payload.top_prediction,
    confidence: payload.confidence,
    unimodal_probs: payload.unimodal_probs,
  };
  renderEmbeddings();
}

function clearEmbedding(kind) {
  embeddings[kind] = null;
  const statusMap = {
    mri: "#mriUploadStatus",
    eeg: "#eegUploadStatus",
    cog: "#cognitiveUploadStatus",
  };
  const label = embeddingLabels[kind];
  setStatus(statusMap[kind], `No ${label.toLowerCase()} embedding.`);
  renderEmbeddings();
}

function embeddingPreview(kind, item) {
  if (!item) return null;
  const previewValues = item.embedding.slice(0, 8).map((value) => Number(value).toPrecision(5));
  const preview = {
    modality: kind,
    embedding_dim: item.embedding_dim,
    preview: previewValues,
    hidden_values: Math.max(0, item.embedding.length - previewValues.length),
  };
  if (item.top_prediction) {
    preview.top_prediction = item.top_prediction;
    preview.confidence = item.confidence;
  }
  return preview;
}

function renderEmbeddings() {
  const active = Object.entries(embeddings).filter(([, item]) => item);
  const summary = $("#embeddingSummary");
  const preview = $("#embeddingPreview");
  const diagnosisStatus = $("#diagnosisEmbeddingStatus");

  if (active.length === 0) {
    summary.innerHTML = '<div class="empty-line">No active embeddings.</div>';
    preview.textContent = "No embeddings generated.";
  } else {
    summary.innerHTML = active.map(([kind, item]) => `
      <div class="embedding-row">
        <div>
          <strong>${embeddingLabels[kind]}</strong>
          <span>${item.embedding_dim} dimensions · ${item.embedding.length} values ready</span>
        </div>
        <button type="button" data-clear-embedding="${kind}">Clear</button>
      </div>
    `).join("");
    const compact = Object.fromEntries(
      active.map(([kind, item]) => [kind, embeddingPreview(kind, item)])
    );
    preview.textContent = JSON.stringify(compact, null, 2);
  }

  diagnosisStatus.innerHTML = Object.entries(embeddingLabels).map(([kind, label]) => {
    const item = embeddings[kind];
    const text = item ? `${label} ${item.embedding_dim}d` : `${label} empty`;
    return `<span class="mini-status ${item ? "is-ready" : ""}">${text}</span>`;
  }).join("");
}

function renderStreamTimeline(statuses = {}) {
  $("#streamTimeline").innerHTML = streamAgents.map((agent) => {
    const status = statuses[agent] || "pending";
    return `
      <div class="timeline-item">
        <div>
          <strong>${escapeHtml(agent.replaceAll("_", " "))}</strong>
          <span>Research evaluation orchestration step</span>
        </div>
        <span class="timeline-badge ${escapeHtml(status)}">${escapeHtml(status)}</span>
      </div>
    `;
  }).join("");
}

function appendStreamLine(text) {
  const raw = $("#streamRaw");
  raw.textContent = raw.textContent === "No stream started." ? text : `${raw.textContent}\n${text}`;
}

async function runStreamingDiagnosis(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const button = $("#streamBtn");
  const statuses = Object.fromEntries(streamAgents.map((agent) => [agent, "pending"]));
  const payload = {
    patient_id: String(form.get("patient_id") || "").trim(),
    query: String(form.get("query") || "Provide model-generated risk profile summary."),
  };
  if (form.get("use_scores")) {
    payload.cognitive_scores = collectCognitiveScores();
  }

  $("#streamRaw").textContent = "Opening stream...";
  renderStreamTimeline(statuses);

  await withButton(button, "Streaming...", async () => {
    const response = await fetch("/api/diagnose/stream", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || response.statusText);
    }
    if (!response.body) {
      appendStreamLine(await response.text());
      return;
    }

    $("#streamRaw").textContent = "";
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const eventText = trimmed.slice(5).trim();
        appendStreamLine(eventText);
        if (eventText === "[DONE]") continue;
        try {
          const eventPayload = JSON.parse(eventText);
          if (eventPayload.agent && streamAgents.includes(eventPayload.agent)) {
            statuses[eventPayload.agent] = eventPayload.status || "updated";
            renderStreamTimeline(statuses);
          }
        } catch {
          appendStreamLine("Unable to parse stream event.");
        }
      }
    }
  }).catch((error) => {
    appendStreamLine(String(error));
  });
}

async function uploadEmbedding(event, kind) {
  event.preventDefault();
  const form = event.currentTarget;
  const input = form.querySelector('input[type="file"]');
  const file = input.files && input.files[0];
  const statusMap = {
    mri: "#mriUploadStatus",
    eeg: "#eegUploadStatus",
  };

  if (!file) {
    setStatus(statusMap[kind], "Select a file first.", "error");
    return;
  }

  const button = form.querySelector("button");
  const originalText = button.textContent;
  const formData = new FormData();
  formData.append("file", file);

  button.disabled = true;
  button.textContent = "Uploading...";
  setStatus(statusMap[kind], `Uploading ${file.name}...`);

  try {
    const payload = await postFormData(`/api/upload/${kind}`, formData);
    setEmbedding(kind, payload);
    setStatus(
      statusMap[kind],
      `${embeddingLabels[kind]} embedding ready · ${payload.embedding_dim} dimensions.`,
      "ready",
    );
  } catch (error) {
    clearEmbedding(kind);
    setStatus(statusMap[kind], String(error), "error");
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
}

async function encodeCognitiveScores() {
  const button = $("#cognitiveUploadBtn");
  const originalText = button.textContent;

  button.disabled = true;
  button.textContent = "Encoding...";
  setStatus("#cognitiveUploadStatus", "Encoding cognitive scores...");

  try {
    const payload = await postJson("/api/upload/cognitive", {scores: collectCognitiveScores()});
    setEmbedding("cog", payload);
    const prediction = payload.top_prediction ? ` · ${payload.top_prediction}` : "";
    setStatus(
      "#cognitiveUploadStatus",
      `Cognitive embedding ready · ${payload.embedding_dim} dimensions${prediction}.`,
      "ready",
    );
  } catch (error) {
    clearEmbedding("cog");
    setStatus("#cognitiveUploadStatus", String(error), "error");
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
}

async function runKgQuery(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const button = $("#kgBtn");
  const payload = {
    patient_id: String(form.get("patient_id") || "").trim(),
    query_type: String(form.get("query_type") || "history"),
  };
  const targetDate = String(form.get("target_date") || "").trim();
  if (targetDate) payload.target_date = targetDate;
  if (payload.query_type === "similar") {
    payload.top_k = Number(form.get("top_k") || 5);
  }

  $("#kgRaw").textContent = "Querying knowledge graph...";
  await withButton(button, "Querying...", async () => {
    const data = await postJson("/api/kg/query", payload);
    $("#kgRaw").textContent = JSON.stringify(data, null, 2);
    renderTable("#kgTable", data.results || data);
  }).catch((error) => {
    $("#kgRaw").textContent = String(error);
    $("#kgTable").innerHTML = '<div class="empty-line">KG query failed.</div>';
  });
}

function renderXaiChart(payload) {
  const chart = $("#xaiChart");
  const importance = payload.feature_importance || {};
  const entries = Object.entries(importance)
    .map(([name, value]) => [name, Number(value || 0)])
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));

  if (entries.length === 0) {
    chart.innerHTML = `<div class="empty-line">${escapeHtml(payload.note || "No feature importance returned.")}</div>`;
    return;
  }

  const max = Math.max(...entries.map(([, value]) => Math.abs(value)), 0.000001);
  chart.innerHTML = entries.map(([name, value]) => `
    <div class="bar-row">
      <div class="bar-name">${escapeHtml(name)}</div>
      <div class="bar-track">
        <div class="bar-fill" style="width: ${(Math.abs(value) / max * 100).toFixed(1)}%"></div>
      </div>
      <div class="bar-value">${escapeHtml(value.toFixed(4))}</div>
    </div>
  `).join("");
}

async function runXaiQuery(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const button = $("#xaiBtn");
  const params = new URLSearchParams({
    patient_id: String(form.get("patient_id") || "").trim(),
    modality: String(form.get("modality") || "cognitive"),
  });
  const targetClass = String(form.get("target_class") || "").trim();
  if (targetClass) params.set("target_class", targetClass);

  $("#xaiRaw").textContent = "Generating explanation...";
  $("#xaiSummary").textContent = "Generating explanation...";
  await withButton(button, "Generating...", async () => {
    const data = await getJson(`/api/xai?${params.toString()}`);
    $("#xaiRaw").textContent = JSON.stringify(data, null, 2);
    $("#xaiSummary").textContent = data.text_summary || data.note || "No explanation summary returned.";
    renderXaiChart(data);
  }).catch((error) => {
    $("#xaiRaw").textContent = String(error);
    $("#xaiSummary").textContent = "XAI request failed.";
    $("#xaiChart").innerHTML = '<div class="empty-line">No XAI chart available.</div>';
  });
}

async function loadXaiStatus() {
  $("#xaiRaw").textContent = "Loading XAI status...";
  await withButton($("#xaiStatusBtn"), "Loading...", async () => {
    renderXaiStatus(await getJson("/api/xai/status"));
  }).catch((error) => {
    $("#xaiRaw").textContent = String(error);
    $("#xaiMethodGrid").innerHTML = '<div class="empty-line">XAI status request failed.</div>';
    $("#xaiPolicySummary").textContent = "XAI status request failed.";
  });
}

async function loadTrustStatus() {
  $("#trustRaw").textContent = "Loading trust status...";
  await withButton($("#trustStatusBtn"), "Loading...", async () => {
    renderTrustStatus(await getJson("/api/governance/status"));
  }).catch((error) => {
    $("#trustRaw").textContent = String(error);
    $("#trustPrivacy").innerHTML = '<div class="empty-line">Trust status request failed.</div>';
    $("#trustSecurity").innerHTML = '<div class="empty-line">Trust status request failed.</div>';
    $("#trustDisclosure").innerHTML = '<div class="empty-line">Trust status request failed.</div>';
  });
}

async function loadDemoReadiness() {
  $("#demoRaw").textContent = "Loading demo readiness...";
  await withButton($("#demoReadinessBtn"), "Loading...", async () => {
    renderDemoReadiness(await getJson("/api/demo/readiness"));
  }).catch((error) => {
    $("#demoRaw").textContent = String(error);
    $("#demoChecklist").innerHTML = '<div class="empty-line">Demo readiness request failed.</div>';
    $("#demoFlow").innerHTML = '<div class="empty-line">Demo readiness request failed.</div>';
    $("#demoAdniRun").innerHTML = '<div class="empty-line">Demo readiness request failed.</div>';
  });
}

function summarizeEvaluation(payload) {
  if (Array.isArray(payload)) {
    return `${payload.length} evaluation history record${payload.length === 1 ? "" : "s"} loaded.`;
  }
  if (payload.status && payload.metrics) {
    const metrics = Object.entries(payload.metrics)
      .slice(0, 5)
      .map(([key, value]) => `${key}: ${Number(value).toFixed(4)}`)
      .join(" · ");
    return `${payload.status}. ${metrics}`;
  }
  if (payload.winner) {
    return `Benchmark complete. Winner: ${payload.winner.model || payload.winner.name || "unknown"}.`;
  }
  if (payload.cv_results) {
    return "Cross-validation results loaded.";
  }
  if (payload.evaluation && payload.scientific_claims) {
    const metrics = payload.evaluation.metrics || {};
    return `Evaluation report loaded. macro_auc ${metrics.macro_auc ?? "--"} · ece ${metrics.ece ?? "--"}. ${payload.scientific_claims.notice || ""}`;
  }
  if (payload.status) {
    return String(payload.status);
  }
  return "Evaluation payload loaded.";
}

function renderEvaluation(payload) {
  $("#evalRaw").textContent = JSON.stringify(payload, null, 2);
  $("#evalSummary").textContent = summarizeEvaluation(payload);
}

function summarizeProduction(payload) {
  if (!payload || typeof payload !== "object") return "No production model loaded.";
  if (payload.status === "no_production_model") return payload.message || "No production model is currently registered.";
  const production = payload.production_model || payload;
  const runId = production.run_id;
  const metrics = production.metrics || {};
  if (runId) {
    const metricText = Object.entries(metrics)
      .slice(0, 3)
      .map(([key, value]) => `${key}: ${typeof value === "number" ? value.toFixed(4) : value}`)
      .join(" · ");
    return metricText ? `Run ${runId} · ${metricText}` : `Run ${runId}`;
  }
  return JSON.stringify(payload);
}

function renderProduction(payload) {
  $("#modelsRaw").textContent = JSON.stringify(payload, null, 2);
  $("#productionModelSummary").textContent = summarizeProduction(payload);
}

async function loadModels() {
  $("#modelsRaw").textContent = "Loading model runs...";
  await withButton($("#modelsLoadBtn"), "Loading...", async () => {
    const data = await getJson("/api/models");
    $("#modelsRaw").textContent = JSON.stringify(data, null, 2);
    renderModelsTable(data);
  }).catch((error) => {
    $("#modelsRaw").textContent = String(error);
    $("#modelsTable").innerHTML = '<div class="empty-line">Model run request failed.</div>';
  });
}

async function loadProductionModel() {
  $("#modelsRaw").textContent = "Loading production model...";
  await withButton($("#productionLoadBtn"), "Loading...", async () => {
    renderProduction(await getJson("/api/models/production"));
  }).catch((error) => {
    $("#modelsRaw").textContent = String(error);
    $("#productionModelSummary").textContent = "Production model request failed.";
  });
}

async function loadCheckpointStatus() {
  $("#modelsRaw").textContent = "Loading checkpoint status...";
  await withButton($("#checkpointLoadBtn"), "Loading...", async () => {
    const data = await getJson("/api/models/checkpoint/status");
    $("#modelsRaw").textContent = JSON.stringify(data, null, 2);
    renderCheckpointStatus(data);
  }).catch((error) => {
    $("#modelsRaw").textContent = String(error);
    $("#checkpointSummary").textContent = "Checkpoint status request failed.";
  });
}

async function loadDataStatus() {
  $("#dataRaw").textContent = "Loading data readiness...";
  await withButton($("#dataStatusBtn"), "Loading...", async () => {
    renderDataStatus(await getJson("/api/data/status"));
  }).catch((error) => {
    $("#dataRaw").textContent = String(error);
    $("#dataSummary").innerHTML = '<div class="empty-line">Data status request failed.</div>';
  });
}

async function loadDemoPatients() {
  $("#dataRaw").textContent = "Loading demo patients...";
  await withButton($("#demoPatientsBtn"), "Loading...", async () => {
    renderDemoPatients(await getJson("/api/data/demo-patients?limit=12"));
  }).catch((error) => {
    $("#dataRaw").textContent = String(error);
    $("#demoPatientsTable").innerHTML = '<div class="empty-line">Demo patient request failed.</div>';
  });
}

async function loadModalityStatus() {
  await withButton($("#modalityStatusBtn"), "Loading...", async () => {
    renderModalityStatus(await getJson("/api/modalities/status"));
  }).catch((error) => {
    $("#modalityContract").innerHTML = `<div class="empty-line">${escapeHtml(String(error))}</div>`;
  });
}

async function promoteRunId(runId) {
  const normalized = String(runId || "").trim();
  if (!normalized) {
    $("#modelsRaw").textContent = "Run ID is required.";
    return;
  }
  if (!window.confirm(`Promote run ${normalized} to production?`)) {
    return;
  }

  $("#modelsRaw").textContent = `Promoting ${normalized}...`;
  await withButton($("#promoteBtn"), "Promoting...", async () => {
    const data = await postJson("/api/models/promote", {run_id: normalized});
    renderProduction(data.production_model || data);
  }).catch((error) => {
    $("#modelsRaw").textContent = String(error);
    $("#productionModelSummary").textContent = "Promotion failed.";
  });
}

async function promoteModel(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  await promoteRunId(form.get("run_id"));
}

async function loadEvaluationGet(endpoint, button, loadingText) {
  $("#evalRaw").textContent = "Loading evaluation payload...";
  await withButton(button, loadingText, async () => {
    renderEvaluation(await getJson(endpoint));
  }).catch((error) => {
    $("#evalRaw").textContent = String(error);
    $("#evalSummary").textContent = "Evaluation request failed.";
  });
}

async function runEvaluationPost(endpoint, button, loadingText) {
  $("#evalRaw").textContent = "Running evaluation job...";
  await withButton(button, loadingText, async () => {
    renderEvaluation(await postJson(endpoint, {}));
  }).catch((error) => {
    $("#evalRaw").textContent = String(error);
    $("#evalSummary").textContent = "Evaluation job failed.";
  });
}

async function runDiagnosis(event) {
  event.preventDefault();
  const button = $("#analyzeBtn");
  const report = $("#report");

  button.disabled = true;
  button.textContent = "Analyzing...";
  report.textContent = "Running analysis...";

  try {
    const form = new FormData(event.currentTarget);
    const payload = {
      patient_id: String(form.get("patient_id") || "").trim(),
      query: String(form.get("query") || "What is the model-generated risk profile?"),
      cognitive_scores: collectCognitiveScores(),
    };

    if ($("#useEmbeddings").checked && !payload.patient_id) {
      if (embeddings.mri) payload.mri_embedding = embeddings.mri.embedding;
      if (embeddings.eeg) payload.eeg_embedding = embeddings.eeg.embedding;
      if (embeddings.cog) payload.cog_embedding = embeddings.cog.embedding;
    }

    setResult(await postJson("/api/diagnose", payload));
  } catch (error) {
    $("#badge").textContent = "ERROR";
    $("#badge").style.background = "var(--slate)";
    $("#bar").style.width = "0%";
    report.textContent = String(error);
  } finally {
    button.disabled = false;
    button.textContent = "Analyze";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  $("#backend").textContent = BACKEND;
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.addEventListener("click", () => setView(button.dataset.view));
  });
  $("#healthBtn").addEventListener("click", checkHealth);
  $("#demoReadinessBtn").addEventListener("click", loadDemoReadiness);
  $("#dataStatusBtn").addEventListener("click", loadDataStatus);
  $("#demoPatientsBtn").addEventListener("click", loadDemoPatients);
  $("#modalityStatusBtn").addEventListener("click", loadModalityStatus);
  $("#diagnosisForm").addEventListener("submit", runDiagnosis);
  $("#streamForm").addEventListener("submit", runStreamingDiagnosis);
  $("#kgForm").addEventListener("submit", runKgQuery);
  $("#xaiForm").addEventListener("submit", runXaiQuery);
  $("#xaiStatusBtn").addEventListener("click", loadXaiStatus);
  $("#trustStatusBtn").addEventListener("click", loadTrustStatus);
  $("#mriUploadForm").addEventListener("submit", (event) => uploadEmbedding(event, "mri"));
  $("#eegUploadForm").addEventListener("submit", (event) => uploadEmbedding(event, "eeg"));
  $("#cognitiveUploadBtn").addEventListener("click", encodeCognitiveScores);
  $("#clearEmbeddingsBtn").addEventListener("click", () => {
    clearEmbedding("mri");
    clearEmbedding("eeg");
    clearEmbedding("cog");
  });
  $("#embeddingSummary").addEventListener("click", (event) => {
    const button = event.target.closest("[data-clear-embedding]");
    if (button) clearEmbedding(button.dataset.clearEmbedding);
  });
  $("#evalMetricsBtn").addEventListener("click", () => {
    loadEvaluationGet("/api/eval/metrics", $("#evalMetricsBtn"), "Loading...");
  });
  $("#evalHistoryBtn").addEventListener("click", () => {
    loadEvaluationGet("/api/eval/history", $("#evalHistoryBtn"), "Loading...");
  });
  $("#evalReportBtn").addEventListener("click", () => {
    loadEvaluationGet("/api/eval/report", $("#evalReportBtn"), "Loading...");
  });
  $("#evalRunBtn").addEventListener("click", () => {
    runEvaluationPost("/api/eval/run", $("#evalRunBtn"), "Running...");
  });
  $("#evalBenchmarkBtn").addEventListener("click", () => {
    runEvaluationPost("/api/eval/benchmark", $("#evalBenchmarkBtn"), "Benchmarking...");
  });
  $("#evalCvBtn").addEventListener("click", () => {
    runEvaluationPost("/api/eval/cv", $("#evalCvBtn"), "Running CV...");
  });
  $("#modelsLoadBtn").addEventListener("click", loadModels);
  $("#productionLoadBtn").addEventListener("click", loadProductionModel);
  $("#checkpointLoadBtn").addEventListener("click", loadCheckpointStatus);
  $("#promoteForm").addEventListener("submit", promoteModel);
  $("#modelsTable").addEventListener("click", (event) => {
    const button = event.target.closest("[data-promote-run]");
    if (!button) return;
    const runId = button.dataset.promoteRun;
    const input = document.querySelector('#promoteForm input[name="run_id"]');
    input.value = runId;
    promoteRunId(runId);
  });
  renderEmbeddings();
  renderStreamTimeline();
  checkHealth();
});

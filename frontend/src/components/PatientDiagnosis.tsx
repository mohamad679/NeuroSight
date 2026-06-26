import * as Select from "@radix-ui/react-select";
import { ChevronDown, UserRound } from "lucide-react";
import { useEffect, useState } from "react";
import { diagnosePatient, getDemoPatients } from "@/lib/api";
import type { DiagnoseResponse } from "@/lib/types";
import { ResultPanel } from "./ResultPanel";

interface SyntheticPatient {
  id: string;
  cohort: string;
  AGE: number;
  MMSE: number;
  MOCA: number;
  CDRSB: number;
  ADAS11: number;
  RAVLT_immediate: number;
  RAVLT_learning: number;
  FAQ: number;
  modalities: string[];
  expectedClass: string;
}

const SYNTHETIC_PATIENTS: SyntheticPatient[] = [
  {
    id: "SYN_0001",
    cohort: "Fictitious ADNI-style workflow record",
    AGE: 72,
    MMSE: 23,
    MOCA: 19,
    CDRSB: 1,
    ADAS11: 18,
    RAVLT_immediate: 34,
    RAVLT_learning: 2,
    FAQ: 6,
    modalities: ["Cognitive profile", "MRI placeholder", "EEG placeholder"],
    expectedClass: "MCI"
  }
];

function syntheticResult(patient: SyntheticPatient): DiagnoseResponse {
  return {
    diagnosis: "mci",
    confidence: 0.64,
    requires_review: true,
    blocked_by_safety: false,
    modality_weights: { mri: 0.32, eeg: 0.24, cog: 0.44 },
    feature_importance: {
      MMSE: 0.22,
      MOCA: 0.2,
      CDRSB: 0.16,
      ADAS11: 0.14,
      RAVLT_immediate: 0.12,
      RAVLT_learning: 0.07,
      FAQ: 0.06,
      AGE: 0.03
    },
    modality_weights_source: "synthetic patient workflow profile",
    feature_importance_source: "synthetic explanation values",
    backend_source: "synthetic",
    model_mode: "synthetic_frontend_demo",
    checkpoint_id: null,
    trained_on_real_data: false,
    clinical_validated: false,
    requires_expert_review: true,
    disclaimer:
      "Not clinical software. This synthetic fallback demonstrates UI behavior only.",
    warnings: ["Synthetic patient fallback.", "No trained clinical checkpoint.", "Expert review required."],
    reliability_note:
      "This is a fictitious patient record generated in the frontend only to demonstrate the Patient tab workflow when backend dataset endpoints are unavailable.",
    report_text: `### Research Output Summary

#### Patient Profile
- Patient ID: ${patient.id}
- Cohort: ${patient.cohort}.
- Available modalities: ${patient.modalities.join(", ")}.

#### Cognitive Snapshot
- MMSE: ${patient.MMSE}
- MOCA: ${patient.MOCA}
- CDRSB: ${patient.CDRSB}
- ADAS11: ${patient.ADAS11}
- RAVLT immediate: ${patient.RAVLT_immediate}
- RAVLT learning: ${patient.RAVLT_learning}
- FAQ: ${patient.FAQ}
- AGE: ${patient.AGE}

#### Workflow Output
- Demonstration class: ${patient.expectedClass}.
- Demonstration score: 64.0%.
- Review status: requires clinical review.

#### Purpose
This record exists only so the Patient tab can show the full NeuroSight workflow when the configured backend has no demo-patient API available.

#### Test Project Disclaimer
This is synthetic demonstration data. It is not from a real patient, not medical evidence, and not suitable for patient-care decisions.`
  };
}

export function PatientDiagnosis() {
  const [patientId, setPatientId] = useState("SYN_0001");
  const [demoPatients, setDemoPatients] = useState<string[]>([]);
  const [demoSource, setDemoSource] = useState<"backend" | "synthetic">("synthetic");
  const [result, setResult] = useState<DiagnoseResponse | null>(null);
  const [isLoadingPatients, setIsLoadingPatients] = useState(true);
  const [isLoadingResult, setIsLoadingResult] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const selectedSyntheticPatient =
    SYNTHETIC_PATIENTS.find((patient) => patient.id === patientId.trim()) ?? null;

  useEffect(() => {
    let isMounted = true;

    async function loadPatients() {
      setIsLoadingPatients(true);
      try {
        const patients = await getDemoPatients();
        if (!isMounted) {
          return;
        }
        if (patients.length === 0) {
          setDemoPatients(SYNTHETIC_PATIENTS.map((patient) => patient.id));
          setDemoSource("synthetic");
          setPatientId(SYNTHETIC_PATIENTS[0].id);
          setError(null);
          return;
        }
        setDemoPatients(patients);
        setDemoSource("backend");
        if (patients.length > 0) {
          setPatientId(patients[0]);
        }
        setError(null);
      } catch {
        if (isMounted) {
          setDemoPatients(SYNTHETIC_PATIENTS.map((patient) => patient.id));
          setDemoSource("synthetic");
          setPatientId(SYNTHETIC_PATIENTS[0].id);
          setError(null);
        }
      } finally {
        if (isMounted) {
          setIsLoadingPatients(false);
        }
      }
    }

    void loadPatients();
    return () => {
      isMounted = false;
    };
  }, []);

  async function loadPatient() {
    const trimmed = patientId.trim();
    if (!trimmed) {
      setError("Patient ID is required.");
      return;
    }

    setIsLoadingResult(true);
    setError(null);
    try {
      const syntheticPatient =
        SYNTHETIC_PATIENTS.find((patient) => patient.id === trimmed) ?? null;
      if (syntheticPatient) {
        setResult(syntheticResult(syntheticPatient));
        return;
      }
      const response = await diagnosePatient(trimmed);
      setResult(response);
    } catch (loadError: unknown) {
      setResult(null);
      setError(loadError instanceof Error ? loadError.message : "Patient evaluation failed");
    } finally {
      setIsLoadingResult(false);
    }
  }

  return (
    <section className="grid grid-cols-[360px_1fr] gap-5">
      <div className="rounded-xl border border-borderSubtle bg-bgSurface p-5">
        <div className="mb-6 flex items-center gap-3">
          <div className="grid h-10 w-10 place-items-center rounded-lg bg-accent text-white">
            <UserRound className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-[20px] font-semibold text-textPrimary">Patient Dataset Evaluation</h2>
            <p className="mt-1 text-[13px] text-textSecondary">
              Load a patient from the configured ADNI-style dataset
            </p>
          </div>
        </div>

        <div className="space-y-5">
          <label className="space-y-2">
            <span className="text-[15px] font-medium text-textPrimary">Patient ID</span>
            <input
              value={patientId}
              onChange={(event) => setPatientId(event.target.value)}
              className="h-11 w-full rounded-lg border border-borderSubtle bg-bgElevated px-3 text-[13px] text-textPrimary outline-none focus:shadow-active"
            />
          </label>

          <div className="space-y-2">
            <span className="text-[15px] font-medium text-textPrimary">Or select demo</span>
            <Select.Root value={patientId} onValueChange={setPatientId} disabled={demoPatients.length === 0}>
              <Select.Trigger className="flex h-11 w-full items-center justify-between rounded-lg border border-borderSubtle bg-bgElevated px-3 text-[13px] text-textPrimary outline-none focus:shadow-active disabled:opacity-60">
                <Select.Value placeholder={isLoadingPatients ? "Loading patients" : "No demo patients"} />
                <Select.Icon>
                  <ChevronDown className="h-4 w-4 text-textSecondary" />
                </Select.Icon>
              </Select.Trigger>
              <Select.Portal>
                <Select.Content className="overflow-hidden rounded-lg border border-borderSubtle bg-bgSurface shadow-xl">
                  <Select.Viewport className="max-h-72 p-1">
                    {demoPatients.map((id) => (
                      <Select.Item
                        key={id}
                        value={id}
                        className="cursor-pointer rounded-md px-3 py-2 text-[13px] text-textPrimary outline-none hover:bg-bgElevated"
                      >
                        <Select.ItemText>{id}</Select.ItemText>
                      </Select.Item>
                    ))}
                  </Select.Viewport>
                </Select.Content>
              </Select.Portal>
            </Select.Root>
          </div>

          {isLoadingPatients ? (
            <div className="space-y-2 rounded-lg border border-borderSubtle bg-bgElevated p-3">
              <div className="h-3 w-36 animate-pulse rounded bg-borderSubtle" />
              <div className="h-3 w-52 animate-pulse rounded bg-borderSubtle" />
            </div>
          ) : (
            <div className="rounded-lg border border-borderSubtle bg-bgElevated p-3">
              <p className="text-[11px] font-semibold uppercase tracking-wide text-textSecondary">
                Demo patients list
              </p>
              <p className="mt-2 text-[13px] text-textPrimary">
                {demoSource === "backend"
                  ? `${demoPatients.length} patient IDs loaded from /v1/data/demo-patients`
                  : `${demoPatients.length} synthetic demo patient available`}
              </p>
              {demoSource === "synthetic" ? (
                <p className="mt-2 text-[12px] leading-5 text-textSecondary">
                  Backend demo-patient endpoints are unavailable here, so this tab uses a clearly
                  labeled fictitious patient to demonstrate the workflow.
                </p>
              ) : null}
            </div>
          )}

          {selectedSyntheticPatient ? (
            <div className="rounded-lg border border-warning/30 bg-warning/10 p-3">
              <div className="mb-3 flex items-center justify-between gap-3">
                <p className="text-[11px] font-semibold uppercase tracking-wide text-warning">
                  Synthetic patient profile
                </p>
                <span className="rounded-full border border-warning/30 px-2 py-0.5 text-[10px] font-semibold uppercase text-warning">
                  demo only
                </span>
              </div>
              <div className="space-y-2 text-[12px] leading-5 text-textSecondary">
                <p>
                  <span className="font-semibold text-textPrimary">{selectedSyntheticPatient.id}</span>
                  {" - "}
                  {selectedSyntheticPatient.cohort}
                </p>
                <p>
                  AGE {selectedSyntheticPatient.AGE}; MMSE{" "}
                  {selectedSyntheticPatient.MMSE}, MOCA {selectedSyntheticPatient.MOCA}, CDRSB{" "}
                  {selectedSyntheticPatient.CDRSB}
                </p>
                <p>Modalities: {selectedSyntheticPatient.modalities.join(", ")}</p>
              </div>
            </div>
          ) : null}

          {error ? (
            <div className="rounded-lg border border-danger/40 bg-danger/10 px-3 py-2 text-[13px] text-danger">
              {error}
            </div>
          ) : null}

          <button
            type="button"
            disabled={isLoadingResult}
            onClick={() => {
              void loadPatient();
            }}
            className="flex w-full items-center justify-center gap-2 rounded-lg bg-accent px-5 py-3 text-[13px] font-semibold text-white transition hover:bg-accentSoft focus:shadow-active disabled:cursor-wait disabled:opacity-60"
          >
            Load Patient
            <UserRound className="h-4 w-4" />
          </button>
        </div>
      </div>

      <ResultPanel result={result} isLoading={isLoadingResult} />
    </section>
  );
}

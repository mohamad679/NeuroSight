import { motion } from "framer-motion";
import { AlertTriangle, ShieldAlert } from "lucide-react";
import type { DiagnosisLabel } from "@/lib/types";

interface ModalityBadgeProps {
  diagnosis: DiagnosisLabel;
  requiresReview: boolean;
  blockedBySafety: boolean;
}

const diagnosisClass: Record<DiagnosisLabel, string> = {
  normal: "border-success/30 bg-success/10 text-success",
  mci: "border-warning/40 bg-warning/10 text-warning",
  ad: "border-danger/30 bg-danger/10 text-danger",
  ftd: "border-danger/30 bg-danger/10 text-danger",
  lbd: "border-danger/30 bg-danger/10 text-danger",
  vd: "border-danger/30 bg-danger/10 text-danger"
};

export function ModalityBadge({
  diagnosis,
  requiresReview,
  blockedBySafety
}: ModalityBadgeProps) {
  return (
    <div className="space-y-3">
      <motion.div
        initial={{ y: 10, scale: 0.94, opacity: 0 }}
        animate={{ y: 0, scale: 1, opacity: 1 }}
        transition={{ type: "spring", stiffness: 220, damping: 22 }}
        className={`inline-flex min-h-16 items-center rounded-[8px] border px-5 font-display text-[clamp(2.2rem,5vw,4.4rem)] font-extrabold uppercase leading-none tracking-[-0.045em] ${diagnosisClass[diagnosis]}`}
      >
        {diagnosis}
      </motion.div>

      {requiresReview ? (
        <div className="flex items-center gap-2 rounded-[8px] border border-warning/30 bg-warning/10 px-3 py-2 text-sm font-bold text-warning">
          <AlertTriangle className="h-4 w-4" />
          Requires expert review
        </div>
      ) : null}

      {blockedBySafety ? (
        <div className="flex items-center gap-2 rounded-[8px] border border-danger/30 bg-danger/10 px-3 py-2 text-sm font-bold text-danger">
          <ShieldAlert className="h-4 w-4" />
          Blocked by safety guardian
        </div>
      ) : null}
    </div>
  );
}

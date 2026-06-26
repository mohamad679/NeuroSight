import { motion } from "framer-motion";

interface ConfidenceMeterProps {
  confidence: number;
  label?: string;
}

function meterClass(confidence: number): string {
  if (confidence > 0.8) {
    return "stroke-success";
  }
  if (confidence >= 0.5) {
    return "stroke-warning";
  }
  return "stroke-danger";
}

export function ConfidenceMeter({ confidence, label = "model score" }: ConfidenceMeterProps) {
  const value = Math.max(0, Math.min(1, confidence));

  return (
    <div className="luxury-panel reveal reveal-delay-1 p-5">
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-extrabold uppercase tracking-[0.18em] text-textMuted">
            confidence
          </p>
          <h3 className="font-display mt-2 text-3xl font-extrabold tracking-[-0.04em] text-textPrimary">
            {(value * 100).toFixed(1)}%
          </h3>
        </div>
        <span className="rounded-full border border-borderSubtle px-3 py-1 text-[0.65rem] font-extrabold uppercase tracking-[0.16em] text-textSecondary">
          {label}
        </span>
      </div>
      <svg viewBox="0 0 220 130" className="h-32 w-full" role="img" aria-label="Model score gauge">
        <path
          d="M 25 110 A 85 85 0 0 1 195 110"
          fill="none"
          strokeWidth="12"
          className="stroke-white/10"
          strokeLinecap="round"
          pathLength={1}
        />
        <motion.path
          d="M 25 110 A 85 85 0 0 1 195 110"
          fill="none"
          strokeWidth="12"
          className={meterClass(value)}
          strokeLinecap="round"
          pathLength={1}
          initial={{ pathLength: 0 }}
          animate={{ pathLength: value }}
          transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
        />
        <text x="25" y="126" textAnchor="middle" className="fill-textMuted text-[11px] font-bold">
          0
        </text>
        <text x="195" y="126" textAnchor="middle" className="fill-textMuted text-[11px] font-bold">
          100
        </text>
      </svg>
    </div>
  );
}

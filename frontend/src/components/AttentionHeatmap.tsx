import { motion } from "framer-motion";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import type { ModalityWeights } from "@/lib/types";

interface AttentionHeatmapProps {
  weights: ModalityWeights;
  featureImportance: Record<string, number>;
  weightsSource?: string;
  featureSource?: string;
}

interface ModalityRow {
  label: string;
  value: number;
  colorClass: string;
}

interface FeatureRow {
  name: string;
  value: number;
}

function FeatureTooltip({
  active,
  payload,
  label
}: {
  active?: boolean;
  payload?: Array<{ value?: number; name?: string }>;
  label?: string;
}) {
  if (!active || !payload || payload.length === 0) {
    return null;
  }
  return (
    <div className="rounded-[8px] border border-borderSubtle bg-bgSurface px-3 py-2 text-[11px] text-textPrimary shadow-luxury">
      <p className="font-semibold">{label}</p>
      {payload.map((item) => (
        <p key={item.name ?? "value"} className="text-textSecondary">
          {item.name}: {typeof item.value === "number" ? item.value.toFixed(4) : "—"}
        </p>
      ))}
    </div>
  );
}

export function AttentionHeatmap({
  weights,
  featureImportance,
  weightsSource,
  featureSource
}: AttentionHeatmapProps) {
  const modalities: ModalityRow[] = [
    { label: "MRI", value: weights.mri, colorClass: "bg-accent" },
    { label: "EEG", value: weights.eeg, colorClass: "bg-teal" },
    { label: "Cognitive", value: weights.cog, colorClass: "bg-warning" }
  ];

  const features: FeatureRow[] = Object.entries(featureImportance)
    .map(([name, value]) => ({ name, value: Number(value.toFixed(4)) }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 8);

  return (
    <div className="luxury-panel h-full min-h-0 overflow-hidden p-5">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h3 className="font-display text-[1.65rem] font-extrabold tracking-[-0.04em] text-textPrimary">Modality Attention</h3>
          <p className="mt-1 text-[11px] text-textSecondary">
            Source: {weightsSource ?? "backend"}
          </p>
        </div>
      </div>

      <div className="luxury-panel-muted p-4">
        <div className="space-y-3">
          {modalities.map((item) => {
            const pct = Math.max(0, Math.min(100, item.value * 100));
            return (
              <div key={item.label} className="grid grid-cols-[90px_1fr_52px] items-center gap-3">
                <span className="text-[13px] font-medium text-textSecondary">{item.label}</span>
                <div className="h-2.5 overflow-hidden rounded-full bg-white/[0.07]">
                  <motion.div
                    className={`h-full rounded-full ${item.colorClass}`}
                    initial={{ width: 0 }}
                    animate={{ width: `${pct}%` }}
                    transition={{ duration: 0.65, ease: "easeOut" }}
                  />
                </div>
                <span className="text-right text-[13px] font-medium text-textPrimary">
                  {pct.toFixed(1)}%
                </span>
              </div>
            );
          })}
        </div>
      </div>

      <div className="mt-4 border-t border-borderSubtle pt-4">
        <div className="mb-3 flex items-center justify-between">
          <h4 className="text-[13px] font-extrabold uppercase tracking-[0.14em] text-textPrimary">Cognitive Feature Importance</h4>
          <span className="text-[11px] text-textSecondary">{featureSource ?? "backend XAI"}</span>
        </div>
        {features.length > 0 ? (
          <div className="h-44">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={features} layout="vertical" margin={{ left: 14, right: 20 }}>
                <CartesianGrid stroke="rgba(223,241,232,0.10)" horizontal={false} />
                <XAxis type="number" stroke="rgba(167,178,169,0.55)" tick={{ fill: "var(--color-text-muted)", fontSize: 11 }} />
                <YAxis
                  dataKey="name"
                  type="category"
                  width={96}
                  stroke="rgba(167,178,169,0.55)"
                  tick={{ fill: "var(--color-text-muted)", fontSize: 11 }}
                />
                <Tooltip content={<FeatureTooltip />} cursor={{ fill: "rgba(244,184,96,0.08)" }} />
                <Bar dataKey="value" fill="var(--color-accent)" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="luxury-panel-muted p-4">
            <p className="text-[13px] font-semibold text-textPrimary">
              Feature importance unavailable
            </p>
            <p className="mt-1 text-[12px] leading-5 text-textSecondary">
              The current backend response did not include structured feature-importance values.
              When the FastAPI XAI endpoint returns numeric values, this area will render them as
              native NeuroSight bars instead of external plot images.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

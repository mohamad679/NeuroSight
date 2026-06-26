import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import type { BenchmarkClassMetric } from "@/lib/types";

const ablationRows = [
  { configuration: "All modalities", accuracy: "71.4%", macroF1: "0.694", auc: "0.945" },
  { configuration: "MRI only", accuracy: "—", macroF1: "—", auc: "—" },
  { configuration: "EEG only", accuracy: "—", macroF1: "—", auc: "—" },
  { configuration: "Cognitive only", accuracy: "71.4%*", macroF1: "0.694", auc: "0.945" }
];

const perClassMetrics: BenchmarkClassMetric[] = [
  { className: "normal", f1: 0.909, auc: 1.0 },
  { className: "mci", f1: 0.75, auc: 0.969 },
  { className: "ad", f1: 0.769, auc: 0.957 },
  { className: "ftd", f1: 0.667, auc: 0.904 },
  { className: "lbd", f1: 0.571, auc: 0.913 },
  { className: "vd", f1: 0.5, auc: 0.927 }
];

const calibrationRows = [
  { bin: "0.1", expected: 0.1, observed: 0.0 },
  { bin: "0.3", expected: 0.3, observed: 0.18 },
  { bin: "0.5", expected: 0.5, observed: 0.32 },
  { bin: "0.7", expected: 0.7, observed: 0.51 },
  { bin: "0.9", expected: 0.9, observed: 0.68 }
];

function BenchmarkTooltip({
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
      <p className="font-semibold uppercase">{label}</p>
      {payload.map((item) => (
        <p key={item.name ?? "metric"} className="text-textSecondary">
          {item.name}: {typeof item.value === "number" ? item.value.toFixed(3) : "—"}
        </p>
      ))}
    </div>
  );
}

export function BenchmarkPanel() {
  return (
    <section className="grid gap-5">
      <div className="luxury-panel p-5">
        <div className="mb-5">
          <p className="luxury-kicker">
            Synthetic benchmark
          </p>
          <h2 className="font-display mt-2 text-[2.4rem] font-extrabold tracking-[-0.045em] text-textPrimary">Ablation Study</h2>
        </div>

        <div className="overflow-hidden rounded-[8px] border border-borderSubtle">
          <table className="w-full border-collapse text-left text-[13px]">
            <thead className="bg-white/[0.045] text-textSecondary">
              <tr>
                <th className="px-4 py-3 font-semibold">Configuration</th>
                <th className="px-4 py-3 font-semibold">Accuracy</th>
                <th className="px-4 py-3 font-semibold">Macro F1</th>
                <th className="px-4 py-3 font-semibold">AUC</th>
              </tr>
            </thead>
            <tbody>
              {ablationRows.map((row) => (
                <tr key={row.configuration} className="zebra-row border-t border-borderSubtle">
                  <td className="px-4 py-3 text-textPrimary">{row.configuration}</td>
                  <td className="px-4 py-3 text-textSecondary">{row.accuracy}</td>
                  <td className="px-4 py-3 text-textSecondary">{row.macroF1}</td>
                  <td className="px-4 py-3 text-textSecondary">{row.auc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-[11px] text-textSecondary">
          *Synthetic results, single-modality training. MRI/EEG public-demo values are not
          claimed because the current synthetic cohort is cognitive-centered.
        </p>
      </div>

      <div className="grid grid-cols-[minmax(0,1fr)_360px] gap-5">
        <div className="luxury-panel p-5">
          <h3 className="font-display text-[1.8rem] font-extrabold tracking-[-0.04em] text-textPrimary">Per-class F1</h3>
          <div className="mt-4 h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={perClassMetrics}>
                <CartesianGrid stroke="rgba(223,241,232,0.10)" vertical={false} />
                <XAxis dataKey="className" stroke="rgba(167,178,169,0.55)" tick={{ fill: "var(--color-text-muted)", fontSize: 11 }} />
                <YAxis stroke="rgba(167,178,169,0.55)" tick={{ fill: "var(--color-text-muted)", fontSize: 11 }} />
                <Tooltip content={<BenchmarkTooltip />} cursor={{ fill: "rgba(244,184,96,0.08)" }} />
                <Bar dataKey="f1" name="F1" fill="var(--color-accent)" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="luxury-panel p-5">
          <h3 className="font-display text-[1.8rem] font-extrabold tracking-[-0.04em] text-textPrimary">Calibration</h3>
          <div className="mt-4 h-56 rounded-lg border border-borderSubtle bg-bgElevated p-3">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={calibrationRows}>
                <CartesianGrid stroke="rgba(223,241,232,0.10)" />
                <XAxis dataKey="bin" stroke="rgba(167,178,169,0.55)" tick={{ fill: "var(--color-text-muted)", fontSize: 11 }} />
                <YAxis domain={[0, 1]} stroke="rgba(167,178,169,0.55)" tick={{ fill: "var(--color-text-muted)", fontSize: 11 }} />
                <Tooltip content={<BenchmarkTooltip />} />
                <Line type="monotone" dataKey="expected" name="Expected" stroke="rgba(167,178,169,0.85)" strokeDasharray="4 4" dot={false} />
                <Line type="monotone" dataKey="observed" name="Observed" stroke="var(--color-accent)" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-3 text-[11px] leading-5 text-textSecondary">
            Calibration reliability diagram from the synthetic evaluation artifact.
          </p>
          <div className="mt-4 rounded-lg border border-warning/40 bg-warning/10 p-3">
            <p className="text-[13px] font-semibold text-warning">ECE = 0.255</p>
            <p className="mt-2 text-[13px] leading-6 text-textSecondary">
              ECE of 0.255 indicates overconfidence on this synthetic dataset. Temperature
              scaling is applied during inference with initial T=1.5, but further post-hoc
              calibration on held-out validation data is recommended before research use.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

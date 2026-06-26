type StatusTone = "ok" | "warning" | "error" | "muted";

interface StatusDotProps {
  tone: StatusTone;
  label?: string;
}

const toneClass: Record<StatusTone, string> = {
  ok: "bg-success",
  warning: "bg-warning",
  error: "bg-danger",
  muted: "bg-textMuted"
};

export function StatusDot({ tone, label }: StatusDotProps) {
  return (
    <span className="inline-flex items-center gap-2">
      <span className={`h-2.5 w-2.5 rounded-full ${toneClass[tone]}`} />
      {label ? <span>{label}</span> : null}
    </span>
  );
}

import { CheckCircle2, X, XCircle } from "lucide-react";

interface ToastNotificationProps {
  tone: "success" | "error";
  message: string;
  onDismiss: () => void;
}

export function ToastNotification({ tone, message, onDismiss }: ToastNotificationProps) {
  const isSuccess = tone === "success";
  const Icon = isSuccess ? CheckCircle2 : XCircle;

  return (
    <div
      role="status"
      className={`toast-enter fixed right-6 top-6 z-50 flex max-w-sm items-start gap-3 rounded-[8px] border px-4 py-3 shadow-luxury backdrop-blur-xl ${
        isSuccess
          ? "border-success/30 bg-success/10 text-success"
          : "border-danger/30 bg-danger/10 text-danger"
      }`}
    >
      <Icon className="mt-0.5 h-5 w-5 shrink-0" />
      <p className="text-sm font-bold leading-6">{message}</p>
      <button
        type="button"
        onClick={onDismiss}
        className="btn btn-ghost ml-1 h-8 w-8 shrink-0 p-0"
        aria-label="Dismiss notification"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  );
}

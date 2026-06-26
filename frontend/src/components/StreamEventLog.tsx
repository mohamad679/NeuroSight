import { Terminal } from "lucide-react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { atomOneDark } from "react-syntax-highlighter/dist/cjs/styles/hljs";
import type { StreamEvent } from "@/lib/types";

interface StreamEventLogProps {
  events: StreamEvent[];
}

export function StreamEventLog({ events }: StreamEventLogProps) {
  return (
    <div className="luxury-panel flex h-full min-h-[440px] flex-col overflow-hidden">
      <div className="flex items-center justify-between border-b border-borderSubtle px-5 py-4">
        <div>
          <p className="luxury-kicker">event log</p>
          <h3 className="font-display mt-1 text-2xl font-extrabold tracking-[-0.04em]">
            Runtime trace
          </h3>
        </div>
        <Terminal className="h-5 w-5 text-accent" />
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        {events.length > 0 ? (
          <div className="space-y-3">
            {events.map((event, index) => (
              <SyntaxHighlighter
                key={`${event.agent}-${event.status}-${index}`}
                language="json"
                style={atomOneDark}
                className="syntax-event"
              >
                {JSON.stringify(event, null, 2)}
              </SyntaxHighlighter>
            ))}
          </div>
        ) : (
          <div className="grid min-h-[420px] place-items-center text-center">
            <div className="max-w-xs">
              <p className="font-display text-3xl font-extrabold tracking-[-0.04em] text-textPrimary">
                No packets yet
              </p>
              <p className="mt-3 text-sm leading-6 text-textSecondary">
                Stream events will slide into this log as the backend emits agent state.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

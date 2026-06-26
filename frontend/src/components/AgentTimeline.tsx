import { motion } from "framer-motion";
import { Check } from "lucide-react";
import { useAppStore } from "@/lib/store";
import type { AgentState } from "@/lib/types";

const agents = ["supervisor", "kg_retriever", "report_writer", "safety_guardian"];

interface AgentTimelineProps {
  states?: Record<string, AgentState>;
  large?: boolean;
  timings?: Record<string, number>;
}

function NodeIcon({ state, large }: { state: AgentState; large: boolean }) {
  const sizeClass = large ? "h-7 w-7" : "h-5 w-5";

  if (state === "completed") {
    return (
      <span className={`grid shrink-0 place-items-center rounded-full bg-success text-bgBase ${sizeClass}`}>
        <Check className={large ? "h-4 w-4" : "h-3 w-3"} />
      </span>
    );
  }

  if (state === "running") {
    return (
      <motion.span
        className={`shrink-0 rounded-full bg-accent ${sizeClass}`}
        animate={{ opacity: [0.55, 1, 0.55], scale: [0.95, 1.08, 0.95] }}
        transition={{ duration: 1.1, repeat: Infinity, ease: "easeInOut" }}
      />
    );
  }

  return <span className={`shrink-0 rounded-full border border-textMuted ${sizeClass}`} />;
}

export function AgentTimeline({ states, large = false, timings = {} }: AgentTimelineProps) {
  const storeStates = useAppStore((state) => state.streamingAgentStates);
  const currentStates = states ?? storeStates;

  return (
    <div className="space-y-1">
      {agents.map((agent, index) => {
        const state = currentStates[agent] ?? "pending";
        const isCompleted = state === "completed";
        const isRunning = state === "running";
        return (
          <div key={agent} className="relative flex gap-4 pb-5 last:pb-0">
            {index < agents.length - 1 ? (
              <span
                className={`absolute left-2.5 top-6 h-[calc(100%-1.5rem)] w-px ${
                  large ? "left-3.5 top-8 h-[calc(100%-2rem)]" : ""
                } ${isCompleted ? "bg-success/50" : "bg-borderSubtle"}`}
              />
            ) : null}
            <NodeIcon state={state} large={large} />
            <div className="min-w-0 flex-1">
              <div
                className={`font-semibold ${
                  large ? "text-[18px]" : "text-[13px]"
                } ${isCompleted ? "text-textPrimary" : isRunning ? "text-accentSoft" : "text-textMuted"}`}
              >
                {agent}
              </div>
              <div className={`mt-1 text-textSecondary ${large ? "text-[13px]" : "text-[11px]"}`}>
                {isRunning ? "running..." : isCompleted ? "completed" : "pending"}
                {timings[agent] !== undefined ? ` · ${timings[agent].toFixed(2)}s` : ""}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

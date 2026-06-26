import { create } from "zustand";
import type {
  AgentState,
  CheckpointStatus,
  CognitiveScores,
  DiagnoseResponse,
  EvalHistoryEntry,
  EvalMetrics,
  ModelRun,
  StreamEvent
} from "./types";

const agentNames = ["supervisor", "kg_retriever", "report_writer", "safety_guardian"] as const;

function initialAgentStates(): Record<string, AgentState> {
  return Object.fromEntries(agentNames.map((agent) => [agent, "pending"])) as Record<
    string,
    AgentState
  >;
}

interface AppState {
  cogScores: CognitiveScores;
  mriFile: File | null;
  eegFile: File | null;
  query: string;
  isLoading: boolean;
  result: DiagnoseResponse | null;
  error: string | null;
  streamEvents: StreamEvent[];
  isStreaming: boolean;
  streamingAgentStates: Record<string, AgentState>;
  evalMetrics: EvalMetrics | null;
  evalHistory: EvalHistoryEntry[];
  isRunningEval: boolean;
  models: ModelRun[];
  productionModel: ModelRun | null;
  checkpointStatus: CheckpointStatus | null;
  systemHealth: Record<string, unknown>;
  lastHealthCheck: Date | null;
  setCog: <K extends keyof CognitiveScores>(key: K, val: CognitiveScores[K]) => void;
  setMriFile: (f: File | null) => void;
  setEegFile: (f: File | null) => void;
  setQuery: (q: string) => void;
  setLoading: (b: boolean) => void;
  setResult: (r: DiagnoseResponse | null) => void;
  setError: (e: string | null) => void;
  addStreamEvent: (e: StreamEvent) => void;
  setStreaming: (b: boolean) => void;
  setAgentState: (agent: string, state: AgentState) => void;
  resetStream: () => void;
  setEvalMetrics: (metrics: EvalMetrics | null) => void;
  setEvalHistory: (history: EvalHistoryEntry[]) => void;
  setRunningEval: (b: boolean) => void;
  setModels: (models: ModelRun[]) => void;
  setProductionModel: (model: ModelRun | null) => void;
  setCheckpointStatus: (status: CheckpointStatus | null) => void;
  setSystemHealth: (health: Record<string, unknown>) => void;
  setLastHealthCheck: (date: Date | null) => void;
}

const initialScores: CognitiveScores = {
  MMSE: 24,
  MOCA: 20,
  CDRSB: 1,
  ADAS11: 18,
  RAVLT_immediate: 34,
  RAVLT_learning: 2,
  FAQ: 6,
  AGE: 70
};

export const useAppStore = create<AppState>((set) => ({
  cogScores: initialScores,
  mriFile: null,
  eegFile: null,
  query: "What should the research/demo workflow inspect?",
  isLoading: false,
  result: null,
  error: null,
  streamEvents: [],
  isStreaming: false,
  streamingAgentStates: initialAgentStates(),
  evalMetrics: null,
  evalHistory: [],
  isRunningEval: false,
  models: [],
  productionModel: null,
  checkpointStatus: null,
  systemHealth: {},
  lastHealthCheck: null,
  setCog: (key, val) =>
    set((state) => ({
      cogScores: {
        ...state.cogScores,
        [key]: val
      }
    })),
  setMriFile: (f) => set({ mriFile: f }),
  setEegFile: (f) => set({ eegFile: f }),
  setQuery: (q) => set({ query: q }),
  setLoading: (b) => set({ isLoading: b }),
  setResult: (r) => set({ result: r }),
  setError: (e) => set({ error: e }),
  addStreamEvent: (e) =>
    set((state) => ({
      streamEvents: [e, ...state.streamEvents]
    })),
  setStreaming: (b) => set({ isStreaming: b }),
  setAgentState: (agent, agentState) =>
    set((state) => ({
      streamingAgentStates: {
        ...state.streamingAgentStates,
        [agent]: agentState
      }
    })),
  resetStream: () =>
    set({
      streamEvents: [],
      isStreaming: false,
      streamingAgentStates: initialAgentStates()
    }),
  setEvalMetrics: (metrics) => set({ evalMetrics: metrics }),
  setEvalHistory: (history) => set({ evalHistory: history }),
  setRunningEval: (b) => set({ isRunningEval: b }),
  setModels: (models) => set({ models }),
  setProductionModel: (model) => set({ productionModel: model }),
  setCheckpointStatus: (status) => set({ checkpointStatus: status }),
  setSystemHealth: (health) => set({ systemHealth: health }),
  setLastHealthCheck: (date) => set({ lastHealthCheck: date })
}));

export type CogKey = keyof CognitiveScores;

/**
 * pi-reason-harness — Shared types
 */

export interface ReasonHarnessRuntime {
  sessionName: string | null;
  taskType: string | null;
  status: string;
  iterationCount: number;
  bestScore: number;
  solved: boolean;
  expertCount: number;
  models: string[];
  totalTokens: number;
  totalCost: number;
  adaptations: number;
  budget: {
    costUsed: number;
    timeUsed: number;
    problemsSolved: number;
    problemsAttempted: number;
  };
}

/**
 * pi-reason-harness — Runtime state management
 */

import type { ReasonHarnessRuntime } from '../types/index.js';

export function createRuntimeState(): ReasonHarnessRuntime {
  return {
    sessionName: null,
    taskType: null,
    status: 'idle',
    iterationCount: 0,
    bestScore: 0,
    solved: false,
    expertCount: 0,
    models: [],
    totalTokens: 0,
    totalCost: 0,
    adaptations: 0,
    budget: {
      costUsed: 0,
      timeUsed: 0,
      problemsSolved: 0,
      problemsAttempted: 0,
    },
  };
}

export interface RuntimeStore {
  ensure(key: string): ReasonHarnessRuntime;
  clear(key: string): void;
}

export function createRuntimeStore(): RuntimeStore {
  const runtimes = new Map<string, ReasonHarnessRuntime>();
  return {
    ensure(key: string): ReasonHarnessRuntime {
      let runtime = runtimes.get(key);
      if (!runtime) {
        runtime = createRuntimeState();
        runtimes.set(key, runtime);
      }
      return runtime;
    },
    clear(key: string): void {
      runtimes.delete(key);
    },
  };
}

/**
 * pi-reason-harness — Dashboard rendering
 */

import type { ReasonHarnessRuntime } from '../types/index.js';

export function renderDashboardLines(
  runtime: ReasonHarnessRuntime,
  width: number,
  theme: { fg: (color: string, text: string) => string },
  maxLines: number = 6
): string[] {
  const lines: string[] = [];

  lines.push(
    `  Type: ${runtime.taskType}  │  Status: ${runtime.status}  │  Experts: ${runtime.expertCount}`
  );
  lines.push(
    `  ★ Best score: ${runtime.bestScore.toFixed(2)}  │  ${runtime.solved ? '✅ Solved' : '❌ Not solved'}  │  ${runtime.iterationCount} iterations`
  );

  if (runtime.totalTokens > 0 || runtime.totalCost > 0) {
    lines.push(
      `  Tokens: ${runtime.totalTokens}  │  Cost: $${runtime.totalCost.toFixed(4)}`
    );
  }

  if (runtime.budget.problemsAttempted > 0) {
    lines.push(
      `  Budget: ${runtime.budget.problemsSolved}/${runtime.budget.problemsAttempted} solved  │  $${runtime.budget.costUsed.toFixed(4)} spent  │  ${runtime.budget.timeUsed.toFixed(1)}s`
    );
  }

  if (runtime.adaptations > 0) {
    lines.push(
      `  🧠 ${runtime.adaptations} strategy adaptation(s) active`
    );
  }

  if (runtime.models.length > 0) {
    lines.push(
      `  Models: ${runtime.models.join(', ')}`
    );
  }

  return lines.slice(0, maxLines);
}

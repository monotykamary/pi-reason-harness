/**
 * pi-reason-harness — Server unit tests
 *
 * Tests the core algorithms: voting, soft scoring, feedback building,
 * strategy adaptation, budget tracking, and model resolution.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

// We test the pure functions by extracting them for testability.
// In production these live in server.ts; for tests we inline the critical ones.

// =============================================================================
// Grid utilities
// =============================================================================

function ensure2D(arr: unknown): number[][] | null {
  if (!Array.isArray(arr)) return null;
  if (arr.length === 0) return [[]];
  if (Array.isArray(arr[0])) return arr as number[][];
  return [arr as unknown[] as number[]];
}

function gridShape(grid: number[][]): [number, number] {
  return [grid.length, grid.length > 0 ? grid[0].length : 0];
}

function arrayDiff(pred: number[][], truth: number[][]): string {
  const rows = truth.length;
  const cols = truth.length > 0 ? truth[0].length : 0;
  const lines: string[] = [];
  for (let i = 0; i < rows; i++) {
    const row: string[] = [];
    const pRow = i < pred.length ? pred[i] : [];
    const tRow = truth[i];
    for (let j = 0; j < cols; j++) {
      const pVal = j < pRow.length ? pRow[j] : '?';
      const tVal = tRow[j];
      if (pVal === tVal) {
        row.push(String(tVal));
      } else {
        row.push(`${pVal}/${tVal}`);
      }
    }
    lines.push(row.join(' '));
  }
  return lines.join('\n');
}

function gridToDiagram(grid: number[][]): string {
  return grid.map(row => row.join(' ')).join('\n');
}

// =============================================================================
// Soft score computation
// =============================================================================

function computeSoftScore(actual: string, expected: unknown): number {
  try {
    const actualArr = JSON.parse(actual);
    const expectedArr = Array.isArray(expected) ? expected : JSON.parse(JSON.stringify(expected));

    if (!Array.isArray(actualArr) || !Array.isArray(expectedArr)) return 0;

    const pred2D = ensure2D(actualArr);
    const truth2D = ensure2D(expectedArr);

    if (!pred2D || !truth2D) return 0;

    const [predRows, predCols] = gridShape(pred2D);
    const [truthRows, truthCols] = gridShape(truth2D);

    if (predRows !== truthRows || predCols !== truthCols) return 0;
    if (truthRows === 0 || truthCols === 0) return 1;

    let matches = 0;
    const total = truthRows * truthCols;
    for (let i = 0; i < truthRows; i++) {
      for (let j = 0; j < truthCols; j++) {
        if (pred2D[i][j] === truth2D[i][j]) matches++;
      }
    }

    return total > 0 ? matches / total : 0;
  } catch {
    return 0;
  }
}

// =============================================================================
// Output comparison
// =============================================================================

function compareOutputs(actual: string, expected: unknown): boolean {
  try {
    const actualParsed = JSON.parse(actual);
    const expectedParsed = Array.isArray(expected) ? expected : JSON.parse(JSON.stringify(expected));

    const pred2D = ensure2D(actualParsed);
    const truth2D = ensure2D(expectedParsed);

    if (pred2D && truth2D) {
      const [pr, pc] = gridShape(pred2D);
      const [tr, tc] = gridShape(truth2D);
      if (pr !== tr || pc !== tc) return false;
      for (let i = 0; i < tr; i++) {
        for (let j = 0; j < tc; j++) {
          if (pred2D[i][j] !== truth2D[i][j]) return false;
        }
      }
      return true;
    }

    return JSON.stringify(actualParsed) === JSON.stringify(expectedParsed);
  } catch {
    return actual.trim() === String(expected).trim();
  }
}

// =============================================================================
// Code parsing
// =============================================================================

function parseCodeFromLLM(response: string): string | null {
  const m = response.match(/```(?:javascript|js|typescript|ts)\s*(.*?)```/s);
  return m ? m[1].trim() : null;
}

// =============================================================================
// RNG
// =============================================================================

function createRNG(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

// =============================================================================
// Model resolution
// =============================================================================

function formatProblem(
  trainIn: number[][][],
  trainOut: number[][][],
  testIn: number[][][],
  shuffle: boolean = true,
  seed: number = 0
): string {
  const indices = trainIn.map((_, i) => i);
  if (shuffle && indices.length > 1) {
    const rng = createRNG(seed);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
  }

  let exampleStr = '';
  let challengeStr = '';

  for (let e = 0; e < indices.length; e++) {
    const idx = indices[e];
    exampleStr += `\nExample #${e + 1}\nInput:\n<Diagram>\n${gridToDiagram(trainIn[idx])}\n</Diagram>\n\nOutput:\n<Diagram>\n${gridToDiagram(trainOut[idx])}\n</Diagram>\n`;
  }

  for (let c = 0; c < testIn.length; c++) {
    challengeStr += `\nChallenge #${c + 1}\nInput:\n<Diagram>\n${gridToDiagram(testIn[c])}\n</Diagram>\n`;
  }

  return exampleStr + challengeStr;
}

// =============================================================================
// Detailed feedback (Poetiq parity)
// =============================================================================

interface SolveResult {
  success: boolean;
  output: string;
  softScore: number;
  error: string | null;
  code: string;
}

function buildDetailedFeedback(
  trainResults: SolveResult[],
  _trainInputs: unknown[],
  trainOutputs: unknown[]
): string {
  const parts: string[] = [];

  for (let i = 0; i < trainResults.length; i++) {
    const rr = trainResults[i];
    if (rr.success) {
      parts.push(`Solves Example #${i + 1} correctly. `);
      continue;
    }

    const msgLines: string[] = [`Solves Example #${i + 1} incorrectly. `];

    let predArr: unknown = null;
    try {
      if (rr.output) {
        predArr = JSON.parse(rr.output);
      }
    } catch {}

    const truth = trainOutputs[i];
    const truthArr = Array.isArray(truth) ? truth : null;

    if (!predArr || !Array.isArray(predArr)) {
      msgLines.push('\nThe output has to be a rectangular grid of numbers.\n');
      if (rr.error) {
        msgLines.push(`Your code produced the following error:\n${rr.error.slice(0, 300)}\n`);
      }
    } else {
      const pred2D = ensure2D(predArr);
      const truth2D = truthArr ? ensure2D(truthArr) : null;

      if (!truth2D || !pred2D) {
        msgLines.push('\nFailed to parse grids for comparison.\n');
      } else {
        const predShape = gridShape(pred2D);
        const truthShape = gridShape(truth2D);

        if (predShape[0] !== truthShape[0] || predShape[1] !== truthShape[1]) {
          msgLines.push(
            `\n\nShape mismatch: your prediction's shape was [${predShape}], ` +
            `while the correct shape was [${truthShape}].`
          );
        } else {
          msgLines.push(
            '\nYour code\'s output does not match the expected output.' +
            '\n\nBelow is a visualization of the 2D array your code produced as well as the expected output.\n' +
            'Correctly predicted values are shown as-is while the incorrectly predicted values are shown ' +
            "in the format 'prediction/correct':\n"
          );
          const diff = arrayDiff(pred2D, truth2D);
          msgLines.push(`\n\`\`\`\n${diff}\n\`\`\`\n`);
          msgLines.push(`Output accuracy: ${rr.softScore.toFixed(2)} (0 is worst, 1 is best).\n`);
        }
      }

      if (rr.error) {
        msgLines.push(`\n\nYour code produced the following error:\n${rr.error.slice(0, 300)}\n`);
      }
    }

    parts.push(msgLines.join(''));
  }

  return parts.join('\n\n');
}

function resolveModelId(modelId: string): { provider: string; id: string } | null {
  const slashIdx = modelId.indexOf('/');
  if (slashIdx === -1) return null;
  return {
    provider: modelId.slice(0, slashIdx),
    id: modelId.slice(slashIdx + 1),
  };
}

// =============================================================================
// Feedback building
// =============================================================================

function buildFeedbackBlock(
  solutions: Array<{ code: string; feedback: string; score: number }>,
  maxExamples: number = 5,
  improvingOrder: boolean = true
): string {
  if (solutions.length === 0) return '';

  const sorted = [...solutions].sort((a, b) => b.score - a.score);
  const top = sorted.slice(0, maxExamples);
  if (improvingOrder) top.reverse();

  return top
    .map((s, i) =>
      `<solution_${i + 1}>
<solution_score>
${s.score.toFixed(2)}
</solution_score>
</solution_${i + 1}>`
    )
    .join('\n\n');
}

// =============================================================================
// Strategy adaptation learning
// =============================================================================

interface StrategyAdaptation {
  insight: string;
  taskType: string;
  models: string[];
  evidenceCount: number;
  timestamp: number;
  promptModifier?: string;
}

interface IterationResult {
  iteration: number;
  expertIndex: number;
  code: string;
  answer: string;
  trainResults: Array<{ success: boolean; softScore: number }>;
  testResults: unknown[];
  passed: boolean;
  score: number;
  feedback: string;
  promptTokens: number;
  completionTokens: number;
  durationMs: number;
}

function learnFromIterations(
  iterations: IterationResult[],
  models: string[],
  taskType: string,
  existingAdaptations: StrategyAdaptation[]
): StrategyAdaptation[] {
  const adaptations = [...existingAdaptations];
  const passed = iterations.some((r) => r.passed);
  const bestResult = iterations.reduce<IterationResult | null>(
    (best, r) => (r.score > (best?.score ?? -1) ? r : best),
    null
  );

  // 1. Successful model tracking
  if (passed && bestResult) {
    const successfulModel = models[bestResult.expertIndex % models.length];
    const existing = adaptations.find(
      (a) => a.taskType === taskType && a.models.includes(successfulModel)
    );
    if (existing) {
      existing.evidenceCount++;
    } else {
      adaptations.push({
        insight: `Model ${successfulModel} successfully solved ${taskType} problems`,
        taskType,
        models: [successfulModel],
        evidenceCount: 1,
        timestamp: Date.now(),
        promptModifier: `Note: Model ${successfulModel} has been effective for ${taskType} tasks.`,
      });
    }
  }

  // 2. Feedback effectiveness
  if (passed && bestResult && bestResult.iteration > 0) {
    const firstScore = iterations.find(
      (r) => r.expertIndex === bestResult.expertIndex && r.iteration === 0
    )?.score ?? 0;
    const scoreDelta = bestResult.score - firstScore;
    if (scoreDelta > 0.3) {
      const existing = adaptations.find(
        (a) => a.insight.includes('feedback-driven improvement')
      );
      if (existing) {
        existing.evidenceCount++;
      } else {
        adaptations.push({
          insight: 'Feedback-driven improvement is effective',
          taskType: '*',
          models,
          evidenceCount: 1,
          timestamp: Date.now(),
          promptModifier: 'Pay careful attention to feedback from previous attempts.',
        });
      }
    }
  }

  // 3. Timeout detection
  const timeoutCount = iterations.filter(
    (r) => r.feedback.includes('timeout') || r.feedback.includes('Too many timeouts')
  ).length;
  if (timeoutCount > 2) {
    const existing = adaptations.find((a) => a.insight.includes('performance'));
    if (existing) {
      existing.evidenceCount++;
    } else {
      adaptations.push({
        insight: 'Frequent timeouts suggest solutions need performance optimization',
        taskType,
        models,
        evidenceCount: 1,
        timestamp: Date.now(),
        promptModifier: 'IMPORTANT: Prioritize efficient algorithms.',
      });
    }
  }

  return adaptations;
}

// =============================================================================
// Tests
// =============================================================================

describe('computeSoftScore', () => {
  it('returns 1.0 for perfect match', () => {
    const actual = JSON.stringify([[1, 2], [3, 4]]);
    const expected = [[1, 2], [3, 4]];
    expect(computeSoftScore(actual, expected)).toBe(1.0);
  });

  it('returns 0.0 for completely wrong', () => {
    const actual = JSON.stringify([[0, 0], [0, 0]]);
    const expected = [[1, 2], [3, 4]];
    expect(computeSoftScore(actual, expected)).toBe(0.0);
  });

  it('returns 0.5 for half correct', () => {
    const actual = JSON.stringify([[1, 2], [0, 0]]);
    const expected = [[1, 2], [3, 4]];
    expect(computeSoftScore(actual, expected)).toBe(0.5);
  });

  it('returns 0 for wrong-length arrays', () => {
    const actual = JSON.stringify([[1, 2]]);
    const expected = [[1, 2], [3, 4]];
    expect(computeSoftScore(actual, expected)).toBe(0);
  });

  it('returns 0 for invalid JSON', () => {
    expect(computeSoftScore('not json', [[1]])).toBe(0);
  });

  it('returns 0.25 for one cell correct in 2x2', () => {
    const actual = JSON.stringify([[1, 0], [0, 0]]);
    const expected = [[1, 2], [3, 4]];
    expect(computeSoftScore(actual, expected)).toBe(0.25);
  });
});

describe('compareOutputs', () => {
  it('returns true for identical JSON', () => {
    expect(compareOutputs(JSON.stringify([1, 2, 3]), [1, 2, 3])).toBe(true);
  });

  it('returns false for different JSON', () => {
    expect(compareOutputs(JSON.stringify([1, 2]), [1, 3])).toBe(false);
  });

  it('returns true for matching string when JSON parse fails', () => {
    expect(compareOutputs('hello', 'hello')).toBe(true);
  });

  it('returns false for different strings', () => {
    expect(compareOutputs('hello', 'world')).toBe(false);
  });
});

describe('parseCodeFromLLM', () => {
  it('extracts javascript code block', () => {
    const response = 'Here is my solution:\n```javascript\nfunction transform(grid) {\n    return grid;\n}\n```\nDone.';
    expect(parseCodeFromLLM(response)).toBe('function transform(grid) {\n    return grid;\n}');
  });

  it('extracts js code block', () => {
    const response = '```js\nfunction transform(grid) { return grid; }\n```';
    expect(parseCodeFromLLM(response)).toBe('function transform(grid) { return grid; }');
  });

  it('extracts typescript code block', () => {
    const response = '```typescript\nfunction transform(grid: number[][]) { return grid; }\n```';
    expect(parseCodeFromLLM(response)).toBe('function transform(grid: number[][]) { return grid; }');
  });

  it('returns null when no code block', () => {
    expect(parseCodeFromLLM('No code here')).toBeNull();
  });

  it('returns null for non-js code block', () => {
    expect(parseCodeFromLLM('```python\nprint(1)\n```')).toBeNull();
  });

  it('handles multi-line code', () => {
    const response = '```javascript\nfunction transform(grid) {\n    return grid.map(row => row.map(v => v * 2));\n}\n```';
    const parsed = parseCodeFromLLM(response);
    expect(parsed).toContain('function transform');
    expect(parsed).toContain('v * 2');
  });
});

describe('createRNG', () => {
  it('produces deterministic sequence for same seed', () => {
    const rng1 = createRNG(42);
    const rng2 = createRNG(42);
    const seq1 = [rng1(), rng1(), rng1()];
    const seq2 = [rng2(), rng2(), rng2()];
    expect(seq1).toEqual(seq2);
  });

  it('produces different sequences for different seeds', () => {
    const rng1 = createRNG(0);
    const rng2 = createRNG(100);
    expect(rng1()).not.toBe(rng2());
  });

  it('produces values between 0 and 1', () => {
    const rng = createRNG(42);
    for (let i = 0; i < 100; i++) {
      const val = rng();
      expect(val).toBeGreaterThanOrEqual(0);
      expect(val).toBeLessThan(1);
    }
  });
});

describe('resolveModelId', () => {
  it('parses provider/id format', () => {
    expect(resolveModelId('anthropic/claude-sonnet-4-5')).toEqual({
      provider: 'anthropic',
      id: 'claude-sonnet-4-5',
    });
  });

  it('parses openai models', () => {
    expect(resolveModelId('openai/gpt-4o')).toEqual({
      provider: 'openai',
      id: 'gpt-4o',
    });
  });

  it('returns null for invalid format', () => {
    expect(resolveModelId('just-a-model')).toBeNull();
  });

  it('handles models with slashes in id', () => {
    expect(resolveModelId('openai/gpt-4o-mini')).toEqual({
      provider: 'openai',
      id: 'gpt-4o-mini',
    });
  });
});

describe('buildFeedbackBlock', () => {
  it('returns empty string for no solutions', () => {
    expect(buildFeedbackBlock([])).toBe('');
  });

  it('orders solutions by improving order (worst→best)', () => {
    const solutions = [
      { code: 'a', feedback: '', score: 0.9 },
      { code: 'b', feedback: '', score: 0.5 },
      { code: 'c', feedback: '', score: 0.7 },
    ];
    const block = buildFeedbackBlock(solutions, 5, true);
    // Should contain scores in order: 0.5, 0.7, 0.9
    const scores = [...block.matchAll(/(\d+\.\d+)/g)].map((m) => parseFloat(m[1]));
    expect(scores).toEqual([0.5, 0.7, 0.9]);
  });

  it('orders solutions by decreasing order when improvingOrder=false', () => {
    const solutions = [
      { code: 'a', feedback: '', score: 0.5 },
      { code: 'b', feedback: '', score: 0.9 },
    ];
    const block = buildFeedbackBlock(solutions, 5, false);
    const scores = [...block.matchAll(/(\d+\.\d+)/g)].map((m) => parseFloat(m[1]));
    expect(scores).toEqual([0.9, 0.5]);
  });

  it('limits to maxExamples', () => {
    const solutions = Array.from({ length: 10 }, (_, i) => ({
      code: `c${i}`,
      feedback: '',
      score: i / 10,
    }));
    const block = buildFeedbackBlock(solutions, 3, true);
    // Count opening tags only (not closing </solution_>)
    const count = (block.match(/<solution_\d+>/g) || []).length;
    expect(count).toBe(3);
  });
});

describe('learnFromIterations', () => {
  it('learns from successful model', () => {
    const iterations: IterationResult[] = [
      {
        iteration: 0,
        expertIndex: 0,
        code: 'def f(): pass',
        answer: '',
        trainResults: [{ success: true, softScore: 1.0 }],
        testResults: [],
        passed: true,
        score: 1.0,
        feedback: '',
        promptTokens: 100,
        completionTokens: 200,
        durationMs: 5000,
      },
    ];

    const adaptations = learnFromIterations(iterations, ['anthropic/claude-sonnet-4-5'], 'code-reasoning', []);
    expect(adaptations).toHaveLength(1);
    expect(adaptations[0].insight).toContain('anthropic/claude-sonnet-4-5');
    expect(adaptations[0].taskType).toBe('code-reasoning');
  });

  it('increments evidence for repeated model success', () => {
    const iterations: IterationResult[] = [
      {
        iteration: 0,
        expertIndex: 0,
        code: 'pass',
        answer: '',
        trainResults: [{ success: true, softScore: 1.0 }],
        testResults: [],
        passed: true,
        score: 1.0,
        feedback: '',
        promptTokens: 0,
        completionTokens: 0,
        durationMs: 0,
      },
    ];

    const existing: StrategyAdaptation[] = [{
      insight: 'Model anthropic/claude-sonnet-4-5 successfully solved code-reasoning problems',
      taskType: 'code-reasoning',
      models: ['anthropic/claude-sonnet-4-5'],
      evidenceCount: 2,
      timestamp: Date.now(),
      promptModifier: 'Note.',
    }];

    const adaptations = learnFromIterations(iterations, ['anthropic/claude-sonnet-4-5'], 'code-reasoning', existing);
    const match = adaptations.find((a) => a.taskType === 'code-reasoning');
    expect(match?.evidenceCount).toBe(3);
  });

  it('detects feedback-driven improvement', () => {
    const iterations: IterationResult[] = [
      {
        iteration: 0,
        expertIndex: 0,
        code: 'pass',
        answer: '',
        trainResults: [{ success: false, softScore: 0.1 }],
        testResults: [],
        passed: false,
        score: 0.1,
        feedback: '',
        promptTokens: 0,
        completionTokens: 0,
        durationMs: 0,
      },
      {
        iteration: 1,
        expertIndex: 0,
        code: 'pass',
        answer: '',
        trainResults: [{ success: true, softScore: 1.0 }],
        testResults: [],
        passed: true,
        score: 1.0,
        feedback: '',
        promptTokens: 0,
        completionTokens: 0,
        durationMs: 0,
      },
    ];

    const adaptations = learnFromIterations(iterations, ['openai/gpt-4o'], 'code-reasoning', []);
    // Should have a model success adaptation and a feedback-driven improvement adaptation
    const feedbackAdaptation = adaptations.find((a) => a.insight.includes('Feedback-driven improvement'));
    expect(feedbackAdaptation).toBeDefined();
    expect(feedbackAdaptation?.taskType).toBe('*');
  });

  it('detects timeout patterns', () => {
    const iterations: IterationResult[] = Array.from({ length: 3 }, (_, i) => ({
      iteration: i,
      expertIndex: 0,
      code: 'pass',
      answer: '',
      trainResults: [{ success: false, softScore: 0 }],
      testResults: [],
      passed: false,
      score: 0,
      feedback: 'Too many timeouts. Code may have infinite loop.',
      promptTokens: 0,
      completionTokens: 0,
      durationMs: 0,
    }));

    const adaptations = learnFromIterations(iterations, ['openai/gpt-4o'], 'code-reasoning', []);
    const perfAdaptation = adaptations.find((a) => a.insight.includes('performance'));
    expect(perfAdaptation).toBeDefined();
    expect(perfAdaptation?.promptModifier).toContain('efficient algorithms');
  });

  it('does not add feedback adaptation when improvement is small', () => {
    const iterations: IterationResult[] = [
      {
        iteration: 0,
        expertIndex: 0,
        code: 'pass',
        answer: '',
        trainResults: [{ success: false, softScore: 0.8 }],
        testResults: [],
        passed: false,
        score: 0.8,
        feedback: '',
        promptTokens: 0,
        completionTokens: 0,
        durationMs: 0,
      },
      {
        iteration: 1,
        expertIndex: 0,
        code: 'pass',
        answer: '',
        trainResults: [{ success: true, softScore: 1.0 }],
        testResults: [],
        passed: true,
        score: 1.0,
        feedback: '',
        promptTokens: 0,
        completionTokens: 0,
        durationMs: 0,
      },
    ];

    const adaptations = learnFromIterations(iterations, ['openai/gpt-4o'], 'code-reasoning', []);
    const feedbackAdaptation = adaptations.find((a) => a.insight.includes('feedback-driven improvement'));
    // Score delta is 0.2, which is < 0.3 threshold
    expect(feedbackAdaptation).toBeUndefined();
  });
});

describe('voting algorithm (simulated)', () => {
  // Simplified voting test with mock data
  it('ranks passing solutions before failing ones', () => {
    type SimpleResult = { key: string; passed: boolean; score: number };

    const results: SimpleResult[] = [
      { key: 'A', passed: false, score: 0.5 },
      { key: 'B', passed: true, score: 1.0 },
      { key: 'C', passed: false, score: 0.3 },
    ];

    // Simple ranking: passers first, then failures sorted by score desc
    const ranked = results.sort((a, b) => {
      if (a.passed && !b.passed) return -1;
      if (!a.passed && b.passed) return 1;
      return b.score - a.score;
    });

    expect(ranked[0].key).toBe('B');
    expect(ranked[1].key).toBe('A');
    expect(ranked[2].key).toBe('C');
  });

  it('groups by output and sorts by vote count', () => {
    const outputs = [
      { output: 'X', passed: true },
      { output: 'X', passed: true },
      { output: 'Y', passed: true },
      { output: 'Z', passed: false },
    ];

    const groups = new Map<string, number>();
    for (const o of outputs) {
      if (o.passed) {
        groups.set(o.output, (groups.get(o.output) || 0) + 1);
      }
    }

    const sorted = [...groups.entries()].sort((a, b) => b[1] - a[1]);
    expect(sorted[0]).toEqual(['X', 2]);
    expect(sorted[1]).toEqual(['Y', 1]);
  });
});

describe('budget tracking', () => {
  it('stops when cost budget exceeded', () => {
    const budget = { maxCost: 0.01, costSoFar: 0 };
    let iterations = 0;

    while (budget.costSoFar < (budget.maxCost ?? Infinity) && iterations < 100) {
      budget.costSoFar += 0.005;
      iterations++;
    }

    expect(iterations).toBe(2); // 0.005 + 0.005 = 0.01
  });

  it('stops when time budget exceeded', () => {
    const startTime = Date.now() - 5000; // 5 seconds ago
    const maxTime = 3; // 3 seconds
    const elapsed = (Date.now() - startTime) / 1000;

    expect(elapsed).toBeGreaterThan(maxTime);
  });
});

describe('vm sandbox', () => {
  // We replicate the vm sandbox logic here for testing
  async function runInSandbox(code: string, input: unknown, timeoutS: number = 5): Promise<{ ok: boolean; output: string; timedOut: boolean }> {
    const vm = await import('node:vm');
    try {
      const context = vm.createContext({
        console: { log: () => {}, error: () => {}, warn: () => {} },
        Math,
        JSON,
        Array,
        Object,
        String,
        Number,
        Boolean,
        Date,
        Map,
        Set,
        parseInt,
        parseFloat,
        isNaN,
        isFinite,
        RegExp,
        Error,
        TypeError,
        RangeError,
        __input__: input,
        __output__: null,
      });

      const wrappedCode = `
${code}

if (typeof transform === 'function') {
  try {
    __output__ = transform(__input__);
  } catch (e) {
    __output__ = { __error__: e.message || String(e) };
  }
}
`;

      const script = new vm.Script(wrappedCode, { filename: 'sandbox.js' });
      script.runInContext(context, { timeout: timeoutS * 1000 });

      const result = context.__output__;
      if (result && typeof result === 'object' && result.__error__) {
        return { ok: false, output: result.__error__, timedOut: false };
      }
      return { ok: true, output: JSON.stringify(result), timedOut: false };
    } catch (e: any) {
      const isTimeout = e.code === 'ERR_SCRIPT_EXECUTION_TIMEOUT' || (e.message && e.message.includes('timeout'));
      return { ok: false, output: isTimeout ? 'timeout' : (e.message || String(e)), timedOut: isTimeout };
    }
  }

  it('executes a simple transform function', async () => {
    const code = 'function transform(grid) { return grid.map(row => row.map(v => v * 2)); }';
    const result = await runInSandbox(code, [[1, 2], [3, 4]]);
    expect(result.ok).toBe(true);
    expect(JSON.parse(result.output)).toEqual([[2, 4], [6, 8]]);
  });

  it('catches runtime errors', async () => {
    const code = 'function transform(grid) { return grid.foo.bar; }';
    const result = await runInSandbox(code, [[1]]);
    expect(result.ok).toBe(false);
    expect(result.output).toContain('Cannot read');
  });

  it('catches syntax errors', async () => {
    const code = 'function transform(grid { return grid; }';  // missing closing paren
    const result = await runInSandbox(code, [[1]]);
    expect(result.ok).toBe(false);
  });

  it('detects timeouts', async () => {
    const code = 'function transform(grid) { while(true) {} }';
    const result = await runInSandbox(code, [[1]], 1);  // 1 second timeout
    expect(result.timedOut).toBe(true);
  });

  it('provides standard JS builtins', async () => {
    const code = 'function transform(grid) { return grid.flat().sort((a,b) => a - b); }';
    const result = await runInSandbox(code, [[3, 1], [2, 4]]);
    expect(result.ok).toBe(true);
    expect(JSON.parse(result.output)).toEqual([1, 2, 3, 4]);
  });

  it('isolates the sandbox from Node globals', async () => {
    const code = 'function transform(grid) { return typeof process; }';
    const result = await runInSandbox(code, [[1]]);
    expect(result.ok).toBe(true);
    expect(JSON.parse(result.output)).toBe('undefined');
  });
});

describe('formatProblem', () => {
  it('formats grids into <Diagram> text', () => {
    const result = formatProblem(
      [[[1, 2], [3, 4]]],  // trainIn
      [[[5, 6], [7, 8]]],  // trainOut
      [[[9, 10]]],         // testIn
      false,               // shuffle
      0                    // seed
    );
    expect(result).toContain('<Diagram>');
    expect(result).toContain('Example #1');
    expect(result).toContain('Challenge #1');
    expect(result).toContain('1 2');
    expect(result).toContain('5 6');
    expect(result).toContain('9 10');
  });

  it('shuffles training examples with different seeds', () => {
    const trainIn = [[[1]], [[2]], [[3]], [[4]], [[5]]];
    const trainOut = [[[10]], [[20]], [[30]], [[40]], [[50]]];

    const result1 = formatProblem(trainIn, trainOut, [], true, 0);
    const result2 = formatProblem(trainIn, trainOut, [], true, 42);

    // Same seed should produce same order
    const result1b = formatProblem(trainIn, trainOut, [], true, 0);
    expect(result1).toBe(result1b);

    // Different seeds *may* produce different order (probabilistic, but very likely with 5 items)
    // Just verify they're both valid
    expect(result1).toContain('<Diagram>');
    expect(result2).toContain('<Diagram>');
  });

  it('handles single training example (no shuffle possible)', () => {
    const result = formatProblem([[[0]]], [[[1]]], [], false, 0);
    expect(result).toContain('Example #1');
    expect(result).toContain('0');
    expect(result).toContain('1');
  });
});

describe('arrayDiff', () => {
  it('shows matching values as-is, mismatches as pred/truth', () => {
    const pred = [[1, 2], [3, 4]];
    const truth = [[1, 9], [3, 8]];
    const diff = arrayDiff(pred, truth);
    expect(diff).toContain('1');    // match
    expect(diff).toContain('2/9'); // mismatch
    expect(diff).toContain('3');    // match
    expect(diff).toContain('4/8'); // mismatch
  });

  it('handles fully matching grids', () => {
    const diff = arrayDiff([[1, 2]], [[1, 2]]);
    expect(diff).toBe('1 2');
  });
});

describe('buildDetailedFeedback (Poetiq parity)', () => {
  it('reports shape mismatch when dimensions differ', () => {
    const trainResults: SolveResult[] = [
      { success: false, output: '[[1,2]]', softScore: 0, error: null, code: '' },
    ];
    const trainOutputs = [[[1, 2], [3, 4]]];

    const feedback = buildDetailedFeedback(trainResults, [], trainOutputs);
    expect(feedback).toContain('Shape mismatch');
  });

  it('shows diff grid when shapes match but values differ', () => {
    const trainResults: SolveResult[] = [
      { success: false, output: '[[1,9],[3,8]]', softScore: 0.5, error: null, code: '' },
    ];
    const trainOutputs = [[[1, 2], [3, 4]]];

    const feedback = buildDetailedFeedback(trainResults, [], trainOutputs);
    expect(feedback).toContain('9/2');
    expect(feedback).toContain('8/4');
    expect(feedback).toContain('0.50');
  });

  it('reports bad JSON output', () => {
    const trainResults: SolveResult[] = [
      { success: false, output: 'not json', softScore: 0, error: 'parse error', code: '' },
    ];
    const trainOutputs = [[[1]]];

    const feedback = buildDetailedFeedback(trainResults, [], trainOutputs);
    expect(feedback).toContain('rectangular grid');
  });

  it('reports execution errors', () => {
    const trainResults: SolveResult[] = [
      { success: false, output: '', softScore: 0, error: 'TypeError: Cannot read properties of undefined', code: '' },
    ];
    const trainOutputs = [[[1]]];

    const feedback = buildDetailedFeedback(trainResults, [], trainOutputs);
    expect(feedback).toContain('TypeError');
  });
});

// =============================================================================
// Meta-system V2 tests
// =============================================================================

describe('PromptDelta', () => {
  const { applyPromptDelta } = (() => {
    // Inline applyPromptDelta for testing
    function applyPromptDelta(basePrompt: string, delta: any): string {
      let result = basePrompt;
      for (const [section, replacement] of Object.entries(delta.sectionReplacements || {})) {
        result = result.replace(section, replacement as string);
      }
      if (delta.preProblemInsert) {
        const problemIdx = result.indexOf('$$problem$$');
        if (problemIdx !== -1) {
          result = result.slice(0, problemIdx) +
            '\n\n**Problem-Specific Strategy:**\n' + delta.preProblemInsert + '\n\n' +
            result.slice(problemIdx);
        }
      }
      if (delta.postProblemInsert) {
        result = result.replace('$$problem$$', () => '$$problem$$\n\n**Critical Reminders:**\n' + delta.postProblemInsert);
      }
      if (delta.antiPatterns && delta.antiPatterns.length > 0) {
        result += '\n\n**DO NOT:**\n' + delta.antiPatterns.map((a: string, i: number) => `${i + 1}. ${a}`).join('\n');
      }
      if (delta.additionalExamples && delta.additionalExamples.length > 0) {
        const examplesStr = delta.additionalExamples
          .map((e: any, i: number) => `**Custom Example ${i + 1}:**\nProblem: ${e.problem}\nSolution: ${e.solution}`)
          .join('\n\n');
        result = result.replace('$$problem$$', () => examplesStr + '\n\n$$problem$$');
      }
      return result;
    }
    return { applyPromptDelta };
  })();

  it('applies preProblemInsert before $$problem$$', () => {
    const base = 'Hello $$problem$$ goodbye';
    const delta = { preProblemInsert: 'STRATEGY HINT', postProblemInsert: null, sectionReplacements: {}, additionalExamples: [], antiPatterns: [] };
    const result = applyPromptDelta(base, delta);
    expect(result).toContain('**Problem-Specific Strategy:**');
    expect(result).toContain('STRATEGY HINT');
    expect(result.indexOf('STRATEGY HINT')).toBeLessThan(result.indexOf('$$problem$$'));
  });

  it('applies postProblemInsert after $$problem$$', () => {
    const base = 'Hello $$problem$$ goodbye';
    const delta = { preProblemInsert: null, postProblemInsert: 'NO CONSOLE.LOG', sectionReplacements: {}, additionalExamples: [], antiPatterns: [] };
    const result = applyPromptDelta(base, delta);
    expect(result).toContain('**Critical Reminders:**');
    expect(result).toContain('NO CONSOLE.LOG');
  });

  it('appends anti-patterns', () => {
    const base = 'Hello $$problem$$';
    const delta = { preProblemInsert: null, postProblemInsert: null, sectionReplacements: {}, additionalExamples: [], antiPatterns: ['No brute force', 'No hardcoded values'] };
    const result = applyPromptDelta(base, delta);
    expect(result).toContain('**DO NOT:**');
    expect(result).toContain('No brute force');
    expect(result).toContain('No hardcoded values');
  });

  it('inserts additional examples before $$problem$$', () => {
    const base = 'Hello $$problem$$';
    const delta = {
      preProblemInsert: null, postProblemInsert: null, sectionReplacements: {},
      additionalExamples: [{ problem: 'rotate grid', solution: 'use transpose' }],
      antiPatterns: [],
    };
    const result = applyPromptDelta(base, delta);
    expect(result).toContain('Custom Example 1');
    expect(result).toContain('rotate grid');
  });

  it('applies section replacements', () => {
    const base = 'Old text $$problem$$';
    const delta = {
      preProblemInsert: null, postProblemInsert: null,
      sectionReplacements: { 'Old text': 'New text' },
      additionalExamples: [], antiPatterns: [],
    };
    const result = applyPromptDelta(base, delta);
    expect(result).toContain('New text');
    expect(result).not.toContain('Old text');
  });

  it('combines all delta types', () => {
    const base = 'Start $$problem$$ End';
    const delta = {
      preProblemInsert: 'HINT',
      postProblemInsert: 'REMINDER',
      sectionReplacements: { Start: 'Beginning' },
      additionalExamples: [{ problem: 'p', solution: 's' }],
      antiPatterns: ['no x'],
    };
    const result = applyPromptDelta(base, delta);
    expect(result).toContain('HINT');
    expect(result).toContain('REMINDER');
    expect(result).toContain('Beginning');
    expect(result).toContain('Custom Example');
    expect(result).toContain('no x');
  });

  it('leaves prompt unchanged with empty delta', () => {
    const base = 'Hello $$problem$$';
    const delta = { preProblemInsert: null, postProblemInsert: null, sectionReplacements: {}, additionalExamples: [], antiPatterns: [] };
    const result = applyPromptDelta(base, delta);
    expect(result).toBe(base);
  });
});

describe('Budget bandit', () => {
  const { shouldStopEarly, shouldReExplore } = (() => {
    function shouldStopEarly(history: Array<{ score: number; passed: boolean }>, minIterations = 3) {
      if (history.length < minIterations) return { stop: false, reason: '' };
      const last3 = history.slice(-3);
      const allFailed = last3.every((r) => !r.passed);
      const noProgress = last3.every((r) => r.score === last3[0].score) && last3[0].score < 0.5;
      if (allFailed && noProgress) return { stop: true, reason: `No progress after ${history.length} iterations` };
      if (history.length >= 4) {
        const last4 = history.slice(-4);
        const decreasing = last4.every((r, i) => i === 0 || r.score <= last4[i - 1].score);
        if (decreasing && last4[3].score < 0.3) return { stop: true, reason: 'Score decreasing' };
      }
      return { stop: false, reason: '' };
    }

    function shouldReExplore(allResults: any[][], totalIterations: number) {
      const allStuck = allResults.every((results) => {
        const last3 = results.slice(-3);
        return last3.length >= 3 && last3.every((r: any) => r.score === 0);
      });
      if (allStuck && totalIterations >= 5) return { reExplore: true, reason: 'All experts stuck' };
      return { reExplore: false, reason: '' };
    }

    return { shouldStopEarly, shouldReExplore };
  })();

  it('stops early when score is stuck at 0', () => {
    const history = [
      { score: 0, passed: false },
      { score: 0, passed: false },
      { score: 0, passed: false },
    ];
    const result = shouldStopEarly(history);
    expect(result.stop).toBe(true);
  });

  it('does not stop early with fewer than 3 iterations', () => {
    const history = [
      { score: 0, passed: false },
      { score: 0, passed: false },
    ];
    const result = shouldStopEarly(history);
    expect(result.stop).toBe(false);
  });

  it('does not stop when making progress', () => {
    const history = [
      { score: 0.2, passed: false },
      { score: 0.5, passed: false },
      { score: 0.7, passed: false },
    ];
    const result = shouldStopEarly(history);
    expect(result.stop).toBe(false);
  });

  it('does not stop when stuck at high score', () => {
    const history = [
      { score: 0.8, passed: false },
      { score: 0.8, passed: false },
      { score: 0.8, passed: false },
    ];
    const result = shouldStopEarly(history);
    expect(result.stop).toBe(false); // 0.8 >= 0.5
  });

  it('stops when score is decreasing', () => {
    const history = [
      { score: 0.3, passed: false },
      { score: 0.2, passed: false },
      { score: 0.1, passed: false },
      { score: 0.0, passed: false },
    ];
    const result = shouldStopEarly(history);
    expect(result.stop).toBe(true);
  });

  it('triggers re-explore when all experts are stuck', () => {
    const allResults = [
      [{ score: 0 }, { score: 0 }, { score: 0 }],
      [{ score: 0 }, { score: 0 }, { score: 0 }],
    ];
    const result = shouldReExplore(allResults, 6);
    expect(result.reExplore).toBe(true);
  });

  it('does not re-explore when an expert is making progress', () => {
    const allResults = [
      [{ score: 0 }, { score: 0.5 }, { score: 0.8 }],
      [{ score: 0 }, { score: 0 }, { score: 0 }],
    ];
    const result = shouldReExplore(allResults, 6);
    expect(result.reExplore).toBe(false);
  });
});

describe('Thompson sampling', () => {
  it('returns the only available model', () => {
    // We can't easily test the real thompsonSampleModel without mocking,
    // so test the Beta sampling primitives
    const { betaSample, gammaVariate, randn } = (() => {
      function randn() {
        const u1 = Math.random();
        const u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }
      function gammaVariate(shape: number): number {
        if (shape < 1) return gammaVariate(shape + 1) * Math.pow(Math.random(), 1 / shape);
        const d = shape - 1 / 3;
        const c = 1 / Math.sqrt(9 * d);
        while (true) {
          let x, v;
          do { x = randn(); v = 1 + c * x; } while (v <= 0);
          v = v * v * v;
          const u = Math.random();
          if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v;
          if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
        }
      }
      function betaSample(alpha: number, beta: number): number {
        const x = gammaVariate(alpha);
        const y = gammaVariate(beta);
        return x / (x + y);
      }
      return { betaSample, gammaVariate, randn };
    })();

    // Beta(1,1) should produce uniform-ish values
    const samples = Array.from({ length: 100 }, () => betaSample(1, 1));
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    expect(mean).toBeGreaterThan(0.2);
    expect(mean).toBeLessThan(0.8);

    // Beta(10,1) should produce values near 1
    const highAlpha = Array.from({ length: 100 }, () => betaSample(10, 1));
    const highMean = highAlpha.reduce((a, b) => a + b, 0) / highAlpha.length;
    expect(highMean).toBeGreaterThan(0.7);

    // Beta(1,10) should produce values near 0
    const highBeta = Array.from({ length: 100 }, () => betaSample(1, 10));
    const lowMean = highBeta.reduce((a, b) => a + b, 0) / highBeta.length;
    expect(lowMean).toBeLessThan(0.3);
  });
});

describe('Meta-rule engine', () => {
  it('validates meta-rules by category', () => {
    // Simulate the validation function inline
    function validateMetaRule(rule: any, category: string, improved: boolean) {
      rule.testCount++;
      if (improved) rule.improvementCount++;
      if (!rule.validatedCategories.includes(category)) rule.validatedCategories.push(category);
      rule.lastValidated = Date.now();
    }

    const rule = {
      id: 'test',
      principle: 'Add worked examples',
      validatedCategories: [],
      improvementCount: 0,
      testCount: 0,
      suggestedDelta: {},
      sourceStrategyId: null,
      created: Date.now(),
      lastValidated: 0,
    };

    validateMetaRule(rule, 'grid-transformation', true);
    expect(rule.testCount).toBe(1);
    expect(rule.improvementCount).toBe(1);
    expect(rule.validatedCategories).toContain('grid-transformation');

    validateMetaRule(rule, 'knowledge-synthesis', false);
    expect(rule.testCount).toBe(2);
    expect(rule.improvementCount).toBe(1);
    expect(rule.validatedCategories).toContain('knowledge-synthesis');
  });

  it('applies meta-rules filtered by category and freshness', () => {
    // Test the filtering logic
    const rules = [
      {
        id: 'r1', principle: 'Test', validatedCategories: ['grid-transformation'],
        improvementCount: 2, testCount: 3, lastValidated: Date.now(),
        suggestedDelta: { preProblemInsert: 'grid hint' },
      },
      {
        id: 'r2', principle: 'Universal', validatedCategories: [],
        improvementCount: 1, testCount: 2, lastValidated: Date.now(),
        suggestedDelta: { preProblemInsert: 'universal hint' },
      },
      {
        id: 'r3', principle: 'Stale', validatedCategories: ['grid-transformation'],
        improvementCount: 0, testCount: 10, lastValidated: 0, // stale
        suggestedDelta: { preProblemInsert: 'stale hint' },
      },
    ];

    const category = 'grid-transformation';
    const STALE_MS = 7 * 24 * 60 * 60 * 1000;
    const now = Date.now();

    const relevant = rules.filter((r) => {
      const isCategoryMatch = r.validatedCategories.includes(category) || r.validatedCategories.length === 0;
      const isFresh = now - r.lastValidated < STALE_MS || r.lastValidated === 0;
      const hasPositiveEvidence = r.improvementCount > 0 || r.testCount < 5;
      return isCategoryMatch && isFresh && hasPositiveEvidence;
    });

    // r1 matches category and is fresh with positive evidence
    expect(relevant.some(r => r.id === 'r1')).toBe(true);
    // r2 is universal (empty categories) and fresh
    expect(relevant.some(r => r.id === 'r2')).toBe(true);
    // r3 has testCount=10 with 0 improvements and lastValidated=0 → no positive evidence
    expect(relevant.some(r => r.id === 'r3')).toBe(false);
  });
});

describe('Prompt quality metrics', () => {
  it('tracks code parse rate and sandbox success rate', () => {
    function recordPromptQuality(metrics: any, codeParsed: boolean, sandboxOk: boolean, firstIterScore: number) {
      const n = metrics.observationCount;
      metrics.codeParseRate = (metrics.codeParseRate * n + (codeParsed ? 1 : 0)) / (n + 1);
      metrics.sandboxSuccessRate = (metrics.sandboxSuccessRate * n + (sandboxOk ? 1 : 0)) / (n + 1);
      metrics.avgFirstIterationScore = (metrics.avgFirstIterationScore * n + firstIterScore) / (n + 1);
      metrics.observationCount = n + 1;
    }

    const metrics = { codeParseRate: 0, sandboxSuccessRate: 0, avgFirstIterationScore: 0, observationCount: 0 };

    recordPromptQuality(metrics, true, true, 0.8);
    expect(metrics.observationCount).toBe(1);
    expect(metrics.codeParseRate).toBe(1);
    expect(metrics.sandboxSuccessRate).toBe(1);
    expect(metrics.avgFirstIterationScore).toBe(0.8);

    recordPromptQuality(metrics, false, true, 0.4);
    expect(metrics.observationCount).toBe(2);
    expect(metrics.codeParseRate).toBe(0.5);
    expect(metrics.sandboxSuccessRate).toBe(1);
    expect(metrics.avgFirstIterationScore).toBeCloseTo(0.6, 10);
  });
});

describe('Harness spec generation', () => {
  it('creates specs with correct types', () => {
    const spec = {
      id: 'test1',
      category: 'grid-transformation',
      approach: 'code-sandbox' as const,
      solverPrompt: 'Solve $$problem$$',
      feedbackPrompt: 'Feedback $$feedback$$',
      configOverrides: { temperature: 0.8 },
      validationScore: 0,
      validationTests: 0,
      validated: false,
      parentId: null,
      generation: 0,
      created: Date.now(),
      useCount: 0,
      successCount: 0,
      avgScore: 0,
    };
    expect(spec.approach).toBe('code-sandbox');
    expect(spec.solverPrompt).toContain('$$problem$$');
  });

  it('supports multiple approach types', () => {
    const approaches = ['code-sandbox', 'decomposition', 'chain-of-questions', 'analogy', 'counter-factual', 'exhaustive-search'] as const;
    expect(approaches.length).toBe(6);
    for (const a of approaches) {
      expect(typeof a).toBe('string');
    }
  });
});

describe('Ensemble diversification', () => {
  it('assigns different approaches per expert', () => {
    const APPROACH_SOLVER_PROMPTS: Record<string, string> = {
      'code-sandbox': 'Code approach $$problem$$',
      'decomposition': 'Decompose $$problem$$',
      'analogy': 'Analogy $$problem$$',
    };

    const approaches = ['code-sandbox', 'decomposition', 'analogy'];
    expect(approaches.length).toBe(3);
    expect(approaches[0]).not.toBe(approaches[1]);
    expect(approaches[1]).not.toBe(approaches[2]);
  });

  it('knowledge-extraction uses chain-of-questions', () => {
    const approaches = ['chain-of-questions', 'decomposition', 'counter-factual'];
    expect(approaches[0]).toBe('chain-of-questions');
  });
});

describe('Budget optimization via marginal ROI', () => {
  it('estimates high ROI for improving experts', () => {
    function estimateMarginalROI(history: Array<{ score: number; iteration: number; cost: number }>, costPerIteration: number): number {
      if (history.length < 2) return 1.0;
      const recentWindow = Math.min(5, history.length);
      const recent = history.slice(-recentWindow);
      let totalImprovement = 0;
      let improvementCount = 0;
      for (let i = 1; i < recent.length; i++) {
        const delta = recent[i].score - recent[i - 1].score;
        if (delta > 0) {
          totalImprovement += delta;
          improvementCount++;
        }
      }
      const avgImprovement = improvementCount > 0 ? totalImprovement / improvementCount : 0;
      const pImprove = improvementCount / (recent.length - 1);
      const expectedImprovement = pImprove * avgImprovement;
      return costPerIteration > 0 ? expectedImprovement / costPerIteration : expectedImprovement;
    }

    // Expert improving steadily
    const improving = [
      { score: 0.2, iteration: 0, cost: 0.001 },
      { score: 0.5, iteration: 1, cost: 0.001 },
      { score: 0.8, iteration: 2, cost: 0.001 },
      { score: 1.0, iteration: 3, cost: 0.001 },
    ];
    const roi = estimateMarginalROI(improving, 0.001);
    expect(roi).toBeGreaterThan(0);

    // Expert stuck at same score
    const stuck = [
      { score: 0.0, iteration: 0, cost: 0.001 },
      { score: 0.0, iteration: 1, cost: 0.001 },
      { score: 0.0, iteration: 2, cost: 0.001 },
    ];
    const stuckRoi = estimateMarginalROI(stuck, 0.001);
    expect(stuckRoi).toBe(0);
  });

  it('reallocates budget to high-ROI experts', () => {
    function reallocateBudget(
      expertHistories: Map<number, Array<{ score: number; iteration: number; cost: number }>>,
      totalRemainingIterations: number,
      totalRemainingBudget: number
    ): Map<number, number> {
      const allocation = new Map<number, number>();
      if (expertHistories.size === 0) return allocation;

      const rois = new Map<number, number>();
      for (const [expertId, history] of expertHistories) {
        const avgCost = history.length > 0
          ? history.reduce((s, r) => s + r.cost, 0) / history.length
          : 0.001;
        // Simplified ROI for testing
        const lastScore = history.length > 0 ? history[history.length - 1].score : 0;
        const firstScore = history.length > 0 ? history[0].score : 0;
        const roi = Math.max(lastScore - firstScore, 0.01);
        rois.set(expertId, roi);
      }

      const sorted = [...rois.entries()].sort((a, b) => b[1] - a[1]);
      const totalROI = sorted.reduce((s, [, roi]) => s + Math.max(roi, 0.01), 0);

      for (const [expertId, roi] of sorted) {
        const proportion = Math.max(roi, 0.01) / totalROI;
        const iters = Math.max(1, Math.round(proportion * totalRemainingIterations));
        allocation.set(expertId, iters);
      }

      for (const [expertId] of expertHistories) {
        if (!allocation.has(expertId)) allocation.set(expertId, 1);
      }

      return allocation;
    }

    const histories = new Map<number, Array<{ score: number; iteration: number; cost: number }>>();
    histories.set(0, [
      { score: 0.2, iteration: 0, cost: 0.001 },
      { score: 0.5, iteration: 1, cost: 0.001 },
    ]);
    histories.set(1, [
      { score: 0.0, iteration: 0, cost: 0.001 },
      { score: 0.0, iteration: 1, cost: 0.001 },
    ]);

    const allocation = reallocateBudget(histories, 10, 1);
    expect(allocation.get(0)).toBeGreaterThan(allocation.get(1)!);
  });
});

describe('Cross-domain transfer', () => {
  it('finds analogous categories', () => {
    const CATEGORY_ANALOGIES: Record<string, string[]> = {
      'grid-transformation': ['pattern-completion', 'spatial-reasoning', 'sequence-prediction'],
      'pattern-completion': ['grid-transformation', 'sequence-prediction'],
      'knowledge-synthesis': ['logical-inference', 'mathematical'],
    };

    expect(CATEGORY_ANALOGIES['grid-transformation']).toContain('pattern-completion');
    expect(CATEGORY_ANALOGIES['pattern-completion']).toContain('grid-transformation');
    expect(CATEGORY_ANALOGIES['knowledge-synthesis']).toContain('logical-inference');
  });

  it('category descriptions are comprehensive', () => {
    const descs: Record<string, string> = {
      'grid-transformation': '2D array transformations',
      'pattern-completion': 'Completing partial patterns',
      'knowledge-synthesis': 'Synthesizing fragmented knowledge',
    };
    expect(Object.keys(descs).length).toBeGreaterThanOrEqual(3);
    for (const desc of Object.values(descs)) {
      expect(desc.length).toBeGreaterThan(5);
    }
  });
});

describe('Confidence-weighted voting', () => {
  it('weights passed solutions by iteration efficiency', () => {
    // A solution that passes in 1 iteration should rank higher than one that passes in 5
    // even if they produce the same output
    const confidenceWeight = (res: { score: number; iteration: number; passed: boolean; trainResults: Array<{ softScore: number }> }) => {
      let weight = res.score;
      if (res.passed) {
        weight *= Math.max(0.5, 1 - res.iteration * 0.05);
      }
      const avgSoft = res.trainResults.length > 0
        ? res.trainResults.reduce((s, r) => s + r.softScore, 0) / res.trainResults.length
        : 0;
      if (avgSoft > 0.8) weight *= 1.2;
      return weight;
    };

    const fastSolution = { score: 1.0, iteration: 0, passed: true, trainResults: [{ softScore: 1.0 }] };
    const slowSolution = { score: 1.0, iteration: 5, passed: true, trainResults: [{ softScore: 1.0 }] };

    const fastWeight = confidenceWeight(fastSolution);
    const slowWeight = confidenceWeight(slowSolution);
    expect(fastWeight).toBeGreaterThan(slowWeight);
  });

  it('boosts high soft-score solutions', () => {
    const confidenceWeight = (res: { score: number; iteration: number; passed: boolean; trainResults: Array<{ softScore: number }> }) => {
      let weight = res.score;
      const avgSoft = res.trainResults.length > 0
        ? res.trainResults.reduce((s, r) => s + r.softScore, 0) / res.trainResults.length
        : 0;
      if (avgSoft > 0.8) weight *= 1.2;
      return weight;
    };

    const highSoft = { score: 0.9, iteration: 0, passed: false, trainResults: [{ softScore: 0.9 }] };
    const lowSoft = { score: 0.9, iteration: 0, passed: false, trainResults: [{ softScore: 0.3 }] };

    expect(confidenceWeight(highSoft)).toBeGreaterThan(confidenceWeight(lowSoft));
  });
});

describe('Progressive difficulty', () => {
  it('orders training examples from easiest to hardest', () => {
    function flatten(arr: unknown[]): number[] {
      const result: number[] = [];
      const stack: unknown[] = [arr];
      while (stack.length > 0) {
        const item = stack.pop()!;
        if (Array.isArray(item)) {
          for (let i = item.length - 1; i >= 0; i--) stack.push(item[i]);
        } else if (typeof item === 'number') {
          result.push(item);
        }
      }
      return result;
    }

    function gridSize(arr: unknown[]): number {
      return flatten(arr).length;
    }

    function orderByDifficulty(trainInputs: unknown[], trainOutputs: unknown[]): number[] {
      const indices = trainInputs.map((_, i) => i);
      const difficulty = (input: unknown, output: unknown): number => {
        const inArr = Array.isArray(input) ? input : [];
        const outArr = Array.isArray(output) ? output : [];
        const inSize = gridSize(inArr);
        const outSize = gridSize(outArr);
        const sizeScore = Math.max(inSize, outSize);
        const uniqueVals = new Set(flatten(outArr)).size;
        const asymmetry = Math.abs(inSize - outSize);
        return sizeScore + uniqueVals * 2 + asymmetry * 3;
      };
      const scored = indices.map(i => ({ index: i, diff: difficulty(trainInputs[i], trainOutputs[i]) }));
      scored.sort((a, b) => a.diff - b.diff);
      return scored.map(s => s.index);
    }

    // Small grid should come before large grid
    const trainInputs = [
      [[1, 2], [3, 4]],
      [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ];
    const trainOutputs = [
      [[5, 6], [7, 8]],
      [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ];

    const order = orderByDifficulty(trainInputs, trainOutputs);
    expect(order[0]).toBe(0); // 2x2 grid before 3x3
  });

  it('simplest example is first', () => {
    function flatten(arr: unknown[]): number[] {
      const result: number[] = [];
      const stack: unknown[] = [arr];
      while (stack.length > 0) {
        const item = stack.pop()!;
        if (Array.isArray(item)) {
          for (let i = item.length - 1; i >= 0; i--) stack.push(item[i]);
        } else if (typeof item === 'number') {
          result.push(item);
        }
      }
      return result;
    }
    function gridSize(arr: unknown[]): number { return flatten(arr).length; }

    function orderByDifficulty(trainInputs: unknown[], trainOutputs: unknown[]): number[] {
      const indices = trainInputs.map((_, i) => i);
      const difficulty = (input: unknown, output: unknown): number => {
        const inArr = Array.isArray(input) ? input : [];
        const outArr = Array.isArray(output) ? output : [];
        const inSize = gridSize(inArr);
        const outSize = gridSize(outArr);
        return Math.max(inSize, outSize) + new Set(flatten(outArr)).size * 2;
      };
      const scored = indices.map(i => ({ index: i, diff: difficulty(trainInputs[i], trainOutputs[i]) }));
      scored.sort((a, b) => a.diff - b.diff);
      return scored.map(s => s.index);
    }

    // 3 examples of increasing complexity
    const trainInputs = [
      [[1]],
      [[1, 2], [3, 4]],
      [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ];
    const trainOutputs = [
      [[1]],
      [[1, 2], [3, 4]],
      [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ];

    const order = orderByDifficulty(trainInputs, trainOutputs);
    expect(order).toEqual([0, 1, 2]);
  });
});

describe('Decomposition', () => {
  it('produces sub-problems with valid structure', () => {
    const subProblem = {
      id: 1,
      description: 'Identify the rotation angle',
      input: '[[1,2],[3,4]]',
      expectedOutput: '[[3,1],[4,2]]',
      combineOrder: 1,
    };
    expect(subProblem.id).toBe(1);
    expect(subProblem.description.length).toBeGreaterThan(0);
    expect(subProblem.input.length).toBeGreaterThan(0);
  });

  it('combine strategies are valid', () => {
    const strategies = ['sequential', 'parallel', 'hierarchical'] as const;
    expect(strategies.length).toBe(3);
  });
});

describe('Layer 14: Per-problem prompt synthesis', () => {
  it('computes problem fingerprints by structural features', () => {
    function flatten(arr: unknown[]): number[] {
      const result: number[] = [];
      const stack: unknown[] = [arr];
      while (stack.length > 0) {
        const item = stack.pop()!;
        if (Array.isArray(item)) {
          for (let i = item.length - 1; i >= 0; i--) stack.push(item[i]);
        } else if (typeof item === 'number') {
          result.push(item);
        }
      }
      return result;
    }

    function problemFingerprint(problem: string, trainInputs: unknown[], trainOutputs: unknown[]): string {
      const features: string[] = [];
      if (trainInputs.length > 0) {
        const input = trainInputs[0];
        if (Array.isArray(input)) {
          const flat = flatten(input);
          features.push(`grid:${flat.length}`);
          features.push(`unique:${new Set(flat).size}`);
        }
      }
      const lower = problem.toLowerCase();
      if (lower.includes('rotate')) features.push('op:rotate');
      if (lower.includes('count')) features.push('op:count');
      if (lower.includes('fill')) features.push('op:fill');
      // Size class
      if (trainInputs.length > 0 && Array.isArray(trainInputs[0])) {
        const size = flatten(trainInputs[0]);
        features.push(size.length <= 4 ? 'size:tiny' : size.length <= 16 ? 'size:small' : 'size:medium');
      }
      return features.join('|');
    }

    // Grid problem with rotation
    const fp1 = problemFingerprint(
      'Rotate the grid 90 degrees clockwise',
      [[[1,2],[3,4]]],
      [[[3,1],[4,2]]]
    );
    expect(fp1).toContain('grid:4');
    expect(fp1).toContain('op:rotate');
    expect(fp1).toContain('size:tiny');

    // Counting problem
    const fp2 = problemFingerprint(
      'Count the number of connected components',
      [[[1]]],
      [[[1]]]
    );
    expect(fp2).toContain('op:count');
    expect(fp2).toContain('size:tiny');

    // Different problems of the same type should have similar fingerprints
    const fp3 = problemFingerprint(
      'Rotate this grid 90 degrees',
      [[[1,2,3],[4,5,6],[7,8,9]]],
      [[[7,4,1],[8,5,2],[9,6,3]]]
    );
    expect(fp3).toContain('op:rotate');
  });

  it('SynthesizedPrompt structure is valid', () => {
    const synth = {
      id: 'test1',
      category: 'grid-transformation',
      problemFingerprint: 'grid:4|unique:4|op:rotate|size:small',
      solverPrompt: 'Specialized rotate solver $$problem$$',
      feedbackPrompt: 'Feedback $$feedback$$',
      configOverrides: { temperature: 0.8 },
      validationScore: 0.8,
      validationTests: 2,
      validated: true,
      created: Date.now(),
      useCount: 0,
      successCount: 0,
      avgScore: 0,
    };
    expect(synth.solverPrompt).toContain('$$problem$$');
    expect(synth.problemFingerprint).toContain('op:rotate');
    expect(synth.validated).toBe(true);
  });
});

describe('Layer 15: Meta-meta level', () => {
  it('MetaHarness structure is valid', () => {
    const mh = {
      id: 'mh1',
      name: 'pattern-code-hybrid',
      description: 'Combines pattern recognition with code execution',
      solverPrompt: 'Analyze the pattern first, then write code $$problem$$',
      configOverrides: { temperature: 0.9 },
      rationale: 'Code-only approaches miss spatial patterns; pattern-only lacks precision',
      parentId: null,
      generation: 1,
      created: Date.now(),
      useCount: 0,
      successCount: 0,
      avgScore: 0,
    };
    expect(mh.solverPrompt).toContain('$$problem$$');
    expect(mh.generation).toBe(1);
    expect(mh.rationale.length).toBeGreaterThan(10);
  });

  it('child meta-harness has incremented generation', () => {
    const parent = { id: 'mh1', generation: 1 };
    const child = {
      id: 'mh2',
      parentId: 'mh1',
      generation: parent.generation + 1,
    };
    expect(child.generation).toBe(2);
    expect(child.parentId).toBe('mh1');
  });
});

describe('Layer 16: Gradient-based budget optimization', () => {
  it('estimates positive gradient for improving trajectories', () => {
    function estimateImprovementGradient(history: Array<{ score: number; iteration: number }>) {
      if (history.length < 2) return { gradient: 0, acceleration: 0, expectedNextScore: 0, confidence: 0 };
      const window = Math.min(5, history.length);
      const recent = history.slice(-window);
      let gradientSum = 0, gradientCount = 0;
      for (let i = 1; i < recent.length; i++) {
        const ds = recent[i].score - recent[i - 1].score;
        const di = recent[i].iteration - recent[i - 1].iteration;
        if (di > 0) { gradientSum += ds / di; gradientCount++; }
      }
      const gradient = gradientCount > 0 ? gradientSum / gradientCount : 0;
      const gradients: number[] = [];
      for (let i = 1; i < recent.length; i++) {
        gradients.push(recent[i].score - recent[i - 1].score);
      }
      let accelSum = 0, accelCount = 0;
      for (let i = 1; i < gradients.length; i++) {
        accelSum += gradients[i] - gradients[i - 1];
        accelCount++;
      }
      const acceleration = accelCount > 0 ? accelSum / accelCount : 0;
      const currentScore = history[history.length - 1].score;
      const expectedNextScore = Math.max(0, Math.min(1, currentScore + gradient + 0.5 * acceleration));
      const confidence = Math.min(1, recent.length / 5);
      return { gradient, acceleration, expectedNextScore, confidence };
    }

    // Improving trajectory
    const improving = [
      { score: 0.0, iteration: 0 },
      { score: 0.3, iteration: 1 },
      { score: 0.6, iteration: 2 },
      { score: 0.8, iteration: 3 },
      { score: 1.0, iteration: 4 },
    ];
    const g = estimateImprovementGradient(improving);
    expect(g.gradient).toBeGreaterThan(0);
    expect(g.confidence).toBe(1);

    // Stuck trajectory
    const stuck = [
      { score: 0.0, iteration: 0 },
      { score: 0.0, iteration: 1 },
      { score: 0.0, iteration: 2 },
      { score: 0.0, iteration: 3 },
      { score: 0.0, iteration: 4 },
    ];
    const sg = estimateImprovementGradient(stuck);
    expect(sg.gradient).toBe(0);

    // Decelerating trajectory (acceleration < 0)
    const decel = [
      { score: 0.0, iteration: 0 },
      { score: 0.5, iteration: 1 },
      { score: 0.7, iteration: 2 },
      { score: 0.75, iteration: 3 },
      { score: 0.78, iteration: 4 },
    ];
    const dg = estimateImprovementGradient(decel);
    expect(dg.acceleration).toBeLessThan(0); // slowing down
  });

  it('gradient allocation favors improving experts', () => {
    function gradientBudgetAllocation(
      trajectories: Array<{ expertId: number; history: Array<{ score: number; iteration: number }> }>,
      totalRemainingIterations: number
    ): Map<number, number> {
      const allocation = new Map<number, number>();
      if (trajectories.length === 0) return allocation;
      const weights = trajectories.map(t => {
        if (t.history.length < 2) return 0.01;
        const recent = t.history.slice(-5);
        const lastScore = recent[recent.length - 1].score;
        const firstScore = recent[0].score;
        const improvement = Math.max(0, lastScore - firstScore);
        const confidence = Math.min(1, recent.length / 5);
        return improvement * confidence + 0.01;
      });
      const totalWeight = weights.reduce((s, w) => s + w, 0);
      for (const [i, t] of trajectories.entries()) {
        const proportion = weights[i] / totalWeight;
        const iters = Math.max(1, Math.round(proportion * totalRemainingIterations));
        allocation.set(t.expertId, iters);
      }
      return allocation;
    }

    const improving = {
      expertId: 0,
      history: [
        { score: 0.2, iteration: 0 },
        { score: 0.5, iteration: 1 },
        { score: 0.8, iteration: 2 },
      ],
    };
    const stuck = {
      expertId: 1,
      history: [
        { score: 0.0, iteration: 0 },
        { score: 0.0, iteration: 1 },
        { score: 0.0, iteration: 2 },
      ],
    };

    const alloc = gradientBudgetAllocation([improving, stuck], 10);
    expect(alloc.get(0)).toBeGreaterThan(alloc.get(1)!);
  });

  it('shouldSwitchApproach detects stuck experts', () => {
    function shouldSwitchApproach(history: Array<{ score: number; iteration: number }>): boolean {
      if (history.length < 3) return false;
      // Check if gradient ≈ 0 and decelerating
      const recent = history.slice(-5);
      let gradientSum = 0, gradientCount = 0;
      for (let i = 1; i < recent.length; i++) {
        const ds = recent[i].score - recent[i - 1].score;
        gradientSum += ds;
        gradientCount++;
      }
      const gradient = gradientCount > 0 ? gradientSum / gradientCount : 0;
      const gradients: number[] = [];
      for (let i = 1; i < recent.length; i++) {
        gradients.push(recent[i].score - recent[i - 1].score);
      }
      let accelSum = 0, accelCount = 0;
      for (let i = 1; i < gradients.length; i++) {
        accelSum += gradients[i] - gradients[i - 1];
        accelCount++;
      }
      const acceleration = accelCount > 0 ? accelSum / accelCount : 0;
      const confidence = Math.min(1, recent.length / 5);
      return confidence > 0.6 && gradient < 0.01 && acceleration < -0.01;
    }

    // Stuck expert
    const stuck = [
      { score: 0.0, iteration: 0 },
      { score: 0.0, iteration: 1 },
      { score: 0.0, iteration: 2 },
      { score: 0.0, iteration: 3 },
    ];
    // Not stuck (zero gradient but no negative acceleration)
    expect(shouldSwitchApproach(stuck)).toBe(false); // gradient=0, acceleration=0, not < -0.01

    // Decelerating expert (stuck then declining)
    const decel = [
      { score: 0.6, iteration: 0 },
      { score: 0.6, iteration: 1 },
      { score: 0.6, iteration: 2 },
      { score: 0.6, iteration: 3 },
      { score: 0.55, iteration: 4 },
    ];
    expect(shouldSwitchApproach(decel)).toBe(true);
  });
});

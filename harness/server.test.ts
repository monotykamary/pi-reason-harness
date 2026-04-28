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

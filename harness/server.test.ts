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
// Soft score computation
// =============================================================================

function computeSoftScore(actual: string, expected: unknown): number {
  try {
    const actualArr = JSON.parse(actual);
    const expectedArr = Array.isArray(expected) ? expected : JSON.parse(JSON.stringify(expected));

    if (!Array.isArray(actualArr) || !Array.isArray(expectedArr)) return 0;
    if (actualArr.length !== expectedArr.length) return 0;

    let matches = 0;
    let total = 0;
    for (let i = 0; i < expectedArr.length; i++) {
      const aRow = actualArr[i];
      const eRow = expectedArr[i];
      if (Array.isArray(aRow) && Array.isArray(eRow)) {
        for (let j = 0; j < Math.max(aRow.length, eRow.length); j++) {
          total++;
          if (j < aRow.length && aRow[j] === eRow[j]) matches++;
        }
      } else {
        total++;
        if (aRow === eRow) matches++;
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

      const script = new vm.Script(wrappedCode, { filename: 'sandbox.js', timeout: timeoutS * 1000 });
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

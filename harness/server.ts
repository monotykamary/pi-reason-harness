/**
 * Pi Reason Harness — Server
 *
 * Long-lived HTTP server holding reasoning session state.
 * Implements iterative solve-verify-feedback loops with multi-expert
 * ensembling and voting. The core insight: LLMs contain fragmented
 * knowledge that requires intelligent probing strategies to surface
 * and reassemble.
 *
 * Architecture:
 *   1. Meta-system: discovers task-specific reasoning strategies
 *      (prompt templates, iteration limits, feedback formats)
 *   2. Experts: parallel solve loops with code sandbox + verification
 *   3. Voting: group-by-output ranking with diversity-first ordering
 *
 * JS-exclusive — LLM calls via @mariozechner/pi-ai, sandbox via Node vm module.
 * Zero Python dependency.
 *
 * Endpoints (bind 127.0.0.1:9880 by default; override with $PI_REASON_HARNESS_PORT):
 *   POST /action   body = JSON { action, ...params }
 *                  Headers: x-session-id
 *                  Response: { ok: true, result: { text, details } } | { ok: false, error: string }
 *   GET  /health   { ok: true, uptime, sessions }
 *   POST /quit     graceful shutdown
 */

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { randomBytes } from 'node:crypto';
import { spawn } from 'node:child_process';
import * as fs from 'node:fs';
import * as vm from 'node:vm';

// =============================================================================
// Pi-ai LLM access
// =============================================================================

import {
  getModel,
  getEnvApiKey,
  completeSimple,
  type Model,
  type Context,
  type UserMessage,
  type SimpleStreamOptions,
} from '@mariozechner/pi-ai';

/** Resolve a model from provider/id notation (e.g. "anthropic/claude-sonnet-4-5") */
function resolveModel(modelId: string): Model<any> | null {
  const slashIdx = modelId.indexOf('/');
  if (slashIdx === -1) return null;
  const provider = modelId.slice(0, slashIdx);
  const id = modelId.slice(slashIdx + 1);
  return getModel(provider as any, id) ?? null;
}

/** Get API key for a provider (checks env vars) */
function getApiKey(provider: string): string | undefined {
  return getEnvApiKey(provider as any);
}

// =============================================================================
// Types
// =============================================================================

export interface ExpertConfig {
  /** Prompt template for the solver — use $$problem$$ as placeholder */
  solverPrompt: string;
  /** Prompt template for feedback injection — use $$feedback$$ as placeholder */
  feedbackPrompt: string;
  /** LLM model identifier (provider/model format for pi-ai) */
  llmId: string;
  /** Temperature for LLM sampling */
  temperature: number;
  /** Max iterations per expert before giving up */
  maxIterations: number;
  /** Max past solutions to include in feedback */
  maxSolutions: number;
  /** Probability of including each past solution in feedback [0-1] */
  selectionProbability: number;
  /** Base seed for reproducibility */
  seed: number;
  /** Shuffle training examples each iteration */
  shuffleExamples: boolean;
  /** Show past solutions in improving order (worst→best) */
  improvingOrder: boolean;
  /** Return best-seen result if no perfect solution found */
  returnBestResult: boolean;
  /** Request timeout in seconds */
  requestTimeout: number;
  /** Max total timeouts before giving up */
  maxTotalTimeouts: number;
  /** Per-iteration retries on transient errors */
  perIterationRetries: number;
  /** Code sandbox timeout in seconds */
  sandboxTimeout: number;
  /** Whether code execution is required (vs direct answer) */
  requiresCode: boolean;
  /** Reasoning level for thinking models */
  reasoning: 'off' | 'minimal' | 'low' | 'medium' | 'high';
  /** Voting config */
  voting: {
    useNewVoting: boolean;
    countFailedMatches: boolean;
    itersTiebreak: boolean;
    lowToHighIters: boolean;
  };
}

export interface TaskConfig {
  /** Human-readable name */
  name: string;
  /** Task type determines the strategy template */
  type: 'code-reasoning' | 'knowledge-extraction' | 'hybrid';
  /** Custom expert configs (overrides auto-generated ones) */
  experts: ExpertConfig[];
  /** Number of parallel experts */
  numExperts: number;
  /** Models to use (first available wins, or ensemble all) */
  models: string[];
  /** Verification method */
  verification: 'sandbox' | 'self-audit' | 'external' | 'none';
  /** External verification command (for 'external' type) */
  verifyCommand?: string;
  /** Max cost per problem in USD */
  maxCostPerProblem?: number;
  /** Max time per problem in seconds */
  maxTimePerProblem?: number;
}

export interface SolveResult {
  success: boolean;
  output: string;
  softScore: number;
  error: string | null;
  code: string;
}

export interface IterationResult {
  iteration: number;
  expertIndex: number;
  code: string;
  answer: string;
  trainResults: SolveResult[];
  testResults: SolveResult[];
  passed: boolean;
  score: number;
  feedback: string;
  promptTokens: number;
  completionTokens: number;
  durationMs: number;
}

export interface ProblemHistory {
  problem: string;
  passed: boolean;
  expertIndex: number;
  iterationCount: number;
  bestScore: number;
  strategyType: string;
  modelsUsed: string[];
  timestamp: number;
}

export interface StrategyAdaptation {
  insight: string;
  taskType: string;
  models: string[];
  evidenceCount: number;
  timestamp: number;
  promptModifier?: string;
}

export interface SessionState {
  id: string;
  taskConfig: TaskConfig | null;
  iterations: IterationResult[];
  bestResult: IterationResult | null;
  totalPromptTokens: number;
  totalCompletionTokens: number;
  totalCost: number;
  startTime: number;
  status: 'idle' | 'solving' | 'complete' | 'error';
  error: string | null;
  solutions: Array<{
    code: string;
    feedback: string;
    score: number;
  }>;
  problemHistory: ProblemHistory[];
  strategyAdaptations: StrategyAdaptation[];
  budget: {
    costUsed: number;
    timeUsed: number;
    problemsSolved: number;
    problemsAttempted: number;
  };
}

// =============================================================================
// Strategy templates — the meta-system's output
// =============================================================================

const CODE_REASONING_SOLVER = `You are a world-class expert in solving problems by writing executable JavaScript code. Your approach is methodical, creative, and highly effective. You produce elegant, efficient, and well-documented solutions.

Your goal is to analyze a set of input-output examples and devise a JavaScript function that accurately transforms any input into its corresponding output. The key is to identify a *single, consistent transformation rule* that generalizes across *all* examples.

Follow this iterative process:

**Part 1: Initial Analysis and Hypothesis Generation**
1. Carefully examine the input and output for each example. Note patterns, symmetries, and relationships.
2. Formulate several candidate transformation rules. Start with simpler rules and gradually increase complexity.
3. Consider: value transformations, element isolation, spatial operations, pattern generation.

**Part 2: Iterative Testing and Refinement**
1. Implement your strongest candidate as a JavaScript function.
2. Test against *all* training examples. A single failure indicates an incorrect rule.
3. Analyze feedback carefully when your code fails.
4. Refine or discard the rule and try again.

**Part 3: Coding Guidelines**
1. Available: all standard JavaScript built-ins. No external libraries — pure JS only.
2. Write modular, clear code with comments.
3. Handle edge cases gracefully.
4. For 2D array (grid) operations, use nested Array methods or indexed loops.

**Part 4: Output**
1. Begin with a concise explanation, then provide code.
2. The code section must be a single markdown code block tagged as javascript.
3. Main function: \`function transform(grid)\` — takes a 2D array, returns a 2D array.
4. Do NOT include any I/O code, test harness, or console.log calls.
5. The function must be pure — no side effects, no reading from stdin/stdout.

$$problem$$`;

const CODE_REASONING_FEEDBACK = `**EXISTING PARTIAL/INCORRECT SOLUTIONS:**

Below are some of the best (but not completely correct) solutions so far, with their code, feedback on training examples, and a numeric score (0 = worst, 1 = best). Study these, understand what went wrong, and produce a new solution that fixes all issues. Follow the output format specified earlier.

$$feedback$$`;

const KNOWLEDGE_EXTRACTION_SOLVER = `You are a world-class expert in answering complex questions that require synthesizing fragmented knowledge. Your approach is methodical and thorough.

Key principles:
1. **Probe deeply.** The information you need is often hidden in fragments. Ask yourself: what pieces of information do I need? How do they connect?
2. **Think step-by-step.** Break the question into sub-questions. Answer each one carefully.
3. **Chain of questions.** For each sub-question, ask a follow-up that narrows down the answer space. Build a hierarchy:
   - Start with broad category identification
   - Then narrow to specific domain
   - Then drill into exact details
4. **Verify internally.** Before giving your final answer, check it against known facts. Cross-reference multiple knowledge sources.
5. **Consider multiple angles.** If one approach doesn't yield a clear answer, try another.
6. **Be precise.** When asked for specific dates, names, or values, give the exact answer — not approximations.
7. **Confidence bucketing.** Assess your confidence level:
   - HIGH: Answer verified from multiple angles
   - MEDIUM: Answer plausible but not fully verified
   - LOW: Answer is speculative, needs more probing
   State your confidence level for each component of your answer.

$$problem$$`;

const KNOWLEDGE_EXTRACTION_FEEDBACK = `**PREVIOUS ATTEMPTS:**

Below are previous attempts at this problem, with feedback on what was correct and incorrect. Learn from these mistakes and produce a better answer.

$$feedback$$`;

const HYBRID_SOLVER = `You are a world-class expert in solving complex problems that require both deep knowledge retrieval and systematic reasoning.

Your approach:
1. **Analyze** the problem to determine what knowledge is needed and what reasoning steps are required.
2. **Retrieve** relevant knowledge from your training data — be precise about facts, dates, and relationships.
3. **Chain of questions.** Break the problem into sub-questions, then drill into each one systematically.
4. **Reason** step-by-step, using the retrieved knowledge to build toward the answer.
5. **Verify** your reasoning chain. Check each step for logical consistency.
6. **Synthesize** the pieces into a final, confident answer.
7. **Confidence bucketing.** State HIGH/MEDIUM/LOW confidence for each component.

When code would help (e.g., for computation, data transformation, or systematic checking), write JavaScript code in a markdown code block tagged as javascript. When direct reasoning suffices, explain your thinking clearly.

$$problem$$`;

const HYBRID_FEEDBACK = `**PREVIOUS ATTEMPTS:**

Below are previous attempts with feedback. Each shows what was tried, what worked, and what failed. Learn from these and produce a better solution.

$$feedback$$`;

/**
 * Auto-generate expert configs from a task config.
 * This IS the meta-system — it adapts strategy to task type.
 * Strategy adaptations from past problems modify the prompts.
 */
function generateExpertConfigs(taskConfig: TaskConfig, adaptations: StrategyAdaptation[] = []): ExpertConfig[] {
  const basePrompt =
    taskConfig.type === 'code-reasoning'
      ? CODE_REASONING_SOLVER
      : taskConfig.type === 'knowledge-extraction'
        ? KNOWLEDGE_EXTRACTION_SOLVER
        : HYBRID_SOLVER;

  const baseFeedback =
    taskConfig.type === 'code-reasoning'
      ? CODE_REASONING_FEEDBACK
      : taskConfig.type === 'knowledge-extraction'
        ? KNOWLEDGE_EXTRACTION_FEEDBACK
        : HYBRID_FEEDBACK;

  const requiresCode =
    taskConfig.type === 'code-reasoning' ||
    (taskConfig.type === 'hybrid' && taskConfig.verification === 'sandbox');

  // Apply learned strategy adaptations to prompts
  let promptModifier = '';
  for (const adaptation of adaptations) {
    if (adaptation.taskType === taskConfig.type || adaptation.taskType === '*') {
      promptModifier += '\n\n' + adaptation.promptModifier;
    }
  }

  const solverPrompt = promptModifier ? basePrompt + promptModifier : basePrompt;

  if (taskConfig.experts.length > 0) return taskConfig.experts;

  const experts: ExpertConfig[] = [];
  const numExperts = taskConfig.numExperts || 1;

  for (let i = 0; i < numExperts; i++) {
    const model = taskConfig.models[i % taskConfig.models.length] || 'openai/gpt-4o';
    experts.push({
      solverPrompt,
      feedbackPrompt: baseFeedback,
      llmId: model,
      temperature: 1.0,
      maxIterations: 10,
      maxSolutions: 5,
      selectionProbability: 1.0,
      seed: i * 100,
      shuffleExamples: true,
      improvingOrder: true,
      returnBestResult: true,
      requestTimeout: 3600,
      maxTotalTimeouts: 15,
      perIterationRetries: 2,
      sandboxTimeout: requiresCode ? 5 : 0,
      requiresCode,
      reasoning: 'off',
      voting: {
        useNewVoting: true,
        countFailedMatches: true,
        itersTiebreak: false,
        lowToHighIters: false,
      },
    });
  }

  return experts;
}

// =============================================================================
// Session management
// =============================================================================

const sessions = new Map<string, SessionState>();

function createSession(id?: string): SessionState {
  return {
    id: id || randomBytes(8).toString('hex'),
    taskConfig: null,
    iterations: [],
    bestResult: null,
    totalPromptTokens: 0,
    totalCompletionTokens: 0,
    totalCost: 0,
    startTime: Date.now(),
    status: 'idle',
    error: null,
    solutions: [],
    problemHistory: [],
    strategyAdaptations: [],
    budget: { costUsed: 0, timeUsed: 0, problemsSolved: 0, problemsAttempted: 0 },
  };
}

function getSession(sessionId?: string): SessionState {
  const key = sessionId || 'default';
  let session = sessions.get(key);
  if (!session) {
    session = createSession(sessionId);
    sessions.set(key, session);
  }
  return session;
}

// =============================================================================
// Code sandbox — JS-exclusive via Node vm module
// =============================================================================

interface SandboxResult {
  ok: boolean;
  output: string;
  exitCode: number | null;
  timedOut: boolean;
  durationMs: number;
}

/**
 * Execute user-provided JS code in a sandboxed Node vm context.
 *
 * The code must export/define a `transform` function (or any named function
 * the problem specifies). We run it in an isolated context with a timeout.
 * The input is passed as a global `__input__`, and the output is captured
 * from `__output__`.
 */
async function runInSandbox(
  code: string,
  input: unknown,
  timeoutS: number = 5
): Promise<SandboxResult> {
  const t0 = Date.now();

  try {
    // Create an isolated context
    const context = vm.createContext({
      // Standard JS builtins
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
      // Input data
      __input__: input,
      __output__: null,
      // Allow structuredClone for deep cloning
      structuredClone: typeof structuredClone !== 'undefined' ? structuredClone : undefined,
    });

    // Wrap the user code to capture the transform function's output
    const wrappedCode = `
${code}

// Auto-invoke: if transform is defined, run it on __input__
if (typeof transform === 'function') {
  try {
    __output__ = transform(__input__);
  } catch (e) {
    __output__ = { __error__: e.message || String(e) };
  }
} else if (typeof solve === 'function') {
  try {
    __output__ = solve(__input__);
  } catch (e) {
    __output__ = { __error__: e.message || String(e) };
  }
}
`;

    const script = new vm.Script(wrappedCode, {
      filename: 'sandbox.js',
    });

    script.runInContext(context, {
      timeout: timeoutS * 1000,
    });

    const result = context.__output__;

    // Check for execution errors
    if (result && typeof result === 'object' && result.__error__) {
      return {
        ok: false,
        output: result.__error__,
        exitCode: 1,
        timedOut: false,
        durationMs: Date.now() - t0,
      };
    }

    return {
      ok: true,
      output: JSON.stringify(result),
      exitCode: 0,
      timedOut: false,
      durationMs: Date.now() - t0,
    };
  } catch (e: any) {
    const isTimeout = e.code === 'ERR_SCRIPT_EXECUTION_TIMEOUT' ||
      (e.message && e.message.includes('timeout'));

    return {
      ok: false,
      output: isTimeout ? 'timeout' : (e.message || String(e)).slice(0, 500),
      exitCode: isTimeout ? null : 1,
      timedOut: isTimeout,
      durationMs: Date.now() - t0,
    };
  }
}

// =============================================================================
// External verification command
// =============================================================================

async function runExternalVerify(
  code: string,
  input: unknown,
  expectedOutput: unknown,
  command: string,
  timeoutS: number = 10
): Promise<SandboxResult> {
  const tmpDir = join(tmpdir(), `pi-reason-verify-${randomBytes(4).toString('hex')}`);
  fs.mkdirSync(tmpDir, { recursive: true });
  const codePath = join(tmpDir, 'solution.js');
  const inputPath = join(tmpDir, 'input.json');
  const expectedPath = join(tmpDir, 'expected.json');

  try {
    fs.writeFileSync(codePath, code, 'utf-8');
    fs.writeFileSync(inputPath, JSON.stringify(input), 'utf-8');
    fs.writeFileSync(expectedPath, JSON.stringify(expectedOutput), 'utf-8');

    const fullCommand = command
      .replace('{code}', codePath)
      .replace('{input}', inputPath)
      .replace('{expected}', expectedPath);

    const t0 = Date.now();
    const child = spawn('bash', ['-c', fullCommand], {
      cwd: tmpDir,
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    const chunks: Buffer[] = [];
    const errChunks: Buffer[] = [];
    child.stdout.on('data', (c: Buffer) => chunks.push(c));
    child.stderr.on('data', (c: Buffer) => errChunks.push(c));

    let timedOut = false;
    const timeoutHandle = setTimeout(() => {
      timedOut = true;
      try { child.kill('SIGKILL'); } catch {}
    }, timeoutS * 1000);

    const exitCode = await new Promise<number | null>((resolve) => {
      child.on('close', (code) => {
        clearTimeout(timeoutHandle);
        resolve(code);
      });
      child.on('error', () => {
        clearTimeout(timeoutHandle);
        resolve(1);
      });
    });

    const stdout = Buffer.concat(chunks).toString('utf-8');
    const durationMs = Date.now() - t0;

    return {
      ok: exitCode === 0 && !timedOut,
      output: stdout.trim().slice(0, 500),
      exitCode,
      timedOut,
      durationMs,
    };
  } finally {
    try { fs.rmSync(tmpDir, { recursive: true, force: true }); } catch {}
  }
}

// =============================================================================
// LLM caller — JS-exclusive via @mariozechner/pi-ai
// =============================================================================

interface LLMResponse {
  content: string;
  promptTokens: number;
  completionTokens: number;
  durationMs: number;
  cost: number;
}

async function callLLM(
  modelId: string,
  message: string,
  temperature: number = 1.0,
  timeoutS: number = 3600,
  retries: number = 2,
  reasoning: 'off' | 'minimal' | 'low' | 'medium' | 'high' = 'off'
): Promise<LLMResponse> {
  const model = resolveModel(modelId);
  if (!model) {
    return {
      content: `[error: unknown model "${modelId}"]`,
      promptTokens: 0,
      completionTokens: 0,
      durationMs: 0,
      cost: 0,
    };
  }

  const apiKey = getApiKey(model.provider);
  if (!apiKey) {
    return {
      content: `[error: no API key for provider "${model.provider}"]`,
      promptTokens: 0,
      completionTokens: 0,
      durationMs: 0,
      cost: 0,
    };
  }

  const context: Context = {
    messages: [
      {
        role: 'user',
        content: message,
        timestamp: Date.now(),
      } as UserMessage,
    ],
  };

  const options: SimpleStreamOptions = {
    temperature,
    apiKey,
    timeoutMs: timeoutS * 1000,
    maxRetries: retries,
    ...(reasoning !== 'off' ? { reasoning } : {}),
  };

  const t0 = Date.now();

  try {
    const assistantMsg = await completeSimple(model, context, options);
    const usage = assistantMsg.usage;
    const textParts = assistantMsg.content
      .filter((c: any) => c.type === 'text')
      .map((c: any) => c.text);
    const content = textParts.join('\n');

    let cost = 0;
    if (usage) {
      const inputCost = (model.cost.input / 1_000_000) * (usage.input || 0);
      const outputCost = (model.cost.output / 1_000_000) * (usage.output || 0);
      const cacheReadCost = (model.cost.cacheRead / 1_000_000) * (usage.cacheRead || 0);
      const cacheWriteCost = (model.cost.cacheWrite / 1_000_000) * (usage.cacheWrite || 0);
      cost = inputCost + outputCost + cacheReadCost + cacheWriteCost;
    }

    return {
      content,
      promptTokens: usage?.input || 0,
      completionTokens: usage?.output || 0,
      durationMs: Date.now() - t0,
      cost,
    };
  } catch (e) {
    return {
      content: `[error: ${e instanceof Error ? e.message : String(e)}]`,
      promptTokens: 0,
      completionTokens: 0,
      durationMs: Date.now() - t0,
      cost: 0,
    };
  }
}

// =============================================================================
// Code parsing
// =============================================================================

function parseCodeFromLLM(response: string): string | null {
  // Try javascript, js, typescript, ts code blocks
  const m = response.match(/```(?:javascript|js|typescript|ts)\s*(.*?)```/s);
  return m ? m[1].trim() : null;
}

function parseAnswerFromLLM(response: string): string {
  const answerMatch = response.match(/\*\*Answer:\*\*\s*(.*?)(?:\n\n|\n\*\*|$)/s);
  if (answerMatch) return answerMatch[1].trim();

  const confMatch = response.match(/^(.*?)\n\s*(?:Confidence|CONFIDENCE|confidence)/s);
  if (confMatch) return confMatch[1].trim();

  const codeIdx = response.indexOf('```');
  if (codeIdx > 0) return response.slice(0, codeIdx).trim();

  return response.trim();
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
<solution_code>
\`\`\`javascript
${s.code}
\`\`\`
</solution_code>
<solution_evaluation>
${s.feedback}
</solution_evaluation>
<solution_score>
${s.score.toFixed(2)}
</solution_score>
</solution_${i + 1}>`
    )
    .join('\n\n');
}

// =============================================================================
// Self-audit verification
// =============================================================================

interface AuditResult {
  verified: boolean;
  confidence: 'HIGH' | 'MEDIUM' | 'LOW';
  issues: string[];
}

async function selfAudit(
  modelId: string,
  answer: string,
  problem: string,
  temperature: number = 0.3,
  timeoutS: number = 120
): Promise<AuditResult> {
  const auditPrompt = `You are a rigorous fact-checker. Evaluate the following answer to a question.

**Question:** ${problem}

**Proposed Answer:** ${answer}

Check for:
1. Factual accuracy — are specific claims correct?
2. Logical consistency — does the reasoning hold?
3. Completeness — are all parts of the question answered?
4. Precision — are specific values/names/dates exact?

Respond in this exact format:
VERIFIED: yes|no
CONFIDENCE: HIGH|MEDIUM|LOW
ISSUES: [comma-separated list of problems found, or "none"]`;

  const result = await callLLM(modelId, auditPrompt, temperature, timeoutS, 1, 'high');

  const verified = /VERIFIED:\s*yes/i.test(result.content);
  const confMatch = result.content.match(/CONFIDENCE:\s*(HIGH|MEDIUM|LOW)/i);
  const confidence = (confMatch?.[1]?.toUpperCase() as AuditResult['confidence']) || 'LOW';
  const issuesMatch = result.content.match(/ISSUES:\s*(.*?)(?:\n|$)/i);
  const issuesStr = issuesMatch?.[1]?.trim() || 'none';
  const issues = issuesStr.toLowerCase() === 'none' ? [] : issuesStr.split(',').map(s => s.trim()).filter(Boolean);

  return { verified, confidence, issues };
}

// =============================================================================
// Problem formatting — mirrors Poetiq's format_problem()
// Converts raw grid data into <Diagram> text with optional per-iteration shuffle
// =============================================================================

/**
 * Format a problem from raw grid arrays into <Diagram> text.
 * Mirrors Poetiq's format_problem + _example_to_diagram.
 *
 * @param trainIn  Training input grids (3D: [example][row][col])
 * @param trainOut Training output grids
 * @param testIn   Test input grids
 * @param shuffle  Whether to shuffle training example order
 * @param seed     Seed for the shuffle RNG
 */
function formatProblem(
  trainIn: number[][][],
  trainOut: number[][][],
  testIn: number[][][],
  shuffle: boolean = true,
  seed: number = 0
): string {
  // Build index array for shuffling
  const indices = trainIn.map((_, i) => i);
  if (shuffle && indices.length > 1) {
    const rng = createRNG(seed);
    // Fisher-Yates shuffle
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
  }

  let exampleStr = '';
  let challengeStr = '';

  for (let e = 0; e < indices.length; e++) {
    const idx = indices[e];
    exampleStr += `
Example #${e + 1}
Input:
<Diagram>
${gridToDiagram(trainIn[idx])}
</Diagram>

Output:
<Diagram>
${gridToDiagram(trainOut[idx])}
</Diagram>
`;
  }

  for (let c = 0; c < testIn.length; c++) {
    challengeStr += `
Challenge #${c + 1}
Input:
<Diagram>
${gridToDiagram(testIn[c])}
</Diagram>
`;
  }

  return exampleStr + challengeStr;
}

/** Convert a 2D grid to an ASCII diagram (space-separated values, one row per line) */
function gridToDiagram(grid: number[][]): string {
  return grid.map(row => row.join(' ')).join('\n');
}

// =============================================================================
// Solve loop — the core iterative refinement engine
// =============================================================================

async function solveWithExpert(
  session: SessionState,
  expertConfig: ExpertConfig,
  expertIndex: number,
  problem: string,
  trainInputs: unknown[],
  trainOutputs: unknown[],
  testInputs: unknown[],
  verification: TaskConfig['verification'],
  verifyCommand: string | undefined,
  budget: { maxCost?: number; maxTime?: number; startTime: number; costSoFar: number },
  signal?: AbortSignal
): Promise<IterationResult[]> {
  const results: IterationResult[] = [];
  const solutions: Array<{ code: string; feedback: string; score: number }> = [];
  const rng = createRNG(expertConfig.seed);

  let bestScore = -1;
  let bestIteration: IterationResult | null = null;
  let totalTimeouts = 0;

  // Pre-format problem from raw grid data if available (mirrors Poetiq's format_problem)
  const hasGridData = trainInputs.length > 0 && trainOutputs.length > 0;

  for (let it = 0; it < expertConfig.maxIterations; it++) {
    if (signal?.aborted) break;

    // Budget checks
    if (budget.maxCost && budget.costSoFar >= budget.maxCost) break;
    if (budget.maxTime && (Date.now() - budget.startTime) / 1000 >= budget.maxTime) break;

    // Format problem with per-iteration shuffle (Poetiq: seed + it)
    let formattedProblem = problem;
    if (hasGridData && !problem.includes('<Diagram>')) {
      formattedProblem = formatProblem(
        trainInputs as number[][][],
        trainOutputs as number[][][],
        testInputs as number[][][],
        expertConfig.shuffleExamples,
        expertConfig.seed + it
      );
    }

    // Build prompt with optional feedback from past solutions
    let message = expertConfig.solverPrompt.replace('$$problem$$', formattedProblem);

    if (solutions.length > 0) {
      const selected = solutions.filter(() => rng() < expertConfig.selectionProbability);
      if (selected.length > 0) {
        const feedbackBlock = buildFeedbackBlock(
          selected,
          expertConfig.maxSolutions,
          expertConfig.improvingOrder
        );
        message += '\n\n' + expertConfig.feedbackPrompt.replace('$$feedback$$', feedbackBlock);
      }
    }

    // Call LLM via pi-ai
    const llmResult = await callLLM(
      expertConfig.llmId,
      message,
      expertConfig.temperature,
      expertConfig.requestTimeout,
      expertConfig.perIterationRetries,
      expertConfig.reasoning
    );

    session.totalPromptTokens += llmResult.promptTokens;
    session.totalCompletionTokens += llmResult.completionTokens;
    session.totalCost += llmResult.cost;
    budget.costSoFar += llmResult.cost;

    // Parse code from response
    const code = expertConfig.requiresCode ? parseCodeFromLLM(llmResult.content) : null;
    const answer = parseAnswerFromLLM(llmResult.content);

    if (expertConfig.requiresCode && !code) {
      results.push({
        iteration: it,
        expertIndex,
        code: '',
        answer: llmResult.content,
        trainResults: [],
        testResults: [],
        passed: false,
        score: 0,
        feedback: 'No code block found in LLM response.',
        promptTokens: llmResult.promptTokens,
        completionTokens: llmResult.completionTokens,
        durationMs: llmResult.durationMs,
      });
      continue;
    }

    // === Verification ===

    if (verification === 'sandbox' && code && trainInputs.length > 0) {
      const trainResults: SolveResult[] = [];
      let totalScore = 0;

      for (let j = 0; j < trainInputs.length; j++) {
        const sandboxResult = await runInSandbox(code, trainInputs[j], expertConfig.sandboxTimeout);
        if (sandboxResult.timedOut) {
          totalTimeouts++;
          if (totalTimeouts > expertConfig.maxTotalTimeouts) break;
        }
        const expected = trainOutputs[j];
        const success = sandboxResult.ok && compareOutputs(sandboxResult.output, expected);
        const softScore = sandboxResult.ok ? computeSoftScore(sandboxResult.output, expected) : 0;
        totalScore += softScore;

        trainResults.push({
          success,
          output: sandboxResult.output,
          softScore,
          error: sandboxResult.ok ? null : sandboxResult.output,
          code,
        });
      }

      if (totalTimeouts > expertConfig.maxTotalTimeouts) {
        results.push({
          iteration: it,
          expertIndex,
          code: code || '',
          answer,
          trainResults,
          testResults: [],
          passed: false,
          score: 0,
          feedback: `Too many timeouts (${totalTimeouts}). Code may have infinite loop or be too slow.`,
          promptTokens: llmResult.promptTokens,
          completionTokens: llmResult.completionTokens,
          durationMs: llmResult.durationMs,
        });
        break;
      }

      const avgScore = trainResults.length > 0 ? totalScore / trainResults.length : 0;
      const passed = trainResults.length > 0 && trainResults.every((r) => r.success);

      // Evaluate on test data (no ground truth available)
      const testResults: SolveResult[] = [];
      for (const testInput of testInputs) {
        const sandboxResult = await runInSandbox(code, testInput, expertConfig.sandboxTimeout);
        testResults.push({
          success: false,
          output: sandboxResult.output,
          softScore: 0,
          error: sandboxResult.ok ? null : sandboxResult.output,
          code,
        });
      }

      const feedback = buildDetailedFeedback(trainResults, trainInputs, trainOutputs);
      solutions.push({ code, feedback, score: avgScore });

      const iteration: IterationResult = {
        iteration: it,
        expertIndex,
        code,
        answer,
        trainResults,
        testResults,
        passed,
        score: avgScore,
        feedback,
        promptTokens: llmResult.promptTokens,
        completionTokens: llmResult.completionTokens,
        durationMs: llmResult.durationMs,
      };

      results.push(iteration);
      if (avgScore > bestScore) {
        bestScore = avgScore;
        bestIteration = iteration;
      }
      if (passed) break;
    } else if (verification === 'external' && code && verifyCommand && trainInputs.length > 0) {
      const trainResults: SolveResult[] = [];
      let totalScore = 0;

      for (let j = 0; j < trainInputs.length; j++) {
        const vResult = await runExternalVerify(
          code,
          trainInputs[j],
          trainOutputs[j],
          verifyCommand,
          expertConfig.sandboxTimeout * 2
        );
        const success = vResult.ok;
        totalScore += success ? 1 : 0;

        trainResults.push({
          success,
          output: vResult.output,
          softScore: success ? 1 : 0,
          error: vResult.ok ? null : vResult.output,
          code,
        });
      }

      const avgScore = trainResults.length > 0 ? totalScore / trainResults.length : 0;
      const passed = trainResults.every((r) => r.success);

      const feedback = buildDetailedFeedback(trainResults, trainInputs, trainOutputs);
      solutions.push({ code, feedback, score: avgScore });

      const testResults: SolveResult[] = [];

      const iteration: IterationResult = {
        iteration: it,
        expertIndex,
        code,
        answer,
        trainResults,
        testResults,
        passed,
        score: avgScore,
        feedback,
        promptTokens: llmResult.promptTokens,
        completionTokens: llmResult.completionTokens,
        durationMs: llmResult.durationMs,
      };

      results.push(iteration);
      if (avgScore > bestScore) {
        bestScore = avgScore;
        bestIteration = iteration;
      }
      if (passed) break;
    } else if (verification === 'self-audit') {
      const audit = await selfAudit(expertConfig.llmId, answer, problem);

      const passed = audit.verified && audit.confidence !== 'LOW';
      const score = audit.verified
        ? audit.confidence === 'HIGH' ? 1.0 : 0.5
        : 0;

      const feedback = audit.issues.length > 0
        ? `Self-audit issues: ${audit.issues.join('; ')}\nConfidence: ${audit.confidence}`
        : `Self-audit passed. Confidence: ${audit.confidence}`;

      solutions.push({ code: code || '', feedback, score });

      const iteration: IterationResult = {
        iteration: it,
        expertIndex,
        code: code || '',
        answer,
        trainResults: [{ success: passed, output: answer, softScore: score, error: null, code: '' }],
        testResults: [],
        passed,
        score,
        feedback,
        promptTokens: llmResult.promptTokens,
        completionTokens: llmResult.completionTokens,
        durationMs: llmResult.durationMs,
      };

      results.push(iteration);
      if (score > bestScore) {
        bestScore = score;
        bestIteration = iteration;
      }
      if (passed && audit.confidence === 'HIGH') break;
    } else {
      // No verification
      const iteration: IterationResult = {
        iteration: it,
        expertIndex,
        code: code || '',
        answer,
        trainResults: [],
        testResults: [],
        passed: false,
        score: 0,
        feedback: 'No verification method configured.',
        promptTokens: llmResult.promptTokens,
        completionTokens: llmResult.completionTokens,
        durationMs: llmResult.durationMs,
      };

      results.push(iteration);
    }
  }

  return results;
}

function createRNG(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

function compareOutputs(actual: string, expected: unknown): boolean {
  try {
    const actualParsed = JSON.parse(actual);
    const expectedParsed = Array.isArray(expected) ? expected : JSON.parse(JSON.stringify(expected));

    // Use 2D normalization for grid comparison
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

    // Shape mismatch → 0 (matches Poetiq's _soft_score)
    if (predRows !== truthRows || predCols !== truthCols) return 0;

    // Empty grid → 1 (trivially correct)
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

/**
 * Build detailed feedback for each training example.
 * Mirrors Poetiq's _build_feedback() — produces element-by-element
 * diff grids, shape mismatch info, and bad-JSON diagnostics.
 */
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

    // Try to parse the model's output as a 2D array
    let predArr: unknown = null;
    try {
      if (rr.output) {
        predArr = JSON.parse(rr.output);
      }
    } catch {}

    const truth = trainOutputs[i];
    const truthArr = Array.isArray(truth) ? truth : null;

    if (!predArr || !Array.isArray(predArr)) {
      // Model output couldn't be parsed as a JSON array
      msgLines.push('\nThe output has to be a rectangular grid of numbers.\n');
      if (rr.error) {
        msgLines.push(`Your code produced the following error:\n${rr.error.slice(0, 300)}\n`);
      }
    } else {
      // Normalize to 2D
      const pred2D = ensure2D(predArr);
      const truth2D = truthArr ? ensure2D(truthArr) : null;

      if (!truth2D || !pred2D) {
        msgLines.push('\nFailed to parse grids for comparison.\n');
      } else {
        const predShape = gridShape(pred2D);
        const truthShape = gridShape(truth2D);

        if (predShape[0] !== truthShape[0] || predShape[1] !== truthShape[1]) {
          // Shape mismatch
          msgLines.push(
            `\n\nShape mismatch: your prediction's shape was [${predShape}], ` +
            `while the correct shape was [${truthShape}].`
          );
        } else {
          // Same shape: show element-by-element diff grid
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

// =============================================================================
// Grid utilities — mirrors Poetiq's numpy-based grid ops in pure JS
// =============================================================================

/** Ensure a value is a 2D array (expand 1D → 2D if needed) */
function ensure2D(arr: unknown): number[][] | null {
  if (!Array.isArray(arr)) return null;
  if (arr.length === 0) return [[]];
  if (Array.isArray(arr[0])) return arr as number[][];
  // 1D array → single row
  return [arr as unknown[] as number[]];
}

/** Get [rows, cols] of a 2D grid */
function gridShape(grid: number[][]): [number, number] {
  return [grid.length, grid.length > 0 ? grid[0].length : 0];
}

/** Element-by-element diff: matching values as-is, mismatches as 'pred/truth' */
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

// =============================================================================
// Voting
// =============================================================================

function rankByVoting(allResults: IterationResult[], config: ExpertConfig): IterationResult[] {
  const { useNewVoting, countFailedMatches, itersTiebreak, lowToHighIters } = config.voting;

  const candidateBuckets = new Map<string, IterationResult[]>();
  const failureBuckets = new Map<string, IterationResult[]>();

  for (const res of allResults) {
    const isPasser = res.trainResults.length > 0 && res.trainResults.every((r) => r.success);
    const key = canonicalKey(res.testResults);

    if (isPasser) {
      (candidateBuckets.get(key) || candidateBuckets.set(key, []).get(key)!).push(res);
    } else {
      (failureBuckets.get(key) || failureBuckets.set(key, []).get(key)!).push(res);
    }
  }

  if (useNewVoting) {
    if (countFailedMatches) {
      for (const [k, failures] of failureBuckets) {
        if (candidateBuckets.has(k)) {
          candidateBuckets.get(k)!.push(...failures);
          failureBuckets.delete(k);
        }
      }
    }

    let passerGroups = [...candidateBuckets.values()].sort((a, b) => b.length - a.length);

    if (itersTiebreak) {
      passerGroups = passerGroups.map((ps) =>
        [...ps].sort((a, b) =>
          lowToHighIters ? a.iteration - b.iteration : b.iteration - a.iteration
        )
      );
    }

    for (const fs of failureBuckets.values()) {
      fs.sort((a, b) => meanSoft(b) - meanSoft(a));
    }
    const failureGroups = [...failureBuckets.values()].sort(
      (a, b) => b.length - a.length || meanSoft(b[0]) - meanSoft(a[0])
    );

    const ordered: IterationResult[] = [];
    ordered.push(...passerGroups.map((g) => g[0]).filter(Boolean));
    ordered.push(...failureGroups.map((g) => g[0]).filter(Boolean));
    ordered.push(...passerGroups.flatMap((g) => g.slice(1)));
    ordered.push(...failureGroups.flatMap((g) => g.slice(1)));

    return ordered;
  }

  const passerGroups = [...candidateBuckets.values()].sort((a, b) => b.length - a.length);
  const firsts = passerGroups.map((g) => g[0]);
  const rest = passerGroups.flatMap((g) => g.slice(1));
  const failed = [...failureBuckets.values()]
    .flat()
    .sort((a, b) => meanSoft(b) - meanSoft(a));

  return [...firsts, ...failed, ...rest];
}

function canonicalKey(testResults: SolveResult[]): string {
  if (testResults.length === 0) return '__no_test__';
  return testResults.map((r) => r.output).join('|');
}

function meanSoft(result: IterationResult): number {
  const trs = result.trainResults;
  if (trs.length === 0) return 0;
  return trs.reduce((sum, r) => sum + r.softScore, 0) / trs.length;
}

// =============================================================================
// Self-improvement
// =============================================================================

function learnFromProblem(session: SessionState): void {
  if (session.iterations.length === 0) return;

  const passed = session.iterations.some((r) => r.passed);
  const config = session.taskConfig;
  if (!config) return;

  const bestResult = session.bestResult;
  const historyEntry: ProblemHistory = {
    problem: '[last problem]',
    passed,
    expertIndex: bestResult?.expertIndex ?? 0,
    iterationCount: session.iterations.length,
    bestScore: bestResult?.score ?? 0,
    strategyType: config.type,
    modelsUsed: config.models,
    timestamp: Date.now(),
  };
  session.problemHistory.push(historyEntry);
  session.budget.problemsAttempted++;
  if (passed) session.budget.problemsSolved++;

  // 1. Model preference
  if (passed && bestResult) {
    const successfulModel = config.models[bestResult.expertIndex % config.models.length];
    const existing = session.strategyAdaptations.find(
      (a) => a.taskType === config.type && a.models.includes(successfulModel)
    );
    if (existing) {
      existing.evidenceCount++;
      existing.timestamp = Date.now();
    } else {
      session.strategyAdaptations.push({
        insight: `Model ${successfulModel} successfully solved ${config.type} problems`,
        taskType: config.type,
        models: [successfulModel],
        evidenceCount: 1,
        timestamp: Date.now(),
        promptModifier: `Note: Model ${successfulModel} has been effective for ${config.type} tasks. Prefer approaches that leverage its strengths.`,
      });
    }
  }

  // 2. Feedback effectiveness
  if (passed && bestResult && bestResult.iteration > 0) {
    const firstScore = session.iterations.find(
      (r) => r.expertIndex === bestResult.expertIndex && r.iteration === 0
    )?.score ?? 0;
    const scoreDelta = bestResult.score - firstScore;

    if (scoreDelta > 0.3) {
      const existing = session.strategyAdaptations.find(
        (a) => a.insight.includes('Feedback-driven improvement')
      );
      if (existing) {
        existing.evidenceCount++;
      } else {
        session.strategyAdaptations.push({
          insight: 'Feedback-driven improvement is effective — past solutions with feedback significantly improve results',
          taskType: '*',
          models: config.models,
          evidenceCount: 1,
          timestamp: Date.now(),
          promptModifier: 'Pay careful attention to feedback from previous attempts. The iterative refinement process is key — analyze what went wrong in past solutions and specifically address those issues.',
        });
      }
    }
  }

  // 3. Partial solution awareness
  const bestFailed = session.iterations
    .filter((r) => !r.passed && r.score > 0.5)
    .sort((a, b) => b.score - a.score)[0];

  if (bestFailed && !passed) {
    const existing = session.strategyAdaptations.find(
      (a) => a.insight.includes('partial solutions are close')
    );
    if (existing) {
      existing.evidenceCount++;
    } else if (session.problemHistory.length >= 2) {
      session.strategyAdaptations.push({
        insight: 'Partial solutions are often close to correct — small adjustments can fix them',
        taskType: config.type,
        models: config.models,
        evidenceCount: 1,
        timestamp: Date.now(),
        promptModifier: 'When a previous solution has a high accuracy score but isn\'t quite right, focus on understanding the specific differences rather than starting from scratch. Often a small fix to the logic resolves the problem.',
      });
    }
  }

  // 4. Performance adaptation
  const timeoutCount = session.iterations.filter(
    (r) => r.feedback.includes('timeout') || r.feedback.includes('Too many timeouts')
  ).length;

  if (timeoutCount > 2) {
    const existing = session.strategyAdaptations.find(
      (a) => a.insight.includes('performance')
    );
    if (existing) {
      existing.evidenceCount++;
    } else {
      session.strategyAdaptations.push({
        insight: 'Frequent timeouts suggest solutions need performance optimization',
        taskType: config.type,
        models: config.models,
        evidenceCount: 1,
        timestamp: Date.now(),
        promptModifier: 'IMPORTANT: Prioritize efficient algorithms. Previous solutions have timed out. Avoid brute-force approaches. Use Array methods and avoid deeply nested loops where possible. Consider time complexity carefully.',
      });
    }
  }
}

// =============================================================================
// Action dispatch
// =============================================================================

async function dispatchAction(
  action: string,
  params: Record<string, unknown>,
  sessionId?: string
): Promise<{ text: string; details: unknown }> {
  const session = getSession(sessionId);

  switch (action) {
    case 'init': {
      const name = params.name as string;
      const type = (params.type as TaskConfig['type']) || 'code-reasoning';
      const models = (params.models as string[]) || ['openai/gpt-4o'];
      const numExperts = (params.numExperts as number) || 1;
      const verification = (params.verification as TaskConfig['verification']) || 'sandbox';
      const verifyCommand = params.verifyCommand as string | undefined;
      const maxCostPerProblem = params.maxCostPerProblem as number | undefined;
      const maxTimePerProblem = params.maxTimePerProblem as number | undefined;
      const customExperts = (params.experts as ExpertConfig[]) || [];

      // Validate models
      const validModels: string[] = [];
      const invalidModels: string[] = [];
      for (const m of models) {
        const resolved = resolveModel(m);
        if (resolved) {
          const key = getApiKey(resolved.provider);
          if (key) {
            validModels.push(m);
          } else {
            invalidModels.push(`${m} (no API key)`);
          }
        } else {
          invalidModels.push(`${m} (unknown model)`);
        }
      }

      if (validModels.length === 0) {
        return {
          text: `❌ No valid models with API keys found.\nChecked: ${models.join(', ')}\nInvalid: ${invalidModels.join(', ')}\n\nSet API keys via environment variables (e.g. ANTHROPIC_API_KEY, OPENAI_API_KEY) or pi auth.`,
          details: { invalidModels },
        };
      }

      const taskConfig: TaskConfig = {
        name,
        type,
        experts: customExperts,
        numExperts,
        models: validModels,
        verification,
        verifyCommand,
        maxCostPerProblem,
        maxTimePerProblem,
      };

      session.taskConfig = taskConfig;
      session.status = 'idle';
      session.iterations = [];
      session.solutions = [];
      session.bestResult = null;

      const expertConfigs = generateExpertConfigs(taskConfig, session.strategyAdaptations);

      const warnings = invalidModels.length > 0
        ? `\n⚠️ Skipped models: ${invalidModels.join(', ')}`
        : '';

      const adaptationsNote = session.strategyAdaptations.length > 0
        ? `\n🧠 Applying ${session.strategyAdaptations.length} strategy adaptation(s) from past problems`
        : '';

      return {
        text: `✅ Reasoning session initialized: "${name}"\nType: ${type}\nExperts: ${expertConfigs.length} (models: ${validModels.join(', ')})\nVerification: ${verification}${warnings}${adaptationsNote}\n\nReady to solve. Call pi-reason-harness solve with a problem.`,
        details: {
          sessionId: session.id,
          taskConfig,
          expertConfigs: expertConfigs.length,
          invalidModels,
        },
      };
    }

    case 'solve': {
      if (!session.taskConfig) {
        return { text: '❌ No session initialized. Call init first.', details: {} };
      }

      const problem = (params.problem as string) || '';
      const trainInputs = (params.trainInputs as unknown[]) || [];
      const trainOutputs = (params.trainOutputs as unknown[]) || [];
      const testInputs = (params.testInputs as unknown[]) || [];

      if (!problem && trainInputs.length === 0) {
        return { text: '❌ solve requires a problem or trainInputs/trainOutputs.', details: {} };
      }

      const maxCost = session.taskConfig.maxCostPerProblem;
      const maxTime = session.taskConfig.maxTimePerProblem;

      session.status = 'solving';
      session.iterations = [];
      session.solutions = [];
      session.bestResult = null;
      session.totalPromptTokens = 0;
      session.totalCompletionTokens = 0;
      session.totalCost = 0;

      const solveStart = Date.now();
      const expertConfigs = generateExpertConfigs(session.taskConfig, session.strategyAdaptations);
      const allResults: IterationResult[] = [];

      const budget = {
        maxCost,
        maxTime,
        startTime: solveStart,
        costSoFar: session.totalCost,
      };

      const expertPromises = expertConfigs.map((cfg, i) =>
        solveWithExpert(
          session,
          cfg,
          i,
          problem,
          trainInputs,
          trainOutputs,
          testInputs,
          session.taskConfig!.verification,
          session.taskConfig!.verifyCommand,
          budget
        )
      );

      const expertResults = await Promise.all(expertPromises);
      for (const results of expertResults) {
        allResults.push(...results);
      }

      session.iterations = allResults;

      const ranked = expertConfigs.length > 0
        ? rankByVoting(allResults, expertConfigs[0])
        : allResults;

      session.bestResult = ranked[0] || null;
      session.status = 'complete';

      const passed = allResults.some((r) => r.passed);
      const bestScore = ranked[0]?.score ?? 0;
      const totalIters = allResults.length;
      const totalTokens = session.totalPromptTokens + session.totalCompletionTokens;
      const totalDuration = (Date.now() - solveStart) / 1000;
      const cost = session.totalCost;

      learnFromProblem(session);

      session.budget.costUsed += cost;
      session.budget.timeUsed += totalDuration;

      let text = passed
        ? `✅ SOLVED in ${totalIters} iterations (${totalDuration.toFixed(1)}s, ${totalTokens} tokens, $${cost.toFixed(4)})`
        : `❌ Not fully solved after ${totalIters} iterations (best score: ${bestScore.toFixed(2)}, ${totalDuration.toFixed(1)}s, $${cost.toFixed(4)})`;

      text += `\nExperts: ${expertConfigs.length} | Models: ${session.taskConfig.models.join(', ')}`;
      text += `\nPrompt tokens: ${session.totalPromptTokens} | Completion tokens: ${session.totalCompletionTokens} | Cost: $${cost.toFixed(4)}`;

      if (session.strategyAdaptations.length > 0) {
        text += `\n🧠 ${session.strategyAdaptations.length} strategy adaptation(s) active`;
      }

      if (ranked[0]) {
        text += `\n\nBest solution (expert ${ranked[0].expertIndex}, iteration ${ranked[0].iteration}):`;
        if (ranked[0].code) {
          text += `\n\`\`\`javascript\n${ranked[0].code.slice(0, 2000)}\n\`\`\``;
        } else if (ranked[0].answer) {
          text += `\n${ranked[0].answer.slice(0, 2000)}`;
        }
      }

      return {
        text,
        details: {
          passed,
          bestScore,
          totalIterations: totalIters,
          totalTokens,
          cost,
          totalDuration,
          rankedResults: ranked.map((r) => ({
            expert: r.expertIndex,
            iteration: r.iteration,
            passed: r.passed,
            score: r.score,
          })),
        },
      };
    }

    case 'status': {
      if (!session.taskConfig) {
        return { text: 'No session initialized.', details: { status: 'uninitialized' } };
      }

      const passed = session.iterations.some((r) => r.passed);
      const bestScore = session.bestResult?.score ?? 0;
      const totalTokens = session.totalPromptTokens + session.totalCompletionTokens;

      let text = `🧠 ${session.taskConfig.name}\n`;
      text += `Status: ${session.status}\n`;
      text += `Type: ${session.taskConfig.type} | Verification: ${session.taskConfig.verification}\n`;
      text += `Experts: ${session.taskConfig.numExperts} | Models: ${session.taskConfig.models.join(', ')}\n`;
      text += `Iterations: ${session.iterations.length}\n`;
      text += `Best score: ${bestScore.toFixed(2)} | Solved: ${passed ? '✅' : '❌'}\n`;
      text += `Tokens: ${totalTokens} | Cost: $${session.totalCost.toFixed(4)}\n`;
      text += `Budget: ${session.budget.problemsSolved}/${session.budget.problemsAttempted} solved | $${session.budget.costUsed.toFixed(4)} spent\n`;

      if (session.strategyAdaptations.length > 0) {
        text += `\n🧠 Strategy adaptations (${session.strategyAdaptations.length}):`;
        for (const a of session.strategyAdaptations.slice(-5)) {
          text += `\n  • ${a.insight} (evidence: ${a.evidenceCount}, type: ${a.taskType})`;
        }
      }

      return {
        text,
        details: {
          sessionId: session.id,
          taskConfig: session.taskConfig,
          status: session.status,
          iterationCount: session.iterations.length,
          bestScore,
          solved: passed,
          totalTokens,
          cost: session.totalCost,
          budget: session.budget,
          adaptations: session.strategyAdaptations.length,
        },
      };
    }

    case 'results': {
      const last = (params.last as number) || 10;
      const results = session.iterations.slice(-last);

      if (results.length === 0) {
        return { text: 'No iterations yet.', details: {} };
      }

      const lines = results.map(
        (r) =>
          `#${r.iteration} E${r.expertIndex} | score=${r.score.toFixed(2)} | ${r.passed ? '✅' : '❌'} | ${r.durationMs}ms | ${r.promptTokens + r.completionTokens} tok`
      );

      return {
        text: lines.join('\n'),
        details: { results },
      };
    }

    case 'learn': {
      if (session.strategyAdaptations.length === 0) {
        return {
          text: 'No strategy adaptations learned yet. Solve some problems first.',
          details: { adaptations: [], history: session.problemHistory },
        };
      }

      const lines = session.strategyAdaptations.map(
        (a, i) =>
          `${i + 1}. [${a.taskType}] ${a.insight}\n   Evidence: ${a.evidenceCount} | Models: ${a.models.join(', ')}${a.promptModifier ? `\n   Prompt mod: "${a.promptModifier.slice(0, 100)}..."` : ''}`
      );

      return {
        text: `🧠 Learned strategy adaptations:\n\n${lines.join('\n\n')}\n\nTotal problems: ${session.budget.problemsAttempted} | Solved: ${session.budget.problemsSolved}`,
        details: {
          adaptations: session.strategyAdaptations,
          history: session.problemHistory,
        },
      };
    }

    case 'clear': {
      sessions.delete(session.id);
      return { text: 'Session cleared.', details: {} };
    }

    case 'reset-learn': {
      session.strategyAdaptations = [];
      session.problemHistory = [];
      session.budget = { costUsed: 0, timeUsed: 0, problemsSolved: 0, problemsAttempted: 0 };
      return { text: '🧠 Strategy adaptations and problem history cleared.', details: {} };
    }

    default:
      throw new Error(`Unknown action: ${action}`);
  }
}

// =============================================================================
// HTTP Server
// =============================================================================

const PORT = Number(process.env.PI_REASON_HARNESS_PORT ?? 9880);
const startedAt = Date.now();
const LOG = process.env.PI_REASON_HARNESS_LOG ?? '/tmp/pi-reason-harness.log';

const TEXT_JSON = { 'content-type': 'application/json; charset=utf-8' } as const;

function serverLog(msg: string): void {
  const ts = new Date().toISOString();
  try { fs.appendFileSync(LOG, `[${ts}] ${msg}\n`); } catch {}
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on('data', (chunk: Buffer) => chunks.push(chunk));
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf-8')));
    req.on('error', reject);
  });
}

function header(req: IncomingMessage, name: string): string | undefined {
  const val = req.headers[name.toLowerCase()];
  if (typeof val === 'string') return val;
  if (Array.isArray(val)) return val[0];
  return undefined;
}

const server = createServer(async (req: IncomingMessage, res: ServerResponse) => {
  const url = new URL(req.url ?? '/', `http://127.0.0.1:${PORT}`);

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  if (req.method === 'GET' && url.pathname === '/health') {
    res.writeHead(200, TEXT_JSON);
    res.end(
      JSON.stringify({
        ok: true,
        uptime: Math.floor((Date.now() - startedAt) / 1000),
        sessions: sessions.size,
      })
    );
    return;
  }

  if (req.method === 'POST' && url.pathname === '/action') {
    let body: string;
    try {
      body = await readBody(req);
    } catch {
      res.writeHead(400, TEXT_JSON);
      res.end(JSON.stringify({ ok: false, error: 'failed to read body' }));
      return;
    }

    let params: Record<string, unknown>;
    try {
      params = JSON.parse(body);
    } catch {
      res.writeHead(400, TEXT_JSON);
      res.end(JSON.stringify({ ok: false, error: 'invalid JSON' }));
      return;
    }

    const action = params.action;
    if (!action || typeof action !== 'string') {
      res.writeHead(400, TEXT_JSON);
      res.end(JSON.stringify({ ok: false, error: 'missing action' }));
      return;
    }

    const sessionId = header(req, 'x-session-id');

    try {
      serverLog(`action=${action} session=${sessionId || 'default'}`);
      const { text, details } = await dispatchAction(action, params, sessionId);
      res.writeHead(200, TEXT_JSON);
      res.end(JSON.stringify({ ok: true, result: { text, details } }));
    } catch (e) {
      serverLog(`error: ${e instanceof Error ? e.message : String(e)}`);
      res.writeHead(500, TEXT_JSON);
      res.end(JSON.stringify({ ok: false, error: e instanceof Error ? e.message : String(e) }));
    }
    return;
  }

  if (req.method === 'POST' && url.pathname === '/quit') {
    res.writeHead(200, TEXT_JSON);
    res.end(JSON.stringify({ ok: true }));
    server.close();
    return;
  }

  res.writeHead(404, TEXT_JSON);
  res.end(JSON.stringify({ ok: false, error: 'not found' }));
});

server.listen(PORT, '127.0.0.1', () => {
  serverLog(`Pi Reason Harness server listening on 127.0.0.1:${PORT}`);
});

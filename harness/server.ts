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
// Meta-system — the proprietary layer, rebuilt from first principles
//
// What Poetiq's open-source code shows: a static harness with fixed prompts
// and hardcoded config. What their blog results prove: a dynamic system that
// adapts. The gap is THIS code.
//
// Five core insights derived from first principles:
//
// 1. CRITIQUE, DON'T CREATE — The analyzer should propose TARGETED DELTAS to
//    proven templates, not write prompts from scratch. Our old analyzer
//    generated 1-liner prompts that failed. Code review > writing from zero.
//
// 2. BUDGET IS A BANDIT — Allocate compute where ROI is highest. Stop experts
//    that show no progress. Re-explore when stuck. This is how Poetiq hits
//    "half the cost" — not hard limits, but intelligent allocation.
//
// 3. META-RULES COMPOUND — Every strategy improvement extracts a generalizable
//    PRINCIPLE (e.g., "add worked examples", "lower temp for hard problems").
//    These principles bias ALL future improvements. The meta-system gets
//    smarter over time, not just per-category.
//
// 4. CROSS-DOMAIN TRANSFER — Grid strategies don't transfer to knowledge
//    extraction, but the PRINCIPLE does. "Provide concrete examples" works
//    everywhere. We extract principles from successful strategies and
//    test them on other categories.
//
// 5. AUTO-TRIGGER — The meta-improver runs automatically after every N
//    problems, on success rate drops, on new categories, and on staleness.
//    No human in the loop. The system self-improves continuously.
//
// Architecture:
//   Layer 0: Problem Critic — inspects problem, proposes delta to template
//   Layer 1: Strategy Library — persistent proven strategies with ROI data
//   Layer 2: Meta-Rule Engine — cross-strategy principles that compound
//   Layer 3: Budget Bandit — Thompson sampling for compute allocation
//   Layer 4: Recursive Improvement — evolves strategies, extracts principles
//   Layer 5: Auto-Trigger — runs improvement automatically
// =============================================================================

// ---------------------------------------------------------------------------
// Meta-System Types
// ---------------------------------------------------------------------------

interface ProblemFeatures {
  /** Brief description */
  summary: string;
  /** Problem category */
  category: string;
  /** Estimated difficulty 0-1 */
  difficulty: number;
  /** Key patterns detected */
  keyPatterns: string[];
  /** Suggested approach description */
  suggestedApproach: string;
  /** Whether code execution is the right path */
  requiresCode: boolean;
  /** Suggested number of iterations */
  suggestedMaxIterations: number;
  /** Suggested temperature */
  suggestedTemperature: number;
  /** Suggested reasoning level */
  suggestedReasoning: 'off' | 'minimal' | 'low' | 'medium' | 'high';
  /** Sub-questions for chain-of-questions decomposition */
  subQuestions: string[];
  /** Best model for this problem type (null if unknown) */
  preferredModel: string | null;
  /** TARGETED DELTA to the default prompt (NOT a full prompt) */
  promptDelta: PromptDelta;
}

/** A targeted modification to a proven prompt template */
interface PromptDelta {
  /** Section to insert BEFORE the $$problem$$ placeholder */
  preProblemInsert: string | null;
  /** Section to insert AFTER the $$problem$$ placeholder */
  postProblemInsert: string | null;
  /** Specific instructions to REPLACE in the template (section header → replacement) */
  sectionReplacements: Record<string, string>;
  /** Additional examples to include (problem → solution pairs) */
  additionalExamples: Array<{ problem: string; solution: string }>;
  /** Anti-patterns: things the solver should NOT do for this problem type */
  antiPatterns: string[];
}

interface StrategyEntry {
  id: string;
  created: number;
  category: string;
  /** The FULL solver prompt (base template + delta applied) */
  solverPrompt: string;
  /** The FULL feedback prompt */
  feedbackPrompt: string;
  /** Config overrides */
  configOverrides: Partial<ExpertConfig>;
  /** The delta that was applied to produce this prompt (for lineage) */
  appliedDelta: PromptDelta | null;
  useCount: number;
  successCount: number;
  avgScore: number;
  totalCost: number;
  totalTime: number;
  testedModels: string[];
  parentId: string | null;
  generation: number;
  /** Prompt quality metrics — fast feedback signals */
  qualityMetrics: PromptQualityMetrics;
}

interface PromptQualityMetrics {
  /** How often does this prompt produce valid code blocks? [0-1] */
  codeParseRate: number;
  /** How often does the code run without sandbox errors? [0-1] */
  sandboxSuccessRate: number;
  /** Average first-iteration score (before feedback kicks in) [0-1] */
  avgFirstIterationScore: number;
  /** Total observations used to compute these rates */
  observationCount: number;
}

/** A meta-rule — a generalizable principle extracted from strategy evolution */
interface MetaRule {
  id: string;
  /** The principle in natural language */
  principle: string;
  /** Which categories it has been validated on */
  validatedCategories: string[];
  /** How many times applying this rule improved a strategy */
  improvementCount: number;
  /** How many times it was tested */
  testCount: number;
  /** The delta this rule suggests (template for modification) */
  suggestedDelta: Partial<PromptDelta>;
  /** Source strategy that inspired this rule */
  sourceStrategyId: string | null;
  /** When it was created */
  created: number;
  /** Last time it was validated */
  lastValidated: number;
}

/** Model routing stats — per model × category */
interface ModelRouteStats {
  modelId: string;
  category: string;
  uses: number;
  successes: number;
  avgScore: number;
  avgCost: number;
  avgTime: number;
}

// ---------------------------------------------------------------------------
// Layer 0: Problem Critic — "critique, don't create"
// ---------------------------------------------------------------------------

const CRITIC_PROMPT = `You are a meta-reasoning expert. You will receive a PROVEN solver prompt template and a PROBLEM to solve. Your job is NOT to rewrite the prompt. Your job is to propose TARGETED MODIFICATIONS (deltas) that adapt the proven template to this specific problem type.

Think of this as code review: you don't rewrite the whole file, you propose specific insertions, replacements, and anti-patterns.

## PROVEN SOLVER TEMPLATE (the base prompt that works well in general)

$$baseTemplate$$

## PROBLEM TO ANALYZE

$$problem$$

## YOUR TASK

Analyze the problem and propose targeted modifications. Respond in EXACT JSON format (no markdown, no backticks, just raw JSON):

{
  "summary": "one-line description",
  "category": "one of: grid-transformation, pattern-completion, sequence-prediction, spatial-reasoning, knowledge-synthesis, mathematical, logical-inference, code-generation, other",
  "difficulty": 0.7,
  "keyPatterns": ["pattern 1", "pattern 2"],
  "suggestedApproach": "detailed description of how to attack this",
  "requiresCode": true,
  "suggestedMaxIterations": 8,
  "suggestedTemperature": 1.0,
  "suggestedReasoning": "high",
  "subQuestions": ["sub-question 1", "sub-question 2"],
  "preferredModel": null,
  "promptDelta": {
    "preProblemInsert": "Specific instructions to add RIGHT BEFORE the problem examples. Focus on what's unique about this problem type. E.g., 'For this type of problem, the key insight is to look for connected components using BFS/DFS.'",
    "postProblemInsert": "Additional instructions to add AFTER the problem examples. E.g., 'CRITICAL: Your function MUST be named transform() and take a single 2D array. No console.log or test code.'",
    "sectionReplacements": {},
    "additionalExamples": [],
    "antiPatterns": ["Don't write test harness code", "Don't use console.log", "Don't hardcode values from training examples"]
  }
}

Rules:
- preProblemInsert: Focus on problem-specific STRATEGY adjustments (what approach to use, what to look for)
- postProblemInsert: Focus on FORMAT enforcement (function signature, no I/O, pure function)
- antiPatterns: Things the solver commonly gets wrong for this problem type
- Keep the base template's structure — only ADD what's needed
- For grid/array problems: requiresCode=true, add spatial reasoning hints
- For factual questions: requiresCode=false, add chain-of-questions decomposition
- For hard problems (difficulty > 0.7): more iterations, lower temperature, higher reasoning
- preferredModel: provider/id format or null`;

async function critiqueAndAdapt(
  problem: string,
  baseSolverPrompt: string,
  modelId: string
): Promise<ProblemFeatures> {
  // Truncate the base template to fit context window (keep structure, trim examples)
  const truncatedTemplate = baseSolverPrompt.length > 3000
    ? baseSolverPrompt.slice(0, 3000) + '\n... [template truncated for analysis]'
    : baseSolverPrompt;

  const prompt = CRITIC_PROMPT
    .replace('$$baseTemplate$$', truncatedTemplate)
    .replace('$$problem$$', problem.slice(0, 4000));

  const result = await callLLM(modelId, prompt, 0.3, 120, 1, 'high');

  try {
    let content = result.content.trim();
    const jsonMatch = content.match(/```(?:json)?\s*([\s\S]*?)```/);
    if (jsonMatch) content = jsonMatch[1].trim();

    const parsed = JSON.parse(content);
    const delta: PromptDelta = {
      preProblemInsert: parsed.promptDelta?.preProblemInsert || null,
      postProblemInsert: parsed.promptDelta?.postProblemInsert || null,
      sectionReplacements: parsed.promptDelta?.sectionReplacements || {},
      additionalExamples: parsed.promptDelta?.additionalExamples || [],
      antiPatterns: parsed.promptDelta?.antiPatterns || [],
    };

    return {
      summary: parsed.summary || 'Unknown problem',
      category: parsed.category || 'other',
      difficulty: typeof parsed.difficulty === 'number' ? parsed.difficulty : 0.5,
      keyPatterns: Array.isArray(parsed.keyPatterns) ? parsed.keyPatterns : [],
      suggestedApproach: parsed.suggestedApproach || '',
      requiresCode: parsed.requiresCode !== false,
      suggestedMaxIterations: parsed.suggestedMaxIterations || 10,
      suggestedTemperature: parsed.suggestedTemperature ?? 1.0,
      suggestedReasoning: parsed.suggestedReasoning || 'off',
      subQuestions: Array.isArray(parsed.subQuestions) ? parsed.subQuestions : [],
      preferredModel: parsed.preferredModel || null,
      promptDelta: delta,
    };
  } catch {
    return {
      summary: 'Analysis failed — using defaults',
      category: 'other',
      difficulty: 0.5,
      keyPatterns: [],
      suggestedApproach: '',
      requiresCode: true,
      suggestedMaxIterations: 10,
      suggestedTemperature: 1.0,
      suggestedReasoning: 'off',
      subQuestions: [],
      preferredModel: null,
      promptDelta: { preProblemInsert: null, postProblemInsert: null, sectionReplacements: {}, additionalExamples: [], antiPatterns: [] },
    };
  }
}

/** Apply a PromptDelta to a base prompt, producing a modified prompt */
function applyPromptDelta(basePrompt: string, delta: PromptDelta): string {
  let result = basePrompt;

  // Apply section replacements
  for (const [section, replacement] of Object.entries(delta.sectionReplacements)) {
    result = result.replace(section, replacement);
  }

  // Insert before $$problem$$
  if (delta.preProblemInsert) {
    const problemIdx = result.indexOf('$$problem$$');
    if (problemIdx !== -1) {
      result = result.slice(0, problemIdx) +
        '\n\n**Problem-Specific Strategy:**\n' + delta.preProblemInsert + '\n\n' +
        result.slice(problemIdx);
    }
  }

  // Insert after $$problem$$
  if (delta.postProblemInsert) {
    result = result.replace('$$problem$$', () => '$$problem$$\n\n**Critical Reminders:**\n' + delta.postProblemInsert);
  }

  // Add anti-patterns
  if (delta.antiPatterns.length > 0) {
    result += '\n\n**DO NOT:**\n' + delta.antiPatterns.map((a, i) => `${i + 1}. ${a}`).join('\n');
  }

  // Add additional examples
  if (delta.additionalExamples.length > 0) {
    const examplesStr = delta.additionalExamples
      .map((e, i) => `**Custom Example ${i + 1}:**\nProblem: ${e.problem}\nSolution: ${e.solution}`)
      .join('\n\n');
    result = result.replace('$$problem$$', () => examplesStr + '\n\n$$problem$$');
  }

  return result;
}

// ---------------------------------------------------------------------------
// Layer 1: Strategy Library — persistent store with ROI data
// ---------------------------------------------------------------------------

const STRATEGY_LIB_PATH = process.env.PI_REASON_HARNESS_STRATEGIES ??
  join(process.env.HOME || '/tmp', '.pi-reason-harness', 'strategies.json');

const META_RULES_PATH = process.env.PI_REASON_HARNESS_META_RULES ??
  join(process.env.HOME || '/tmp', '.pi-reason-harness', 'meta-rules.json');

const MODEL_ROUTE_PATH = process.env.PI_REASON_HARNESS_MODEL_ROUTES ??
  join(process.env.HOME || '/tmp', '.pi-reason-harness', 'model-routes.json');

let strategyLibrary: StrategyEntry[] = [];
let strategyLibLoaded = false;
let metaRules: MetaRule[] = [];
let metaRulesLoaded = false;
let modelRouteStats: ModelRouteStats[] = [];
let modelRouteLoaded = false;

function loadJSON<T>(path: string, fallback: T[]): T[] {
  try {
    const dir = join(path, '..');
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    if (fs.existsSync(path)) return JSON.parse(fs.readFileSync(path, 'utf-8'));
  } catch {}
  return fallback;
}

function saveJSON(path: string, data: unknown): void {
  try {
    const dir = join(path, '..');
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path, JSON.stringify(data, null, 2), 'utf-8');
  } catch (e) {
    serverLog(`save error for ${path}: ${e}`);
  }
}

function loadStrategyLibrary(): StrategyEntry[] {
  if (strategyLibLoaded) return strategyLibrary;
  strategyLibrary = loadJSON(STRATEGY_LIB_PATH, []);
  strategyLibLoaded = true;
  return strategyLibrary;
}

function saveStrategyLibrary(): void {
  saveJSON(STRATEGY_LIB_PATH, strategyLibrary);
}

function loadMetaRules(): MetaRule[] {
  if (metaRulesLoaded) return metaRules;
  metaRules = loadJSON(META_RULES_PATH, []);
  metaRulesLoaded = true;
  return metaRules;
}

function saveMetaRules(): void {
  saveJSON(META_RULES_PATH, metaRules);
}

function loadModelRoutes(): ModelRouteStats[] {
  if (modelRouteLoaded) return modelRouteStats;
  modelRouteStats = loadJSON(MODEL_ROUTE_PATH, []);
  modelRouteLoaded = true;
  return modelRouteStats;
}

function saveModelRoutes(): void {
  saveJSON(MODEL_ROUTE_PATH, modelRouteStats);
}

const DEFAULT_QUALITY: PromptQualityMetrics = {
  codeParseRate: 0,
  sandboxSuccessRate: 0,
  avgFirstIterationScore: 0,
  observationCount: 0,
};

function recordStrategyUse(
  strategyId: string,
  success: boolean,
  score: number,
  cost: number,
  timeS: number,
  model: string
): void {
  const entry = strategyLibrary.find((s) => s.id === strategyId);
  if (!entry) return;
  entry.useCount++;
  if (success) entry.successCount++;
  entry.avgScore = (entry.avgScore * (entry.useCount - 1) + score) / entry.useCount;
  entry.totalCost += cost;
  entry.totalTime += timeS;
  if (!entry.testedModels.includes(model)) entry.testedModels.push(model);
  saveStrategyLibrary();
}

/** Record prompt quality metrics (fast feedback signals) */
function recordPromptQuality(
  strategyId: string,
  codeParsed: boolean,
  sandboxOk: boolean,
  firstIterationScore: number
): void {
  const entry = strategyLibrary.find((s) => s.id === strategyId);
  if (!entry) return;
  const m = entry.qualityMetrics;
  const n = m.observationCount;
  m.codeParseRate = (m.codeParseRate * n + (codeParsed ? 1 : 0)) / (n + 1);
  m.sandboxSuccessRate = (m.sandboxSuccessRate * n + (sandboxOk ? 1 : 0)) / (n + 1);
  m.avgFirstIterationScore = (m.avgFirstIterationScore * n + firstIterationScore) / (n + 1);
  m.observationCount = n + 1;
  saveStrategyLibrary();
}

function findBestStrategy(category: string, models: string[]): StrategyEntry | null {
  loadStrategyLibrary();
  const candidates = strategyLibrary
    .filter((s) => s.category === category || s.category === '*')
    .filter((s) => s.useCount >= 1)
    .sort((a, b) => {
      // Composite ROI: success * score / cost * quality_boost
      const qualA = 1 + (a.qualityMetrics.codeParseRate * 0.5 + a.qualityMetrics.sandboxSuccessRate * 0.3);
      const qualB = 1 + (b.qualityMetrics.codeParseRate * 0.5 + b.qualityMetrics.sandboxSuccessRate * 0.3);
      const roiA = (a.successCount / Math.max(a.useCount, 1)) * a.avgScore / Math.max(a.totalCost, 0.001) * qualA;
      const roiB = (b.successCount / Math.max(b.useCount, 1)) * b.avgScore / Math.max(b.totalCost, 0.001) * qualB;
      return roiB - roiA;
    });
  return candidates[0] || null;
}

// ---------------------------------------------------------------------------
// Layer 2: Meta-Rule Engine — cross-strategy principles that compound
// ---------------------------------------------------------------------------

/** Extract meta-rules from a strategy evolution (parent → child) */
const RULE_EXTRACTOR_PROMPT = `You are a meta-reasoning expert. You have observed a strategy evolution where a child strategy outperformed its parent. Your job is to extract GENERALIZABLE PRINCIPLES from this improvement.

## Parent Strategy
Category: $$category$$
Solver prompt (excerpt):
$$parentPrompt$$
Config overrides: $$parentConfig$$
Score: $$parentScore$$ | Uses: $$parentUses$$

## Child Strategy (the improvement)
Solver prompt (excerpt):
$$childPrompt$$
Config overrides: $$childConfig$$
Score: $$childScore$$ | Uses: $$childUses$$

## Key Difference
$$diffSummary$$

---

Extract 1-3 generalizable principles from this improvement. A principle should be:
- Abstract enough to apply to OTHER problem categories (not just $$category$$)
- Concrete enough to translate into specific prompt modifications
- Falsifiable (we can test it and see if it helps)

Respond in EXACT JSON format (no markdown, no backticks, just raw JSON):

{
  "rules": [
    {
      "principle": "A concise statement of the principle",
      "suggestedDelta": {
        "preProblemInsert": "text to insert before the problem, or null",
        "postProblemInsert": "text to insert after the problem, or null",
        "antiPatterns": ["things to avoid"]
      },
      "rationale": "Why this principle should generalize"
    }
  ]
}`;

async function extractMetaRules(
  parent: StrategyEntry,
  child: StrategyEntry,
  modelId: string
): Promise<MetaRule[]> {
  // Compute diff summary
  const diffLines: string[] = [];
  if (child.solverPrompt.length !== parent.solverPrompt.length) {
    diffLines.push(`Prompt length changed: ${parent.solverPrompt.length} → ${child.solverPrompt.length} chars`);
  }
  if (child.configOverrides.temperature !== parent.configOverrides.temperature) {
    diffLines.push(`Temperature: ${parent.configOverrides.temperature} → ${child.configOverrides.temperature}`);
  }
  if (child.configOverrides.maxIterations !== parent.configOverrides.maxIterations) {
    diffLines.push(`Max iterations: ${parent.configOverrides.maxIterations} → ${child.configOverrides.maxIterations}`);
  }
  if (child.configOverrides.reasoning !== parent.configOverrides.reasoning) {
    diffLines.push(`Reasoning: ${parent.configOverrides.reasoning} → ${child.configOverrides.reasoning}`);
  }
  if (diffLines.length === 0) diffLines.push('Subtle prompt wording changes');

  const prompt = RULE_EXTRACTOR_PROMPT
    .replace('$$category$$', child.category)
    .replace('$$parentPrompt$$', parent.solverPrompt.slice(0, 1500))
    .replace('$$parentConfig$$', JSON.stringify(parent.configOverrides))
    .replace('$$parentScore$$', parent.avgScore.toFixed(3))
    .replace('$$parentUses$$', String(parent.useCount))
    .replace('$$childPrompt$$', child.solverPrompt.slice(0, 1500))
    .replace('$$childConfig$$', JSON.stringify(child.configOverrides))
    .replace('$$childScore$$', child.avgScore.toFixed(3))
    .replace('$$childUses$$', String(child.useCount))
    .replace('$$diffSummary$$', diffLines.join('; '));

  const result = await callLLM(modelId, prompt, 0.3, 60, 1, 'high');

  try {
    let content = result.content.trim();
    const jsonMatch = content.match(/```(?:json)?\s*([\s\S]*?)```/);
    if (jsonMatch) content = jsonMatch[1].trim();
    const parsed = JSON.parse(content);

    if (!Array.isArray(parsed.rules)) return [];

    return parsed.rules
      .filter((r: any) => r.principle && r.suggestedDelta)
      .map((r: any): MetaRule => ({
        id: randomBytes(3).toString('hex'),
        principle: r.principle,
        validatedCategories: [],
        improvementCount: 0,
        testCount: 0,
        suggestedDelta: r.suggestedDelta || {},
        sourceStrategyId: child.id,
        created: Date.now(),
        lastValidated: 0,
      }));
  } catch {
    return [];
  }
}

/** Apply relevant meta-rules as additional deltas to a prompt */
function applyMetaRules(
  category: string,
  difficulty: number
): PromptDelta {
  loadMetaRules();
  const now = Date.now();
  const STALE_MS = 7 * 24 * 60 * 60 * 1000; // 7 days

  // Filter to rules validated for this category or universal rules
  const relevant = metaRules
    .filter((r) => {
      const isCategoryMatch = r.validatedCategories.includes(category) || r.validatedCategories.length === 0;
      const isFresh = now - r.lastValidated < STALE_MS || r.lastValidated === 0;
      const hasPositiveEvidence = r.improvementCount > 0 || r.testCount < 5; // untested = worth trying
      return isCategoryMatch && isFresh && hasPositiveEvidence;
    })
    .sort((a, b) => {
      // Prioritize rules with more positive evidence
      const scoreA = a.improvementCount / Math.max(a.testCount, 1);
      const scoreB = b.improvementCount / Math.max(b.testCount, 1);
      return scoreB - scoreA;
    });

  // Merge top rules into a combined delta
  const combinedDelta: PromptDelta = {
    preProblemInsert: null,
    postProblemInsert: null,
    sectionReplacements: {},
    additionalExamples: [],
    antiPatterns: [],
  };

  for (const rule of relevant.slice(0, 3)) { // max 3 rules to avoid prompt bloat
    const d = rule.suggestedDelta;
    if (d.preProblemInsert) {
      combinedDelta.preProblemInsert = (combinedDelta.preProblemInsert || '') + '\n' + d.preProblemInsert;
    }
    if (d.postProblemInsert) {
      combinedDelta.postProblemInsert = (combinedDelta.postProblemInsert || '') + '\n' + d.postProblemInsert;
    }
    if (d.antiPatterns) {
      combinedDelta.antiPatterns.push(...d.antiPatterns);
    }
    Object.assign(combinedDelta.sectionReplacements, d.sectionReplacements || {});
  }

  // Apply difficulty-based rules (these are hardcoded meta-principles)
  if (difficulty > 0.7) {
    combinedDelta.antiPatterns.push(
      'Do not use brute-force approaches — they will time out on hard problems',
      'Do not hardcode values from training examples'
    );
  }

  return combinedDelta;
}

/** Validate a meta-rule by testing it on a category */
function validateMetaRule(ruleId: string, category: string, improved: boolean): void {
  loadMetaRules();
  const rule = metaRules.find((r) => r.id === ruleId);
  if (!rule) return;
  rule.testCount++;
  if (improved) rule.improvementCount++;
  if (!rule.validatedCategories.includes(category)) rule.validatedCategories.push(category);
  rule.lastValidated = Date.now();
  saveMetaRules();
}

// ---------------------------------------------------------------------------
// Layer 3: Model Router — Thompson sampling for model selection
// ---------------------------------------------------------------------------

function recordModelRoute(
  modelId: string,
  category: string,
  success: boolean,
  score: number,
  cost: number,
  timeS: number
): void {
  loadModelRoutes();
  const existing = modelRouteStats.find(
    (m) => m.modelId === modelId && m.category === category
  );
  if (existing) {
    existing.uses++;
    if (success) existing.successes++;
    existing.avgScore = (existing.avgScore * (existing.uses - 1) + score) / existing.uses;
    existing.avgCost = (existing.avgCost * (existing.uses - 1) + cost) / existing.uses;
    existing.avgTime = (existing.avgTime * (existing.uses - 1) + timeS) / existing.uses;
  } else {
    modelRouteStats.push({
      modelId,
      category,
      uses: 1,
      successes: success ? 1 : 0,
      avgScore: score,
      avgCost: cost,
      avgTime: timeS,
    });
  }
  saveModelRoutes();
}

/** Thompson sampling: pick the model with highest expected reward,
 *  with exploration bonus for under-tested models */
function thompsonSampleModel(
  category: string,
  availableModels: string[]
): string {
  if (availableModels.length <= 1) return availableModels[0];

  loadModelRoutes();

  const categoryStats = modelRouteStats.filter(
    (m) => m.category === category && availableModels.includes(m.modelId)
  );

  // Thompson sampling with Beta distribution
  // α = successes + 1, β = (uses - successes) + 1 (Laplace smoothing)
  let bestModel = availableModels[0];
  let bestSample = -Infinity;

  for (const model of availableModels) {
    const stats = categoryStats.find((s) => s.modelId === model);
    const alpha = (stats ? stats.successes : 0) + 1;
    const beta = (stats ? stats.uses - stats.successes : 0) + 1;

    // Sample from Beta(alpha, beta) using the gamma trick
    const sample = betaSample(alpha, beta);
    // Adjust for cost efficiency
    const costEfficiency = stats && stats.avgCost > 0
      ? Math.min(1 / (stats.avgCost * 10), 2) // penalize expensive models
      : 1;
    const adjustedSample = sample * costEfficiency;

    if (adjustedSample > bestSample) {
      bestSample = adjustedSample;
      bestModel = model;
    }
  }

  return bestModel;
}

/** Simple Beta distribution sampler using gamma ratio */
function betaSample(alpha: number, beta: number): number {
  const x = gammaVariate(alpha);
  const y = gammaVariate(beta);
  return x / (x + y);
}

/** Gamma variate using Marsaglia and Tsang's method */
function gammaVariate(shape: number): number {
  if (shape < 1) {
    // Use Ahrens-Dieter: Gamma(shape) = Gamma(shape+1) * U^(1/shape)
    return gammaVariate(shape + 1) * Math.pow(Math.random(), 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  while (true) {
    let x, v;
    do {
      x = randn();
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = Math.random();
    if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

/** Standard normal random variate (Box-Muller) */
function randn(): number {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ---------------------------------------------------------------------------
// Layer 4: Budget Bandit — intelligent compute allocation
// ---------------------------------------------------------------------------

/** Early stopping: should we stop an expert that's making no progress? */
function shouldStopEarly(
  iterationHistory: Array<{ score: number; passed: boolean }>,
  minIterations: number = 3
): { stop: boolean; reason: string } {
  if (iterationHistory.length < minIterations) return { stop: false, reason: '' };

  // Check if we've been stuck at the same low score
  const last3 = iterationHistory.slice(-3);
  const allFailed = last3.every((r) => !r.passed);
  const noProgress = last3.every((r) => r.score === last3[0].score) && last3[0].score < 0.5;

  if (allFailed && noProgress) {
    return { stop: true, reason: `No progress after ${iterationHistory.length} iterations (score stuck at ${last3[0].score.toFixed(2)})` };
  }

  // Check if score is decreasing (getting worse)
  if (iterationHistory.length >= 4) {
    const last4 = iterationHistory.slice(-4);
    const decreasing = last4.every((r, i) => i === 0 || r.score <= last4[i - 1].score);
    if (decreasing && last4[3].score < 0.3) {
      return { stop: true, reason: `Score decreasing over last 4 iterations (latest: ${last4[3].score.toFixed(2)})` };
    }
  }

  return { stop: false, reason: '' };
}

/** Determine if we should re-explore with a different strategy */
function shouldReExplore(
  allExpertResults: IterationResult[][],
  totalIterations: number
): { reExplore: boolean; reason: string } {
  // If ALL experts are stuck at 0 after enough iterations, re-explore
  const allStuck = allExpertResults.every((results) => {
    const last3 = results.slice(-3);
    return last3.length >= 3 && last3.every((r) => r.score === 0);
  });

  if (allStuck && totalIterations >= 5) {
    return { reExplore: true, reason: 'All experts stuck at score 0 — need a different approach' };
  }

  return { reExplore: false, reason: '' };
}

// ---------------------------------------------------------------------------
// Layer 5: Recursive Strategy Improvement + Auto-Trigger
// ---------------------------------------------------------------------------

const IMPROVER_PROMPT = `You are a meta-reasoning expert. Below is a problem-solving strategy with its test results. Your job is to generate an IMPROVED version.

## Current Strategy
Category: $$category$$ | Generation: $$generation$$
Solver prompt (excerpt):
$$solverPrompt$$
Feedback prompt (excerpt):
$$feedbackPrompt$$
Config overrides: $$configOverrides$$

## Test Results
Uses: $$useCount$$ | Successes: $$successCount$$
Average score: $$avgScore$$
Total cost: $$$$totalCost$$ | Total time: $$totalTime$$s
Models tested: $$testedModels$$

## Prompt Quality Metrics
Code parse rate: $$codeParseRate$$ | Sandbox success rate: $$sandboxSuccessRate$$
First-iteration avg score: $$firstIterScore$$

## Recent Problem Details
$$recentProblems$$

## Active Meta-Rules (principles that have improved other strategies)
$$metaRules$$

---

Generate an improved strategy. The key insight: make TARGETED modifications, don't rewrite from scratch.

Respond in EXACT JSON format (no markdown, no backticks, just raw JSON):

{
  "solverPrompt": "improved solver prompt with $$problem$$ placeholder — modify the existing prompt, don't rewrite",
  "feedbackPrompt": "improved feedback prompt with $$feedback$$ placeholder",
  "configOverrides": {
    "temperature": 0.8,
    "maxIterations": 12,
    "reasoning": "high"
  },
  "appliedDelta": {
    "preProblemInsert": "what you added before the problem",
    "postProblemInsert": "what you added after the problem",
    "antiPatterns": ["what you told it to avoid"]
  },
  "rationale": "brief explanation of what you changed and why"
}

Rules:
- Keep what works, fix what doesn't — TARGETED MODIFICATIONS
- If codeParseRate is low: enforce code output format more strongly in the prompt
- If sandboxSuccessRate is low: add error handling instructions
- If firstIterationScore is low: improve the initial strategy guidance
- If the strategy has low success rate, try a RADICALLY different approach
- If it has high success rate but low avgScore, focus on quality improvements
- If it's expensive (high cost/low ROI), optimize for fewer iterations
- solverPrompt MUST include $$problem$$ placeholder
- feedbackPrompt MUST include $$feedback$$ placeholder
- Only include configOverrides that differ from defaults
- Apply relevant meta-rules where appropriate`;

async function improveStrategy(
  strategy: StrategyEntry,
  recentProblems: ProblemHistory[],
  modelId: string
): Promise<StrategyEntry | null> {
  loadMetaRules();
  const recentStr = recentProblems.slice(-5).map((p) =>
    `  - ${p.strategyType}: score=${p.bestScore} passed=${p.passed} iters=${p.iterationCount} models=${p.modelsUsed.join(',')}`
  ).join('\n');

  const rulesStr = metaRules
    .filter((r) => r.improvementCount > 0 || r.testCount < 3)
    .slice(0, 5)
    .map((r) => `  - [${r.id}] "${r.principle}" (improvements: ${r.improvementCount}/${r.testCount}, categories: ${r.validatedCategories.join(',') || 'untested'})`)
    .join('\n') || '  (none yet)';

  const m = strategy.qualityMetrics;
  const prompt = IMPROVER_PROMPT
    .replace('$$category$$', strategy.category)
    .replace('$$generation$$', String(strategy.generation))
    .replace('$$solverPrompt$$', strategy.solverPrompt.slice(0, 2000))
    .replace('$$feedbackPrompt$$', strategy.feedbackPrompt.slice(0, 800))
    .replace('$$configOverrides$$', JSON.stringify(strategy.configOverrides))
    .replace('$$useCount$$', String(strategy.useCount))
    .replace('$$successCount$$', String(strategy.successCount))
    .replace('$$avgScore$$', strategy.avgScore.toFixed(3))
    .replace('$$totalCost$$', strategy.totalCost.toFixed(4))
    .replace('$$totalTime$$', strategy.totalTime.toFixed(1))
    .replace('$$testedModels$$', strategy.testedModels.join(', ') || 'none')
    .replace('$$codeParseRate$$', m.codeParseRate.toFixed(2))
    .replace('$$sandboxSuccessRate$$', m.sandboxSuccessRate.toFixed(2))
    .replace('$$firstIterScore$$', m.avgFirstIterationScore.toFixed(2))
    .replace('$$recentProblems$$', recentStr || 'no recent data')
    .replace('$$metaRules$$', rulesStr);

  const result = await callLLM(modelId, prompt, 0.5, 120, 1, 'high');

  try {
    let content = result.content.trim();
    const jsonMatch = content.match(/```(?:json)?\s*([\s\S]*?)```/);
    if (jsonMatch) content = jsonMatch[1].trim();
    const parsed = JSON.parse(content);

    if (!parsed.solverPrompt || !parsed.feedbackPrompt) return null;

    const newStrategy: StrategyEntry = {
      id: randomBytes(4).toString('hex'),
      created: Date.now(),
      category: strategy.category,
      solverPrompt: parsed.solverPrompt,
      feedbackPrompt: parsed.feedbackPrompt,
      configOverrides: parsed.configOverrides || {},
      appliedDelta: parsed.appliedDelta || null,
      useCount: 0,
      successCount: 0,
      avgScore: 0,
      totalCost: 0,
      totalTime: 0,
      testedModels: [],
      parentId: strategy.id,
      generation: strategy.generation + 1,
      qualityMetrics: { ...DEFAULT_QUALITY },
    };

    return newStrategy;
  } catch {
    return null;
  }
}

/** Auto-trigger conditions for meta-improvement */
function shouldAutoImprove(session: SessionState): { improve: boolean; reason: string } {
  const history = session.problemHistory;
  if (history.length < 3) return { improve: false, reason: '' };

  // Condition 1: Recent success rate dropped
  const recent = history.slice(-5);
  const recentSuccessRate = recent.filter((p) => p.passed).length / recent.length;
  const overall = history.filter((p) => p.passed).length / history.length;
  if (recentSuccessRate < overall * 0.5 && recent.length >= 3) {
    return { improve: true, reason: `Recent success rate (${(recentSuccessRate * 100).toFixed(0)}%) dropped below half of overall (${(overall * 100).toFixed(0)}%)` };
  }

  // Condition 2: New category encountered
  const categories = new Set(history.slice(0, -1).map((p) => p.strategyType));
  const lastCategory = history[history.length - 1].strategyType;
  if (!categories.has(lastCategory)) {
    return { improve: true, reason: `New category encountered: ${lastCategory}` };
  }

  // Condition 3: Every 5 problems, run improvement
  if (history.length % 5 === 0) {
    return { improve: true, reason: `Periodic improvement (every 5 problems, now at ${history.length})` };
  }

  return { improve: false, reason: '' };
}

/** Run auto-improvement if conditions are met */
async function maybeAutoImprove(
  session: SessionState,
  modelId: string
): Promise<{ improved: boolean; reason: string; childId?: string }> {
  const { improve, reason } = shouldAutoImprove(session);
  if (!improve) return { improved: false, reason };

  loadStrategyLibrary();
  const candidates = strategyLibrary
    .filter((s) => s.useCount >= 1)
    .sort((a, b) => {
      // Worst ROI first — most room for improvement
      const roiA = (a.successCount / Math.max(a.useCount, 1)) * a.avgScore;
      const roiB = (b.successCount / Math.max(b.useCount, 1)) * b.avgScore;
      return roiA - roiB;
    });

  if (candidates.length === 0) return { improved: false, reason: `${reason} but no strategies in library` };

  const target = candidates[0];
  const child = await improveStrategy(target, session.problemHistory, modelId);

  if (!child) return { improved: false, reason: `${reason} but LLM didn't return valid strategy` };

  strategyLibrary.push(child);
  saveStrategyLibrary();

  // Extract meta-rules from the evolution
  const rules = await extractMetaRules(target, child, modelId);
  if (rules.length > 0) {
    metaRules.push(...rules);
    saveMetaRules();
    serverLog(`auto-meta: extracted ${rules.length} meta-rule(s) from ${target.id} → ${child.id}`);
  }

  serverLog(`auto-meta: improved strategy ${target.id} → ${child.id} (reason: ${reason})`);
  return { improved: true, reason, childId: child.id };
}

// =============================================================================
// Strategy templates — the base prompts the meta-system modifies
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
      const useMeta = (params.meta as boolean) || false;

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

      // =================================================================
      // META-SYSTEM V2: critique-don't-create + meta-rules + bandit
      // =================================================================
      let metaFeatures: ProblemFeatures | null = null;
      let usedStrategyId: string | null = null;
      let usedCustomPrompts = false;

      if (useMeta && session.taskConfig.models.length > 0) {
        // Build the problem text for the critic
        let analyzeText = problem;
        if (!analyzeText && trainInputs.length > 0) {
          const trainIn = trainInputs as number[][][];
          const trainOut = trainOutputs as number[][][];
          analyzeText = `Grid transformation problem with ${trainInputs.length} training examples:\n`;
          for (let e = 0; e < Math.min(trainIn.length, 2); e++) {
            analyzeText += `Input ${e+1}: ${JSON.stringify(trainIn[e])} → Output ${e+1}: ${JSON.stringify(trainOut[e])}\n`;
          }
        }

        // Layer 1: Check strategy library for a proven strategy
        const categoryGuess = session.taskConfig.type === 'code-reasoning' ? 'grid-transformation' : session.taskConfig.type;
        const bestStrategy = findBestStrategy(categoryGuess, session.taskConfig.models);

        // Layer 0: Critique the base template and propose deltas
        const basePrompt = session.taskConfig.type === 'code-reasoning'
          ? CODE_REASONING_SOLVER
          : session.taskConfig.type === 'knowledge-extraction'
            ? KNOWLEDGE_EXTRACTION_SOLVER
            : HYBRID_SOLVER;

        metaFeatures = await critiqueAndAdapt(analyzeText, basePrompt, session.taskConfig.models[0]);
        serverLog(`meta-critic: category=${metaFeatures.category} difficulty=${metaFeatures.difficulty} delta=${JSON.stringify({pre: !!metaFeatures.promptDelta.preProblemInsert, post: !!metaFeatures.promptDelta.postProblemInsert, anti: metaFeatures.promptDelta.antiPatterns.length})}`);

        // Layer 1: If library has a proven strategy with high ROI, prefer it
        if (bestStrategy && bestStrategy.avgScore > 0.5 && (bestStrategy.useCount >= 2 || bestStrategy.qualityMetrics.observationCount >= 1)) {
          usedStrategyId = bestStrategy.id;
          usedCustomPrompts = true;
          serverLog(`meta: using library strategy ${bestStrategy.id} (avgScore=${bestStrategy.avgScore.toFixed(2)}, uses=${bestStrategy.useCount})`);
        } else {
          // Use delta-modified prompt (critique-don't-create)
          usedCustomPrompts = true;
          serverLog(`meta: applying delta to base template`);
        }

        // Layer 3: Thompson sampling for model routing
        if (session.taskConfig.models.length > 1) {
          const routed = thompsonSampleModel(metaFeatures.category, session.taskConfig.models);
          if (routed !== session.taskConfig.models[0]) {
            serverLog(`meta: Thompson sampling routed to ${routed} for category=${metaFeatures.category}`);
          }
        }
      }

      // Generate expert configs
      let expertConfigs = generateExpertConfigs(session.taskConfig, session.strategyAdaptations);

      if (usedCustomPrompts && metaFeatures) {
        const bestStrategy = usedStrategyId
          ? loadStrategyLibrary().find((s) => s.id === usedStrategyId)
          : null;

        expertConfigs = expertConfigs.map((cfg, i) => {
          const customCfg = { ...cfg };

          if (bestStrategy) {
            // Use proven strategy from library
            customCfg.solverPrompt = bestStrategy.solverPrompt;
            customCfg.feedbackPrompt = bestStrategy.feedbackPrompt;
            if (bestStrategy.configOverrides.temperature !== undefined)
              customCfg.temperature = bestStrategy.configOverrides.temperature!;
            if (bestStrategy.configOverrides.maxIterations !== undefined)
              customCfg.maxIterations = bestStrategy.configOverrides.maxIterations!;
            if (bestStrategy.configOverrides.reasoning !== undefined)
              customCfg.reasoning = bestStrategy.configOverrides.reasoning!;
          } else {
            // Apply delta to base template (critique-don't-create)
            const basePrompt = cfg.solverPrompt;

            // First apply meta-rules (Layer 2)
            const metaRuleDelta = applyMetaRules(metaFeatures.category, metaFeatures.difficulty);

            // Then apply the critic's delta (Layer 0)
            const combinedDelta: PromptDelta = {
              preProblemInsert: [
                metaRuleDelta.preProblemInsert,
                metaFeatures.promptDelta.preProblemInsert
              ].filter(Boolean).join('\n') || null,
              postProblemInsert: [
                metaRuleDelta.postProblemInsert,
                metaFeatures.promptDelta.postProblemInsert
              ].filter(Boolean).join('\n') || null,
              sectionReplacements: {
                ...metaRuleDelta.sectionReplacements,
                ...metaFeatures.promptDelta.sectionReplacements,
              },
              additionalExamples: [
                ...metaRuleDelta.additionalExamples,
                ...metaFeatures.promptDelta.additionalExamples,
              ],
              antiPatterns: [
                ...metaRuleDelta.antiPatterns,
                ...metaFeatures.promptDelta.antiPatterns,
              ],
            };

            customCfg.solverPrompt = applyPromptDelta(basePrompt, combinedDelta);

            // Apply analyzer suggestions for config
            if (metaFeatures.suggestedMaxIterations) {
              customCfg.maxIterations = metaFeatures.suggestedMaxIterations;
            }
            if (metaFeatures.suggestedTemperature) {
              customCfg.temperature = metaFeatures.suggestedTemperature;
            }
            if (metaFeatures.suggestedReasoning) {
              customCfg.reasoning = metaFeatures.suggestedReasoning;
            }
          }

          // Layer 3: Thompson model routing
          if (metaFeatures && session.taskConfig!.models.length > 1) {
            const routed = thompsonSampleModel(metaFeatures.category, session.taskConfig!.models);
            const resolved = resolveModel(routed);
            if (resolved) {
              const key = getApiKey(resolved.provider);
              if (key) customCfg.llmId = routed;
            }
          } else if (metaFeatures?.preferredModel && i === 0) {
            const resolved = resolveModel(metaFeatures.preferredModel);
            if (resolved) {
              const key = getApiKey(resolved.provider);
              if (key) customCfg.llmId = metaFeatures.preferredModel;
            }
          }

          return customCfg;
        });
      }

      // =================================================================
      // SOLVE with budget bandit (Layer 4: early stopping)
      // =================================================================
      const allResults: IterationResult[] = [];
      const budget = {
        maxCost,
        maxTime,
        startTime: solveStart,
        costSoFar: session.totalCost,
      };

      // Track per-expert iteration history for early stopping
      const expertHistories: Map<number, Array<{ score: number; passed: boolean }>> = new Map();
      for (let i = 0; i < expertConfigs.length; i++) {
        expertHistories.set(i, []);
      }

      // Run experts — with early stopping per expert
      const expertResults: IterationResult[][] = [];

      if (expertConfigs.length === 1) {
        // Single expert: run directly
        const results = await solveWithExpert(
          session, expertConfigs[0], 0,
          problem, trainInputs, trainOutputs, testInputs,
          session.taskConfig!.verification, session.taskConfig!.verifyCommand, budget
        );
        expertResults.push(results);
      } else {
        // Multiple experts: run in parallel with early stopping via custom wrapper
        const expertPromises = expertConfigs.map(async (cfg, i) => {
          // Run with reduced max iterations first, check progress, then continue
          const halfConfig = { ...cfg, maxIterations: Math.min(3, cfg.maxIterations) };
          const firstHalf = await solveWithExpert(
            session, halfConfig, i,
            problem, trainInputs, trainOutputs, testInputs,
            session.taskConfig!.verification, session.taskConfig!.verifyCommand, budget
          );

          // Check if we should continue or stop early
          const history = firstHalf.map((r) => ({ score: r.score, passed: r.passed }));
          const stopCheck = shouldStopEarly(history);

          if (stopCheck.stop) {
            serverLog(`budget-bandit: stopping expert ${i} early — ${stopCheck.reason}`);
            return firstHalf;
          }

          // Continue with remaining iterations
          const remainingIters = cfg.maxIterations - firstHalf.length;
          if (remainingIters <= 0) return firstHalf;

          const secondConfig = { ...cfg, maxIterations: remainingIters };
          const secondHalf = await solveWithExpert(
            session, secondConfig, i,
            problem, trainInputs, trainOutputs, testInputs,
            session.taskConfig!.verification, session.taskConfig!.verifyCommand, budget
          );

          return [...firstHalf, ...secondHalf];
        });

        const results = await Promise.all(expertPromises);
        for (const r of results) expertResults.push(r);
      }

      for (const results of expertResults) {
        allResults.push(...results);
        for (const r of results) {
          const hist = expertHistories.get(r.expertIndex) || [];
          hist.push({ score: r.score, passed: r.passed });
          expertHistories.set(r.expertIndex, hist);
        }
      }

      // Layer 4: Check if re-exploration is needed
      const reExploreCheck = shouldReExplore(expertResults, allResults.length);
      let reExploreNote = '';
      if (reExploreCheck.reExplore) {
        serverLog(`budget-bandit: ${reExploreCheck.reason}`);
        reExploreNote = `\n⚠️ ${reExploreCheck.reason}`;
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

      // Record prompt quality metrics (fast feedback)
      if (usedStrategyId && allResults.length > 0) {
        const firstIter = allResults.find((r) => r.iteration === 0);
        const codeParsed = allResults.some((r) => r.code && r.code.length > 0);
        const sandboxOk = allResults.some((r) => r.trainResults.some((tr) => tr.softScore > 0));
        recordPromptQuality(
          usedStrategyId,
          codeParsed,
          sandboxOk,
          firstIter?.score ?? 0
        );
      }

      // Record strategy result in library
      if (usedStrategyId) {
        recordStrategyUse(
          usedStrategyId, passed, bestScore, cost, totalDuration,
          session.taskConfig.models[0]
        );

        // Layer 3: Record model routing stats
        recordModelRoute(
          expertConfigs[0]?.llmId || session.taskConfig.models[0],
          metaFeatures?.category || 'other',
          passed, bestScore, cost, totalDuration
        );
      }

      // Save successful strategies to library
      if (metaFeatures && (passed || bestScore > 0.3)) {
        if (usedCustomPrompts && !usedStrategyId) {
          loadStrategyLibrary();
          const firstIter = allResults.find((r) => r.iteration === 0);
          const codeParsed = allResults.some((r) => r.code && r.code.length > 0);
          const sandboxOk = allResults.some((r) => r.trainResults.some((tr) => tr.softScore > 0));

          const newEntry: StrategyEntry = {
            id: randomBytes(4).toString('hex'),
            created: Date.now(),
            category: metaFeatures.category || session.taskConfig.type,
            solverPrompt: expertConfigs[0]?.solverPrompt || CODE_REASONING_SOLVER,
            feedbackPrompt: expertConfigs[0]?.feedbackPrompt || CODE_REASONING_FEEDBACK,
            configOverrides: {
              temperature: expertConfigs[0]?.temperature,
              maxIterations: expertConfigs[0]?.maxIterations,
              reasoning: expertConfigs[0]?.reasoning,
            },
            appliedDelta: metaFeatures.promptDelta,
            useCount: 1,
            successCount: passed ? 1 : 0,
            avgScore: bestScore,
            totalCost: cost,
            totalTime: totalDuration,
            testedModels: session.taskConfig.models,
            parentId: null,
            generation: 0,
            qualityMetrics: {
              codeParseRate: codeParsed ? 1 : 0,
              sandboxSuccessRate: sandboxOk ? 1 : 0,
              avgFirstIterationScore: firstIter?.score ?? 0,
              observationCount: 1,
            },
          };
          strategyLibrary.push(newEntry);
          saveStrategyLibrary();
          serverLog(`meta: saved new strategy ${newEntry.id} (category=${newEntry.category}, score=${bestScore.toFixed(2)})`);
        }

        // Record model routing even without strategy ID
        recordModelRoute(
          expertConfigs[0]?.llmId || session.taskConfig.models[0],
          metaFeatures.category,
          passed, bestScore, cost, totalDuration
        );
      }

      session.budget.costUsed += cost;
      session.budget.timeUsed += totalDuration;

      // Layer 5: Auto-trigger meta-improvement
      const autoResult = await maybeAutoImprove(session, session.taskConfig.models[0]);
      if (autoResult.improved) {
        serverLog(`auto-meta: triggered improvement — ${autoResult.reason} → child ${autoResult.childId}`);
      }

      let text = passed
        ? `✅ SOLVED in ${totalIters} iterations (${totalDuration.toFixed(1)}s, ${totalTokens} tokens, $${cost.toFixed(4)})`
        : `❌ Not fully solved after ${totalIters} iterations (best score: ${bestScore.toFixed(2)}, ${totalDuration.toFixed(1)}s, $${cost.toFixed(4)})`;

      text += `\nExperts: ${expertConfigs.length} | Models: ${expertConfigs.map(c => c.llmId).join(', ')}`;
      text += `\nPrompt tokens: ${session.totalPromptTokens} | Completion tokens: ${session.totalCompletionTokens} | Cost: $${cost.toFixed(4)}`;

      if (metaFeatures) {
        const delta = metaFeatures.promptDelta;
        text += `\n🧠 Meta: category=${metaFeatures.category} difficulty=${metaFeatures.difficulty.toFixed(1)}${usedStrategyId ? ` strategy=${usedStrategyId}` : ' (delta-modified)'} delta={pre:${!!delta.preProblemInsert} post:${!!delta.postProblemInsert} anti:${delta.antiPatterns.length}}`;
      }

      if (autoResult.improved) {
        text += `\n🔄 Auto-improved: ${autoResult.reason} → ${autoResult.childId}`;
      }

      if (reExploreNote) {
        text += reExploreNote;
      }

      if (session.strategyAdaptations.length > 0) {
        text += `\n🧠 ${session.strategyAdaptations.length} strategy adaptation(s) active`;
      }

      loadMetaRules();
      if (metaRules.length > 0) {
        text += `\n📐 ${metaRules.length} meta-rule(s) active`;
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
          autoImproved: autoResult.improved,
          autoImproveReason: autoResult.reason,
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

    case 'meta-analyze': {
      if (!session.taskConfig) {
        return { text: '❌ No session initialized. Call init first.', details: {} };
      }

      const analyzeProblemText = params.problem as string;
      if (!analyzeProblemText) {
        return { text: '❌ meta-analyze requires a problem.', details: {} };
      }

      // Use critique-don't-create: pass the base template to the critic
      const basePrompt = session.taskConfig.type === 'code-reasoning'
        ? CODE_REASONING_SOLVER
        : session.taskConfig.type === 'knowledge-extraction'
          ? KNOWLEDGE_EXTRACTION_SOLVER
          : HYBRID_SOLVER;

      const features = await critiqueAndAdapt(analyzeProblemText, basePrompt, session.taskConfig.models[0]);

      const delta = features.promptDelta;
      let text = `🧠 Problem Analysis (critique-don't-create):\n`;
      text += `Category: ${features.category}\n`;
      text += `Difficulty: ${features.difficulty.toFixed(1)}/1.0\n`;
      text += `Summary: ${features.summary}\n`;
      text += `Requires code: ${features.requiresCode ? '✅' : '❌'}\n`;
      text += `Key patterns: ${features.keyPatterns.join(', ') || 'none detected'}\n`;
      text += `Suggested approach: ${features.suggestedApproach}\n`;
      text += `Suggested: ${features.suggestedMaxIterations} iters, temp=${features.suggestedTemperature}, reasoning=${features.suggestedReasoning}\n`;
      if (features.preferredModel) text += `Preferred model: ${features.preferredModel}\n`;
      if (features.subQuestions.length > 0) {
        text += `Sub-questions:\n`;
        for (const sq of features.subQuestions) {
          text += `  • ${sq}\n`;
        }
      }
      text += `\n--- Prompt Delta ---`;
      text += `\nPre-problem insert: ${delta.preProblemInsert ? '✅ ' + delta.preProblemInsert.slice(0, 200) : '❌ none'}`;
      text += `\nPost-problem insert: ${delta.postProblemInsert ? '✅ ' + delta.postProblemInsert.slice(0, 200) : '❌ none'}`;
      text += `\nAnti-patterns (${delta.antiPatterns.length}): ${delta.antiPatterns.join(', ') || 'none'}`;
      text += `\nAdditional examples: ${delta.additionalExamples.length}`;

      // Show what the modified prompt would look like
      const modifiedPrompt = applyPromptDelta(basePrompt, delta);
      text += `\n\n--- Modified Prompt Preview (${modifiedPrompt.length} chars) ---\n${modifiedPrompt.slice(0, 500)}...`;

      // Check library for matching strategy
      const bestStrategy = findBestStrategy(features.category, session.taskConfig.models);
      if (bestStrategy) {
        text += `\n\n📚 Best library strategy: ${bestStrategy.id} (score=${bestStrategy.avgScore.toFixed(2)}, uses=${bestStrategy.useCount}, gen=${bestStrategy.generation})`;
      } else {
        text += `\n\n📚 No matching strategy in library.`;
      }

      // Show applicable meta-rules
      const metaRuleDelta = applyMetaRules(features.category, features.difficulty);
      if (metaRuleDelta.preProblemInsert || metaRuleDelta.antiPatterns.length > 0) {
        text += `\n📐 Applicable meta-rules: pre=${!!metaRuleDelta.preProblemInsert} anti=${metaRuleDelta.antiPatterns.length}`;
      }

      return { text, details: { features, bestStrategyId: bestStrategy?.id || null } };
    }

    case 'meta-improve': {
      loadStrategyLibrary();
      if (strategyLibrary.length === 0) {
        return { text: '📚 No strategies in library to improve. Solve some problems with meta=true first.', details: {} };
      }

      if (!session.taskConfig) {
        return { text: '❌ No session initialized. Call init first.', details: {} };
      }

      // Pick the strategy with lowest ROI to improve (worst first)
      const candidates = [...strategyLibrary]
        .filter((s) => s.useCount >= 1)
        .sort((a, b) => {
          const roiA = (a.successCount / Math.max(a.useCount, 1)) * a.avgScore;
          const roiB = (b.successCount / Math.max(b.useCount, 1)) * b.avgScore;
          return roiA - roiB; // worst first
        });

      const target = candidates[0];
      if (!target) {
        return { text: '📚 No improvable strategies found.', details: {} };
      }

      const improved = await improveStrategy(target, session.problemHistory, session.taskConfig.models[0]);

      if (!improved) {
        return { text: `❌ Failed to improve strategy ${target.id}. LLM didn't return valid JSON.`, details: { targetId: target.id } };
      }

      strategyLibrary.push(improved);
      saveStrategyLibrary();

      // Extract meta-rules from the evolution
      const rules = await extractMetaRules(target, improved, session.taskConfig.models[0]);
      if (rules.length > 0) {
        metaRules.push(...rules);
        saveMetaRules();
      }

      let text = `🧠 Strategy Evolution:\n`;
      text += `Parent: ${target.id} (gen=${target.generation}, score=${target.avgScore.toFixed(2)}, uses=${target.useCount})\n`;
      text += `Child:  ${improved.id} (gen=${improved.generation})\n`;
      text += `\nNew solver prompt (${improved.solverPrompt.length} chars):\n${improved.solverPrompt.slice(0, 300)}...\n`;
      text += `\nNew config overrides: ${JSON.stringify(improved.configOverrides)}`;

      if (improved.appliedDelta) {
        text += `\n\nApplied delta: pre=${!!improved.appliedDelta.preProblemInsert} post=${!!improved.appliedDelta.postProblemInsert} anti=${improved.appliedDelta.antiPatterns.length}`;
      }

      if (rules.length > 0) {
        text += `\n📐 Extracted ${rules.length} meta-rule(s):`;
        for (const r of rules) {
          text += `\n  • [${r.id}] ${r.principle}`;
        }
      }

      return { text, details: { parentId: target.id, childId: improved.id, generation: improved.generation, extractedRules: rules.length } };
    }

    case 'strategies': {
      loadStrategyLibrary();

      if (strategyLibrary.length === 0) {
        return { text: '📚 Strategy library is empty. Solve problems with meta=true to build it.', details: { strategies: [] } };
      }

      const lines = strategyLibrary
        .sort((a, b) => {
          const qualA = 1 + (a.qualityMetrics.codeParseRate * 0.5 + a.qualityMetrics.sandboxSuccessRate * 0.3);
          const qualB = 1 + (b.qualityMetrics.codeParseRate * 0.5 + b.qualityMetrics.sandboxSuccessRate * 0.3);
          const roiA = (a.successCount / Math.max(a.useCount, 1)) * a.avgScore / Math.max(a.totalCost, 0.001) * qualA;
          const roiB = (b.successCount / Math.max(b.useCount, 1)) * b.avgScore / Math.max(b.totalCost, 0.001) * qualB;
          return roiB - roiA;
        })
        .map((s, i) => {
          const successRate = s.useCount > 0 ? (s.successCount / s.useCount * 100).toFixed(0) : '0';
          const roi = (s.successCount / Math.max(s.useCount, 1)) * s.avgScore / Math.max(s.totalCost, 0.001);
          const qual = `parse=${(s.qualityMetrics.codeParseRate * 100).toFixed(0)}% sandbox=${(s.qualityMetrics.sandboxSuccessRate * 100).toFixed(0)}%`;
          return `${i + 1}. [${s.id}] cat=${s.category} gen=${s.generation} score=${s.avgScore.toFixed(2)} win=${successRate}% uses=${s.useCount} cost=$${s.totalCost.toFixed(3)} ROI=${roi.toFixed(1)} qual={${qual}}${s.parentId ? ` parent=${s.parentId}` : ''}`;
        });

      return {
        text: `📚 Strategy Library (${strategyLibrary.length} strategies):\n${lines.join('\n')}`,
        details: { strategies: strategyLibrary },
      };
    }

    case 'meta-rules': {
      loadMetaRules();

      if (metaRules.length === 0) {
        return { text: '📐 No meta-rules yet. Strategies need to evolve first (use meta-improve or solve with meta=true).', details: { rules: [] } };
      }

      const lines = metaRules
        .sort((a, b) => b.improvementCount - a.improvementCount)
        .map((r, i) => {
          const successRate = r.testCount > 0 ? (r.improvementCount / r.testCount * 100).toFixed(0) : '?';
          return `${i + 1}. [${r.id}] "${r.principle}"\n   validated=${r.validatedCategories.join(',') || 'untested'} improvements=${r.improvementCount}/${r.testCount} (${successRate}%)`;
        });

      return {
        text: `📐 Meta-Rules (${metaRules.length} rules):\n${lines.join('\n')}`,
        details: { rules: metaRules },
      };
    }

    case 'model-routes': {
      loadModelRoutes();

      if (modelRouteStats.length === 0) {
        return { text: '🚦 No model routing data yet. Solve problems with meta=true to build it.', details: { routes: [] } };
      }

      const lines = modelRouteStats
        .sort((a, b) => b.avgScore - a.avgScore)
        .map((m, i) => {
          const successRate = m.uses > 0 ? (m.successes / m.uses * 100).toFixed(0) : '?';
          return `${i + 1}. ${m.modelId} (${m.category}): score=${m.avgScore.toFixed(2)} win=${successRate}% uses=${m.uses} cost=$${m.avgCost.toFixed(4)} time=${m.avgTime.toFixed(1)}s`;
        });

      return {
        text: `🚦 Model Routes (${modelRouteStats.length} entries):\n${lines.join('\n')}`,
        details: { routes: modelRouteStats },
      };
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

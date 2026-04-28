# pi-reason-harness

Recursive self-improving reasoning harness — the proprietary layer rebuilt from first principles.

Builds task-specific reasoning strategies on top of any LLM by running iterative solve-verify-feedback loops with multi-expert ensembling, voting, and a **5-layer meta-system** that discovers, adapts, and evolves strategies autonomously.

**JS-exclusive** — LLM calls go through pi's native LLM infrastructure (`@mariozechner/pi-ai`). Code sandbox uses Node's `vm` module. Zero Python dependency.

## How It Works

The core insight (from first-principles analysis of SOTA reasoning systems): **LLMs are knowledge stores that require intelligent probing strategies to extract reliable answers.** The harness layer (open-source) iteratively generates, verifies, and refines. The meta-system layer (proprietary, rebuilt here) discovers and evolves the strategies themselves.

### The 5-Layer Meta-System

| Layer | Name | What it does |
|-------|------|-------------|
| 0 | **Problem Critic** | Inspects problems, proposes *targeted deltas* to proven templates (not writing from scratch) |
| 1 | **Strategy Library** | Persistent store of proven strategies with ROI + quality metrics |
| 2 | **Meta-Rule Engine** | Extracts cross-strategy principles that compound over time |
| 3 | **Model Router** | Thompson sampling for intelligent model selection per category |
| 4 | **Budget Bandit** | Early stopping, budget reallocation, re-exploration when stuck |
| 5 | **Auto-Trigger** | Self-improvement runs automatically (on success rate drops, new categories, periodic) |

### Layer 0: Critique, Don't Create

The key insight: asking an LLM to write a solver prompt from scratch produces terse, bad prompts. Instead, the critic receives the **proven default template** and proposes **targeted modifications** (deltas):

- `preProblemInsert` — Strategy hints inserted before the problem
- `postProblemInsert` — Format enforcement after the problem
- `antiPatterns` — Things the solver should NOT do
- `additionalExamples` — Worked examples for this problem type
- `sectionReplacements` — Replacement for specific template sections

This is code review, not code writing. The delta gets applied to the base template via `applyPromptDelta()`, producing a modified prompt that keeps what works and adds what's needed.

### Layer 2: Meta-Rules Compound

When a child strategy outperforms its parent, the meta-rule extractor analyzes the evolution and extracts **1-3 generalizable principles**. These principles apply to OTHER problem categories, not just the source:

> *"Adding worked examples improved grid-transformation by 30% → try it on pattern-completion"*

Rules are filtered by category match, freshness (7-day stale), and positive evidence. Top 3 are merged into solve deltas. Rules get validated every time they're tested, compounding improvements over time.

### Layer 3: Thompson Sampling for Model Routing

Each model is an arm in a multi-armed bandit. Beta(α, β) sampling with Laplace smoothing picks the model most likely to succeed for a given category, with exploration bonus for under-tested models. Cost efficiency adjustment penalizes expensive models.

### Layer 4: Budget Bandit

- **Early stopping**: Stop experts stuck at same low score for 3+ iterations
- **Decreasing score detection**: Stop when score trends downward over 4+ iterations
- **Re-exploration**: When ALL experts are stuck at 0, trigger a different approach

### Layer 5: Auto-Trigger

Meta-improvement runs automatically when:
- Recent success rate drops below half of overall
- A new problem category is encountered
- Every 5th problem (periodic)

Each auto-trigger runs strategy improvement + meta-rule extraction.

### The Harness Layer

Below the meta-system, the harness implements the iterative solve loop:

1. **Iterative solve-verify-feedback loops** — Generate code, sandbox-execute, build detailed feedback, inject into next iteration
2. **Multi-expert ensembling** — Run parallel experts with different seeds/models
3. **Voting and ranking** — Group by output, rank by vote count, diversity-first
4. **Poetiq-parity feedback** — Element-by-element diff grids (`prediction/correct` format), shape mismatch detection, bad-JSON diagnostics
5. **Poetiq-parity formatting** — `<Diagram>` text with Fisher-Yates shuffle per iteration
6. **Self-audit verification** — LLM checks its own answers for accuracy, consistency, completeness
7. **Budget tracking** — Per-problem cost/time limits, session-level cumulative tracking

### Task Types

| Type | Strategy | Verification |
|------|----------|--------------|
| `code-reasoning` | Generate JavaScript code → sandbox execute → verify against examples → feedback loop | Sandbox (default) or external |
| `knowledge-extraction` | Chain-of-questions probing → self-audit → confidence bucketing | Self-audit (recommended) |
| `hybrid` | Decide per-problem: code or direct answer → verify → feedback | Any method |

### Verification Methods

| Method | How it works |
|--------|-------------|
| `sandbox` | Execute JS code in Node vm sandbox, compare output to expected |
| `self-audit` | LLM checks its own answer for accuracy, consistency, completeness |
| `external` | Run a custom shell command to verify |
| `none` | No verification, just record the answer |

### Persistent Data

The meta-system persists across server restarts at `~/.pi-reason-harness/`:

| File | Contents |
|------|----------|
| `strategies.json` | Strategy library with ROI, quality metrics, lineage |
| `meta-rules.json` | Cross-strategy principles with validation stats |
| `model-routes.json` | Per model×category routing stats |

## Quick Start

```bash
# Initialize a reasoning session
pi-reason-harness init --name "ARC solver" --type code-reasoning \
  --models '["anthropic/claude-sonnet-4-5","openai/gpt-4o"]' --num-experts 2

# Solve with the full meta-system pipeline
pi-reason-harness solve --meta --problem "Transform the grid..." \
  --train-inputs '[[1,2],[3,4]]' \
  --train-outputs '[[4,3],[2,1]]' \
  --test-inputs '[[5,6]]'

# Analyze a problem without solving
pi-reason-harness meta-analyze --problem "Rotate a 2x2 grid 90 degrees clockwise"

# Check the strategy library
pi-reason-harness strategies

# Check meta-rules
pi-reason-harness meta-rules

# Manually trigger strategy improvement
pi-reason-harness meta-improve

# Check model routing stats
pi-reason-harness model-routes

# Check what the system has learned
pi-reason-harness learn

# Budget-constrained solving
pi-reason-harness init --name "budget test" --type code-reasoning --max-cost 0.50 --max-time 120
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  META-SYSTEM V2 (5 layers — the proprietary layer)      │
│                                                         │
│  Layer 0: Problem Critic                                │
│    - critiqueAndAdapt(): receives base template,        │
│      proposes PromptDelta (not writing from scratch)     │
│    - applyPromptDelta(): base + delta = modified prompt  │
│                                                         │
│  Layer 1: Strategy Library                               │
│    - Persistent strategies with ROI + quality metrics    │
│    - findBestStrategy(): retrieves highest-ROI match     │
│    - recordPromptQuality(): fast feedback signals        │
│                                                         │
│  Layer 2: Meta-Rule Engine                               │
│    - extractMetaRules(): parent→child evolution → rules  │
│    - applyMetaRules(): merges relevant rules into delta  │
│    - validateMetaRule(): tracks improvement/test counts  │
│                                                         │
│  Layer 3: Model Router (Thompson Sampling)               │
│    - thompsonSampleModel(): Beta(α,β) sampling          │
│    - Cost efficiency adjustment                          │
│    - recordModelRoute(): per model×category stats        │
│                                                         │
│  Layer 4: Budget Bandit                                  │
│    - shouldStopEarly(): stuck at same low score          │
│    - shouldReExplore(): all experts stuck at 0           │
│                                                         │
│  Layer 5: Auto-Trigger                                   │
│    - shouldAutoImprove(): success rate drop, new cat,   │
│      periodic (every 5 problems)                         │
│    - maybeAutoImprove(): runs improvement + extraction   │
└───────────────────────┬─────────────────────────────────┘
                        │ generates (with deltas + rules)
                        ▼
┌─────────────────────────────────────────────────────────┐
│  HARNESS (iterative solve-verify-feedback)              │
│                                                         │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐          │
│  │ Expert 1  │  │ Expert 2  │  │ Expert N  │          │
│  │ (pi-ai    │  │ (pi-ai    │  │ (pi-ai    │          │
│  │ LLM call  │  │ LLM call  │  │ LLM call  │          │
│  │ + sandbox │  │ + sandbox │  │ + sandbox │          │
│  │ + verify  │  │ + verify  │  │ + verify  │          │
│  │ + feedback│  │ + feedback│  │ + feedback│          │
│  │ loop)     │  │ loop)     │  │ loop)     │          │
│  └────┬──────┘  └───┬───────┘  └──┬────────┘          │
│       └─────────────┼─────────────┘                    │
│                     ▼                                   │
│              VOTING / RANKING                           │
│              (group by output,                          │
│               most-voted wins,                          │
│               diversity-first)                          │
│                     │                                   │
│                     ▼                                   │
│              LEARN & ADAPT                              │
│              (update strategies, record routes,         │
│               extract rules, auto-improve)              │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
                   Best Solution + Insights
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `init` | Initialize session with task config, models, verification |
| `solve` | Run iterative solve-verify-feedback loop |
| `status` | Show session state, budget, learned adaptations |
| `results` | Show iteration results |
| `learn` | Inspect strategy adaptations |
| `reset-learn` | Clear learned strategies |
| `clear` | Clear session |
| `meta-analyze` | Analyze a problem with the critic (no solving) |
| `meta-improve` | Manually trigger strategy evolution + rule extraction |
| `strategies` | List strategy library with ROI + quality metrics |
| `meta-rules` | List meta-rules with validation stats |
| `model-routes` | List model routing stats per model×category |

### init flags

`--name`, `--type`, `--models`, `--num-experts`, `--verification`, `--verify-command`, `--max-cost`, `--max-time`

### solve flags

`--problem`, `--train-inputs`, `--train-outputs`, `--test-inputs`, `--meta` / `-m`

## LLM Integration

The harness uses `@mariozechner/pi-ai` for all LLM calls. Models are specified in `provider/model` format (e.g., `anthropic/claude-sonnet-4-5`, `openai/gpt-4o`). API keys are resolved from the same environment variables pi uses:

- `ANTHROPIC_API_KEY` — Anthropic models
- `OPENAI_API_KEY` — OpenAI models
- `GEMINI_API_KEY` — Google models
- `GROQ_API_KEY` — Groq models
- etc.

No additional setup required — if pi can call the model, so can the harness.

## Tests

```bash
npm test
```

69 tests covering: vm sandbox, formatProblem, arrayDiff, buildDetailedFeedback, PromptDelta application, budget bandit (early stopping, re-exploration), Thompson sampling (Beta distribution), meta-rule engine (validation, filtering), prompt quality metrics.

## License

MIT

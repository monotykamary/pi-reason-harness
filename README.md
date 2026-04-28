# pi-reason-harness

Recursive self-improving reasoning harness — the proprietary layer rebuilt from first principles.

Builds task-specific reasoning strategies on top of any LLM by running iterative solve-verify-feedback loops with multi-expert ensembling, voting, and a **13-layer meta-system** that discovers, adapts, evolves, transfers, and validates strategies autonomously.

**JS-exclusive** — LLM calls go through pi's native LLM infrastructure (`@mariozechner/pi-ai`). Code sandbox uses Node's `vm` module. Zero Python dependency.

## How It Works

The core insight (from first-principles analysis of SOTA reasoning systems): **LLMs are knowledge stores that require intelligent probing strategies to extract reliable answers.** The harness layer (open-source) iteratively generates, verifies, and refines. The meta-system layer (proprietary, rebuilt here) discovers and evolves the strategies themselves.

### The 13-Layer Meta-System

| Layer | Name | What it does |
|-------|------|-------------|
| 0 | **Problem Critic** | Inspects problems, proposes *targeted deltas* to proven templates (not writing from scratch) |
| 1 | **Strategy Library** | Persistent store of proven strategies with ROI + quality metrics |
| 2 | **Meta-Rule Engine** | Extracts cross-strategy principles that compound over time |
| 3 | **Model Router** | Thompson sampling for intelligent model selection per category |
| 4 | **Budget Bandit** | Early stopping, budget reallocation, re-exploration when stuck |
| 5 | **Auto-Trigger** | Self-improvement runs automatically (on success rate drops, new categories, periodic) |
| 6 | **Recursive Harness Generation** | Generates entire solve approach configurations (the "solver of solvers") |
| 7 | **Ensemble Diversification** | Each expert uses a fundamentally different approach strategy |
| 8 | **Sub-problem Decomposition** | Break hard problems into sub-problems, solve independently, combine |
| 9 | **Budget Optimization** | Marginal ROI estimation, reallocate iterations to high-ROI experts |
| 10 | **Cross-Domain Transfer** | Transfer proven strategies across analogous categories automatically |
| 11 | **Confidence-Weighted Voting** | Weight votes by self-assessed quality, not just output match |
| 12 | **Progressive Difficulty** | Train on easiest examples first, build up to harder ones |
| 13 | **Auto-Transfer** | Automatically transfer strategies when new categories are encountered |

### Layer 6: Recursive Harness Generation — the "solver of solvers"

The biggest gap with Poetiq: their open-source code shows ONE harness configuration with fixed prompts. Their blog results prove they generate MULTIPLE different configurations per problem type.

A **HarnessSpec** defines a complete solve approach:
- **Approach type**: code-sandbox, decomposition, chain-of-questions, analogy, counter-factual, exhaustive-search, code-direct
- **Solver/feedback prompts**: Full templates with `$$problem$$` placeholders
- **Config overrides**: Temperature, iterations, reasoning level
- **Decomposition config**: Max sub-problems, depth, combine strategy
- **Validation data**: Score on held-out data, production stats

The system generates multiple specs per problem, validates them, and evolves them over time.

### Layer 7: Ensemble Diversification

Instead of N experts with the same prompt (just different seeds/models), each expert uses a **fundamentally different approach**:

| Expert | Approach | When to use |
|--------|----------|-------------|
| 1 | code-sandbox | Grid/array problems — generate code, execute, verify |
| 2 | decomposition | Complex problems — break into sub-problems |
| 3 | analogy | Hard problems — solve simpler version first |
| 4 | chain-of-questions | Knowledge tasks — hierarchical probing |
| 5 | counter-factual | Stubborn problems — generate wrong solutions, invert |
| 6 | exhaustive-search | Small search spaces — enumerate, filter |

Each approach has its own specialized prompt template.

### Layer 8: Sub-problem Decomposition

For hard problems that resist direct solving, the decomposer breaks them into independent sub-problems:
1. LLM analyzes the problem and proposes 2-4 sub-problems
2. Each sub-problem is solved independently
3. Sub-solutions are combined (sequentially, in parallel, or hierarchically)

Example: "Rotate 90° clockwise" → Sub-problem 1: "Transpose the grid" → Sub-problem 2: "Reverse each row"

### Layer 9: Budget Optimization via Marginal ROI

Not just "stop when stuck" but **"spend where ROI is highest"**:
- Estimate marginal ROI per expert based on recent improvement rate
- Reallocate remaining iterations to experts with highest expected improvement
- Phase-based execution: run all experts for half-iterations, then reallocate

### Layer 10-13: Cross-Domain Transfer + Auto-Transfer

When a strategy works in one domain, the system **automatically transfers** it to analogous domains:
- Category similarity map: grid-transformation ↔ pattern-completion ↔ spatial-reasoning
- Transfer adapts domain-specific parts while keeping universal insights
- Auto-triggered when a new category is encountered with no existing strategies
- Creates both a strategy entry and a harness spec for the new category

### Layer 11: Confidence-Weighted Voting

Voting is weighted by **self-assessed quality**:
- Solutions that pass in fewer iterations count more (efficiency bonus)
- Solutions with high soft scores count more (partial accuracy bonus)
- Failed solutions grouped by output similarity, ranked by total confidence

### Layer 12: Progressive Difficulty

Training examples are ordered from **easiest to hardest**:
- Difficulty proxy: grid size + unique value count + input/output asymmetry
- The solver sees simpler patterns first, building up to complex ones
- Mirrors Poetiq's per-iteration shuffle but with intelligence

### Layers 0-5: Core Meta-System

These were implemented in the previous iteration and remain the foundation:

- **Layer 0: Critique, Don't Create** — The critic receives proven templates and proposes targeted deltas (insertions, anti-patterns, examples). This is code review, not writing from zero.
- **Layer 2: Meta-Rules Compound** — When a child strategy outperforms its parent, generalizable principles are extracted and applied to other categories.
- **Layer 3: Thompson Sampling** — Beta(α,β) sampling with Laplace smoothing picks the best model per category.
- **Layer 4: Budget Bandit** — Early stopping, re-exploration when all experts fail.
- **Layer 5: Auto-Trigger** — Runs automatically on success rate drops, new categories, and every 5th problem.

### The Harness Layer

Below the meta-system, the harness implements the iterative solve loop with Poetiq-parity features:

1. **Iterative solve-verify-feedback loops** — Generate code, sandbox-execute, build detailed feedback
2. **Multi-expert ensembling** — Parallel experts with diverse approaches
3. **Confidence-weighted voting** — Group by output, rank by confidence
4. **Poetiq-parity feedback** — Element-by-element diff grids, shape mismatch detection
5. **Poetiq-parity formatting** — `<Diagram>` text with Fisher-Yates shuffle
6. **Self-audit verification** — LLM checks its own answers
7. **Budget tracking** — Per-problem cost/time limits

### Task Types

| Type | Strategy | Verification |
|------|----------|--------------|
| `code-reasoning` | Generate JavaScript code → sandbox execute → verify against examples → feedback loop | Sandbox (default) or external |
| `knowledge-extraction` | Chain-of-questions probing → self-audit → confidence bucketing | Self-audit (recommended) |
| `hybrid` | Decide per-problem: code or direct answer → verify → feedback | Any method |

### Approach Types (for ensemble diversification)

| Approach | Description | Best for |
|----------|-------------|----------|
| `code-sandbox` | Generate JS code, execute in sandbox, verify output | Grid/array transformations |
| `code-direct` | Generate code, extract answer without execution | Computation-heavy |
| `decomposition` | Break into sub-problems, solve each, combine | Multi-step problems |
| `chain-of-questions` | Hierarchical probing from broad to specific | Knowledge questions |
| `analogy` | Solve simpler version first, then scale up | Hard spatial problems |
| `counter-factual` | Generate wrong solutions, analyze failures, invert | Stubborn problems |
| `exhaustive-search` | Enumerate possibilities, filter by constraints | Small search spaces |

### Persistent Data

The meta-system persists across server restarts at `~/.pi-reason-harness/`:

| File | Contents |
|------|----------|
| `strategies.json` | Strategy library with ROI, quality metrics, lineage |
| `meta-rules.json` | Cross-strategy principles with validation stats |
| `model-routes.json` | Per model×category routing stats |
| `harness-specs.json` | Complete harness specifications per category×approach |

## Quick Start

```bash
# Initialize a reasoning session
pi-reason-harness init --name "ARC solver" --type code-reasoning \
  --models '["anthropic/claude-sonnet-4-5","openai/gpt-4o"]' --num-experts 3

# Solve with the full 13-layer meta-system pipeline
pi-reason-harness solve --meta --problem "Transform the grid..." \
  --train-inputs '[[1,2],[3,4]]' \
  --train-outputs '[[4,3],[2,1]]' \
  --test-inputs '[[5,6]]'

# Analyze a problem without solving
pi-reason-harness meta-analyze --problem "Rotate a 2x2 grid 90 degrees clockwise"

# Decompose a hard problem into sub-problems
pi-reason-harness decompose --problem "Rotate a 3x3 grid 90 degrees clockwise. Input: [[1,2,3],[4,5,6],[7,8,9]]"

# Check harness specs
pi-reason-harness harness-specs

# Evolve the worst-performing spec
pi-reason-harness evolve-harness

# Check the strategy library
pi-reason-harness strategies

# Transfer a strategy from grid-transformation to pattern-completion
pi-reason-harness transfer --source-category grid-transformation --target-category pattern-completion

# Check meta-rules
pi-reason-harness meta-rules

# Check model routing stats
pi-reason-harness model-routes
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  META-SYSTEM V3 (13 layers — the proprietary layer)      │
│                                                         │
│  Layer 0: Problem Critic (critique-don't-create)       │
│  Layer 1: Strategy Library (ROI + quality metrics)      │
│  Layer 2: Meta-Rule Engine (cross-strategy principles)  │
│  Layer 3: Model Router (Thompson sampling)              │
│  Layer 4: Budget Bandit (early stopping + re-explore)   │
│  Layer 5: Auto-Trigger (self-improving loop)            │
│  Layer 6: Recursive Harness Generation (solver-of-solvers)│
│  Layer 7: Ensemble Diversification (different approaches)│
│  Layer 8: Sub-problem Decomposition (break & combine)   │
│  Layer 9: Budget Optimization (marginal ROI realloc)   │
│  Layer 10: Cross-Domain Transfer (analogous categories) │
│  Layer 11: Confidence-Weighted Voting (quality-ranked)  │
│  Layer 12: Progressive Difficulty (easiest-first)       │
│  Layer 13: Auto-Transfer (new category handling)        │
└───────────────────────┬─────────────────────────────────┘
                        │ generates (with deltas + rules + specs)
                        ▼
┌─────────────────────────────────────────────────────────┐
│  HARNESS (iterative solve-verify-feedback)              │
│                                                         │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐          │
│  │ Expert 1  │  │ Expert 2  │  │ Expert N  │          │
│  │ code-sandbox│ │ decomposition│ │ analogy   │          │
│  │ (pi-ai    │  │ (pi-ai    │  │ (pi-ai    │          │
│  │  LLM call │  │  LLM call │  │  LLM call │          │
│  │  + sandbox│  │  + sub-   │  │  + analogy │          │
│  │  + verify │  │  solve    │  │  + verify  │          │
│  │  + feedback│ │  + combine│  │  + feedback│          │
│  └────┬──────┘  └───┬───────┘  └──┬────────┘          │
│       └─────────────┼─────────────┘                    │
│                     ▼                                   │
│     CONFIDENCE-WEIGHTED VOTING                         │
│     (group by output, rank by confidence)              │
│                     │                                   │
│                     ▼                                   │
│     LEARN + ADAPT + EVOLVE + TRANSFER                  │
│     (update strategies, extract rules, evolve specs,    │
│      transfer to new categories, auto-improve)          │
└─────────────────────────────────────────────────────────┘
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
| `harness-specs` | List harness specifications with validation + production stats |
| `evolve-harness` | Evolve the worst-performing harness spec |
| `transfer` | Transfer strategy from one category to another |
| `decompose` | Decompose a problem into sub-problems |

### init flags

`--name`, `--type`, `--models`, `--num-experts`, `--verification`, `--verify-command`, `--max-cost`, `--max-time`

### solve flags

`--problem`, `--train-inputs`, `--train-outputs`, `--test-inputs`, `--meta` / `-m`

### transfer flags

`--source-category`, `--target-category`

### decompose flags

`--problem`

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

83 tests covering: vm sandbox, formatProblem, arrayDiff, buildDetailedFeedback, PromptDelta application, budget bandit (early stopping, re-exploration), Thompson sampling (Beta distribution), meta-rule engine (validation, filtering), prompt quality metrics, harness spec generation, ensemble diversification, budget optimization (marginal ROI), cross-domain transfer, confidence-weighted voting, progressive difficulty, decomposition.

## License

MIT

---
name: reason-harness
description: Run iterative solve-verify-feedback reasoning loops on top of any LLM with a 13-layer self-improving meta-system. Use when you need to solve problems requiring code generation with verification, knowledge extraction through chain-of-questions probing, or hybrid reasoning tasks. The system learns from past problems, adapts strategy, generates diverse approaches, decomposes hard problems, and transfers insights across domains. Works by generating candidate solutions, executing them in sandboxes or self-auditing, building feedback from failures, and confidence-weighted voting across parallel experts with different strategies.
---

# Reason Harness

Recursive self-improving reasoning harness with 13 meta-system layers. Generate solutions, test them, learn from failures, evolve strategies, decompose hard problems, transfer across domains, and improve autonomously.

The `pi-reason-harness` CLI auto-spawns a long-lived harness server on first use. Every call dispatches an action to the harness, which holds session state across calls. The system uses pi's own LLM infrastructure (pi-ai) and runs code in Node's vm sandbox — zero Python dependency.

## Setup (once, first use)

Symlink the CLI if not on PATH:

```bash
# macOS (Apple Silicon + Homebrew)
command -v pi-reason-harness >/dev/null || ln -sf <skill-dir>/harness/cli.ts /opt/homebrew/bin/pi-reason-harness

# macOS (Intel) / most Linux
command -v pi-reason-harness >/dev/null || ln -sf <skill-dir>/harness/cli.ts /usr/local/bin/pi-reason-harness
```

## How to use

```bash
# Initialize a reasoning session
pi-reason-harness init --name "ARC solver" --type code-reasoning \
  --models '["anthropic/claude-sonnet-4-5","openai/gpt-4o"]' --num-experts 3

# Solve with the full 13-layer meta-system pipeline
pi-reason-harness solve --meta --problem "Transform the grid by..." \
  --train-inputs '[[1,2],[3,4]]' \
  --train-outputs '[[4,3],[2,1]]' \
  --test-inputs '[[5,6]]'

# Decompose a hard problem into sub-problems
pi-reason-harness decompose --problem "Rotate a 3x3 grid 90 degrees clockwise. Input: [[1,2,3],[4,5,6],[7,8,9]]"

# Analyze a problem without solving
pi-reason-harness meta-analyze --problem "Rotate a 2x2 grid 90 degrees clockwise"

# Check harness specs (different solve approaches)
pi-reason-harness harness-specs

# Evolve the worst-performing harness spec
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

## CLI commands

| Command | Behavior |
|---|---|
| `pi-reason-harness init` | Initialize a reasoning session with task config |
| `pi-reason-harness solve` | Run the iterative solve-verify-feedback loop |
| `pi-reason-harness status` | Show session state, budget, and learned adaptations |
| `pi-reason-harness results` | Show iteration results |
| `pi-reason-harness learn` | Inspect strategy adaptations learned from past problems |
| `pi-reason-harness reset-learn` | Clear all learned adaptations and history |
| `pi-reason-harness clear` | Clear session state |
| `pi-reason-harness meta-analyze` | Analyze a problem with the critic (no solving) |
| `pi-reason-harness meta-improve` | Evolve worst-performing strategy + extract rules |
| `pi-reason-harness strategies` | List strategy library with ROI + quality metrics |
| `pi-reason-harness meta-rules` | List cross-strategy principles with validation stats |
| `pi-reason-harness model-routes` | List model routing stats per model×category |
| `pi-reason-harness harness-specs` | List harness specifications with stats |
| `pi-reason-harness evolve-harness` | Evolve the worst-performing spec |
| `pi-reason-harness transfer` | Transfer strategy between categories |
| `pi-reason-harness decompose` | Decompose a problem into sub-problems |
| `pi-reason-harness synth-prompts` | List synthesized prompts |
| `pi-reason-harness meta-harnesses` | List meta-harnesses |
| `pi-reason-harness generate-meta-harness` | Generate new approach type |

## The 16-Layer Meta-System

| Layer | Name | What it does |
|-------|------|-------------|
| 0 | Problem Critic | Proposes targeted deltas to proven templates |
| 1 | Strategy Library | Persistent store with ROI + quality metrics |
| 2 | Meta-Rule Engine | Cross-strategy principles that compound |
| 3 | Model Router | Thompson sampling for model selection |
| 4 | Budget Bandit | Early stopping + re-exploration |
| 5 | Auto-Trigger | Self-improvement runs automatically |
| 6 | Recursive Harness Generation | Generates entire approach configurations |
| 7 | Ensemble Diversification | Different approach per expert |
| 8 | Sub-problem Decomposition | Break hard problems, solve, combine |
| 9 | Budget Optimization | Marginal ROI reallocation |
| 10 | Cross-Domain Transfer | Transfer strategies across categories |
| 11 | Confidence-Weighted Voting | Weight votes by quality |
| 12 | Progressive Difficulty | Easiest examples first |
| 14 | Per-Problem Prompt Synthesis | Generate + validate prompts for novel types |
| 15 | Meta-Meta Level | Harness-of-harnesses — new approach types |
| 16 | Gradient-Based Budget Optimization | Trajectory-based improvement estimation |

## Approach Types (Ensemble Diversification)

When using `--meta` with multiple experts, each gets a different approach:

| Approach | Description | Best for |
|----------|-------------|----------|
| code-sandbox | Generate JS, execute, verify output | Grid/array transformations |
| decomposition | Break into sub-problems, solve each | Multi-step problems |
| chain-of-questions | Hierarchical broad→specific probing | Knowledge questions |
| analogy | Solve simpler version first, scale up | Hard spatial problems |
| counter-factual | Generate wrong solutions, analyze failures | Stubborn problems |
| exhaustive-search | Enumerate possibilities, filter | Small search spaces |

## Verification Methods

| Method | How it works |
|---|---|
| `sandbox` | Execute JS code in Node vm sandbox, compare output |
| `self-audit` | LLM checks its own answer for accuracy |
| `external` | Run a custom shell command to verify |
| `none` | No verification |

## Persistent Data

All meta-system data persists at `~/.pi-reason-harness/`:

| File | Contents |
|------|----------|
| `strategies.json` | Strategy library with ROI, quality metrics, lineage |
| `meta-rules.json` | Cross-strategy principles with validation stats |
| `model-routes.json` | Per model×category routing stats |
| `harness-specs.json` | Complete harness specs per category×approach |
| `synthesized-prompts.json` | Per-problem-type specialized prompts |
| `meta-harnesses.json` | Generated approach types with evolution |

## Key Principles

1. **The loop is the intelligence** — The multi-step verify-feedback loop, not the prompt
2. **Critique, don't create** — Modify proven templates with targeted deltas
3. **Meta-rules compound** — Cross-strategy principles bias all future improvements
4. **Diversity beats depth** — Different approaches per expert > more iterations with same approach
5. **Budget is a bandit** — Allocate compute where marginal ROI is highest
6. **Transfer compounds** — Grid strategies inform knowledge strategies via meta-rules
7. **JS-exclusive** — Zero Python dependency

## init Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--name` | Yes | — | Session name |
| `--type` | No | `code-reasoning` | Task type |
| `--models` | No | `["openai/gpt-4o"]` | JSON array of model IDs |
| `--num-experts` | No | `1` | Number of parallel experts |
| `--verification` | No | `sandbox` | Verification method |
| `--max-cost` | No | — | Max cost per problem (USD) |
| `--max-time` | No | — | Max time per problem (seconds) |

## solve Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--problem` | No* | Problem description (*or use --train-inputs) |
| `--train-inputs` | No | JSON array of training inputs |
| `--train-outputs` | No | JSON array of training outputs |
| `--test-inputs` | No | JSON array of test inputs |
| `--meta` / `-m` | No | Enable the 13-layer meta-system |

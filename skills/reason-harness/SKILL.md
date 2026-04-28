---
name: reason-harness
description: Run iterative solve-verify-feedback reasoning loops on top of any LLM. Use when you need to solve problems requiring code generation with verification, knowledge extraction through chain-of-questions probing, or hybrid reasoning tasks. The system learns from past problems and adapts its strategy over time. Works by generating candidate solutions, executing them in sandboxes or self-auditing, building feedback from failures, and voting across parallel experts.
---

# Reason Harness

Iterative reasoning with verification: generate solutions, test them, learn from failures, improve. Works on top of any LLM — the intelligence is the loop, not the prompt.

The `pi-reason-harness` CLI auto-spawns a long-lived harness server on first use. Every call dispatches an action to the harness, which holds session state across calls. The system uses pi's own LLM infrastructure (pi-ai) and runs code in Node's vm sandbox — zero Python dependency.

## Setup (once, first use)

Symlink the CLI if not on PATH:

```bash
# macOS (Apple Silicon + Homebrew)
command -v pi-reason-harness >/dev/null || ln -sf <skill-dir>/harness/cli.ts /opt/homebrew/bin/pi-reason-harness

# macOS (Intel) / most Linux
command -v pi-reason-harness >/dev/null || ln -sf <skill-dir>/harness/cli.ts /usr/local/bin/pi-reason-harness

# Linux without sudo
command -v pi-reason-harness >/dev/null || { mkdir -p ~/.local/bin && ln -sf <skill-dir>/harness/cli.ts ~/.local/bin/pi-reason-harness; }
```

## How to use

```bash
# Initialize a reasoning session
pi-reason-harness init --name "ARC solver" --type code-reasoning --models '["anthropic/claude-sonnet-4-5"]' --num-experts 2

# Solve a problem (with training data for verification)
pi-reason-harness solve --problem "Transform the grid by..." \
  --train-inputs '[[1,2],[3,4]]' \
  --train-outputs '[[4,3],[2,1]]' \
  --test-inputs '[[5,6]]'

# Solve a knowledge question (self-audit verification)
pi-reason-harness init --name "HLE questions" --type knowledge-extraction --verification self-audit
pi-reason-harness solve --problem "Who was the first person to..."

# Check status
pi-reason-harness status

# View iteration results
pi-reason-harness results --last 10

# Inspect what the system has learned
pi-reason-harness learn

# Reset learned strategies
pi-reason-harness reset-learn
```

Also accepts JSON for programmatic use:

```bash
pi-reason-harness '{ "action": "solve", "problem": "..." }'
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

### Server management

| Command | Behavior |
|---|---|
| `pi-reason-harness --status` | Print health JSON or exit 1 if down |
| `pi-reason-harness --start` | Start the harness server |
| `pi-reason-harness --stop` | Graceful shutdown |
| `pi-reason-harness --restart` | Stop + start fresh |
| `pi-reason-harness --logs` | `tail -f` the server log |

Env vars: `PI_REASON_HARNESS_PORT` (default `9880`), `PI_REASON_HARNESS_LOG` (default `/tmp/pi-reason-harness.log`).

## Task Types

The harness adapts its strategy based on task type:

| Type | Strategy | Verification | When to use |
|---|---|---|---|
| `code-reasoning` | Generate JavaScript code → sandbox execute → verify against examples → feedback loop | Sandbox (default) or external command | Problems where code can verify correctness (puzzles, algorithms, transformations) |
| `knowledge-extraction` | Chain-of-questions probing → self-audit → confidence bucketing | Self-audit (recommended) or none | Factual questions requiring scattered knowledge retrieval |
| `hybrid` | Decide per-problem: code or direct answer → verify → feedback | Any method | Mixed tasks with both reasoning and knowledge components |

## Verification Methods

| Method | How it works | When to use |
|---|---|---|
| `sandbox` | Execute generated JavaScript code in Node vm sandbox, compare output to expected | Code-reasoning tasks with known input/output pairs |
| `self-audit` | LLM checks its own answer for accuracy, consistency, completeness | Knowledge questions where no sandbox verification is possible |
| `external` | Run a custom shell command to verify | When you have a custom checker script |
| `none` | No verification, just record the answer | Exploration mode or when no verification is possible |

## How It Works

### The Iterative Loop

For each expert (running in parallel):

```
┌─────────────────────────────────────────────────────┐
│  1. Build prompt (problem + optional past solutions) │
│  2. Call LLM via pi-ai → generate candidate        │
│  3. Parse code and/or answer from response          │
│  4. Verify:                                         │
│     - sandbox: execute code, compare to expected    │
│     - self-audit: LLM audits its own answer         │
│     - external: run custom verification command     │
│  5. If all training examples pass → DONE            │
│  6. If not: build detailed feedback                 │
│     - Per-example pass/fail                         │
│     - Soft scores (partial credit)                  │
│     - Error messages / audit issues                  │
│  7. Add solution + feedback to history              │
│  8. Next iteration (with feedback from past)        │
│  9. Budget check: stop if cost/time exceeded        │
└─────────────────────────────────────────────────────┘
```

### Feedback Injection

Past solutions are injected probabilistically into subsequent prompts. Each past solution includes:
- The code attempted
- Per-example feedback (pass/fail, accuracy score, errors)
- Overall score (0–1)

The feedback format teaches the model *why* past attempts failed, not just *that* they failed.

### Multi-Expert Ensembling

Multiple experts run in parallel with different:
- Seeds (different example orderings)
- Potentially different models
- Independent iteration histories

### Voting and Ranking

Results from all experts are grouped by canonical test output and ranked:
1. **Passers** (solutions that pass all training examples) sorted by vote count
2. **Failed-but-matching** outputs can be merged into passer groups
3. **Failures** sorted by best soft score
4. **Diversity-first** ordering: one representative per group before duplicates

### Self-Improvement (The Meta-System)

The harness learns from every problem it attempts:

1. **Model preference tracking** — When a model succeeds on a task type, the meta-system records this and biases future expert configs toward that model
2. **Feedback effectiveness detection** — When later iterations with feedback succeed but first iterations fail, the system reinforces feedback-driven improvement
3. **Partial solution awareness** — When solutions score high but don't pass, the system learns that small adjustments matter more than starting from scratch
4. **Performance adaptation** — When timeouts occur, the system injects timing-awareness prompt modifiers

These adaptations accumulate across problems in a session, making the system progressively better at the task type.

### Budget Tracking

Each solve call tracks:
- Token usage (prompt + completion)
- Cost (using model pricing data from pi-ai)
- Time elapsed
- Problems solved vs attempted

Budget limits can be set per-problem via `--max-cost` and `--max-time`.

### Chain-of-Questions Probing (Knowledge Extraction)

For knowledge-extraction tasks, the strategy template encourages:
1. Breaking the question into sub-questions
2. Building a hierarchy from broad to narrow
3. Cross-referencing multiple knowledge sources
4. Confidence bucketing (HIGH/MEDIUM/LOW)

### Self-Audit Verification

For tasks where sandbox verification isn't possible:
1. The LLM generates an initial answer
2. A second LLM call (low temperature, reasoning enabled) audits the answer
3. The audit checks: factual accuracy, logical consistency, completeness, precision
4. Issues are fed back as feedback for the next iteration
5. Only HIGH-confidence verified answers trigger early exit

## Key Principles

1. **The prompt is an interface, not the intelligence.** The multi-step loop with verification IS the intelligence; the prompt just sets the format.
2. **Inconsistency exploitation.** LLMs solve many problems inconsistently. Verification + retry catches cases where a model *can* solve it but didn't on the first try.
3. **Soft scores guide search.** Not just pass/fail — partial credit guides the search toward better solutions.
4. **Diversity via stochasticity.** Temperature=1.0, shuffled examples, different seeds, probabilistic solution selection = diverse attempts.
5. **Self-auditing.** The system decides when it has enough information to terminate (early exit on all-pass or HIGH-confidence audit).
6. **Model-agnostic.** Works with any LLM through pi-ai — the harness orchestrates, the model provides.
7. **Self-improving.** The meta-system learns what works and adapts strategy for future problems.
8. **JS-exclusive.** Zero Python dependency — LLM calls go through pi's native LLM infrastructure, code sandbox runs in Node's vm module.

## init Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--name` | Yes | — | Human-readable session name |
| `--type` | No | `code-reasoning` | Task type: `code-reasoning`, `knowledge-extraction`, `hybrid` |
| `--models` | No | `["openai/gpt-4o"]` | JSON array of model identifiers (provider/model format) |
| `--num-experts` | No | `1` | Number of parallel experts |
| `--verification` | No | `sandbox` | Verification method: `sandbox`, `self-audit`, `external`, `none` |
| `--verify-command` | No | — | External verification command (for `external` type). Use `{code}`, `{input}`, `{expected}` placeholders |
| `--max-cost` | No | — | Max cost per problem in USD |
| `--max-time` | No | — | Max time per problem in seconds |

## solve Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--problem` | Yes | The problem description/formatted examples |
| `--train-inputs` | No | JSON array of training input data |
| `--train-outputs` | No | JSON array of training output data (ground truth) |
| `--test-inputs` | No | JSON array of test input data |

## learn Output

Shows strategy adaptations the meta-system has discovered:

- Model preferences per task type
- Feedback effectiveness insights
- Partial solution patterns
- Performance adaptations

Each adaptation includes:
- The insight learned
- Which task type it applies to
- Evidence count (how many problems confirmed it)
- Any prompt modifiers applied to future problems

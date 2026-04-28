# pi-reason-harness

Recursive self-improving reasoning harness for [pi](https://github.com/badlogic/pi-mono) — iterate, verify, improve.

Builds task-specific reasoning strategies on top of any LLM by running iterative solve-verify-feedback loops with multi-expert ensembling, voting, and a meta-system that learns from past problems.

**JS-exclusive** — LLM calls go through pi's native LLM infrastructure (`@mariozechner/pi-ai`). Code sandbox uses Node's `vm` module. Zero Python dependency.

## How It Works

The core insight: **LLMs are amazing knowledge stores, but naive usage fails to extract that knowledge reliably.** Information is fragmented inside the model's weights, and you need intelligent probing strategies to surface and reassemble it.

The harness implements:

1. **Iterative solve-verify-feedback loops** — Generate candidate solutions, verify them (sandbox/self-audit/external), build detailed feedback from failures, inject that feedback into subsequent iterations
2. **Multi-expert ensembling** — Run multiple experts in parallel with different seeds and models
3. **Voting and ranking** — Group results by canonical output, rank by vote count with diversity-first ordering
4. **Self-auditing** — The system decides when it has enough information to terminate (early exit on all-pass or HIGH-confidence audit)
5. **Soft-score-guided search** — Not just pass/fail; partial credit guides the search toward better solutions
6. **Self-improving meta-system** — Learns from every problem: model preferences, feedback effectiveness, partial solution patterns, performance adaptations
7. **Budget tracking** — Token usage, cost, and time tracking with per-problem limits
8. **Chain-of-questions probing** — For knowledge extraction: hierarchical sub-question decomposition, confidence bucketing, cross-referencing

### Task Types

| Type | Strategy | Verification |
|------|----------|--------------|
| `code-reasoning` | Generate JavaScript code → sandbox execute → verify against examples → feedback loop | Sandbox (default) or external |
| `knowledge-extraction` | Chain-of-questions probing → self-audit → confidence bucketing | Self-audit (recommended) |
| `hybrid` | Decide per-problem: code or direct answer → verify → feedback | Any method |

### Verification Methods

| Method | How it works |
|--------|-------------|
| `sandbox` | Execute generated JavaScript code in Node vm sandbox, compare output to expected |
| `self-audit` | LLM checks its own answer for accuracy, consistency, completeness |
| `external` | Run a custom shell command to verify |
| `none` | No verification, just record the answer |

### Self-Improvement

The meta-system learns from every problem it attempts:
- **Model preference tracking** — Records which models succeed on which task types
- **Feedback effectiveness** — Detects when feedback-driven improvement is working
- **Partial solution awareness** — Learns that high-scoring near-misses need small fixes
- **Performance adaptation** — Injects timing-awareness when timeouts are frequent

These adaptations accumulate across problems, making the system progressively better at the task type.

## Quick Start

```bash
# Initialize a reasoning session
pi-reason-harness init --name "ARC solver" --type code-reasoning \
  --models '["anthropic/claude-sonnet-4-5","openai/gpt-4o"]' --num-experts 2

# Solve a problem with verification
pi-reason-harness solve --problem "Transform the grid..." \
  --train-inputs '[[1,2],[3,4]]' \
  --train-outputs '[[4,3],[2,1]]' \
  --test-inputs '[[5,6]]'

# Knowledge extraction with self-audit
pi-reason-harness init --name "HLE" --type knowledge-extraction --verification self-audit
pi-reason-harness solve --problem "Who was the first person to..."

# Check what the system has learned
pi-reason-harness learn

# Budget-constrained solving
pi-reason-harness init --name "budget test" --type code-reasoning --max-cost 0.50 --max-time 120
```

## Architecture

```
┌──────────────────────────────────────────────────┐
│  META-SYSTEM (strategy discovery + learning)     │
│  - Selects prompt templates by task type         │
│  - Applies learned strategy adaptations          │
│  - Tracks model preferences                      │
│  - Detects feedback effectiveness                │
│  - Budget enforcement                            │
└──────────────┬───────────────────────────────────┘
               │ generates (with adaptations)
               ▼
┌──────────────────────────────────────────────────┐
│  TASK-SPECIFIC HARNESS                           │
│                                                  │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐     │
│  │ Expert 1  │  │ Expert 2  │  │ Expert N  │     │
│  │ (pi-ai    │  │ (pi-ai    │  │ (pi-ai    │     │
│  │ LLM call  │  │ LLM call  │  │ LLM call  │     │
│  │ + sandbox │  │ + sandbox │  │ + sandbox │     │
│  │ /audit    │  │ /audit    │  │ /audit    │     │
│  │ + verify  │  │ + verify  │  │ + verify  │     │
│  │ + feedback│  │ + feedback│  │ + feedback│     │
│  │ loop)     │  │ loop)     │  │ loop)     │     │
│  └────┬───── ┘  └───┬───────┘  └──┬────────┘     │
│       └─────────────┼─────────────┘              │
│                     ▼                            │
│              VOTING / RANKING                    │
│              (group by output,                   │
│               most-voted wins,                   │
│               diversity-first)                   │
│                     │                            │
│                     ▼                            │
│              LEARN & ADAPT                       │
│              (update strategy adaptations)       │
└──────────────────────────────────────────────────┘
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

### init flags

`--name`, `--type`, `--models`, `--num-experts`, `--verification`, `--verify-command`, `--max-cost`, `--max-time`

### solve flags

`--problem`, `--train-inputs`, `--train-outputs`, `--test-inputs`

## Installation

```bash
pi install git:github.com/user/pi-reason-harness
```

Or clone and link:

```bash
git clone https://github.com/user/pi-reason-harness.git
cd pi-reason-harness
npm install
```

## LLM Integration

The harness uses `@mariozechner/pi-ai` for all LLM calls. Models are specified in `provider/model` format (e.g., `anthropic/claude-sonnet-4-5`, `openai/gpt-4o`). API keys are resolved from the same environment variables pi uses:

- `ANTHROPIC_API_KEY` — Anthropic models
- `OPENAI_API_KEY` — OpenAI models
- `GEMINI_API_KEY` — Google models
- etc.

No additional setup required — if pi can call the model, so can the harness.

## Tests

```bash
npm test
```

## License

MIT

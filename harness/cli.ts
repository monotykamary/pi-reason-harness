#!/usr/bin/env node
/**
 * pi-reason-harness — CLI for iterative reasoning with verification.
 *
 * Usage:
 *   pi-reason-harness init --name "..." --type code-reasoning --models '["openai/gpt-4o"]'
 *   pi-reason-harness solve --problem "..." [--train-inputs '[[...]]'] [--train-outputs '[[...]]'] [--test-inputs '[[...]]']
 *   pi-reason-harness status
 *   pi-reason-harness results [--last 10]
 *   pi-reason-harness learn
 *   pi-reason-harness reset-learn
 *   pi-reason-harness clear
 *
 * Server management:
 *   pi-reason-harness --status
 *   pi-reason-harness --start
 *   pi-reason-harness --stop
 *   pi-reason-harness --restart
 *   pi-reason-harness --logs
 */

import { spawn as spawnChild } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as http from 'node:http';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PI_REASON_HARNESS_PORT ?? 9880);
const HOST = '127.0.0.1';
const BASE_URL = `http://${HOST}:${PORT}`;
const LOG = process.env.PI_REASON_HARNESS_LOG ?? '/tmp/pi-reason-harness.log';

// =============================================================================
// HTTP helpers
// =============================================================================

function httpGet(url: string): Promise<{ status: number; body: string }> {
  return new Promise((resolve) => {
    const req = http.get(url, { timeout: 2000 }, (res) => {
      const chunks: Buffer[] = [];
      res.on('data', (c: Buffer) => chunks.push(c));
      res.on('end', () => {
        resolve({ status: res.statusCode ?? 0, body: Buffer.concat(chunks).toString('utf-8') });
      });
    });
    req.on('error', (err) => resolve({ status: 0, body: err.message }));
    req.on('timeout', () => { req.destroy(); resolve({ status: 0, body: 'timeout' }); });
  });
}

function httpPost(
  url: string,
  body: string,
  extraHeaders?: Record<string, string>
): Promise<{ status: number; body: string }> {
  return new Promise((resolve) => {
    const data = Buffer.from(body, 'utf-8');
    const req = http.request(
      url,
      {
        method: 'POST',
        headers: { 'content-type': 'application/json; charset=utf-8', 'content-length': data.length, ...extraHeaders },
        timeout: 60 * 60 * 1000,
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on('data', (c: Buffer) => chunks.push(c));
        res.on('end', () => resolve({ status: res.statusCode ?? 0, body: Buffer.concat(chunks).toString('utf-8') }));
      }
    );
    req.on('error', (err) => resolve({ status: 0, body: err.message }));
    req.on('timeout', () => { req.destroy(); resolve({ status: 0, body: 'timeout' }); });
    req.write(data);
    req.end();
  });
}

// =============================================================================
// Session ID resolution
// =============================================================================

function readSessionIdFromFile(): string | undefined {
  try {
    const cwd = process.cwd();
    const sessionFilePath = path.join(cwd, '.pi', 'reason-harness', 'session-id');
    if (fs.existsSync(sessionFilePath)) {
      const id = fs.readFileSync(sessionFilePath, 'utf-8').trim();
      if (id) return id;
    }
  } catch {}
  return undefined;
}

function agentHeaders(): Record<string, string> {
  const headers: Record<string, string> = {};
  const sessionId = readSessionIdFromFile();
  if (sessionId) headers['x-session-id'] = sessionId;
  return headers;
}

// =============================================================================
// Server lifecycle
// =============================================================================

async function isUp(): Promise<boolean> {
  const { status } = await httpGet(`${BASE_URL}/health`);
  return status === 200;
}

async function startServer(): Promise<boolean> {
  if (await isUp()) return true;

  let serverScript = path.resolve(__dirname, 'server.js');
  if (!fs.existsSync(serverScript)) {
    const tsPath = path.resolve(__dirname, 'server.ts');
    if (fs.existsSync(tsPath)) serverScript = tsPath;
  }

  const useTsx = serverScript.endsWith('.ts');
  const cmd = useTsx ? 'npx' : 'node';
  const args = useTsx ? ['tsx', serverScript] : [serverScript];

  const child = spawnChild(cmd, args, {
    cwd: process.cwd(),
    stdio: ['ignore', 'ignore', 'ignore'],
    detached: true,
    env: { ...process.env, PI_REASON_HARNESS_PORT: String(PORT), PI_REASON_HARNESS_LOG: LOG },
  });
  child.unref();

  for (let i = 0; i < 150; i++) {
    await new Promise((r) => setTimeout(r, 100));
    if (await isUp()) return true;
  }

  process.stderr.write(`pi-reason-harness: server failed to start on ${BASE_URL} (see ${LOG})\n`);
  return false;
}

// =============================================================================
// Action dispatch
// =============================================================================

async function postAction(jsonBody: string): Promise<void> {
  const { status, body } = await httpPost(`${BASE_URL}/action`, jsonBody, agentHeaders());
  if (status === 200) {
    try {
      const parsed = JSON.parse(body);
      if (parsed.ok && parsed.result?.text) {
        process.stdout.write(parsed.result.text + '\n');
      } else if (!parsed.ok) {
        process.stderr.write(`Error: ${parsed.error}\n`);
        process.exit(1);
      }
    } catch {
      if (body.trim()) process.stdout.write(body + '\n');
    }
  } else if (status === 0) {
    process.stderr.write(`Error: cannot reach harness server at ${BASE_URL}\n`);
    process.exit(1);
  } else {
    try {
      const parsed = JSON.parse(body);
      process.stderr.write(`Error: ${parsed.error ?? body}\n`);
    } catch {
      process.stderr.write(`Error: HTTP ${status} — ${body}\n`);
    }
    process.exit(1);
  }
}

// =============================================================================
// Argument parser
// =============================================================================

function extractFlag(args: string[], name: string): string | undefined {
  const idx = args.findIndex((a) => a === `--${name}`);
  if (idx !== -1 && idx + 1 < args.length) {
    const val = args[idx + 1];
    args.splice(idx, 2);
    return val;
  }
  return undefined;
}

// =============================================================================
// CLI entrypoint
// =============================================================================

async function main(): Promise<void> {
  const rawArgs = process.argv.slice(2);

  if (rawArgs.length === 0) {
    process.stderr.write(`pi-reason-harness — iterative reasoning with verification CLI

Usage:
  pi-reason-harness init --name "..." --type code-reasoning|knowledge-extraction|hybrid [--models '["openai/gpt-4o"]'] [--num-experts 1] [--verification sandbox|self-audit|external|none] [--verify-command "..."] [--max-cost 1.0] [--max-time 300]
  pi-reason-harness solve --problem "..." [--train-inputs '[[...]]'] [--train-outputs '[[...]]'] [--test-inputs '[[...]]']
  pi-reason-harness status
  pi-reason-harness results [--last 10]
  pi-reason-harness learn
  pi-reason-harness reset-learn
  pi-reason-harness clear

Server management:
  pi-reason-harness --status
  pi-reason-harness --start
  pi-reason-harness --stop
  pi-reason-harness --restart
  pi-reason-harness --logs

Environment:
  PI_REASON_HARNESS_PORT  Server port (default: 9880)
  PI_REASON_HARNESS_LOG  Log file (default: /tmp/pi-reason-harness.log)
`);
    return;
  }

  const first = rawArgs[0];

  // Server management
  if (first === '--status') {
    const { status, body } = await httpGet(`${BASE_URL}/health`);
    process.stdout.write(status === 200 ? body + '\n' : '{"ok":false,"error":"down"}\n');
    process.exit(status === 200 ? 0 : 1);
  }
  if (first === '--start') {
    await startServer();
    const { body } = await httpGet(`${BASE_URL}/health`);
    process.stdout.write(body + '\n');
    return;
  }
  if (first === '--stop') {
    if (await isUp()) {
      await httpPost(`${BASE_URL}/quit`, '');
      process.stdout.write('{"ok":true,"stopped":true}\n');
    } else {
      process.stdout.write('{"ok":true,"stopped":false,"note":"already down"}\n');
    }
    return;
  }
  if (first === '--restart') {
    if (await isUp()) {
      await httpPost(`${BASE_URL}/quit`, '');
      await new Promise((r) => setTimeout(r, 200));
    }
    await startServer();
    const { body } = await httpGet(`${BASE_URL}/health`);
    process.stdout.write(body + '\n');
    return;
  }
  if (first === '--logs') {
    const { spawn } = await import('node:child_process');
    spawn('tail', ['-f', LOG], { stdio: 'inherit' });
    return;
  }

  // JSON passthrough
  if (first.startsWith('{')) {
    if (!(await startServer())) process.exit(1);
    await postAction(first);
    return;
  }

  // Action subcommands
  if (!(await startServer())) process.exit(1);

  const args = [...rawArgs];
  const action = args.shift()!;

  switch (action) {
    case 'init': {
      const name = extractFlag(args, 'name');
      const type = extractFlag(args, 'type');
      const modelsRaw = extractFlag(args, 'models');
      const numExperts = extractFlag(args, 'num-experts');
      const verification = extractFlag(args, 'verification');
      const verifyCommand = extractFlag(args, 'verify-command');
      const maxCost = extractFlag(args, 'max-cost');
      const maxTime = extractFlag(args, 'max-time');

      if (!name) {
        process.stderr.write('Error: init requires --name.\n');
        process.exit(1);
      }

      let models: string[] = ['openai/gpt-4o'];
      if (modelsRaw) {
        try { models = JSON.parse(modelsRaw); } catch {
          process.stderr.write('Error: --models must be a JSON array of strings.\n');
          process.exit(1);
        }
      }

      await postAction(
        JSON.stringify({
          action: 'init',
          name,
          type: type || 'code-reasoning',
          models,
          numExperts: numExperts ? Number(numExperts) : 1,
          verification: verification || 'sandbox',
          verifyCommand,
          maxCostPerProblem: maxCost ? Number(maxCost) : undefined,
          maxTimePerProblem: maxTime ? Number(maxTime) : undefined,
        })
      );
      break;
    }

    case 'solve': {
      const problem = extractFlag(args, 'problem') ?? '';
      const trainInputsRaw = extractFlag(args, 'train-inputs');
      const trainOutputsRaw = extractFlag(args, 'train-outputs');
      const testInputsRaw = extractFlag(args, 'test-inputs');
      const meta = args.includes('--meta') || args.includes('-m');

      if (!problem && !trainInputsRaw) {
        process.stderr.write('Error: solve requires --problem or --train-inputs/--train-outputs.\n');
        process.exit(1);
      }

      let trainInputs: unknown[] = [];
      let trainOutputs: unknown[] = [];
      let testInputs: unknown[] = [];

      if (trainInputsRaw) {
        try { trainInputs = JSON.parse(trainInputsRaw); } catch {
          process.stderr.write('Error: --train-inputs must be a JSON array.\n');
          process.exit(1);
        }
      }
      if (trainOutputsRaw) {
        try { trainOutputs = JSON.parse(trainOutputsRaw); } catch {
          process.stderr.write('Error: --train-outputs must be a JSON array.\n');
          process.exit(1);
        }
      }
      if (testInputsRaw) {
        try { testInputs = JSON.parse(testInputsRaw); } catch {
          process.stderr.write('Error: --test-inputs must be a JSON array.\n');
          process.exit(1);
        }
      }

      await postAction(
        JSON.stringify({
          action: 'solve',
          problem,
          trainInputs,
          trainOutputs,
          testInputs,
          meta,
        })
      );
      break;
    }

    case 'status': {
      await postAction(JSON.stringify({ action: 'status' }));
      break;
    }

    case 'results': {
      const last = extractFlag(args, 'last');
      await postAction(JSON.stringify({ action: 'results', last: last ? Number(last) : 10 }));
      break;
    }

    case 'learn': {
      await postAction(JSON.stringify({ action: 'learn' }));
      break;
    }

    case 'reset-learn': {
      await postAction(JSON.stringify({ action: 'reset-learn' }));
      break;
    }

    case 'meta-analyze': {
      const problem = extractFlag(args, 'problem');
      if (!problem) {
        process.stderr.write('meta-analyze requires --problem\n');
        process.exit(1);
      }
      await postAction(JSON.stringify({ action: 'meta-analyze', problem }));
      break;
    }

    case 'meta-improve': {
      await postAction(JSON.stringify({ action: 'meta-improve' }));
      break;
    }

    case 'strategies': {
      await postAction(JSON.stringify({ action: 'strategies' }));
      break;
    }

    case 'meta-rules': {
      await postAction(JSON.stringify({ action: 'meta-rules' }));
      break;
    }

    case 'model-routes': {
      await postAction(JSON.stringify({ action: 'model-routes' }));
      break;
    }

    case 'harness-specs': {
      await postAction(JSON.stringify({ action: 'harness-specs' }));
      break;
    }

    case 'evolve-harness': {
      await postAction(JSON.stringify({ action: 'evolve-harness' }));
      break;
    }

    case 'transfer': {
      const sourceCategory = extractFlag(args, 'source-category');
      const targetCategory = extractFlag(args, 'target-category');
      if (!sourceCategory || !targetCategory) {
        process.stderr.write('transfer requires --source-category and --target-category\n');
        process.exit(1);
      }
      await postAction(JSON.stringify({ action: 'transfer', sourceCategory, targetCategory }));
      break;
    }

    case 'decompose': {
      const problem = extractFlag(args, 'problem');
      if (!problem) {
        process.stderr.write('decompose requires --problem\n');
        process.exit(1);
      }
      await postAction(JSON.stringify({ action: 'decompose', problem }));
      break;
    }

    case 'synth-prompts': {
      await postAction(JSON.stringify({ action: 'synth-prompts' }));
      break;
    }

    case 'meta-harnesses': {
      await postAction(JSON.stringify({ action: 'meta-harnesses' }));
      break;
    }

    case 'generate-meta-harness': {
      await postAction(JSON.stringify({ action: 'generate-meta-harness' }));
      break;
    }

    case 'clear': {
      await postAction(JSON.stringify({ action: 'clear' }));
      break;
    }

    default: {
      process.stderr.write(`Unknown command: ${action}. Use --help for usage.\n`);
      process.exit(1);
    }
  }
}

main().catch((err) => {
  process.stderr.write(`pi-reason-harness: ${err instanceof Error ? err.message : err}\n`);
  process.exit(1);
});

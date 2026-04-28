/**
 * pi-reason-harness — Pi Extension
 *
 * Thin lifecycle shell for the reason harness server.
 * All reasoning interactions happen through the `pi-reason-harness` CLI,
 * which dispatches to a long-lived harness server holding session state.
 *
 * This extension:
 *   - Installs the CLI shell alias on session start
 *   - Starts/stops the harness server
 *   - Manages the status widget
 *   - Provides the /reason command
 *   - Registers tools for LLM-driven reasoning
 *   - Writes session ID to disk for the harness server
 */

import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn as spawnChild, type ChildProcess } from 'node:child_process';
import type { ExtensionAPI, ExtensionContext } from '@mariozechner/pi-coding-agent';
import { createRuntimeStore } from './src/state/index.js';
import { createWidgetUpdater, clearSessionUi } from './src/ui/index.js';
import { installShellAlias, writeSessionId, getDirs } from './src/lifecycle/index.js';
import { registerTools } from './src/tools/index.js';

// ---------------------------------------------------------------------------
// CLI path resolution
// ---------------------------------------------------------------------------

const __dirname = fileURLToPath(new URL('.', import.meta.url));

function getProjectRoot(): string {
  return join(__dirname, '..', '..');
}

function getCliPath(): string {
  return join(getProjectRoot(), 'harness', 'cli.ts');
}

// ---------------------------------------------------------------------------
// Harness server lifecycle
// ---------------------------------------------------------------------------

interface HarnessServerController {
  start(): void;
  stop(): void;
}

function createHarnessServer(): HarnessServerController {
  let harnessProcess: ChildProcess | null = null;

  function start(): void {
    if (harnessProcess) return;
    if (process.env.PI_SWARM_SPAWNED === '1') return;

    const cliPath = getCliPath();
    const projectRoot = getProjectRoot();

    try {
      harnessProcess = spawnChild('npx', ['tsx', cliPath, '--start'], {
        cwd: projectRoot,
        stdio: ['ignore', 'ignore', 'ignore'],
        detached: true,
      });
      harnessProcess.unref();
    } catch {}
  }

  function stop(): void {
    if (!harnessProcess) return;
    try {
      harnessProcess.kill('SIGTERM');
    } catch {}
    harnessProcess = null;
  }

  return { start, stop };
}

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

export default function reasonHarnessExtension(pi: ExtensionAPI) {
  const runtimeStore = createRuntimeStore();
  const getSessionKey = (ctx: ExtensionContext) => ctx.sessionManager.getSessionId();
  const getRuntime = (ctx: ExtensionContext) => runtimeStore.ensure(getSessionKey(ctx));

  const updateWidget = createWidgetUpdater(getRuntime);
  const harnessServer = createHarnessServer();

  // ===========================================================================
  // Register tools
  // ===========================================================================

  registerTools(pi);

  // ===========================================================================
  // /reason command
  // ===========================================================================

  pi.registerCommand('reason', {
    description: 'Manage reasoning harness (init, solve, status, learn, clear)',
    handler: async (args, extCtx) => {
      const runtime = getRuntime(extCtx);
      const trimmedArgs = (args ?? '').trim();
      const command = trimmedArgs.toLowerCase();

      if (!trimmedArgs) {
        extCtx.ui.notify('Usage: /reason [off|clear|status|learn|<config>]', 'info');
        return;
      }

      if (command === 'off') {
        runtime.sessionName = null;
        runtime.status = 'idle';
        updateWidget(extCtx);
        extCtx.ui.notify('Reason harness OFF', 'info');
        return;
      }

      if (command === 'clear') {
        runtime.sessionName = null;
        runtime.taskType = null;
        runtime.status = 'idle';
        runtime.iterationCount = 0;
        runtime.bestScore = 0;
        runtime.solved = false;
        runtime.totalTokens = 0;
        runtime.totalCost = 0;
        runtime.adaptations = 0;
        runtime.budget = { costUsed: 0, timeUsed: 0, problemsSolved: 0, problemsAttempted: 0 };
        updateWidget(extCtx);
        extCtx.ui.notify('Reason harness cleared', 'info');
        return;
      }

      if (command === 'status') {
        pi.sendUserMessage('Run `pi-reason-harness status` to see session details.');
        return;
      }

      if (command === 'learn') {
        pi.sendUserMessage('Run `pi-reason-harness learn` to see strategy adaptations.');
        return;
      }

      // Treat as init configuration
      pi.sendUserMessage(
        `Initializing reason harness: ${trimmedArgs}. Use pi-reason-harness CLI to configure and solve.`
      );
    },
  });

  // ===========================================================================
  // Keyboard shortcuts
  // ===========================================================================

  pi.registerShortcut('ctrl+shift+r', {
    description: 'Toggle reason harness widget',
    handler: async (ctx) => {
      const runtime = getRuntime(ctx);
      if (!runtime.sessionName) {
        ctx.ui.notify('No reason harness session active', 'info');
        return;
      }
      updateWidget(ctx);
      ctx.ui.notify('Reason harness widget updated', 'info');
    },
  });

  // ===========================================================================
  // Lifecycle
  // ===========================================================================

  pi.on('session_start', async (_event, ctx) => {
    installShellAlias(getCliPath(), getProjectRoot());
    writeSessionId(ctx);
    harnessServer.start();
  });

  pi.on('session_shutdown', async (_e, ctx) => {
    runtimeStore.clear(getSessionKey(ctx));
    clearSessionUi(ctx);
    harnessServer.stop();
  });
}

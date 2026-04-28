/**
 * pi-reason-harness — Registered tools
 */

import type { ExtensionAPI } from '@mariozechner/pi-coding-agent';
import { Type } from '@sinclair/typebox';

export function registerTools(pi: ExtensionAPI): void {
  pi.registerTool({
    name: 'reason_harness_init',
    label: 'Reason Harness Init',
    description: 'Initialize a reasoning harness session for iterative solve-verify-feedback problem solving. Creates experts with verification strategies.',
    parameters: Type.Object({
      name: Type.String({ description: 'Session name' }),
      type: Type.Union([
        Type.Literal('code-reasoning'),
        Type.Literal('knowledge-extraction'),
        Type.Literal('hybrid'),
      ], { description: 'Task type' }),
      models: Type.Array(Type.String(), { description: 'Models to use (provider/id format, e.g. "anthropic/claude-sonnet-4-5")', default: ['openai/gpt-4o'] }),
      numExperts: Type.Number({ description: 'Number of parallel experts', default: 1 }),
      verification: Type.Union([
        Type.Literal('sandbox'),
        Type.Literal('self-audit'),
        Type.Literal('external'),
        Type.Literal('none'),
      ], { description: 'Verification method', default: 'sandbox' }),
    }),
    execute: async (_toolCallId, params, _signal?, _onUpdate?) => {
      const { execFile } = await import('node:child_process');
      const payload = JSON.stringify({ action: 'init', ...params });
      return new Promise((resolve) => {
        execFile('pi-reason-harness', [payload], { timeout: 10000 }, (err, stdout, stderr) => {
          if (err) {
            resolve({ content: [{ type: 'text' as const, text: `Error: ${stderr || err.message}` }], details: {} });
          } else {
            resolve({ content: [{ type: 'text' as const, text: stdout.trim() }], details: {} });
          }
        });
      });
    },
  });

  pi.registerTool({
    name: 'reason_harness_solve',
    label: 'Reason Harness Solve',
    description: 'Run the iterative solve-verify-feedback loop on a problem. Generates candidate solutions, verifies them, builds feedback from failures, and votes across parallel experts.',
    parameters: Type.Object({
      problem: Type.String({ description: 'The problem to solve' }),
      trainInputs: Type.Array(Type.Any(), { description: 'Training input data for verification' }),
      trainOutputs: Type.Array(Type.Any(), { description: 'Training output data (ground truth) for verification' }),
      testInputs: Type.Array(Type.Any(), { description: 'Test input data' }),
    }),
    execute: async (_toolCallId, params, _signal?, _onUpdate?) => {
      const { execFile } = await import('node:child_process');
      const payload = JSON.stringify({ action: 'solve', ...params });
      return new Promise((resolve) => {
        execFile('pi-reason-harness', [payload], { timeout: 600000 }, (err, stdout, stderr) => {
          if (err) {
            resolve({ content: [{ type: 'text' as const, text: `Error: ${stderr || err.message}` }], details: {} });
          } else {
            resolve({ content: [{ type: 'text' as const, text: stdout.trim() }], details: {} });
          }
        });
      });
    },
  });

  pi.registerTool({
    name: 'reason_harness_status',
    label: 'Reason Harness Status',
    description: 'Check the current reasoning harness session status, including iterations, scores, cost, and learned strategy adaptations.',
    parameters: Type.Object({}),
    execute: async (_toolCallId, _params, _signal?, _onUpdate?) => {
      const { execFile } = await import('node:child_process');
      const payload = JSON.stringify({ action: 'status' });
      return new Promise((resolve) => {
        execFile('pi-reason-harness', [payload], { timeout: 10000 }, (err, stdout, stderr) => {
          if (err) {
            resolve({ content: [{ type: 'text' as const, text: `Error: ${stderr || err.message}` }], details: {} });
          } else {
            resolve({ content: [{ type: 'text' as const, text: stdout.trim() }], details: {} });
          }
        });
      });
    },
  });
}

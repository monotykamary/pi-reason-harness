/**
 * pi-reason-harness — UI widgets
 */

import type { ExtensionContext } from '@mariozechner/pi-coding-agent';
import { Text, truncateToWidth } from '@mariozechner/pi-tui';
import type { ReasonHarnessRuntime } from '../types/index.js';

export function createWidgetUpdater(
  getRuntime: (ctx: ExtensionContext) => ReasonHarnessRuntime
) {
  return function updateWidget(extCtx: ExtensionContext): void {
    if (!extCtx.hasUI) return;
    const runtime = getRuntime(extCtx);
    const width = process.stdout.columns || 120;

    if (!runtime.sessionName) {
      extCtx.ui.setWidget('reason-harness', undefined);
      return;
    }

    extCtx.ui.setWidget('reason-harness', (_tui, theme) => {
      const parts = [
        theme.fg('accent', '🧠'),
        theme.fg('text', ` ${runtime.sessionName}`),
        theme.fg('dim', ` │ ${runtime.taskType}`),
        theme.fg('dim', ' │ '),
        theme.fg(
          runtime.status === 'solving' ? 'warning' : runtime.solved ? 'success' : 'muted',
          runtime.status
        ),
        theme.fg('dim', ' │ '),
        theme.fg('muted', `E:${runtime.expertCount}`),
        theme.fg('dim', ' │ '),
        theme.fg(runtime.solved ? 'success' : 'warning', `★ ${runtime.bestScore.toFixed(2)}`),
        theme.fg('dim', ` │ ${runtime.iterationCount} iters`),
      ];

      if (runtime.totalTokens > 0) {
        const tokStr =
          runtime.totalTokens > 1000000
            ? `${(runtime.totalTokens / 1000000).toFixed(1)}M`
            : runtime.totalTokens > 1000
              ? `${(runtime.totalTokens / 1000).toFixed(1)}K`
              : `${runtime.totalTokens}`;
        parts.push(theme.fg('dim', ` │ ${tokStr} tok`));
      }

      if (runtime.totalCost > 0) {
        const costStr =
          runtime.totalCost < 0.01
            ? `$${runtime.totalCost.toFixed(4)}`
            : `$${runtime.totalCost.toFixed(2)}`;
        parts.push(theme.fg('dim', ` │ ${costStr}`));
      }

      if (runtime.adaptations > 0) {
        parts.push(theme.fg('accent', ` │ 🧠${runtime.adaptations}`));
      }

      if (runtime.solved) {
        parts.push(theme.fg('success', ' ✓'));
      }

      parts.push(theme.fg('dim', '  (ctrl+shift+r)'));

      return new Text(truncateToWidth(parts.join(''), width), width);
    });
  };
}

export function clearSessionUi(extCtx: ExtensionContext): void {
  if (extCtx.hasUI) {
    extCtx.ui.setWidget('reason-harness', undefined);
  }
}

/**
 * pi-reason-harness — Lifecycle helpers
 */

import { homedir } from 'node:os';
import * as fs from 'node:fs';
import { join } from 'node:path';
import type { ExtensionContext } from '@mariozechner/pi-coding-agent';

export function getDirs() {
  const baseDir = join(process.cwd(), '.pi', 'reason-harness');
  return { base: baseDir };
}

export function writeSessionId(ctx: ExtensionContext): void {
  try {
    const dirs = getDirs();
    const sessionId = ctx.sessionManager.getSessionId();
    if (sessionId) {
      if (!fs.existsSync(dirs.base)) fs.mkdirSync(dirs.base, { recursive: true });
      fs.writeFileSync(join(dirs.base, 'session-id'), sessionId, 'utf-8');
    }
  } catch {}
}

export function installShellAlias(cliPath: string, projectRoot: string): void {
  try {
    const agentBinDir = join(homedir(), '.pi', 'agent', 'bin');
    if (!fs.existsSync(agentBinDir)) {
      fs.mkdirSync(agentBinDir, { recursive: true });
    }
    const linkPath = join(agentBinDir, 'pi-reason-harness');

    const wrapperContent = `#!/bin/sh
cd "${projectRoot}" 2>/dev/null
exec npx tsx "${cliPath}" "$@"
`;

    let currentContent: string | null = null;
    try {
      currentContent = fs.readFileSync(linkPath, 'utf-8');
    } catch {}
    if (currentContent !== wrapperContent) {
      fs.writeFileSync(linkPath, wrapperContent, { mode: 0o755 });
    }
  } catch {}
}

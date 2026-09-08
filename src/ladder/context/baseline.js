// kolm code — per-SHA baseline of typecheck / lint / tests (spec 5.5). Owner C.
//
// A repo with 12 pre-existing type errors and one already-failing test must not
// fail every candidate. So the untouched base is measured ONCE per SHA and every
// stage computes a DELTA against it: new error ids, newly failing test names.
// The result is cached in context_cache so a run with N candidates pays for it
// once, and a resumed run pays nothing.

import { detectRunner, parseTestOutput, parseTypecheckErrors, parseLintErrors } from './detect.js';

const DEFAULT_TIMEOUT_MS = 600000;
const OUTPUT_KEEP = 4096;

function repoKeyFor(wt, opts = {}) {
  return String(opts.repoKey || wt?.repoKey || wt?.repoPath || wt?.path || 'repo');
}

function toolFor(cmd, kind) {
  const c = String(cmd || '').toLowerCase();
  if (kind === 'typecheck') {
    if (/tsc\b/.test(c)) return 'tsc';
    if (/pyright/.test(c)) return 'pyright';
    if (/mypy/.test(c)) return 'mypy';
    if (/go\s+vet/.test(c)) return 'go';
    if (/cargo\s+check/.test(c)) return 'cargo';
    return undefined;
  }
  if (/eslint/.test(c)) return 'eslint';
  if (/ruff/.test(c)) return 'ruff';
  return undefined;
}

/**
 * Run one detected command in the worktree and keep a bounded slice of output.
 * @returns {Promise<{ran:boolean, cmd:string|null, exitCode:number|null, durationMs:number,
 *                    timedOut:boolean, output:string}>}
 */
export async function runCommand(wt, cmd, { timeoutMs = DEFAULT_TIMEOUT_MS, signal, env } = {}) {
  if (!cmd) return { ran: false, cmd: null, exitCode: null, durationMs: 0, timedOut: false, output: '' };
  const started = Date.now();
  try {
    const res = (await wt.exec(cmd, { timeoutMs, env, signal })) || {};
    const output = `${res.stdout || ''}${res.stderr ? `\n${res.stderr}` : ''}`;
    return {
      ran: true,
      cmd,
      exitCode: res.exitCode ?? null,
      durationMs: res.durationMs ?? Date.now() - started,
      timedOut: res.timedOut === true,
      output,
    };
  } catch (err) {
    return {
      ran: true,
      cmd,
      exitCode: null,
      durationMs: Date.now() - started,
      timedOut: false,
      output: `exec failed: ${err?.message || String(err)}`,
      error: err?.message || String(err),
    };
  }
}

/**
 * Measure the untouched base once per SHA.
 * @param {object|null} store openStore() result, or null to skip caching
 * @param {object} wt worktree at the base sha
 * @param {string} sha
 * @param {object} commands detectCommands() output
 * @param {{repoKey?:string, config?:object, signal?:AbortSignal, force?:boolean}} [opts]
 * @returns {Promise<object>} baseline
 */
export async function baseline(store, wt, sha, commands = {}, opts = {}) {
  const repoKey = repoKeyFor(wt, opts);
  if (store && !opts.force) {
    const cached = store.cacheGet(repoKey, sha, 'baseline');
    if (cached) return cached;
  }
  const timeoutMs = Number(opts.config?.stageTimeoutMs) > 0 ? Number(opts.config.stageTimeoutMs) : DEFAULT_TIMEOUT_MS;
  const root = wt?.path;
  const durations = {};

  const tc = await runCommand(wt, commands.typecheck, { timeoutMs, signal: opts.signal });
  durations.typecheck = tc.durationMs;
  const typecheck = {
    ran: tc.ran,
    cmd: tc.cmd,
    exitCode: tc.exitCode,
    timedOut: tc.timedOut,
    errors: tc.ran ? parseTypecheckErrors(tc.output, toolFor(commands.typecheck, 'typecheck'), { root }) : [],
    output: tc.output.slice(0, OUTPUT_KEEP),
  };

  const lt = await runCommand(wt, commands.lint, { timeoutMs, signal: opts.signal });
  durations.lint = lt.durationMs;
  const lint = {
    ran: lt.ran,
    cmd: lt.cmd,
    exitCode: lt.exitCode,
    timedOut: lt.timedOut,
    errors: lt.ran ? parseLintErrors(lt.output, toolFor(commands.lint, 'lint'), { root }) : [],
    output: lt.output.slice(0, OUTPUT_KEEP),
  };

  const tt = await runCommand(wt, commands.test, { timeoutMs, signal: opts.signal });
  durations.test = tt.durationMs;
  const parsedTests = tt.ran
    ? parseTestOutput(tt.output, detectRunner(commands.test))
    : { passed: 0, failed: 0, failing: [], parsed: false };
  const tests = {
    ran: tt.ran,
    cmd: tt.cmd,
    exitCode: tt.exitCode,
    timedOut: tt.timedOut,
    runner: detectRunner(commands.test),
    passed: parsedTests.passed,
    failed: parsedTests.failed,
    failing: parsedTests.failing,
    parsed: parsedTests.parsed,
    output: tt.output.slice(0, OUTPUT_KEEP),
  };

  const result = {
    sha: sha ?? null,
    generatedAt: new Date().toISOString(),
    commands: {
      test: commands.test ?? null,
      typecheck: commands.typecheck ?? null,
      lint: commands.lint ?? null,
      coverage: commands.coverage ?? null,
    },
    typecheck,
    lint,
    tests,
    durations,
  };
  if (store) store.cachePut(repoKey, sha, 'baseline', result);
  return result;
}

export default { baseline, runCommand };

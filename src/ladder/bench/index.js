// kolm code - the bundled benchmark (owner BENCH, SOTA spec section 5).
//
//   const result = await runBench({ config, store, driver, providers });
//   process.stdout.write(formatBenchTable(result) + '\n');
//
// A bench task is a directory under `tests/fixtures/ladder-bench/<id>/` with a
// `task.json`, a `repo/` tree seeded into a fresh git repository, and a
// `hidden/` tree copied in only AFTER the run. The hidden tree is what decides
// the score: it is invisible to the coder, so a candidate cannot pass by
// rewriting the tests it was shown.
//
// House rules honoured here: ASCII only, no new dependencies, never print a
// secret, and never throw for an environment that simply lacks a tool (a task
// whose requirements are missing is SKIPped with a reason).

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import { LadderError } from '../errors.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));

/** Where the bundled task set lives (repo-relative, resolved once). */
export const BENCH_TASKS_DIR = path.resolve(HERE, '..', '..', '..', 'tests', 'fixtures', 'ladder-bench');

/** Schema version of the object `runBench` returns and writes to disk. */
export const BENCH_RESULT_VERSION = 1;

/** Default ceiling for one hidden-test command, in milliseconds. */
export const DEFAULT_HIDDEN_TIMEOUT_MS = 300000;

/** Verdict used for a task whose requirements are not installed on this host. */
export const SKIP_VERDICT = 'SKIP';

/**
 * The orchestrator is imported lazily so that loading the bench (to list tasks
 * or render a stored result) never pays for the whole ladder.
 * @returns {Promise<Function>} `runTask`
 */
async function loadRunTask() {
  const mod = await import('../orchestrator.js');
  if (typeof mod.runTask !== 'function') {
    throw new LadderError('bad-config', 'src/ladder/orchestrator.js does not export runTask()', {});
  }
  return mod.runTask;
}

// ---------------------------------------------------------------------------
// small helpers (kept local so bench never imports the CLI)
// ---------------------------------------------------------------------------

/** Never let a store/driver read take the whole bench down. */
function attempt(fn, fallback = null) {
  try {
    const value = fn();
    return value === undefined ? fallback : value;
  } catch {
    return fallback;
  }
}

function usd(value) {
  const n = Number(value);
  return Number.isFinite(n) ? `$${n.toFixed(4)}` : '$0.0000';
}

function seconds(ms) {
  const n = Number(ms);
  if (!Number.isFinite(n) || n < 0) return '-';
  if (n < 1000) return `${Math.round(n)}ms`;
  if (n < 60000) return `${(n / 1000).toFixed(1)}s`;
  return `${Math.floor(n / 60000)}m ${Math.round((n % 60000) / 1000)}s`;
}

/**
 * Space-padded ASCII table with a dashed rule. Deliberately a copy of the
 * CLI's renderer rather than an import: `src/ladder/**` must not depend on
 * `cli/**`.
 * @param {string[]} headers
 * @param {Array<Array<string|number|null>>} rows
 * @returns {string}
 */
function renderTable(headers, rows) {
  const cells = [
    headers.map((h) => String(h)),
    ...rows.map((r) => r.map((c) => (c === null || c === undefined ? '-' : String(c)))),
  ];
  const widths = headers.map((_, i) => Math.max(...cells.map((row) => (row[i] ?? '').length)));
  const line = (row) => row.map((cell, i) => String(cell ?? '').padEnd(widths[i])).join('  ').trimEnd();
  const rule = widths.map((w) => '-'.repeat(w)).join('  ');
  return [line(cells[0]), rule, ...cells.slice(1).map(line)].join('\n');
}

/** Recursive copy with no symlink following; `fs.cpSync` is not on Node 22 LTS everywhere. */
function copyTree(from, to) {
  fs.mkdirSync(to, { recursive: true });
  for (const entry of fs.readdirSync(from, { withFileTypes: true })) {
    const src = path.join(from, entry.name);
    const dst = path.join(to, entry.name);
    if (entry.isDirectory()) copyTree(src, dst);
    else if (entry.isFile()) fs.copyFileSync(src, dst);
  }
}

/**
 * Every file under `root`, as `{ rel, abs }` with a posix `rel`, so the same
 * list can drive both an on-disk copy and `wt.writeFile()`.
 * @param {string} root
 * @param {string} [prefix]
 * @returns {Array<{rel: string, abs: string}>}
 */
function walkFiles(root, prefix = '') {
  const found = [];
  let entries = [];
  try {
    entries = fs.readdirSync(root, { withFileTypes: true });
  } catch {
    return found;
  }
  for (const entry of entries.sort((a, b) => (a.name < b.name ? -1 : 1))) {
    const abs = path.join(root, entry.name);
    const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) found.push(...walkFiles(abs, rel));
    else if (entry.isFile()) found.push({ rel, abs });
  }
  return found;
}

function runProcess(argv, cwd, timeoutMs) {
  const [cmd, ...args] = argv;
  return spawnSync(cmd, args, {
    cwd,
    encoding: 'utf8',
    timeout: timeoutMs,
    windowsHide: true,
    shell: false,
    env: hiddenTestEnv(),
  });
}

/**
 * The environment a hidden test runs in. `NODE_TEST_CONTEXT` is stripped
 * because a `node --test` child that inherits it refuses to run its files and
 * still exits 0 - which would score every hidden test as a pass whenever the
 * bench itself is invoked from inside a test runner.
 * @returns {Record<string, string|undefined>}
 */
function hiddenTestEnv() {
  const env = { ...process.env };
  delete env.NODE_TEST_CONTEXT;
  return env;
}

// ---------------------------------------------------------------------------
// task loading
// ---------------------------------------------------------------------------

function parseTaskJson(dir) {
  const file = path.join(dir, 'task.json');
  let raw;
  try {
    raw = fs.readFileSync(file, 'utf8');
  } catch (err) {
    throw new LadderError('not-found', `bench task ${path.basename(dir)} has no task.json`, {
      dir,
      cause: err?.message,
    });
  }
  let parsed;
  try {
    // Editors on Windows sometimes leave a BOM on a JSON file; JSON.parse does not.
    parsed = JSON.parse(raw.replace(/^\uFEFF+/, ''));
  } catch (err) {
    throw new LadderError('bad-config', `bench task ${path.basename(dir)} has invalid task.json: ${err?.message}`, { file });
  }
  for (const field of ['id', 'title', 'text']) {
    if (typeof parsed[field] !== 'string' || parsed[field].trim() === '') {
      throw new LadderError('bad-config', `bench task ${path.basename(dir)} is missing "${field}"`, { file, field });
    }
  }
  return parsed;
}

/**
 * The bundled task set, newest schema first. The four fields the spec names
 * (`id`, `title`, `language`, `dir`) are always present; the rest of the
 * parsed `task.json` rides along so `runBench` does not have to read it twice.
 *
 * @param {{dir?: string, tasks?: string[]|string|null}} [opts]
 *   `dir` overrides the fixture root; `tasks` filters (and orders) by id.
 * @returns {Promise<Array<{id: string, title: string, language: string, dir: string, text: string, expect: object, requires: string[], forbiddenPaths: string[]}>>}
 */
export async function listBenchTasks(opts = {}) {
  const root = opts.dir ? path.resolve(String(opts.dir)) : BENCH_TASKS_DIR;
  let entries = [];
  try {
    entries = fs.readdirSync(root, { withFileTypes: true });
  } catch (err) {
    throw new LadderError('not-found', `no bench task directory at ${root}`, { root, cause: err?.message });
  }
  const all = [];
  for (const entry of entries.sort((a, b) => (a.name < b.name ? -1 : 1))) {
    if (!entry.isDirectory()) continue;
    const dir = path.join(root, entry.name);
    if (!fs.existsSync(path.join(dir, 'task.json'))) continue;
    const parsed = parseTaskJson(dir);
    all.push({
      id: String(parsed.id),
      title: String(parsed.title),
      language: typeof parsed.language === 'string' && parsed.language !== '' ? parsed.language : 'javascript',
      dir,
      text: String(parsed.text),
      expect: parsed.expect && typeof parsed.expect === 'object' ? parsed.expect : {},
      requires: Array.isArray(parsed.requires) ? parsed.requires.map(String) : [],
      forbiddenPaths: Array.isArray(parsed.forbiddenPaths) ? parsed.forbiddenPaths.map(String) : [],
    });
  }

  const wanted = normalizeIds(opts.tasks);
  if (wanted.length === 0) return all;
  const byId = new Map(all.map((task) => [task.id, task]));
  const picked = [];
  for (const id of wanted) {
    const task = byId.get(id);
    if (!task) {
      throw new LadderError('not-found', `unknown bench task "${id}" (have: ${all.map((t) => t.id).join(', ')})`, {
        id,
        known: all.map((t) => t.id),
      });
    }
    picked.push(task);
  }
  return picked;
}

function normalizeIds(value) {
  if (value === null || value === undefined) return [];
  const list = Array.isArray(value) ? value : String(value).split(',');
  return list.map((v) => String(v).trim()).filter((v) => v !== '');
}

// ---------------------------------------------------------------------------
// host requirements
// ---------------------------------------------------------------------------

/** How each named requirement is probed. `node` is whatever runs this process. */
const REQUIREMENT_PROBES = Object.freeze({
  node: () => [process.execPath, ['--version']],
  git: () => ['git', ['--version']],
  python: () => ['python', ['--version']],
  pytest: () => ['python', ['-m', 'pytest', '--version']],
});

/**
 * Is a named tool usable on this host? Never throws and never runs anything
 * heavier than a `--version` call, so a machine without python simply skips
 * the python task instead of failing the bench.
 *
 * @param {string} name one of `node`, `git`, `python`, `pytest`
 * @param {Map<string, {ok: boolean, detail: string}>} [cache]
 * @returns {{ok: boolean, detail: string}}
 */
export function probeRequirement(name, cache = null) {
  const key = String(name);
  if (cache && cache.has(key)) return cache.get(key);
  let result;
  const probe = REQUIREMENT_PROBES[key];
  if (!probe) {
    result = { ok: false, detail: `unknown requirement "${key}"` };
  } else {
    try {
      const [cmd, args] = probe();
      const res = spawnSync(cmd, args, { encoding: 'utf8', timeout: 20000, windowsHide: true });
      if (res.error || res.status !== 0) {
        result = { ok: false, detail: `${key} is not available on this host` };
      } else {
        result = { ok: true, detail: String(res.stdout || res.stderr || '').trim().split('\n')[0] || key };
      }
    } catch {
      result = { ok: false, detail: `${key} is not available on this host` };
    }
  }
  if (cache) cache.set(key, result);
  return result;
}

// ---------------------------------------------------------------------------
// repository seeding
// ---------------------------------------------------------------------------

function git(args, cwd) {
  const res = spawnSync('git', args, { cwd, encoding: 'utf8', timeout: 120000, windowsHide: true });
  if (res.error || res.status !== 0) {
    const detail = String(res.stderr || res.stdout || res.error?.message || '').trim().split('\n')[0];
    throw new LadderError('execution', `git ${args[0]} failed in ${cwd}: ${detail || 'unknown error'}`, { args, cwd });
  }
  return String(res.stdout ?? '');
}

/**
 * Seed `dest` with the task's `repo/` tree and turn it into a one-commit git
 * repository. `core.autocrlf false` is set BEFORE the first `git add` so the
 * CRLF fixture keeps its bytes on Windows, and identity is set locally so the
 * commit works on a host with no global git identity.
 *
 * @param {{id: string, dir: string}} task
 * @param {string} dest absolute path; replaced if it already exists
 * @returns {{repoPath: string, baseSha: string}}
 */
export function seedBenchRepo(task, dest) {
  const source = path.join(task.dir, 'repo');
  if (!fs.existsSync(source)) {
    throw new LadderError('not-found', `bench task ${task.id} has no repo/ directory`, { dir: task.dir });
  }
  fs.rmSync(dest, { recursive: true, force: true });
  copyTree(source, dest);
  git(['init', '-q'], dest);
  // A fixed branch name keeps the bench readable on hosts whose git still
  // defaults to `master`.
  git(['symbolic-ref', 'HEAD', 'refs/heads/main'], dest);
  git(['config', 'user.name', 'kolm bench'], dest);
  git(['config', 'user.email', 'bench@kolm.local'], dest);
  git(['config', 'core.autocrlf', 'false'], dest);
  git(['config', 'commit.gpgsign', 'false'], dest);
  git(['add', '-A'], dest);
  git(['commit', '-q', '-m', 'bench base', '--no-verify'], dest);
  const baseSha = git(['rev-parse', 'HEAD'], dest).trim();
  return { repoPath: dest, baseSha };
}

// ---------------------------------------------------------------------------
// hidden judgement
// ---------------------------------------------------------------------------

/**
 * The command that judges a task. `expect.hiddenCommand` wins when present;
 * otherwise `.py` hidden tests run under pytest and everything else runs under
 * `node --test`.
 *
 * @param {{expect?: object}} task
 * @returns {{argv: string[], display: string, file: string}}
 */
export function hiddenCommandFor(task) {
  const expect = task?.expect ?? {};
  const file = typeof expect.hiddenTest === 'string' && expect.hiddenTest !== ''
    ? expect.hiddenTest
    : 'tests/hidden.test.js';
  const argv = Array.isArray(expect.hiddenCommand) && expect.hiddenCommand.length > 0
    ? expect.hiddenCommand.map(String)
    : file.endsWith('.py')
      ? ['python', '-m', 'pytest', '-q', file]
      : ['node', '--test', file];
  return { argv, display: argv.join(' '), file };
}

/** `node` in a task's command means "the node running this bench", not $PATH. */
function resolveArgv(argv) {
  return argv.map((part, i) => (i === 0 && part === 'node' ? process.execPath : part));
}

/**
 * Copy `hidden/` over the winner and run the hidden command there.
 *
 * Three layers, in order of fidelity: a fresh detached checkout of the winner
 * commit (what a reviewer would do), the winner worktree the run kept, and --
 * for the in-memory fake driver, whose worktrees are not on disk -- the
 * worktree's own writeFile/exec surface. When none of the three is available
 * `hiddenPass` is null rather than false: "not judged" is not "failed".
 */
async function judgeHidden({ task, repoPath, driver, runId, winner, sha, timeoutMs }) {
  const hiddenDir = path.join(task.dir, 'hidden');
  const files = walkFiles(hiddenDir);
  if (files.length === 0) {
    return { hiddenPass: null, reason: 'task has no hidden/ tree', where: null };
  }
  const command = hiddenCommandFor(task);

  // 1. fresh checkout of the winner commit
  if (typeof sha === 'string' && /^[0-9a-f]{7,64}$/i.test(sha) && fs.existsSync(path.join(repoPath, '.git'))) {
    const verifyDir = `${repoPath}__hidden`;
    let added = false;
    try {
      fs.rmSync(verifyDir, { recursive: true, force: true });
      git(['worktree', 'add', '--detach', '-f', verifyDir, sha], repoPath);
      added = true;
      for (const file of files) {
        const dst = path.join(verifyDir, ...file.rel.split('/'));
        fs.mkdirSync(path.dirname(dst), { recursive: true });
        fs.copyFileSync(file.abs, dst);
      }
      const res = runProcess(resolveArgv(command.argv), verifyDir, timeoutMs);
      const timedOut = res.error?.code === 'ETIMEDOUT' || res.signal !== null;
      return {
        hiddenPass: !timedOut && res.status === 0,
        reason: timedOut ? `hidden test timed out after ${timeoutMs} ms` : null,
        where: 'checkout',
      };
    } catch {
      // fall through to the worktree layers
    } finally {
      if (added) {
        attempt(() => git(['worktree', 'remove', '--force', verifyDir], repoPath));
        attempt(() => fs.rmSync(verifyDir, { recursive: true, force: true }));
      }
    }
  }

  // 2 and 3. the worktree the run kept
  if (!winner || typeof driver?.open !== 'function') {
    return { hiddenPass: null, reason: 'no winner worktree to judge', where: null };
  }
  let wt = null;
  try {
    wt = await driver.open({ repoPath, runId, candidateId: winner });
  } catch {
    return { hiddenPass: null, reason: 'the winner worktree is gone', where: null };
  }
  if (!wt) return { hiddenPass: null, reason: 'the winner worktree is gone', where: null };

  const onDisk = typeof wt.path === 'string' && wt.path !== '' && fs.existsSync(wt.path);
  if (onDisk) {
    for (const file of files) {
      const dst = path.join(wt.path, ...file.rel.split('/'));
      fs.mkdirSync(path.dirname(dst), { recursive: true });
      fs.copyFileSync(file.abs, dst);
    }
    const res = runProcess(resolveArgv(command.argv), wt.path, timeoutMs);
    const timedOut = res.error?.code === 'ETIMEDOUT' || res.signal !== null;
    return {
      hiddenPass: !timedOut && res.status === 0,
      reason: timedOut ? `hidden test timed out after ${timeoutMs} ms` : null,
      where: 'worktree',
    };
  }

  if (typeof wt.writeFile !== 'function' || typeof wt.exec !== 'function') {
    return { hiddenPass: null, reason: 'the winner worktree cannot run commands', where: null };
  }
  for (const file of files) {
    await wt.writeFile(file.rel, fs.readFileSync(file.abs, 'utf8'));
  }
  const res = await wt.exec(command.display, { timeoutMs });
  const exitCode = Number(res?.exitCode);
  return {
    hiddenPass: Number.isFinite(exitCode) ? exitCode === 0 : false,
    reason: Number.isFinite(exitCode) ? null : 'the worktree did not report an exit code',
    where: 'driver',
  };
}

// ---------------------------------------------------------------------------
// per-run counters
// ---------------------------------------------------------------------------

/**
 * Counters read back from the run's event stream. Every name is matched
 * defensively: an orchestrator build that does not emit subtask events yet
 * simply reports zero subtasks instead of crashing the bench.
 *
 * @param {object} store ladder store
 * @param {string} runId
 * @returns {{rungReached: number, candidates: number, turns: number, escalations: number, repairs: number, subtasks: number}}
 */
export function benchCountersFor(store, runId) {
  const events = attempt(() => store.eventsSince(runId, 0, 1000000), []) ?? [];
  let sawRung = false;
  let maxRung = 0;
  let turns = 0;
  let escalations = 0;
  let repairs = 0;
  const subtaskIds = new Set();

  for (const event of events) {
    const type = String(event?.type ?? '');
    const payload = event?.payload && typeof event.payload === 'object' ? event.payload : {};
    if (Number.isFinite(payload.rung) && (type === 'candidate.start' || type === 'escalate')) {
      sawRung = true;
      maxRung = Math.max(maxRung, Number(payload.rung));
    }
    if (type === 'coder.turn') turns += 1;
    if (type === 'escalate' || type === 'run.escalate') escalations += 1;
    if (type === 'repair' || type === 'repair.start') repairs += 1;
    if (type === 'subtask.start' || type === 'subtask.end') {
      if (payload.id !== undefined && payload.id !== null) subtaskIds.add(String(payload.id));
    }
    if (type === 'subtasks.plan' && Array.isArray(payload.subtasks)) {
      for (const sub of payload.subtasks) {
        if (sub && sub.id !== undefined && sub.id !== null) subtaskIds.add(String(sub.id));
      }
    }
  }

  const candidates = attempt(() => store.listCandidates(runId).length, 0) ?? 0;
  return {
    rungReached: sawRung ? maxRung + 1 : 0,
    candidates,
    turns,
    escalations,
    repairs,
    subtasks: subtaskIds.size,
  };
}

// ---------------------------------------------------------------------------
// the runner
// ---------------------------------------------------------------------------

function ladderOf(config, override) {
  if (override && typeof override === 'object') return override;
  return {
    coders: Array.isArray(config?.coderLadder) ? [...config.coderLadder] : [],
    reviewer: Array.isArray(config?.reviewerLadder) ? (config.reviewerLadder[0] ?? null) : null,
    orchestrator: config?.orchestratorRef ?? null,
    judge: config?.judgeRef ?? null,
  };
}

function benchDirOf(config) {
  if (typeof config?.benchDir === 'string' && config.benchDir !== '') return config.benchDir;
  if (typeof config?.codeDir === 'string' && config.codeDir !== '') return path.join(config.codeDir, 'bench');
  return path.join(os.tmpdir(), 'kolm-code-bench');
}

/** File-system-safe timestamp: `2026-09-09T14-05-12-345Z`. */
function stampOf(date) {
  return date.toISOString().replace(/[:.]/g, '-');
}

/**
 * Run the bundled benchmark: seed one fresh git repository per task, run the
 * real ladder over it, then judge the winner with the task's hidden tests.
 *
 * Runs are real ladder runs: they land in the store and show up in
 * `kolm code status`.
 *
 * @param {object} opts
 * @param {string[]|string|null} [opts.tasks] task ids to run (default: all)
 * @param {object} opts.config ladder config (`loadConfig` output)
 * @param {object} opts.store open ladder store
 * @param {object} opts.driver worktree driver
 * @param {object} [opts.providers] `{orchestrator, judge, coders, reviewer}`
 * @param {(event: {type: string, payload: object}) => void} [opts.onEvent]
 * @param {AbortSignal} [opts.signal]
 * @param {string} [opts.workDir] where the seeded repositories go
 * @param {number} [opts.concurrency] tasks in flight at once (default 1)
 * @param {string} [opts.dir] fixture root override
 * @param {object} [opts.ladder] ladder description recorded in the result
 * @param {boolean} [opts.write] write `<benchDir>/<ts>.json` (default true)
 * @param {boolean} [opts.keepWork] keep the seeded repositories (default false)
 * @param {number} [opts.hiddenTimeoutMs] per-task hidden command ceiling
 * @param {Function} [opts.runTask] seam for tests; defaults to the orchestrator
 * @returns {Promise<object>} the bench result document (schema version 1)
 */
export async function runBench(opts = {}) {
  const {
    config = {},
    store,
    driver,
    providers = {},
    onEvent,
    signal,
    dir,
    write = true,
    keepWork = false,
  } = opts;

  if (!store) throw new LadderError('bad-arg', 'runBench needs a store');
  if (!driver) throw new LadderError('bad-arg', 'runBench needs a worktree driver');
  const runTask = typeof opts.runTask === 'function' ? opts.runTask : await loadRunTask();

  const probes = new Map();
  const gitProbe = probeRequirement('git', probes);
  if (!gitProbe.ok) {
    throw new LadderError('not-armed', 'kolm code bench needs git on PATH to seed each task repository', {});
  }

  const tasks = await listBenchTasks({ dir, tasks: opts.tasks });
  const concurrency = Math.max(1, Math.min(Number(opts.concurrency) || 1, Math.max(1, tasks.length)));
  const hiddenTimeoutMs = Number.isFinite(opts.hiddenTimeoutMs)
    ? Number(opts.hiddenTimeoutMs)
    : Math.min(Number(config.stageTimeoutMs) || DEFAULT_HIDDEN_TIMEOUT_MS, DEFAULT_HIDDEN_TIMEOUT_MS);

  const startedAtDate = new Date();
  const stamp = stampOf(startedAtDate);
  const workDir = opts.workDir
    ? path.resolve(String(opts.workDir))
    : path.join(benchDirOf(config), 'work', stamp);
  fs.mkdirSync(workDir, { recursive: true });

  const emit = (type, payload) => {
    if (typeof onEvent === 'function') {
      try {
        onEvent({ type, payload });
      } catch {
        // an observer must never break the bench
      }
    }
  };

  const results = new Array(tasks.length).fill(null);
  let cursor = 0;
  const startedAt = Date.now();

  const worker = async () => {
    for (;;) {
      const index = cursor;
      cursor += 1;
      if (index >= tasks.length) return;
      if (signal?.aborted) {
        results[index] = skippedResult(tasks[index], 'aborted before the task started');
        continue;
      }
      results[index] = await runOneTask({
        task: tasks[index],
        config,
        store,
        driver,
        providers,
        onEvent,
        signal,
        workDir,
        probes,
        hiddenTimeoutMs,
        keepWork,
        runTask,
        emit,
      });
    }
  };

  await Promise.all(Array.from({ length: concurrency }, () => worker()));

  const finishedAt = Date.now();
  const judged = results.filter((r) => r.hiddenPass !== null);
  const hiddenPass = results.filter((r) => r.hiddenPass === true).length;
  const document = {
    version: BENCH_RESULT_VERSION,
    startedAt: startedAtDate.toISOString(),
    finishedAt: new Date(finishedAt).toISOString(),
    ladder: ladderOf(config, opts.ladder),
    mode: config.mode ?? 'cascade',
    results,
    summary: {
      tasks: results.length,
      pass: results.filter((r) => r.verdict === 'PASS').length,
      hiddenPass,
      passRate: judged.length > 0 ? hiddenPass / judged.length : 0,
      costUsd: results.reduce((sum, r) => sum + (Number(r.costUsd) || 0), 0),
      durationMs: finishedAt - startedAt,
      escalations: results.reduce((sum, r) => sum + (Number(r.escalations) || 0), 0),
      repairs: results.reduce((sum, r) => sum + (Number(r.repairs) || 0), 0),
    },
  };

  if (write) {
    const outDir = benchDirOf(config);
    fs.mkdirSync(outDir, { recursive: true });
    const file = path.join(outDir, `${stamp}.json`);
    fs.writeFileSync(file, `${JSON.stringify(document, null, 2)}\n`, 'utf8');
    document.resultPath = file;
  } else {
    document.resultPath = null;
  }
  if (!keepWork) attempt(() => fs.rmSync(workDir, { recursive: true, force: true }));
  return document;
}

function emptyResult(task, extra = {}) {
  return {
    id: task.id,
    verdict: null,
    hiddenPass: null,
    costUsd: 0,
    durationMs: 0,
    rungReached: 0,
    candidates: 0,
    turns: 0,
    escalations: 0,
    repairs: 0,
    subtasks: 0,
    runId: null,
    reason: null,
    ...extra,
  };
}

function skippedResult(task, reason) {
  return emptyResult(task, { verdict: SKIP_VERDICT, reason });
}

async function runOneTask(ctx) {
  const {
    task, config, store, driver, providers, onEvent, signal, workDir,
    probes, hiddenTimeoutMs, keepWork, runTask, emit,
  } = ctx;

  for (const requirement of task.requires) {
    const probe = probeRequirement(requirement, probes);
    if (!probe.ok) {
      emit('bench.task.skip', { id: task.id, reason: probe.detail });
      return skippedResult(task, probe.detail);
    }
  }

  const repoPath = path.join(workDir, task.id);
  const startedAt = Date.now();
  emit('bench.task.start', { id: task.id, title: task.title, language: task.language });

  let seeded = null;
  try {
    seeded = seedBenchRepo(task, repoPath);
  } catch (err) {
    const reason = err?.message ? String(err.message) : 'could not seed the task repository';
    emit('bench.task.end', { id: task.id, verdict: 'ERROR', reason });
    return emptyResult(task, { verdict: 'ERROR', reason, durationMs: Date.now() - startedAt });
  }

  let runResult = null;
  let reason = null;
  try {
    runResult = await runTask({
      task: {
        title: task.title,
        text: task.text,
        benchTask: task.id,
        forbiddenPaths: task.forbiddenPaths.length > 0 ? task.forbiddenPaths : undefined,
      },
      repoPath: seeded.repoPath,
      config,
      store,
      driver,
      providers,
      signal,
      mode: config.mode,
      onEvent,
    });
  } catch (err) {
    reason = err?.message ? String(err.message) : 'the run failed';
  }

  const runId = runResult?.runId ?? null;
  const counters = runId ? benchCountersFor(store, runId) : {
    rungReached: 0, candidates: 0, turns: 0, escalations: 0, repairs: 0, subtasks: 0,
  };

  let hidden = { hiddenPass: null, reason: reason ?? 'the run produced no winner', where: null };
  if (runResult) {
    try {
      hidden = await judgeHidden({
        task,
        repoPath: seeded.repoPath,
        driver,
        runId,
        winner: runResult.winner ?? null,
        sha: runResult.sha ?? null,
        timeoutMs: hiddenTimeoutMs,
      });
    } catch (err) {
      hidden = { hiddenPass: null, reason: err?.message ? String(err.message) : 'hidden judgement failed', where: null };
    }
  }

  const durationMs = Date.now() - startedAt;
  const result = emptyResult(task, {
    verdict: runResult?.verdict ?? (reason ? 'ERROR' : null),
    hiddenPass: hidden.hiddenPass,
    costUsd: Number(runResult?.costUsd) || 0,
    durationMs,
    runId,
    reason: reason ?? hidden.reason ?? null,
    ...counters,
  });
  emit('bench.task.end', { id: task.id, verdict: result.verdict, hiddenPass: result.hiddenPass, costUsd: result.costUsd });
  if (!keepWork) attempt(() => fs.rmSync(repoPath, { recursive: true, force: true }));
  return result;
}

// ---------------------------------------------------------------------------
// rendering
// ---------------------------------------------------------------------------

function hiddenCell(value) {
  if (value === true) return 'PASS';
  if (value === false) return 'FAIL';
  return '-';
}

/**
 * The bench table an operator reads: one row per task plus a summary line.
 * ASCII only, no box drawing.
 *
 * @param {object} result a `runBench` document
 * @returns {string}
 */
export function formatBenchTable(result) {
  const results = Array.isArray(result?.results) ? result.results : [];
  const summary = result?.summary ?? {};
  const rows = results.map((r) => [
    r.id,
    r.verdict ?? '-',
    hiddenCell(r.hiddenPass),
    r.rungReached ?? 0,
    r.candidates ?? 0,
    r.turns ?? 0,
    r.escalations ?? 0,
    r.repairs ?? 0,
    r.subtasks ?? 0,
    usd(r.costUsd),
    seconds(r.durationMs),
  ]);
  const table = renderTable(
    ['TASK', 'VERDICT', 'HIDDEN', 'RUNG', 'CAND', 'TURNS', 'ESC', 'REP', 'SUB', 'COST', 'TIME'],
    rows,
  );
  const skipped = results.filter((r) => r.hiddenPass === null).length;
  const judged = results.length - skipped;
  const rate = Number(summary.passRate) || 0;
  const lines = [
    table,
    '',
    `${summary.tasks ?? results.length} tasks  ${summary.hiddenPass ?? 0}/${judged} hidden pass`
      + ` (${Math.round(rate * 100)}%)  ${skipped} not judged`
      + `  ${usd(summary.costUsd)}  ${seconds(summary.durationMs)}`,
  ];
  const notes = results.filter((r) => typeof r.reason === 'string' && r.reason !== '');
  if (notes.length > 0) {
    lines.push('');
    for (const r of notes) lines.push(`note ${r.id}: ${r.reason}`);
  }
  return lines.join('\n');
}

function indexResults(document) {
  const map = new Map();
  for (const r of Array.isArray(document?.results) ? document.results : []) map.set(String(r.id), r);
  return map;
}

function changeOf(before, after) {
  if (!before) return 'new';
  if (!after) return 'gone';
  if (before.hiddenPass === after.hiddenPass) return 'same';
  if (after.hiddenPass === true) return 'fixed';
  if (before.hiddenPass === true) return 'broken';
  return 'changed';
}

function delta(before, after, format) {
  if (before === null || after === null) return '-';
  const diff = Number(after) - Number(before);
  if (!Number.isFinite(diff)) return '-';
  const sign = diff > 0 ? '+' : diff < 0 ? '-' : '';
  return `${sign}${format(Math.abs(diff))}`;
}

/**
 * Per-task delta between two bench documents, oldest first. Tasks that exist
 * on only one side are shown as `new` or `gone` rather than dropped, because a
 * task that quietly stopped running is exactly what a regression looks like.
 *
 * @param {object} a the earlier bench document
 * @param {object} b the later bench document
 * @returns {string}
 */
export function compareBench(a, b) {
  const before = indexResults(a);
  const after = indexResults(b);
  const ids = [...new Set([...before.keys(), ...after.keys()])].sort();
  const rows = ids.map((id) => {
    const x = before.get(id) ?? null;
    const y = after.get(id) ?? null;
    return [
      id,
      hiddenCell(x?.hiddenPass ?? null),
      hiddenCell(y?.hiddenPass ?? null),
      changeOf(x, y),
      x ? usd(x.costUsd) : '-',
      y ? usd(y.costUsd) : '-',
      x && y ? delta(x.costUsd, y.costUsd, (v) => v.toFixed(4)) : '-',
      x && y ? delta(x.durationMs, y.durationMs, (v) => seconds(v)) : '-',
    ];
  });
  const table = renderTable(
    ['TASK', 'HIDDEN A', 'HIDDEN B', 'CHANGE', 'COST A', 'COST B', 'D COST', 'D TIME'],
    rows,
  );
  const fixed = rows.filter((r) => r[3] === 'fixed').length;
  const broken = rows.filter((r) => r[3] === 'broken').length;
  const passA = Number(a?.summary?.hiddenPass) || 0;
  const passB = Number(b?.summary?.hiddenPass) || 0;
  const costA = Number(a?.summary?.costUsd) || 0;
  const costB = Number(b?.summary?.costUsd) || 0;
  return [
    table,
    '',
    `hidden pass ${passA} -> ${passB}  fixed ${fixed}  broken ${broken}`
      + `  cost ${usd(costA)} -> ${usd(costB)} (${delta(costA, costB, (v) => v.toFixed(4))})`,
  ].join('\n');
}

export default {
  BENCH_TASKS_DIR,
  BENCH_RESULT_VERSION,
  listBenchTasks,
  runBench,
  formatBenchTable,
  compareBench,
  seedBenchRepo,
  hiddenCommandFor,
  benchCountersFor,
  probeRequirement,
};

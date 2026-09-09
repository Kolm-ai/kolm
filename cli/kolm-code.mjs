#!/usr/bin/env node
// kolm code - the confidence-cascade coding agent CLI (owner E, spec section 7).
//
//   kolm code run "fix the flaky retry in src/queue.js"
//
// Everything this file needs comes from `src/ladder/index.js` (the single public
// surface) plus the GitHub and Truffle helpers, which are imported lazily so a
// plain `kolm code help` never pays for them.
//
// House rules honoured here: ASCII only (no emoji, no box drawing), never print
// a secret VALUE (only the variable name that would carry it), never crash with
// a stack trace on a LadderError, and map every failure onto kolm's exit codes.

import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';

import * as ladder from '../src/ladder/index.js';

const { LadderError, isLadderError } = ladder;

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..');

/** kolm's exit-code convention (cli/kolm.js line 77). */
export const EXIT = Object.freeze({
  OK: 0,
  BAD_ARGS: 1,
  GATE_FAIL: 2,
  MISSING_PREREQ: 3,
  EXECUTION: 4,
  NOT_FOUND: 5,
  USAGE: 64,
});

/** LadderError.code -> exit code. Anything unlisted is an execution failure. */
const CODE_EXIT = Object.freeze({
  'not-armed': EXIT.MISSING_PREREQ,
  'sqlite-unavailable': EXIT.MISSING_PREREQ,
  'not-found': EXIT.NOT_FOUND,
  aborted: EXIT.EXECUTION,
  'bad-args': EXIT.BAD_ARGS,
  'bad-arg': EXIT.BAD_ARGS,
  'bad-config': EXIT.BAD_ARGS,
  'bad-ref': EXIT.BAD_ARGS,
  'bad-bands': EXIT.BAD_ARGS,
  'budget-exceeded': EXIT.GATE_FAIL,
});

const VERBS = Object.freeze([
  'run',
  'resume',
  'status',
  'show',
  'funnel',
  'eval',
  'bands',
  'bench',
  'lessons',
  'clean',
  'receipt',
  'keys',
  'doctor',
  'serve',
  'truffle-worker',
  'help',
]);

/** Stage ids in cascade order, plus the judge rung the viewer also shows. */
const STAGE_IDS = Object.freeze(['l0-meta', 'l1-static', 'l2-dynamic', 'l3-semantic', 'l4-judge']);

// ---------------------------------------------------------------------------
// argument parsing
// ---------------------------------------------------------------------------

/** Flags that never consume the next token. */
const BOOLEAN_FLAGS = new Set([
  'json',
  'quiet',
  'keepWorktrees',
  'pr',
  'live',
  'open',
  'suggest',
  'apply',
  'yes',
  'stdin',
  'events',
  'dryRun',
  'clear',
  'keepWork',
  'help',
  'version',
]);

/** Statuses after which a run's worktrees are safe to remove. */
export const FINISHED_RUN_STATUSES = Object.freeze(['done', 'error', 'fail']);

/** Flags that accumulate; `--coder a --coder b,c` -> ['a','b','c']. */
const LIST_FLAGS = new Set(['coder', 'reviewer']);

function camel(name) {
  return String(name).replace(/-([a-z0-9])/g, (_, ch) => ch.toUpperCase());
}

/**
 * A dependency-free flag parser: `--k v`, `--k=v`, boolean flags, `--` ends
 * option parsing. Exported so the tests can exercise it without a subprocess.
 *
 * @param {string[]} argv
 * @param {{booleans?: Set<string>, lists?: Set<string>}} [opts]
 * @returns {{positional: string[], flags: Record<string, string|boolean|string[]>, passthrough: string[]}}
 */
export function parseArgs(argv = [], opts = {}) {
  const booleans = opts.booleans ?? BOOLEAN_FLAGS;
  const lists = opts.lists ?? LIST_FLAGS;
  const positional = [];
  const passthrough = [];
  const flags = {};
  let terminated = false;

  for (let i = 0; i < argv.length; i += 1) {
    const token = String(argv[i]);
    if (terminated) {
      passthrough.push(token);
      continue;
    }
    if (token === '--') {
      terminated = true;
      continue;
    }
    if (token === '-h') {
      flags.help = true;
      continue;
    }
    if (token.startsWith('--') && token.length > 2) {
      let key = token.slice(2);
      let value;
      let inline = false;
      const eq = key.indexOf('=');
      if (eq !== -1) {
        value = key.slice(eq + 1);
        key = key.slice(0, eq);
        inline = true;
      }
      const name = camel(key);
      if (!inline) {
        const next = argv[i + 1];
        if (booleans.has(name) || next === undefined || (String(next).startsWith('--') && String(next).length > 2)) {
          value = true;
        } else {
          value = String(next);
          i += 1;
        }
      }
      if (lists.has(name)) {
        const parts = value === true
          ? []
          : String(value)
            .split(',')
            .map((s) => s.trim())
            .filter(Boolean);
        flags[name] = (flags[name] ?? []).concat(parts);
      } else {
        flags[name] = value;
      }
      continue;
    }
    positional.push(token);
  }

  return { positional, flags, passthrough };
}

// ---------------------------------------------------------------------------
// tiny output helpers (ASCII only)
// ---------------------------------------------------------------------------

function out(line = '') {
  process.stdout.write(`${line}\n`);
}

function note(line = '') {
  process.stderr.write(`${line}\n`);
}

function usd(value) {
  const n = Number(value);
  return `$${(Number.isFinite(n) ? n : 0).toFixed(4)}`;
}

function seconds(ms) {
  const n = Number(ms);
  if (!Number.isFinite(n)) return '-';
  return `${(n / 1000).toFixed(1)}s`;
}

function truncate(text, limit) {
  const s = String(text ?? '');
  return s.length > limit ? `${s.slice(0, limit - 1)}~` : s;
}

/**
 * Space-padded ASCII table with a dashed rule. No box-drawing characters.
 * @param {string[]} headers
 * @param {Array<Array<string|number>>} rows
 * @returns {string}
 */
export function renderTable(headers, rows) {
  const cells = [headers.map((h) => String(h)), ...rows.map((r) => r.map((c) => (c === null || c === undefined ? '-' : String(c))))];
  const widths = headers.map((_, i) => Math.max(...cells.map((row) => (row[i] ?? '').length)));
  const line = (row) => row.map((cell, i) => String(cell ?? '').padEnd(widths[i])).join('  ').trimEnd();
  const rule = widths.map((w) => '-'.repeat(w)).join('  ');
  return [line(cells[0]), rule, ...cells.slice(1).map(line)].join('\n');
}

function jsonOut(value) {
  out(JSON.stringify(value, null, 2));
}

// ---------------------------------------------------------------------------
// owner-D surface guard
// ---------------------------------------------------------------------------

/**
 * Resolve a function from the ladder index, with a legible error when the
 * orchestrator half of the build has not exported it (yet).
 * @param {string} name
 * @returns {Function}
 */
function api(name) {
  const fn = ladder[name];
  if (typeof fn !== 'function') {
    throw new LadderError('bad-config', `src/ladder/index.js does not export ${name}()`, { missing: name });
  }
  return fn;
}

function requireSqlite() {
  if (typeof ladder.sqliteAvailable === 'function' && !ladder.sqliteAvailable()) {
    throw new LadderError(
      'sqlite-unavailable',
      `node:sqlite is unavailable on Node ${process.version}; kolm code needs Node >= 22.5`,
      { node: process.version },
    );
  }
}

// ---------------------------------------------------------------------------
// config overlay: CLI flags are applied as environment overrides, because
// loadConfig() returns a deep-frozen object that cannot be patched afterwards.
// ---------------------------------------------------------------------------

/**
 * @param {Record<string, any>} flags
 * @param {Record<string, string|undefined>} [base]
 * @returns {Record<string, string|undefined>}
 */
export function envOverlay(flags = {}, base = process.env) {
  const env = { ...base };
  const set = (name, value) => {
    if (value === undefined || value === null || value === '' || value === true) return;
    env[name] = String(value);
  };
  set('KOLM_CODE_MODE', flags.mode);
  set('KOLM_CODE_DRIVER', flags.driver);
  set('KOLM_CODE_CANDIDATES', flags.candidates);
  set('KOLM_CODE_ORCHESTRATOR', flags.orchestrator);
  set('KOLM_CODE_JUDGE', flags.judge);
  set('KOLM_CODE_MAX_COST_USD', flags.maxCost);
  set('KOLM_CODE_BANDS_PATH', flags.bandsPath);
  set('KOLM_CODE_DB_PATH', flags.db);
  set('TRUFFLE_PARTNER_POLL_MS', flags.pollMs);
  if (Array.isArray(flags.coder) && flags.coder.length > 0) {
    env.KOLM_CODE_CODER_LADDER = flags.coder.join(',');
  }
  if (Array.isArray(flags.reviewer) && flags.reviewer.length > 0) {
    env.KOLM_CODE_REVIEWER_LADDER = flags.reviewer.join(',');
  }
  if (flags.keepWorktrees === true) env.KOLM_CODE_KEEP_WORKTREES = 'all';
  return env;
}

/**
 * Secrets an operator stored with `kolm code keys set`, ready to hand to
 * `loadConfig` as `opts.fileSecrets`. `KOLM_CODE_NO_KEYSTORE=1` makes this an
 * empty object, so CI and the Truffle worker stay env-only. A corrupt keystore
 * raises a LadderError naming the file rather than silently arming nothing.
 * @param {Record<string, string|undefined>} env
 * @returns {Record<string, string>}
 */
function fileSecretsFrom(env) {
  const store = api('readKeystore')(env);
  return store.enabled ? store.secrets : {};
}

function configFrom(flags) {
  const env = envOverlay(flags);
  return api('loadConfig')(env, { fileSecrets: fileSecretsFrom(env) });
}

/**
 * Config for verbs that will actually call models. Ollama rungs are armed by
 * reachability, which `loadConfig` cannot know without a probe, so probe once
 * here when any configured ref points at Ollama (doctor --live does the same).
 * @param {Record<string, any>} flags
 * @returns {Promise<object>}
 */
async function configForModels(flags) {
  const config = configFrom(flags);
  if (config.mockProvider === true || config.ollamaReachable !== null) return config;
  const refs = [config.orchestratorRef, config.judgeRef, ...(config.coderLadder ?? []), ...(config.reviewerLadder ?? [])];
  if (!refs.some((ref) => typeof ref === 'string' && ref.startsWith('ollama:'))) return config;
  const probe = await api('probeOllama')(config);
  return api('withOllamaReachable')(config, probe.reachable);
}

function openStoreFor(config) {
  requireSqlite();
  return api('openStore')(config.dbPath);
}

// ---------------------------------------------------------------------------
// usage
// ---------------------------------------------------------------------------

const USAGE = `kolm code - confidence-cascade coding agent

usage: kolm code <verb> [options]

verbs:
  run "<task>"            generate candidates, run them up the cascade, keep the winner
  resume <runId>          continue an interrupted run, or reopen one that did not pass
  status [runId]          list runs newest-first, or show one run's candidates and stages
  show <runId>            full detail for one run
  funnel                  how many candidates cleared cheaply vs went to the judge
  eval                    cascade vs direct comparison over stored runs
  bands                   print, suggest or apply the confidence bands
  bench                   run the bundled coding benchmark and score the hidden tests
  lessons <repo>          print (or clear) the lessons remembered for a repository
  clean                   remove worktrees left behind by finished runs
  receipt verify|show     verify or print a signed evidence receipt
  keys set|list|unset|path|test
                          store your own provider keys once, in <data>/code/keys.json
  doctor                  arming table for every model rung, the driver and the store
  serve                   run the local viewer
  truffle-worker          claim and settle Truffle partner tasks
  help                    this text

run options:
  --repo <path>           repository to work in (default: the current directory)
  --repo-url <url>        clone source recorded on the task
  --base <ref>            base ref for the worktrees (default: HEAD)
  --mode cascade|direct   verification mode (default: cascade)
  --driver local|cloudflare|fake
  --candidates <n>        how many candidates to generate
  --coder <ref,...>       coder ladder, repeatable (e.g. --coder ollama:qwen3.8:latest)
  --reviewer <ref,...>    reviewer ladder for the L3 rubric rung
  --judge <ref>           L4 judge model
  --orchestrator <ref>    planner / repair model
  --max-cost <usd>        budget ceiling for the run
  --keep-worktrees        keep every worktree, not only the winner's
  --pr                    open a draft pull request when the run passes
  --pr-base <branch>      base branch for that pull request (default: main)
  --json                  print the run summary as JSON
  --quiet                 do not stream stage events to stderr

bench options:
  --tasks <a,b>           only these bench task ids (default: all ten)
  --concurrency <n>       bench tasks in flight at once (default 1)
  --compare <file>        also print a delta table against an earlier result file
  --keep-work             keep the seeded task repositories for inspection
  --json                  print the bench result document as JSON
                          run's model flags apply here too (--coder, --judge, ...)
                          exit 0 when every non-skipped task passes its hidden test

lessons options:
  <repo>                  repository path or url whose lessons to show
  --clear                 forget every lesson for that repository
  --json                  machine readable output

keys options:
  names                   anthropic cerebras deepseek openrouter github sandbox
  --from-env <VAR>        keys set: take the value from that variable, unprinted
  --from-file <path>      keys set: take the first line of that file
  --stdin                 keys set: read the value from stdin (the default when
                          stdin is a pipe; a TTY gets a hidden prompt instead)
  --json                  keys list / keys test: machine readable output
  environment             KOLM_CODE_KEYS_PATH moves the file,
                          KOLM_CODE_NO_KEYSTORE=1 makes every verb env-only

clean options:
  --run <id>              clean one run instead of every finished run
  --older-than <days>     only runs last touched more than N days ago
  --dry-run               list what would go without removing anything
  --json                  machine readable output
                          running runs are never touched and branches
                          (kolm/code/<runId>/<candidate>) are never deleted

other options:
  --since <iso>           funnel/eval window
  --limit <n>             status list length
  --events                show: include the event list in --json output
  --suggest               bands: print proposed thresholds from stored runs
  --apply --yes           bands: write the proposal (never over the bundled file)
  --live                  doctor: probe Ollama, gh, the driver and every armed model
  --port <n>              serve: listen port (default 7474)
  --host <addr>           serve: bind address (default 127.0.0.1)
  --open                  serve: open the viewer in the OS browser
  --max-tasks <n>         truffle-worker: stop after N settled tasks
  --poll-ms <n>           truffle-worker: claim poll interval

exit codes:
  0 ok   1 bad arguments   2 cascade FAIL or JUDGE   3 not armed
  4 execution error        5 not found               64 usage

environment and the full operator guide: docs/KOLM_CODE.md`;

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

function formatEvent(event) {
  const type = String(event?.type ?? 'event');
  const payload = event && typeof event.payload === 'object' && event.payload !== null ? event.payload : {};
  const stage = payload.stage ?? payload.stageId ?? null;
  const who = payload.candidateId ?? payload.candidate_id ?? payload.candidate ?? null;
  const head = `[${stage || type}${who ? ` ${who}` : ''}]`;
  const bits = [];
  if (payload.decision) bits.push(String(payload.decision));
  if (payload.verdict) bits.push(String(payload.verdict));
  if (Number.isFinite(payload.turn)) bits.push(`turn ${payload.turn}`);
  if (Array.isArray(payload.tools) && payload.tools.length > 0) bits.push(payload.tools.join(','));
  if (payload.tool) bits.push(String(payload.tool));
  if (Number.isFinite(payload.durationMs)) bits.push(seconds(payload.durationMs));
  if (Number.isFinite(payload.costUsd) && payload.costUsd > 0) bits.push(usd(payload.costUsd));
  if (bits.length === 0 && payload.message) bits.push(truncate(payload.message, 140));
  if (bits.length === 0 && stage) bits.push(type);
  if (bits.length === 0) {
    // Scalars only: an object would stringify to "[object Object]" and an empty
    // string would render as a bare "key=" that tells the operator nothing.
    const rest = Object.entries(payload)
      .filter(([, v]) => (typeof v === 'string' ? v !== '' : ['number', 'boolean'].includes(typeof v)))
      .slice(0, 3)
      .map(([k, v]) => `${k}=${truncate(v, 40)}`);
    if (rest.length > 0) bits.push(rest.join(' '));
    else bits.push(type);
  }
  return `${head} ${bits.join(' ')}`.trimEnd();
}

function summarizeRun(result) {
  const rows = [
    ['run', result.runId],
    ['verdict', result.verdict],
    ['winner', result.winner ?? '-'],
    ['branch', result.branch ?? '-'],
    ['sha', result.sha ?? '-'],
    ['cost', usd(result.costUsd)],
    ['elapsed', seconds(result.durationMs)],
    ['receipt', result.receiptPath ?? '-'],
  ];
  if (result.stopReason) rows.push(['stopped', String(result.stopReason)]);
  const funnel = result.funnel;
  if (funnel && typeof funnel === 'object') {
    const cleared = Number(funnel.cleared_cheaply ?? funnel.clearedCheaply ?? 0);
    rows.push([
      'funnel',
      `${funnel.candidates ?? 0} candidates, ${(cleared * 100).toFixed(0)}% cleared cheaply, ${funnel.judged ?? 0} judged`,
    ]);
  }
  return rows.map(([k, v]) => `${k.padEnd(9)}${v}`).join('\n');
}

async function cmdRun(parsed) {
  const flags = parsed.flags;
  const text = parsed.positional.slice(1).join(' ').trim();
  if (!text) {
    throw new LadderError('bad-args', 'run needs a task: kolm code run "<task>"');
  }

  const config = await configForModels(flags);
  const arming = api('armingReport')(config);
  const ladders = api('effectiveLadders')(config, arming);

  const repoPath = path.resolve(String(flags.repo ?? process.cwd()));
  // Everything checkable BEFORE a run row exists is checked here: a typo'd
  // --repo used to cost an error row and an empty worktree directory, and the
  // message came back three times (run.error event, logger, reportError).
  validateRunPreconditions({
    repoPath,
    baseRef: flags.base ? String(flags.base) : null,
    config,
    arming,
    ladders,
  });
  for (const line of droppedRungLines(arming, ladders)) note(line);

  const store = openStoreFor(config);
  const driver = api('createDriver')(config.driver, config);
  const createProvider = api('createProvider');
  const providers = {
    orchestrator: createProvider(config.orchestratorRef, config),
    judge: createProvider(config.judgeRef, config),
    coders: ladders.coders.map((ref) => createProvider(ref, config)),
    reviewer: ladders.reviewer ? createProvider(ladders.reviewer, config) : null,
  };

  const task = {
    text,
    title: flags.title ? String(flags.title) : truncate(text, 72),
    repoUrl: flags.repoUrl ? String(flags.repoUrl) : undefined,
    baseRef: flags.base ? String(flags.base) : undefined,
  };

  const controller = new AbortController();
  const streaming = flags.quiet !== true;
  let observedRunId = null;
  let interrupted = false;

  const onSigint = () => {
    interrupted = true;
    controller.abort();
    note('');
    note(`aborted; resume with: kolm code resume ${observedRunId ?? '<runId>'}`);
  };
  process.on('SIGINT', onSigint);

  try {
    const result = await api('runTask')({
      task,
      repoPath,
      repoUrl: task.repoUrl,
      baseRef: task.baseRef,
      config,
      store,
      driver,
      providers,
      logger: console,
      signal: controller.signal,
      mode: config.mode,
      bands: config.bandsPath ? api('loadBands')(config.bandsPath) : undefined,
      onEvent: (event) => {
        if (event?.payload?.runId) observedRunId = event.payload.runId;
        if (streaming) note(formatEvent(event));
      },
    });
    observedRunId = result?.runId ?? observedRunId;

    if (flags.json) jsonOut(sanitizeRunResult(result));
    else out(summarizeRun(result));

    if (flags.pr === true && result.verdict === 'PASS') {
      const prUrl = await openPullRequestFor({ config, result, repoPath, flags });
      if (prUrl) out(`pr       ${prUrl}`);
    }

    if (interrupted) return EXIT.EXECUTION;
    return result.verdict === 'PASS' ? EXIT.OK : EXIT.GATE_FAIL;
  } finally {
    process.off('SIGINT', onSigint);
    try {
      store.close();
    } catch {
      // the store is already closed; nothing to unwind
    }
  }
}

/**
 * One line per ladder rung that `effectiveLadders` left out because it is
 * not armed. The run still works (the orchestrator codes when no coder rung
 * is armed), but a ladder that quietly collapsed onto one model is the kind
 * of surprise an operator should see at run start, not in the ledger.
 *
 * @param {object} arming output of `armingReport`
 * @param {object} ladders output of `effectiveLadders`
 * @returns {string[]} lines for stderr; empty when every rung is armed
 */
export function droppedRungLines(arming, ladders) {
  const rungs = arming?.rungs ?? [];
  const lines = [];
  const coders = new Set(ladders?.coders ?? []);
  const unarmedCoders = rungs.filter((r) => r.kind === 'coder' && !r.armed);
  for (const r of unarmedCoders) {
    if (coders.has(r.ref)) continue;
    lines.push(`skipping coder rung ${r.ref}: ${r.reason ?? 'not armed'}`);
  }
  const anyCoderArmed = rungs.some((r) => r.kind === 'coder' && r.armed);
  if (unarmedCoders.length > 0 && !anyCoderArmed) {
    lines.push(`no coder rung is armed; the orchestrator ${ladders?.coders?.[0] ?? ''} writes the candidates`.trimEnd());
  }
  const unarmedReviewers = rungs.filter((r) => r.kind === 'reviewer' && !r.armed);
  if (unarmedReviewers.length > 0 && !ladders?.reviewer) {
    for (const r of unarmedReviewers) {
      lines.push(`skipping reviewer rung ${r.ref}: ${r.reason ?? 'not armed'}`);
    }
    lines.push('no reviewer rung is armed; l3-semantic is skipped and unsure candidates go straight to the judge');
  }
  return lines;
}

/**
 * Everything `run` can check before `createRun` writes a row: a real directory,
 * a real git repository, a base ref git can resolve, and an armed model on
 * every rung this run will actually use. Failing here costs nothing and prints
 * one line; failing later leaves a status=error row behind.
 *
 * @param {object} args `{repoPath, baseRef, config, arming, ladders}`
 * @param {object} [deps] `{statSync, git}` injected by the tests
 * @returns {true}
 */
export function validateRunPreconditions(args, deps = {}) {
  const { repoPath, baseRef, config, arming, ladders } = args;
  const statSync = deps.statSync ?? fs.statSync;
  const git = deps.git ?? ((gitArgs, cwd) => spawnSync('git', gitArgs, { cwd, encoding: 'utf8' }));

  if ((config?.driver ?? 'local') === 'local') {
    let stat = null;
    try {
      stat = statSync(repoPath);
    } catch {
      stat = null;
    }
    if (!stat || !stat.isDirectory()) {
      throw new LadderError('bad-args', `--repo is not a directory: ${repoPath}`, { repoPath });
    }
    const inside = git(['rev-parse', '--git-dir'], repoPath);
    if (!inside || inside.status !== 0) {
      throw new LadderError('bad-args', `not a git repository: ${repoPath}`, { repoPath });
    }
    if (baseRef) {
      const resolved = git(['rev-parse', '--verify', `${baseRef}^{commit}`], repoPath);
      if (!resolved || resolved.status !== 0) {
        throw new LadderError('bad-ref', `--base ref does not resolve in ${repoPath}: ${baseRef}`, {
          repoPath,
          baseRef,
        });
      }
    }
  }

  if (config?.mockProvider === true) return true;

  const rungs = arming?.rungs ?? [];
  const used = [];
  const orchestrator = rungs.find((r) => r.id === 'orchestrator');
  if (orchestrator) used.push(orchestrator);
  if ((config?.mode ?? 'cascade') === 'cascade') {
    const judge = rungs.find((r) => r.id === 'judge');
    if (judge) used.push(judge);
  }
  const coderRefs = new Set(ladders?.coders ?? []);
  for (const rung of rungs) {
    if (rung.kind === 'coder' && coderRefs.has(rung.ref)) used.push(rung);
  }
  const unarmed = used.filter((r) => !r.armed);
  if (unarmed.length > 0) {
    const detail = unarmed.map((r) => `${r.id} (${r.ref ?? '-'}): ${r.reason}`).join('; ');
    throw new LadderError(
      'not-armed',
      `no armed model for ${unarmed.map((r) => r.id).join(', ')} - ${detail}. Run "kolm code doctor" for the full table.`,
      { rungs: unarmed.map((r) => r.id) },
    );
  }
  return true;
}

/** Drop the diff blobs so `--json` stays a summary, not a patch dump. */
function sanitizeRunResult(result) {
  const copy = { ...result };
  if (Array.isArray(copy.candidates)) {
    copy.candidates = copy.candidates.map((c) => {
      const entry = { ...c };
      if (typeof entry.diff === 'string') entry.diffBytes = Buffer.byteLength(entry.diff);
      delete entry.diff;
      delete entry.diff_text;
      return entry;
    });
  }
  return copy;
}

async function openPullRequestFor({ config, result, repoPath, flags }) {
  const [{ resolveGitHubToken }, { openPrForRun, buildPrBody }] = await Promise.all([
    import('../src/ladder/github/app-auth.js'),
    import('../src/ladder/github/pr.js'),
  ]);
  const resolved = await resolveGitHubToken({ config });
  if (!resolved?.token) {
    throw new LadderError(
      'not-armed',
      'no GitHub credential: set KOLM_GITHUB_TOKEN or GITHUB_TOKEN, install the GitHub App, or run "gh auth login"',
      { reasons: resolved?.reasons ?? [] },
    );
  }
  const body = buildPrBody({
    receipt: result.receipt ?? null,
    runSummary: result,
    receiptPath: result.receiptPath,
  });
  const pr = await openPrForRun({
    repoPath,
    branch: result.branch,
    base: flags.prBase ? String(flags.prBase) : 'main',
    title: result.receipt?.task?.title ?? `kolm code ${result.runId}`,
    body,
    token: resolved.token,
    draft: true,
  });
  return pr.htmlUrl ?? pr.url ?? null;
}

// ---------------------------------------------------------------------------
// resume
// ---------------------------------------------------------------------------

async function cmdResume(parsed) {
  const runId = parsed.positional[1];
  if (!runId) throw new LadderError('bad-args', 'resume needs a run id: kolm code resume <runId>');
  const flags = parsed.flags;
  const config = await configForModels(flags);
  const store = openStoreFor(config);
  try {
    const stored = store.getRun(runId);
    if (!stored) {
      throw new LadderError('not-found', `no run ${runId} in ${config.dbPath}`, { runId });
    }
    // resumeTask() delegates straight to runTask(), which still validates the
    // task and the repo path, so rehydrate both from the stored run row.
    const task = stored.task && typeof stored.task === 'object' ? stored.task : null;
    if (!task || !String(task.text ?? '').trim()) {
      throw new LadderError('bad-config', `run ${runId} has no stored task text to resume from`, { runId });
    }
    const repoPath = stored.repo_path ?? task.repoPath ?? process.cwd();
    const driver = api('createDriver')(config.driver, config);
    const arming = api('armingReport')(config);
    const ladders = api('effectiveLadders')(config, arming);
    for (const line of droppedRungLines(arming, ladders)) note(line);
    const createProvider = api('createProvider');
    const controller = new AbortController();
    const onSigint = () => {
      controller.abort();
      note('');
      note(`aborted; resume with: kolm code resume ${runId}`);
    };
    process.on('SIGINT', onSigint);
    try {
      const result = await api('resumeTask')({
        runId,
        task,
        repoPath,
        repoUrl: task.repoUrl ?? null,
        baseRef: task.baseRef ?? stored.base_sha ?? undefined,
        mode: stored.mode ?? config.mode,
        config,
        store,
        driver,
        providers: {
          orchestrator: createProvider(config.orchestratorRef, config),
          judge: createProvider(config.judgeRef, config),
          coders: ladders.coders.map((ref) => createProvider(ref, config)),
          reviewer: ladders.reviewer ? createProvider(ladders.reviewer, config) : null,
        },
        logger: console,
        signal: controller.signal,
        onEvent: (event) => {
          if (flags.quiet !== true) note(formatEvent(event));
        },
      });
      if (flags.json) jsonOut(sanitizeRunResult(result));
      else out(summarizeRun(result));
      return result.verdict === 'PASS' ? EXIT.OK : EXIT.GATE_FAIL;
    } finally {
      process.off('SIGINT', onSigint);
    }
  } finally {
    store.close();
  }
}

// ---------------------------------------------------------------------------
// status / show
// ---------------------------------------------------------------------------

function stageDecision(results, stage) {
  const entry = results?.[stage];
  if (!entry || !entry.verdict) return '-';
  return String(entry.verdict.decision ?? 'skip');
}

async function cmdStatus(parsed) {
  const flags = parsed.flags;
  const config = configFrom(flags);
  const store = openStoreFor(config);
  try {
    const runId = parsed.positional[1];
    if (!runId) {
      const limit = Number(flags.limit ?? 20);
      const runs = store.listRuns({ limit: Number.isFinite(limit) ? limit : 20 });
      if (flags.json) {
        jsonOut({ runs });
        return EXIT.OK;
      }
      if (runs.length === 0) {
        out(`no runs recorded in ${config.dbPath}`);
        return EXIT.OK;
      }
      out(
        renderTable(
          ['RUN', 'STATUS', 'WINNER', 'COST', 'MODE', 'CREATED', 'TASK'],
          runs.map((run) => [
            run.id,
            run.status ?? '-',
            run.winner_candidate ?? '-',
            usd(run.total_cost_usd),
            run.mode ?? '-',
            String(run.created_at ?? '').slice(0, 19),
            truncate(run.task?.title ?? run.task?.text ?? '-', 48),
          ]),
        ),
      );
      return EXIT.OK;
    }

    const run = store.getRun(runId);
    if (!run) throw new LadderError('not-found', `no run ${runId} in ${config.dbPath}`, { runId });
    const candidates = store.listCandidates(runId);
    if (flags.json) {
      jsonOut({
        run,
        candidates: candidates.map((c) => ({ ...c, diff_text: undefined, results: store.latestResults(runId, c.candidate_id) })),
      });
      return EXIT.OK;
    }
    out(`run ${run.id}  ${run.status ?? '-'}  ${usd(run.total_cost_usd)}`);
    out(`task ${truncate(run.task?.title ?? run.task?.text ?? '-', 100)}`);
    out('');
    out(
      renderTable(
        ['CANDIDATE', 'GENERATOR', 'STATUS', ...STAGE_IDS],
        candidates.map((c) => {
          const results = store.latestResults(runId, c.candidate_id);
          return [
            c.candidate_id,
            truncate(c.generator_ref ?? '-', 30),
            c.status ?? '-',
            ...STAGE_IDS.map((stage) => stageDecision(results, stage)),
          ];
        }),
      ),
    );
    return EXIT.OK;
  } finally {
    store.close();
  }
}

/**
 * Fold the orchestrator's subtask events (and the receipt's `subtasks` array,
 * when one is present) into rows the viewer can print. Defensive on purpose:
 * a run that never split into subtasks yields an empty list rather than an
 * error, and an unknown payload shape degrades to a dash.
 *
 * @param {Array<{type: string, payload: object}>} events
 * @param {object|null} [receipt] the stored receipt, if any
 * @returns {{subtasks: Array<object>, integration: {merged: number|null, conflicts: number|null, verdict: string|null}}}
 */
export function subtaskViewFrom(events = [], receipt = null) {
  const order = [];
  const byId = new Map();
  const integration = { merged: null, conflicts: null, verdict: null };

  const touch = (rawId) => {
    const id = rawId === undefined || rawId === null || rawId === '' ? null : String(rawId);
    if (id === null) return null;
    if (!byId.has(id)) {
      byId.set(id, { id, title: null, status: null, verdict: null, winner: null, branch: null, sha: null, costUsd: 0 });
      order.push(id);
    }
    return byId.get(id);
  };

  for (const event of Array.isArray(events) ? events : []) {
    const type = String(event?.type ?? '');
    const payload = event && typeof event.payload === 'object' && event.payload !== null ? event.payload : {};
    if (type === 'subtasks.plan') {
      const list = Array.isArray(payload.subtasks) ? payload.subtasks : [];
      for (const entry of list) {
        const row = touch(entry?.id ?? entry);
        if (row && typeof entry === 'object' && entry !== null) {
          row.title = entry.title ?? entry.text ?? row.title;
        }
      }
      continue;
    }
    if (type === 'subtask.retry') {
      // A resume is redoing a subtask that failed or ended JUDGE; the
      // `subtask.start` right behind this event moves the row to 'running'.
      const row = touch(payload.id);
      if (row) row.status = 'retrying';
      continue;
    }
    if (type === 'subtask.skipped') {
      // A dependent of a failed subtask never ran, so it has no start, no end
      // and no cost of its own.
      const row = touch(payload.id);
      if (row) {
        row.status = 'skipped';
        row.verdict = 'SKIPPED';
      }
      continue;
    }
    if (type === 'subtask.start') {
      const row = touch(payload.id);
      if (row) {
        row.status = 'running';
        row.title = row.title ?? payload.title ?? null;
      }
      continue;
    }
    if (type === 'subtask.end') {
      const row = touch(payload.id);
      if (row) {
        row.status = 'done';
        row.verdict = payload.verdict ?? row.verdict;
        row.winner = payload.winner ?? row.winner;
        row.costUsd += Number(payload.costUsd) || 0;
      }
      continue;
    }
    if (type === 'integration.merge') {
      if (payload.merged !== undefined) integration.merged = Number(payload.merged) || 0;
      else integration.merged = (integration.merged ?? 0) + 1;
      if (payload.conflicts !== undefined) {
        integration.conflicts = Array.isArray(payload.conflicts) ? payload.conflicts.length : Number(payload.conflicts) || 0;
      }
      continue;
    }
    if (type === 'integration.verdict') {
      integration.verdict = payload.verdict ?? payload.value ?? integration.verdict;
    }
  }

  const fromReceipt = Array.isArray(receipt?.subtasks) ? receipt.subtasks : [];
  for (const entry of fromReceipt) {
    const row = touch(entry?.id);
    if (!row) continue;
    row.winner = entry.winner ?? row.winner;
    row.branch = entry.branch ?? row.branch;
    row.sha = entry.sha ?? row.sha;
  }

  return { subtasks: order.map((id) => byId.get(id)), integration };
}

/**
 * One line each for the two events that say what the run was given and how it
 * was routed; null when the run predates them, so `show` prints nothing.
 * @param {Array<{type?: string, payload?: object}>} events
 * @returns {{context: string|null, route: string|null}}
 */
export function briefLinesFrom(events = []) {
  let context = null;
  let route = null;
  for (const event of Array.isArray(events) ? events : []) {
    const payload = event?.payload && typeof event.payload === 'object' ? event.payload : {};
    if (event?.type === 'context.pack' && context === null) {
      const files = Array.isArray(payload.instructionFiles) ? payload.instructionFiles.length : 0;
      context = `instructions ${Number(payload.instructionsBytes) || 0}B from ${files} file${files === 1 ? '' : 's'}, `
        + `lessons ${Number(payload.lessons) || 0}, `
        + `retrieval ${Number(payload.retrievalFiles) || 0} files/${Number(payload.retrievalBytes) || 0}B`
        + (payload.retrievalCached === true ? ' (cached)' : '');
    } else if (event?.type === 'run.route' && route === null) {
      route = `${payload.complexity ?? '-'}/${payload.risk ?? '-'} -> rung ${payload.startRung ?? '-'}, `
        + `${payload.candidates ?? '-'} candidate${payload.candidates === 1 ? '' : 's'}, `
        + `${payload.repairRounds ?? '-'} repair${payload.repairRounds === 1 ? '' : 's'}`
        + (payload.reason ? ` (${payload.reason})` : '');
    }
    if (context !== null && route !== null) break;
  }
  return { context, route };
}

async function cmdShow(parsed) {
  const flags = parsed.flags;
  const runId = parsed.positional[1];
  if (!runId) throw new LadderError('bad-args', 'show needs a run id: kolm code show <runId>');
  const config = configFrom(flags);
  const store = openStoreFor(config);
  try {
    const run = store.getRun(runId);
    if (!run) throw new LadderError('not-found', `no run ${runId} in ${config.dbPath}`, { runId });
    const candidates = store.listCandidates(runId).map((c) => ({
      candidate_id: c.candidate_id,
      generator_ref: c.generator_ref,
      status: c.status,
      summary: c.summary,
      diff_sha256: c.diff_sha256,
      diff_bytes: c.diff_text ? Buffer.byteLength(c.diff_text) : 0,
      results: store.latestResults(runId, c.candidate_id),
    }));
    const events = store.eventsSince(runId, 0, 100000);
    const cost = store.costSummary(runId);
    const receiptPath = typeof ladder.receiptPathFor === 'function' ? ladder.receiptPathFor(config, runId) : null;
    const view = subtaskViewFrom(events, run.receipt ?? null);
    const brief = briefLinesFrom(events);
    const detail = {
      run: { ...run, receipt: run.receipt ? { present: true } : null },
      context: brief.context,
      route: brief.route,
      subtasks: view.subtasks,
      integration: view.integration,
      candidates,
      judgments: store.listJudgments(runId),
      funnel: store.funnel({ runIds: [runId] }),
      cost,
      eventCount: events.length,
      // `--events` swaps the count for the list; without it the payload stays
      // a summary, because a long run carries thousands of events.
      events: flags.events === true ? events : events.length,
      receiptPath,
      receiptOnDisk: receiptPath ? fs.existsSync(receiptPath) : false,
    };

    if (flags.json) {
      jsonOut(detail);
      return EXIT.OK;
    }

    out(`run       ${run.id}`);
    out(`status    ${run.status ?? '-'}`);
    out(`task      ${truncate(run.task?.title ?? run.task?.text ?? '-', 100)}`);
    out(`repo      ${run.repo_path ?? '-'}`);
    out(`base sha  ${run.base_sha ?? '-'}`);
    out(`mode      ${run.mode ?? '-'}  driver ${run.driver ?? '-'}`);
    out(`winner    ${run.winner_candidate ?? '-'}`);
    out(`cost      ${usd(cost.total_usd)}`);
    out(`events    ${events.length}`);
    if (brief.context !== null) out(`context   ${brief.context}`);
    if (brief.route !== null) out(`route     ${brief.route}`);
    out(`receipt   ${receiptPath ?? '-'}${detail.receiptOnDisk ? '' : ' (not on disk)'}`);
    out('');
    out(
      renderTable(
        ['CANDIDATE', 'GENERATOR', 'STATUS', ...STAGE_IDS],
        candidates.map((c) => [
          c.candidate_id,
          truncate(c.generator_ref ?? '-', 30),
          c.status ?? '-',
          ...STAGE_IDS.map((stage) => stageDecision(c.results, stage)),
        ]),
      ),
    );
    // A run that was never split has no subtask rows; printing an empty table
    // would only add noise, so the whole block is conditional.
    if (view.subtasks.length > 0) {
      out('');
      out(
        renderTable(
          ['SUBTASK', 'STATUS', 'VERDICT', 'WINNER', 'BRANCH', 'COST'],
          view.subtasks.map((s) => [
            truncate(s.id, 24),
            s.status ?? '-',
            s.verdict ?? '-',
            truncate(s.winner ?? '-', 24),
            truncate(s.branch ?? '-', 40),
            usd(s.costUsd),
          ]),
        ),
      );
    }
    if (view.integration.verdict !== null || view.integration.merged !== null) {
      out('');
      out(
        `integration  verdict ${view.integration.verdict ?? '-'}  merged ${view.integration.merged ?? '-'}  conflicts ${view.integration.conflicts ?? '-'}`,
      );
    }
    const byModel = Object.entries(cost.by_model ?? {});
    if (byModel.length > 0) {
      out('');
      out(renderTable(['MODEL', 'CALLS', 'IN', 'OUT', 'COST'], byModel.map(([model, b]) => [model, b.calls, b.input_tokens, b.output_tokens, usd(b.cost_usd)])));
    }
    if (flags.events === true) {
      out('');
      for (const event of events) out(formatEvent(event));
    }
    return EXIT.OK;
  } finally {
    store.close();
  }
}

// ---------------------------------------------------------------------------
// clean
// ---------------------------------------------------------------------------

/**
 * Which stored runs `clean` may touch, and why the others were left alone.
 * Pure, so the tests can pin the policy without a driver or a filesystem.
 *
 * @param {Array<object>} runs store rows, newest first
 * @param {{olderThanDays?: number|null, now?: number}} [opts]
 * @returns {{clean: object[], skipped: Array<{runId:string, reason:string}>}}
 */
export function selectRunsToClean(runs = [], opts = {}) {
  const olderThanDays = opts.olderThanDays ?? null;
  const now = Number.isFinite(opts.now) ? opts.now : Date.now();
  const cutoff = olderThanDays === null ? null : now - olderThanDays * 86400000;
  const clean = [];
  const skipped = [];
  for (const run of runs) {
    const status = String(run?.status ?? '');
    if (!FINISHED_RUN_STATUSES.includes(status)) {
      // A running run owns its worktrees; removing them under it is sabotage.
      skipped.push({ runId: run?.id ?? '-', reason: `status ${status || 'unknown'}` });
      continue;
    }
    if (cutoff !== null) {
      const ts = Date.parse(run.updated_at ?? run.created_at ?? '');
      if (!Number.isFinite(ts) || ts > cutoff) {
        skipped.push({ runId: run.id, reason: `newer than ${olderThanDays} day(s)` });
        continue;
      }
    }
    const kind = run.driver ? String(run.driver) : 'local';
    if (kind !== 'local' && kind !== 'fake') {
      skipped.push({ runId: run.id, reason: `driver ${kind}` });
      continue;
    }
    clean.push(run);
  }
  return { clean, skipped };
}

async function cmdClean(parsed) {
  const flags = parsed.flags;
  const config = configFrom(flags);
  const dryRun = flags.dryRun === true;
  const olderThanDays = flags.olderThan === undefined ? null : Number(flags.olderThan);
  if (olderThanDays !== null && (!Number.isFinite(olderThanDays) || olderThanDays < 0)) {
    throw new LadderError('bad-args', `--older-than takes a number of days, got "${flags.olderThan}"`);
  }
  const store = openStoreFor(config);
  try {
    const runId = flags.run && flags.run !== true ? String(flags.run) : null;
    let runs;
    if (runId) {
      const run = store.getRun(runId);
      if (!run) throw new LadderError('not-found', `no run ${runId} in ${config.dbPath}`, { runId });
      runs = [run];
    } else {
      const limit = Number(flags.limit ?? 500);
      runs = store.listRuns({ limit: Number.isFinite(limit) ? limit : 500 });
    }

    const { clean, skipped } = selectRunsToClean(runs, { olderThanDays });
    const drivers = new Map();
    const removed = [];
    for (const run of clean) {
      const kind = run.driver ? String(run.driver) : 'local';
      if (!drivers.has(kind)) drivers.set(kind, api('createDriver')(kind, config));
      const result = await drivers.get(kind).destroyRun(run.id, {
        repoPath: run.repo_path ?? undefined,
        dryRun,
      });
      removed.push({
        runId: run.id,
        status: run.status ?? '-',
        worktrees: dryRun ? (result?.paths?.length ?? 0) : (result?.removed ?? 0),
        directory: result?.runDir ?? null,
        existed: result?.existed !== false,
      });
    }

    const worktrees = removed.reduce((sum, r) => sum + r.worktrees, 0);
    if (flags.json) {
      jsonOut({ dryRun, runs: removed, worktrees, skipped });
      return EXIT.OK;
    }
    if (removed.length === 0) {
      out('nothing to clean');
    } else {
      out(
        renderTable(
          ['RUN', 'STATUS', 'WORKTREES', 'DIRECTORY'],
          removed.map((r) => [r.runId, r.status, r.worktrees, r.directory ?? '-']),
        ),
      );
      out('');
    }
    out(`${dryRun ? 'would remove' : 'removed'}  ${removed.length} run(s), ${worktrees} worktree(s)`);
    if (skipped.length > 0) {
      const head = skipped.slice(0, 5).map((s) => `${s.runId} (${s.reason})`).join(', ');
      out(`skipped      ${skipped.length} run(s): ${head}${skipped.length > 5 ? ', ...' : ''}`);
    }
    out('branches kolm/code/<runId>/<candidate> are never deleted');
    return EXIT.OK;
  } finally {
    store.close();
  }
}

// ---------------------------------------------------------------------------
// funnel / eval
// ---------------------------------------------------------------------------

async function cmdFunnel(parsed) {
  const flags = parsed.flags;
  const config = configFrom(flags);
  const store = openStoreFor(config);
  try {
    const result = store.funnel({ since: flags.since ? String(flags.since) : undefined });
    if (flags.json) {
      jsonOut(result);
      return EXIT.OK;
    }
    const stages = result.stages ?? {};
    const ids = STAGE_IDS.filter((s) => stages[s]).concat(Object.keys(stages).filter((s) => !STAGE_IDS.includes(s)));
    out(
      renderTable(
        ['STAGE', 'GOOD', 'UNSURE', 'BAD', 'SKIP'],
        ids.map((id) => {
          const b = stages[id] ?? { good: 0, unsure: 0, bad: 0, skip: 0 };
          return [id, b.good, b.unsure, b.bad, b.skip];
        }),
      ),
    );
    out('');
    out(`runs             ${result.runs}`);
    out(`candidates       ${result.candidates}`);
    out(`cleared cheaply  ${(Number(result.cleared_cheaply ?? 0) * 100).toFixed(1)}%`);
    out(`judged           ${result.judged}`);
    out(`hard fail        ${result.hard_fail}`);
    out(`verdicts         PASS ${result.verdicts.PASS}  FAIL ${result.verdicts.FAIL}  JUDGE ${result.verdicts.JUDGE}`);
    return EXIT.OK;
  } finally {
    store.close();
  }
}

async function cmdEval(parsed) {
  const flags = parsed.flags;
  const config = configFrom(flags);
  const store = openStoreFor(config);
  try {
    const result = await api('evalRuns')({ store, since: flags.since ? String(flags.since) : undefined });
    if (flags.json) {
      jsonOut(result);
      return EXIT.OK;
    }
    out(api('formatEvalTable')(result));
    return EXIT.OK;
  } finally {
    store.close();
  }
}

// ---------------------------------------------------------------------------
// bands
// ---------------------------------------------------------------------------

function diffBands(before, after) {
  const changes = [];
  for (const [stage, keys] of Object.entries(after ?? {})) {
    for (const [key, spec] of Object.entries(keys ?? {})) {
      const old = before?.[stage]?.[key];
      for (const field of ['lo', 'hi']) {
        const from = old ? old[field] : undefined;
        const to = spec[field];
        if (from !== to) changes.push({ where: `${stage}.${key}.${field}`, from, to });
      }
    }
  }
  return changes;
}

async function cmdBands(parsed) {
  const flags = parsed.flags;
  const config = configFrom(flags);
  const bands = api('loadBands')(config.bandsPath || undefined);
  const source = config.bandsPath || process.env.KOLM_CODE_BANDS_PATH || ladder.BUNDLED_BANDS_PATH;

  if (!flags.suggest && !flags.apply) {
    if (flags.json) {
      jsonOut({ source, bands });
      return EXIT.OK;
    }
    out(`bands ${source}`);
    out('');
    const rows = [];
    for (const [stage, keys] of Object.entries(bands)) {
      for (const [key, spec] of Object.entries(keys)) {
        rows.push([stage, key, spec.lo, spec.hi, spec.low_is_bad ? 'low is bad' : 'high is bad']);
      }
    }
    out(renderTable(['STAGE', 'METRIC', 'LO', 'HI', 'ORIENTATION'], rows));
    return EXIT.OK;
  }

  // The confirmation is an argument, so a missing one is refused before the
  // stored runs are read: `bands --suggest` shows the proposal without writing.
  if (flags.apply && flags.yes !== true) {
    throw new LadderError('bad-args', 'bands --apply rewrites your bands file; add --yes to confirm');
  }

  const store = openStoreFor(config);
  try {
    const proposal = await api('suggestBands')({
      store,
      bands,
      minDisagreements: config.calibrationMinDisagreements,
    });
    const patched = proposal.bandsPatched ?? bands;
    const changes = diffBands(bands, patched);
    const skipped = proposal.skipped ?? [];

    if (flags.json) {
      jsonOut({ source, proposals: proposal.proposals ?? [], changes, skipped, applied: false });
    } else {
      out(`bands ${source}`);
      if (changes.length === 0) out('no threshold would change from the stored runs');
      for (const change of changes) out(`${change.where}: ${change.from} -> ${change.to}`);
      // A key that disagreed but did not move says more than silence: locked is
      // a deliberate safety band, n<N is not enough evidence yet.
      for (const skip of skipped) {
        out(`${skip.stage}.${skip.key}: not proposed (${skip.reason}, ${skip.disagreements} disagreement(s))`);
      }
    }

    if (!flags.apply) return EXIT.OK;

    const bundled = path.resolve(ladder.BUNDLED_BANDS_PATH);
    let target = config.bandsPath ? path.resolve(config.bandsPath) : path.join(config.codeDir, 'bands.json');
    if (path.resolve(target) === bundled) {
      target = path.join(config.codeDir, 'bands.json');
    }
    if (path.resolve(target) === bundled) {
      throw new LadderError('bad-args', 'refusing to overwrite the bundled bands file', { target });
    }
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, `${JSON.stringify(patched, null, 2)}\n`, 'utf8');
    out(`wrote ${target}`);
    if (!config.bandsPath || path.resolve(config.bandsPath) !== path.resolve(target)) {
      out(`set KOLM_CODE_BANDS_PATH=${target} to use it`);
    }
    return EXIT.OK;
  } finally {
    store.close();
  }
}

// ---------------------------------------------------------------------------
// bench
// ---------------------------------------------------------------------------

async function cmdBench(parsed) {
  const flags = parsed.flags;
  const config = await configForModels(flags);
  const arming = api('armingReport')(config);
  const ladders = api('effectiveLadders')(config, arming);
  for (const line of droppedRungLines(arming, ladders)) note(line);

  const concurrency = flags.concurrency === undefined ? 1 : Number(flags.concurrency);
  if (!Number.isInteger(concurrency) || concurrency < 1 || concurrency > 16) {
    throw new LadderError('bad-args', `--concurrency must be an integer between 1 and 16, got "${flags.concurrency}"`);
  }

  // Read the comparison file before the bench runs: a typo in --compare should
  // cost one line, not a whole benchmark.
  let previous = null;
  if (flags.compare !== undefined && flags.compare !== true) {
    const comparePath = path.resolve(String(flags.compare));
    if (!fs.existsSync(comparePath)) {
      throw new LadderError('not-found', `--compare file does not exist: ${comparePath}`, { path: comparePath });
    }
    try {
      previous = JSON.parse(fs.readFileSync(comparePath, 'utf8'));
    } catch (err) {
      throw new LadderError('bad-args', `--compare file is not readable JSON: ${comparePath}`, { path: comparePath, detail: err?.message });
    }
  }

  // Likewise the task selection: an unknown id is refused before the store is
  // opened, so the message is the same on every Node version.
  const taskIds = flags.tasks === undefined || flags.tasks === true ? null : String(flags.tasks);
  await api('listBenchTasks')({ tasks: taskIds });

  const store = openStoreFor(config);
  const driver = api('createDriver')(config.driver, config);
  const createProvider = api('createProvider');
  const providers = {
    orchestrator: createProvider(config.orchestratorRef, config),
    judge: createProvider(config.judgeRef, config),
    coders: ladders.coders.map((ref) => createProvider(ref, config)),
    reviewer: ladders.reviewer ? createProvider(ladders.reviewer, config) : null,
  };

  const controller = new AbortController();
  const streaming = flags.quiet !== true;
  const onSigint = () => {
    controller.abort();
    note('');
    note('aborted; the tasks still pending are recorded as skipped');
  };
  process.on('SIGINT', onSigint);

  try {
    const result = await api('runBench')({
      tasks: taskIds,
      concurrency,
      config,
      store,
      driver,
      providers,
      signal: controller.signal,
      keepWork: flags.keepWork === true,
      ladder: {
        coders: ladders.coders,
        reviewer: ladders.reviewer,
        orchestrator: config.orchestratorRef,
        judge: config.judgeRef,
      },
      onEvent: (event) => {
        if (streaming) note(formatEvent(event));
      },
    });

    if (flags.json) {
      jsonOut(previous ? { ...result, compare: api('compareBench')(previous, result) } : result);
    } else {
      out(api('formatBenchTable')(result));
      if (result.resultPath) {
        out('');
        out(`result   ${result.resultPath}`);
      }
      if (previous) {
        out('');
        out(api('compareBench')(previous, result));
      }
    }

    // A skipped task (no python, say) must not turn into a failing bench: the
    // gate is every task that actually ran and was judgeable.
    const graded = (result.results ?? []).filter((r) => r.verdict !== 'SKIP');
    const allPassed = graded.length > 0 && graded.every((r) => r.hiddenPass === true);
    return allPassed ? EXIT.OK : EXIT.GATE_FAIL;
  } finally {
    process.off('SIGINT', onSigint);
    try {
      store.close();
    } catch {
      // already closed; nothing to unwind
    }
  }
}

// ---------------------------------------------------------------------------
// lessons
// ---------------------------------------------------------------------------

async function cmdLessons(parsed) {
  const flags = parsed.flags;
  const target = parsed.positional[1] ?? flags.repo;
  if (!target || target === true) {
    throw new LadderError('bad-args', 'lessons needs a repository: kolm code lessons <repo> [--clear]');
  }
  const config = configFrom(flags);
  // A local path is stored under its resolved absolute form; a url is stored
  // verbatim, so `lessons <url>` and a run against that url agree on the key.
  const repoKey = /^[a-z][a-z0-9+.-]*:\/\//i.test(String(target)) ? String(target) : path.resolve(String(target));
  const file = api('lessonsPath')(config, repoKey);

  if (flags.clear === true) {
    // Removing the file rather than writing an empty one, so `--clear` still
    // works when KOLM_CODE_LESSONS=0 makes saveLessons a no-op.
    const existed = fs.existsSync(file);
    fs.rmSync(file, { force: true });
    if (flags.json) jsonOut({ repo: repoKey, path: file, cleared: true, existed, lessons: [] });
    else out(`cleared  ${file}${existed ? '' : ' (was not on disk)'}`);
    return EXIT.OK;
  }

  const document = api('loadLessons')(config, repoKey);
  const lessons = Array.isArray(document?.lessons) ? document.lessons : [];
  if (flags.json) {
    jsonOut({ repo: repoKey, path: file, enabled: config.lessons !== false, count: lessons.length, lessons });
    return EXIT.OK;
  }
  out(`repo     ${repoKey}`);
  out(`file     ${file}${fs.existsSync(file) ? '' : ' (not on disk)'}`);
  out(`lessons  ${lessons.length}${config.lessons === false ? '  (KOLM_CODE_LESSONS=0: reads as empty)' : ''}`);
  if (lessons.length === 0) return EXIT.OK;
  out('');
  out(
    renderTable(
      ['N', 'HITS', 'RUN', 'TEXT'],
      lessons.map((lesson, i) => [
        i + 1,
        lesson?.hits ?? '-',
        truncate(lesson?.runId ?? '-', 20),
        truncate(lesson?.text ?? '-', 90),
      ]),
    ),
  );
  return EXIT.OK;
}

// ---------------------------------------------------------------------------
// receipt
// ---------------------------------------------------------------------------

function locateReceipt(config, target) {
  if (!target) return null;
  const asPath = path.resolve(String(target));
  if (fs.existsSync(asPath) && fs.statSync(asPath).isFile()) return asPath;
  if (typeof ladder.receiptPathFor === 'function') {
    const derived = ladder.receiptPathFor(config, String(target));
    if (derived && fs.existsSync(derived)) return derived;
  }
  return null;
}

async function cmdReceipt(parsed) {
  const flags = parsed.flags;
  const sub = parsed.positional[1];
  const target = parsed.positional[2];
  if (sub !== 'verify' && sub !== 'show') {
    throw new LadderError('bad-args', 'receipt takes a subcommand: kolm code receipt verify|show <path|runId>');
  }
  if (!target) throw new LadderError('bad-args', `receipt ${sub} needs a path or a run id`);

  const config = configFrom(flags);
  const receiptPath = locateReceipt(config, target);

  let receipt = null;
  if (receiptPath) {
    receipt = api('readReceipt')(receiptPath);
  } else if (typeof ladder.sqliteAvailable === 'function' && ladder.sqliteAvailable()) {
    const store = api('openStore')(config.dbPath);
    try {
      receipt = store.getRun(String(target))?.receipt ?? null;
    } finally {
      store.close();
    }
  }

  if (!receipt) {
    note(`kolm code: not-found: no receipt for ${target}`);
    return EXIT.NOT_FOUND;
  }

  if (sub === 'show') {
    jsonOut(receipt);
    return EXIT.OK;
  }

  const result = api('verifyReceipt')(receipt);
  if (flags.json) jsonOut({ path: receiptPath, ...result });
  else if (result.ok) out(`VERIFIED  kid ${result.kid ?? '-'}  ${receiptPath ?? target}`);
  else out(`UNVERIFIED  ${result.reason ?? 'signature did not verify'}  ${receiptPath ?? target}`);
  return result.ok ? EXIT.OK : EXIT.GATE_FAIL;
}

// ---------------------------------------------------------------------------
// keys - the bring-your-own-key keystore
//
// The value never travels as a command-line argument (shell history, `ps`, CI
// logs), is never echoed back and never appears in a table. Only the SHA-256
// fingerprint prefix is ever printed.
// ---------------------------------------------------------------------------

const KEYS_SUBVERBS = Object.freeze(['set', 'list', 'unset', 'path', 'test']);

/** Read all of stdin. Used when the value is piped in. */
async function readStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(Buffer.from(chunk));
  return Buffer.concat(chunks).toString('utf8');
}

/**
 * Ask for a value on a terminal without echoing it: the prompt goes to stdout
 * directly and readline writes its echo into a sink that discards everything.
 * @param {string} label
 * @returns {Promise<string>}
 */
async function promptHiddenValue(label) {
  const readline = await import('node:readline');
  const { Writable } = await import('node:stream');
  const sink = new Writable({
    write(_chunk, _encoding, done) {
      done();
    },
  });
  process.stdout.write(`${label}: `);
  const rl = readline.createInterface({ input: process.stdin, output: sink, terminal: true });
  try {
    return await new Promise((resolve) => {
      rl.question('', resolve);
    });
  } finally {
    rl.close();
    process.stdout.write('\n');
  }
}

/** First line of a file, trimmed. Keys are one line by definition. */
function firstLineOf(file) {
  let text;
  try {
    text = fs.readFileSync(file, 'utf8');
  } catch (err) {
    throw new LadderError('bad-args', `--from-file ${file} could not be read (${err.code ?? err.message})`, {
      path: file,
    });
  }
  return String(text).split(/\r?\n/, 1)[0].trim();
}

/**
 * Where the value for `keys set` comes from, in the documented order.
 * @param {string} name
 * @param {Record<string, any>} flags
 * @returns {Promise<string>}
 */
async function collectKeyValue(name, flags) {
  if (typeof flags.fromEnv === 'string') {
    const varName = flags.fromEnv;
    const value = process.env[varName];
    if (value === undefined || String(value).trim() === '') {
      throw new LadderError('bad-args', `--from-env ${varName}: that variable is not set in this shell`, {
        variable: varName,
      });
    }
    return String(value);
  }
  if (typeof flags.fromFile === 'string') return firstLineOf(flags.fromFile);
  if (flags.stdin === true || process.stdin.isTTY !== true) {
    const piped = await readStdin();
    return String(piped).split(/\r?\n/, 1)[0];
  }
  return await promptHiddenValue(`${name} key (not echoed)`);
}

async function cmdKeys(parsed) {
  const flags = parsed.flags;
  const sub = parsed.positional[1] ?? 'list';
  if (!KEYS_SUBVERBS.includes(sub)) {
    throw new LadderError('bad-args', `keys takes a subcommand: kolm code keys ${KEYS_SUBVERBS.join('|')}`, {
      sub,
    });
  }
  const env = envOverlay(flags);
  const keyNames = ladder.KEY_NAMES ?? [];

  if (sub === 'path') {
    const store = api('readKeystore')(env);
    out(store.path);
    if (!store.enabled) note('note: KOLM_CODE_NO_KEYSTORE=1 is set, so no verb reads this file');
    else if (!fs.existsSync(store.path)) note('note: no keystore file yet; kolm code keys set <name> creates it');
    return EXIT.OK;
  }

  if (sub === 'set') {
    const name = parsed.positional[2];
    if (!name) {
      throw new LadderError('bad-args', `keys set needs a name: kolm code keys set <${keyNames.join('|')}>`, {});
    }
    if (parsed.positional[3] !== undefined) {
      note('kolm code: pass the key on stdin, --from-env or --from-file; never on the command line (shell history)');
      return EXIT.USAGE;
    }
    api('assertKeyName')(name);
    const value = await collectKeyValue(name, flags);
    const result = api('setKey')(env, name, value);
    out(`set ${result.name} (fingerprint ${result.fingerprint}) at ${result.path}`);
    if (api('keystoreDisabled')(env)) {
      note('note: KOLM_CODE_NO_KEYSTORE=1 is set, so this file is written but not read');
    }
    return EXIT.OK;
  }

  if (sub === 'unset') {
    const name = parsed.positional[2];
    if (!name) {
      throw new LadderError('bad-args', `keys unset needs a name: kolm code keys unset <${keyNames.join('|')}>`, {});
    }
    const result = api('unsetKey')(env, name);
    out(result.removed ? `unset ${result.name} at ${result.path}` : `${result.name} was not in ${result.path}`);
    return EXIT.OK;
  }

  if (sub === 'list') {
    const described = api('describeKeys')(env);
    if (flags.json) {
      jsonOut({
        path: described.path,
        enabled: described.enabled,
        keys: described.rows.map(({ name, source, fingerprint: fp, set_at: setAt }) => ({
          name,
          source,
          fingerprint: fp,
          set_at: setAt,
        })),
      });
      return EXIT.OK;
    }
    out(`keystore ${described.path}${described.enabled ? '' : '  (disabled by KOLM_CODE_NO_KEYSTORE)'}`);
    out('');
    out(
      renderTable(
        ['NAME', 'SOURCE', 'FINGERPRINT', 'SET AT'],
        described.rows.map((row) => [row.name, row.source, row.fingerprint ?? '-', row.set_at ?? '-']),
      ),
    );
    out('');
    out('values are never printed; the fingerprint is the first 12 hex of sha256');
    return EXIT.OK;
  }

  // keys test - one authenticated GET per key that is actually set.
  const only = parsed.positional[2];
  if (only) api('assertKeyName')(only);
  const config = configFrom(flags);
  const described = api('describeKeys')(env);
  const rows = described.rows.filter((row) => row.source !== 'none' && (!only || row.name === only));
  if (rows.length === 0) {
    const scope = only ? `${only} is not set` : 'no keys are set';
    if (flags.json) jsonOut({ ok: true, results: [], reason: scope });
    else out(`${scope}; nothing to test`);
    return EXIT.OK;
  }

  const secretFor = (name) => config.getSecret(name === 'github' ? 'githubToken' : name);
  const results = [];
  for (const row of rows) {
    results.push({
      source: row.source,
      ...(await api('probeKey')({ name: row.name, secret: secretFor(row.name), config })),
    });
  }
  const bad = results.filter((r) => r.status === 'bad-key');

  if (flags.json) {
    jsonOut({
      ok: bad.length === 0,
      results: results.map(({ name, source, status, httpStatus, detail }) => ({
        name,
        source,
        status,
        httpStatus,
        detail,
      })),
    });
    return bad.length === 0 ? EXIT.OK : EXIT.GATE_FAIL;
  }

  out(
    renderTable(
      ['NAME', 'SOURCE', 'RESULT', 'DETAIL'],
      results.map((r) => [
        r.name,
        r.source,
        r.status === 'error' ? `error ${r.httpStatus ?? r.detail}` : r.status,
        truncate(r.detail, 50),
      ]),
    ),
  );
  out('');
  out(bad.length === 0 ? 'no key was rejected' : `rejected: ${bad.map((r) => r.name).join(', ')}`);
  return bad.length === 0 ? EXIT.OK : EXIT.GATE_FAIL;
}

// ---------------------------------------------------------------------------
// doctor
// ---------------------------------------------------------------------------

function gitAvailable() {
  try {
    const res = spawnSync('git', ['--version'], { encoding: 'utf8', timeout: 10000 });
    if (res.status === 0) return { ok: true, detail: String(res.stdout || '').trim() };
    return { ok: false, detail: 'git exited non-zero' };
  } catch (err) {
    return { ok: false, detail: `git not on PATH (${err.code ?? err.message})` };
  }
}

async function liveProbeProvider(ref, config) {
  const started = Date.now();
  try {
    const provider = api('createProvider')(ref, config);
    const res = await provider.chat({ messages: [{ role: 'user', content: 'ping' }], maxTokens: 8 });
    return `ok ${seconds(Date.now() - started)} ${usd(res?.costUsd ?? 0)}`;
  } catch (err) {
    return `failed: ${truncate(err?.message ?? String(err), 70)}`;
  }
}

async function liveProbeDriver(config) {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'kolm-code-doctor-'));
  try {
    const git = (args) => spawnSync('git', args, { cwd: tmp, encoding: 'utf8', timeout: 30000 });
    if (git(['init', '-q']).status !== 0) return 'git init failed';
    git(['config', 'user.email', 'kolm@localhost']);
    git(['config', 'user.name', 'kolm code doctor']);
    fs.writeFileSync(path.join(tmp, 'README'), 'kolm code doctor probe\n');
    git(['add', 'README']);
    if (git(['commit', '-qm', 'probe']).status !== 0) return 'git commit failed';
    const driver = api('createDriver')('local', config);
    const worktree = await driver.create({ repoPath: tmp, runId: 'r_doctor', candidateId: 'c0a0' });
    const at = worktree.path;
    await worktree.destroy();
    await driver.destroyRun('r_doctor');
    return `create + destroy ok (${path.basename(at)})`;
  } catch (err) {
    return `failed: ${truncate(err?.message ?? String(err), 70)}`;
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
}

async function cmdDoctor(parsed) {
  const flags = parsed.flags;
  const env = envOverlay(flags);
  const config = configFrom(flags);
  const live = flags.live === true;

  let ollamaProbe = null;
  let ghProbe = null;
  let effectiveConfig = config;
  if (live) {
    ollamaProbe = await api('probeOllama')(config);
    effectiveConfig = api('withOllamaReachable')(config, ollamaProbe.reachable);
    ghProbe = await api('probeGhCli')({});
  }

  const arming = api('armingReport')(effectiveConfig, ollamaProbe ? { probe: { ollama: ollamaProbe } } : {});
  const { describeGitHubArming } = await import('../src/ladder/github/app-auth.js');
  const { describeTruffleArming, truffleEnvFromConfig, REQUIRED_TRUFFLE_ENV } = await import('../src/ladder/truffle/worker.js');

  const githubDetail = describeGitHubArming({ config: effectiveConfig });
  const truffleDetail = describeTruffleArming(truffleEnvFromConfig(effectiveConfig));

  const sqliteOk = typeof ladder.sqliteAvailable === 'function' ? ladder.sqliteAvailable() : false;
  const git = gitAvailable();

  const providerKinds = ['orchestrator', 'judge', 'coder', 'reviewer'];
  const rows = arming.rungs.map((rung) => {
    const row = {
      id: rung.id,
      kind: rung.kind,
      ref: rung.ref,
      armed: rung.armed,
      reason: rung.reason,
    };
    // Which of env / keystore supplied the key behind this rung (never the key).
    if (providerKinds.includes(rung.kind) && typeof rung.ref === 'string') {
      const provider = rung.ref.split(':', 1)[0];
      row.keySource = effectiveConfig.keySource?.[provider] ?? null;
    }
    if (rung.id === 'github') row.keySource = effectiveConfig.keySource?.github ?? null;
    if (rung.id === 'driver') row.keySource = effectiveConfig.keySource?.sandbox ?? null;
    return row;
  });

  const github = rows.find((r) => r.id === 'github');
  if (github) {
    github.armed = githubDetail.armed || github.armed;
    // `sources` lists only credential sources actually present (gh cli is
    // reported separately as unknown), so an unarmed row names the variables.
    const real = githubDetail.sources ?? [];
    const fromKeystore = effectiveConfig.github?.tokenSource === 'keystore';
    github.reason = github.armed
      ? `token from ${fromKeystore ? 'keystore' : real.join(', ') || 'gh cli'}`
      : 'set KOLM_GITHUB_TOKEN or GITHUB_TOKEN, run kolm code keys set github, install the App, or run gh auth login';
  }
  const truffle = rows.find((r) => r.id === 'truffle');
  if (truffle) {
    truffle.reason = truffleDetail.armed
      ? `partner contract vars set (${REQUIRED_TRUFFLE_ENV.length} names)`
      : `missing ${truffleDetail.missing.join(', ')}`;
    truffle.armed = truffleDetail.armed;
  }

  rows.push({
    id: 'sqlite',
    kind: 'store',
    ref: effectiveConfig.dbPath,
    armed: sqliteOk,
    reason: sqliteOk ? `node:sqlite on ${process.version}` : `node:sqlite unavailable on ${process.version}`,
  });
  rows.push({
    id: 'git',
    kind: 'tool',
    ref: null,
    armed: git.ok,
    reason: git.detail,
  });

  if (live) {
    for (const row of rows) {
      if (!['orchestrator', 'judge', 'coder', 'reviewer'].includes(row.kind)) continue;
      if (!row.armed) {
        row.live = 'skipped (not armed)';
        continue;
      }
      row.live = await liveProbeProvider(row.ref, effectiveConfig);
    }
    const driverRow = rows.find((r) => r.id === 'driver');
    if (driverRow) {
      driverRow.live = effectiveConfig.driver === 'local' ? await liveProbeDriver(effectiveConfig) : `no live probe for the ${effectiveConfig.driver} driver`;
    }
    const ollamaRowExists = rows.some((r) => String(r.ref ?? '').startsWith('ollama:'));
    if (ollamaRowExists || ollamaProbe) {
      rows.push({
        id: 'ollama',
        kind: 'probe',
        ref: effectiveConfig.ollamaUrl,
        armed: Boolean(ollamaProbe?.reachable),
        reason: ollamaProbe?.reachable ? `${ollamaProbe.models.length} models` : (ollamaProbe?.error ?? 'unreachable'),
      });
    }
    rows.push({
      id: 'gh-cli',
      kind: 'probe',
      ref: null,
      armed: Boolean(ghProbe?.authenticated),
      reason: ghProbe?.reason ?? 'not probed',
    });
  }

  // The runtime toggles: none of them arm or disarm a rung, but every one of
  // them changes how a run behaves, so an operator debugging "why did it not
  // split / not retrieve / not stream" should see them here.
  const runtime = {
    routing: effectiveConfig.routing ?? 'auto',
    subtasks: effectiveConfig.subtasks !== false,
    subtaskConcurrency: effectiveConfig.subtaskConcurrency ?? 2,
    coach: effectiveConfig.coach !== false,
    selfReview: effectiveConfig.selfReview !== false,
    lessons: effectiveConfig.lessons !== false,
    contextTokens: effectiveConfig.contextTokens ?? 0,
    retrievalBytes: effectiveConfig.retrievalBytes ?? 0,
    textTools: Array.isArray(effectiveConfig.textTools) ? effectiveConfig.textTools : [],
    providerConcurrency: effectiveConfig.providerConcurrency ?? {},
    stream: effectiveConfig.stream === true,
    benchDir: effectiveConfig.benchDir ?? null,
  };

  const orchestratorArmed = rows.find((r) => r.id === 'orchestrator')?.armed === true;
  const coderArmed = rows.some((r) => r.kind === 'coder' && r.armed);
  const driverArmed = rows.find((r) => r.id === 'driver')?.armed === true;
  const ok = orchestratorArmed && coderArmed && driverArmed && sqliteOk;

  if (flags.json) {
    jsonOut({
      ok,
      node: process.version,
      dbPath: effectiveConfig.dbPath,
      dataDir: effectiveConfig.dataDir,
      keystore: { path: api('keystorePath')(env), enabled: !api('keystoreDisabled')(env) },
      keySource: effectiveConfig.keySource,
      driver: effectiveConfig.driver,
      mode: effectiveConfig.mode,
      mockProvider: effectiveConfig.mockProvider === true,
      live,
      runtime,
      rungs: rows,
    });
    return ok ? EXIT.OK : EXIT.MISSING_PREREQ;
  }

  out(`kolm code doctor  node ${process.version}  data ${effectiveConfig.codeDir}`);
  out('');
  const headers = live
    ? ['ROW', 'KIND', 'REF', 'STATUS', 'KEY', 'REASON', 'LIVE']
    : ['ROW', 'KIND', 'REF', 'STATUS', 'KEY', 'REASON'];
  out(
    renderTable(
      headers,
      rows.map((row) => {
        const base = [
          row.id,
          row.kind,
          truncate(row.ref ?? '-', 40),
          row.armed ? 'ARMED' : 'NOT ARMED',
          row.keySource ?? '-',
          truncate(row.reason ?? '-', 60),
        ];
        return live ? [...base, truncate(row.live ?? '-', 40)] : base;
      }),
    ),
  );
  out('');
  const onOff = (value) => (value ? 'on' : 'off');
  const concurrencyPairs = Object.entries(runtime.providerConcurrency);
  out(
    renderTable(
      ['SETTING', 'VALUE'],
      [
        ['routing', `${runtime.routing} (subtasks ${onOff(runtime.subtasks)}, up to ${runtime.subtaskConcurrency} at a time)`],
        ['coach', onOff(runtime.coach)],
        ['self-review', onOff(runtime.selfReview)],
        ['lessons', onOff(runtime.lessons)],
        ['context tokens', runtime.contextTokens === 0 ? '0 (no compaction)' : runtime.contextTokens],
        ['retrieval bytes', runtime.retrievalBytes],
        ['text tools', runtime.textTools.length > 0 ? truncate(runtime.textTools.join(', '), 60) : 'none'],
        [
          'provider concurrency',
          concurrencyPairs.length > 0
            ? truncate(concurrencyPairs.map(([name, n]) => `${name}=${n}`).join(', '), 60)
            : 'defaults',
        ],
        ['stream', onOff(runtime.stream)],
        ['bench dir', truncate(runtime.benchDir ?? '-', 60)],
      ],
    ),
  );
  out('');
  if (effectiveConfig.mockProvider === true) out('note: KOLM_TEST_MOCK_PROVIDER=1 is set; every provider call is short-circuited');
  if (ok) {
    out('ready: orchestrator, a coder rung, the driver and the store are armed');
  } else {
    out('not ready: see the NOT ARMED rows above (env var names only, never values)');
    out('hint: export the variable, or store the key once with: kolm code keys set <provider>');
  }
  return ok ? EXIT.OK : EXIT.MISSING_PREREQ;
}

// ---------------------------------------------------------------------------
// serve
// ---------------------------------------------------------------------------

function openInBrowser(url) {
  try {
    const [cmd, args] = process.platform === 'win32'
      ? ['cmd', ['/c', 'start', '', url]]
      : process.platform === 'darwin'
        ? ['open', [url]]
        : ['xdg-open', [url]];
    const child = spawn(cmd, args, { detached: true, stdio: 'ignore' });
    child.on('error', () => {});
    child.unref();
  } catch {
    // best effort only: the URL is already printed
  }
}

async function cmdServe(parsed) {
  const flags = parsed.flags;
  const config = configFrom(flags);
  const store = openStoreFor(config);
  const { createViewerServer } = await import('../src/ladder/viewer/server.js');
  const port = Number(flags.port ?? 7474);
  const viewer = createViewerServer({
    store,
    config,
    host: flags.host ? String(flags.host) : '127.0.0.1',
    port: Number.isFinite(port) ? port : 7474,
    verifyReceipt: typeof ladder.verifyReceipt === 'function' ? ladder.verifyReceipt : null,
    evalRuns: typeof ladder.evalRuns === 'function' ? ladder.evalRuns : null,
    arming: () => api('armingReport')(config),
  });
  const { url } = await viewer.listen();
  out(`kolm code viewer  ${url}`);
  out(`store             ${config.dbPath}`);
  out('press ctrl-c to stop');
  if (flags.open === true) openInBrowser(url);

  await new Promise((resolve) => {
    const stop = () => resolve();
    process.once('SIGINT', stop);
    process.once('SIGTERM', stop);
  });
  await viewer.close();
  store.close();
  return EXIT.OK;
}

// ---------------------------------------------------------------------------
// truffle worker
// ---------------------------------------------------------------------------

async function cmdTruffleWorker(parsed) {
  const flags = parsed.flags;
  const config = await configForModels(flags);
  const { createWorkerFromEnv, describeTruffleArming, truffleEnvFromConfig, REQUIRED_TRUFFLE_ENV } = await import('../src/ladder/truffle/worker.js');
  const env = truffleEnvFromConfig(config);
  const { armed, missing } = describeTruffleArming(env);
  if (!armed) {
    note('kolm code: not-armed: the Truffle partner worker is not configured');
    note(`missing variables: ${missing.join(', ')}`);
    // Only worth restating when some of the required set is already present.
    if (missing.length < REQUIRED_TRUFFLE_ENV.length) {
      note(`required: ${REQUIRED_TRUFFLE_ENV.join(', ')}`);
    }
    return EXIT.MISSING_PREREQ;
  }

  const store = openStoreFor(config);
  const driver = api('createDriver')(config.driver, config);
  const arming = api('armingReport')(config);
  const ladders = api('effectiveLadders')(config, arming);
  for (const line of droppedRungLines(arming, ladders)) note(line);
  const createProvider = api('createProvider');
  const controller = new AbortController();
  const onSigint = () => controller.abort();
  process.on('SIGINT', onSigint);

  const handler = async (request, ctx = {}) => {
    const result = await api('runTask')({
      task: {
        text: request.text ?? request.instructions ?? '',
        title: truncate(request.text ?? request.taskId ?? 'kolm code task', 72),
        repoUrl: request.repoUrl ?? undefined,
        baseRef: request.baseRef ?? undefined,
      },
      repoPath: process.cwd(),
      repoUrl: request.repoUrl ?? undefined,
      baseRef: request.baseRef ?? undefined,
      config,
      store,
      driver,
      providers: {
        orchestrator: createProvider(config.orchestratorRef, config),
        judge: createProvider(config.judgeRef, config),
        coders: ladders.coders.map((ref) => createProvider(ref, config)),
        reviewer: ladders.reviewer ? createProvider(ladders.reviewer, config) : null,
      },
      logger: console,
      signal: ctx.signal ?? controller.signal,
      mode: config.mode,
      onEvent: (event) => {
        if (flags.quiet !== true) note(formatEvent(event));
      },
    });
    return {
      verdict: result.verdict,
      receipt: result.receipt ?? null,
      prUrl: result.prUrl ?? null,
      costUsd: result.costUsd ?? 0,
      summary: result.summary ?? result.receipt?.summary ?? null,
      title: request.text ? truncate(request.text, 200) : null,
      stages: result.funnel?.stages ?? null,
      runId: result.runId,
    };
  };

  try {
    const maxTasks = flags.maxTasks === undefined ? null : Number(flags.maxTasks);
    const stats = await createWorkerFromEnv({
      config,
      handler,
      logger: console,
      signal: controller.signal,
      maxTasks: Number.isFinite(maxTasks) ? maxTasks : null,
    });
    out(`processed ${stats.processed}  failed ${stats.failed}  stopped by ${stats.stoppedBy}`);
    return stats.failed > 0 ? EXIT.GATE_FAIL : EXIT.OK;
  } finally {
    process.off('SIGINT', onSigint);
    store.close();
  }
}

// ---------------------------------------------------------------------------
// dispatch
// ---------------------------------------------------------------------------

/**
 * @param {string[]} argv arguments after the `code` verb
 * @returns {Promise<number>} process exit code
 */
export async function main(argv = []) {
  const parsed = parseArgs(argv);
  const verb = parsed.positional[0];

  if (parsed.flags.version === true) {
    out(ladder.VERSION ?? 'kolm-code');
    return EXIT.OK;
  }
  if (!verb || verb === 'help' || parsed.flags.help === true) {
    out(USAGE);
    return EXIT.OK;
  }
  if (!VERBS.includes(verb)) {
    note(`kolm code: unknown verb "${verb}"`);
    note('');
    note(USAGE);
    return EXIT.USAGE;
  }

  try {
    switch (verb) {
      case 'run':
        return await cmdRun(parsed);
      case 'resume':
        return await cmdResume(parsed);
      case 'status':
        return await cmdStatus(parsed);
      case 'show':
        return await cmdShow(parsed);
      case 'funnel':
        return await cmdFunnel(parsed);
      case 'eval':
        return await cmdEval(parsed);
      case 'bands':
        return await cmdBands(parsed);
      case 'bench':
        return await cmdBench(parsed);
      case 'lessons':
        return await cmdLessons(parsed);
      case 'clean':
        return await cmdClean(parsed);
      case 'receipt':
        return await cmdReceipt(parsed);
      case 'keys':
        return await cmdKeys(parsed);
      case 'doctor':
        return await cmdDoctor(parsed);
      case 'serve':
        return await cmdServe(parsed);
      case 'truffle-worker':
        return await cmdTruffleWorker(parsed);
      default:
        out(USAGE);
        return EXIT.USAGE;
    }
  } catch (err) {
    return reportError(err);
  }
}

/**
 * LadderErrors print one legible line, never a stack trace.
 * @param {unknown} err
 * @returns {number}
 */
function reportError(err) {
  if (isLadderError(err)) {
    note(`kolm code: ${err.code}: ${err.message}`);
    return CODE_EXIT[err.code] ?? EXIT.EXECUTION;
  }
  if (err && (err.name === 'AbortError' || err.code === 'ABORT_ERR')) {
    note('kolm code: aborted');
    return EXIT.EXECUTION;
  }
  const message = err && err.message ? err.message : String(err);
  note(`kolm code: ${message}`);
  if (process.env.KOLM_DEBUG === '1' && err && err.stack) note(err.stack);
  return EXIT.EXECUTION;
}

export { USAGE, VERBS, REPO_ROOT };

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main(process.argv.slice(2)).then(
    (code) => {
      process.exitCode = code;
    },
    (err) => {
      process.exitCode = reportError(err);
    },
  );
}

// kolm code — the twelve live-run hardening fixes (one section per defect).
//
// Everything here is offline: the scripted fake provider, the in-memory fake
// worktree driver, injected shell/stat/git resolvers, and CLI subprocesses that
// run with a scrubbed environment. No network, no live model, no operator store.

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import { runTask, repairSkipReason, RUNG_LEVEL_ERROR_CODES } from '../src/ladder/orchestrator.js';
import { runCoder, runRepair, turnDetail, allowedCommandsFor, NO_PROGRESS_WINDOW } from '../src/ladder/coder.js';
import {
  resolveShell,
  allowedForShell,
  POSIX_ONLY_COMMANDS,
  createLocalGitDriver,
} from '../src/ladder/worktree/local-git.js';
import { createFakeDriver } from '../src/ladder/worktree/fake.js';
import { createFakeProvider } from '../src/ladder/providers/fake.js';
import { openStore, sqliteAvailable } from '../src/ladder/store.js';
import { loadBands } from '../src/ladder/bands.js';
import { loadRubric } from '../src/ladder/rubric.js';
import { parseTestOutput, isRunnerNoise } from '../src/ladder/context/detect.js';
import { suggestBands, runJudge } from '../src/ladder/stages/l4-judge.js';
import { buildReceipt, signReceipt, verifyReceipt } from '../src/ladder/receipt.js';
import { generateKeyPair, keyFingerprint } from '../src/ed25519.js';
import { run as l2Run } from '../src/ladder/stages/l2-dynamic.js';
import { run as l1Run } from '../src/ladder/stages/l1-static.js';
import { run as l3Run } from '../src/ladder/stages/l3-semantic.js';
import { runCascade } from '../src/ladder/stages/index.js';
import { plan as makePlan } from '../src/ladder/planner.js';
import { isLadderError, LadderError } from '../src/ladder/errors.js';
import {
  EXIT,
  VERBS,
  USAGE,
  FINISHED_RUN_STATUSES,
  validateRunPreconditions,
  droppedRungLines,
  selectRunsToClean,
} from '../cli/kolm-code.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..');
const CLI = path.join(REPO_ROOT, 'cli', 'kolm-code.mjs');

const HAS_SQLITE = sqliteAvailable();
const options = HAS_SQLITE ? {} : { skip: 'node:sqlite is unavailable on this Node' };
const HAS_GIT = spawnSync('git', ['--version'], { encoding: 'utf8' }).status === 0;
const gitOptions = HAS_GIT ? {} : { skip: 'git is not on PATH' };

const BANDS = loadBands();
const RUBRIC = loadRubric();
const RUBRIC_IDS = RUBRIC.items.map((item) => item.id);

/* --------------------------------------------------------------- fixtures */

const tempDirs = [];
const openStores = [];

function tempDir(label) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `kolm-hard-${label}-`));
  tempDirs.push(dir);
  return dir;
}

function tempStore(label) {
  const dir = tempDir(label);
  const store = openStore(path.join(dir, 'ladder.db'));
  openStores.push(store);
  return { store, dir };
}

test.after(() => {
  for (const store of openStores) {
    try {
      store.close();
    } catch {
      // an already-closed store is fine
    }
  }
  for (const dir of tempDirs) {
    try {
      fs.rmSync(dir, { recursive: true, force: true });
    } catch {
      // a lingering handle on Windows is not a test failure
    }
  }
});

const TEST_CMD = 'npm test --silent';

const BASE_FILES = {
  'package.json': `${JSON.stringify({ name: 'demo', version: '1.0.0', scripts: { test: 'node --test' } }, null, 2)}\n`,
  'README.md': '# demo\n\nA tiny repository for the hardening tests.\n',
  'src/index.js': "export const name = 'demo';\n",
};

const GOOD_MATH = 'export function add(a, b) {\n  return a + b;\n}\n';
const MATH_DIFF = [
  'diff --git a/src/math.js b/src/math.js',
  'new file mode 100644',
  'index 0000000..1111111',
  '--- /dev/null',
  '+++ b/src/math.js',
  '@@ -0,0 +1,3 @@',
  '+export function add(a, b) {',
  '+  return a + b;',
  '+}',
  '',
].join('\n');
const BROKEN_MATH = 'export function add(a, b) {\n  // TODO: implement\n  return 0;\n}\n';

function scriptedTests(cmd, wt) {
  const math = wt.readFileSync('src/math.js');
  if (math === null || /return a \+ b/.test(math)) {
    return { stdout: 'ok 1 - add adds numbers\n# tests 1\n# pass 1\n# fail 0\n', exitCode: 0 };
  }
  return { stdout: 'not ok 1 - add adds numbers\n# tests 1\n# pass 0\n# fail 1\n', exitCode: 1 };
}

const PLAN_JSON = {
  title: 'add add(a, b)',
  summary: 'Create src/math.js with an add function.',
  acceptance: ['src/math.js exports add(a, b)'],
  files_expected: ['src/math.js'],
  forbidden_paths: ['package.json'],
  test_command: TEST_CMD,
  risk: 'low',
  budget_hint_usd: 0.25,
  clarifications: [],
};

function plannerProvider({ costUsd = 0.002 } = {}) {
  return createFakeProvider({
    ref: 'anthropic:claude-sonnet',
    script: () => ({ json: PLAN_JSON, costUsd }),
  });
}

/** A coder that writes `content` to src/math.js on its first turn, then finishes. */
function coderProvider(ref, content, { costUsd = 0.001 } = {}) {
  return createFakeProvider({
    ref,
    script: (call) => {
      const first = (call.messages || []).length === 1;
      if (first) return { toolCalls: [{ name: 'write_file', input: { path: 'src/math.js', content } }], costUsd };
      return { toolCalls: [{ name: 'finish', input: { summary: 'wrote src/math.js' } }], costUsd };
    },
  });
}

/** A coder rung whose every call dies at the provider. */
function deadCoder(ref, code = 'provider-bad-request') {
  return createFakeProvider({
    ref,
    script: () => ({ error: new LadderError(code, `${ref} rejected the request (400)`) }),
  });
}

function reviewerProvider(score, confidence) {
  return createFakeProvider({
    ref: 'anthropic:claude-haiku',
    script: () => ({
      json: {
        items: RUBRIC_IDS.map((id) => ({ id, score, evidence: 'the diff implements the acceptance criteria' })),
        overall: score,
        confidence,
        blocking: [],
      },
      costUsd: 0.001,
    }),
  });
}

function passingJudge() {
  return createFakeProvider({
    ref: 'anthropic:claude-opus',
    script: () => ({
      json: {
        items: RUBRIC_IDS.map((id) => ({ id, score: 0.95, evidence: 'reviewed the full diff' })),
        overall: 0.95,
        blocking: [],
        rationale: 'the patch satisfies every acceptance criterion',
        decision: 'PASS',
      },
      costUsd: 0.01,
    }),
  });
}

/** A judge whose reply never parses, on either attempt. */
function unparsableJudge() {
  return createFakeProvider({
    ref: 'anthropic:claude-opus',
    script: () => ({ text: 'I am unable to produce a structured judgment here.', costUsd: 0.004 }),
  });
}

function baseConfig(dir, extra = {}) {
  return {
    mode: 'cascade',
    candidates: 1,
    coderTurns: 6,
    repairRounds: 0,
    repairTurns: 6,
    maxCostUsd: 5,
    calibrationRate: 0,
    stageTimeoutMs: 60000,
    codeDir: path.join(dir, 'code'),
    ...extra,
  };
}

function fixture(label, { files = BASE_FILES, exec = { [TEST_CMD]: scriptedTests }, shell } = {}) {
  const { store, dir } = tempStore(label);
  return { store, dir, driver: createFakeDriver(shell ? { files, exec, shell } : { files, exec }) };
}

/** A store stand-in for the coder loop: records events, charges nothing. */
function stubStore(events) {
  return {
    emit(runId, type, payload) {
      events.push({ type, payload });
      return events.length;
    },
    addCost() {},
    upsertCandidate() {},
    costSummary() {
      return { total_usd: 0 };
    },
  };
}

async function loneWorktree(driver, runId = 'r_hard', candidateId = 'c0a0') {
  return driver.create({ repoPath: '/repo/demo', runId, candidateId });
}

/* ------------------------------------------------------------- CLI harness */

const SECRET_ENV_NAMES = [
  'ANTHROPIC_API_KEY', 'CEREBRAS_API_KEY', 'KOLM_CEREBRAS_TOKEN', 'DEEPSEEK_API_KEY',
  'OPENROUTER_API_KEY', 'KOLM_GITHUB_TOKEN', 'GITHUB_TOKEN', 'KOLM_CODE_BANDS_PATH',
  'KOLM_CODE_ORCHESTRATOR', 'KOLM_CODE_JUDGE', 'KOLM_CODE_CODER_LADDER',
  'KOLM_CODE_REVIEWER_LADDER', 'KOLM_CODE_KEYS_PATH', 'KOLM_CODE_NO_KEYSTORE',
];

function runCli(args, { home, extra = {} } = {}) {
  const env = { ...process.env };
  for (const name of SECRET_ENV_NAMES) delete env[name];
  env.HOME = home;
  env.USERPROFILE = home;
  env.KOLM_CODE_HOME = home;
  env.KOLM_DATA_DIR = path.join(home, '.kolm');
  const res = spawnSync(process.execPath, [CLI, ...args], {
    env: { ...env, ...extra },
    encoding: 'utf8',
    cwd: REPO_ROOT,
    timeout: 180_000,
  });
  return {
    status: res.status,
    stdout: res.stdout ?? '',
    stderr: res.stderr ?? '',
    all: `${res.stdout ?? ''}\n${res.stderr ?? ''}`,
  };
}

/** A one-commit git repository the CLI can validate against. */
function makeRepo(label = 'repo') {
  const dir = tempDir(label);
  const git = (args) => spawnSync('git', args, { cwd: dir, encoding: 'utf8', timeout: 60_000 });
  git(['init', '-q', '-b', 'main']);
  git(['config', 'user.email', 'kolm@localhost']);
  git(['config', 'user.name', 'kolm code tests']);
  git(['config', 'commit.gpgsign', 'false']);
  fs.writeFileSync(path.join(dir, 'README.md'), '# fixture\n');
  fs.writeFileSync(path.join(dir, 'index.js'), 'export function add(a, b) {\n  return a + b;\n}\n');
  git(['add', 'README.md', 'index.js']);
  git(['commit', '-qm', 'fixture']);
  return dir;
}

/* =========================================================================
 * Defect 1 — a rung-level provider failure escalates instead of ending the run
 * ========================================================================= */

test('D1: a coder rung that dies at the provider escalates to the next rung', options, async () => {
  const { store, dir, driver } = fixture('d1-escalate');
  const deadRef = 'ollama:qwen3:8b';
  const liveRef = 'anthropic:claude-sonnet';
  const events = [];

  const result = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js' },
    repoPath: '/repo/demo',
    config: baseConfig(dir),
    store,
    driver,
    providers: {
      orchestrator: plannerProvider(),
      judge: passingJudge(),
      coders: [deadCoder(deadRef), coderProvider(liveRef, GOOD_MATH)],
      reviewer: reviewerProvider(0.95, 0.9),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  assert.equal(result.verdict, 'PASS', 'the second rung rescued the run');
  assert.equal(result.winner, 'c1a0');

  const errors = events.filter((e) => e.type === 'candidate.error');
  assert.equal(errors.length, 1);
  assert.equal(errors[0].payload.candidateId, 'c0a0');
  assert.equal(errors[0].payload.code, 'provider-bad-request');
  assert.ok(errors[0].payload.message.includes('400'));

  const escalations = events.filter((e) => e.type === 'escalate');
  assert.equal(escalations.length, 1);
  assert.equal(escalations[0].payload.reason, 'rung-failed');
  assert.equal(escalations[0].payload.from, deadRef);
  assert.equal(escalations[0].payload.to, liveRef);

  // The dead rung's candidate is on record with its failure, not silently
  // missing (the status has since moved on to 'destroyed' with its worktree).
  const deadRow = store.getCandidate(result.runId, 'c0a0');
  assert.ok(deadRow, 'the errored candidate has a row');
  assert.ok(/rejected the request/.test(deadRow.summary ?? ''), deadRow.summary);
});

test('D1: only the last rung turns a provider failure into a run failure', options, async () => {
  const { store, dir, driver } = fixture('d1-last');
  await assert.rejects(
    runTask({
      task: { text: 'add a function add(a,b) to src/math.js' },
      repoPath: '/repo/demo',
      config: baseConfig(dir),
      store,
      driver,
      providers: {
        orchestrator: plannerProvider(),
        judge: passingJudge(),
        coders: [deadCoder('ollama:qwen3:8b')],
        reviewer: reviewerProvider(0.95, 0.9),
      },
      bands: BANDS,
      rubric: RUBRIC,
    }),
    (err) => isLadderError(err, 'provider-bad-request'),
  );
});

test('D1: a mixed rung carries on with the attempts that survived', options, async () => {
  const { store, dir, driver } = fixture('d1-mixed');
  const ref = 'ollama:qwen3:8b';
  const events = [];
  // The rung's FIRST call dies; every later call codes normally, so exactly one
  // of the two attempts is lost and the other must still reach the cascade.
  const flaky = createFakeProvider({
    ref,
    script: (call, index) => {
      if (index === 0) return { error: new LadderError('provider-bad-request', 'model rejected the tool schema') };
      const first = (call.messages || []).length === 1;
      if (first) {
        return { toolCalls: [{ name: 'write_file', input: { path: 'src/math.js', content: GOOD_MATH } }], costUsd: 0.001 };
      }
      return { toolCalls: [{ name: 'finish', input: { summary: 'wrote src/math.js' } }], costUsd: 0.001 };
    },
  });

  const result = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js' },
    repoPath: '/repo/demo',
    config: baseConfig(dir, { candidates: 2 }),
    store,
    driver,
    providers: {
      orchestrator: plannerProvider(),
      judge: passingJudge(),
      coders: [flaky],
      reviewer: reviewerProvider(0.95, 0.9),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  assert.equal(result.verdict, 'PASS');
  assert.equal(events.filter((e) => e.type === 'candidate.error').length, 1);
  assert.equal(events.filter((e) => e.type === 'escalate').length, 0, 'a mixed rung does not escalate');
});

test('D1: the rung-level code list is the one the escalation rule reads', () => {
  for (const code of ['provider-bad-request', 'not-armed', 'provider-unavailable', 'provider-timeout', 'execution']) {
    assert.ok(RUNG_LEVEL_ERROR_CODES.includes(code), code);
  }
  assert.ok(!RUNG_LEVEL_ERROR_CODES.includes('aborted'), 'an abort always propagates');
});

/* =========================================================================
 * Defect 2 — repairRounds (how many passes) vs repairTurns (turns per pass)
 * ========================================================================= */

test('D2: repairRounds counts the passes and both survive into the run row', options, async () => {
  const { store, dir, driver } = fixture('d2-rounds');
  const result = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js' },
    repoPath: '/repo/demo',
    // Two repair passes, five turns inside each: two numbers, two meanings.
    config: baseConfig(dir, { repairRounds: 2, repairTurns: 5 }),
    store,
    driver,
    providers: {
      orchestrator: plannerProvider(),
      judge: passingJudge(),
      coders: [coderProvider('ollama:qwen3:8b', BROKEN_MATH)],
      reviewer: reviewerProvider(0.95, 0.9),
    },
    bands: BANDS,
    rubric: RUBRIC,
  });

  const ids = result.candidates.map((c) => c.candidateId).sort();
  assert.deepEqual(ids, ['c0a0', 'c0a1', 'c0a2'], 'one attempt plus exactly two repair rounds');

  const stored = store.getRun(result.runId);
  assert.equal(stored.config.repairRounds, 2);
  assert.equal(stored.config.repairTurns, 5, 'a resume rehydrates the per-pass turn cap too');
});

test('D2: runRepair caps one pass at config.repairTurns', async () => {
  const driver = createFakeDriver({ files: BASE_FILES });
  const wt = await loneWorktree(driver);
  const events = [];
  const provider = createFakeProvider({
    ref: 'ollama:qwen3:8b',
    script: () => ({ toolCalls: [{ name: 'read_file', input: { path: 'src/index.js' } }] }),
  });

  const out = await runRepair({
    runId: 'r_hard',
    candidateId: 'c0a0',
    wt,
    plan: PLAN_JSON,
    coder: provider,
    store: stubStore(events),
    config: { repairTurns: 2 },
    feedback: { stage: 'l2-dynamic', reasons: ['1 newly failing test(s): add adds numbers'], hits: [], failing: [] },
  });

  assert.equal(out.turns, 2, 'the per-pass turn cap is repairTurns, not repairRounds');
  assert.equal(out.reason, 'max-turns');
});

/* =========================================================================
 * Defect 3 — the node --test summary header is not a failing test
 * ========================================================================= */

const NODE_TEST_OUTPUT = [
  '> demo@1.0.0 test',
  '> node --test',
  '',
  'tests/adder.test.js:',
  '  \u2714 add adds two numbers (1.2345ms)',
  '  \u2716 add rejects a string argument (3.4567ms)',
  '  \u2714 add is commutative (0.9876ms)',
  '',
  '\u2139 tests 3',
  '\u2139 suites 1',
  '\u2139 pass 2',
  '\u2139 fail 1',
  '\u2139 cancelled 0',
  '\u2139 skipped 0',
  '\u2139 todo 0',
  '\u2139 duration_ms 412.5',
  '',
  '\u2716 failing tests:',
  '',
  'test at tests/adder.test.js:12:1',
  '\u2716 add rejects a string argument (3.4567ms)',
  "  AssertionError [ERR_ASSERTION]: Expected values to be strictly equal",
  '',
].join('\n');

test('D3: the node --test summary header never lands in the failing list', () => {
  const parsed = parseTestOutput(NODE_TEST_OUTPUT, 'node');
  assert.equal(parsed.parsed, true);
  assert.deepEqual(parsed.failing, ['add rejects a string argument'], 'only real test names');
  assert.equal(parsed.passed, 2);
  assert.equal(parsed.failed, 1);
  // The header line is the exact shape that used to be reported as a failure.
  assert.ok(!parsed.failing.some((name) => /failing tests/i.test(name)));
});

test('D3: summary and header lines are classified as runner noise', () => {
  for (const noise of ['failing tests:', 'failing tests', 'passing tests:', 'tests 389', 'pass 388', 'duration_ms 412.5', 'test at tests/adder.test.js:12:1', '---']) {
    assert.equal(isRunnerNoise(noise), true, noise);
  }
  assert.equal(isRunnerNoise('add rejects a string argument'), false);
});

/* =========================================================================
 * Defect 4 — calibration cannot move a locked band or act on thin evidence
 * ========================================================================= */

/**
 * A store holding `n` cascade-PASS candidates that the calibration judge called
 * `bad`, each with the given metric hit banded `good`.
 */
function calibrationStore(label, { key, value, n }) {
  const { store } = tempStore(label);
  const runId = 'r_20260101_cal00000';
  store.createRun({ id: runId, task: { text: 'calibrate' }, repoPath: '/repo/demo', mode: 'cascade' });
  for (let i = 0; i < n; i += 1) {
    const candidateId = `c${i}a0`;
    store.upsertCandidate({ runId, candidateId, idx: i, attempt: 0, status: 'passed' });
    store.putResult({
      runId,
      candidateId,
      stage: 'l2-dynamic',
      version: 1,
      verdict: { decision: 'good', reasons: [], hits: [{ key, value, band: 'good' }], metrics: {} },
    });
    store.putJudgment({ runId, candidateId, kind: 'calibration', verdict: { decision: 'bad', rationale: 'wrong' } });
  }
  return store;
}

test('D4: a locked band is never proposed, however many judges disagree', options, () => {
  const store = calibrationStore('d4-locked', { key: 'newly_failing_tests', value: 0, n: 5 });
  const out = suggestBands({ store, bands: BANDS, minDisagreements: 3 });
  assert.equal(out.proposals.length, 0, 'the safety band held');
  assert.deepEqual(out.bandsPatched['l2-dynamic'].newly_failing_tests, BANDS['l2-dynamic'].newly_failing_tests);
  const skip = out.skipped.find((s) => s.key === 'newly_failing_tests');
  assert.ok(skip, 'the skip is reported');
  assert.equal(skip.stage, 'l2-dynamic');
  assert.equal(skip.reason, 'locked');
});

test('D4: one disagreement is not enough to move a threshold', options, () => {
  const single = calibrationStore('d4-thin', { key: 'changed_line_coverage', value: 0.9, n: 1 });
  const thin = suggestBands({ store: single, bands: BANDS, minDisagreements: 3 });
  assert.equal(thin.proposals.length, 0);
  const skip = thin.skipped.find((s) => s.key === 'changed_line_coverage');
  assert.ok(skip);
  assert.equal(skip.reason, 'n<3');
  assert.equal(skip.disagreements, 1);

  // Three independent disagreements are: the proposal goes through.
  const many = calibrationStore('d4-thick', { key: 'changed_line_coverage', value: 0.9, n: 3 });
  const thick = suggestBands({ store: many, bands: BANDS, minDisagreements: 3 });
  const proposal = thick.proposals.find((p) => p.key === 'changed_line_coverage');
  assert.ok(proposal, 'three disagreements are enough');
  assert.equal(proposal.field, 'hi');
  assert.ok(proposal.to > BANDS['l2-dynamic'].changed_line_coverage.hi);
});

/* =========================================================================
 * Defect 5 — edit_file, the tool guidance, and the no-progress watchdog
 * ========================================================================= */

async function coderWith(script, { config = {}, files = BASE_FILES, shell, maxTurns } = {}) {
  const driver = createFakeDriver(shell ? { files, shell } : { files });
  const wt = await loneWorktree(driver);
  const events = [];
  const provider = createFakeProvider({ ref: 'ollama:qwen3:8b', script });
  const out = await runCoder({
    runId: 'r_hard',
    candidateId: 'c0a0',
    wt,
    plan: PLAN_JSON,
    coder: provider,
    store: stubStore(events),
    config,
    maxTurns,
  });
  return { out, events, provider, wt };
}

test('D5: edit_file replaces exactly one occurrence', async () => {
  const calls = [
    { toolCalls: [{ name: 'edit_file', input: { path: 'src/index.js', old_string: "'demo'", new_string: "'demo2'" } }] },
    { toolCalls: [{ name: 'finish', input: { summary: 'renamed the export' } }] },
  ];
  const { out, wt, provider } = await coderWith(calls);
  assert.equal(out.reason, 'finish');
  assert.equal(await wt.readFile('src/index.js'), "export const name = 'demo2';\n");
  const result = provider.calls[1].messages.at(-1).content[0].content;
  assert.ok(result.startsWith('edited src/index.js (1 replacement'), result);
});

test('D5: edit_file refuses with the occurrence count instead of guessing', async () => {
  const files = { 'src/dup.js': 'const a = 1;\nconst a = 1;\n' };
  const calls = [
    { toolCalls: [{ name: 'edit_file', input: { path: 'src/dup.js', old_string: 'const a = 1;', new_string: 'const b = 2;' } }] },
    { toolCalls: [{ name: 'edit_file', input: { path: 'src/dup.js', old_string: 'nowhere', new_string: 'x' } }] },
    { toolCalls: [{ name: 'edit_file', input: { path: 'src/missing.js', old_string: 'a', new_string: 'b' } }] },
    { toolCalls: [{ name: 'finish', input: { summary: 'gave up' } }] },
  ];
  const { provider, wt } = await coderWith(calls, { files });
  const resultOf = (i) => provider.calls[i].messages.at(-1).content[0].content;
  assert.ok(resultOf(1).includes('appears 2 times'), resultOf(1));
  assert.ok(resultOf(2).includes('0 occurrences'), resultOf(2));
  assert.ok(resultOf(3).startsWith('not found:'), resultOf(3));
  assert.equal(await wt.readFile('src/dup.js'), 'const a = 1;\nconst a = 1;\n', 'nothing was written');
});

test('D5: edit_file matches an LF old_string against a CRLF file and keeps CRLF', async () => {
  const files = { 'src/crlf.js': 'const a = 1;\r\nexport { a };\r\n' };
  const calls = [
    { toolCalls: [{ name: 'edit_file', input: { path: 'src/crlf.js', old_string: 'const a = 1;\nexport { a };', new_string: 'const a = 1;\nconst b = 2;\nexport { a, b };' } }] },
    { toolCalls: [{ name: 'finish', input: { summary: 'added b' } }] },
  ];
  const { provider, wt } = await coderWith(calls, { files });
  const result = provider.calls[1].messages.at(-1).content[0].content;
  assert.ok(result.startsWith('edited src/crlf.js (1 replacement'), result);
  assert.equal(await wt.readFile('src/crlf.js'), 'const a = 1;\r\nconst b = 2;\r\nexport { a, b };\r\n');
});

test('D5: a run that hits its timeout says so instead of showing a bare exit code', async () => {
  const files = { ...BASE_FILES };
  const driver = createFakeDriver({ files, exec: { 'npm test': { exitCode: 1, timedOut: true, stdout: '' } } });
  const wt = await loneWorktree(driver);
  const provider = createFakeProvider({
    ref: 'ollama:qwen3:8b',
    script: [
      { toolCalls: [{ name: 'run', input: { cmd: 'npm test' } }] },
      { toolCalls: [{ name: 'finish', input: { summary: 'done' } }] },
    ],
  });
  await runCoder({ runId: 'r_hard', candidateId: 'c0a0', wt, plan: PLAN_JSON, coder: provider, store: stubStore([]), config: {} });
  const result = provider.calls[1].messages.at(-1).content[0].content;
  assert.ok(result.startsWith('timed out after 120000 ms'), result);
  assert.ok(result.includes('do not rerun it'), result);
});

test('D5: the system prompt sends the coder to edit_file and away from run', async () => {
  const { provider } = await coderWith([{ toolCalls: [{ name: 'finish', input: { summary: 'done' } }] }]);
  const system = provider.calls[0].system;
  assert.ok(system.includes('TOOL DISCIPLINE'));
  assert.ok(/use edit_file/i.test(system), 'edit_file is the tool for an existing file');
  assert.ok(/write_file only for a NEW file/i.test(system));
  assert.ok(/Never inspect or edit files through run/i.test(system));
  assert.ok(/Diagnosis is not the task/i.test(system));
});

test('D5: the watchdog nudges once, then stops the coder with no-progress', async () => {
  const { out, events, provider } = await coderWith(
    () => ({ toolCalls: [{ name: 'read_file', input: { path: 'src/index.js' } }] }),
    { maxTurns: 40 },
  );
  assert.equal(out.reason, 'no-progress');
  assert.equal(out.finished, false);
  assert.equal(out.turns, NO_PROGRESS_WINDOW * 2, 'one nudge window, then one more');

  const nudges = events.filter((e) => e.type === 'coder.nudge');
  const stops = events.filter((e) => e.type === 'coder.no-progress');
  assert.equal(nudges.length, 1);
  assert.equal(stops.length, 1);
  assert.equal(stops[0].payload.candidateId, 'c0a0');

  // The nudge is a real user turn the model sees, not just an event.
  const afterNudge = provider.calls[NO_PROGRESS_WINDOW].messages.at(-1);
  assert.equal(afterNudge.role, 'user');
  assert.ok(String(afterNudge.content).includes('without a file change'), String(afterNudge.content));
});

test('D5: a coder that writes keeps its watchdog counter at zero', async () => {
  const script = (call, index) => {
    if (index === 12) return { toolCalls: [{ name: 'finish', input: { summary: 'done' } }] };
    if (index % 3 === 2) {
      return { toolCalls: [{ name: 'write_file', input: { path: 'src/math.js', content: GOOD_MATH } }] };
    }
    return { toolCalls: [{ name: 'read_file', input: { path: 'src/index.js' } }] };
  };
  const { out, events } = await coderWith(script, { maxTurns: 20 });
  assert.equal(out.reason, 'finish');
  assert.equal(events.filter((e) => e.type === 'coder.nudge').length, 0);
});

/* =========================================================================
 * Defect 6 — coder.turn carries bounded, replayable detail
 * ========================================================================= */

test('D6: turnDetail keeps the identifying argument and bounds every field', () => {
  const detail = turnDetail(
    [
      { name: 'write_file', input: { path: 'src/math.js', content: 'x'.repeat(5000) } },
      { name: 'run', input: { cmd: 'c'.repeat(400) } },
      { name: 'search', input: { pattern: 'add\\(' } },
    ],
    ['wrote src/math.js (42 bytes)', 'r'.repeat(900), 'src/math.js:1: add('],
  );
  assert.equal(detail[0].name, 'write_file');
  assert.equal(detail[0].input, 'src/math.js', 'the path, never the content');
  assert.ok(!JSON.stringify(detail).includes('xxxxxxxxxx'), 'file content never reaches the event');
  assert.equal(detail[1].input.length, 200);
  assert.equal(detail[1].result.length, 300);
  assert.equal(detail[2].input, 'add\\(');
  assert.ok(JSON.stringify(detail).length < 4096);
});

test('D6: turnDetail drops trailing calls rather than blow the 4 KB budget', () => {
  const calls = [];
  const results = [];
  for (let i = 0; i < 40; i += 1) {
    calls.push({ name: 'read_file', input: { path: `src/file-${i}.js` } });
    results.push('r'.repeat(300));
  }
  const detail = turnDetail(calls, results);
  assert.ok(detail.length >= 1);
  assert.ok(JSON.stringify(detail).length <= 4000, `payload was ${JSON.stringify(detail).length} bytes`);
});

test('D6: coder.turn events name the tool, the path and the head of the result', async () => {
  const calls = [
    { toolCalls: [{ name: 'read_file', input: { path: 'src/index.js' } }] },
    { toolCalls: [{ name: 'write_file', input: { path: 'src/math.js', content: GOOD_MATH } }] },
    { toolCalls: [{ name: 'finish', input: { summary: 'wrote src/math.js' } }] },
  ];
  const { events } = await coderWith(calls);
  const turns = events.filter((e) => e.type === 'coder.turn');
  assert.equal(turns.length, 3);
  assert.deepEqual(turns[0].payload.detail, [
    { name: 'read_file', input: 'src/index.js', result: "export const name = 'demo';\n" },
  ]);
  assert.equal(turns[1].payload.detail[0].name, 'write_file');
  assert.equal(turns[1].payload.detail[0].input, 'src/math.js');
  assert.ok(turns[1].payload.detail[0].result.startsWith('wrote src/math.js'));
  assert.ok(!JSON.stringify(turns[1].payload).includes('return a + b'), 'the patch body stays out of the event');
});

/* =========================================================================
 * Defect 7 — validate before the run row exists, and print the failure once
 * ========================================================================= */

const ARMED_RUNGS = [
  { id: 'orchestrator', kind: 'orchestrator', ref: 'ollama:qwen3:8b', armed: true, reason: 'ok' },
  { id: 'judge', kind: 'judge', ref: 'ollama:qwen3:32b', armed: true, reason: 'ok' },
  { id: 'coder-0', kind: 'coder', ref: 'ollama:qwen3:8b', armed: true, reason: 'ok' },
];

function preconditionDeps({ dir = true, gitDir = 0, ref = 0 } = {}) {
  return {
    statSync: () => {
      if (!dir) throw new Error('ENOENT');
      return { isDirectory: () => true };
    },
    git: (args) => (args[1] === '--git-dir' ? { status: gitDir } : { status: ref }),
  };
}

test('D7: a repo path that is not a directory is bad-args, before anything is created', () => {
  assert.throws(
    () => validateRunPreconditions(
      { repoPath: '/nope', baseRef: null, config: { driver: 'local' }, arming: { rungs: ARMED_RUNGS }, ladders: { coders: ['ollama:qwen3:8b'] } },
      preconditionDeps({ dir: false }),
    ),
    (err) => isLadderError(err, 'bad-args') && /not a directory/.test(err.message),
  );
});

test('D7: a directory that is not a git repository is bad-args', () => {
  assert.throws(
    () => validateRunPreconditions(
      { repoPath: '/tmp/plain', baseRef: null, config: { driver: 'local' }, arming: { rungs: ARMED_RUNGS }, ladders: { coders: [] } },
      preconditionDeps({ gitDir: 128 }),
    ),
    (err) => isLadderError(err, 'bad-args') && /not a git repository/.test(err.message),
  );
});

test('D7: a base ref git cannot resolve is bad-ref', () => {
  assert.throws(
    () => validateRunPreconditions(
      { repoPath: '/tmp/repo', baseRef: 'no-such-branch', config: { driver: 'local' }, arming: { rungs: ARMED_RUNGS }, ladders: { coders: [] } },
      preconditionDeps({ ref: 128 }),
    ),
    (err) => isLadderError(err, 'bad-ref') && /does not resolve/.test(err.message),
  );
});

test('D7: every rung the run will use has to be armed', () => {
  const rungs = [
    { id: 'orchestrator', kind: 'orchestrator', ref: 'ollama:a', armed: true, reason: 'ok' },
    { id: 'judge', kind: 'judge', ref: 'anthropic:b', armed: false, reason: 'ANTHROPIC_API_KEY is not set' },
    { id: 'coder-0', kind: 'coder', ref: 'ollama:a', armed: true, reason: 'ok' },
  ];
  assert.throws(
    () => validateRunPreconditions(
      { repoPath: '/tmp/repo', baseRef: null, config: { driver: 'fake', mode: 'cascade' }, arming: { rungs }, ladders: { coders: ['ollama:a'] } },
      preconditionDeps(),
    ),
    (err) => isLadderError(err, 'not-armed') && err.message.includes('judge'),
  );
  // An unarmed rung this ladder never reaches is not this run's problem.
  const unusedCoder = [...rungs.slice(0, 1), { id: 'coder-9', kind: 'coder', ref: 'cerebras:z', armed: false, reason: 'no key' }];
  assert.equal(
    validateRunPreconditions(
      { repoPath: '/tmp/repo', baseRef: null, config: { driver: 'fake', mode: 'direct' }, arming: { rungs: unusedCoder }, ladders: { coders: ['ollama:a'] } },
      preconditionDeps(),
    ),
    true,
  );
  // The mock provider arms everything by construction.
  assert.equal(
    validateRunPreconditions(
      { repoPath: '/tmp/repo', baseRef: null, config: { driver: 'fake', mockProvider: true }, arming: { rungs }, ladders: { coders: ['ollama:a'] } },
      preconditionDeps(),
    ),
    true,
  );
});

test('D7: the CLI rejects a bad repo before it writes a run row, once', gitOptions, () => {
  const home = tempDir('d7-home');
  const missing = path.join(tempDir('d7-gone'), 'not-here');
  const res = runCli(['run', 'do a thing', '--repo', missing], { home });
  assert.equal(res.status, EXIT.BAD_ARGS, res.all.slice(-500));
  const message = 'is not a directory';
  assert.equal(res.all.split(message).length - 1, 1, `printed more than once:\n${res.all}`);

  // No run row: the failure cost nothing. Reading the rows needs the store,
  // which a Node without node:sqlite cannot open; the rejection above does not.
  if (HAS_SQLITE) {
    const listed = runCli(['status', '--json'], { home });
    assert.equal(listed.status, EXIT.OK, listed.all.slice(-400));
    assert.deepEqual(JSON.parse(listed.stdout).runs, []);
  }

  const plain = tempDir('d7-plain');
  const notGit = runCli(['run', 'do a thing', '--repo', plain], { home });
  assert.equal(notGit.status, EXIT.BAD_ARGS);
  assert.ok(notGit.all.includes('not a git repository'), notGit.all.slice(-400));

  const repo = makeRepo('d7-repo');
  const badRef = runCli(['run', 'do a thing', '--repo', repo, '--base', 'no-such-ref'], { home });
  assert.equal(badRef.status, EXIT.BAD_ARGS);
  assert.ok(badRef.all.includes('bad-ref'), badRef.all.slice(-400));
});

/* =========================================================================
 * Defect 8 — a receipt names the repository, not the operator's disk
 * ========================================================================= */

function testSigner() {
  const { publicKey, privateKey } = generateKeyPair();
  return { publicKey, privateKey, key_fingerprint: keyFingerprint(publicKey) };
}

test('D8: the receipt carries a basename and never the local path', () => {
  const receipt = buildReceipt({
    run: { id: 'r_20260101_aaaa0000', repo_path: path.join('C:', 'Users', 'someone', 'Desktop', 'secret-project'), base_sha: 'a'.repeat(40) },
    verdict: 'PASS',
    createdAt: '2026-01-01T00:00:00.000Z',
  });
  assert.equal(receipt.version, 2);
  assert.equal(receipt.repo.basename, 'secret-project');
  assert.equal(receipt.repo.path, undefined, 'the operator directory layout never travels');
  assert.equal(receipt.repo.path_sha256.length, 64, 'the hash still identifies the checkout');
  assert.ok(!JSON.stringify(receipt).includes('Users'), JSON.stringify(receipt.repo));
});

test('D8: verify accepts version 1 and version 2 and refuses anything else', () => {
  const signer = testSigner();
  const base = buildReceipt({ run: { id: 'r_20260101_bbbb0000', repo_path: '/srv/demo' }, verdict: 'PASS', createdAt: '2026-01-01T00:00:00.000Z' });
  for (const version of [1, 2]) {
    const signed = signReceipt({ ...base, version }, signer);
    const result = verifyReceipt(signed);
    assert.equal(result.ok, true, `version ${version}: ${result.reason ?? ''}`);
  }
  const future = verifyReceipt(signReceipt({ ...base, version: 3 }, signer));
  assert.equal(future.ok, false);
  assert.ok(/unsupported receipt version 3/.test(future.reason), future.reason);
});

/* =========================================================================
 * Defect 9 — a timeout says what timed out, and model calls have their own cap
 * ========================================================================= */

test('D9: a baseline test timeout skips L2 instead of blaming the candidate', async () => {
  const result = await l2Run({
    wt: null,
    baseline: { tests: { timedOut: true, passed: 0, failed: 0, failing: [] } },
    commands: { test: TEST_CMD },
    bands: BANDS,
    config: { stageTimeoutMs: 60000 },
  });
  assert.equal(result.decision, 'skip');
  assert.deepEqual(result.reasons, ['test command timed out at baseline too (60000 ms)']);
  assert.equal(result.artifacts.baselineTimedOut, true);
  assert.deepEqual(result.hits, [], 'nothing was measured, so nothing is banded');
});

test('D9: a baseline typecheck or lint timeout is reported as such', async () => {
  const driver = createFakeDriver({ files: BASE_FILES });
  const wt = await loneWorktree(driver, 'r_l1', 'c0a0');
  const result = await l1Run({
    wt,
    baseline: { typecheck: { timedOut: true, errors: [] }, lint: { timedOut: true, errors: [] } },
    commands: { typecheck: 'npx tsc --noEmit', lint: 'npx eslint .' },
    bands: BANDS,
    config: { stageTimeoutMs: 60000 },
  });
  assert.equal(result.decision, 'skip');
  assert.ok(result.reasons.includes('typecheck timed out at baseline too (60000 ms)'), result.reasons.join('; '));
  assert.ok(result.reasons.includes('lint timed out at baseline too (60000 ms)'), result.reasons.join('; '));
  assert.equal(result.artifacts.baselineTypecheckTimedOut, true);
  assert.equal(result.artifacts.baselineLintTimedOut, true);
  assert.equal(wt.commands.length, 0, 'neither tool was run again');
});

test('D9: the stage wrapper outlives its command so the real reason surfaces', options, async () => {
  const { store } = tempStore('d9-wrapper');
  const runId = 'r_20260101_cccc0000';
  store.createRun({ id: runId, task: { text: 'x' }, repoPath: '/repo/demo', mode: 'cascade' });
  const driver = createFakeDriver({
    files: BASE_FILES,
    exec: {
      // The command itself takes longer than the stage budget and reports its
      // own timeout. A wrapper without the margin would win the race and report
      // the useless 'stage timed out' instead.
      [TEST_CMD]: async () => {
        await new Promise((resolve) => setTimeout(resolve, 250));
        return { stdout: '', stderr: '', exitCode: null, timedOut: true };
      },
    },
  });
  const wt = await loneWorktree(driver, runId, 'c0a0');
  const { perCandidate } = await runCascade({
    runId,
    candidates: [{ candidateId: 'c0a0', wt, diffText: MATH_DIFF }],
    store,
    config: { stageTimeoutMs: 60 },
    plan: PLAN_JSON,
    commands: { test: TEST_CMD },
    baseline: { tests: { passed: 1, failed: 0, failing: [], exitCode: 0 } },
    bands: BANDS,
    rubric: RUBRIC,
    mode: 'direct',
  });
  const l2 = perCandidate.get('c0a0').results['l2-dynamic'];
  assert.equal(l2.decision, 'unsure');
  assert.ok(
    l2.reasons.some((r) => /test command timed out after 60 ms/.test(r)),
    `the inner reason was lost: ${l2.reasons.join('; ')}`,
  );
  assert.ok(!l2.reasons.includes('stage timed out'));
});

test('D9: model calls carry KOLM_CODE_MODEL_TIMEOUT_MS, not the stage budget', async () => {
  const config = { modelTimeoutMs: 123456, stageTimeoutMs: 60000 };
  const { provider } = await coderWith([{ toolCalls: [{ name: 'finish', input: { summary: 'done' } }] }], { config });
  assert.equal(provider.calls[0].timeoutMs, 123456, 'the coder uses the model timeout');

  const orchestrator = plannerProvider();
  await makePlan({
    task: { text: 'add a function add(a,b)' },
    brief: { root: '/repo/demo', files: [] },
    orchestrator,
    config,
    commands: { test: TEST_CMD },
  });
  assert.equal(orchestrator.calls[0].timeoutMs, 123456, 'the planner uses it too');

  const reviewer = reviewerProvider(0.95, 0.9);
  await l3Run({ diffText: MATH_DIFF, plan: PLAN_JSON, brief: { root: '/repo/demo', files: [] }, providers: { reviewer }, rubric: RUBRIC, bands: BANDS, config });
  assert.equal(reviewer.calls[0].timeoutMs, 123456, 'so does L3');

  const judge = passingJudge();
  await runJudge({ diffText: MATH_DIFF, plan: PLAN_JSON, brief: { root: '/repo/demo', files: [] }, judge, rubric: RUBRIC, config });
  assert.equal(judge.calls[0].timeoutMs, 123456, 'and so does L4');
});

/* =========================================================================
 * Defect 10 — repair rounds are only spent on actionable feedback
 * ========================================================================= */

test('D10: repairSkipReason separates infrastructure from patch feedback', () => {
  const actionable = {
    stage: 'l2-dynamic',
    reasons: ['1 newly failing test(s): add adds numbers'],
    hits: [{ key: 'newly_failing_tests', value: 1, band: 'bad' }],
    failing: ['add adds numbers'],
  };
  assert.equal(repairSkipReason({}, actionable), null);
  assert.equal(
    repairSkipReason({}, { stage: 'l2-dynamic', reasons: ['test command timed out after 600000 ms'], hits: [{ key: 'newly_failing_tests', value: 0, band: 'unsure' }], failing: [] }),
    'timeout',
  );
  assert.equal(repairSkipReason({}, { stage: 'l1-static', reasons: ['stage timed out'], hits: [], failing: [] }), 'timeout');
  assert.equal(
    repairSkipReason({}, { stage: 'l3-semantic', reasons: ['the reviewer could not produce a verdict'], hits: [], failing: [] }),
    'provider-failure',
  );
  assert.equal(
    repairSkipReason({}, { stage: 'l3-semantic', reasons: ['provider-unavailable: the endpoint refused the connection'], hits: [], failing: [] }),
    'provider-failure',
  );
  assert.equal(repairSkipReason({ judgeFailed: true }, actionable), 'judge-failed');
  assert.equal(repairSkipReason({}, { stage: 'unknown', reasons: [], hits: [], failing: [] }), 'no-actionable-reasons');
  // A named failing test is actionable even next to a timeout note.
  assert.equal(
    repairSkipReason({}, { stage: 'l2-dynamic', reasons: ['test command timed out after 600000 ms'], hits: [], failing: ['add adds numbers'] }),
    null,
  );
});

test('D10: a judge that cannot decide costs no repair round', options, async () => {
  const { store, dir, driver } = fixture('d10-judge');
  const events = [];
  await runTask({
    task: { text: 'add a function add(a,b) to src/math.js' },
    repoPath: '/repo/demo',
    config: baseConfig(dir, { repairRounds: 2 }),
    store,
    driver,
    providers: {
      orchestrator: plannerProvider(),
      judge: unparsableJudge(),
      coders: [coderProvider('ollama:qwen3:8b', GOOD_MATH)],
      // An unsure reviewer sends the candidate to a judge that cannot answer.
      reviewer: reviewerProvider(0.7, 0.5),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  const skipped = events.filter((e) => e.type === 'repair.skipped');
  assert.ok(skipped.length > 0, 'the skip is on the record');
  assert.equal(skipped[0].payload.reason, 'judge-failed');
  assert.equal(skipped[0].payload.candidateId, 'c0a0');
  assert.equal(events.filter((e) => e.type === 'repair').length, 0, 'no repair model call was made');
});

/* =========================================================================
 * Defect 11 — worktrees do not leak, and `clean` sweeps up
 * ========================================================================= */

/** A run that opens two worktrees and then dies on its only coder rung. */
function failingRun(store, dir, driver, extra = {}) {
  return runTask({
    task: { text: 'add a function add(a,b) to src/math.js' },
    repoPath: '/repo/demo',
    config: baseConfig(dir, { candidates: 2, ...extra }),
    store,
    driver,
    providers: {
      orchestrator: plannerProvider(),
      judge: passingJudge(),
      coders: [deadCoder('ollama:qwen3:8b')],
      reviewer: reviewerProvider(0.95, 0.9),
    },
    bands: BANDS,
    rubric: RUBRIC,
  });
}

test('D11: the error path destroys every worktree it opened', options, async () => {
  const { store, dir, driver } = fixture('d11-error');
  await assert.rejects(failingRun(store, dir, driver));

  const candidateIds = driver.destroyed.map((d) => d.candidateId);
  assert.ok(candidateIds.includes('c0a0'), `destroyed: ${JSON.stringify(candidateIds)}`);
  assert.ok(candidateIds.includes('c1a0'), `destroyed: ${JSON.stringify(candidateIds)}`);
  assert.ok(candidateIds.includes('base'), 'the base worktree goes too');
});

test('D11: keepWorktrees=all keeps the worktrees even on the error path', options, async () => {
  const { store, dir, driver } = fixture('d11-keep');
  await assert.rejects(failingRun(store, dir, driver, { keepWorktrees: 'all' }));
  assert.deepEqual(driver.destroyed, [], 'the operator asked to keep them');
});

test('D11: clean touches finished runs only, and never a running one', () => {
  const now = Date.parse('2026-09-09T12:00:00.000Z');
  const day = 86400000;
  const runs = [
    { id: 'r_running', status: 'running', updated_at: new Date(now - 10 * day).toISOString(), driver: 'local' },
    { id: 'r_done', status: 'done', updated_at: new Date(now - 10 * day).toISOString(), driver: 'local' },
    { id: 'r_error', status: 'error', updated_at: new Date(now - 10 * day).toISOString(), driver: 'fake' },
    { id: 'r_fail', status: 'fail', updated_at: new Date(now - 1 * day).toISOString(), driver: 'local' },
    { id: 'r_cloud', status: 'done', updated_at: new Date(now - 10 * day).toISOString(), driver: 'cloudflare' },
  ];
  const all = selectRunsToClean(runs, { now });
  assert.deepEqual(all.clean.map((r) => r.id), ['r_done', 'r_error', 'r_fail']);
  assert.deepEqual(
    all.skipped,
    [
      { runId: 'r_running', reason: 'status running' },
      { runId: 'r_cloud', reason: 'driver cloudflare' },
    ],
  );

  const old = selectRunsToClean(runs, { now, olderThanDays: 7 });
  assert.deepEqual(old.clean.map((r) => r.id), ['r_done', 'r_error']);
  assert.ok(old.skipped.some((s) => s.runId === 'r_fail' && s.reason === 'newer than 7 day(s)'));

  assert.deepEqual([...FINISHED_RUN_STATUSES], ['done', 'error', 'fail']);
});

test('D11: clean is a documented verb the CLI dispatches', () => {
  assert.ok(VERBS.includes('clean'));
  assert.ok(USAGE.includes('clean'));
  assert.ok(USAGE.includes('--older-than'));
  assert.ok(USAGE.includes('--dry-run'));
});

test('D11: the local driver can list what it would remove without removing it', gitOptions, async () => {
  const repo = makeRepo('d11-repo');
  const worktreesDir = tempDir('d11-wt');
  const driver = createLocalGitDriver({}, { worktreesDir });
  const runId = 'r_20260101_dddd0000';
  const wt = await driver.create({ repoPath: repo, runId, candidateId: 'c0a0' });
  assert.ok(fs.existsSync(wt.path));
  // Commit so the candidate branch exists and can be shown to survive.
  await wt.writeFile('math.js', GOOD_MATH);
  const committed = await wt.commit('candidate work');
  assert.equal(committed.branch, `kolm/code/${runId}/c0a0`);

  const dry = await driver.destroyRun(runId, { repoPath: repo, dryRun: true });
  assert.equal(dry.dryRun, true);
  assert.equal(dry.removed, 0);
  assert.equal(dry.paths.length, 1);
  assert.ok(fs.existsSync(wt.path), 'a dry run removes nothing');

  const real = await driver.destroyRun(runId, { repoPath: repo });
  assert.equal(real.removed, 1);
  assert.equal(real.existed, true);
  assert.ok(!fs.existsSync(wt.path));
  assert.ok(!fs.existsSync(real.runDir));

  // The branch survives: the candidate's work is not deleted with its checkout.
  const branches = spawnSync('git', ['branch', '--list', `kolm/code/${runId}/c0a0`], { cwd: repo, encoding: 'utf8' });
  assert.ok(branches.stdout.includes(runId), branches.stdout);

  const again = await driver.destroyRun(runId, { repoPath: repo });
  assert.equal(again.existed, false, 'cleaning twice is not an error');
});

/* =========================================================================
 * Defect 12 — `run` goes through a POSIX shell on Windows when there is one
 * ========================================================================= */

test('D12: resolveShell prefers Git for Windows bash on win32', () => {
  const bash = resolveShell({
    platform: 'win32',
    env: {},
    gitPath: '/opt/Git/cmd/git.exe',
    existsSync: (p) => /[\\/]bin[\\/]bash\.exe$/.test(p) && !/usr/.test(p),
  });
  assert.equal(bash.kind, 'bash');
  assert.equal(bash.posix, true);
  assert.ok(/bin[\\/]bash\.exe$/.test(bash.path), bash.path);

  const usrBash = resolveShell({
    platform: 'win32',
    env: {},
    gitPath: '/opt/Git/cmd/git.exe',
    existsSync: (p) => /usr[\\/]bin[\\/]bash\.exe$/.test(p),
  });
  assert.equal(usrBash.kind, 'bash');
  assert.ok(/usr[\\/]bin[\\/]bash\.exe$/.test(usrBash.path), usrBash.path);
});

test('D12: without a bash, win32 falls back to cmd.exe and loses the POSIX prefixes', () => {
  const shell = resolveShell({ platform: 'win32', env: {}, gitPath: null, existsSync: () => false });
  assert.equal(shell.kind, 'cmd');
  assert.equal(shell.posix, false);
  assert.equal(shell.path, null);

  const allowed = allowedForShell(['npm', 'node', 'ls', 'cat', 'grep', 'git diff'], shell);
  assert.deepEqual(allowed, ['npm', 'node', 'git diff']);
  for (const cmd of POSIX_ONLY_COMMANDS) assert.ok(!allowed.includes(cmd), cmd);
  // A POSIX shell keeps them.
  assert.deepEqual(allowedForShell(['npm', 'ls'], { posix: true }), ['npm', 'ls']);
});

test('D12: KOLM_CODE_SHELL overrides the search, and non-win32 stays sh', () => {
  assert.equal(resolveShell({ platform: 'linux', env: {} }).kind, 'sh');
  assert.equal(resolveShell({ platform: 'linux', env: {} }).posix, true);
  const override = resolveShell({ platform: 'win32', env: {}, override: 'C:/msys64/usr/bin/bash.exe' });
  assert.equal(override.kind, 'bash');
  assert.equal(override.path, 'C:/msys64/usr/bin/bash.exe');
  const viaEnv = resolveShell({ platform: 'win32', env: { KOLM_CODE_SHELL: '/usr/bin/bash' } });
  assert.equal(viaEnv.kind, 'bash');
  const forcedCmd = resolveShell({ platform: 'win32', env: {}, override: 'C:/Windows/System32/cmd.exe' });
  assert.equal(forcedCmd.kind, 'cmd');
  assert.equal(forcedCmd.posix, false);
});

test('D12: the coder is told which shell and platform it has', async () => {
  const cmdShell = { kind: 'cmd', path: null, posix: false, platform: 'win32' };
  const { provider, wt } = await coderWith(
    [{ toolCalls: [{ name: 'finish', input: { summary: 'done' } }] }],
    { shell: cmdShell },
  );
  const system = provider.calls[0].system;
  assert.ok(system.includes('ENVIRONMENT'), system.slice(0, 200));
  assert.ok(system.includes('platform: win32'));
  assert.ok(system.includes('run executes through: cmd.exe'));
  assert.ok(system.includes('There is no POSIX shell here'));
  assert.ok(/read_file, list_files and search never go through the shell/.test(system));
  const prefixes = /allowed run prefixes: (.*)/.exec(system)[1].split(', ');
  for (const cmd of POSIX_ONLY_COMMANDS) assert.ok(!prefixes.includes(cmd), `${cmd} was offered to cmd.exe`);
  assert.deepEqual(allowedCommandsFor(wt, {}), prefixes);
});

test('D12: a bash worktree keeps the POSIX prefixes and says so', async () => {
  const bashShell = { kind: 'bash', path: 'C:/Program Files/Git/bin/bash.exe', posix: true, platform: 'win32' };
  const { provider } = await coderWith(
    [{ toolCalls: [{ name: 'finish', input: { summary: 'done' } }] }],
    { shell: bashShell },
  );
  const system = provider.calls[0].system;
  assert.ok(system.includes('run executes through: bash (C:/Program Files/Git/bin/bash.exe)'), system);
  assert.ok(!system.includes('There is no POSIX shell here'));
  assert.ok(/allowed run prefixes: .*\bls\b/.test(system));
});

/* =========================================================================
 * Defect 6 + 11, end to end through the CLI
 * ========================================================================= */

test('D6/D11: show --events returns the event list and clean sweeps the run', options, () => {
  const home = tempDir('cli-home');
  const repo = makeRepo('cli-repo');
  const run = runCli(
    ['run', 'add a subtract helper next to add', '--repo', repo, '--candidates', '1', '--quiet'],
    { home, extra: { KOLM_TEST_MOCK_PROVIDER: '1', KOLM_CODE_DRIVER: 'fake' } },
  );
  assert.ok([EXIT.OK, EXIT.GATE_FAIL].includes(run.status), `run exited ${run.status}: ${run.all.slice(-600)}`);

  const listed = runCli(['status', '--json'], { home });
  const runId = JSON.parse(listed.stdout).runs[0].id;

  const summary = JSON.parse(runCli(['show', runId, '--json'], { home }).stdout);
  assert.equal(typeof summary.eventCount, 'number');
  assert.ok(summary.eventCount > 0);
  assert.equal(summary.events, summary.eventCount, 'without --events the payload stays a count');

  const withEvents = JSON.parse(runCli(['show', runId, '--json', '--events'], { home }).stdout);
  assert.ok(Array.isArray(withEvents.events), 'the event list is included');
  assert.equal(withEvents.events.length, withEvents.eventCount);
  const turn = withEvents.events.find((e) => e.type === 'coder.turn');
  if (turn) {
    assert.ok(Array.isArray(turn.payload.detail), 'coder.turn carries its tool detail');
    assert.ok(JSON.stringify(turn.payload).length < 8192);
  }

  const dry = JSON.parse(runCli(['clean', '--dry-run', '--json'], { home }).stdout);
  assert.equal(dry.dryRun, true);
  assert.ok(Array.isArray(dry.runs));

  const cleaned = runCli(['clean', '--json'], { home });
  assert.equal(cleaned.status, EXIT.OK, cleaned.all.slice(-400));
  const parsed = JSON.parse(cleaned.stdout);
  assert.equal(parsed.dryRun, false);
  assert.ok(Array.isArray(parsed.skipped));

  const text = runCli(['clean'], { home });
  assert.equal(text.status, EXIT.OK);
  assert.ok(text.stdout.includes('branches kolm/code/<runId>/<candidate> are never deleted'));
  assert.ok(/^[\x20-\x7e\n]*$/.test(text.stdout), 'clean output stays inside printable ascii');
});

test('D13: an unarmed coder rung is reported once at run start instead of vanishing', () => {
  const rungs = [
    { id: 'orchestrator', kind: 'orchestrator', ref: 'ollama:a', armed: true, reason: 'ok' },
    { id: 'coder0', kind: 'coder', ref: 'cerebras:qwen-3-coder-480b', armed: false, reason: 'CEREBRAS_API_KEY is not set' },
    { id: 'coder1', kind: 'coder', ref: 'ollama:a', armed: true, reason: 'ok' },
    { id: 'reviewer0', kind: 'reviewer', ref: 'ollama:a', armed: true, reason: 'ok' },
  ];
  const lines = droppedRungLines({ rungs }, { coders: ['ollama:a'], reviewer: 'ollama:a' });
  assert.deepEqual(lines, [
    'skipping coder rung cerebras:qwen-3-coder-480b: CEREBRAS_API_KEY is not set',
  ]);
  assert.ok(lines.every((l) => /^[\x20-\x7e]*$/.test(l)), 'ascii only');
});

test('D13: when every coder rung is unarmed the fallback onto the orchestrator is named', () => {
  const rungs = [
    { id: 'orchestrator', kind: 'orchestrator', ref: 'ollama:a', armed: true, reason: 'ok' },
    { id: 'coder0', kind: 'coder', ref: 'deepseek:deepseek-chat', armed: false, reason: 'DEEPSEEK_API_KEY is not set' },
    { id: 'reviewer0', kind: 'reviewer', ref: 'anthropic:claude-haiku-4-5-20251001', armed: false, reason: 'ANTHROPIC_API_KEY is not set' },
  ];
  const lines = droppedRungLines({ rungs }, { coders: ['ollama:a'], reviewer: null });
  assert.equal(lines.length, 4, lines.join('\n'));
  assert.match(lines[0], /skipping coder rung deepseek:deepseek-chat: DEEPSEEK_API_KEY/);
  assert.match(lines[1], /no coder rung is armed; the orchestrator ollama:a writes the candidates/);
  assert.match(lines[2], /skipping reviewer rung anthropic:claude-haiku-4-5-20251001/);
  assert.match(lines[3], /no reviewer rung is armed; l3-semantic is skipped/);
  assert.deepEqual(droppedRungLines({ rungs: ARMED_RUNGS }, { coders: ['ollama:qwen3:8b'], reviewer: null }), []);
});

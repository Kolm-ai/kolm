// Owner D — runTask end to end (spec 6.4). Three scenarios:
//   1. a clean pass over the fake driver, with a signed receipt,
//   2. a rung that cannot fix its own patch, escalating to the next rung,
//   3. the real local-git driver on a throwaway repository.
// Everything is scripted: no network, no real models, no repository outside a
// temp directory.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

import { runTask } from '../src/ladder/orchestrator.js';
import { verifyReceipt } from '../src/ladder/receipt.js';
import { openStore, sqliteAvailable } from '../src/ladder/store.js';
import { createFakeDriver } from '../src/ladder/worktree/fake.js';
import { createDriver } from '../src/ladder/worktree/index.js';
import { createFakeProvider } from '../src/ladder/providers/fake.js';
import { loadBands } from '../src/ladder/bands.js';
import { loadRubric } from '../src/ladder/rubric.js';
import { evalRuns, formatEvalTable } from '../src/ladder/eval.js';

const HAS_SQLITE = sqliteAvailable();
const options = HAS_SQLITE ? {} : { skip: 'node:sqlite is unavailable on this Node' };

const BANDS = loadBands();
const RUBRIC = loadRubric();
const RUBRIC_IDS = RUBRIC.items.map((item) => item.id);

function hasGit() {
  try {
    execFileSync('git', ['--version'], { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}
const gitOptions = HAS_SQLITE && hasGit()
  ? {}
  : { skip: HAS_SQLITE ? 'git is not on PATH' : 'node:sqlite is unavailable on this Node' };

// ----------------------------------------------------------------- fixtures

const tempDirs = [];
const openStores = [];

function tempDir(label) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `kolm-e2e-${label}-`));
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
      // already closed is fine
    }
  }
  for (const dir of tempDirs) {
    try {
      fs.rmSync(dir, { recursive: true, force: true });
    } catch {
      // a lingering WAL handle on Windows is not a test failure
    }
  }
});

const BASE_FILES = {
  'package.json': `${JSON.stringify({ name: 'demo', version: '1.0.0', scripts: { test: 'node --test' } }, null, 2)}\n`,
  'README.md': '# demo\n\nA tiny repository used by the kolm code end-to-end test.\n',
  'src/index.js': "export const name = 'demo';\n",
};

const GOOD_MATH = [
  '/** Add two numbers. */',
  'export function add(a, b) {',
  '  return a + b;',
  '}',
  '',
].join('\n');

const TODO_MATH = [
  '/** Add two numbers. */',
  'export function add(a, b) {',
  '  // TODO: actually add the arguments',
  '  return 0;',
  '}',
  '',
].join('\n');

const MATH_TEST = [
  "import test from 'node:test';",
  "import assert from 'node:assert/strict';",
  "import { add } from '../src/math.js';",
  '',
  "test('add adds numbers', () => {",
  '  assert.equal(add(2, 3), 5);',
  '});',
  '',
].join('\n');

const TEST_CMD = 'npm test --silent';

/** Scripted test output that depends on what the worktree actually contains. */
function scriptedTests(cmd, wt) {
  const math = wt.readFileSync('src/math.js');
  if (math === null) {
    return { stdout: 'ok 1 - index exports a name\n# tests 1\n# pass 1\n# fail 0\n', exitCode: 0 };
  }
  if (/return a \+ b/.test(math)) {
    return {
      stdout: 'ok 1 - index exports a name\nok 2 - add adds numbers\n# tests 2\n# pass 2\n# fail 0\n',
      exitCode: 0,
    };
  }
  return {
    stdout: 'ok 1 - index exports a name\nnot ok 2 - add adds numbers\n# tests 2\n# pass 1\n# fail 1\n',
    exitCode: 1,
  };
}

const PLAN_JSON = {
  title: 'add add(a, b) to src/math.js',
  summary: 'Create src/math.js exporting add(a, b) and cover it with a node:test file.',
  acceptance: [
    'src/math.js exports a function add(a, b) that returns the sum',
    'test/math.test.js asserts add(2, 3) === 5',
  ],
  files_expected: ['src/math.js', 'test/math.test.js'],
  forbidden_paths: ['package.json'],
  test_command: TEST_CMD,
  risk: 'low',
  budget_hint_usd: 0.5,
  clarifications: [],
};

function plannerProvider(ref = 'anthropic:claude-sonnet') {
  return createFakeProvider({ ref, script: () => ({ json: PLAN_JSON, costUsd: 0.003 }) });
}

function reviewerProvider(score = 0.95, confidence = 0.9) {
  return createFakeProvider({
    ref: 'anthropic:claude-haiku',
    script: () => ({
      json: {
        items: RUBRIC_IDS.map((id) => ({ id, score, evidence: 'the diff adds the function and a test for it' })),
        overall: score,
        confidence,
        blocking: [],
      },
      costUsd: 0.002,
    }),
  });
}

/** A coder that writes `files` on its first turn and finishes on its second. */
function coderProvider(ref, files, { finishOnFirstTurn = false, costUsd = 0.001 } = {}) {
  return createFakeProvider({
    ref,
    script: (call) => {
      const first = (call.messages || []).length === 1;
      const writes = Object.entries(files).map(([p, content]) => ({
        name: 'write_file',
        input: { path: p, content },
      }));
      const finish = { name: 'finish', input: { summary: `wrote ${Object.keys(files).join(', ')}` } };
      if (first) return { toolCalls: finishOnFirstTurn ? [...writes, finish] : writes, costUsd };
      return { toolCalls: [finish], costUsd };
    },
  });
}

function baseConfig(codeDir) {
  return {
    mode: 'cascade',
    candidates: 2,
    coderTurns: 6,
    repairRounds: 1,
    repairTurns: 6,
    maxCostUsd: 5,
    calibrationRate: 0,
    stageTimeoutMs: 60000,
    keepWorktrees: 'losers-destroyed',
    codeDir,
  };
}

// ------------------------------------------------------- 1. the happy path

test('runTask carries a clean patch to a signed PASS receipt', options, async () => {
  const { store, dir } = tempStore('pass');
  const driver = createFakeDriver({ files: BASE_FILES, exec: { [TEST_CMD]: scriptedTests } });
  const judge = createFakeProvider({ ref: 'anthropic:claude-opus', script: () => ({ json: {} }) });
  const events = [];

  const result = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js with a test', title: 'add add()' },
    repoPath: '/repo/demo',
    config: baseConfig(path.join(dir, 'code')),
    store,
    driver,
    providers: {
      orchestrator: plannerProvider(),
      judge,
      coders: [coderProvider('anthropic:claude-haiku', { 'src/math.js': GOOD_MATH, 'test/math.test.js': MATH_TEST })],
      reviewer: reviewerProvider(),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  assert.equal(result.verdict, 'PASS');
  assert.equal(result.winner, 'c0a0');
  assert.equal(result.stopReason, 'winner');
  assert.equal(result.branch, `kolm/code/${result.runId}/c0a0`);
  assert.ok(result.sha && result.sha.length >= 7, 'the winning commit has a sha');
  assert.equal(result.candidates.length, 2);
  assert.ok(result.durationMs >= 0);

  // The cheap rungs cleared everything, so the judge was never asked.
  assert.ok(result.funnel.cleared_cheaply >= 1, `cleared_cheaply was ${result.funnel.cleared_cheaply}`);
  assert.equal(result.funnel.judged, 0);
  assert.equal(judge.calls.length, 0);

  // The ledger recorded real spend and the receipt agrees with it.
  assert.ok(result.costUsd > 0, 'the cost ledger is non-empty');
  assert.equal(result.receipt.cost.total_usd, result.costUsd);
  const verified = verifyReceipt(result.receipt);
  assert.equal(verified.ok, true, verified.reason ?? 'no reason given');
  assert.ok(fs.existsSync(result.receiptPath), 'the receipt was written to disk');
  assert.equal(verifyReceipt(result.receiptPath).ok, true);
  assert.equal(result.receipt.winner.candidate_id, 'c0a0');
  assert.equal(result.receipt.mode, 'cascade');

  // Losers are cleaned up; the winner's worktree survives for the PR step.
  const destroyed = driver.destroyed.map((d) => d.candidateId);
  assert.ok(destroyed.includes('base'), 'the context worktree is destroyed');
  assert.ok(destroyed.includes('c1a0'), 'the losing worktree is destroyed');
  assert.ok(!destroyed.includes('c0a0'), 'the winning worktree is kept');

  // Events arrive in cascade order, and every one of them reached onEvent.
  const types = events.map((e) => e.type);
  assert.ok(types.includes('run.start'));
  assert.ok(types.includes('plan'));
  assert.ok(types.includes('coder.turn'));
  assert.ok(types.includes('stage.result'));
  assert.ok(types.includes('winner'));
  assert.equal(types[0], 'run.start');
  assert.equal(types[types.length - 1], 'run.end');
  assert.ok(types.indexOf('plan') < types.indexOf('coder.turn'));
  assert.ok(types.indexOf('coder.turn') < types.indexOf('stage.result'));
  assert.equal(events.length, store.eventsSince(result.runId, 0, 1000).length);

  // The run row is closed out and the eval view can read the whole thing back.
  const run = store.getRun(result.runId);
  assert.equal(run.status, 'done');
  assert.equal(run.winner_candidate, 'c0a0');
  const evaluated = evalRuns({ store, runIds: [result.runId] });
  assert.equal(evaluated.runs.length, 1);
  assert.equal(evaluated.runs[0].verdict, 'PASS');
  assert.equal(evaluated.cascade_vs_direct.cascade.pass_rate, 1);
  const table = formatEvalTable(evaluated);
  assert.ok(table.includes('CASCADE VS DIRECT'));
  assert.ok(/^[\x00-\x7F]*$/.test(table), 'the eval table is plain ASCII');
});

// ------------------------------------------------------- 2. the escalation

test('a rung that cannot repair its patch escalates to the next rung', options, async () => {
  const { store, dir } = tempStore('escalate');
  const driver = createFakeDriver({ files: BASE_FILES, exec: { [TEST_CMD]: scriptedTests } });
  const cheapRef = 'ollama:qwen3:8b';
  const strongRef = 'anthropic:claude-sonnet';
  const events = [];

  const config = { ...baseConfig(path.join(dir, 'code')), candidates: 1, repairRounds: 1 };
  const result = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js with a test', title: 'add add()' },
    repoPath: '/repo/demo',
    config,
    store,
    driver,
    providers: {
      orchestrator: plannerProvider(),
      judge: createFakeProvider({ ref: 'anthropic:claude-opus', script: () => ({ json: {} }) }),
      coders: [
        // The cheap rung writes a stub with a TODO, and the scripted test fails.
        coderProvider(cheapRef, { 'src/math.js': TODO_MATH, 'test/math.test.js': MATH_TEST }, { finishOnFirstTurn: true }),
        coderProvider(strongRef, { 'src/math.js': GOOD_MATH, 'test/math.test.js': MATH_TEST }),
      ],
      reviewer: reviewerProvider(),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  assert.equal(result.verdict, 'PASS');
  assert.equal(result.winner, 'c1a0', 'the second rung produced the winner');

  const ids = result.candidates.map((c) => c.candidateId).sort();
  assert.deepEqual(ids, ['c0a0', 'c0a1', 'c1a0'], 'ids record rung and repair round');
  const byId = new Map(result.candidates.map((c) => [c.candidateId, c]));
  assert.equal(byId.get('c0a0').verdict, 'FAIL');
  assert.equal(byId.get('c0a1').verdict, 'FAIL');
  assert.equal(byId.get('c1a0').verdict, 'PASS');
  assert.equal(byId.get('c0a0').generatorRef, cheapRef);
  assert.equal(byId.get('c1a0').generatorRef, strongRef);

  const repairEvents = events.filter((e) => e.type === 'repair');
  assert.equal(repairEvents.length, 1);
  assert.equal(repairEvents[0].payload.candidateId, 'c0a1');
  assert.equal(repairEvents[0].payload.round, 1);
  const escalations = events.filter((e) => e.type === 'escalate');
  assert.equal(escalations.length, 1);
  assert.equal(escalations[0].payload.from, cheapRef);
  assert.equal(escalations[0].payload.to, strongRef);

  assert.deepEqual(result.receipt.models.coders, [cheapRef, strongRef]);
  assert.equal(result.receipt.winner.candidate_id, 'c1a0');
  assert.equal(verifyReceipt(result.receipt).ok, true);
  // The repaired attempt shares its predecessor's worktree, which is the winner's
  // rival here, so both cheap-rung ids appear once in the destroy log at most.
  const destroyedIds = driver.destroyed.map((d) => d.candidateId);
  assert.ok(!destroyedIds.includes('c1a0'), 'the winning worktree is never destroyed');
});

// ---------------------------------------------------------- 3. a real repo

test('runTask commits a real branch through the local git driver', gitOptions, async () => {
  const { store, dir } = tempStore('localgit');
  const repo = tempDir('repo');
  const git = (...args) => execFileSync('git', args, { cwd: repo, stdio: 'pipe' });

  fs.writeFileSync(path.join(repo, 'package.json'), `${JSON.stringify({ name: 'demo', version: '1.0.0', scripts: { test: 'node --test' } }, null, 2)}\n`);
  fs.writeFileSync(path.join(repo, 'README.md'), '# demo\n');
  git('init', '-q', '-b', 'main');
  git('config', 'user.email', 'kolm@example.invalid');
  git('config', 'user.name', 'kolm code test');
  git('add', '-A');
  git('-c', 'commit.gpgsign=false', 'commit', '-q', '-m', 'initial commit');

  const driver = createDriver('local', {}, { worktreesDir: path.join(dir, 'worktrees') });
  // `node --test` on an empty repo prints a summary with zero tests, which is a
  // clean baseline; after the coder writes the file it runs for real.
  const testCommand = 'node --test test/';
  const localPlan = { ...PLAN_JSON, test_command: testCommand, forbidden_paths: ['package.json'] };

  const result = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js with a test', title: 'add add()' },
    repoPath: repo,
    baseRef: 'HEAD',
    config: { ...baseConfig(path.join(dir, 'code')), candidates: 1, repairRounds: 0 },
    store,
    driver,
    providers: {
      orchestrator: createFakeProvider({ ref: 'anthropic:claude-sonnet', script: () => ({ json: localPlan, costUsd: 0.001 }) }),
      judge: createFakeProvider({ ref: 'anthropic:claude-opus', script: () => ({ json: {} }) }),
      coders: [coderProvider('anthropic:claude-haiku', { 'src/math.js': GOOD_MATH, 'test/math.test.js': MATH_TEST })],
      reviewer: reviewerProvider(),
    },
    bands: BANDS,
    rubric: RUBRIC,
  });

  assert.equal(result.verdict, 'PASS', `stopped as ${result.verdict} (${result.stopReason})`);
  assert.equal(result.winner, 'c0a0');
  const expectedBranch = `kolm/code/${result.runId}/c0a0`;
  assert.equal(result.branch, expectedBranch);
  assert.match(result.sha, /^[0-9a-f]{40}$/);

  const branches = execFileSync('git', ['branch', '--list', expectedBranch], { cwd: repo, encoding: 'utf8' });
  assert.ok(branches.includes(expectedBranch), `git branch --list did not show ${expectedBranch}`);
  const show = execFileSync('git', ['show', '--stat', '--oneline', result.sha], { cwd: repo, encoding: 'utf8' });
  assert.ok(show.includes('src/math.js'), 'the commit contains the new file');
  assert.equal(verifyReceipt(result.receipt).ok, true);

  // Clean up the worktree the winner kept, so the temp repo can be removed.
  await driver.destroyRun(result.runId).catch(() => {});
});

// --------------------------------------------- 4. a split run, end to end

const SPLIT_BASE_KEYS = new Set(Object.keys(BASE_FILES));

const STRINGS_MODULE = [
  '/** Greet someone. */',
  'export function greet(name) {',
  '  return `hello ${name}`;',
  '}',
  '',
].join('\n');

/**
 * Scripted test output for the split run: every added file passes unless it
 * still carries conflict markers, which is what a bad merge looks like.
 */
function splitTests(cmd, wt) {
  const added = [...wt.files.keys()].filter((p) => !SPLIT_BASE_KEYS.has(p)).sort();
  const broken = added.filter((p) => String(wt.files.get(p)).includes('<<<<<<<'));
  const lines = ['ok 1 - index exports a name'];
  added.forEach((p, i) => lines.push(`${broken.includes(p) ? 'not ok' : 'ok'} ${i + 2} - ${p} works`));
  const total = 1 + added.length;
  lines.push(`# tests ${total}`, `# pass ${total - broken.length}`, `# fail ${broken.length}`);
  return { stdout: `${lines.join('\n')}\n`, exitCode: broken.length > 0 ? 1 : 0 };
}

// The whole-task summary deliberately names no file: every subtask prompt
// quotes it, so a file named there would be written by both coders.
const SPLIT_PLAN = {
  title: 'add two independent modules',
  summary: 'Add two small modules, one per part of the split.',
  acceptance: ['each module exports what its part promises'],
  files_expected: ['src/math.js', 'src/strings.js'],
  forbidden_paths: ['package.json'],
  test_command: TEST_CMD,
  risk: 'low',
  complexity: 'medium',
  budget_hint_usd: 0.5,
  clarifications: [],
  subtasks: [
    {
      id: 'math',
      title: 'add src/math.js',
      summary: 'Create src/math.js exporting add(a, b).',
      files: ['src/math.js'],
      acceptance: ['src/math.js exports add(a, b)'],
      depends_on: [],
    },
    {
      id: 'strings',
      title: 'add src/strings.js',
      summary: 'Create src/strings.js exporting greet(name).',
      files: ['src/strings.js'],
      acceptance: ['src/strings.js exports greet(name)'],
      depends_on: [],
    },
  ],
  integration_test: TEST_CMD,
};

/**
 * One coder provider for both subtasks: it reads the prompt it was handed (the
 * plan block lives in the system message, the task in the first user message)
 * and writes whichever file that part of the split owns.
 */
function splitCoder(ref = 'anthropic:claude-haiku', { costUsd = 0.001 } = {}) {
  return createFakeProvider({
    ref,
    script: (call) => {
      const messages = call.messages || [];
      if (messages.length !== 1) {
        return { toolCalls: [{ name: 'finish', input: { summary: 'done' } }], costUsd };
      }
      const opening = `${String(call.system ?? '')}\n${String(messages[0]?.content ?? '')}`;
      const writes = [];
      if (opening.includes('src/math.js')) {
        writes.push({ name: 'write_file', input: { path: 'src/math.js', content: GOOD_MATH } });
      }
      if (opening.includes('src/strings.js')) {
        writes.push({ name: 'write_file', input: { path: 'src/strings.js', content: STRINGS_MODULE } });
      }
      return {
        toolCalls: [...writes, { name: 'finish', input: { summary: 'wrote my part' } }],
        costUsd,
      };
    },
  });
}

test('a split task runs both halves and merges them into one signed receipt', options, async () => {
  const { store, dir } = tempStore('split');
  const driver = createFakeDriver({ files: BASE_FILES, exec: { [TEST_CMD]: splitTests } });
  const judge = createFakeProvider({ ref: 'anthropic:claude-opus', script: () => ({ json: {} }) });
  const events = [];

  const result = await runTask({
    task: { text: 'add a math module and a strings module', title: 'two modules' },
    repoPath: '/repo/demo',
    config: { ...baseConfig(path.join(dir, 'code')), candidates: 1, repairRounds: 0 },
    store,
    driver,
    providers: {
      orchestrator: createFakeProvider({
        ref: 'anthropic:claude-sonnet',
        script: () => ({ json: SPLIT_PLAN, costUsd: 0.003 }),
      }),
      judge,
      coders: [splitCoder()],
      reviewer: reviewerProvider(),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  assert.equal(result.verdict, 'PASS');
  assert.equal(result.winner, 'integration');
  assert.equal(result.branch, `kolm/code/${result.runId}/integration`);
  assert.ok(result.sha && result.sha.length >= 7, 'the integration commit has a sha');

  // Both halves ran, in one level, and both merged cleanly.
  const types = events.map((e) => e.type);
  assert.ok(types.includes('subtasks.plan'));
  assert.deepEqual(
    events.filter((e) => e.type === 'subtask.end').map((e) => e.payload.id),
    ['math', 'strings'],
  );
  assert.deepEqual(
    events.filter((e) => e.type === 'integration.merge').map((e) => e.payload.id),
    ['math', 'strings'],
  );
  assert.ok(events.filter((e) => e.type === 'integration.merge').every((e) => e.payload.ok));
  assert.equal(types[types.length - 1], 'run.end');
  assert.equal(events.length, store.eventsSince(result.runId, 0, 2000).length);

  // The merged tree really does hold both halves, and nothing else.
  const integration = driver.worktrees.get(`${result.runId}/integration`);
  assert.equal(integration.readFileSync('src/math.js'), GOOD_MATH);
  assert.equal(integration.readFileSync('src/strings.js'), STRINGS_MODULE);
  assert.equal(integration.readFileSync('package.json'), BASE_FILES['package.json'], 'the lockfile is untouched');

  // The receipt carries the per-subtask ledger and still verifies, from memory
  // and from disk. `subtasks` is inside the signed canonical form.
  assert.deepEqual(result.receipt.subtasks.map((s) => s.id), ['math', 'strings']);
  assert.deepEqual(result.receipt.subtasks.map((s) => s.winner), ['s0c0a0', 's1c0a0']);
  assert.ok(result.receipt.subtasks.every((s) => typeof s.sha === 'string' && s.sha.length > 0));
  assert.ok(result.receipt.subtasks.every((s) => typeof s.branch === 'string'));
  assert.equal(result.receipt.winner.candidate_id, 'integration');
  const verified = verifyReceipt(result.receipt);
  assert.equal(verified.ok, true, verified.reason ?? 'no reason given');
  assert.equal(verifyReceipt(result.receiptPath).ok, true);

  // The integration worktree is the one that survives for the PR step.
  const destroyed = driver.destroyed.map((d) => d.candidateId);
  assert.ok(!destroyed.includes('integration'), 'the merged worktree is kept');
  assert.ok(destroyed.includes('base'));

  // The run row and the eval view read a split run back like any other.
  const run = store.getRun(result.runId);
  assert.equal(run.status, 'done');
  assert.equal(run.winner_candidate, 'integration');
  assert.equal(evalRuns({ store, runIds: [result.runId] }).runs.length, 1);
});

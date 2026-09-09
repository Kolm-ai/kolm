// kolm code - the bundled benchmark (spec 5). Everything here is offline:
// the fake worktree driver, scripted fake providers, temp directories and the
// bundled fixtures. No network call, no Ollama, no real model, no git state
// outside a temp directory.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync, spawnSync } from 'node:child_process';

import {
  BENCH_RESULT_VERSION,
  BENCH_TASKS_DIR,
  SKIP_VERDICT,
  benchCountersFor,
  compareBench,
  formatBenchTable,
  hiddenCommandFor,
  listBenchTasks,
  probeRequirement,
  runBench,
  seedBenchRepo,
} from '../src/ladder/bench/index.js';
import { openStore, sqliteAvailable } from '../src/ladder/store.js';
import { createFakeDriver } from '../src/ladder/worktree/fake.js';
import { createFakeProvider } from '../src/ladder/providers/fake.js';
import { loadBands } from '../src/ladder/bands.js';
import { loadRubric } from '../src/ladder/rubric.js';

const HAS_SQLITE = sqliteAvailable();

function hasGit() {
  try {
    execFileSync('git', ['--version'], { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}
const HAS_GIT = hasGit();

const gitOptions = HAS_GIT ? {} : { skip: 'git is not on PATH' };
const runOptions = HAS_SQLITE && HAS_GIT
  ? {}
  : { skip: HAS_SQLITE ? 'git is not on PATH' : 'node:sqlite is unavailable on this Node' };

const BANDS = loadBands();
const RUBRIC = loadRubric();
const RUBRIC_IDS = RUBRIC.items.map((item) => item.id);

/** Every id the bundled fixture tree is expected to carry (spec 5.1). */
const EXPECTED_IDS = [
  'add-cli-json-flag',
  'crlf-edit',
  'fix-duration-units',
  'flaky-baseline',
  'forbidden-path-trap',
  'multi-file-rename',
  'python-fix-if-available',
  'refactor-extract-module',
  'two-subtask-feature',
  'write-tests-for-untested',
];

// ----------------------------------------------------------------- fixtures

const tempDirs = [];
const openStores = [];

function tempDir(label) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `kolm-bench-${label}-`));
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
      // a lingering handle on Windows is not a test failure
    }
  }
});

/** Read one fixture's `repo/` tree into the map the fake driver wants. */
function fixtureFiles(id) {
  const root = path.join(BENCH_TASKS_DIR, id, 'repo');
  const files = {};
  const walk = (dir, prefix) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true }).sort((a, b) => (a.name < b.name ? -1 : 1))) {
      const abs = path.join(dir, entry.name);
      const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
      if (entry.isDirectory()) walk(abs, rel);
      else files[rel] = fs.readFileSync(abs, 'utf8');
    }
  };
  walk(root, '');
  return files;
}

/**
 * Materialize an in-memory worktree into a temp directory and run a real
 * command in it. The fake driver's worktrees have no path on disk, so this is
 * how a scripted `exec` entry runs the actual node test suite offline.
 */
function execInMaterializedTree(argv, wt, label) {
  const dir = tempDir(label);
  for (const [rel, content] of wt.files) {
    const abs = path.join(dir, ...rel.split('/'));
    fs.mkdirSync(path.dirname(abs), { recursive: true });
    fs.writeFileSync(abs, content);
  }
  // NODE_TEST_CONTEXT has to go: a `node --test` child that inherits it skips
  // its files and still exits 0, which would score every run as a pass.
  const env = { ...process.env };
  delete env.NODE_TEST_CONTEXT;
  const res = spawnSync(argv[0] === 'node' ? process.execPath : argv[0], argv.slice(1), {
    cwd: dir,
    encoding: 'utf8',
    timeout: 120000,
    env,
  });
  return {
    stdout: String(res.stdout ?? ''),
    stderr: String(res.stderr ?? ''),
    exitCode: res.status === null ? 1 : res.status,
  };
}

// --------------------------------------------------------- the fixture tree

test('the fixture tree carries the ten bench tasks the spec names', async () => {
  const tasks = await listBenchTasks();
  assert.deepEqual(tasks.map((t) => t.id).sort(), EXPECTED_IDS);
  for (const task of tasks) {
    assert.ok(task.title.length > 0, `${task.id} has a title`);
    assert.ok(task.text.length > 20, `${task.id} has task text`);
    assert.ok(fs.existsSync(path.join(task.dir, 'repo')), `${task.id} has a repo/ tree`);
    assert.ok(fs.existsSync(path.join(task.dir, 'hidden')), `${task.id} has a hidden/ tree`);
    // The task text is what a model reads: it must stay printable ASCII.
    assert.ok(/^[\x20-\x7e\n\r\t]*$/.test(task.text), `${task.id} task text is ascii`);
  }
});

test('every bench repository stays small and dependency free', async () => {
  const tasks = await listBenchTasks();
  for (const task of tasks) {
    const files = [];
    const walk = (dir) => {
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        if (entry.isDirectory()) walk(path.join(dir, entry.name));
        else files.push(path.join(dir, entry.name));
      }
    };
    walk(path.join(task.dir, 'repo'));
    assert.ok(files.length < 25, `${task.id} has ${files.length} files, the cap is 25`);
    const pkg = files.find((f) => path.basename(f) === 'package.json');
    if (pkg) {
      const parsed = JSON.parse(fs.readFileSync(pkg, 'utf8'));
      assert.equal(parsed.dependencies, undefined, `${task.id} declares no dependencies`);
      assert.equal(parsed.devDependencies, undefined, `${task.id} declares no dev dependencies`);
    }
  }
});

test('listBenchTasks filters, orders and rejects an unknown id', async () => {
  const two = await listBenchTasks({ tasks: 'crlf-edit,fix-duration-units' });
  assert.deepEqual(two.map((t) => t.id), ['crlf-edit', 'fix-duration-units']);

  const reversed = await listBenchTasks({ tasks: ['fix-duration-units', 'crlf-edit'] });
  assert.deepEqual(reversed.map((t) => t.id), ['fix-duration-units', 'crlf-edit']);

  await assert.rejects(() => listBenchTasks({ tasks: 'no-such-task' }), (err) => {
    assert.equal(err.code, 'not-found');
    assert.match(err.message, /no-such-task/);
    return true;
  });
});

test('the hidden command follows the task, defaulting to node --test', async () => {
  const [js] = await listBenchTasks({ tasks: 'fix-duration-units' });
  assert.deepEqual(hiddenCommandFor(js).argv, ['node', '--test', 'tests/hidden.test.js']);
  assert.equal(hiddenCommandFor(js).display, 'node --test tests/hidden.test.js');

  const [py] = await listBenchTasks({ tasks: 'python-fix-if-available' });
  assert.deepEqual(py.requires, ['python', 'pytest']);
  assert.deepEqual(hiddenCommandFor(py).argv, ['python', '-m', 'pytest', '-q', 'tests/test_hidden.py']);
});

// ------------------------------------------------------- host requirements

test('probeRequirement answers without ever throwing', () => {
  const node = probeRequirement('node');
  assert.equal(node.ok, true);
  assert.match(node.detail, /v\d+/);

  const unknown = probeRequirement('definitely-not-a-tool');
  assert.equal(unknown.ok, false);
  assert.ok(typeof unknown.detail === 'string' && unknown.detail.length > 0);

  // The cache means a missing tool is probed once per bench, not once per task.
  const cache = new Map();
  const first = probeRequirement('node', cache);
  const second = probeRequirement('node', cache);
  assert.deepEqual(first, second);
  assert.equal(cache.size, 1);
});

// -------------------------------------------------------------- seeding

test('seedBenchRepo commits the fixture as a fresh single-commit repository', gitOptions, async () => {
  const [task] = await listBenchTasks({ tasks: 'fix-duration-units' });
  const dest = path.join(tempDir('seed'), 'repo');
  const { repoPath, baseSha } = seedBenchRepo(task, dest);

  assert.equal(repoPath, dest);
  assert.match(baseSha, /^[0-9a-f]{40}$/);
  assert.ok(fs.existsSync(path.join(repoPath, 'src', 'duration.js')));

  const log = execFileSync('git', ['log', '--oneline'], { cwd: repoPath, encoding: 'utf8' }).trim().split('\n');
  assert.equal(log.length, 1, 'exactly one commit');
  assert.match(log[0], /bench base/);
  const status = execFileSync('git', ['status', '--porcelain'], { cwd: repoPath, encoding: 'utf8' });
  assert.equal(status.trim(), '', 'the seeded tree is clean');
});

test('seeding preserves CRLF bytes exactly, whatever the host autocrlf says', gitOptions, async () => {
  const [task] = await listBenchTasks({ tasks: 'crlf-edit' });
  const dest = path.join(tempDir('crlf'), 'repo');
  seedBenchRepo(task, dest);

  for (const rel of ['src/config.js', 'test/config.test.js']) {
    const buf = fs.readFileSync(path.join(dest, ...rel.split('/')));
    assert.ok(buf.includes(0x0d), `${rel} still has CR bytes`);
    let bareLf = 0;
    for (let i = 0; i < buf.length; i += 1) {
      if (buf[i] === 0x0a && (i === 0 || buf[i - 1] !== 0x0d)) bareLf += 1;
    }
    assert.equal(bareLf, 0, `${rel} has no bare LF after seeding`);
  }
});

// ---------------------------------------------------------------- counters

test('benchCountersFor reads the orchestrator event names defensively', () => {
  const events = [
    { type: 'run.start', payload: {} },
    { type: 'candidate.start', payload: { rung: 0 } },
    { type: 'coder.turn', payload: {} },
    { type: 'coder.turn', payload: {} },
    { type: 'repair.start', payload: {} },
    { type: 'escalate', payload: { rung: 2 } },
    { type: 'subtasks.plan', payload: { subtasks: [{ id: 's1' }, { id: 's2' }] } },
    { type: 'subtask.start', payload: { id: 's1' } },
    { type: 'subtask.end', payload: { id: 's1', verdict: 'PASS', winner: 'c0a0', costUsd: 0.01 } },
  ];
  const store = {
    eventsSince: () => events,
    listCandidates: () => [{ candidate_id: 'c0a0' }, { candidate_id: 'c0a1' }],
  };
  assert.deepEqual(benchCountersFor(store, 'r1'), {
    rungReached: 3,
    candidates: 2,
    turns: 2,
    escalations: 1,
    repairs: 1,
    subtasks: 2,
  });
});

test('a run with no subtask events counts zero rather than crashing', () => {
  const store = {
    eventsSince: () => [{ type: 'run.start', payload: {} }, { type: 'run.end', payload: {} }],
    listCandidates: () => [],
  };
  assert.deepEqual(benchCountersFor(store, 'r2'), {
    rungReached: 0,
    candidates: 0,
    turns: 0,
    escalations: 0,
    repairs: 0,
    subtasks: 0,
  });
});

test('a store that throws degrades to zeroes instead of failing the bench', () => {
  const store = {
    eventsSince() {
      throw new Error('store closed');
    },
    listCandidates() {
      throw new Error('store closed');
    },
  };
  const counters = benchCountersFor(store, 'r3');
  assert.equal(counters.candidates, 0);
  assert.equal(counters.turns, 0);
});

// ------------------------------------------------- runBench with a seam

/** A stubbed runTask: no models, no worktrees, just a verdict per task id. */
function stubRunTask(verdicts) {
  return async ({ task }) => {
    const id = task.benchTask ?? task.title;
    return {
      runId: `run-${id}`,
      verdict: verdicts[id] ?? 'PASS',
      winner: null,
      sha: null,
      costUsd: 0.25,
      candidates: [],
    };
  };
}

test('runBench writes a result document and never judges what it cannot run', runOptions, async () => {
  const { store, dir } = tempStore('seam');
  const benchDir = path.join(dir, 'bench');
  const result = await runBench({
    tasks: ['fix-duration-units', 'multi-file-rename'],
    config: { benchDir, mode: 'cascade', coderLadder: ['fake:coder'], reviewerLadder: [] },
    store,
    driver: { kind: 'fake' },
    runTask: stubRunTask({}),
    workDir: path.join(dir, 'work'),
  });

  assert.equal(result.version, BENCH_RESULT_VERSION);
  assert.equal(result.results.length, 2);
  assert.deepEqual(result.results.map((r) => r.id), ['fix-duration-units', 'multi-file-rename']);
  for (const row of result.results) {
    assert.equal(row.verdict, 'PASS');
    // No driver.open and no winner sha: "not judged" is null, never false.
    assert.equal(row.hiddenPass, null);
    assert.ok(row.durationMs >= 0);
  }
  assert.equal(result.summary.tasks, 2);
  assert.equal(result.summary.pass, 2);
  assert.equal(result.summary.hiddenPass, 0);
  assert.equal(result.summary.costUsd, 0.5);

  assert.ok(result.resultPath && fs.existsSync(result.resultPath), 'the result file is on disk');
  const written = JSON.parse(fs.readFileSync(result.resultPath, 'utf8'));
  assert.equal(written.version, BENCH_RESULT_VERSION);
  assert.equal(written.results.length, 2);
  // The work directory is removed unless the caller asked to keep it.
  assert.equal(fs.existsSync(path.join(dir, 'work')), false);
});

test('a task whose requirements are missing is skipped, not failed', runOptions, async () => {
  const { store, dir } = tempStore('skip');
  const fixtures = tempDir('skip-fixtures');
  const taskDir = path.join(fixtures, 'needs-nothing-real');
  fs.mkdirSync(path.join(taskDir, 'repo'), { recursive: true });
  fs.mkdirSync(path.join(taskDir, 'hidden'), { recursive: true });
  fs.writeFileSync(path.join(taskDir, 'repo', 'README.md'), 'placeholder\n');
  fs.writeFileSync(path.join(taskDir, 'hidden', 'hidden.test.js'), '\n');
  fs.writeFileSync(
    path.join(taskDir, 'task.json'),
    `${JSON.stringify({
      id: 'needs-nothing-real',
      title: 'needs a tool that is not installed',
      text: 'This task can never run on this host, and that is the point of the test.',
      requires: ['definitely-not-a-tool'],
      expect: { hiddenTest: 'hidden.test.js' },
    }, null, 2)}\n`,
  );

  const result = await runBench({
    dir: fixtures,
    config: { benchDir: path.join(dir, 'bench') },
    store,
    driver: { kind: 'fake' },
    runTask: async () => {
      throw new Error('a skipped task must never reach runTask');
    },
    workDir: path.join(dir, 'work'),
    write: false,
  });

  assert.equal(result.results.length, 1);
  assert.equal(result.results[0].verdict, SKIP_VERDICT);
  assert.equal(result.results[0].hiddenPass, null);
  assert.ok(result.results[0].reason.length > 0, 'the skip carries a reason');
  assert.equal(result.resultPath, null);
});

// ------------------------------------------------------------- rendering

function fakeDocument(rows, extra = {}) {
  return {
    version: BENCH_RESULT_VERSION,
    startedAt: '2026-09-09T00:00:00.000Z',
    finishedAt: '2026-09-09T00:10:00.000Z',
    ladder: { coders: ['fake:coder'], reviewer: null, orchestrator: 'fake:orc', judge: 'fake:judge' },
    mode: 'cascade',
    results: rows,
    summary: {
      tasks: rows.length,
      pass: rows.filter((r) => r.verdict === 'PASS').length,
      hiddenPass: rows.filter((r) => r.hiddenPass === true).length,
      passRate: 0.5,
      costUsd: 1.5,
      durationMs: 600000,
      escalations: 1,
      repairs: 2,
    },
    ...extra,
  };
}

const ROW_A = {
  id: 'fix-duration-units', verdict: 'PASS', hiddenPass: true, costUsd: 0.5, durationMs: 1000,
  rungReached: 1, candidates: 2, turns: 3, escalations: 0, repairs: 0, subtasks: 0, runId: 'r1', reason: null,
};
const ROW_B = {
  id: 'crlf-edit', verdict: 'FAIL', hiddenPass: false, costUsd: 1, durationMs: 2000,
  rungReached: 2, candidates: 2, turns: 5, escalations: 1, repairs: 2, subtasks: 0, runId: 'r2', reason: null,
};
const ROW_SKIP = {
  id: 'python-fix-if-available', verdict: SKIP_VERDICT, hiddenPass: null, costUsd: 0, durationMs: 0,
  rungReached: 0, candidates: 0, turns: 0, escalations: 0, repairs: 0, subtasks: 0, runId: null,
  reason: 'python is not on PATH',
};

test('formatBenchTable renders printable ascii with a row per task', () => {
  const table = formatBenchTable(fakeDocument([ROW_A, ROW_B, ROW_SKIP]));
  assert.ok(/^[\x20-\x7e\n]*$/.test(table), 'no box drawing, no emoji, no smart quotes');
  assert.match(table, /TASK\s+VERDICT\s+HIDDEN/);
  assert.match(table, /fix-duration-units/);
  assert.match(table, /crlf-edit/);
  assert.match(table, /python-fix-if-available/);
  // A skipped task explains itself instead of reading as a silent zero.
  assert.match(table, /python is not on PATH/);
});

test('compareBench names what changed between two result documents', () => {
  const before = fakeDocument([ROW_A, { ...ROW_B, hiddenPass: true }]);
  const after = fakeDocument([{ ...ROW_A, hiddenPass: false }, ROW_B, ROW_SKIP]);
  const table = compareBench(before, after);
  assert.ok(/^[\x20-\x7e\n]*$/.test(table), 'the comparison stays ascii');
  assert.match(table, /HIDDEN A\s+HIDDEN B\s+CHANGE/);
  assert.match(table, /fix-duration-units\s+PASS\s+FAIL\s+broken/);
  assert.match(table, /crlf-edit\s+PASS\s+FAIL\s+broken/);
  // A task only the later run has is new, not a regression.
  assert.match(table, /python-fix-if-available\s+-\s+-\s+new/);
});

// ------------------------------------------- the whole loop, fully offline

const PLAN_FOR_DURATION = {
  title: 'fix the seconds divisor in formatDuration',
  summary: 'src/duration.js divides by 100 where it should divide by 1000.',
  acceptance: [
    'formatDuration(1500) returns "1.5s"',
    'formatDuration(90000) returns "1m 30s"',
  ],
  files_expected: ['src/duration.js'],
  forbidden_paths: ['package.json'],
  test_command: 'npm test --silent',
  risk: 'low',
  budget_hint_usd: 0.5,
  clarifications: [],
};

const GOOD_DURATION = [
  "'use strict';",
  '',
  '// Formats a millisecond duration for the run log. Sub-second durations stay',
  '// in milliseconds so short stages do not all render as "0.0s".',
  'function formatDuration(ms) {',
  '  if (!Number.isFinite(ms) || ms < 0) return "0ms";',
  '  if (ms < 1000) return Math.round(ms) + "ms";',
  '  if (ms < 60000) return (ms / 1000).toFixed(1) + "s";',
  '  const minutes = Math.floor(ms / 60000);',
  '  const seconds = Math.round((ms % 60000) / 1000);',
  '  return minutes + "m " + seconds + "s";',
  '}',
  '',
  'module.exports = { formatDuration };',
  '',
].join('\n');

// The plausible near-miss: the first divisor is fixed and the second is not,
// so the visible tests pass and only the hidden test catches it.
const HALF_FIXED_DURATION = GOOD_DURATION.replace('(ms % 60000) / 1000', '(ms % 60000) / 100');

/** A coder that writes one file on its first turn and finishes on its second. */
function coderProvider(ref, files) {
  return createFakeProvider({
    ref,
    script: (call) => {
      const first = (call.messages || []).length === 1;
      const writes = Object.entries(files).map(([p, content]) => ({
        name: 'write_file',
        input: { path: p, content },
      }));
      if (first) return { toolCalls: writes, costUsd: 0.001 };
      return { toolCalls: [{ name: 'finish', input: { summary: 'fixed the divisor' } }], costUsd: 0.001 };
    },
  });
}

function reviewerProvider() {
  return createFakeProvider({
    ref: 'anthropic:claude-haiku',
    script: () => ({
      json: {
        items: RUBRIC_IDS.map((id) => ({ id, score: 0.95, evidence: 'the diff changes only the divisor' })),
        overall: 0.95,
        confidence: 0.9,
        blocking: [],
      },
      costUsd: 0.002,
    }),
  });
}

/**
 * The visible suite the ladder runs inside the fake worktree: really executed
 * against the in-memory tree, so a broken patch really fails.
 */
function scriptedVisibleTests(cmd, wt) {
  return execInMaterializedTree(['node', '--test'], wt, 'visible');
}

/** The hidden suite, run the same way once the bench has copied hidden/ in. */
function scriptedHiddenTests(cmd, wt) {
  return execInMaterializedTree(['node', '--test', 'tests/hidden.test.js'], wt, 'hidden');
}

function benchProviders(files) {
  return {
    orchestrator: createFakeProvider({
      ref: 'anthropic:claude-sonnet',
      script: () => ({ json: PLAN_FOR_DURATION, costUsd: 0.003 }),
    }),
    judge: createFakeProvider({ ref: 'anthropic:claude-opus', script: () => ({ json: {} }) }),
    coders: [coderProvider('anthropic:claude-haiku', files)],
    reviewer: reviewerProvider(),
  };
}

function benchConfig(dir) {
  return {
    mode: 'cascade',
    candidates: 1,
    coderTurns: 4,
    repairRounds: 0,
    repairTurns: 4,
    maxCostUsd: 5,
    calibrationRate: 0,
    stageTimeoutMs: 120000,
    keepWorktrees: 'all',
    codeDir: path.join(dir, 'code'),
    benchDir: path.join(dir, 'bench'),
    coderLadder: ['anthropic:claude-haiku'],
    reviewerLadder: ['anthropic:claude-haiku'],
    orchestratorRef: 'anthropic:claude-sonnet',
    judgeRef: 'anthropic:claude-opus',
  };
}

test('a correct patch clears the hidden test end to end, with no network', runOptions, async () => {
  const { store, dir } = tempStore('good');
  const driver = createFakeDriver({
    files: fixtureFiles('fix-duration-units'),
    exec: {
      'npm test --silent': scriptedVisibleTests,
      'node --test tests/hidden.test.js': scriptedHiddenTests,
    },
  });

  const events = [];
  const result = await runBench({
    tasks: ['fix-duration-units'],
    config: benchConfig(dir),
    store,
    driver,
    providers: benchProviders({ 'src/duration.js': GOOD_DURATION }),
    bands: BANDS,
    rubric: RUBRIC,
    workDir: path.join(dir, 'work'),
    onEvent: (event) => events.push(event),
    write: false,
  });

  const row = result.results[0];
  assert.equal(row.id, 'fix-duration-units');
  assert.equal(row.hiddenPass, true, `hidden test should pass; reason: ${row.reason}`);
  assert.equal(result.summary.hiddenPass, 1);
  assert.equal(result.summary.passRate, 1);
  assert.ok(row.runId, 'the bench records the ladder run id');
  assert.ok(row.candidates >= 1, 'the counters see the candidate');
  assert.ok(events.some((e) => e.type === 'bench.task.start'));
  assert.ok(events.some((e) => e.type === 'bench.task.end'));

  // The bench ran a real ladder run: it is in the store like any other.
  assert.ok(store.getRun(row.runId), 'the run row is in the store');
});

test('a patch that only half fixes the bug fails the hidden test', runOptions, async () => {
  const { store, dir } = tempStore('half');
  const driver = createFakeDriver({
    files: fixtureFiles('fix-duration-units'),
    exec: {
      'npm test --silent': scriptedVisibleTests,
      'node --test tests/hidden.test.js': scriptedHiddenTests,
    },
  });

  const result = await runBench({
    tasks: ['fix-duration-units'],
    config: benchConfig(dir),
    store,
    driver,
    providers: benchProviders({ 'src/duration.js': HALF_FIXED_DURATION }),
    bands: BANDS,
    rubric: RUBRIC,
    workDir: path.join(dir, 'work'),
    write: false,
  });

  const row = result.results[0];
  // The visible suite is happy with the half fix; the hidden suite is not.
  // That gap is the whole reason the benchmark keeps a hidden tree.
  assert.equal(row.hiddenPass, false, `hidden test should fail; reason: ${row.reason}`);
  assert.equal(result.summary.hiddenPass, 0);
});

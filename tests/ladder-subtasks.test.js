// Owner ORCH - subtask decomposition (spec 4.4).
//
// Two halves:
//   1. the pure graph helpers (validate, topo, ids, sub-plans, conflict text,
//      resume replay) - no store, no driver, no provider;
//   2. the orchestrator path that runs a split plan over parallel worktrees and
//      merges the results into one integration branch.
//
// Everything is offline: the fake worktree driver plus scripted providers. No
// network, no real model, no repository outside a temp directory.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import {
  validateSubtasks,
  topoLevels,
  subtaskCandidateIdFor,
  parseSubtaskCandidateId,
  subPlanFor,
  scopedAcceptance,
  conflictFeedback,
  finishedSubtasks,
  priorSubtaskVerdicts,
} from '../src/ladder/subtasks.js';
import { runTask, resumeTask } from '../src/ladder/orchestrator.js';
import { openStore, sqliteAvailable } from '../src/ladder/store.js';
import { createFakeDriver, threeWayMerge } from '../src/ladder/worktree/fake.js';
import { createFakeProvider } from '../src/ladder/providers/fake.js';
import { LadderError, isLadderError } from '../src/ladder/errors.js';
import { loadBands } from '../src/ladder/bands.js';
import { loadRubric } from '../src/ladder/rubric.js';

const HAS_SQLITE = sqliteAvailable();
const options = HAS_SQLITE ? {} : { skip: 'node:sqlite is unavailable on this Node' };

const BANDS = loadBands();
const RUBRIC = loadRubric();
const RUBRIC_IDS = RUBRIC.items.map((item) => item.id);

/* -------------------------------------------------------- 1. pure helpers */

test('validateSubtasks accepts a runnable DAG and names every way it can fail', () => {
  const ok = validateSubtasks({
    subtasks: [
      { id: 'a', title: 'a', files: ['src/a.js'] },
      { id: 'b', title: 'b', files: ['src/b.js'], depends_on: ['a'] },
    ],
  });
  assert.equal(ok.ok, true);
  assert.equal(ok.reason, null);
  assert.equal(ok.subtasks.length, 2);

  // A plan with no split at all is not an error; it is the normal case.
  assert.equal(validateSubtasks({ subtasks: [] }).reason, 'no-split');
  assert.equal(validateSubtasks({ subtasks: [{ id: 'a', files: [] }] }).reason, 'no-split');
  assert.equal(validateSubtasks(null).reason, 'no-split');

  const many = { subtasks: ['a', 'b', 'c', 'd', 'e'].map((id) => ({ id, files: [] })) };
  assert.match(validateSubtasks(many).reason, /too-many-subtasks \(5 > 4\)/);

  const dup = { subtasks: [{ id: 'a', files: [] }, { id: 'a', files: [] }] };
  assert.match(validateSubtasks(dup).reason, /duplicate subtask id a/);

  const nameless = { subtasks: [{ id: '', files: [] }, { id: 'b', files: [] }] };
  assert.match(validateSubtasks(nameless).reason, /without an id/);

  const selfDep = { subtasks: [{ id: 'a', depends_on: ['a'] }, { id: 'b' }] };
  assert.match(validateSubtasks(selfDep).reason, /a depends on itself/);

  const dangling = { subtasks: [{ id: 'a', depends_on: ['zz'] }, { id: 'b' }] };
  assert.match(validateSubtasks(dangling).reason, /unknown depends_on zz in subtask a/);

  const cycle = { subtasks: [{ id: 'a', depends_on: ['b'] }, { id: 'b', depends_on: ['a'] }] };
  assert.match(validateSubtasks(cycle).reason, /cycle in subtasks/);
});

test('topoLevels groups independent subtasks and keeps plan order inside a level', () => {
  const subtasks = [
    { id: 'c', depends_on: ['a'] },
    { id: 'a' },
    { id: 'b' },
    { id: 'd', depends_on: ['b', 'c'] },
  ];
  const levels = topoLevels(subtasks).map((level) => level.map((st) => st.id));
  assert.deepEqual(levels, [['a', 'b'], ['c'], ['d']]);

  // A single level when nothing depends on anything.
  assert.deepEqual(
    topoLevels([{ id: 'x' }, { id: 'y' }]).map((l) => l.map((s) => s.id)),
    [['x', 'y']],
  );

  assert.throws(
    () => topoLevels([{ id: 'a', depends_on: ['b'] }, { id: 'b', depends_on: ['a'] }]),
    (err) => isLadderError(err, 'bad-plan') && /cycle in subtasks/.test(err.message),
  );
  assert.throws(
    () => topoLevels([{ id: 'a', depends_on: ['nope'] }]),
    (err) => isLadderError(err, 'bad-plan') && /unknown depends_on nope/.test(err.message),
  );
});

test('subtask candidate ids prefix todays ids and parse back', () => {
  assert.equal(subtaskCandidateIdFor(0, 0, 0), 's0c0a0');
  assert.equal(subtaskCandidateIdFor(1, 3, 2), 's1c3a2');
  assert.deepEqual(parseSubtaskCandidateId('s1c3a2'), { k: 1, idx: 3, attempt: 2 });
  // A single-unit id is not a subtask id, which is how callers tell them apart.
  assert.equal(parseSubtaskCandidateId('c3a2'), null);
  assert.equal(parseSubtaskCandidateId('integration'), null);
  assert.equal(parseSubtaskCandidateId(null), null);
  // Ids never collide across subtasks or across slots of one rung.
  const ids = new Set();
  for (let k = 0; k < 4; k += 1) {
    for (let idx = 0; idx < 6; idx += 1) ids.add(subtaskCandidateIdFor(k, idx, 0));
  }
  assert.equal(ids.size, 24);
});

test('subPlanFor narrows the files but inherits the guardrails', () => {
  const plan = {
    title: 'wire the thing',
    summary: 'a longer summary of the whole change',
    acceptance: ['the whole thing works'],
    files_expected: ['src/a.js', 'src/b.js'],
    forbidden_paths: ['package-lock.json'],
    test_command: 'npm test',
    subtasks: [{ id: 'a' }, { id: 'b' }],
    integration_test: 'npm run e2e',
  };
  const sub = subPlanFor(plan, { id: 'a', title: 'add a', files: ['src/a.js'] }, 0);
  assert.deepEqual(sub.files_expected, ['src/a.js']);
  assert.deepEqual(sub.forbidden_paths, ['package-lock.json']);
  assert.equal(sub.test_command, 'npm test');
  assert.deepEqual(sub.subtasks, []);
  assert.equal(sub.integration_test, null);
  assert.equal(sub.subtask_id, 'a');
  assert.match(sub.title, /subtask 1: add a/);
  assert.match(sub.summary, /WHOLE TASK: a longer summary/);
  // No acceptance of its own falls back to the plan's generic criteria.
  assert.deepEqual(sub.acceptance, ['the whole thing works']);
  const own = subPlanFor(plan, { id: 'b', files: [], acceptance: ['b works'] }, 1);
  assert.deepEqual(own.acceptance, ['b works']);

  // The planner writes summary:'' for a subtask that has none, so the headline
  // has to fall through the empty string to the title, then to the id.
  assert.match(subPlanFor(plan, { id: 'a', title: 'add a', summary: '', files: [] }, 0).summary, /^add a\n/);
  assert.match(subPlanFor(plan, { id: 'a', title: '', summary: '', files: [] }, 0).summary, /^a\n/);
});

test('conflictFeedback names the files and the markers in plain ASCII', () => {
  const text = conflictFeedback({
    conflicts: ['src/shared.js'],
    subtaskId: 'beta',
    detail: 'CONFLICT (content): Merge conflict in src/shared.js',
  });
  assert.ok(/^[\x00-\x7F]*$/.test(text), 'the feedback is plain ASCII');
  assert.match(text, /MERGE CONFLICT/);
  assert.match(text, /subtask beta/);
  assert.match(text, /src\/shared\.js/);
  assert.match(text, /<<<<<<</);
  assert.match(text, /DETAIL/);
});

test('finishedSubtasks replays only the subtasks a resume can reuse', () => {
  const events = [
    { type: 'run.start', payload: {} },
    { type: 'subtask.start', payload: { id: 'a' } },
    { type: 'subtask.end', payload: { id: 'a', verdict: 'PASS', winner: 's0c0a0', branch: 'b/a', sha: 'aaa', costUsd: 0.01 } },
    { type: 'subtask.end', payload: { id: '', verdict: 'PASS' } },
    { type: 'subtask.end', payload: { id: 'b', verdict: 'FAIL', winner: null } },
    { type: 'subtask.end', payload: { id: 'c', verdict: 'JUDGE', winner: null } },
    // A PASS with no branch has nothing for the integration merge to consume.
    { type: 'subtask.end', payload: { id: 'd', verdict: 'PASS', winner: 's3c0a0', branch: null } },
  ];
  const done = finishedSubtasks(events);
  assert.deepEqual([...done.keys()], ['a'], 'only a PASS with a branch is reusable');
  assert.equal(done.get('a').winner, 's0c0a0');
  assert.equal(done.get('a').sha, 'aaa');
  assert.equal(done.get('a').branch, 'b/a');
  assert.equal(done.get('a').costUsd, 0.01);
  assert.equal(finishedSubtasks(null).size, 0);

  // The last entry for an id wins in both directions: a retry that passes
  // becomes reusable, and one that fails again stops being so.
  const retried = finishedSubtasks([
    ...events,
    { type: 'subtask.end', payload: { id: 'b', verdict: 'PASS', winner: 's1c1a0', branch: 'b/b', sha: 'bbb' } },
    { type: 'subtask.end', payload: { id: 'a', verdict: 'FAIL', winner: null } },
  ]);
  assert.deepEqual([...retried.keys()], ['b']);

  // The verdicts themselves are still readable, which is what the retry event
  // reports as `priorVerdict`.
  const priors = priorSubtaskVerdicts(events);
  assert.deepEqual([...priors.entries()], [['a', 'PASS'], ['b', 'FAIL'], ['c', 'JUDGE'], ['d', 'PASS']]);
  assert.equal(priorSubtaskVerdicts(null).size, 0);
  assert.equal(priorSubtaskVerdicts([{ type: 'subtask.end', payload: { id: 'z' } }]).get('z'), 'FAIL');
});

test('the fake three-way merge keeps both sides and marks a real conflict', () => {
  const base = new Map([['keep.js', 'k\n']]);
  const ours = new Map([['keep.js', 'k\n'], ['a.js', 'A\n']]);
  const theirs = new Map([['keep.js', 'k\n'], ['b.js', 'B\n']]);
  const clean = threeWayMerge(base, ours, theirs, 'ref');
  assert.deepEqual(clean.conflicts, []);
  assert.equal(clean.merged.get('a.js'), 'A\n');
  assert.equal(clean.merged.get('b.js'), 'B\n');

  const both = threeWayMerge(base, new Map([['x.js', 'ours\n']]), new Map([['x.js', 'theirs\n']]), 'beta');
  assert.deepEqual(both.conflicts, ['x.js']);
  const marked = both.merged.get('x.js');
  assert.ok(marked.includes('<<<<<<< ours'));
  assert.ok(marked.includes('======='));
  assert.ok(marked.includes('>>>>>>> beta'));
});

/* ------------------------------------------------------------- fixtures */

const tempDirs = [];
const openStores = [];

function tempStore(label) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `kolm-sub-${label}-`));
  tempDirs.push(dir);
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
  'README.md': '# demo\n\nA tiny repository used by the kolm code subtask tests.\n',
  'src/index.js': "export const name = 'demo';\n",
};
const BASE_KEYS = new Set(Object.keys(BASE_FILES));
const TEST_CMD = 'npm test --silent';

/**
 * Scripted test output derived from what the worktree actually holds: one test
 * per added file, and a failure for every file that still carries a conflict
 * marker. That makes the cascade react to the merge the way a real one would.
 */
function scriptedTests(cmd, wt) {
  const added = [...wt.files.keys()].filter((p) => !BASE_KEYS.has(p)).sort();
  const broken = added.filter((p) => String(wt.files.get(p)).includes('<<<<<<<'));
  const lines = ['ok 1 - index exports a name'];
  added.forEach((p, i) => {
    lines.push(`${broken.includes(p) ? 'not ok' : 'ok'} ${i + 2} - ${p} works`);
  });
  const total = 1 + added.length;
  lines.push(`# tests ${total}`, `# pass ${total - broken.length}`, `# fail ${broken.length}`);
  return { stdout: `${lines.join('\n')}\n`, exitCode: broken.length > 0 ? 1 : 0 };
}

function fakeDriver() {
  return createFakeDriver({ files: BASE_FILES, exec: { [TEST_CMD]: scriptedTests } });
}

function planProvider(planJson, ref = 'anthropic:claude-sonnet') {
  return createFakeProvider({ ref, script: () => ({ json: planJson, costUsd: 0.003 }) });
}

function reviewerProvider(score = 0.95) {
  return createFakeProvider({
    ref: 'anthropic:claude-haiku',
    script: () => ({
      json: {
        items: RUBRIC_IDS.map((id) => ({ id, score, evidence: 'the diff matches the plan for this part' })),
        overall: score,
        confidence: 0.9,
        blocking: [],
      },
      costUsd: 0.002,
    }),
  });
}

const idleJudge = () => createFakeProvider({ ref: 'anthropic:claude-opus', script: () => ({ json: {} }) });

/**
 * A coder that writes whichever entries of `plays` its prompt names. Each play
 * is `{marker, path, text}`: the marker is the string that identifies the
 * subtask (a path from `files_expected`, or a phrase from its title), so one
 * scripted rung can play every subtask without knowing which one it is on.
 */
function fileCoder(ref, plays, { onConflict = null, costUsd = 0.001 } = {}) {
  return createFakeProvider({
    ref,
    script: (call) => {
      // The plan (title, acceptance, files_expected) lives in the system
      // prompt; the task headline lives in the first user message. A scripted
      // coder that has to tell subtasks apart needs both.
      const opening = `${String(call.system ?? '')}\n${String(call.messages?.[0]?.content ?? '')}`;
      const first = (call.messages || []).length === 1;
      if (!first) return { toolCalls: [{ name: 'finish', input: { summary: 'done' } }], costUsd: 0 };
      if (onConflict && opening.includes('MERGE CONFLICT')) {
        return {
          toolCalls: [
            { name: 'write_file', input: { path: onConflict.path, content: onConflict.content } },
            { name: 'finish', input: { summary: 'resolved the conflict' } },
          ],
          costUsd,
        };
      }
      const writes = plays
        .filter((play) => opening.includes(play.marker))
        .map((play) => ({ name: 'write_file', input: { path: play.path, content: play.text } }));
      return {
        toolCalls: [...writes, { name: 'finish', input: { summary: `wrote ${writes.length} file(s)` } }],
        costUsd,
      };
    },
  });
}

/** The two-module coder used by most scenarios below. */
const twoModuleCoder = (ref = 'ollama:qwen3:8b') => fileCoder(ref, [
  { marker: 'src/alpha.js', path: 'src/alpha.js', text: ALPHA },
  { marker: 'src/beta.js', path: 'src/beta.js', text: BETA },
]);

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
    keepWorktrees: 'losers-destroyed',
    codeDir: path.join(dir, 'code'),
    ...extra,
  };
}

const ALPHA = "/** alpha. */\nexport const alpha = 1;\n";
const BETA = "/** beta. */\nexport const beta = 2;\n";

function splitPlan(subtasks, extra = {}) {
  return {
    title: 'ship two independent modules',
    summary: 'Create two independent modules under the source directory, one per part.',
    acceptance: ['both modules exist and export a constant'],
    files_expected: ['src/alpha.js', 'src/beta.js'],
    forbidden_paths: ['package.json'],
    test_command: TEST_CMD,
    risk: 'low',
    complexity: 'medium',
    budget_hint_usd: 0.5,
    clarifications: [],
    subtasks,
    integration_test: null,
    ...extra,
  };
}

const TWO_INDEPENDENT = [
  { id: 'alpha', title: 'add src/alpha.js', files: ['src/alpha.js'], acceptance: ['src/alpha.js exports alpha'] },
  { id: 'beta', title: 'add src/beta.js', files: ['src/beta.js'], acceptance: ['src/beta.js exports beta'] },
];

const typesOf = (events) => events.map((e) => e.type);
const payloads = (events, type) => events.filter((e) => e.type === type).map((e) => e.payload);

/* ------------------------------------------- 2. the split run, end to end */

test('two independent subtasks run in parallel and merge into one integration PASS', options, async () => {
  const { store, dir } = tempStore('split');
  const driver = fakeDriver();
  const events = [];

  const result = await runTask({
    task: { text: 'add alpha and beta modules', title: 'two modules' },
    repoPath: '/repo/demo',
    config: baseConfig(dir),
    store,
    driver,
    providers: {
      orchestrator: planProvider(splitPlan(TWO_INDEPENDENT)),
      judge: idleJudge(),
      coders: [twoModuleCoder()],
      reviewer: reviewerProvider(),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  assert.equal(result.verdict, 'PASS');
  assert.equal(result.winner, 'integration');
  assert.equal(result.stopReason, 'winner');
  assert.equal(result.branch, `kolm/code/${result.runId}/integration`);

  // The plan event describes the DAG the run actually executed.
  const [plan] = payloads(events, 'subtasks.plan');
  assert.deepEqual(plan.levels, [['alpha', 'beta']]);
  assert.equal(plan.concurrency, 2);
  assert.deepEqual(plan.subtasks.map((s) => s.id), ['alpha', 'beta']);
  assert.deepEqual(plan.subtasks[0].files, ['src/alpha.js']);
  assert.deepEqual(plan.subtasks[0].dependsOn, []);

  // Both subtasks started, both finished, and their ids carry the s<k> prefix.
  assert.deepEqual(payloads(events, 'subtask.start').map((p) => p.id).sort(), ['alpha', 'beta']);
  const ends = payloads(events, 'subtask.end').sort((a, b) => a.index - b.index);
  assert.equal(ends.length, 2);
  assert.deepEqual(ends.map((e) => e.verdict), ['PASS', 'PASS']);
  assert.deepEqual(ends.map((e) => e.winner), ['s0c0a0', 's1c0a0']);
  assert.ok(ends.every((e) => typeof e.sha === 'string' && e.sha.length > 0));
  assert.ok(ends.every((e) => e.costUsd >= 0));

  // Both merges were clean, in plan order, and the whole change was judged once.
  const merges = payloads(events, 'integration.merge');
  assert.deepEqual(merges.map((m) => m.id), ['alpha', 'beta']);
  assert.ok(merges.every((m) => m.ok === true), 'both merges are clean');
  const [verdict] = payloads(events, 'integration.verdict');
  assert.equal(verdict.verdict, 'PASS');
  assert.equal(verdict.winner, 'integration');
  assert.deepEqual(verdict.subtasks, ['alpha', 'beta']);

  // The integration worktree holds BOTH halves of the change.
  const integrationWt = driver.worktrees.get(`${result.runId}/integration`);
  assert.equal(integrationWt.readFileSync('src/alpha.js'), ALPHA);
  assert.equal(integrationWt.readFileSync('src/beta.js'), BETA);

  // The result and the receipt both carry the per-subtask ledger.
  assert.deepEqual(result.subtasks.map((s) => s.id), ['alpha', 'beta']);
  assert.deepEqual(result.receipt.subtasks.map((s) => s.id), ['alpha', 'beta']);
  assert.ok(result.receipt.subtasks.every((s) => typeof s.sha === 'string'));

  // Candidate rows exist for every subtask attempt plus the integration.
  const ids = (store.listCandidates(result.runId) || []).map((c) => c.candidate_id ?? c.id);
  assert.ok(ids.includes('s0c0a0'));
  assert.ok(ids.includes('s1c0a0'));
  assert.ok(ids.includes('integration'));
  // A split run never draws a plain single-unit id.
  assert.ok(!ids.includes('c0a0'), `unexpected single-unit id in ${ids.join(',')}`);

  // Ordering: the split is planned before any subtask starts, and every merge
  // happens after the subtask it merges finished.
  const types = typesOf(events);
  assert.ok(types.indexOf('subtasks.plan') < types.indexOf('subtask.start'));
  assert.ok(types.lastIndexOf('subtask.end') < types.indexOf('integration.verdict'));
  assert.equal(types[types.length - 1], 'run.end');
});

test('a plan whose split does not validate falls back to a single-unit run', options, async () => {
  const { store, dir } = tempStore('reject');
  const driver = fakeDriver();
  const events = [];

  const bad = splitPlan([
    { id: 'alpha', title: 'add src/alpha.js', files: ['src/alpha.js'], depends_on: ['ghost'] },
    { id: 'beta', title: 'add src/beta.js', files: ['src/beta.js'] },
  ]);

  const result = await runTask({
    task: { text: 'add alpha and beta modules', title: 'two modules' },
    repoPath: '/repo/demo',
    config: baseConfig(dir),
    store,
    driver,
    providers: {
      orchestrator: planProvider(bad),
      judge: idleJudge(),
      coders: [twoModuleCoder()],
      reviewer: reviewerProvider(),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  const [rejected] = payloads(events, 'subtasks.rejected');
  assert.match(rejected.reason, /unknown depends_on ghost/);
  assert.equal(rejected.count, 2);
  assert.equal(payloads(events, 'subtask.start').length, 0);
  assert.equal(payloads(events, 'subtasks.plan').length, 0);
  // The fallback is today's ladder, with today's ids.
  assert.equal(result.verdict, 'PASS');
  assert.equal(result.winner, 'c0a0');
  assert.deepEqual(result.subtasks, []);
  assert.deepEqual(result.receipt.subtasks, []);
});

test('a driver that cannot merge is a reason to stay on the single-unit ladder', options, async () => {
  const { store, dir } = tempStore('nomerge');
  const driver = fakeDriver();
  delete driver.merge;
  const events = [];

  const result = await runTask({
    task: { text: 'add alpha and beta modules', title: 'two modules' },
    repoPath: '/repo/demo',
    config: baseConfig(dir),
    store,
    driver,
    providers: {
      orchestrator: planProvider(splitPlan(TWO_INDEPENDENT)),
      judge: idleJudge(),
      coders: [twoModuleCoder()],
      reviewer: reviewerProvider(),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  const [rejected] = payloads(events, 'subtasks.rejected');
  assert.match(rejected.reason, /driver fake cannot merge/);
  assert.equal(payloads(events, 'subtask.start').length, 0);
  assert.equal(result.winner, 'c0a0');
});

test('subtasks=false keeps the single-unit ladder without even validating', options, async () => {
  const { store, dir } = tempStore('off');
  const driver = fakeDriver();
  const events = [];

  const result = await runTask({
    task: { text: 'add alpha and beta modules', title: 'two modules' },
    repoPath: '/repo/demo',
    config: baseConfig(dir, { subtasks: false }),
    store,
    driver,
    providers: {
      orchestrator: planProvider(splitPlan(TWO_INDEPENDENT)),
      judge: idleJudge(),
      coders: [twoModuleCoder()],
      reviewer: reviewerProvider(),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  assert.equal(payloads(events, 'subtasks.rejected').length, 0);
  assert.equal(payloads(events, 'subtasks.plan').length, 0);
  assert.equal(result.winner, 'c0a0');
});

test('a merge conflict is repaired in the integration worktree', options, async () => {
  const { store, dir } = tempStore('conflict');
  const driver = fakeDriver();
  const events = [];

  // Both subtasks own the same file, so the second merge cannot be clean.
  const shared = splitPlan(
    [
      { id: 'alpha', title: 'shared, the alpha half', files: ['src/shared.js'] },
      { id: 'beta', title: 'shared, the beta half', files: ['src/shared.js'] },
    ],
    { files_expected: ['src/shared.js'] },
  );
  const RESOLVED = "export const alpha = 1;\nexport const beta = 2;\n";
  const coder = fileCoder(
    'ollama:qwen3:8b',
    [
      { marker: 'alpha half', path: 'src/shared.js', text: 'export const alpha = 1;\n' },
      { marker: 'beta half', path: 'src/shared.js', text: 'export const beta = 2;\n' },
    ],
    { onConflict: { path: 'src/shared.js', content: RESOLVED } },
  );

  // What the integration worktree had committed at the moment the repaired
  // merge was announced: the merge has to be concluded THERE, not at the end
  // of the level, or the next merge of the level starts mid-merge.
  const committedAtRepair = [];

  const result = await runTask({
    task: { text: 'both halves of the shared module', title: 'shared' },
    repoPath: '/repo/demo',
    config: baseConfig(dir),
    store,
    driver,
    providers: {
      orchestrator: planProvider(shared),
      judge: idleJudge(),
      coders: [coder],
      reviewer: reviewerProvider(),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => {
      events.push(evt);
      if (evt.type === 'integration.merge' && evt.payload?.repaired === true) {
        committedAtRepair.push(driver.worktrees.get(`${evt.run_id}/integration`)?.lastCommitMessage ?? null);
      }
    },
  });

  const merges = payloads(events, 'integration.merge');
  assert.equal(merges.length, 3, `expected clean + conflict + repaired, got ${merges.length}`);
  assert.deepEqual(merges[0], { id: 'alpha', ok: true, conflicts: [] });
  assert.equal(merges[1].id, 'beta');
  assert.equal(merges[1].ok, false);
  assert.deepEqual(merges[1].conflicts, ['src/shared.js']);
  assert.match(merges[1].detail, /src\/shared\.js/, 'the failed merge carries what the driver said');
  assert.equal(merges[2].id, 'beta');
  assert.equal(merges[2].ok, true);
  assert.equal(merges[2].repaired, true);

  // The resolution was committed right after the repair, and the branch is
  // really in the integration history.
  assert.equal(committedAtRepair.length, 1);
  assert.match(committedAtRepair[0], /merge beta/, `expected a commit for beta, got ${committedAtRepair[0]}`);
  assert.match(committedAtRepair[0], /conflict resolved/);
  const integrationWt = driver.worktrees.get(`${result.runId}/integration`);
  const betaBranch = payloads(events, 'subtask.end').find((s) => s.id === 'beta').branch;
  assert.equal(await integrationWt.isAncestor(betaBranch), true);

  // The repaired tree carries both halves and no marker survives.
  assert.equal(integrationWt.readFileSync('src/shared.js'), RESOLVED);
  assert.equal(result.verdict, 'PASS');
  assert.equal(result.winner, 'integration');
});

test('a merge that fails with no conflicted file stops the run instead of pretending', options, async () => {
  const { store, dir } = tempStore('stuck');
  const driver = fakeDriver();
  const events = [];

  // The live failure (run r_20260909_6a5142fd): git refused to start the merge
  // because the previous one was never concluded. No file conflicted, so there
  // is nothing a coder could resolve.
  const STUCK = 'fatal: You have not concluded your merge (MERGE_HEAD exists).';
  const realMerge = driver.merge;
  driver.merge = async (wt, ref, opts) => (
    String(ref).endsWith('s1c0a0')
      ? { ok: false, conflicts: [], detail: STUCK }
      : realMerge(wt, ref, opts)
  );

  const result = await runTask({
    task: { text: 'add alpha and beta modules', title: 'two modules' },
    repoPath: '/repo/demo',
    config: baseConfig(dir),
    store,
    driver,
    providers: {
      orchestrator: planProvider(splitPlan(TWO_INDEPENDENT)),
      judge: idleJudge(),
      coders: [twoModuleCoder()],
      reviewer: reviewerProvider(),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  const merges = payloads(events, 'integration.merge');
  assert.equal(merges.length, 2, 'the failure is reported once, with no repair round');
  assert.deepEqual(merges[0], { id: 'alpha', ok: true, conflicts: [] });
  assert.deepEqual(merges[1], { id: 'beta', ok: false, conflicts: [], detail: STUCK });

  // No repair session was started: a coder cannot conclude a merge.
  const integrationTurns = payloads(events, 'coder.turn').filter((p) => p.candidateId === 'integration');
  assert.deepEqual(integrationTurns, [], 'nothing was asked of the coder');

  // The run stops and says why, rather than judging a change that never merged.
  assert.equal(result.stopReason, 'integration-conflict');
  assert.notEqual(result.verdict, 'PASS');
  const [verdict] = payloads(events, 'integration.verdict');
  assert.equal(verdict.verdict, 'FAIL');
  assert.equal(verdict.reasons.length, 1);
  assert.match(verdict.reasons[0], /merge of beta failed/);
  assert.ok(verdict.reasons[0].includes(STUCK), `expected the detail in ${verdict.reasons[0]}`);
});

test('a repair that leaves no markers but no merge either is not accepted', options, async () => {
  const { store, dir } = tempStore('nohistory');
  const driver = fakeDriver();
  const events = [];

  // The tree looks resolved (the coder kept HEAD's side) but the branch never
  // landed. Only the history can tell, so the integration worktree is asked.
  const realCreate = driver.create;
  driver.create = async (opts = {}) => {
    const wt = await realCreate(opts);
    if (opts.candidateId === 'integration') wt.isAncestor = async () => false;
    return wt;
  };

  const shared = splitPlan(
    [
      { id: 'alpha', title: 'shared, the alpha half', files: ['src/shared.js'] },
      { id: 'beta', title: 'shared, the beta half', files: ['src/shared.js'] },
    ],
    { files_expected: ['src/shared.js'] },
  );
  const coder = fileCoder(
    'ollama:qwen3:8b',
    [
      { marker: 'alpha half', path: 'src/shared.js', text: 'export const alpha = 1;\n' },
      { marker: 'beta half', path: 'src/shared.js', text: 'export const beta = 2;\n' },
    ],
    { onConflict: { path: 'src/shared.js', content: 'export const alpha = 1;\n' } },
  );

  const result = await runTask({
    task: { text: 'both halves of the shared module', title: 'shared' },
    repoPath: '/repo/demo',
    config: baseConfig(dir),
    store,
    driver,
    providers: {
      orchestrator: planProvider(shared),
      judge: idleJudge(),
      coders: [coder],
      reviewer: reviewerProvider(),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  const merges = payloads(events, 'integration.merge');
  assert.equal(merges.length, 3);
  assert.equal(merges[2].id, 'beta');
  assert.equal(merges[2].ok, false, 'a clean tree is not proof the branch merged');
  assert.equal(merges[2].repaired, true);
  assert.match(merges[2].detail, /is not in the integration history after repair/);

  assert.equal(result.stopReason, 'integration-conflict');
  assert.notEqual(result.verdict, 'PASS');
  const [verdict] = payloads(events, 'integration.verdict');
  assert.equal(verdict.verdict, 'FAIL');
  assert.match(verdict.reasons[0], /merge of beta failed: branch .* is not in the integration history/);
});

test('a resumed split run skips the subtask it already finished', options, async () => {
  const { store, dir } = tempStore('resume');
  const driver = fakeDriver();
  const config = baseConfig(dir, { keepWorktrees: 'all' });

  // beta depends on alpha, so they land in separate levels and the failure is
  // deterministic: alpha completes, then beta's only rung dies.
  const chained = splitPlan([
    { id: 'alpha', title: 'add src/alpha.js', files: ['src/alpha.js'] },
    { id: 'beta', title: 'add src/beta.js', files: ['src/beta.js'], depends_on: ['alpha'] },
  ]);

  const deadOnBeta = createFakeProvider({
    ref: 'ollama:qwen3:8b',
    script: (call) => {
      const opening = `${String(call.system ?? '')}\n${String(call.messages?.[0]?.content ?? '')}`;
      if (opening.includes('src/beta.js')) {
        return { error: new LadderError('provider-bad-request', 'scripted rung failure') };
      }
      const first = (call.messages || []).length === 1;
      if (!first) return { toolCalls: [{ name: 'finish', input: { summary: 'done' } }], costUsd: 0 };
      return {
        toolCalls: [
          { name: 'write_file', input: { path: 'src/alpha.js', content: ALPHA } },
          { name: 'finish', input: { summary: 'wrote alpha' } },
        ],
        costUsd: 0.001,
      };
    },
  });

  const first = [];
  await assert.rejects(
    () => runTask({
      task: { text: 'add alpha then beta', title: 'chained' },
      repoPath: '/repo/demo',
      config,
      store,
      driver,
      providers: {
        orchestrator: planProvider(chained),
        judge: idleJudge(),
        coders: [deadOnBeta],
        reviewer: reviewerProvider(),
      },
      bands: BANDS,
      rubric: RUBRIC,
      onEvent: (evt) => first.push(evt),
    }),
    (err) => isLadderError(err, 'provider-bad-request'),
  );

  const runId = first[0].run_id;
  const firstEnds = payloads(first, 'subtask.end').map((p) => p.id);
  assert.deepEqual(firstEnds, ['alpha'], 'alpha finished before beta took the run down');

  // Resume with a rung that can write beta. alpha must not be coded again.
  const second = [];
  const result = await resumeTask({
    runId,
    store,
    driver,
    repoPath: '/repo/demo',
    config,
    providers: {
      orchestrator: planProvider(chained),
      judge: idleJudge(),
      coders: [twoModuleCoder()],
      reviewer: reviewerProvider(),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => second.push(evt),
  });

  assert.deepEqual(payloads(second, 'subtask.start').map((p) => p.id), ['beta']);
  assert.deepEqual(payloads(second, 'subtask.end').map((p) => p.id), ['beta']);
  assert.deepEqual(payloads(second, 'integration.merge').map((m) => m.id), ['alpha', 'beta']);
  assert.equal(result.verdict, 'PASS');
  assert.equal(result.winner, 'integration');
  assert.deepEqual(result.subtasks.map((s) => s.id), ['alpha', 'beta']);
  const integrationWt = driver.worktrees.get(`${runId}/integration`);
  assert.equal(integrationWt.readFileSync('src/alpha.js'), ALPHA);
  assert.equal(integrationWt.readFileSync('src/beta.js'), BETA);
});

test('a resumed split run cleans up the worktrees of the subtask it carried over', options, async () => {
  // Found live (r_20260909_7159a980): the killed run never reached its hygiene
  // step, and the resume only destroyed what it had opened itself, so the
  // carried-over subtask's losers stayed on disk. A kill is modelled by a first
  // run that keeps everything and dies on beta.
  const { store, dir } = tempStore('resume-hygiene');
  const driver = fakeDriver();
  const chained = splitPlan([
    { id: 'alpha', title: 'add src/alpha.js', files: ['src/alpha.js'] },
    { id: 'beta', title: 'add src/beta.js', files: ['src/beta.js'], depends_on: ['alpha'] },
  ]);
  const deadOnBeta = createFakeProvider({
    ref: 'ollama:qwen3:8b',
    script: (call) => {
      const opening = `${String(call.system ?? '')}\n${String(call.messages?.[0]?.content ?? '')}`;
      if (opening.includes('src/beta.js')) {
        return { error: new LadderError('provider-bad-request', 'scripted rung failure') };
      }
      const first = (call.messages || []).length === 1;
      if (!first) return { toolCalls: [{ name: 'finish', input: { summary: 'done' } }], costUsd: 0 };
      return {
        toolCalls: [
          { name: 'write_file', input: { path: 'src/alpha.js', content: ALPHA } },
          { name: 'finish', input: { summary: 'wrote alpha' } },
        ],
        costUsd: 0.001,
      };
    },
  });
  const first = [];
  await assert.rejects(
    () => runTask({
      task: { text: 'add alpha then beta', title: 'chained' },
      repoPath: '/repo/demo',
      config: baseConfig(dir, { keepWorktrees: 'all' }),
      store,
      driver,
      providers: {
        orchestrator: planProvider(chained),
        judge: idleJudge(),
        coders: [deadOnBeta],
        reviewer: reviewerProvider(),
      },
      bands: BANDS,
      rubric: RUBRIC,
      onEvent: (evt) => first.push(evt),
    }),
    (err) => isLadderError(err, 'provider-bad-request'),
  );
  const runId = first[0].run_id;
  assert.equal(driver.destroyed.length, 0, 'the killed run destroyed nothing');
  const alphaIds = store.listCandidates(runId).map((c) => c.candidate_id).filter((id) => id.startsWith('s0c')).sort();
  assert.deepEqual(alphaIds, ['s0c0a0']);

  const result = await resumeTask({
    runId,
    store,
    driver,
    repoPath: '/repo/demo',
    config: baseConfig(dir),
    providers: {
      orchestrator: planProvider(chained),
      judge: idleJudge(),
      coders: [twoModuleCoder()],
      reviewer: reviewerProvider(),
    },
    bands: BANDS,
    rubric: RUBRIC,
  });
  assert.equal(result.verdict, 'PASS');
  assert.equal(result.winner, 'integration');
  // alpha was carried over without a subtask.start, and its worktree
  // was still removed on the way out; the integration winner is the only one kept.
  const destroyed = driver.destroyed.map((d) => d.candidateId).sort();
  for (const id of alphaIds) assert.ok(destroyed.includes(id), `${id} was adopted and destroyed`);
  assert.ok(!destroyed.includes('integration'));
  for (const id of alphaIds) assert.equal(store.getCandidate(runId, id).status, 'destroyed');
});

// ---------------------------------------------------------------------------
// failure policy (defects found live in r_20260909_f02fbbb9: both subtasks
// FAILED, nothing merged, the cascade graded the untouched base, and a resume
// would have treated both FAIL entries as finished work)
// ---------------------------------------------------------------------------

/** A file the scripted test command reports as failing, so its subtask FAILs. */
const ALPHA_BROKEN = '<<<<<<< ours\nexport const alpha = 1;\n=======\n>>>>>>> theirs\n';
const GAMMA = "/** gamma. */\nexport const gamma = 3;\n";

/** A coder that writes a broken src/alpha.js and correct beta and gamma. */
const weakCoder = (ref = 'ollama:qwen3:8b') => fileCoder(ref, [
  { marker: 'src/alpha.js', path: 'src/alpha.js', text: ALPHA_BROKEN },
  { marker: 'src/beta.js', path: 'src/beta.js', text: BETA },
  { marker: 'src/gamma.js', path: 'src/gamma.js', text: GAMMA },
]);

/**
 * Which subtasks a scripted coder was actually asked to write, one entry per
 * session it opened, read back from the calls the fake provider recorded. A
 * resumed rung that reuses its cached diff never opens a session at all.
 */
function coderSessions(provider, ids) {
  const seen = [];
  for (const call of provider.calls ?? []) {
    if ((call.messages || []).length !== 1) continue;
    const opening = `${String(call.system ?? '')}\n${String(call.messages[0]?.content ?? '')}`;
    for (const id of ids) if (opening.includes(`src/${id}.js`)) seen.push(id);
  }
  return seen;
}

test('a failed subtask skips its dependents and the integration is not judged', options, async () => {
  const { store, dir } = tempStore('skipped');
  const driver = fakeDriver();
  const events = [];

  // alpha fails every attempt, beta waits on alpha, gamma is independent.
  const plan = splitPlan(
    [
      { id: 'alpha', title: 'add src/alpha.js', files: ['src/alpha.js'] },
      { id: 'beta', title: 'add src/beta.js', files: ['src/beta.js'], depends_on: ['alpha'] },
      { id: 'gamma', title: 'add src/gamma.js', files: ['src/gamma.js'] },
    ],
    { files_expected: ['src/alpha.js', 'src/beta.js', 'src/gamma.js'] },
  );

  const result = await runTask({
    task: { text: 'add alpha, beta and gamma modules', title: 'three modules' },
    repoPath: '/repo/demo',
    // 'all' so the run leaves no candidate row behind on its way out: the
    // assertion below is that NOTHING ever drew an `integration` candidate.
    config: baseConfig(dir, { keepWorktrees: 'all' }),
    store,
    driver,
    providers: {
      orchestrator: planProvider(plan),
      judge: idleJudge(),
      coders: [weakCoder()],
      reviewer: reviewerProvider(),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  // alpha ran and failed; gamma ran and passed; beta never started at all.
  const ends = payloads(events, 'subtask.end');
  assert.deepEqual(ends.find((e) => e.id === 'alpha').verdict, 'FAIL');
  assert.deepEqual(ends.find((e) => e.id === 'gamma').verdict, 'PASS');
  assert.equal(ends.find((e) => e.id === 'beta'), undefined, 'a skipped subtask has no end');
  assert.deepEqual(payloads(events, 'subtask.start').map((p) => p.id).sort(), ['alpha', 'gamma']);

  const [skipped] = payloads(events, 'subtask.skipped');
  assert.equal(skipped.id, 'beta');
  assert.equal(skipped.index, 1);
  assert.deepEqual(skipped.because, ['alpha']);

  // Only the survivor merged, so the integration branch holds the partial work
  // a later resume builds on.
  assert.deepEqual(payloads(events, 'integration.merge').map((m) => m.id), ['gamma']);
  const integrationWt = driver.worktrees.get(`${result.runId}/integration`);
  assert.equal(integrationWt.readFileSync('src/gamma.js'), GAMMA);
  assert.equal(integrationWt.files.has('src/alpha.js'), false);

  // The verdict names every missing subtask instead of grading a partial diff.
  const [verdict] = payloads(events, 'integration.verdict');
  assert.equal(verdict.verdict, 'FAIL');
  assert.equal(verdict.winner, null);
  assert.deepEqual(verdict.reasons, [
    'subtask alpha failed',
    'subtask beta skipped (depends on alpha)',
  ]);
  assert.ok(verdict.reasons.every((r) => /^[\x20-\x7e]*$/.test(r)), 'the reasons are plain ASCII');
  assert.deepEqual(verdict.subtasks.slice().sort(), ['alpha', 'beta', 'gamma']);

  // No integration candidate was drawn and no stage ran on it.
  const ids = (store.listCandidates(result.runId) || []).map((c) => c.candidate_id ?? c.id);
  assert.ok(!ids.includes('integration'), `unexpected integration candidate in ${ids.join(',')}`);
  assert.deepEqual(store.latestResults(result.runId, 'integration'), {});

  assert.equal(result.verdict, 'FAIL');
  assert.equal(result.stopReason, 'subtask-failed');
  assert.equal(result.winner, null);

  // The ledger and the receipt both carry the skipped subtask.
  const ledger = new Map(result.subtasks.map((s) => [s.id, s]));
  assert.deepEqual([...ledger.keys()].sort(), ['alpha', 'beta', 'gamma']);
  assert.equal(ledger.get('beta').verdict, 'SKIPPED');
  assert.equal(ledger.get('beta').branch, null);
  assert.equal(ledger.get('beta').costUsd, 0);
  assert.deepEqual(result.receipt.subtasks.map((s) => s.id).sort(), ['alpha', 'beta', 'gamma']);
  assert.equal(result.receipt.verdict, 'FAIL');
  assert.equal(result.receipt.stop_reason, 'subtask-failed');
});

test('a resumed split run retries the subtask that failed', options, async () => {
  const { store, dir } = tempStore('retry');
  const driver = fakeDriver();
  // 'all' keeps the failed attempt's worktree, and one subtask at a time makes
  // the order deterministic: alpha finishes FAIL before beta takes the run down.
  const config = baseConfig(dir, { keepWorktrees: 'all', subtaskConcurrency: 1 });

  const plan = splitPlan([
    { id: 'alpha', title: 'add src/alpha.js', files: ['src/alpha.js'] },
    { id: 'beta', title: 'add src/beta.js', files: ['src/beta.js'] },
  ]);

  // Run 1: alpha's only rung writes a diff that fails the tests, then beta's
  // rung dies, so the run is interrupted rather than finished (a finished run
  // has a receipt and resumes to 'already-finished').
  const deadOnBeta = createFakeProvider({
    ref: 'ollama:qwen3:8b',
    script: (call) => {
      const opening = `${String(call.system ?? '')}\n${String(call.messages?.[0]?.content ?? '')}`;
      if (opening.includes('src/beta.js')) {
        return { error: new LadderError('provider-bad-request', 'scripted rung failure') };
      }
      const first = (call.messages || []).length === 1;
      if (!first) return { toolCalls: [{ name: 'finish', input: { summary: 'done' } }], costUsd: 0 };
      return {
        toolCalls: [
          { name: 'write_file', input: { path: 'src/alpha.js', content: ALPHA_BROKEN } },
          { name: 'finish', input: { summary: 'wrote alpha' } },
        ],
        costUsd: 0.001,
      };
    },
  });

  const first = [];
  await assert.rejects(
    () => runTask({
      task: { text: 'add alpha and beta', title: 'two modules' },
      repoPath: '/repo/demo',
      config,
      store,
      driver,
      providers: {
        orchestrator: planProvider(plan),
        judge: idleJudge(),
        coders: [deadOnBeta],
        reviewer: reviewerProvider(),
      },
      bands: BANDS,
      rubric: RUBRIC,
      onEvent: (evt) => first.push(evt),
    }),
    (err) => isLadderError(err, 'provider-bad-request'),
  );

  const runId = first[0].run_id;
  const firstEnds = payloads(first, 'subtask.end');
  assert.deepEqual(firstEnds.map((p) => p.id), ['alpha']);
  assert.equal(firstEnds[0].verdict, 'FAIL');
  assert.equal(firstEnds[0].branch, null);

  // Resume with the failing rung still first and a stronger rung behind it -
  // exactly what `resume --coder <stronger>` buys.
  const weak = weakCoder();
  const strong = fileCoder('anthropic:claude-opus', [
    { marker: 'src/alpha.js', path: 'src/alpha.js', text: ALPHA },
    { marker: 'src/beta.js', path: 'src/beta.js', text: BETA },
  ]);
  const second = [];
  const result = await resumeTask({
    runId,
    store,
    driver,
    repoPath: '/repo/demo',
    config,
    providers: {
      orchestrator: planProvider(plan),
      judge: idleJudge(),
      coders: [weak, strong],
      reviewer: reviewerProvider(),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => second.push(evt),
  });

  // alpha is retried, and the retry is announced before its ladder restarts.
  const [retry] = payloads(second, 'subtask.retry');
  assert.deepEqual(retry, { id: 'alpha', index: 0, priorVerdict: 'FAIL' });
  const order = typesOf(second);
  assert.ok(order.indexOf('subtask.retry') < order.indexOf('subtask.start'));
  assert.deepEqual(payloads(second, 'subtask.start').map((p) => p.id), ['alpha', 'beta']);
  assert.deepEqual(payloads(second, 'subtask.end').map((p) => p.id), ['alpha', 'beta']);
  assert.deepEqual(payloads(second, 'subtask.end').map((p) => p.verdict), ['PASS', 'PASS']);
  // beta never ended in the first run, so it is fresh work, not a retry.
  assert.equal(payloads(second, 'subtask.retry').length, 1);

  // The cached rung was reused rather than paid for again: the weak coder was
  // never asked to write alpha, only beta, because s0c0a0 already had a diff.
  assert.deepEqual(coderSessions(weak, ['alpha', 'beta']), ['beta']);
  assert.deepEqual(coderSessions(strong, ['alpha', 'beta']), ['alpha']);

  assert.equal(result.verdict, 'PASS');
  assert.equal(result.winner, 'integration');
  assert.deepEqual(payloads(second, 'integration.merge').map((m) => m.id), ['alpha', 'beta']);
  const integrationWt = driver.worktrees.get(`${runId}/integration`);
  assert.equal(integrationWt.readFileSync('src/alpha.js'), ALPHA);
  assert.equal(integrationWt.readFileSync('src/beta.js'), BETA);
});

test('a split run that ended FAIL normally is reopened and the failed subtask is retried', options, async () => {
  const { store, dir } = tempStore('reopen-split');
  const driver = fakeDriver();
  // The live shape (r_20260909_f02fbbb9): the run ends FAIL on its own terms and
  // the losers' worktrees go with it, so the reopen has nothing cached to open
  // and must not try. One subtask at a time keeps the order deterministic.
  const config = baseConfig(dir, { subtaskConcurrency: 1 });
  const plan = splitPlan([
    { id: 'alpha', title: 'add src/alpha.js', files: ['src/alpha.js'] },
    { id: 'beta', title: 'add src/beta.js', files: ['src/beta.js'], depends_on: ['alpha'] },
  ]);

  // Run 1: alpha's only rung writes a file the tests reject, so alpha FAILs and
  // beta - which depends on it - is skipped. The run finishes normally.
  const weak = weakCoder();
  const firstEvents = [];
  const first = await runTask({
    task: { text: 'add alpha and beta modules', title: 'two modules' },
    repoPath: '/repo/demo',
    config,
    store,
    driver,
    providers: {
      orchestrator: planProvider(plan),
      judge: idleJudge(),
      coders: [weak],
      reviewer: reviewerProvider(),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => firstEvents.push(evt),
  });

  const runId = first.runId;
  assert.equal(first.verdict, 'FAIL');
  assert.equal(first.stopReason, 'subtask-failed');
  assert.equal(payloads(firstEvents, 'subtask.end').find((p) => p.id === 'alpha').verdict, 'FAIL');
  assert.deepEqual(payloads(firstEvents, 'subtask.skipped').map((p) => p.id), ['beta']);
  assert.equal(store.getRun(runId).status, 'done', 'the run finished, receipt and all');
  assert.equal(store.getRun(runId).receipt.verdict, 'FAIL');
  // Only alpha ever drew a candidate; the integration worktree the hygiene
  // step destroyed on the way out was never a graded candidate and leaves no
  // row behind.
  assert.deepEqual(
    store.listCandidates(runId).map((c) => c.candidate_id).sort(),
    ['s0c0a0'],
  );
  const weakCalls = weak.calls.length;

  // Resume with a stronger coder and nothing else: the ladder on the command
  // line takes over above the rung already spent.
  const strong = fileCoder('anthropic:claude-opus', [
    { marker: 'src/alpha.js', path: 'src/alpha.js', text: ALPHA },
    { marker: 'src/beta.js', path: 'src/beta.js', text: BETA },
  ]);
  const second = [];
  const result = await resumeTask({
    runId,
    store,
    driver,
    repoPath: '/repo/demo',
    config,
    providers: {
      orchestrator: planProvider(plan),
      judge: idleJudge(),
      coders: [strong],
      reviewer: reviewerProvider(),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => second.push(evt),
  });

  const [resumed] = payloads(second, 'run.resume');
  assert.equal(resumed.reopened, true);
  assert.equal(resumed.priorVerdict, 'FAIL');
  assert.equal(resumed.priorStopReason, 'subtask-failed');

  // alpha is retried; beta never ended, so it is fresh work rather than a retry.
  assert.deepEqual(payloads(second, 'subtask.retry'), [{ id: 'alpha', index: 0, priorVerdict: 'FAIL' }]);
  const order = typesOf(second);
  assert.ok(order.indexOf('subtask.retry') < order.indexOf('subtask.start'));
  assert.deepEqual(payloads(second, 'subtask.start').map((p) => p.id), ['alpha', 'beta']);
  assert.deepEqual(payloads(second, 'subtask.end').map((p) => p.verdict), ['PASS', 'PASS']);

  // alpha's new candidates carry its own offset; beta, which spent nothing, does
  // not inherit it.
  const starts = payloads(second, 'candidate.start');
  assert.deepEqual(starts.map((p) => p.candidateId), ['s0c1a0', 's1c0a0']);
  assert.deepEqual(starts.map((p) => p.rung), [1, 0]);
  assert.ok(starts.every((p) => p.generatorRef === 'anthropic:claude-opus'));
  assert.equal(weak.calls.length, weakCalls, 'the weak coder was never called again');
  assert.deepEqual(coderSessions(strong, ['alpha', 'beta']), ['alpha', 'beta']);

  // The reopened run integrates and passes, and its receipt carries both
  // attempts' candidates.
  assert.deepEqual(payloads(second, 'integration.merge').map((m) => m.id), ['alpha', 'beta']);
  assert.equal(payloads(second, 'integration.verdict')[0].verdict, 'PASS');
  assert.equal(result.verdict, 'PASS');
  assert.equal(result.winner, 'integration');
  assert.deepEqual(
    result.receipt.candidates.map((c) => c.candidate_id).sort(),
    ['integration', 's0c0a0', 's0c1a0', 's1c0a0'],
  );
  assert.equal(store.getRun(runId).status, 'done');
  assert.equal(store.getRun(runId).receipt.verdict, 'PASS');
});

test('an exhausted budget stops a split run before the first subtask', options, async () => {
  const { store, dir } = tempStore('budget');
  const driver = fakeDriver();
  const events = [];

  // The plan alone costs more than the whole run is allowed to spend.
  const result = await runTask({
    task: { text: 'add alpha and beta modules', title: 'two modules' },
    repoPath: '/repo/demo',
    config: baseConfig(dir, { maxCostUsd: 0.001 }),
    store,
    driver,
    providers: {
      orchestrator: planProvider(splitPlan(TWO_INDEPENDENT)),
      judge: idleJudge(),
      coders: [twoModuleCoder()],
      reviewer: reviewerProvider(),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  const budget = payloads(events, 'run.budget');
  assert.ok(budget.some((b) => b.where === 'subtask'), `expected a subtask budget stop, got ${budget.map((b) => b.where).join(',')}`);
  assert.equal(payloads(events, 'subtask.start').length, 0);
  assert.equal(payloads(events, 'subtasks.plan').length, 1, 'the split was planned before the money ran out');
  assert.equal(result.stopReason, 'budget');
  assert.notEqual(result.verdict, 'PASS');
});

// ---------------------------------------------------------------------------
// scoped acceptance (defect found live: subtasks judged against the parent list)
// ---------------------------------------------------------------------------

test('scopedAcceptance keeps the criteria about this subtask and the generic ones', () => {
  // The exact plan qwen3.8 produced for the two-subtask-feature bench task.
  const plan = {
    acceptance: [
      "src/csv.js exports toCsv which returns 'id,name,score' rows joined by newlines.",
      "src/table.js exports toTable which returns 'id | name | score' rows with a --- separator.",
      'src/csv.js and src/table.js do not import each other.',
      'test/integration.test.js requires both toCsv and toTable and asserts the line counts.',
      'npm test --silent passes (all existing tests plus the new integration test).',
    ],
    subtasks: [
      { id: 'st-csv', title: 'Implement toCsv', files: ['src/csv.js'], acceptance: [] },
      { id: 'st-table', title: 'Implement toTable', files: ['src/table.js'], acceptance: [] },
      {
        id: 'st-it',
        title: 'Write the integration test',
        files: ['test/integration.test.js'],
        acceptance: [],
        depends_on: ['st-csv', 'st-table'],
      },
    ],
  };
  const [csv, table, it] = plan.subtasks;
  // The "do not import each other" criterion names csv.js first, so it is the
  // csv subtask's, annotated with the half that is not this subtask's work.
  const mutual = `${plan.acceptance[2]} (only the src/csv.js part is this subtask's; src/table.js belongs to another subtask)`;
  assert.deepEqual(scopedAcceptance(plan, csv), [plan.acceptance[0], mutual]);
  assert.deepEqual(scopedAcceptance(plan, table), [plan.acceptance[1]]);
  // "toCsv" is not a mention of csv.js; "integration test" is a mention of the
  // test subtask's file stem, so the npm-test criterion belongs to it alone.
  assert.deepEqual(scopedAcceptance(plan, it), [plan.acceptance[3], plan.acceptance[4]]);

  // subPlanFor uses the scoped list, so the reviewer never sees a foreign criterion.
  assert.deepEqual(subPlanFor(plan, csv, 0).acceptance, [plan.acceptance[0], mutual]);
});

test('scopedAcceptance gives a mixed criterion to the subtask whose file it names first', () => {
  // The exact wording from the live rerun r_20260909_6a5142fd: the integration
  // criterion says "../src/csv.js", and the csv and table reviewers failed it.
  const plan = {
    acceptance: [
      'src/csv.js exports toCsv(records) via module.exports; the first line is exactly "id,name,score"',
      'src/table.js exports toTable(records) via module.exports; the second line is exactly "---"',
      'src/csv.js does not require or import src/table.js and vice versa',
      "test/integration.test.js requires toCsv from ../src/csv.js, toTable from ../src/table.js, and sampleRecords from ../src/records.js, and asserts toTable(records).split('\\n').length === toCsv(records).split('\\n').length + 1",
      'npm test --silent passes with zero failures covering both the existing records test and the new integration test',
    ],
    subtasks: [
      { id: 'csv-formatter', title: 'Implement src/csv.js with toCsv(records)', files: ['src/csv.js'], acceptance: [] },
      { id: 'table-formatter', title: 'Implement src/table.js with toTable(records)', files: ['src/table.js'], acceptance: [] },
      { id: 'integration-test', title: 'Write test/integration.test.js', files: ['test/integration.test.js'], acceptance: [] },
    ],
  };
  const [csv, table, it] = plan.subtasks;
  const [a0, a1, a2, a3, a4] = plan.acceptance;
  assert.deepEqual(scopedAcceptance(plan, csv), [
    a0,
    `${a2} (only the src/csv.js part is this subtask's; src/table.js belongs to another subtask)`,
  ]);
  assert.deepEqual(scopedAcceptance(plan, table), [a1]);
  // The test file is the subject of #4 even though csv.js and table.js follow;
  // both are named, so the note lists them in the order the plan does.
  assert.deepEqual(scopedAcceptance(plan, it), [
    `${a3} (only the test/integration.test.js part is this subtask's; src/csv.js, src/table.js belongs to another subtask)`,
    a4,
  ]);
  // A criterion that names no subtask file is kept for everyone, unannotated.
  assert.deepEqual(scopedAcceptance({ ...plan, acceptance: ['eslint passes'] }, csv), ['eslint passes']);
});

test('scopedAcceptance prefers the subtask list, keeps generic criteria, and never returns empty', () => {
  const plan = {
    acceptance: ['lib/parse.js handles quoted fields', 'the CLI prints a summary', 'eslint passes'],
    subtasks: [
      { id: 'p', title: 'parser', files: ['lib/parse.js'] },
      { id: 'c', title: 'cli', files: ['bin/cli.js'] },
    ],
  };
  const [parser, cli] = plan.subtasks;
  // A subtask with its own criteria uses them verbatim (trimmed, blanks dropped).
  assert.deepEqual(scopedAcceptance(plan, { ...parser, acceptance: [' own ', '', 7] }), ['own']);
  // Generic criteria stay for everyone; the one naming lib/parse.js leaves the
  // cli, and the one saying CLI (the stem of bin/cli.js) leaves the parser.
  assert.deepEqual(scopedAcceptance(plan, parser), ['lib/parse.js handles quoted fields', 'eslint passes']);
  assert.deepEqual(scopedAcceptance(plan, cli), ['the CLI prints a summary', 'eslint passes']);
  // Windows separators and case do not defeat the match.
  assert.deepEqual(
    scopedAcceptance({ ...plan, acceptance: ['LIB\PARSE.JS is rewritten'] }, cli),
    ['"cli" is implemented in bin/cli.js and the existing tests still pass.'],
  );
  // Nothing survives: one concrete criterion is synthesised from the subtask.
  const only = { acceptance: ['src/other.js does x'], subtasks: [{ id: 'o', files: ['src/other.js'] }, { id: 'me', files: ['src/me.js'] }] };
  assert.deepEqual(scopedAcceptance(only, only.subtasks[1]), ['"me" is implemented in src/me.js and the existing tests still pass.']);
  // Generic stems (index, test, utils) never attribute a criterion to a subtask.
  const generic = {
    acceptance: ['the index exports everything', 'src/util/index.js re-exports helpers'],
    subtasks: [{ id: 'a', files: ['src/index.js'] }, { id: 'b', files: ['src/util/index.js'] }],
  };
  assert.deepEqual(scopedAcceptance(generic, generic.subtasks[0]), ['the index exports everything']);
  assert.deepEqual(scopedAcceptance(generic, generic.subtasks[1]), generic.acceptance);
  // No plan acceptance at all, no title: still one criterion.
  assert.deepEqual(scopedAcceptance({}, { id: 'z', files: [] }), ['"z" is implemented and the existing tests still pass.']);
});

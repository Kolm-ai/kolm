// Owner D — the orchestrator's control flow (spec 6.4): ranking, the budget
// guard, repair and escalation, direct mode, resume, the error path, the event
// stream and an unresolved judge. Scripted providers and the fake driver only:
// no network, no real repository.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { runTask, resumeTask, rankCandidates, pickWinner } from '../src/ladder/orchestrator.js';
import { verifyReceipt } from '../src/ladder/receipt.js';
import { openStore, sqliteAvailable, newRunId } from '../src/ladder/store.js';
import { createFakeDriver } from '../src/ladder/worktree/fake.js';
import { createFakeProvider } from '../src/ladder/providers/fake.js';
import { isLadderError } from '../src/ladder/errors.js';
import { loadBands } from '../src/ladder/bands.js';
import { loadRubric } from '../src/ladder/rubric.js';
import { evalRuns } from '../src/ladder/eval.js';

const HAS_SQLITE = sqliteAvailable();
const options = HAS_SQLITE ? {} : { skip: 'node:sqlite is unavailable on this Node' };

const BANDS = loadBands();
const RUBRIC = loadRubric();
const RUBRIC_IDS = RUBRIC.items.map((item) => item.id);

/* ------------------------------------------------------------- fixtures */

const tempDirs = [];
const openStores = [];

function tempStore(label) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `kolm-orch-${label}-`));
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
  'README.md': '# demo\n\nA tiny repository for the orchestrator tests.\n',
  'src/index.js': "export const name = 'demo';\n",
};

const GOOD_MATH = 'export function add(a, b) {\n  return a + b;\n}\n';
const BROKEN_MATH = 'export function add(a, b) {\n  // TODO: implement\n  return 0;\n}\n';

/** Passing or failing scripted test output, depending on the worktree. */
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

function planner({ costUsd = 0.002 } = {}) {
  return createFakeProvider({
    ref: 'anthropic:claude-sonnet',
    script: () => ({ json: PLAN_JSON, costUsd }),
  });
}

function coder(ref, content, { costUsd = 0.001 } = {}) {
  return createFakeProvider({
    ref,
    script: (call) => {
      const first = (call.messages || []).length === 1;
      if (first) return { toolCalls: [{ name: 'write_file', input: { path: 'src/math.js', content } }], costUsd };
      return { toolCalls: [{ name: 'finish', input: { summary: 'wrote src/math.js' } }], costUsd };
    },
  });
}

function reviewer(score, confidence) {
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

/** A judge that always passes the candidate. */
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

/** A judge whose reply never parses as JSON, on either attempt. */
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
    repairTurns: 0,
    maxCostUsd: 5,
    calibrationRate: 0,
    stageTimeoutMs: 60000,
    codeDir: path.join(dir, 'code'),
    ...extra,
  };
}

function fixture(label, { files = BASE_FILES, exec = { [TEST_CMD]: scriptedTests } } = {}) {
  const { store, dir } = tempStore(label);
  return { store, dir, driver: createFakeDriver({ files, exec }) };
}

/* ------------------------------------------------------------- ranking */

test('rankCandidates orders by reasons, then cost, then candidate id', options, () => {
  const list = [
    { candidateId: 'c2a0', verdict: 'PASS', reasons: [], costUsd: 0.9 },
    { candidateId: 'c1a0', verdict: 'FAIL', reasons: ['tests fail'], costUsd: 0.1 },
    { candidateId: 'c0a0', verdict: 'PASS', reasons: [], costUsd: 0.9 },
    { candidateId: 'c3a0', verdict: 'PASS', reasons: [], costUsd: 0.2 },
    { candidateId: 'c4a0', verdict: 'JUDGE', reasons: ['rubric_score is unsure', 'low confidence'], costUsd: 0.01 },
  ];
  assert.deepEqual(
    rankCandidates(list).map((c) => c.candidateId),
    ['c3a0', 'c0a0', 'c2a0', 'c1a0', 'c4a0'],
    'fewest reasons first, then cheapest, then id',
  );
  // Only PASS candidates can win, and the cheapest of them does.
  assert.equal(pickWinner(list).candidateId, 'c3a0');
  assert.equal(pickWinner([{ candidateId: 'c0a0', verdict: 'FAIL', reasons: [] }]), null);
  assert.equal(pickWinner([]), null);
  assert.deepEqual(rankCandidates(undefined), []);
  // Ranking does not mutate its input.
  assert.equal(list[0].candidateId, 'c2a0');
});

/* -------------------------------------------------------------- budget */

test('the budget guard stops the run before any coder spends', options, async () => {
  const { store, dir, driver } = fixture('budget');
  const cheapCoder = coder('anthropic:claude-haiku', GOOD_MATH);
  const events = [];

  const result = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js' },
    repoPath: '/repo/demo',
    // The plan alone costs five times the ceiling.
    config: baseConfig(dir, { maxCostUsd: 0.01 }),
    store,
    driver,
    providers: {
      orchestrator: planner({ costUsd: 0.05 }),
      judge: passingJudge(),
      coders: [cheapCoder],
      reviewer: reviewer(0.95, 0.9),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  assert.equal(result.stopReason, 'budget');
  assert.equal(result.verdict, 'FAIL');
  assert.equal(result.winner, null);
  assert.equal(result.candidates.length, 0);
  assert.equal(cheapCoder.calls.length, 0, 'no coder was ever asked for a patch');

  const budgetEvent = events.find((e) => e.type === 'run.budget');
  assert.ok(budgetEvent, 'a run.budget event was emitted');
  assert.equal(budgetEvent.payload.where, 'coder');
  assert.equal(budgetEvent.payload.maxCostUsd, 0.01);
  assert.ok(budgetEvent.payload.spentUsd >= 0.01);

  // A stopped run still produces a signed receipt with no winner.
  assert.equal(result.receipt.winner, null);
  assert.equal(result.receipt.stop_reason, 'budget');
  assert.equal(verifyReceipt(result.receipt).ok, true);
});

/* ------------------------------------------------ repair and escalation */

test('a run escalates through every rung before giving up', options, async () => {
  const { store, dir, driver } = fixture('escalation');
  const cheapRef = 'ollama:qwen3:8b';
  const strongRef = 'anthropic:claude-sonnet';
  const events = [];

  const result = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js' },
    repoPath: '/repo/demo',
    config: baseConfig(dir, { repairTurns: 1 }),
    store,
    driver,
    providers: {
      orchestrator: planner(),
      judge: passingJudge(),
      // Both rungs write the same broken patch, so neither repair nor
      // escalation can rescue the run.
      coders: [coder(cheapRef, BROKEN_MATH), coder(strongRef, BROKEN_MATH)],
      reviewer: reviewer(0.95, 0.9),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  assert.equal(result.verdict, 'FAIL');
  assert.equal(result.stopReason, 'ladder-exhausted');
  assert.equal(result.winner, null);

  const ids = result.candidates.map((c) => c.candidateId).sort();
  assert.deepEqual(ids, ['c0a0', 'c0a1', 'c1a0', 'c1a1'], 'each rung repaired once');
  for (const row of result.candidates) assert.equal(row.verdict, 'FAIL');

  const escalations = events.filter((e) => e.type === 'escalate');
  assert.equal(escalations.length, 1, 'one escalation, from the cheap rung to the strong one');
  assert.equal(escalations[0].payload.from, cheapRef);
  assert.equal(escalations[0].payload.to, strongRef);
  assert.equal(escalations[0].payload.rung, 1);

  // The repair feedback names the stage that rejected the patch.
  const repairs = events.filter((e) => e.type === 'repair');
  assert.equal(repairs.length, 2);
  assert.equal(repairs[0].payload.stage, 'l2-dynamic');
  assert.ok(repairs[0].payload.reasons.length > 0, 'the coder was told why it failed');

  // The escalation happened after the cheap rung's repair, not before it.
  const order = events.map((e) => e.type);
  assert.ok(order.indexOf('repair') < order.indexOf('escalate'));

  assert.equal(result.receipt.winner, null);
  assert.deepEqual(result.receipt.models.coders, [cheapRef, strongRef]);
  assert.equal(verifyReceipt(result.receipt).ok, true);
});

/* --------------------------------------------------------- direct mode */

test('direct mode runs one candidate and always calls the judge', options, async () => {
  const { store, dir, driver } = fixture('direct');
  const judge = passingJudge();
  const reviewerProvider = reviewer(0.95, 0.9);

  const result = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js' },
    repoPath: '/repo/demo',
    // Two candidates and two rungs are configured; direct mode ignores both.
    // Calibration at rate 1 must not re-judge the candidate the judge just ruled on.
    config: baseConfig(dir, { candidates: 2, calibrationRate: 1 }),
    mode: 'direct',
    store,
    driver,
    providers: {
      orchestrator: planner(),
      judge,
      coders: [coder('anthropic:claude-haiku', GOOD_MATH), coder('anthropic:claude-sonnet', GOOD_MATH)],
      reviewer: reviewerProvider,
    },
    bands: BANDS,
    rubric: RUBRIC,
  });

  assert.equal(result.verdict, 'PASS');
  assert.equal(result.candidates.length, 1, 'direct mode builds a single candidate');
  assert.equal(result.winner, 'c0a0');
  assert.equal(judge.calls.length, 1, 'the judge ran even though the cascade was happy');
  assert.equal(reviewerProvider.calls.length, 0, 'l3-semantic is skipped in direct mode');

  const judgments = store.listJudgments(result.runId);
  assert.equal(judgments.length, 1, 'the calibration sampler never re-judges a judged candidate');
  assert.equal(judgments[0].kind, 'unsure');
  assert.equal(judgments[0].candidate_id, 'c0a0');

  assert.equal(result.receipt.mode, 'direct');
  assert.equal(result.receipt.judge.candidate_id, 'c0a0');
  assert.equal(result.receipt.judge.decision, 'good');
  assert.equal(result.receipt.models.coders.length, 1, 'only the first rung is recorded');
  assert.equal(verifyReceipt(result.receipt).ok, true);

  // The funnel and the eval report count the same judged candidate.
  const funnelReport = store.funnel();
  const evalReport = evalRuns({ store, runIds: [result.runId] });
  assert.equal(funnelReport.judged, 1, 'store.funnel sees the judgment');
  assert.equal(evalReport.runs[0].judged, 1, 'evalRuns sees the judgment');
  assert.equal(evalReport.funnel.judged, funnelReport.judged, 'both reports agree');
});

/* -------------------------------------------------------------- resume */

test('resuming a finished run rehydrates it from the store and runs nothing', options, async () => {
  const { store, dir, driver } = fixture('resume');
  const orchestratorProvider = planner();
  const coderProvider = coder('anthropic:claude-haiku', GOOD_MATH);
  const reviewerProvider = reviewer(0.95, 0.9);
  const config = baseConfig(dir, { keepWorktrees: 'all' });
  const providers = {
    orchestrator: orchestratorProvider,
    judge: passingJudge(),
    coders: [coderProvider],
    reviewer: reviewerProvider,
  };
  const opts = {
    task: { text: 'add a function add(a,b) to src/math.js' },
    repoPath: '/repo/demo',
    config,
    store,
    driver,
    providers,
    bands: BANDS,
    rubric: RUBRIC,
  };

  const first = await runTask(opts);
  assert.equal(first.verdict, 'PASS');
  const plannerCalls = orchestratorProvider.calls.length;
  const coderCalls = coderProvider.calls.length;
  const reviewerCalls = reviewerProvider.calls.length;
  const execCalls = driver.commands.length;
  assert.ok(plannerCalls > 0 && coderCalls > 0 && execCalls > 0);

  const events = [];
  // The task, repo path and mode are rehydrated from the stored run row.
  const second = await resumeTask({
    runId: first.runId,
    config,
    store,
    driver,
    providers,
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  assert.equal(second.runId, first.runId, 'the same run is reported');
  assert.equal(second.verdict, 'PASS');
  assert.equal(second.winner, 'c0a0');
  assert.equal(second.stopReason, 'already-finished');
  assert.equal(second.branch, first.branch);
  assert.equal(second.sha, first.sha);
  assert.equal(second.receiptPath, first.receiptPath);
  assert.equal(second.candidates.length, first.candidates.length);
  assert.equal(orchestratorProvider.calls.length, plannerCalls, 'the planner was not called again');
  assert.equal(coderProvider.calls.length, coderCalls, 'no coder turn was spent');
  assert.equal(reviewerProvider.calls.length, reviewerCalls, 'no stage was recomputed');
  assert.equal(driver.commands.length, execCalls, 'no command was re-run');

  const types = events.map((e) => e.type);
  assert.deepEqual(types, ['run.resume', 'run.end'], 'a finished run only reports itself');
  assert.equal(events[0].payload.alreadyFinished, true);
  assert.equal(events[1].payload.stopReason, 'already-finished');
});

test('an interrupted run resumes from its stored shape and a finished run short-circuits', options, async () => {
  const { store, dir, driver } = fixture('resume-interrupted');
  const orchestratorProvider = planner();
  const coderProvider = coder('anthropic:claude-haiku', GOOD_MATH);
  const reviewerProvider = reviewer(0.95, 0.9);
  const judgeProvider = passingJudge();
  const providers = {
    orchestrator: orchestratorProvider,
    judge: judgeProvider,
    coders: [coderProvider],
    reviewer: reviewerProvider,
  };
  const task = { text: 'add a function add(a,b) to src/math.js' };

  // Interrupt the first run the moment the dynamic stage has landed, the way a
  // killed process would: everything up to l2 is in the store, l3 never ran.
  const ac = new AbortController();
  let runId = null;
  await assert.rejects(
    runTask({
      task,
      repoPath: '/repo/demo',
      config: baseConfig(dir, { candidates: 1, keepWorktrees: 'all' }),
      store,
      driver,
      providers,
      bands: BANDS,
      rubric: RUBRIC,
      signal: ac.signal,
      onEvent: (evt) => {
        if (evt.type === 'run.start') runId = evt.run_id;
        if (evt.type === 'stage.result' && evt.payload.stage === 'l2-dynamic') ac.abort();
      },
    }),
    (err) => isLadderError(err) && err.code === 'aborted',
  );
  assert.ok(runId, 'the run started');
  assert.equal(store.getRun(runId).status, 'error');
  assert.equal(reviewerProvider.calls.length, 0, 'the semantic stage never ran');
  const coderCalls = coderProvider.calls.length;

  // Resume with a WIDER caller config: the stored run row (one candidate) must
  // win, or the resume would generate two fresh candidates for a run that was
  // one candidate wide.
  const events = [];
  const second = await resumeTask({
    runId,
    config: baseConfig(dir, { candidates: 3, keepWorktrees: 'all' }),
    store,
    driver,
    providers,
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });
  assert.equal(second.verdict, 'PASS');
  assert.equal(second.winner, 'c0a0');
  assert.equal(second.stopReason, 'winner');
  assert.equal(store.listCandidates(runId).length, 1, 'no extra candidate was generated');
  assert.equal(coderProvider.calls.length, coderCalls, 'the stored diff was reused');
  assert.equal(reviewerProvider.calls.length, 1, 'the semantic stage ran exactly once');
  const cachedStages = events.filter((e) => e.type === 'stage.cached').map((e) => e.payload.stage);
  assert.deepEqual(cachedStages, ['l0-meta', 'l1-static', 'l2-dynamic']);
  assert.deepEqual(
    events.filter((e) => e.type === 'stage.result').map((e) => e.payload.stage),
    ['l3-semantic'],
  );
  assert.equal(store.getRun(runId).config.candidates, 1, 'the stored shape is untouched');

  // Resuming a finished run does nothing but report the stored outcome.
  const before = {
    coder: coderProvider.calls.length,
    reviewer: reviewerProvider.calls.length,
    judge: judgeProvider.calls.length,
    exec: driver.commands.length,
  };
  const again = [];
  const third = await resumeTask({
    runId,
    config: baseConfig(dir, { candidates: 3, keepWorktrees: 'all' }),
    store,
    driver,
    providers,
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => again.push(evt),
  });
  assert.equal(third.stopReason, 'already-finished');
  assert.equal(third.verdict, 'PASS');
  assert.equal(third.winner, 'c0a0');
  assert.equal(third.branch, second.branch);
  assert.equal(third.sha, second.sha);
  assert.deepEqual(again.map((e) => e.type), ['run.resume', 'run.end']);
  assert.equal(again[0].payload.alreadyFinished, true);
  assert.deepEqual({
    coder: coderProvider.calls.length,
    reviewer: reviewerProvider.calls.length,
    judge: judgeProvider.calls.length,
    exec: driver.commands.length,
  }, before, 'nothing ran');
});

/* --------------------------------------------------------- error path */

test('a driver failure marks the run as errored and rethrows a LadderError', options, async () => {
  const { store, dir } = tempStore('error');
  const runId = newRunId();
  const broken = {
    kind: 'fake',
    async create() {
      // A realistic failure message that happens to carry a credential.
      throw new Error('cannot reach the sandbox with token sk-ant-api03-NOTAREALVALUE99');
    },
    async open() {
      throw new Error('unreachable');
    },
    async destroyRun() {
      return { removed: 0 };
    },
  };
  const events = [];

  await assert.rejects(
    () => runTask({
      task: { text: 'add a function add(a,b) to src/math.js' },
      repoPath: '/repo/demo',
      runId,
      config: baseConfig(dir),
      store,
      driver: broken,
      providers: {
        orchestrator: planner(),
        judge: passingJudge(),
        coders: [coder('anthropic:claude-haiku', GOOD_MATH)],
        reviewer: reviewer(0.95, 0.9),
      },
      bands: BANDS,
      rubric: RUBRIC,
      onEvent: (evt) => events.push(evt),
    }),
    (err) => {
      assert.ok(isLadderError(err, 'execution'), `expected a LadderError(execution), got ${err?.code ?? err}`);
      assert.ok(!err.message.includes('NOTAREALVALUE99'), 'the credential never reaches the caller');
      assert.ok(err.message.includes('[redacted]'));
      return true;
    },
  );

  assert.equal(store.getRun(runId).status, 'error');
  const errorEvent = events.find((e) => e.type === 'run.error');
  assert.ok(errorEvent, 'a run.error event was emitted');
  assert.equal(errorEvent.payload.code, 'execution');
  assert.ok(!errorEvent.payload.message.includes('NOTAREALVALUE99'));

  // Bad arguments fail before any run row is created.
  await assert.rejects(
    () => runTask({ task: { text: 'x' }, store, driver: broken, providers: {} }),
    (err) => isLadderError(err, 'bad-arg'),
  );
});

/* --------------------------------------------------------- event order */

test('onEvent observes the whole run in cascade order', options, async () => {
  const { store, dir, driver } = fixture('events');
  const events = [];

  const result = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js' },
    repoPath: '/repo/demo',
    config: baseConfig(dir),
    store,
    driver,
    providers: {
      orchestrator: planner(),
      judge: passingJudge(),
      coders: [coder('anthropic:claude-haiku', GOOD_MATH)],
      reviewer: reviewer(0.95, 0.9),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  const types = events.map((e) => e.type);
  const at = (type) => {
    const idx = types.indexOf(type);
    assert.notEqual(idx, -1, `expected a ${type} event, saw ${types.join(', ')}`);
    return idx;
  };
  assert.ok(at('run.start') < at('plan'));
  assert.ok(at('plan') < at('coder.turn'));
  assert.ok(at('coder.turn') < at('stage.result'));
  assert.ok(at('stage.result') < at('run.end'));
  assert.equal(types[0], 'run.start');
  assert.equal(types[types.length - 1], 'run.end');

  // Every event carries the run id and reaches the store as well as onEvent.
  for (const evt of events) assert.equal(evt.run_id, result.runId);
  const stored = store.eventsSince(result.runId, 0, 1000);
  assert.deepEqual(stored.map((e) => e.type), types, 'onEvent mirrors the stored stream');

  const turn = events.find((e) => e.type === 'coder.turn');
  assert.equal(turn.payload.candidateId, 'c0a0');
  assert.ok(Array.isArray(turn.payload.toolCalls));
  assert.ok(turn.payload.toolCalls.includes('write_file'));
});

/* --------------------------------------------------- unresolved judge */

test('a judge that cannot answer leaves the run at the JUDGE verdict', options, async () => {
  const { store, dir, driver } = fixture('unresolved');
  const judge = unparsableJudge();
  const events = [];

  const result = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js' },
    repoPath: '/repo/demo',
    config: baseConfig(dir),
    store,
    driver,
    providers: {
      orchestrator: planner(),
      judge,
      coders: [coder('anthropic:claude-haiku', GOOD_MATH)],
      // Mid-band scores put l3-semantic in the unsure band, which is what
      // sends the candidate to the judge in the first place.
      reviewer: reviewer(0.7, 0.6),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  assert.equal(result.verdict, 'JUDGE');
  assert.equal(result.stopReason, 'judge-unresolved');
  assert.equal(result.winner, null);
  assert.equal(result.branch, null);
  assert.equal(result.candidates[0].verdict, 'JUDGE');
  assert.equal(judge.calls.length, 2, 'the judge was retried exactly once');

  const failed = events.find((e) => e.type === 'judge.failed');
  assert.ok(failed, 'the unresolved judgment was reported');
  assert.equal(failed.payload.candidateId, 'c0a0');

  // The cost of the failed judge attempts is still on the ledger.
  assert.ok(result.costUsd > 0);
  assert.equal(result.receipt.verdict, 'JUDGE');
  assert.equal(result.receipt.judge, null, 'no judgment row means no judge section');
  assert.equal(verifyReceipt(result.receipt).ok, true);
});

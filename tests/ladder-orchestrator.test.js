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
import { routeRun } from '../src/ladder/routing.js';

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
    repairRounds: 0,
    repairTurns: 6,
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
    config: baseConfig(dir, { repairRounds: 1 }),
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

/* ------------------------------------------------------------- reopen */

/**
 * How many coder SESSIONS a scripted provider opened. Only the first turn of a
 * session carries a one-message conversation, so counting those counts sessions
 * rather than turns. A rung that reused a cached diff opened none.
 */
function coderSessions(provider) {
  return (provider.calls ?? []).filter((call) => (call.messages || []).length === 1).length;
}

/**
 * Run a task with one weak rung that writes a patch the tests reject, so the run
 * ends FAIL the normal way: a receipt, `status: 'done'`, no throw.
 */
async function failedRun({ store, driver, coderProvider, config }) {
  const result = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js' },
    repoPath: '/repo/demo',
    config,
    store,
    driver,
    providers: {
      orchestrator: planner(),
      judge: passingJudge(),
      coders: [coderProvider],
      reviewer: reviewer(0.95, 0.9),
    },
    bands: BANDS,
    rubric: RUBRIC,
  });
  assert.equal(result.verdict, 'FAIL', 'the fixture run ends FAIL');
  assert.equal(result.stopReason, 'ladder-exhausted');
  const row = store.getRun(result.runId);
  assert.equal(row.status, 'done', 'a FAIL finishes like any other run');
  assert.equal(row.receipt.verdict, 'FAIL', 'and it leaves a receipt behind');
  return result;
}

test('a run that ended FAIL is reopened by resume with a stronger coder', options, async () => {
  const { store, dir, driver } = fixture('reopen-fail');
  const weakRef = 'ollama:qwen3:8b';
  const strongRef = 'anthropic:claude-sonnet';
  const weak = coder(weakRef, BROKEN_MATH);
  // Two candidates per rung, so the offset is visible: the stored rung owns idx
  // 0 and 1, and the coder on the resume command line must start at idx 2.
  const config = baseConfig(dir, { candidates: 2 });

  const first = await failedRun({ store, driver, coderProvider: weak, config });
  const runId = first.runId;
  assert.deepEqual(
    store.listCandidates(runId).map((c) => c.candidate_id).sort(),
    ['c0a0', 'c1a0'],
  );
  const weakSessions = coderSessions(weak);

  const strong = coder(strongRef, GOOD_MATH);
  const events = [];
  const second = await resumeTask({
    runId,
    config,
    store,
    driver,
    providers: {
      orchestrator: planner(),
      judge: passingJudge(),
      coders: [strong],
      reviewer: reviewer(0.95, 0.9),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  // The run was reopened, not re-printed.
  const [resumed] = events.filter((e) => e.type === 'run.resume').map((e) => e.payload);
  assert.equal(resumed.reopened, true);
  assert.equal(resumed.priorVerdict, 'FAIL');
  assert.equal(resumed.priorStopReason, 'ladder-exhausted');
  assert.deepEqual(resumed.coders, [strongRef]);
  assert.notEqual(second.stopReason, 'already-finished');

  // The new coder took the rung ABOVE the stored one, so nothing cached was
  // re-coded and no destroyed candidate id was written a second time.
  const starts = events.filter((e) => e.type === 'candidate.start').map((e) => e.payload);
  assert.deepEqual(starts.map((p) => p.candidateId), ['c2a0', 'c3a0']);
  assert.ok(starts.every((p) => p.generatorRef === strongRef), 'the strong rung wrote both');
  assert.deepEqual(starts.map((p) => p.rung), [1, 1], 'the ids and the rung agree');
  assert.equal(coderSessions(weak), weakSessions, 'the weak coder was never called again');

  assert.equal(second.verdict, 'PASS');
  assert.equal(second.winner, 'c2a0');
  assert.equal(verifyReceipt(second.receipt).ok, true);
  assert.equal(second.receipt.winner.candidate_id, 'c2a0');
  assert.deepEqual(
    second.receipt.candidates.map((c) => c.candidate_id).sort(),
    ['c0a0', 'c1a0', 'c2a0', 'c3a0'],
    'the receipt carries both attempts',
  );
  const byId = new Map(second.receipt.candidates.map((c) => [c.candidate_id, c]));
  assert.equal(byId.get('c0a0').verdict, 'FAIL');
  assert.equal(byId.get('c0a0').generator, weakRef);
  assert.equal(byId.get('c2a0').generator, strongRef);

  // The receipt file is replaced in place and the row is finished again.
  assert.equal(second.receiptPath, first.receiptPath);
  assert.equal(JSON.parse(fs.readFileSync(second.receiptPath, 'utf8')).verdict, 'PASS');
  const row = store.getRun(runId);
  assert.equal(row.status, 'done');
  assert.equal(row.receipt.verdict, 'PASS');
  assert.equal(row.winner_candidate, 'c2a0');
});

test('a reopened run with the same coder codes fresh candidates instead of re-coding the old ones', options, async () => {
  const { store, dir, driver } = fixture('reopen-same-coder');
  const weakRef = 'ollama:qwen3:8b';
  const config = baseConfig(dir, { candidates: 2 });

  const first = await failedRun({
    store,
    driver,
    coderProvider: coder(weakRef, BROKEN_MATH),
    config,
  });
  const runId = first.runId;
  const before = new Map(store.listCandidates(runId).map((c) => [c.candidate_id, c]));
  assert.deepEqual([...before.keys()].sort(), ['c0a0', 'c1a0']);
  const createdBefore = driver.created.length;

  // The same model, told to write a patch that works this time. The stored
  // attempts were destroyed with the run, so they must not be touched.
  const again = coder(weakRef, GOOD_MATH);
  const second = await resumeTask({
    runId,
    config,
    store,
    driver,
    providers: {
      orchestrator: planner(),
      judge: passingJudge(),
      coders: [again],
      reviewer: reviewer(0.95, 0.9),
    },
    bands: BANDS,
    rubric: RUBRIC,
  });

  assert.equal(second.verdict, 'PASS');
  assert.equal(second.winner, 'c2a0');
  assert.equal(coderSessions(again), 2, 'one session per candidate of the new rung');

  const after = new Map(store.listCandidates(runId).map((c) => [c.candidate_id, c]));
  assert.deepEqual([...after.keys()].sort(), ['c0a0', 'c1a0', 'c2a0', 'c3a0']);
  for (const id of ['c0a0', 'c1a0']) {
    assert.equal(after.get(id).diff_sha256, before.get(id).diff_sha256, `${id} kept its diff`);
    assert.equal(after.get(id).status, 'destroyed', `${id} was not revived`);
    assert.equal(after.get(id).generator_ref, weakRef);
  }
  // No worktree was created for a destroyed id: only the base and the two new
  // candidates were opened.
  assert.deepEqual(
    driver.created.slice(createdBefore).map((c) => c.candidateId).sort(),
    ['base', 'c2a0', 'c3a0'],
  );
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

/* ------------------------------------ fixtures for routing, coach, hooks */

/** The planner, with extra plan fields (complexity, risk, subtasks...). */
function plannerWith(extra, { costUsd = 0.002 } = {}) {
  return createFakeProvider({
    ref: 'anthropic:claude-sonnet',
    script: () => ({ json: { ...PLAN_JSON, ...extra }, costUsd }),
  });
}

/**
 * One orchestrator provider that plays both roles: it writes the plan, and it
 * answers the coach's call (recognised by the coach's own JSON schema).
 */
function plannerCoach(hints, { planExtra = {} } = {}) {
  return createFakeProvider({
    ref: 'anthropic:claude-sonnet',
    script: (call) => (call?.jsonSchema?.name === 'kolm_coach_hints'
      ? { json: { hints }, costUsd: 0.0005 }
      : { json: { ...PLAN_JSON, ...planExtra }, costUsd: 0.002 }),
  });
}

/** A coder that fails its first attempt and gets it right on the repair turn. */
function repairingCoder(ref, { costUsd = 0.001 } = {}) {
  return createFakeProvider({
    ref,
    script: (call) => {
      const messages = call.messages || [];
      if (messages.length !== 1) {
        return { toolCalls: [{ name: 'finish', input: { summary: 'wrote src/math.js' } }], costUsd };
      }
      const opening = String(messages[0]?.content ?? '');
      const content = opening.includes('REPAIR TURN') ? GOOD_MATH : BROKEN_MATH;
      return { toolCalls: [{ name: 'write_file', input: { path: 'src/math.js', content } }], costUsd };
    },
  });
}

/* ------------------------------------------------------------- routing */

test('routing turns plan complexity into a start rung, a width and a repair cap', options, () => {
  const config = { candidates: 2, repairRounds: 2 };
  const rungs = ['cheap', 'strong'];

  // A plan written before Plan v2 has no complexity at all: it must route to
  // exactly today's numbers.
  const legacy = routeRun({ plan: { risk: 'low' }, config, coderRungs: rungs });
  assert.equal(legacy.complexity, 'medium');
  assert.equal(legacy.startRung, 0);
  assert.equal(legacy.candidates, 2);
  assert.equal(legacy.repairRounds, 2);

  const trivial = routeRun({ plan: { complexity: 'trivial', risk: 'low' }, config, coderRungs: rungs });
  assert.equal(trivial.startRung, 0);
  assert.equal(trivial.candidates, 1, 'a one-line fix does not pay for a second opinion');
  assert.equal(trivial.repairRounds, 1, 'repair rounds are capped, not raised');

  const large = routeRun({ plan: { complexity: 'large', risk: 'low' }, config, coderRungs: rungs });
  assert.equal(large.startRung, 1, 'a large change skips the cheapest rung');
  assert.equal(large.candidates, 3, 'one more candidate than configured');

  // The cap holds whatever the config asks for, and a single-rung ladder cannot
  // start above rung 0.
  const wide = routeRun({ plan: { complexity: 'large' }, config: { candidates: 9 }, coderRungs: rungs });
  assert.equal(wide.candidates, 4);
  const oneRung = routeRun({ plan: { complexity: 'large' }, config, coderRungs: ['only'] });
  assert.equal(oneRung.startRung, 0);

  // Risk floors the width even when complexity says one candidate is enough.
  const risky = routeRun({
    plan: { complexity: 'trivial', risk: 'high' },
    config: { candidates: 1, repairRounds: 2 },
    coderRungs: rungs,
  });
  assert.equal(risky.candidates, 2);
  assert.match(risky.reason, /risk=high/);

  // Routing never raises a budget the operator set.
  const noRepairs = routeRun({
    plan: { complexity: 'trivial' },
    config: { candidates: 2, repairRounds: 0 },
    coderRungs: rungs,
  });
  assert.equal(noRepairs.repairRounds, 0);

  // 'fixed' is the escape hatch: today's numbers, whatever the plan says.
  const fixed = routeRun({
    plan: { complexity: 'large', risk: 'high' },
    config: { ...config, routing: 'fixed' },
    coderRungs: rungs,
  });
  assert.equal(fixed.startRung, 0);
  assert.equal(fixed.candidates, 2);
  assert.equal(fixed.repairRounds, 2);
  assert.equal(fixed.reason, 'routing=fixed');
});

test('a trivial plan routes to one candidate and says so in run.route', options, async () => {
  const { store, dir, driver } = fixture('route-trivial');
  const events = [];

  const result = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js' },
    repoPath: '/repo/demo',
    // Two candidates are configured; routing spends one.
    config: baseConfig(dir, { candidates: 2, repairRounds: 2 }),
    store,
    driver,
    providers: {
      orchestrator: plannerWith({ complexity: 'trivial' }),
      judge: passingJudge(),
      coders: [coder('ollama:qwen3:8b', GOOD_MATH)],
      reviewer: reviewer(0.95, 0.9),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  assert.equal(result.verdict, 'PASS');
  assert.deepEqual(result.candidates.map((c) => c.candidateId), ['c0a0'], 'one candidate, not two');

  const route = events.find((e) => e.type === 'run.route');
  assert.ok(route, 'the routing decision is on the event stream');
  assert.equal(route.payload.complexity, 'trivial');
  assert.equal(route.payload.risk, 'low');
  assert.equal(route.payload.routing, 'auto');
  assert.equal(route.payload.startRung, 0);
  assert.equal(route.payload.candidates, 1);
  assert.equal(route.payload.repairRounds, 1);
  assert.ok(typeof route.payload.reason === 'string' && route.payload.reason.length > 0);

  // The decision is made after the plan and before any coder runs.
  const order = events.map((e) => e.type);
  assert.ok(order.indexOf('plan') < order.indexOf('run.route'));
  assert.ok(order.indexOf('run.route') < order.indexOf('candidate.start'));
});

test('a large plan starts one rung up and widens the search', options, async () => {
  const { store, dir, driver } = fixture('route-large');
  const cheap = coder('ollama:qwen3:8b', BROKEN_MATH);
  const strong = coder('anthropic:claude-sonnet-coder', GOOD_MATH);
  const events = [];

  const result = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js' },
    repoPath: '/repo/demo',
    config: baseConfig(dir, { candidates: 1 }),
    store,
    driver,
    providers: {
      orchestrator: plannerWith({ complexity: 'large' }),
      judge: passingJudge(),
      coders: [cheap, strong],
      reviewer: reviewer(0.95, 0.9),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  assert.equal(result.verdict, 'PASS');
  assert.equal(cheap.calls.length, 0, 'the cheapest rung was skipped entirely');
  // idx = rung * candidates + slot, so rung 1 of a two-wide ladder starts at 2.
  assert.deepEqual(result.candidates.map((c) => c.candidateId).sort(), ['c2a0', 'c3a0']);

  const route = events.find((e) => e.type === 'run.route');
  assert.equal(route.payload.startRung, 1);
  assert.equal(route.payload.candidates, 2);
  assert.match(route.payload.reason, /large/);
});

test('routing=fixed reproduces the configured ladder whatever the plan claims', options, async () => {
  const { store, dir, driver } = fixture('route-fixed');
  const cheap = coder('ollama:qwen3:8b', GOOD_MATH);
  const events = [];

  const result = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js' },
    repoPath: '/repo/demo',
    config: baseConfig(dir, { candidates: 1, routing: 'fixed' }),
    store,
    driver,
    providers: {
      orchestrator: plannerWith({ complexity: 'large', risk: 'high' }),
      judge: passingJudge(),
      coders: [cheap, coder('anthropic:claude-sonnet-coder', GOOD_MATH)],
      reviewer: reviewer(0.95, 0.9),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  assert.equal(result.verdict, 'PASS');
  assert.deepEqual(result.candidates.map((c) => c.candidateId), ['c0a0'], 'rung 0, one candidate');
  const route = events.find((e) => e.type === 'run.route');
  assert.equal(route.payload.routing, 'fixed');
  assert.equal(route.payload.reason, 'routing=fixed');
  assert.equal(route.payload.startRung, 0);
  assert.equal(route.payload.candidates, 1);
});

/* --------------------------------------------------------------- coach */

test('the coach turns a failed round into hints the repair prompt carries', options, async () => {
  const { store, dir, driver } = fixture('coach');
  const hints = [
    'src/math.js: add() returns 0; return a + b instead.',
    'Run the test command before finishing.',
  ];
  const orchestrator = plannerCoach(hints);
  const coderProvider = repairingCoder('ollama:qwen3:8b');
  const events = [];

  const result = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js' },
    repoPath: '/repo/demo',
    config: baseConfig(dir, { repairRounds: 1 }),
    store,
    driver,
    providers: {
      orchestrator,
      judge: passingJudge(),
      coders: [coderProvider],
      reviewer: reviewer(0.95, 0.9),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  assert.equal(result.verdict, 'PASS');
  assert.equal(result.winner, 'c0a1', 'the repaired attempt won');

  const coach = events.filter((e) => e.type === 'repair.coach');
  assert.equal(coach.length, 1, 'one coaching call, on the one repair round');
  assert.equal(coach[0].payload.candidateId, 'c0a1');
  assert.equal(coach[0].payload.round, 1);
  assert.deepEqual(coach[0].payload.hints, hints);
  assert.ok(coach[0].payload.costUsd > 0, 'the coaching call is on the ledger');

  // The coach runs after the repair is announced and before the coder retries.
  const order = events.map((e) => e.type);
  assert.ok(order.indexOf('repair') < order.indexOf('repair.coach'));

  // The coach saw the failing evidence and the diff, and its hints reached the
  // coder's repair prompt.
  const coachCall = orchestrator.calls.find((c) => c.jsonSchema?.name === 'kolm_coach_hints');
  assert.ok(coachCall, 'the coach was asked with its own schema');
  const coachPrompt = String(coachCall.messages[0].content);
  assert.match(coachPrompt, /FAILING EVIDENCE/);
  assert.match(coachPrompt, /DIFF THE CODER PRODUCED/);
  assert.match(coachPrompt, /TODO: implement/, 'the coach read the diff the coder produced');

  const repairPrompt = coderProvider.calls
    .map((c) => String(c.messages?.[0]?.content ?? ''))
    .find((text) => text.includes('REPAIR TURN'));
  assert.ok(repairPrompt, 'the coder was given a repair turn');
  assert.match(repairPrompt, /HINTS FROM THE ORCHESTRATOR/);
  assert.ok(repairPrompt.includes(hints[0]), 'the first hint reached the coder verbatim');

  // Every string the coach produced is ASCII, like the rest of the prompts.
  for (const hint of coach[0].payload.hints) {
    assert.ok(/^[\x20-\x7e]*$/.test(hint), `hint is ASCII: ${hint}`);
  }
});

test('the coach is skipped when it cannot pay for itself', options, async () => {
  // 1. The orchestrator IS the coder: coaching itself buys nothing.
  const same = fixture('coach-same');
  const selfOrchestrator = plannerCoach(['never asked for']);
  const sameEvents = [];
  const selfResult = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js' },
    repoPath: '/repo/demo',
    config: baseConfig(same.dir, { repairRounds: 1 }),
    store: same.store,
    driver: same.driver,
    providers: {
      orchestrator: selfOrchestrator,
      judge: passingJudge(),
      // The same ref as the orchestrator above.
      coders: [repairingCoder('anthropic:claude-sonnet')],
      reviewer: reviewer(0.95, 0.9),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => sameEvents.push(evt),
  });
  assert.equal(selfResult.verdict, 'PASS');
  assert.ok(sameEvents.some((e) => e.type === 'repair'), 'there was a repair round to coach');
  assert.equal(sameEvents.filter((e) => e.type === 'repair.coach').length, 0);
  assert.equal(
    selfOrchestrator.calls.filter((c) => c.jsonSchema?.name === 'kolm_coach_hints').length,
    0,
    'no coaching call was spent',
  );

  // 2. The operator turned it off.
  const off = fixture('coach-off');
  const offOrchestrator = plannerCoach(['never asked for']);
  const offEvents = [];
  const offResult = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js' },
    repoPath: '/repo/demo',
    config: baseConfig(off.dir, { repairRounds: 1, coach: false }),
    store: off.store,
    driver: off.driver,
    providers: {
      orchestrator: offOrchestrator,
      judge: passingJudge(),
      coders: [repairingCoder('ollama:qwen3:8b')],
      reviewer: reviewer(0.95, 0.9),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => offEvents.push(evt),
  });
  assert.equal(offResult.verdict, 'PASS');
  assert.ok(offEvents.some((e) => e.type === 'repair'));
  assert.equal(offEvents.filter((e) => e.type === 'repair.coach').length, 0);
  assert.equal(
    offOrchestrator.calls.filter((c) => c.jsonSchema?.name === 'kolm_coach_hints').length,
    0,
  );
});

test('a coach that answers with nothing usable leaves the repair exactly as it was', options, async () => {
  const { store, dir, driver } = fixture('coach-empty');
  const orchestrator = createFakeProvider({
    ref: 'anthropic:claude-sonnet',
    script: (call) => (call?.jsonSchema?.name === 'kolm_coach_hints'
      // Not a list of hints, and not even the right shape.
      ? { json: { hints: 'sorry' }, costUsd: 0.0005 }
      : { json: PLAN_JSON, costUsd: 0.002 }),
  });
  const coderProvider = repairingCoder('ollama:qwen3:8b');
  const events = [];

  const result = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js' },
    repoPath: '/repo/demo',
    config: baseConfig(dir, { repairRounds: 1 }),
    store,
    driver,
    providers: {
      orchestrator,
      judge: passingJudge(),
      coders: [coderProvider],
      reviewer: reviewer(0.95, 0.9),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  assert.equal(result.verdict, 'PASS', 'a useless coach cannot break a run');
  assert.equal(events.filter((e) => e.type === 'repair.coach').length, 0, 'no hints, no event');
  const repairPrompt = coderProvider.calls
    .map((c) => String(c.messages?.[0]?.content ?? ''))
    .find((text) => text.includes('REPAIR TURN'));
  assert.ok(repairPrompt);
  assert.ok(!repairPrompt.includes('HINTS FROM THE ORCHESTRATOR'), 'no empty hint block');
});

/* -------------------------------------------------- context pack hooks */

test('context.pack reports what actually reached the brief', options, async () => {
  const { store, dir, driver } = fixture('context-pack', {
    files: {
      ...BASE_FILES,
      'AGENTS.md': '# House rules\n\nNever edit package.json. Always run the tests before finishing.\n',
    },
  });
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
      coders: [coder('ollama:qwen3:8b', GOOD_MATH)],
      reviewer: reviewer(0.95, 0.9),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  assert.equal(result.verdict, 'PASS');
  const packs = events.filter((e) => e.type === 'context.pack');
  assert.equal(packs.length, 1, 'exactly one context pack line per run');
  const pack = packs[0].payload;
  assert.deepEqual(pack.instructionFiles, ['AGENTS.md']);
  assert.ok(pack.instructionsBytes > 0, 'the instruction text was measured');
  assert.equal(pack.lessons, 0, 'the first run on a fresh data dir has no lessons yet');
  assert.equal(typeof pack.retrievalFiles, 'number');
  assert.equal(typeof pack.retrievalBytes, 'number');
  assert.equal(pack.retrievalCached, false, 'a cold cache is reported as cold');

  // It is emitted before the plan, because the plan is written against it.
  const order = events.map((e) => e.type);
  assert.ok(order.indexOf('context.pack') < order.indexOf('plan'));
});

test('a run records its lessons and the next run carries them into the brief', options, async () => {
  const { store, dir, driver } = fixture('lessons');
  const providers = () => ({
    orchestrator: planner(),
    judge: passingJudge(),
    coders: [coder('ollama:qwen3:8b', GOOD_MATH)],
    reviewer: reviewer(0.95, 0.9),
  });
  const first = [];
  const second = [];

  const firstResult = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js' },
    repoPath: '/repo/demo',
    config: baseConfig(dir),
    store,
    driver,
    providers: providers(),
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => first.push(evt),
  });
  assert.equal(firstResult.verdict, 'PASS');

  const learned = first.find((e) => e.type === 'run.lessons');
  assert.ok(learned, 'the run wrote down what it learned');
  assert.ok(learned.payload.added >= 1);
  assert.ok(learned.payload.total >= learned.payload.added);
  const firstOrder = first.map((e) => e.type);
  assert.ok(firstOrder.indexOf('run.lessons') < firstOrder.indexOf('run.end'), 'lessons land before run.end');
  assert.equal(first.filter((e) => e.type === 'context.pack')[0].payload.lessons, 0);

  // Second run, same repo and same data directory: the lessons are in the pack.
  const secondResult = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js' },
    repoPath: '/repo/demo',
    config: baseConfig(dir),
    store,
    driver,
    providers: providers(),
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => second.push(evt),
  });
  assert.equal(secondResult.verdict, 'PASS');
  const pack = second.find((e) => e.type === 'context.pack').payload;
  assert.ok(pack.lessons >= 1, 'the second run opened with what the first one learned');
  assert.equal(pack.retrievalCached, true, 'the retrieval came from the context cache');
});

test('config.lessons=false records nothing', options, async () => {
  const { store, dir, driver } = fixture('lessons-off');
  const events = [];

  const result = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js' },
    repoPath: '/repo/demo',
    config: baseConfig(dir, { lessons: false }),
    store,
    driver,
    providers: {
      orchestrator: planner(),
      judge: passingJudge(),
      coders: [coder('ollama:qwen3:8b', GOOD_MATH)],
      reviewer: reviewer(0.95, 0.9),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  assert.equal(result.verdict, 'PASS');
  assert.equal(events.filter((e) => e.type === 'run.lessons').length, 0);
  assert.equal(events.find((e) => e.type === 'context.pack').payload.lessons, 0);
});

test('a context hook that throws degrades the run instead of failing it', options, async () => {
  const { store, dir, driver } = fixture('context-warn');
  // Only the retrieval lookup breaks; the repo brief's own cache still works.
  const brokenStore = Object.create(store);
  brokenStore.cacheGet = (repoKey, sha, kind) => {
    if (String(kind).startsWith('retrieval:')) throw new Error('the context cache is unreadable');
    return store.cacheGet(repoKey, sha, kind);
  };
  const events = [];

  const result = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js' },
    repoPath: '/repo/demo',
    config: baseConfig(dir),
    store: brokenStore,
    driver,
    providers: {
      orchestrator: planner(),
      judge: passingJudge(),
      coders: [coder('ollama:qwen3:8b', GOOD_MATH)],
      reviewer: reviewer(0.95, 0.9),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  assert.equal(result.verdict, 'PASS', 'a degraded context pack is still a working run');
  const warn = events.filter((e) => e.type === 'context.warn');
  assert.equal(warn.length, 1);
  assert.equal(warn[0].payload.kind, 'retrieval');
  assert.match(warn[0].payload.message, /context cache is unreadable/);

  // A degraded hook contributes zeros rather than a missing field.
  const pack = events.find((e) => e.type === 'context.pack').payload;
  assert.equal(pack.retrievalFiles, 0);
  assert.equal(pack.retrievalBytes, 0);
  assert.equal(pack.retrievalCached, false);
});

/* ------------------------------------- a coder that never calls a tool */

/**
 * A coder that describes its tool call in prose and never emits one, in either
 * protocol. Live run r_20260909_b4d9777c, coder ollama:mistral:latest: "To fix
 * the seconds divisor in formatDuration, I will use the `edit_file` function ...
 * Here's the command to do that:", twice, then reason=no-tool-calls.
 */
function proseCoder(ref, { costUsd = 0.001 } = {}) {
  return createFakeProvider({
    ref,
    script: () => ({
      text: 'To fix the seconds divisor in formatDuration, I will use the `edit_file` function.\n'
        + "Here's the command to do that:\n\nedit_file src/math.js \"return 0\" \"return a + b\"",
      costUsd,
    }),
  });
}

/** Plans when asked for JSON, and writes a good patch when handed the tools. */
function planningCoder(ref, content) {
  return createFakeProvider({
    ref,
    script: (call) => {
      if (!Array.isArray(call.tools) || call.tools.length === 0) return { json: PLAN_JSON, costUsd: 0.002 };
      const first = (call.messages || []).length === 1;
      if (first) {
        return { toolCalls: [{ name: 'write_file', input: { path: 'src/math.js', content } }], costUsd: 0.001 };
      }
      return { toolCalls: [{ name: 'finish', input: { summary: 'wrote src/math.js' } }], costUsd: 0.001 };
    },
  });
}

test('a rung that never calls a tool hands its repair to the next rung', options, async () => {
  const { store, dir, driver } = fixture('fallback-rung');
  const cheapRef = 'ollama:mistral:latest';
  const strongRef = 'anthropic:claude-sonnet';
  const events = [];

  const result = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js' },
    repoPath: '/repo/demo',
    config: baseConfig(dir, { repairRounds: 1 }),
    store,
    driver,
    providers: {
      orchestrator: planner(),
      judge: passingJudge(),
      coders: [proseCoder(cheapRef), coder(strongRef, GOOD_MATH)],
      reviewer: reviewer(0.95, 0.9),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  // The run passes on the fallback, without ever reaching rung 1's own attempts.
  assert.equal(result.verdict, 'PASS');
  assert.equal(result.winner, 'c0a1');

  const empty = events.find((e) => e.type === 'candidate.diff').payload;
  assert.equal(empty.candidateId, 'c0a0');
  assert.equal(empty.reason, 'no-tool-calls');
  assert.equal(empty.changedFiles, 0);
  assert.equal(empty.protocol, 'text', 'the coder was moved to the text protocol first');
  assert.equal(events.filter((e) => e.type === 'coder.protocol').length, 1);

  const fallbacks = events.filter((e) => e.type === 'coder.fallback');
  assert.equal(fallbacks.length, 1);
  assert.deepEqual(fallbacks[0].payload, {
    candidateId: 'c0a1',
    from: cheapRef,
    to: strongRef,
    why: 'no-tool-calls',
  });

  // `show` and the receipt name the model that actually wrote the diff.
  const repaired = result.candidates.find((c) => c.candidateId === 'c0a1');
  assert.equal(repaired.generatorRef, strongRef);
  assert.equal(result.candidates.find((c) => c.candidateId === 'c0a0').generatorRef, cheapRef);
  assert.equal(store.getCandidate(result.runId, 'c0a1').generator_ref, strongRef);
  assert.equal(
    result.receipt.candidates.find((c) => c.candidate_id === 'c0a1').generator,
    strongRef,
  );
  assert.equal(verifyReceipt(result.receipt).ok, true);
});

test('the last rung hands its repair to the orchestrator when that is another model', options, async () => {
  const { store, dir, driver } = fixture('fallback-orchestrator');
  const cheapRef = 'ollama:mistral:latest';
  const orchRef = 'anthropic:claude-sonnet';
  const events = [];

  const result = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js' },
    repoPath: '/repo/demo',
    config: baseConfig(dir, { repairRounds: 1 }),
    store,
    driver,
    providers: {
      orchestrator: planningCoder(orchRef, GOOD_MATH),
      judge: passingJudge(),
      coders: [proseCoder(cheapRef)],
      reviewer: reviewer(0.95, 0.9),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  assert.equal(result.verdict, 'PASS');
  assert.equal(result.winner, 'c0a1');
  const fallbacks = events.filter((e) => e.type === 'coder.fallback');
  assert.equal(fallbacks.length, 1);
  assert.equal(fallbacks[0].payload.from, cheapRef);
  assert.equal(fallbacks[0].payload.to, orchRef);
  assert.equal(result.candidates.find((c) => c.candidateId === 'c0a1').generatorRef, orchRef);
});

test('one rung whose ref is the orchestrator ref has nobody to fall back to', options, async () => {
  const { store, dir, driver } = fixture('fallback-none');
  const ref = 'ollama:mistral:latest';
  const events = [];

  const result = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js' },
    repoPath: '/repo/demo',
    config: baseConfig(dir, { repairRounds: 1 }),
    store,
    driver,
    providers: {
      orchestrator: createFakeProvider({ ref, script: () => ({ json: PLAN_JSON, costUsd: 0.002 }) }),
      judge: passingJudge(),
      coders: [proseCoder(ref)],
      reviewer: reviewer(0.95, 0.9),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  assert.equal(result.verdict, 'FAIL');
  assert.equal(events.filter((e) => e.type === 'coder.fallback').length, 0, 'no other model to try');
  assert.equal(events.filter((e) => e.type === 'repair').length, 1, 'the repair still ran, as it does today');
  assert.equal(result.candidates.find((c) => c.candidateId === 'c0a1').generatorRef, ref);
});

/* --------------------------------------------------- the plan audit (4.8) */
//
// Found live in run r_20260909_f02fbbb9: the task text claimed two LF-only
// files used CRLF, the planner turned the claim into acceptance criterion #2,
// and every candidate of every round was rejected for it. Two repair rounds
// over three candidates reproduced the identical rejection and nothing noticed.
//
// The fixtures below reproduce that shape offline: a reviewer that refuses any
// criterion mentioning CRLF, a coder whose patch is correct and identical every
// round, and an orchestrator that also answers the audit.

const CRLF_CRITERION =
  'Both src/math.js and tests/math.test.js use CRLF (\\r\\n) line endings throughout - no bare LF introduced.';
const CRLF_PLAN = { ...PLAN_JSON, acceptance: [...PLAN_JSON.acceptance, CRLF_CRITERION] };

/** The numbered criteria the reviewer was actually shown, in order. */
function criteriaInPrompt(text) {
  const out = [];
  const lines = String(text).split('\n');
  const at = lines.findIndex((l) => l.startsWith('ACCEPTANCE CRITERIA'));
  if (at < 0) return out;
  for (const line of lines.slice(at + 1)) {
    const m = /^(\d+)\.\s(.*)$/.exec(line);
    if (!m) break;
    out.push({ index: Number(m[1]), text: m[2] });
  }
  return out;
}

/**
 * A reviewer that scores the rubric well and attests to every criterion it was
 * shown - except one that asks for CRLF, which these LF-only files can never
 * satisfy. This is the live reviewer's behaviour, reduced to its one decision.
 */
function crlfReviewer() {
  return createFakeProvider({
    ref: 'anthropic:claude-haiku',
    script: (call) => ({
      json: {
        items: RUBRIC_IDS.map((id) => ({ id, score: 0.95, evidence: 'the diff implements add(a, b)' })),
        overall: 0.95,
        confidence: 0.9,
        blocking: [],
        acceptance: criteriaInPrompt(call.messages[0].content).map((c) => ({
          index: c.index,
          met: !/CRLF/i.test(c.text),
          evidence: /CRLF/i.test(c.text) ? 'every line in the diff ends with a bare LF' : 'src/math.js exports add',
        })),
      },
      costUsd: 0.001,
    }),
  });
}

/**
 * One orchestrator provider playing three roles: the planner, the coach (no
 * hints) and the plan audit, told apart by the JSON schema of the call.
 */
function plannerAudit(verdicts) {
  return createFakeProvider({
    ref: 'anthropic:claude-sonnet',
    script: (call) => {
      const schema = call?.jsonSchema?.name;
      if (schema === 'kolm_plan_audit') return { json: { verdicts }, costUsd: 0.0004 };
      if (schema === 'kolm_coach_hints') return { json: { hints: [] }, costUsd: 0.0003 };
      return { json: CRLF_PLAN, costUsd: 0.002 };
    },
  });
}

test('a criterion no candidate can satisfy is retired and the run finishes', options, async () => {
  const { store, dir, driver } = fixture('audit-retire');
  // Criterion #2 is the CRLF one; the events number criteria from 0, so it is
  // index 1 there.
  const orchestrator = plannerAudit([
    { index: 1, verdict: 'retire', reason: 'src/math.js is LF-only in the base and the task does not ask to convert it' },
  ]);
  const events = [];

  const result = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js. The files use CRLF; keep them that way.' },
    repoPath: '/repo/demo',
    config: baseConfig(dir, { repairRounds: 2 }),
    store,
    driver,
    providers: {
      orchestrator,
      judge: passingJudge(),
      // A correct patch, written identically every round: the diff was never
      // the problem.
      coders: [coder('ollama:qwen3:8b', GOOD_MATH)],
      reviewer: crlfReviewer(),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  const stuck = events.filter((e) => e.type === 'repair.stuck');
  assert.equal(stuck.length, 1, 'one stuck round, detected once');
  assert.deepEqual(stuck[0].payload.criteria, [1], 'acceptance #2 is index 1');
  assert.deepEqual(stuck[0].payload.candidates, ['c0a1'], 'the repair came back exactly as it went in');
  assert.equal(stuck[0].payload.unit, null, 'a single-unit run has no subtask index');
  assert.equal(stuck[0].payload.round, 1);

  const amended = events.filter((e) => e.type === 'plan.criterion');
  assert.equal(amended.length, 1, 'one criterion changed');
  assert.equal(amended[0].payload.verdict, 'retire');
  assert.equal(amended[0].payload.index, 1);
  assert.equal(amended[0].payload.from, CRLF_CRITERION);
  assert.equal(amended[0].payload.to, null);
  assert.ok(amended[0].payload.costUsd > 0, 'the audit call is accounted for');

  assert.equal(result.verdict, 'PASS', 'the patch was right all along');
  assert.equal(result.winner, 'c0a1');

  // The re-grade costs no coder session: the second repair round never happens.
  assert.deepEqual(result.candidates.map((c) => c.candidateId).sort(), ['c0a0', 'c0a1']);
  assert.equal(
    events.filter((e) => e.type === 'coder.turn' && e.payload.candidateId === 'c0a2').length,
    0,
    'no coder turn was spent on a third attempt',
  );
  assert.equal(events.filter((e) => e.type === 'repair').length, 1, 'one repair round, not two');

  // The winner's verdict does not quote the criterion that no longer exists.
  const l3 = store.latestResults(result.runId, 'c0a1')['l3-semantic'];
  assert.equal(l3.verdict.decision, 'good');
  for (const reason of l3.verdict.reasons) {
    assert.ok(!/CRLF/i.test(reason), `a retired criterion is not quoted back: ${reason}`);
  }
});

test('a criterion the audit keeps stops the repair loop instead of repeating it', options, async () => {
  const { store, dir, driver } = fixture('audit-keep');
  const orchestrator = plannerAudit([
    { index: 1, verdict: 'keep', reason: 'the coder could still rewrite the file with CRLF endings' },
  ]);
  const events = [];

  const result = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js. The files use CRLF; keep them that way.' },
    repoPath: '/repo/demo',
    config: baseConfig(dir, { repairRounds: 2 }),
    store,
    driver,
    providers: {
      orchestrator,
      judge: passingJudge(),
      coders: [coder('ollama:qwen3:8b', GOOD_MATH)],
      reviewer: crlfReviewer(),
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  assert.equal(result.verdict, 'FAIL', 'a kept criterion still fails, as it does today');
  assert.equal(events.filter((e) => e.type === 'plan.criterion').length, 0, 'the plan is untouched');
  assert.equal(events.filter((e) => e.type === 'repair.stuck').length, 1);
  assert.equal(
    orchestrator.calls.filter((c) => c.jsonSchema?.name === 'kolm_plan_audit').length,
    1,
    'at most one audit per unit: a wrong keep cannot loop',
  );
  // Round 2 has nothing to say to a candidate that already failed twice the
  // same way, so it is not asked again.
  assert.deepEqual(result.candidates.map((c) => c.candidateId).sort(), ['c0a0', 'c0a1']);
  assert.equal(events.filter((e) => e.type === 'repair').length, 1, 'the second round was not spent');
});

test('a rewritten criterion is the one the reviewer is asked about next', options, async () => {
  const { store, dir, driver } = fixture('audit-rewrite');
  const rewritten = 'src/math.js keeps the line endings it already has.';
  const orchestrator = plannerAudit([
    { index: 1, verdict: 'rewrite', reason: 'line endings are not checkable from this diff', rewrite: rewritten },
  ]);
  const reviewerProvider = crlfReviewer();
  const events = [];

  const result = await runTask({
    task: { text: 'add a function add(a,b) to src/math.js. The files use CRLF; keep them that way.' },
    repoPath: '/repo/demo',
    config: baseConfig(dir, { repairRounds: 2 }),
    store,
    driver,
    providers: {
      orchestrator,
      judge: passingJudge(),
      coders: [coder('ollama:qwen3:8b', GOOD_MATH)],
      reviewer: reviewerProvider,
    },
    bands: BANDS,
    rubric: RUBRIC,
    onEvent: (evt) => events.push(evt),
  });

  const amended = events.filter((e) => e.type === 'plan.criterion');
  assert.equal(amended.length, 1);
  assert.equal(amended[0].payload.verdict, 'rewrite');
  assert.equal(amended[0].payload.to, rewritten);
  assert.equal(result.verdict, 'PASS');

  // The plan the reviewer was handed after the audit carries the new wording
  // and no longer carries the old one.
  const last = String(reviewerProvider.calls.at(-1).messages[0].content);
  assert.deepEqual(
    criteriaInPrompt(last).map((c) => c.text),
    [PLAN_JSON.acceptance[0], rewritten],
    'the reviewer is asked about the rewritten criterion',
  );
  assert.ok(!last.includes('no bare LF introduced'), 'the old wording is gone from the plan too');
});

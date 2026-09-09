// Owner C — the confidence cascade end to end (spec 5.6): gating, resumability,
// worker bounds, abort, and the four stages' own helpers.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { STAGES, runCascade } from '../src/ladder/stages/index.js';
import { globToRegExp, matchGlobs, shannonEntropy, yamlSanity } from '../src/ladder/stages/l0-meta.js';
import { parseCoverage } from '../src/ladder/stages/l2-dynamic.js';
import { capText, priorSummary, REVIEW_SCHEMA } from '../src/ladder/stages/l3-semantic.js';
import { openStore, sqliteAvailable, newRunId } from '../src/ladder/store.js';
import { loadBands } from '../src/ladder/bands.js';
import { loadRubric } from '../src/ladder/rubric.js';
import { LadderError } from '../src/ladder/errors.js';

// Node 20 has no node:sqlite; the cascade scenarios need a real store, so they
// skip rather than fail. The pure helper tests below always run.
const HAS_SQLITE = sqliteAvailable();
const options = HAS_SQLITE ? {} : { skip: 'node:sqlite is unavailable on this Node' };

const BANDS = loadBands();
const RUBRIC = loadRubric();
const RUBRIC_IDS = RUBRIC.items.map((item) => item.id);

// Key-shaped fixtures are assembled at runtime so public Git never stores a
// scanner-shaped literal (same convention as tests/sensitive-data.test.js).
const fakeKey = (...parts) => parts.join('');

// --------------------------------------------------------------- fixtures

const CLEAN_DIFF = [
  'diff --git a/src/util.js b/src/util.js',
  'index 1111111..2222222 100644',
  '--- a/src/util.js',
  '+++ b/src/util.js',
  '@@ -1,3 +1,6 @@',
  ' export const one = 1;',
  '+export function twice(n) {',
  '+  return n * 2;',
  '+}',
  ' export const two = 2;',
  ' export const three = 3;',
  '',
].join('\n');

const SECRET_DIFF = [
  'diff --git a/src/config.js b/src/config.js',
  'index 3333333..4444444 100644',
  '--- a/src/config.js',
  '+++ b/src/config.js',
  '@@ -1,2 +1,3 @@',
  ' export const region = "us-east-1";',
  `+const key = '${fakeKey('AKIA', 'IOSFODNN7EXAMPLE')}';`,
  ' export const stage = "prod";',
  '',
].join('\n');

const TODO_DIFF = [
  'diff --git a/src/util.js b/src/util.js',
  'index 5555555..6666666 100644',
  '--- a/src/util.js',
  '+++ b/src/util.js',
  '@@ -1,2 +1,3 @@',
  ' export const one = 1;',
  '+  // TODO: handle the empty case',
  ' export const two = 2;',
  '',
].join('\n');

const GREEN_TESTS = [
  '✔ adds two numbers together (1.2ms)',
  '✔ handles the empty list case (0.4ms)',
  'ℹ pass 2',
  'ℹ fail 0',
].join('\n');

const RED_TESTS = [
  '✔ adds two numbers together (1.2ms)',
  '✖ handles the empty list case (0.9ms)',
  'ℹ pass 1',
  'ℹ fail 1',
].join('\n');

const GREEN_BASELINE = Object.freeze({
  typecheck: { errors: [] },
  lint: { errors: [] },
  tests: { failing: [], exitCode: 0 },
});

// ----------------------------------------------------------------- fakes

function tempStore(label) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `kolm-cascade-${label}-`));
  const store = openStore(path.join(dir, 'ladder.db'));
  return {
    store,
    dir,
    cleanup() {
      try {
        store.close();
      } catch {
        // already closed is fine
      }
      try {
        fs.rmSync(dir, { recursive: true, force: true });
      } catch {
        // a lingering WAL handle on Windows is not a test failure
      }
    },
  };
}

/** Worktree stand-in with the surface L0..L3 actually touch. */
function fakeWorktree({ root = path.join(os.tmpdir(), 'kolm-fake-wt'), files = {}, exec } = {}) {
  const calls = { exec: [], readFile: [], diff: 0 };
  return {
    path: root,
    calls,
    async readFile(file) {
      calls.readFile.push(file);
      if (Object.prototype.hasOwnProperty.call(files, file)) return files[file];
      throw new Error(`no such file: ${file}`);
    },
    async listFiles() {
      return Object.keys(files);
    },
    async diff() {
      calls.diff += 1;
      return '';
    },
    async exec(cmd, opts = {}) {
      calls.exec.push(cmd);
      if (typeof exec === 'function') return exec(cmd, opts);
      return { stdout: '', stderr: '', exitCode: 0, durationMs: 1 };
    },
  };
}

/** Provider stand-in with owner A's chat() shape (never imported from fake.js). */
function fakeReviewer({ score = 0.95, confidence = 0.9, json, boom = false } = {}) {
  const reviewer = {
    id: 'fake:reviewer',
    provider: 'fake',
    model: 'fake-reviewer-1',
    calls: 0,
    lastRequest: null,
    async chat(req) {
      reviewer.calls += 1;
      reviewer.lastRequest = req;
      if (boom) throw new Error('the reviewer must not be called here');
      const body = json !== undefined
        ? json
        : {
          items: RUBRIC_IDS.map((id) => ({ id, score, evidence: `fake evidence for ${id}` })),
          overall: score,
          confidence,
          blocking: [],
        };
      return {
        json: body,
        text: JSON.stringify(body),
        provider: 'fake',
        model: 'fake-reviewer-1',
        usage: { input_tokens: 120, output_tokens: 40 },
        costUsd: 0.002,
        stopReason: 'end_turn',
      };
    },
  };
  return reviewer;
}

/**
 * Wrap every stage's run with a call counter, optionally substituting one.
 * @param {Record<string, Function>} [overrides] stage id -> replacement run
 */
function instrument(overrides = {}) {
  const counts = {};
  const originals = new Map();
  for (const stage of STAGES) {
    counts[stage.id] = 0;
    originals.set(stage, stage.run);
    const original = stage.run;
    const override = overrides[stage.id];
    stage.run = async (args) => {
      counts[stage.id] += 1;
      if (override) return override(args, original);
      return original(args);
    };
  }
  return {
    counts,
    restore() {
      for (const [stage, run] of originals) stage.run = run;
    },
  };
}

function makeCandidates(count, { diffText, wt }) {
  const out = [];
  for (let i = 0; i < count; i += 1) {
    out.push({ candidateId: `c${i}a0`, wt: typeof wt === 'function' ? wt(i) : wt, diffText });
  }
  return out;
}

function cascadeCtx({ runId, candidates, store, commands = {}, providers = {}, baseline, mode, signal }) {
  return {
    runId,
    candidates,
    store,
    signal,
    mode,
    config: { stageTimeoutMs: 5000 },
    plan: { prompt: 'add a small helper to src/util.js', forbidden_paths: ['docs/**'] },
    brief: { sha: 'base0000', tree: ['src/util.js'], commands, languages: { '.js': 3 } },
    baseline: baseline || GREEN_BASELINE,
    commands,
    bands: BANDS,
    rubric: RUBRIC,
    providers,
  };
}

function seedRun(store, runId) {
  store.createRun({ id: runId, task: { prompt: 'cascade test' }, repoPath: 'C:/repo', baseSha: 'base0000', driver: 'fake', mode: 'worktree' });
}

// ------------------------------------------------------- stage table shape

test('every cascade stage exports the contract the driver relies on', () => {
  assert.deepEqual(STAGES.map((s) => s.id), ['l0-meta', 'l1-static', 'l2-dynamic', 'l3-semantic']);
  assert.deepEqual(STAGES.map((s) => s.level), [0, 1, 2, 3]);
  for (const stage of STAGES) {
    assert.equal(typeof stage.version, 'number');
    assert.ok(stage.workers >= 1, `${stage.id} needs a positive worker bound`);
    assert.equal(typeof stage.gate, 'function');
    assert.equal(typeof stage.run, 'function');
  }
  assert.equal(STAGES[0].gate('bad'), true, 'L0 is the floor: everything runs');
  assert.equal(STAGES[1].gate('bad'), false);
  assert.equal(STAGES[1].gate('unsure'), true);
  assert.equal(STAGES[3].gate('skip'), true);
});

// ------------------------------------------------------------ scenario A

test('scenario A: a clean patch clears every rung and passes', options, async () => {
  const { store, cleanup } = tempStore('pass');
  const runId = newRunId();
  const reviewer = fakeReviewer();
  try {
    seedRun(store, runId);
    const wt = fakeWorktree({
      exec: async (cmd) => {
        if (/node --test/.test(cmd)) return { stdout: GREEN_TESTS, stderr: '', exitCode: 0 };
        return { stdout: '', stderr: '', exitCode: 0 };
      },
    });
    const commands = { test: 'node --test', typecheck: 'tsc --noEmit', lint: 'eslint .' };
    const { perCandidate } = await runCascade(cascadeCtx({
      runId,
      store,
      commands,
      providers: { reviewer },
      candidates: makeCandidates(1, { diffText: CLEAN_DIFF, wt }),
    }));

    const state = perCandidate.get('c0a0');
    assert.deepEqual(state.decisions, {
      'l0-meta': 'good',
      'l1-static': 'good',
      'l2-dynamic': 'good',
      'l3-semantic': 'good',
    });
    assert.equal(state.verdict.verdict, 'PASS');
    assert.equal(reviewer.calls, 1);
    assert.equal(reviewer.lastRequest.jsonSchema.name, REVIEW_SCHEMA.name);
    assert.equal(state.results['l3-semantic'].model, 'fake:reviewer');
    assert.ok(Math.abs(state.results['l3-semantic'].costUsd - 0.002) < 1e-9);

    // Every rung persisted a row and an event.
    for (const stage of STAGES) {
      const row = store.getResult(runId, 'c0a0', stage.id, stage.version);
      assert.ok(row, `${stage.id} wrote no result row`);
      assert.equal(row.decision, 'good');
    }
    const events = store.eventsSince(runId, 0).filter((e) => e.type === 'stage.result');
    assert.equal(events.length, 4);
    const cost = store.costSummary(runId);
    assert.ok(Math.abs(cost.total_usd - 0.002) < 1e-9);
    assert.equal(cost.by_stage['l3-semantic'].input_tokens, 120);
  } finally {
    cleanup();
  }
});

// ------------------------------------------------------------ scenario B

test('scenario B: a secret in the patch dies at L0 and nothing above it runs', options, async () => {
  const { store, cleanup } = tempStore('secret');
  const runId = newRunId();
  const reviewer = fakeReviewer({ boom: true });
  const probe = instrument();
  try {
    seedRun(store, runId);
    const { perCandidate } = await runCascade(cascadeCtx({
      runId,
      store,
      commands: { test: 'node --test', typecheck: 'tsc --noEmit', lint: 'eslint .' },
      providers: { reviewer },
      candidates: makeCandidates(1, { diffText: SECRET_DIFF, wt: fakeWorktree() }),
    }));

    const state = perCandidate.get('c0a0');
    assert.equal(state.decisions['l0-meta'], 'bad');
    assert.deepEqual(Object.keys(state.decisions), ['l0-meta']);
    assert.equal(state.verdict.verdict, 'FAIL');
    assert.equal(probe.counts['l0-meta'], 1);
    assert.equal(probe.counts['l1-static'], 0);
    assert.equal(probe.counts['l2-dynamic'], 0);
    assert.equal(probe.counts['l3-semantic'], 0);
    assert.equal(reviewer.calls, 0);

    const row = store.getResult(runId, 'c0a0', 'l0-meta', 1);
    assert.equal(row.artifacts.secrets[0].kind, 'aws-access-key');
    assert.ok(!JSON.stringify(row).includes(fakeKey('AKIA', 'IOSFODNN7EXAMPLE')), 'the stored row must redact the credential');
    assert.equal(store.getResult(runId, 'c0a0', 'l1-static', 1), null);
  } finally {
    probe.restore();
    cleanup();
  }
});

// ------------------------------------------------------------ scenario C

test('scenario C: a newly failing test fails at L2 and the fixed one is recorded', options, async () => {
  const { store, cleanup } = tempStore('regress');
  const runId = newRunId();
  const probe = instrument();
  try {
    seedRun(store, runId);
    const wt = fakeWorktree({
      exec: async (cmd) => {
        if (/node --test/.test(cmd)) return { stdout: RED_TESTS, stderr: '', exitCode: 1 };
        return { stdout: '', stderr: '', exitCode: 0 };
      },
    });
    const { perCandidate } = await runCascade(cascadeCtx({
      runId,
      store,
      commands: { test: 'node --test', typecheck: 'tsc --noEmit' },
      providers: { reviewer: fakeReviewer({ boom: true }) },
      baseline: {
        typecheck: { errors: [] },
        lint: { errors: [] },
        tests: { failing: ['legacy parser accepts tabs'], exitCode: 1 },
      },
      candidates: makeCandidates(1, { diffText: CLEAN_DIFF, wt }),
    }));

    const state = perCandidate.get('c0a0');
    assert.equal(state.decisions['l1-static'], 'good');
    assert.equal(state.decisions['l2-dynamic'], 'bad');
    assert.equal(state.decisions['l3-semantic'], undefined, 'L3 must not run after a confident bad');
    assert.equal(probe.counts['l3-semantic'], 0);
    assert.equal(state.verdict.verdict, 'FAIL');

    const row = store.getResult(runId, 'c0a0', 'l2-dynamic', 1);
    assert.deepEqual(row.artifacts.newlyFailing, ['handles the empty list case']);
    assert.deepEqual(row.artifacts.fixed, ['legacy parser accepts tabs']);
    assert.equal(row.metrics.newly_failing_tests, 1);
    assert.equal(row.metrics.fixed_tests, 1);
    assert.ok(row.reasons.some((r) => r.includes('legacy parser accepts tabs')));
  } finally {
    probe.restore();
    cleanup();
  }
});

// ------------------------------------------------------------ scenario D

test('scenario D: a TODO marker stays unsure through the cascade and reaches the judge', options, async () => {
  const { store, cleanup } = tempStore('unsure');
  const runId = newRunId();
  const reviewer = fakeReviewer({ score: 0.7, confidence: 0.6 });
  try {
    seedRun(store, runId);
    const { perCandidate } = await runCascade(cascadeCtx({
      runId,
      store,
      commands: {},
      providers: { reviewer },
      candidates: makeCandidates(1, { diffText: TODO_DIFF, wt: fakeWorktree() }),
    }));

    const state = perCandidate.get('c0a0');
    assert.equal(state.decisions['l0-meta'], 'unsure');
    assert.equal(state.decisions['l1-static'], 'skip');
    assert.equal(state.decisions['l2-dynamic'], 'skip');
    assert.equal(state.decisions['l3-semantic'], 'unsure');
    assert.equal(state.verdict.verdict, 'JUDGE');
    assert.deepEqual(state.verdict.unsure, ['l0-meta', 'l3-semantic']);

    const meta = store.getResult(runId, 'c0a0', 'l0-meta', 1);
    assert.equal(meta.metrics.todos, 1);
    assert.ok(meta.artifacts.todos[0].text.includes('TODO'));
    const semantic = store.getResult(runId, 'c0a0', 'l3-semantic', 1);
    assert.ok(Math.abs(semantic.metrics.rubric_score - 0.7) < 1e-9);
    assert.ok(Math.abs(semantic.metrics.reviewer_confidence - 0.6) < 1e-9);
    // No adapter retry happened; the field is present so an operator can tell
    // "no retry" from "an adapter that does not report one".
    assert.equal(semantic.artifacts.emptyRetry, null);
  } finally {
    cleanup();
  }
});

test('an adapter empty-on-length retry is recorded on the L3 result', options, async () => {
  const { store, cleanup } = tempStore('retry');
  const runId = newRunId();
  const reviewer = fakeReviewer({ score: 0.7, confidence: 0.6 });
  const plain = reviewer.chat.bind(reviewer);
  reviewer.chat = async (req) => ({
    ...(await plain(req)),
    meta: { emptyRetry: { from: 2000, to: 8000 }, reasoningChars: 5120 },
  });
  try {
    seedRun(store, runId);
    await runCascade(cascadeCtx({
      runId,
      store,
      commands: {},
      providers: { reviewer },
      candidates: makeCandidates(1, { diffText: TODO_DIFF, wt: fakeWorktree() }),
    }));
    const semantic = store.getResult(runId, 'c0a0', 'l3-semantic', 1);
    assert.deepEqual(semantic.artifacts.emptyRetry, { from: 2000, to: 8000 });
  } finally {
    cleanup();
  }
});

// ------------------------------------------------------------ scenario E

test('scenario E: missing tooling and no reviewer skip their rungs and still pass', options, async () => {
  const { store, cleanup } = tempStore('skips');
  const runId = newRunId();
  try {
    seedRun(store, runId);
    const wt = fakeWorktree();
    const { perCandidate } = await runCascade(cascadeCtx({
      runId,
      store,
      commands: { typecheck: 'tsc --noEmit', lint: null, test: null },
      providers: {},
      candidates: makeCandidates(1, { diffText: CLEAN_DIFF, wt }),
    }));

    const state = perCandidate.get('c0a0');
    assert.equal(state.decisions['l1-static'], 'good');
    assert.equal(state.decisions['l2-dynamic'], 'skip');
    assert.equal(state.decisions['l3-semantic'], 'skip');
    assert.equal(state.verdict.verdict, 'PASS');
    assert.deepEqual(state.verdict.skipped, ['l2-dynamic', 'l3-semantic']);
    assert.deepEqual(wt.calls.exec, ['tsc --noEmit'], 'no test command means no test process');

    const dynamic = store.getResult(runId, 'c0a0', 'l2-dynamic', 1);
    assert.ok(dynamic.reasons[0].includes('no test command'));
    const semantic = store.getResult(runId, 'c0a0', 'l3-semantic', 1);
    assert.ok(semantic.reasons[0].includes('no reviewer'));
  } finally {
    cleanup();
  }
});

// ------------------------------------------------------------ scenario F

test('scenario F: an existing result row is reused and its stage never re-runs', options, async () => {
  const { store, cleanup } = tempStore('resume');
  const runId = newRunId();
  const probe = instrument();
  try {
    seedRun(store, runId);
    // A row from an earlier, interrupted run. Its decision disagrees with what
    // L0 would say about this diff, which is what proves it was reused.
    store.putResult({
      runId,
      candidateId: 'c0a0',
      stage: 'l0-meta',
      version: 1,
      verdict: {
        stage: 'l0-meta',
        version: 1,
        decision: 'good',
        reasons: ['pre-seeded by an earlier interrupted run'],
        hits: [],
        metrics: {},
        artifacts: {},
        costUsd: 0,
        latencyMs: 0,
        model: null,
      },
    });

    const { perCandidate } = await runCascade(cascadeCtx({
      runId,
      store,
      commands: {},
      providers: {},
      candidates: makeCandidates(1, { diffText: SECRET_DIFF, wt: fakeWorktree() }),
    }));

    const state = perCandidate.get('c0a0');
    assert.equal(probe.counts['l0-meta'], 0, 'a cached row must not re-invoke the stage');
    assert.equal(state.decisions['l0-meta'], 'good');
    assert.deepEqual(state.results['l0-meta'].reasons, ['pre-seeded by an earlier interrupted run']);
    assert.equal(probe.counts['l1-static'], 1, 'the rungs above a cached row still run');
    assert.equal(state.verdict.verdict, 'PASS');

    const cached = store.eventsSince(runId, 0).filter((e) => e.type === 'stage.cached');
    assert.equal(cached.length, 1);
    assert.equal(cached[0].payload.stage, 'l0-meta');
  } finally {
    probe.restore();
    cleanup();
  }
});

// ------------------------------------------------------------ scenario G

test('scenario G: direct mode records L3 as skipped without calling the reviewer', options, async () => {
  const { store, cleanup } = tempStore('direct');
  const runId = newRunId();
  const reviewer = fakeReviewer({ boom: true });
  try {
    seedRun(store, runId);
    const { perCandidate } = await runCascade(cascadeCtx({
      runId,
      store,
      mode: 'direct',
      commands: {},
      providers: { reviewer },
      candidates: makeCandidates(1, { diffText: CLEAN_DIFF, wt: fakeWorktree() }),
    }));

    const state = perCandidate.get('c0a0');
    assert.equal(state.decisions['l3-semantic'], 'skip');
    assert.deepEqual(state.results['l3-semantic'].reasons, ['direct mode']);
    assert.equal(reviewer.calls, 0);
    assert.equal(state.verdict.verdict, 'PASS');
    assert.equal(store.getResult(runId, 'c0a0', 'l3-semantic', 1).decision, 'skip');
  } finally {
    cleanup();
  }
});

// ------------------------------------------------------------ scenario H

test('scenario H: a stage that throws becomes unsure and the cascade keeps going', options, async () => {
  const { store, cleanup } = tempStore('throws');
  const runId = newRunId();
  const probe = instrument({
    'l1-static': async () => {
      throw new Error('typecheck driver blew up');
    },
  });
  try {
    seedRun(store, runId);
    const wt = fakeWorktree({
      exec: async () => ({ stdout: GREEN_TESTS, stderr: '', exitCode: 0 }),
    });
    const { perCandidate } = await runCascade(cascadeCtx({
      runId,
      store,
      commands: { test: 'node --test', typecheck: 'tsc --noEmit' },
      providers: {},
      candidates: makeCandidates(1, { diffText: CLEAN_DIFF, wt }),
    }));

    const state = perCandidate.get('c0a0');
    assert.equal(state.decisions['l1-static'], 'unsure');
    assert.deepEqual(state.results['l1-static'].reasons, ['typecheck driver blew up']);
    assert.equal(state.results['l1-static'].artifacts.error, 'typecheck driver blew up');
    assert.equal(probe.counts['l2-dynamic'], 1, 'the rung above a thrown stage still runs');
    assert.equal(state.decisions['l2-dynamic'], 'good');
    assert.equal(state.verdict.verdict, 'JUDGE');
    assert.equal(store.getResult(runId, 'c0a0', 'l1-static', 1).decision, 'unsure');
  } finally {
    probe.restore();
    cleanup();
  }
});

// ------------------------------------------------------------ scenario I

test('scenario I: a level never exceeds its declared worker bound', options, async () => {
  const { store, cleanup } = tempStore('workers');
  const runId = newRunId();
  const l2Workers = STAGES.find((s) => s.id === 'l2-dynamic').workers;
  assert.equal(l2Workers, 2, 'this scenario is written against an L2 bound of 2');
  let inFlight = 0;
  let maxInFlight = 0;
  try {
    seedRun(store, runId);
    const exec = async () => {
      inFlight += 1;
      maxInFlight = Math.max(maxInFlight, inFlight);
      await new Promise((resolve) => setTimeout(resolve, 5));
      inFlight -= 1;
      return { stdout: GREEN_TESTS, stderr: '', exitCode: 0 };
    };
    const { perCandidate } = await runCascade(cascadeCtx({
      runId,
      store,
      // Only the test command is present, so L2's exec is the only one measured.
      commands: { test: 'node --test', typecheck: null, lint: null },
      providers: {},
      candidates: makeCandidates(5, { diffText: CLEAN_DIFF, wt: () => fakeWorktree({ exec }) }),
    }));

    assert.equal(perCandidate.size, 5);
    assert.ok(maxInFlight <= l2Workers, `observed ${maxInFlight} concurrent execs, bound is ${l2Workers}`);
    assert.ok(maxInFlight >= 2, 'the pool must actually run candidates in parallel');
    for (const [candidateId, state] of perCandidate) {
      assert.equal(state.decisions['l2-dynamic'], 'good', candidateId);
      assert.equal(state.verdict.verdict, 'PASS', candidateId);
    }
    assert.equal(store.resultsFor(runId).filter((r) => r.stage === 'l2-dynamic').length, 5);
  } finally {
    cleanup();
  }
});

// ------------------------------------------------------------ scenario J

test('scenario J: aborting mid-run throws aborted and keeps the finished rows', options, async () => {
  const { store, cleanup } = tempStore('abort');
  const runId = newRunId();
  const controller = new AbortController();
  try {
    seedRun(store, runId);
    const wt = fakeWorktree({
      exec: async (cmd) => {
        if (/tsc/.test(cmd)) controller.abort();
        return { stdout: '', stderr: '', exitCode: 0 };
      },
    });
    await assert.rejects(
      () => runCascade(cascadeCtx({
        runId,
        store,
        signal: controller.signal,
        commands: { test: 'node --test', typecheck: 'tsc --noEmit' },
        providers: { reviewer: fakeReviewer({ boom: true }) },
        candidates: makeCandidates(1, { diffText: CLEAN_DIFF, wt }),
      })),
      (err) => {
        assert.ok(err instanceof LadderError);
        assert.equal(err.code, 'aborted');
        assert.equal(err.detail.stage, 'l1-static');
        return true;
      },
    );

    assert.ok(store.getResult(runId, 'c0a0', 'l0-meta', 1), 'the finished rung keeps its row');
    assert.ok(store.getResult(runId, 'c0a0', 'l1-static', 1), 'the rung that completed before the check keeps its row');
    assert.equal(store.getResult(runId, 'c0a0', 'l2-dynamic', 1), null, 'nothing after the abort may be written');
  } finally {
    cleanup();
  }
});

// -------------------------------------------------------- stage helpers

test('globToRegExp and matchGlobs cover the forbidden path patterns', () => {
  assert.ok(globToRegExp('*.pem').test('key.pem'));
  assert.ok(!globToRegExp('*.pem').test('dir/key.pem'));
  assert.ok(globToRegExp('**/*.pem').test('deep/dir/key.pem'));
  assert.equal(matchGlobs('config/.env.local', ['.env*']), '.env*');
  assert.equal(matchGlobs('src/node_modules/x/index.js', ['**/node_modules/**']), '**/node_modules/**');
  assert.equal(matchGlobs('docs/plan.md', ['docs/**']), 'docs/**');
  assert.equal(matchGlobs('src\\win\\path.key', ['**/*.key']), '**/*.key');
  assert.equal(matchGlobs('src/app.js', ['docs/**', '*.pem']), null);
});

test('shannonEntropy stays under the secret threshold for hex digests', () => {
  assert.equal(shannonEntropy(''), 0);
  assert.equal(shannonEntropy('aaaaaaaa'), 0);
  // Pure hex tops out at 4 bits per character, below the 4.5 detector cutoff.
  assert.ok(shannonEntropy('0123456789abcdef0123456789abcdef') <= 4);
  assert.ok(shannonEntropy('Zx9_Qw2-Lm7+Pv4/Rt6=Nb1Kc8Jd5Hf3Ga0') > 4.5);
});

test('yamlSanity flags tab indentation and unbalanced quotes', () => {
  assert.deepEqual(yamlSanity('name: ok\nvalue: 2\n'), { errors: [], warnings: [] });
  const tabbed = yamlSanity('root:\n\tchild: 1\n');
  assert.equal(tabbed.errors.length, 1);
  assert.ok(tabbed.errors[0].includes('line 2'));
  const quoted = yamlSanity('name: "unterminated\n');
  assert.equal(quoted.errors.length, 0);
  assert.equal(quoted.warnings.length, 1);
  assert.equal(yamlSanity('# a "comment\n').warnings.length, 0);
});

test('parseCoverage reads json-summary and the node coverage table', () => {
  assert.deepEqual(parseCoverage('{"total":{"lines":{"pct":84.5}}}'), { value: 0.845, source: 'json-summary' });
  assert.deepEqual(parseCoverage('noise before {"total":{"lines":{"pct":100}}}'), { value: 1, source: 'json-summary' });
  const table = ['# start of coverage report', '# all files      | 72.31 | 55.00 |', '# end of coverage report'].join('\n');
  assert.ok(Math.abs(parseCoverage(table).value - 0.7231) < 1e-9);
  assert.equal(parseCoverage(table).source, 'node-coverage');
  assert.deepEqual(parseCoverage('nothing useful here'), { value: null, source: null });
  assert.deepEqual(parseCoverage(''), { value: null, source: null });
});

test('capText truncates on a byte budget and says how much it dropped', () => {
  assert.equal(capText('short', 1024), 'short');
  assert.equal(capText('', 10), '');
  const long = 'x'.repeat(5000);
  const capped = capText(long, 100);
  assert.ok(Buffer.byteLength(capped, 'utf8') < 200);
  assert.ok(capped.includes('[truncated '));
  // Multi-byte input must not be cut into an invalid sequence.
  const wide = '\u00e9'.repeat(200);
  assert.ok(!capText(wide, 64).includes('\uFFFD'));
});

test('priorSummary renders both the plain and the stored result shapes', () => {
  const plain = priorSummary({ 'l0-meta': { decision: 'good', reasons: ['small patch'] } });
  assert.ok(plain.includes('l0-meta: good'));
  assert.ok(plain.includes('small patch'));
  const stored = priorSummary({ 'l2-dynamic': { verdict: { decision: 'unsure', reasons: ['flaky suite'] } } });
  assert.ok(stored.includes('l2-dynamic: unsure'));
  assert.ok(stored.includes('flaky suite'));
  assert.ok(priorSummary({}).includes('no cheap-rung results'));
  assert.ok(priorSummary(null).includes('no cheap-rung results'));
});

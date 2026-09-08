// Owner C — sqlite state store round trip (spec 5.1).
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { openStore, sqliteAvailable, newRunId, candidateIdFor, SCHEMA_VERSION } from '../src/ladder/store.js';

// Node 20 has no node:sqlite; the whole file skips rather than failing CI.
const HAS_SQLITE = sqliteAvailable();
const options = HAS_SQLITE ? {} : { skip: 'node:sqlite is unavailable on this Node' };

function tempDb(label) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `kolm-store-${label}-`));
  return { dir, file: path.join(dir, 'nested', 'ladder.db') };
}

function cleanup(dir) {
  try {
    fs.rmSync(dir, { recursive: true, force: true });
  } catch {
    // a still-open WAL handle on Windows is not a test failure
  }
}

test('newRunId matches the r_yyyymmdd_hex8 identifier format', options, () => {
  const id = newRunId(new Date(Date.UTC(2026, 0, 5)));
  assert.match(id, /^r_\d{8}_[0-9a-f]{8}$/);
  assert.ok(id.startsWith('r_20260105_'));
  assert.notEqual(newRunId(), newRunId());
  assert.equal(candidateIdFor(3, 1), 'c3a1');
  assert.equal(candidateIdFor(0), 'c0a0');
});

test('openStore creates parent directories and records the schema version', options, () => {
  const { dir, file } = tempDb('open');
  const store = openStore(file);
  try {
    assert.ok(fs.existsSync(file));
    assert.equal(store.path, path.resolve(file));
    const meta = store.db.prepare('SELECT value FROM schema_meta WHERE key = ?').get('schema_version');
    assert.equal(meta.value, String(SCHEMA_VERSION));
    const tables = store.db
      .prepare("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
      .all()
      .map((r) => r.name);
    for (const expected of ['candidates', 'context_cache', 'cost_ledger', 'events', 'judgments', 'results', 'runs', 'schema_meta']) {
      assert.ok(tables.includes(expected), `missing table ${expected}`);
    }
    const journal = store.db.prepare('PRAGMA journal_mode').get();
    assert.equal(String(journal.journal_mode).toLowerCase(), 'wal');
  } finally {
    store.close();
    cleanup(dir);
  }
});

test('runs round trip through create update get and list', options, () => {
  const { dir, file } = tempDb('runs');
  const store = openStore(file);
  try {
    const run = store.createRun({
      task: { prompt: 'add a flag', forbidden_paths: ['.env'] },
      repoPath: '/tmp/repo',
      baseSha: 'aaaa1111',
      driver: 'local',
      mode: 'cascade',
      config: { n: 4 },
    });
    assert.match(run.id, /^r_\d{8}_[0-9a-f]{8}$/);
    assert.equal(run.status, 'running');
    assert.deepEqual(run.task, { prompt: 'add a flag', forbidden_paths: ['.env'] });
    assert.deepEqual(run.config, { n: 4 });
    assert.equal(run.total_cost_usd, 0);

    const updated = store.updateRun(run.id, {
      status: 'done',
      winnerCandidate: 'c1a0',
      totalCostUsd: 1.25,
      receipt: { verdict: 'PASS' },
      nonsense_key: 'ignored',
    });
    assert.equal(updated.status, 'done');
    assert.equal(updated.winner_candidate, 'c1a0');
    assert.equal(updated.total_cost_usd, 1.25);
    assert.deepEqual(updated.receipt, { verdict: 'PASS' });
    assert.ok(updated.updated_at >= run.updated_at);
    assert.ok(!('nonsense_key' in updated));

    const other = store.createRun({ status: 'running' });
    assert.equal(store.listRuns().length, 2);
    assert.deepEqual(store.listRuns({ status: 'done' }).map((r) => r.id), [run.id]);
    assert.equal(store.listRuns({ limit: 1 }).length, 1);
    assert.equal(store.getRun(other.id).status, 'running');
    assert.equal(store.getRun('r_nope'), null);
  } finally {
    store.close();
    cleanup(dir);
  }
});

test('upsertCandidate updates in place and hashes the diff', options, () => {
  const { dir, file } = tempDb('cands');
  const store = openStore(file);
  try {
    const run = store.createRun({});
    const first = store.upsertCandidate({
      runId: run.id,
      candidateId: 'c0a0',
      idx: 0,
      attempt: 0,
      generatorRef: 'anthropic:claude',
      worktree: { id: 'wt0', path: '/tmp/wt0' },
      status: 'generated',
      diffText: 'diff --git a/x b/x\n',
      summary: 'first pass',
    });
    assert.equal(first.candidate_id, 'c0a0');
    assert.equal(first.idx, 0);
    assert.deepEqual(first.worktree, { id: 'wt0', path: '/tmp/wt0' });
    assert.match(first.diff_sha256, /^[0-9a-f]{64}$/);

    const second = store.upsertCandidate({
      runId: run.id,
      candidateId: 'c0a0',
      status: 'verified',
      diffText: 'diff --git a/y b/y\n',
    });
    assert.equal(second.status, 'verified');
    assert.notEqual(second.diff_sha256, first.diff_sha256);
    assert.equal(second.idx, 0, 'omitted fields are preserved');
    assert.equal(second.generator_ref, 'anthropic:claude');

    store.upsertCandidate({ runId: run.id, candidateId: 'c1a0', idx: 1 });
    store.upsertCandidate({ runId: run.id, candidateId: 'c0a1', idx: 0, attempt: 1 });
    const listed = store.listCandidates(run.id);
    assert.equal(listed.length, 3, 'the upsert did not duplicate c0a0');
    assert.deepEqual(listed.map((c) => c.candidate_id), ['c0a0', 'c0a1', 'c1a0']);
    assert.equal(store.getCandidate(run.id, 'c1a0').idx, 1);
    assert.equal(store.getCandidate(run.id, 'missing'), null);
    assert.throws(() => store.upsertCandidate({ runId: run.id }), (err) => err.code === 'bad-arg');
  } finally {
    store.close();
    cleanup(dir);
  }
});

test('latestResults keeps the highest version of each stage', options, () => {
  const { dir, file } = tempDb('results');
  const store = openStore(file);
  try {
    const run = store.createRun({});
    store.putResult({ runId: run.id, candidateId: 'c0a0', stage: 'l0-meta', version: 1, verdict: { decision: 'good', reasons: ['v1'] } });
    store.putResult({ runId: run.id, candidateId: 'c0a0', stage: 'l0-meta', version: 2, verdict: { decision: 'unsure', reasons: ['v2'] } });
    store.putResult({ runId: run.id, candidateId: 'c0a0', stage: 'l1-static', version: 1, verdict: { decision: 'good', reasons: ['static'] } });
    store.putResult({ runId: run.id, candidateId: 'c1a0', stage: 'l0-meta', version: 1, verdict: { decision: 'bad', reasons: ['secret'] } });

    assert.deepEqual(store.getResult(run.id, 'c0a0', 'l0-meta', 1).reasons, ['v1']);
    assert.equal(store.getResult(run.id, 'c0a0', 'l0-meta', 9), null);

    const latest = store.latestResults(run.id, 'c0a0');
    assert.deepEqual(Object.keys(latest).sort(), ['l0-meta', 'l1-static']);
    assert.equal(latest['l0-meta'].version, 2);
    assert.equal(latest['l0-meta'].verdict.decision, 'unsure');
    assert.equal(latest['l1-static'].version, 1);

    // INSERT OR REPLACE on the same (run, candidate, stage, version)
    store.putResult({ runId: run.id, candidateId: 'c0a0', stage: 'l0-meta', version: 2, verdict: { decision: 'good', reasons: ['rerun'] } });
    assert.equal(store.latestResults(run.id, 'c0a0')['l0-meta'].verdict.decision, 'good');

    const all = store.resultsFor(run.id);
    assert.equal(all.length, 4);
    assert.equal(all.filter((r) => r.candidate_id === 'c1a0').length, 1);
    assert.throws(() => store.putResult({ runId: run.id, candidateId: 'c0a0' }), (err) => err.code === 'bad-arg');
  } finally {
    store.close();
    cleanup(dir);
  }
});

test('results and cache survive closing and reopening the database', options, () => {
  const { dir, file } = tempDb('reopen');
  const first = openStore(file);
  let runId;
  try {
    const run = first.createRun({ baseSha: 'sha-1' });
    runId = run.id;
    first.putResult({ runId, candidateId: 'c0a0', stage: 'l2-dynamic', version: 1, verdict: { decision: 'good', metrics: { newly_failing_tests: 0 } } });
    first.cachePut('/repo', 'sha-1', 'brief', { tree: ['a.js'], bytes: 12 });
    first.emit(runId, 'run.start', { n: 1 });
  } finally {
    first.close();
  }
  const second = openStore(file);
  try {
    assert.equal(second.getRun(runId).base_sha, 'sha-1');
    assert.equal(second.getResult(runId, 'c0a0', 'l2-dynamic', 1).metrics.newly_failing_tests, 0);
    assert.deepEqual(second.cacheGet('/repo', 'sha-1', 'brief'), { tree: ['a.js'], bytes: 12 });
    assert.equal(second.eventsSince(runId).length, 1);
  } finally {
    second.close();
    cleanup(dir);
  }
});

test('judgments are appended and filtered per candidate', options, () => {
  const { dir, file } = tempDb('judge');
  const store = openStore(file);
  try {
    const run = store.createRun({});
    store.putJudgment({ runId: run.id, candidateId: 'c0a0', kind: 'l4-judge', verdict: { verdict: 'PASS', score: 0.9 } });
    store.putJudgment({ runId: run.id, candidateId: 'c1a0', kind: 'l4-judge', verdict: { verdict: 'FAIL' } });
    store.putJudgment({ runId: run.id, candidateId: 'c0a0', kind: 'tiebreak', verdict: { winner: 'c0a0' } });
    assert.equal(store.listJudgments(run.id).length, 3);
    const forCandidate = store.listJudgments(run.id, 'c0a0');
    assert.equal(forCandidate.length, 2);
    assert.equal(forCandidate[0].verdict.score, 0.9);
    assert.equal(forCandidate[1].kind, 'tiebreak');
    assert.ok(forCandidate[0].id < forCandidate[1].id, 'judgments keep insertion order');
  } finally {
    store.close();
    cleanup(dir);
  }
});

test('context cache stores and replaces payloads per repo sha and kind', options, () => {
  const { dir, file } = tempDb('cache');
  const store = openStore(file);
  try {
    assert.equal(store.cacheGet('/repo', 'sha', 'brief'), null);
    store.cachePut('/repo', 'sha', 'brief', { v: 1 });
    store.cachePut('/repo', 'sha', 'baseline', { v: 2 });
    store.cachePut('/other', 'sha', 'brief', { v: 3 });
    assert.deepEqual(store.cacheGet('/repo', 'sha', 'brief'), { v: 1 });
    assert.deepEqual(store.cacheGet('/repo', 'sha', 'baseline'), { v: 2 });
    assert.deepEqual(store.cacheGet('/other', 'sha', 'brief'), { v: 3 });
    store.cachePut('/repo', 'sha', 'brief', { v: 9 });
    assert.deepEqual(store.cacheGet('/repo', 'sha', 'brief'), { v: 9 });
    assert.equal(store.db.prepare('SELECT COUNT(*) AS n FROM context_cache').get().n, 3);
  } finally {
    store.close();
    cleanup(dir);
  }
});

test('eventsSince pages forward by monotonic event id', options, () => {
  const { dir, file } = tempDb('events');
  const store = openStore(file);
  try {
    const run = store.createRun({});
    const other = store.createRun({});
    for (let i = 0; i < 7; i += 1) store.emit(run.id, 'stage.result', { i });
    store.emit(other.id, 'stage.result', { i: 'other' });

    const first = store.eventsSince(run.id, 0, 3);
    assert.equal(first.length, 3);
    assert.deepEqual(first.map((e) => e.payload.i), [0, 1, 2]);
    assert.ok(first[0].ts.endsWith('Z'));
    assert.equal(first[0].type, 'stage.result');

    const second = store.eventsSince(run.id, first[first.length - 1].id, 3);
    assert.deepEqual(second.map((e) => e.payload.i), [3, 4, 5]);
    const third = store.eventsSince(run.id, second[second.length - 1].id, 3);
    assert.deepEqual(third.map((e) => e.payload.i), [6]);
    assert.equal(store.eventsSince(run.id, third[0].id, 3).length, 0);
    assert.equal(store.eventsSince(run.id).length, 7, 'the other run is not mixed in');
  } finally {
    store.close();
    cleanup(dir);
  }
});

test('costSummary sums the ledger by model and by stage', options, () => {
  const { dir, file } = tempDb('cost');
  const store = openStore(file);
  try {
    const run = store.createRun({});
    store.addCost({ runId: run.id, candidateId: 'c0a0', stage: 'l3-semantic', provider: 'anthropic', model: 'haiku', inputTokens: 1000, outputTokens: 200, cacheRead: 50, costUsd: 0.01 });
    store.addCost({ runId: run.id, candidateId: 'c1a0', stage: 'l3-semantic', provider: 'anthropic', model: 'haiku', inputTokens: 2000, outputTokens: 300, cacheRead: 0, costUsd: 0.02 });
    store.addCost({ runId: run.id, candidateId: 'c0a0', stage: 'l4-judge', provider: 'openai', model: 'o4', inputTokens: 500, outputTokens: 100, costUsd: 0.05 });
    store.addCost({ runId: 'r_other', stage: 'l3-semantic', provider: 'anthropic', model: 'haiku', costUsd: 9 });

    const summary = store.costSummary(run.id);
    assert.ok(Math.abs(summary.total_usd - 0.08) < 1e-9);
    const haiku = summary.by_model['anthropic:haiku'];
    assert.equal(haiku.input_tokens, 3000);
    assert.equal(haiku.output_tokens, 500);
    assert.equal(haiku.cache_read, 50);
    assert.equal(haiku.calls, 2);
    assert.ok(Math.abs(haiku.cost_usd - 0.03) < 1e-9);
    assert.equal(summary.by_model['openai:o4'].calls, 1);
    assert.ok(Math.abs(summary.by_stage['l3-semantic'].cost_usd - 0.03) < 1e-9);
    assert.ok(Math.abs(summary.by_stage['l4-judge'].cost_usd - 0.05) < 1e-9);
    assert.equal(store.costSummary('r_missing').total_usd, 0);
  } finally {
    store.close();
    cleanup(dir);
  }
});

test('store funnel reads the latest result rows and the judgment table', options, () => {
  const { dir, file } = tempDb('funnel');
  const store = openStore(file);
  try {
    const run = store.createRun({});
    const put = (candidateId, stage, decision, version = 1) =>
      store.putResult({ runId: run.id, candidateId, stage, version, verdict: { decision, reasons: [] } });

    put('c0a0', 'l0-meta', 'good');
    put('c0a0', 'l1-static', 'good');
    put('c1a0', 'l0-meta', 'bad');
    put('c2a0', 'l0-meta', 'good');
    put('c2a0', 'l1-static', 'unsure');
    // A re-run at a higher version must win over the stale row.
    put('c3a0', 'l0-meta', 'bad', 1);
    put('c3a0', 'l0-meta', 'good', 2);
    store.putJudgment({ runId: run.id, candidateId: 'c2a0', kind: 'l4-judge', verdict: { verdict: 'PASS' } });

    const f = store.funnel({ runIds: [run.id] });
    assert.equal(f.candidates, 4);
    assert.equal(f.runs, 1);
    assert.deepEqual(f.verdicts, { PASS: 2, FAIL: 1, JUDGE: 1 });
    assert.equal(f.hard_fail, 1);
    assert.equal(f.judged, 1);
    assert.equal(f.cleared_cheaply, 0.75);
    assert.deepEqual(f.stages['l0-meta'], { good: 3, unsure: 0, bad: 1, skip: 0 });

    const scoped = store.funnel({ runIds: ['r_nothing'] });
    assert.equal(scoped.candidates, 0);
    const future = store.funnel({ since: '2999-01-01T00:00:00.000Z' });
    assert.equal(future.candidates, 0);
    assert.equal(store.funnel({}).candidates, 4);
  } finally {
    store.close();
    cleanup(dir);
  }
});

test('a closed store refuses further calls', options, () => {
  const { dir, file } = tempDb('closed');
  const store = openStore(file);
  store.createRun({});
  store.close();
  store.close(); // idempotent
  assert.throws(() => store.listRuns(), (err) => err.code === 'store-closed');
  cleanup(dir);
});

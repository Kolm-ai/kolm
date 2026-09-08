// kolm code — viewer server + single-file viewer document (owner E, spec 7.2).
//
// Every request here goes to a server this file started on 127.0.0.1:0, so the
// suite stays offline as the testing policy requires.

import { after, before, describe, it } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { openStore, sqliteAvailable } from '../src/ladder/index.js';
import { createViewerServer, DEFAULT_INDEX_HTML, VIEWER_CSP, formatUrl } from '../src/ladder/viewer/server.js';

const RUN_ID = 'r_20260101_deadbeef';
const STAGES = ['l0-meta', 'l1-static', 'l2-dynamic', 'l3-semantic'];
const HAS_SQLITE = sqliteAvailable();

/**
 * Read an event-stream response until `needle` shows up or the budget expires.
 * Each read races a timer so a quiet stream cannot block the suite.
 * @param {Response} res
 * @param {string} needle
 * @param {number} budgetMs
 * @returns {Promise<string>}
 */
async function readUntil(res, needle, budgetMs) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  const deadline = Date.now() + budgetMs;
  let buffer = '';
  while (Date.now() < deadline && !buffer.includes(needle)) {
    const remaining = Math.max(1, deadline - Date.now());
    let timer;
    const timeout = new Promise((resolve) => {
      timer = setTimeout(() => resolve(null), remaining);
    });
    const chunk = await Promise.race([reader.read(), timeout]);
    clearTimeout(timer);
    if (!chunk || chunk.done) break;
    buffer += decoder.decode(chunk.value, { stream: true });
  }
  return buffer;
}

/** A store stub good enough for the constructor guards, with no sqlite. */
function stubStore() {
  return {
    listRuns: () => [],
    getRun: () => null,
    funnel: () => ({ stages: {} }),
    eventsSince: () => [],
  };
}

function seed(store) {
  store.createRun({
    id: RUN_ID,
    task: { text: 'make the retry loop stop hammering the queue', title: 'fix retry loop' },
    repoPath: '/repo',
    baseSha: 'a'.repeat(40),
    driver: 'fake',
    mode: 'cascade',
    config: { candidates: 2 },
  });
  for (const [idx, candidateId] of ['c0a0', 'c1a0'].entries()) {
    store.upsertCandidate({
      runId: RUN_ID,
      candidateId,
      idx,
      attempt: 0,
      generatorRef: idx === 0 ? 'fake:coder-one' : 'fake:coder-two',
      status: 'evaluated',
      diffText: `diff --git a/x b/x\n+line ${idx}\n`,
      summary: `candidate ${candidateId}`,
    });
    for (const [version, stage] of STAGES.entries()) {
      store.putResult({
        runId: RUN_ID,
        candidateId,
        stage,
        version: 1,
        verdict: {
          decision: idx === 0 ? 'good' : 'unsure',
          reasons: [`${stage} decided for ${candidateId}`],
          metrics: { checked: version + 1 },
          hits: [],
        },
      });
    }
  }
  store.putJudgment({
    runId: RUN_ID,
    candidateId: 'c1a0',
    kind: 'l4-judge',
    verdict: { decision: 'good', reasons: ['the residue reads correct'] },
  });
  store.addCost({
    runId: RUN_ID,
    candidateId: 'c0a0',
    stage: 'l3-semantic',
    provider: 'fake',
    model: 'coder-one',
    inputTokens: 120,
    outputTokens: 40,
    costUsd: 0.0031,
  });
  store.emit(RUN_ID, 'run.started', { runId: RUN_ID });
  store.emit(RUN_ID, 'stage.finished', { runId: RUN_ID, stage: 'l0-meta', candidateId: 'c0a0', decision: 'good' });
  store.updateRun(RUN_ID, {
    status: 'done',
    winnerCandidate: 'c0a0',
    totalCostUsd: 0.0031,
    receipt: { schema: 'kolm-code-receipt-v1', run_id: RUN_ID, verdict: 'PASS' },
  });
}

describe('kolm code viewer server', { skip: HAS_SQLITE ? false : 'node:sqlite unavailable on this Node' }, () => {
  let dir;
  let store;
  let viewer;
  let base;

  before(async () => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kolm-viewer-'));
    store = openStore(path.join(dir, 'ladder.db'));
    seed(store);
    viewer = createViewerServer({
      store,
      config: { dbPath: store.path },
      port: 0,
      verifyReceipt: (receipt) => ({ ok: Boolean(receipt), kid: 'kolm-test-kid', reason: 'stubbed verifier' }),
      evalRuns: () => ({ cascade: { runs: 1 }, direct: { runs: 0 } }),
      arming: () => ({ rungs: [{ id: 'driver', armed: true, reason: 'fake' }] }),
    });
    const listening = await viewer.listen();
    base = listening.url;
  });

  after(async () => {
    if (viewer) await viewer.close();
    if (store) store.close();
    if (dir) fs.rmSync(dir, { recursive: true, force: true });
  });

  it('serves the viewer document with a strict content policy', async () => {
    const res = await fetch(`${base}/`);
    assert.equal(res.status, 200);
    assert.match(res.headers.get('content-type'), /text\/html/);
    assert.equal(res.headers.get('cache-control'), 'no-store');
    assert.equal(res.headers.get('content-security-policy'), VIEWER_CSP);
    const html = await res.text();
    assert.ok(html.includes('kolm code'), 'the document names the product');
  });

  it('lists stored runs newest first with the limit applied', async () => {
    const res = await fetch(`${base}/api/runs?limit=5`);
    assert.equal(res.status, 200);
    assert.equal(res.headers.get('content-type'), 'application/json; charset=utf-8');
    assert.equal(res.headers.get('cache-control'), 'no-store');
    const body = await res.json();
    assert.equal(body.runs.length, 1);
    assert.equal(body.runs[0].id, RUN_ID);
    assert.equal(body.runs[0].winner_candidate, 'c0a0');
  });

  it('returns one run with candidates, results, costs and judgments', async () => {
    const res = await fetch(`${base}/api/runs/${RUN_ID}`);
    assert.equal(res.status, 200);
    const body = await res.json();
    assert.equal(body.run.id, RUN_ID);
    assert.equal(body.candidates.length, 2);
    const first = body.candidates[0];
    assert.equal(first.candidate_id, 'c0a0');
    assert.equal(first.diff_text, undefined, 'the diff blob stays out of the payload');
    assert.ok(first.diff_bytes > 0);
    for (const stage of STAGES) {
      assert.equal(first.results[stage].verdict.decision, 'good');
    }
    assert.equal(body.judgments.length, 1);
    assert.ok(body.cost.total_usd > 0);
    assert.equal(body.funnel.candidates, 2);
  });

  it('returns events after a cursor for one run', async () => {
    const all = await (await fetch(`${base}/api/runs/${RUN_ID}/events`)).json();
    assert.ok(all.events.length >= 2);
    const firstId = all.events[0].id;
    const rest = await (await fetch(`${base}/api/runs/${RUN_ID}/events?after=${firstId}`)).json();
    assert.equal(rest.events.length, all.events.length - 1);
    assert.ok(rest.events.every((event) => event.id > firstId));
    assert.equal(typeof all.events[0].ts, 'string');
  });

  it('streams newly stored events over server sent events', async () => {
    const controller = new AbortController();
    const res = await fetch(`${base}/api/runs/${RUN_ID}/stream`, { signal: controller.signal });
    assert.equal(res.status, 200);
    assert.match(res.headers.get('content-type'), /text\/event-stream/);

    const eventId = store.emit(RUN_ID, 'stage.finished', {
      runId: RUN_ID,
      stage: 'l2-dynamic',
      candidateId: 'c0a0',
      decision: 'good',
    });
    const buffer = await readUntil(res, 'l2-dynamic', 4000);
    controller.abort();

    assert.ok(buffer.includes(`id: ${eventId}`), `stream carried the event id; got: ${buffer.slice(0, 300)}`);
    assert.ok(buffer.includes('l2-dynamic'), 'stream carried the new event payload');
    assert.ok(buffer.includes('run.started'), 'an opening stream replays the run history');
  });

  it('replays only what follows the last event id header', async () => {
    const before = await (await fetch(`${base}/api/runs/${RUN_ID}/events`)).json();
    const highest = before.events[before.events.length - 1].id;

    const controller = new AbortController();
    const res = await fetch(`${base}/api/runs/${RUN_ID}/stream`, {
      headers: { 'last-event-id': String(highest) },
      signal: controller.signal,
    });
    const eventId = store.emit(RUN_ID, 'run.finished', { runId: RUN_ID, verdict: 'PASS' });
    const buffer = await readUntil(res, 'run.finished', 4000);
    controller.abort();

    assert.ok(buffer.includes(`id: ${eventId}`));
    assert.ok(!buffer.includes('run.started'), 'events at or below the cursor are not replayed');
  });

  it('serves the funnel and the eval summary', async () => {
    const funnel = await (await fetch(`${base}/api/funnel`)).json();
    assert.equal(funnel.funnel.candidates, 2);
    assert.equal(funnel.funnel.stages['l0-meta'].good, 1);

    const evaluation = await (await fetch(`${base}/api/eval`)).json();
    assert.equal(evaluation.available, true);
    assert.equal(evaluation.eval.cascade.runs, 1);
  });

  it('serves the receipt together with its verification result', async () => {
    const res = await fetch(`${base}/api/receipt/${RUN_ID}`);
    assert.equal(res.status, 200);
    const body = await res.json();
    assert.equal(body.receipt.run_id, RUN_ID);
    assert.equal(body.verification.ok, true);
    assert.equal(body.verification.kid, 'kolm-test-kid');
  });

  it('answers unknown paths and unknown runs with json errors', async () => {
    const missingPath = await fetch(`${base}/api/nothing-here`);
    assert.equal(missingPath.status, 404);
    assert.equal(missingPath.headers.get('content-type'), 'application/json; charset=utf-8');
    assert.equal((await missingPath.json()).error, 'not-found');

    const missingRun = await fetch(`${base}/api/runs/r_absent`);
    assert.equal(missingRun.status, 404);
    assert.equal((await missingRun.json()).error, 'not-found');
  });

  it('rejects request methods other than get and head', async () => {
    const res = await fetch(`${base}/api/runs`, { method: 'POST' });
    assert.equal(res.status, 405);
    assert.equal((await res.json()).error, 'method-not-allowed');
  });
});

describe('kolm code viewer binding rules', () => {
  it('refuses a wildcard bind unless the operator opts in', () => {
    assert.throws(
      () => createViewerServer({ store: stubStore(), host: '0.0.0.0', env: {} }),
      (err) => err.code === 'bad-arg' && /KOLM_CODE_VIEWER_PUBLIC/.test(err.message),
    );
    const allowed = createViewerServer({
      store: stubStore(),
      host: '0.0.0.0',
      env: { KOLM_CODE_VIEWER_PUBLIC: '1' },
    });
    assert.equal(allowed.host, '0.0.0.0');
    assert.equal(allowed.url, null, 'no url before listen()');
  });

  it('requires an open store to construct', () => {
    assert.throws(() => createViewerServer({}), (err) => err.code === 'bad-arg');
  });

  it('brackets ipv6 literals when formatting the url', () => {
    assert.equal(formatUrl('127.0.0.1', 7474), 'http://127.0.0.1:7474');
    assert.equal(formatUrl('::1', 7474), 'http://[::1]:7474');
  });
});

describe('kolm code viewer document', () => {
  const html = fs.readFileSync(DEFAULT_INDEX_HTML, 'utf8');

  it('builds its dom without innerHTML and without remote requests', () => {
    assert.ok(!/innerHTML/.test(html), 'no innerHTML anywhere in the viewer');
    assert.ok(!/https?:\/\//.test(html), 'no absolute http or https url in the viewer');
    assert.ok(!new RegExp(['hon','est'].join(''), 'i').test(html), 'house vocabulary rule');
    assert.ok(html.includes('createElement'), 'the viewer builds nodes explicitly');
    assert.ok(html.includes('textContent'), 'the viewer writes text, not markup');
  });

  it('uses only the kolm brand palette', () => {
    const allowed = new Set([
      '08090A', '0F1011', '15171A', 'F7F8F8', 'D0D6E0',
      '8A8F98', '3FE5A0', 'F4F2EC', 'C9A94B', 'E5634B',
    ]);
    const found = [...html.matchAll(/#([0-9a-fA-F]{6})/g)].map((m) => m[1].toUpperCase());
    assert.ok(found.length > 0, 'the document declares colors');
    const offenders = [...new Set(found)].filter((hex) => !allowed.has(hex));
    assert.deepEqual(offenders, [], `unexpected colors: ${offenders.join(', ')}`);
  });

  it('keeps every corner radius at three pixels or less', () => {
    const radii = [...html.matchAll(/border-radius:\s*([0-9.]+)px/g)].map((m) => Number(m[1]));
    assert.deepEqual(radii.filter((value) => value > 3), []);
  });

  it('renders at both the wide and the narrow layout width', () => {
    assert.ok(/@media \(max-width: 1099px\)/.test(html), 'a breakpoint below 1280 exists');
    assert.ok(/grid-template-columns:\s*300px/.test(html), 'the wide layout keeps a run rail');
    assert.ok(html.includes('<title>kolm code</title>'));
  });

  it('wires the keyboard shortcuts the guide documents', () => {
    assert.ok(html.includes("event.key === 'j'"));
    assert.ok(html.includes("event.key === 'k'"));
    assert.ok(html.includes("event.key === 'r'"));
  });
});

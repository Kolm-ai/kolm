// End-to-end test that boots the server and exercises every layer through HTTP.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';
import net from 'node:net';

import { killAndWait, rmSyncBestEffort } from './_spawn-helpers.js';

let PORT;
let BASE;
let proc, demoKey, adminKey = 'ks_admin_change_me';
const testDataDir = path.join(os.tmpdir(), `kolm-e2e-${process.pid}-${Date.now()}`);

async function freePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.once('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const port = srv.address().port;
      srv.close(() => resolve(port));
    });
  });
}

async function fetchJson(p, opts = {}) {
  const res = await fetch(BASE + p, opts);
  const text = await res.text();
  let body; try { body = JSON.parse(text); } catch { body = { _raw: text }; }
  return { status: res.status, body };
}

async function authedFetch(p, opts = {}, key = adminKey) {
  return fetchJson(p, {
    ...opts,
    headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + key, ...(opts.headers || {}) },
  });
}

async function waitForHealth(base = BASE, retries = 120) {
  for (let i = 0; i < retries; i++) {
    try {
      const r = await fetch(base + '/health');
      if (r.ok) return await r.json();
    } catch {} // deliberate: cleanup
    await new Promise(r => setTimeout(r, 100));
  }
  throw new Error('server did not come up');
}

test('e2e: synthesize → register → run → search → compose', async (t) => {
  PORT = await freePort();
  BASE = `http://127.0.0.1:${PORT}`;
  demoKey = null;

  // Keep the real ./data directory untouched; the server reads KOLM_DATA_DIR.
  rmSyncBestEffort(testDataDir);
  fs.mkdirSync(testDataDir, { recursive: true });
  // Drop the child BEFORE the tmp dir so the OS releases SQLite/log handles
  // and rmSync doesn't trip EPERM on the next line. after() runs LIFO so
  // the kill registers last and fires first.
  t.after(() => rmSyncBestEffort(testDataDir));

  // boot server
  proc = spawn(process.execPath, ['server.js'], {
    // KOLM_STORE_DRIVER=json: assertions below read tenants.json directly, so the file-shape contract
    // must hold even when .env defaults the deploy driver to sqlite.
    env: { ...process.env, PORT: String(PORT), DEFAULT_TENANT: 'test', ANTHROPIC_API_KEY: '', KOLM_DATA_DIR: testDataDir, KOLM_STORE_DRIVER: 'json' },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  proc.stdout.on('data', () => {});
  proc.stderr.on('data', d => process.stderr.write(d));
  t.after(() => killAndWait(proc));

  await waitForHealth();
  const signup = await fetchJson('/v1/signup', {
    method: 'POST',
    body: JSON.stringify({ email: `e2e-${process.pid}-${Date.now()}@example.com`, name: 'e2e' }),
    headers: { 'Content-Type': 'application/json' },
  });
  assert.equal(signup.status, 201, JSON.stringify(signup.body));
  demoKey = signup.body.api_key;
  assert.match(demoKey, /^ks_[0-9a-f]{32}$/);

  const tenants = JSON.parse(fs.readFileSync(path.join(testDataDir, 'tenants.json'), 'utf8'));
  const e2eTenant = tenants.find(tn => tn.email === signup.body.tenant?.email || tn.id === signup.body.tenant?.id);
  assert.ok(e2eTenant, 'signup tenant should be persisted');
  assert.equal(e2eTenant.api_key, undefined, 'raw tenant api key should not be persisted');
  assert.match(e2eTenant.api_key_hash, /^sha256:[0-9a-f]{64}$/);

  // health
  const h = await fetchJson('/health');
  assert.equal(h.status, 200);
  assert.equal(h.body.status, 'ok');
  const ready = await fetchJson('/ready');
  assert.equal(ready.status, 200);
  assert.equal(ready.body.status, 'ready');
  assert.ok(Array.isArray(ready.body.checks));

  // synthesize a boolean classifier
  const synth = await authedFetch('/v1/synthesize', {
    method: 'POST',
    body: JSON.stringify({
      name: 'test-spam',
      tags: ['t'],
      visibility: 'private',
      output_spec: { type: 'boolean' },
      positives: [
        { input: 'FREE viagra click', expected: true },
        { input: 'URGENT prize claim', expected: true },
        { input: 'lunch tomorrow', expected: false },
        { input: 'project review', expected: false },
      ],
      negatives: [{ input: 'sync at 3pm', expected_not: true }],
    }),
  }, demoKey);
  assert.equal(synth.status, 200, JSON.stringify(synth.body));
  assert.equal(synth.body.accepted, true);
  assert.ok(synth.body.concept_id, 'concept_id present');

  const conceptId = synth.body.concept_id;

  // run it
  const run1 = await authedFetch('/v1/run', { method: 'POST', body: JSON.stringify({ concept_id: conceptId, input: 'WIN a free reward' }) }, demoKey);
  assert.equal(run1.status, 200);
  assert.equal(typeof run1.body.output, 'boolean');

  // run cached
  const run2 = await authedFetch('/v1/run', { method: 'POST', body: JSON.stringify({ concept_id: conceptId, input: 'WIN a free reward' }) }, demoKey);
  assert.equal(run2.body.cache, 'L1', 'second run should hit L1 cache');

  // list concepts
  const list = await authedFetch('/v1/concepts', {}, demoKey);
  assert.equal(list.status, 200);
  assert.ok(list.body.concepts.length >= 1);

  // search
  const search = await authedFetch('/v1/search', { method: 'POST', body: JSON.stringify({ query: 'classify spam', k: 5 }) }, demoKey);
  assert.equal(search.status, 200);
  assert.ok(search.body.matches.length >= 1);

  // compose
  const compose = await authedFetch('/v1/compose', { method: 'POST', body: JSON.stringify({ query: 'spam', input: 'win a reward', strategy: 'voting', k: 3 }) }, demoKey);
  assert.equal(compose.status, 200);
  assert.ok(compose.body.dispatched.length >= 1);

  // telemetry
  const telem = await authedFetch('/v1/telemetry', {}, demoKey);
  assert.equal(telem.status, 200);
  assert.ok(telem.body.total_invocations >= 2);

  // verify
  const verify = await authedFetch('/v1/verify', {
    method: 'POST',
    body: JSON.stringify({
      source: 'function generate(input, lib){ return lib.containsAny(input, ["free","prize","win"]); }',
      positives: [{ input: 'FREE prize', expected: true }, { input: 'lunch', expected: false }],
      negatives: [],
    }),
  }, demoKey);
  assert.equal(verify.status, 200);
  assert.ok(verify.body.quality_score >= 0.8);

  // spend-capable provider-backed endpoints require auth before validation/provider work
  const publicVerified = await fetchJson('/v1/verified-inference', {
    method: 'POST',
    body: JSON.stringify({ prompt: 'hello', test_cases: [{ input: 'x', expected: 'y' }] }),
    headers: { 'Content-Type': 'application/json' },
  });
  assert.equal(publicVerified.status, 401);

  const publicWrap = await fetchJson('/v1/wrap/verified', {
    method: 'POST',
    body: JSON.stringify({ messages: [{ role: 'user', content: 'hello' }], verified: { k: 1, test_cases: [{ input: 'x', expected: 'y' }] } }),
    headers: { 'Content-Type': 'application/json' },
  });
  assert.equal(publicWrap.status, 401);

  const authedVerifiedNoProvider = await authedFetch('/v1/verified-inference', {
    method: 'POST',
    body: JSON.stringify({ prompt: 'hello', test_cases: [{ input: 'x', expected: 'y' }] }),
  }, demoKey);
  assert.equal(authedVerifiedNoProvider.status, 503);

  // public listing (no auth)
  const pub = await fetchJson('/v1/public/concepts');
  assert.equal(pub.status, 200);

  // 401 without key
  const noauth = await fetchJson('/v1/concepts');
  assert.equal(noauth.status, 401);
});

test('hosted runtime rejects the development admin fallback key', async (t) => {
  const port = await freePort();
  const base = `http://127.0.0.1:${port}`;
  const dataDir = path.join(os.tmpdir(), `kolm-hosted-auth-${process.pid}-${Date.now()}`);

  rmSyncBestEffort(dataDir);
  fs.mkdirSync(dataDir, { recursive: true });
  t.after(() => rmSyncBestEffort(dataDir));

  const env = {
    ...process.env,
    PORT: String(port),
    DEFAULT_TENANT: 'hosted',
    ANTHROPIC_API_KEY: '',
    KOLM_DATA_DIR: dataDir,
    AWS_EXECUTION_ENV: 'AWS_ECS_FARGATE',
  };
  delete env.ADMIN_KEY;
  delete env.NODE_ENV;
  delete env.RECIPE_RECEIPT_SECRET;
  delete env.KOLM_ARTIFACT_SECRET;

  const child = spawn(process.execPath, ['server.js'], {
    env,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  child.stdout.on('data', () => {});
  child.stderr.on('data', d => process.stderr.write(d));
  t.after(() => killAndWait(child));

  await waitForHealth(base);
  const readiness = await fetch(base + '/ready');
  assert.equal(readiness.status, 503);
  assert.equal((await readiness.json()).status, 'not_ready');

  const res = await fetch(base + '/v1/signin', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ api_key: 'ks_admin_change_me' }),
  });

  assert.equal(res.status, 401);

  const verifyReceipt = await fetch(base + '/v1/receipts/verify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ receipt: { spec: 'rs-1' } }),
  });
  assert.equal(verifyReceipt.status, 503);
});

// kolm code — owner F tests: Truffle partner client + worker (spec 11.2).
//
// The wire-compatibility proof is an in-process Truffle stand-in on
// 127.0.0.1:0 that validates the service headers, the timestamp skew and
// re-derives the HMAC from the raw request bytes it received, spelling the
// canonical string out by hand (contract.ts:665-673) instead of importing it.

import test from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import http from 'node:http';

import {
  MAX_OUTPUT_BYTES,
  clampMessage,
  createTruffleClient,
  formatSettlementMessage,
  readinessDocument,
  signedRequestHeaders,
  stepOutputBudget,
  taskToKolmRequest,
  verifyProcessorSignature,
} from '../src/ladder/truffle/partner-task.js';
import {
  DEFAULT_PROCESSOR_TIMEOUT_MS,
  MAX_CLAIM_BACKOFF_MS,
  REQUIRED_TRUFFLE_ENV,
  createWorkerFromEnv,
  describeTruffleArming,
  parseHealthHost,
  runWorker,
  truffleEnvFromConfig,
} from '../src/ladder/truffle/worker.js';

const SERVICE_TOKEN = crypto.randomBytes(32).toString('base64url');
const SIGNING_SECRET = crypto.randomBytes(32).toString('base64url');
const WORKSPACE_ID = 'WS-Kolm-Code';
const KEY_ID = 'kolm-code-processor';
const QUIET = { log() {}, warn() {}, error() {}, info() {} };

function hex64(seed) {
  return crypto.createHash('sha256').update(seed).digest('hex');
}

function uuid(seed) {
  const h = crypto.createHash('sha256').update(seed).digest('hex');
  return [
    h.slice(0, 8),
    h.slice(8, 12),
    `4${h.slice(13, 16)}`,
    `8${h.slice(17, 20)}`,
    h.slice(20, 32),
  ].join('-');
}

function sampleClaim(seed = 'run-1') {
  return {
    schema: 'truffle.partner-work-claim.v1',
    runId: uuid(seed),
    version: 1,
    attempt: 1,
    idempotencyKey: hex64(`idem:${seed}`),
    triggerText: 'please fix the parser in https://github.com/kolm-ai/kolm base:release/1.2',
    threadRootId: 'thread-root-1',
    leaseToken: hex64(`lease:${seed}`),
    task: {
      schema: 'truffle.channel-partner-task.v1',
      stepCapability: hex64(`capability:${seed}`),
      stepId: 'step-1',
      role: 'maintainer',
      instructions: 'fix the parser',
      output: { kind: 'thread-message', maxBytes: MAX_OUTPUT_BYTES, rootMessageId: 'thread-root-1' },
    },
  };
}

/**
 * An in-process Truffle gateway. Every request is validated the way the real
 * gateway validates it; anything that fails is recorded and answered 401/403
 * so a signing regression fails the test loudly.
 */
async function startFakeTruffle({ claims = [], settleStatus = 200 } = {}) {
  const requests = [];
  const problems = [];
  let port = 0;
  let pending = [...claims];

  const server = http.createServer((req, res) => {
    const chunks = [];
    req.on('data', (chunk) => chunks.push(chunk));
    req.on('end', () => {
      const raw = Buffer.concat(chunks);
      const url = new URL(req.url, `http://${req.headers.host}`);
      const bad = [];

      if (req.method !== 'POST') bad.push('method');
      if (req.headers.authorization !== `Bearer ${SERVICE_TOKEN}`) bad.push('authorization');
      if (req.headers['x-truffle-service-audience'] !== 'truffle-collaboration') bad.push('audience');
      if (req.headers['x-truffle-workspace-id'] !== WORKSPACE_ID.toLowerCase()) bad.push('workspace');
      if (req.headers['x-truffle-partner-key-id'] !== KEY_ID) bad.push('key-id');
      if (req.headers['x-truffle-partner-contract'] !== 'truffle.partner-work-request.v1') {
        bad.push('contract');
      }
      if (req.headers['x-truffle-partner-replay-contract'] !== 'truffle.partner-durable-replay.v1') {
        bad.push('replay-contract');
      }

      const timestamp = req.headers['x-truffle-partner-timestamp'];
      const skewMs = Math.abs(Date.now() - Number(timestamp) * 1000);
      if (!/^[1-9][0-9]{9}$/.test(String(timestamp)) || skewMs > 300_000) bad.push('timestamp');

      const digest = crypto.createHash('sha256').update(raw).digest('hex');
      if (req.headers['x-truffle-partner-content-sha256'] !== digest) bad.push('content-sha256');

      const idempotencyKey = String(req.headers['idempotency-key'] ?? '');
      if (!/^[0-9a-f]{64}$/.test(idempotencyKey)) bad.push('idempotency-key');

      // contract.ts:665-673 — the canonical request string, spelled out.
      const canonical = [
        'truffle:partner-processor-request:v1',
        'POST',
        `http://${req.headers.host}${url.pathname}${url.search}`,
        String(timestamp),
        KEY_ID,
        idempotencyKey,
        digest,
      ].join('\n');
      const expected = `v1=${crypto
        .createHmac('sha256', Buffer.from(SIGNING_SECRET, 'base64url'))
        .update(canonical, 'utf8')
        .digest('hex')}`;
      if (req.headers['x-truffle-partner-signature'] !== expected) bad.push('signature');

      let body = null;
      try {
        body = JSON.parse(raw.toString('utf8'));
      } catch {
        bad.push('json');
      }

      requests.push({
        path: url.pathname,
        headers: req.headers,
        body,
        rawBody: raw.toString('utf8'),
        signatureVerified: !bad.includes('signature'),
        problems: bad,
      });
      if (bad.length > 0) problems.push({ path: url.pathname, bad });

      const send = (status, payload) => {
        const text = JSON.stringify(payload);
        res.writeHead(status, { 'content-type': 'application/json' });
        res.end(text);
      };

      if (bad.length > 0) {
        send(401, { error: 'unauthorized', problems: bad });
        return;
      }
      if (url.pathname.endsWith('/claim')) {
        const next = pending.length > 0 ? pending.shift() : null;
        send(200, { schema: 'truffle.partner-work-claim-result.v1', claim: next });
        return;
      }
      if (url.pathname.endsWith('/settle')) {
        if (settleStatus !== 200) {
          send(settleStatus, { error: 'settlement rejected' });
          return;
        }
        send(200, {
          schema: 'truffle.partner-work-settlement-result.v1',
          runId: body?.run?.id ?? null,
          status: 'succeeded',
          messageId: 'msg-1',
          replayed: false,
        });
        return;
      }
      send(404, { error: 'not_found' });
    });
  });

  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  port = server.address().port;

  return {
    url: `http://127.0.0.1:${port}`,
    requests,
    problems,
    setClaims(next) {
      pending = [...next];
    },
    async close() {
      await new Promise((resolve) => server.close(() => resolve()));
    },
  };
}

function clientFor(server, overrides = {}) {
  return createTruffleClient({
    apiUrl: server.url,
    serviceToken: SERVICE_TOKEN,
    workspaceId: WORKSPACE_ID,
    keyId: KEY_ID,
    signingSecret: SIGNING_SECRET,
    ...overrides,
  });
}

async function waitFor(predicate, label) {
  for (let i = 0; i < 500; i += 1) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  assert.fail(`timed out waiting for ${label}`);
}

/* ------------------------------------------------------------------ *
 * wire compatibility
 * ------------------------------------------------------------------ */

test('worker claims, runs and settles against a signature-checking gateway', async (t) => {
  const server = await startFakeTruffle({ claims: [sampleClaim('wire')] });
  t.after(() => server.close());

  const seen = [];
  const result = await runWorker({
    client: clientFor(server),
    handler: (request) => {
      seen.push(request);
      return { verdict: 'PASS', summary: 'one candidate passed L0-L3', costUsd: 0.25, prUrl: 'https://github.com/kolm-ai/kolm/pull/3' };
    },
    logger: QUIET,
    maxTasks: 1,
    pollMs: 5,
  });

  assert.deepEqual(result, { processed: 1, failed: 0, stoppedBy: 'maxTasks' });
  assert.deepEqual(server.problems, []);
  assert.equal(server.requests.length, 2);
  assert.ok(server.requests.every((r) => r.signatureVerified));

  const [claimRequest, settleRequest] = server.requests;
  assert.ok(claimRequest.path.endsWith('/api/channel-workflow-partner/v1/claim'));
  assert.deepEqual(Object.keys(claimRequest.body).sort(), ['processorReadiness', 'schema']);
  assert.equal(claimRequest.body.schema, 'truffle.partner-work-claim-request.v1');
  assert.equal(
    claimRequest.body.processorReadiness.schema,
    'truffle.partner-processor-readiness.v1',
  );

  assert.ok(settleRequest.path.endsWith('/api/channel-workflow-partner/v1/settle'));
  const claim = sampleClaim('wire');
  assert.deepEqual(Object.keys(settleRequest.body).sort(), [
    'capability', 'operationId', 'output', 'run', 'schema',
  ]);
  assert.equal(settleRequest.body.schema, 'truffle.partner-work-settlement.v1');
  assert.deepEqual(settleRequest.body.run, {
    id: claim.runId,
    version: 1,
    leaseToken: claim.leaseToken,
  });
  assert.equal(settleRequest.body.capability, claim.task.stepCapability);
  assert.equal(settleRequest.body.operationId, `partner.settle.${claim.idempotencyKey}`);
  assert.equal(settleRequest.body.output.kind, 'thread-message');
  assert.match(settleRequest.body.output.text, /^kolm code: PASS -- /);
  assert.ok(settleRequest.body.output.text.includes('pull request: https://github.com/kolm-ai/kolm/pull/3'));
  assert.ok(settleRequest.body.output.text.includes('Verify: kolm code receipt verify'));
  assert.ok(!settleRequest.body.output.text.includes('|'));

  // the settle idempotency key is the claim's, so Truffle can dedupe a replay
  assert.equal(settleRequest.headers['idempotency-key'], claim.idempotencyKey);
  assert.notEqual(claimRequest.headers['idempotency-key'], settleRequest.headers['idempotency-key']);

  assert.equal(seen.length, 1);
  assert.equal(seen[0].repoUrl, 'https://github.com/kolm-ai/kolm');
});

test('a failing handler settles the task with a clean error message', async (t) => {
  const server = await startFakeTruffle({ claims: [sampleClaim('boom')] });
  t.after(() => server.close());

  const result = await runWorker({
    client: clientFor(server),
    handler: () => {
      const err = new Error(`sandbox exploded with token ${SERVICE_TOKEN}`);
      throw err;
    },
    logger: QUIET,
    maxTasks: 1,
    pollMs: 5,
  });

  assert.deepEqual(result, { processed: 0, failed: 1, stoppedBy: 'maxTasks' });
  const settle = server.requests.find((r) => r.path.endsWith('/settle'));
  const text = settle.body.output.text;
  assert.match(text, /^kolm code: ERROR -- /);
  assert.ok(text.includes('sandbox exploded'));
  assert.ok(!text.includes('\n    at '));
  assert.ok(!text.includes('Error: Error'));
  // the worker's own credentials never reach a channel, whatever the sandbox printed
  assert.ok(!text.includes(SERVICE_TOKEN), 'service token redacted from the settlement');
  assert.ok(!text.includes(SIGNING_SECRET), 'signing secret redacted from the settlement');
  assert.ok(text.includes('[redacted]'));
});

/* ------------------------------------------------------------------ *
 * retries and timeouts
 * ------------------------------------------------------------------ */

test('a 500 answer is retried once with the documented backoff', async () => {
  const slept = [];
  const responses = [
    { ok: false, status: 500, text: async () => '{"error":"upstream"}' },
    {
      ok: true,
      status: 200,
      text: async () => JSON.stringify({ schema: 'x', claim: null }),
    },
  ];
  let calls = 0;
  const client = createTruffleClient({
    apiUrl: 'https://truffle.invalid',
    serviceToken: SERVICE_TOKEN,
    workspaceId: WORKSPACE_ID,
    keyId: KEY_ID,
    signingSecret: SIGNING_SECRET,
    fetchImpl: async () => {
      calls += 1;
      return responses.shift();
    },
    sleep: async (ms) => {
      slept.push(ms);
    },
  });

  const claimed = await client.claim({});
  assert.equal(claimed.task, null);
  assert.equal(calls, 2);
  assert.deepEqual(slept, [250]);
});

test('a 4xx answer is never retried', async () => {
  let calls = 0;
  const slept = [];
  const client = createTruffleClient({
    apiUrl: 'https://truffle.invalid',
    serviceToken: SERVICE_TOKEN,
    workspaceId: WORKSPACE_ID,
    keyId: KEY_ID,
    signingSecret: SIGNING_SECRET,
    fetchImpl: async () => {
      calls += 1;
      return { ok: false, status: 403, text: async () => '{"error":"forbidden"}' };
    },
    sleep: async (ms) => {
      slept.push(ms);
    },
  });

  const thrown = await client.claim({}).then(() => null, (e) => e);
  assert.equal(thrown.code, 'truffle-gateway');
  assert.equal(thrown.detail.status, 403);
  assert.equal(thrown.detail.retryable, false);
  assert.equal(calls, 1);
  assert.deepEqual(slept, []);
});

test('gateway failures never echo the service token back', async () => {
  const client = createTruffleClient({
    apiUrl: 'https://truffle.invalid',
    serviceToken: SERVICE_TOKEN,
    workspaceId: WORKSPACE_ID,
    keyId: KEY_ID,
    signingSecret: SIGNING_SECRET,
    fetchImpl: async () => ({
      ok: false,
      status: 401,
      text: async () => `{"error":"bad token ${SERVICE_TOKEN}"}`,
    }),
    sleep: async () => {},
  });
  const thrown = await client.claim({}).then(() => null, (e) => e);
  assert.equal(thrown.detail.status, 401);
  assert.ok(!JSON.stringify(thrown.detail).includes(SERVICE_TOKEN));
});

test('a handler that overruns the processor budget is settled as a failure', async (t) => {
  const server = await startFakeTruffle({ claims: [sampleClaim('slow')] });
  t.after(() => server.close());

  const result = await runWorker({
    client: clientFor(server),
    handler: (_request, { signal }) => new Promise((resolve) => {
      signal.addEventListener('abort', () => resolve('aborted'), { once: true });
    }),
    logger: QUIET,
    maxTasks: 1,
    pollMs: 5,
    processorTimeoutMs: 25,
  });

  assert.deepEqual(result, { processed: 0, failed: 1, stoppedBy: 'maxTasks' });
  const settle = server.requests.find((r) => r.path.endsWith('/settle'));
  assert.match(settle.body.output.text, /^kolm code: ERROR -- /);
  assert.ok(settle.body.output.text.includes('exceeded 25 ms'));
});

/* ------------------------------------------------------------------ *
 * messages
 * ------------------------------------------------------------------ */

test('thread messages are clamped to the contract output budget', () => {
  const message = formatSettlementMessage({
    verdict: 'PASS',
    title: 'huge run',
    summary: 'x'.repeat(60_000),
    runId: 'run-1',
  });
  assert.ok(Buffer.byteLength(message, 'utf8') <= MAX_OUTPUT_BYTES);
  assert.ok(message.endsWith('[truncated]'));
  assert.match(message, /^kolm code: PASS -- huge run/);

  const wide = clampMessage(`${'é'.repeat(40_000)}`);
  assert.ok(Buffer.byteLength(wide, 'utf8') <= MAX_OUTPUT_BYTES);
  assert.ok(wide.endsWith('[truncated]'));
});

test('settlement messages are plain text with a verify line', () => {
  const message = formatSettlementMessage({
    verdict: 'fail',
    title: 'fix the parser',
    costUsd: 1.5,
    stages: [{ stage: 'L1', decision: 'fail' }],
    runId: 'run-42',
  });
  const lines = message.split('\n');
  assert.equal(lines[0], 'kolm code: FAIL -- fix the parser');
  assert.ok(lines.some((line) => line === '- cost: $1.5000'));
  assert.ok(lines.some((line) => line === '- cascade: L1: fail'));
  assert.equal(lines[lines.length - 1], 'Verify: kolm code receipt verify run-42');
  assert.ok(!message.includes('|'));
});

/* ------------------------------------------------------------------ *
 * task mapping, readiness, signatures
 * ------------------------------------------------------------------ */

test('a claimed task maps onto the kolm code request shape', () => {
  const claim = sampleClaim('mapping');
  const request = taskToKolmRequest(claim);
  assert.equal(request.taskId, claim.runId);
  assert.equal(request.text, claim.triggerText);
  assert.equal(request.repoUrl, 'https://github.com/kolm-ai/kolm');
  assert.equal(request.baseRef, 'release/1.2');
  assert.equal(request.threadId, 'thread-root-1');
  assert.equal(request.requestedBy, 'maintainer');
  assert.equal(request.stepId, 'step-1');
  assert.equal(request.idempotencyKey, claim.idempotencyKey);
  assert.equal(request.raw, claim);

  const bare = taskToKolmRequest({ task: { instructions: 'no links here' } });
  assert.equal(bare.repoUrl, null);
  assert.equal(bare.baseRef, null);

  // People write in sentences: the period after "base:main." is not part of the ref.
  const prose = taskToKolmRequest({
    task: { instructions: 'Implement it in https://github.com/kolm-ai/kolm. Branch from base:main.' },
  });
  assert.equal(prose.repoUrl, 'https://github.com/kolm-ai/kolm');
  assert.equal(prose.baseRef, 'main');
  assert.equal(taskToKolmRequest({ task: { instructions: 'base:release/1.2)' } }).baseRef, 'release/1.2');
});

test('the thread message is clamped to the step budget, not only the protocol ceiling', async () => {
  // Truffle dead-letters a settlement whose text exceeds the step's output.maxBytes
  // (0118 settle_channel_workflow_partner_step), so the budget is the step's.
  const claim = sampleClaim('budget');
  claim.task.output.maxBytes = 8_000;
  assert.equal(stepOutputBudget(claim), 8_000);
  assert.equal(stepOutputBudget(claim.task), 8_000);
  assert.equal(stepOutputBudget({ task: { output: { maxBytes: 0 } } }), MAX_OUTPUT_BYTES);
  assert.equal(stepOutputBudget({ task: { output: { maxBytes: MAX_OUTPUT_BYTES + 1 } } }), MAX_OUTPUT_BYTES);
  assert.equal(stepOutputBudget(null), MAX_OUTPUT_BYTES);
  assert.equal(taskToKolmRequest(claim).maxOutputBytes, 8_000);

  const settled = [];
  let claims = 0;
  const client = {
    claim: async () => {
      claims += 1;
      return claims === 1 ? { task: claim } : { task: null };
    },
    settle: async (args) => {
      settled.push(args);
      return { status: 'succeeded' };
    },
  };
  const long = ['kolm code: PASS -- long report', ...Array(400).fill('- changed: src/a/very/long/path/to/a/file.js')].join('\n');
  assert.ok(Buffer.byteLength(long, 'utf8') > 8_000);
  const result = await runWorker({ client, handler: () => long, logger: QUIET, maxTasks: 1, pollMs: 5 });
  assert.deepEqual(result, { processed: 1, failed: 0, stoppedBy: 'maxTasks' });
  assert.ok(Buffer.byteLength(settled[0].message, 'utf8') <= 8_000);
  assert.ok(settled[0].message.endsWith(' [truncated]'));

  // The client clamps again on the way out, so a caller-supplied message is bounded too.
  let body = null;
  const direct = createTruffleClient({
    apiUrl: 'https://truffle.test',
    serviceToken: 'a'.repeat(43),
    workspaceId: '11111111-1111-4111-8111-111111111111',
    keyId: 'key-1',
    signingSecret: 's'.repeat(43),
    fetchImpl: async (_url, init) => {
      body = JSON.parse(init.body);
      return new Response(JSON.stringify({
        schema: 'truffle.partner-work-settlement-result.v1',
        runId: claim.runId,
        status: 'succeeded',
        messageId: 'm1',
        replayed: false,
      }), { status: 200 });
    },
  });
  await direct.settle({ task: claim, status: 'completed', message: long });
  assert.ok(Buffer.byteLength(body.output.text, 'utf8') <= 8_000);
});

test('the readiness attestation carries exactly the contract keys', () => {
  const doc = readinessDocument({ processorId: 'kolm-code', keyId: KEY_ID, now: () => 1_700_000_000_000 });
  assert.deepEqual(Object.keys(doc).sort(), [
    'leaseToken',
    'leaseTtlSeconds',
    'observedAt',
    'partnerAdapterKey',
    'processStartedAt',
    'processorEndpointDigest',
    'processorInstanceId',
    'processorKeyId',
    'releaseSha',
    'schema',
  ]);
  assert.equal(doc.schema, 'truffle.partner-processor-readiness.v1');
  assert.equal(doc.partnerAdapterKey, 'truffle.channel-partner-task.v1');
  assert.equal(doc.leaseTtlSeconds, 60);
  assert.match(doc.processorEndpointDigest, /^[0-9a-f]{64}$/);
  assert.match(doc.leaseToken, /^[0-9a-f]{64}$/);
  assert.deepEqual(JSON.parse(JSON.stringify(doc)).capabilities, undefined);

  const repeat = readinessDocument({ processorId: 'kolm-code', keyId: KEY_ID, now: () => 1_700_000_000_000 });
  assert.deepEqual(repeat, doc);
});

test('processor signatures verify symmetrically and reject tampering', () => {
  const target = 'https://kolm.ai/kolm-code/processor?v=1';
  const body = JSON.stringify({ schema: 'truffle.partner-work-request.v1' });
  const idempotencyKey = hex64('signature-case');
  const now = 1_700_000_000_000;
  const headers = signedRequestHeaders({
    target, body, idempotencyKey, keyId: KEY_ID, signingSecret: SIGNING_SECRET, now: () => now,
  });

  assert.equal(headers['x-truffle-partner-contract'], 'truffle.partner-work-request.v1');
  assert.match(headers['x-truffle-partner-signature'], /^v1=[0-9a-f]{64}$/);

  const args = { headers, body, target, signingSecret: SIGNING_SECRET, now: () => now };
  assert.equal(verifyProcessorSignature(args), true);
  assert.equal(verifyProcessorSignature({ ...args, body: `${body} ` }), false);
  assert.equal(verifyProcessorSignature({ ...args, target: `${target}&x=2` }), false);
  assert.equal(verifyProcessorSignature({ ...args, now: () => now + 600_000 }), false);
  assert.equal(verifyProcessorSignature({ ...args, signingSecret: crypto.randomBytes(32).toString('base64url') }), false);
  assert.equal(verifyProcessorSignature({ ...args, expectedKeyId: 'someone-else' }), false);
});

/* ------------------------------------------------------------------ *
 * worker loop behaviour
 * ------------------------------------------------------------------ */

test('the health endpoint answers with the readiness document', async () => {
  const controller = new AbortController();
  const health = { host: '127.0.0.1', port: 0 };
  const client = {
    readiness: () => readinessDocument({ processorId: 'kolm-code', keyId: KEY_ID }),
    claim: async () => ({ task: null, claim: null }),
    settle: async () => assert.fail('nothing to settle'),
  };

  const run = runWorker({
    client,
    handler: () => assert.fail('no task expected'),
    logger: QUIET,
    pollMs: 5,
    signal: controller.signal,
    health,
  });

  await waitFor(() => health.port > 0, 'the health server to listen');
  const response = await fetch(`http://127.0.0.1:${health.port}/healthz`);
  const payload = await response.json();
  assert.equal(response.status, 200);
  assert.equal(payload.ok, true);
  assert.equal(payload.readiness.schema, 'truffle.partner-processor-readiness.v1');

  const missing = await fetch(`http://127.0.0.1:${health.port}/nope`);
  assert.equal(missing.status, 404);
  await missing.text();

  controller.abort();
  const result = await run;
  assert.equal(result.stoppedBy, 'signal');

  await assert.rejects(() => fetch(`http://127.0.0.1:${health.port}/healthz`));
});

test('an abort stops the loop only after the in-flight task settles', async () => {
  const controller = new AbortController();
  const settled = [];
  let claims = 0;
  const client = {
    claim: async () => {
      claims += 1;
      return claims === 1 ? { task: sampleClaim('abort'), claim: sampleClaim('abort') } : { task: null, claim: null };
    },
    settle: async (args) => {
      settled.push(args);
      return { status: 'succeeded' };
    },
  };

  const result = await runWorker({
    client,
    handler: async () => {
      controller.abort();
      return { verdict: 'PASS', summary: 'done' };
    },
    logger: QUIET,
    pollMs: 5,
    signal: controller.signal,
  });

  assert.deepEqual(result, { processed: 1, failed: 0, stoppedBy: 'signal' });
  assert.equal(claims, 1);
  assert.equal(settled.length, 1);
  assert.equal(settled[0].status, 'completed');
  assert.match(settled[0].message, /^kolm code: PASS -- /);
});

test('claim failures back off exponentially and stop at the cap', async () => {
  const controller = new AbortController();
  const slept = [];
  const client = {
    claim: async () => {
      throw new Error('gateway unreachable');
    },
    settle: async () => assert.fail('nothing to settle'),
  };

  const result = await runWorker({
    client,
    handler: () => assert.fail('no task expected'),
    logger: QUIET,
    pollMs: 1_000,
    signal: controller.signal,
    sleep: async (ms) => {
      slept.push(ms);
      if (slept.length >= 6) controller.abort();
    },
  });

  assert.deepEqual(slept, [1_000, 2_000, 4_000, 8_000, 16_000, MAX_CLAIM_BACKOFF_MS]);
  assert.equal(result.stoppedBy, 'signal');
  assert.equal(result.processed, 0);
});

test('a string handler result is used verbatim as the thread message', async () => {
  const settled = [];
  let claims = 0;
  const client = {
    claim: async () => {
      claims += 1;
      return claims === 1 ? { task: sampleClaim('verbatim') } : { task: null };
    },
    settle: async (args) => {
      settled.push(args);
      return { status: 'succeeded' };
    },
  };
  const result = await runWorker({
    client,
    handler: () => 'kolm code: PASS -- handled by the caller',
    logger: QUIET,
    maxTasks: 1,
    pollMs: 5,
  });
  assert.deepEqual(result, { processed: 1, failed: 0, stoppedBy: 'maxTasks' });
  assert.equal(settled[0].message, 'kolm code: PASS -- handled by the caller');
});

/* ------------------------------------------------------------------ *
 * arming
 * ------------------------------------------------------------------ */

test('worker construction refuses to start and names the missing variables', () => {
  let thrown = null;
  try {
    createWorkerFromEnv({ env: { TRUFFLE_PARTNER_API_URL: 'https://truffle.invalid' }, handler: () => {} });
  } catch (err) {
    thrown = err;
  }
  assert.equal(thrown.code, 'not-armed');
  assert.ok(thrown.message.includes('missing:'));
  for (const name of REQUIRED_TRUFFLE_ENV.slice(1)) {
    assert.ok(thrown.detail.missing.includes(name), `${name} should be listed as missing`);
  }
  assert.ok(!thrown.detail.missing.includes('TRUFFLE_PARTNER_API_URL'));
});

test('arming description is a pure function of the environment', () => {
  assert.deepEqual(describeTruffleArming({}), { armed: false, missing: [...REQUIRED_TRUFFLE_ENV] });
  const full = Object.fromEntries(REQUIRED_TRUFFLE_ENV.map((name) => [name, 'value']));
  assert.deepEqual(describeTruffleArming(full), { armed: true, missing: [] });
  assert.deepEqual(describeTruffleArming({ ...full, TRUFFLE_PARTNER_WORKSPACE_ID: '  ' }), {
    armed: false,
    missing: ['TRUFFLE_PARTNER_WORKSPACE_ID'],
  });
});

test('health host strings parse into a host and a port', () => {
  assert.deepEqual(parseHealthHost('127.0.0.1:9091'), { host: '127.0.0.1', port: 9091 });
  assert.deepEqual(parseHealthHost('0.0.0.0'), { host: '0.0.0.0', port: 8080 });
  assert.equal(parseHealthHost(''), null);
  assert.equal(parseHealthHost(undefined), null);
});

test('the config adapter projects owner A config onto the truffle env names', () => {
  const env = truffleEnvFromConfig({
    truffle: {
      apiUrl: 'https://truffle.example',
      workspaceId: 'ws-1',
      keyId: KEY_ID,
      harnessId: 'kolm-code',
      healthHost: '0.0.0.0:9091',
      pollMs: 1500,
      serviceToken: true,
      signingSecret: true,
    },
    getSecret: (name) => (name === 'truffleServiceToken' ? SERVICE_TOKEN : SIGNING_SECRET),
  });
  assert.equal(env.TRUFFLE_PARTNER_API_URL, 'https://truffle.example');
  assert.equal(env.TRUFFLE_PARTNER_SERVICE_TOKEN, SERVICE_TOKEN);
  assert.equal(env.TRUFFLE_PARTNER_PROCESSOR_SIGNING_SECRET, SIGNING_SECRET);
  assert.equal(env.TRUFFLE_PARTNER_POLL_MS, '1500');
  assert.equal(describeTruffleArming(env).armed, true);
  assert.deepEqual(truffleEnvFromConfig(null), {});
});

test('the processor budget is a coding run, not a chat turn', () => {
  assert.equal(DEFAULT_PROCESSOR_TIMEOUT_MS, 1_800_000);
});

// kolm code — Truffle partner worker loop (owner F, spec 9 + 11.2).
//
// claim -> run -> settle, one task at a time, exactly the shape of Truffle's
// reference worker (`reference-worker.ts:748-784`) with one deliberate
// difference: the processor budget. Truffle's reference allows a step 20 s
// (`PROCESSOR_TIMEOUT_MS`, reference-worker.ts:37) because its steps are
// chat-latency LLM calls; a kolm code run plans, generates candidates, runs a
// verification cascade and may call a judge, so the budget here defaults to
// 30 minutes (`KOLM_CODE_TRUFFLE_PROCESSOR_TIMEOUT_MS`).

import http from 'node:http';
import { setTimeout as delay } from 'node:timers/promises';

import { LadderError } from '../errors.js';
import {
  DEFAULT_POLL_MS,
  REFERENCE_PROCESSOR_TIMEOUT_MS,
  clampMessage,
  createTruffleClient,
  formatSettlementMessage,
  readinessDocument,
  taskToKolmRequest,
} from './partner-task.js';

/** A coding run is not a chat turn: 30 minutes by default. */
export const DEFAULT_PROCESSOR_TIMEOUT_MS = 1_800_000;
/** Claim failures back off exponentially from `pollMs` up to this cap. */
export const MAX_CLAIM_BACKOFF_MS = 30_000;
export const DEFAULT_HEALTH_PORT = 8080;

export const REQUIRED_TRUFFLE_ENV = Object.freeze([
  'TRUFFLE_PARTNER_API_URL',
  'TRUFFLE_PARTNER_SERVICE_TOKEN',
  'TRUFFLE_PARTNER_WORKSPACE_ID',
  'TRUFFLE_PARTNER_PROCESSOR_KEY_ID',
  'TRUFFLE_PARTNER_PROCESSOR_SIGNING_SECRET',
]);

function log(logger, level, message, detail) {
  const sink = logger && typeof logger[level] === 'function'
    ? logger[level].bind(logger)
    : logger && typeof logger.log === 'function'
      ? logger.log.bind(logger)
      : null;
  if (!sink) return;
  if (detail === undefined) sink(message);
  else sink(message, detail);
}

async function defaultSleep(ms, signal) {
  try {
    await delay(ms, undefined, { signal });
  } catch {
    // an aborted sleep is a normal shutdown, not a failure
  }
}

/**
 * A one-line, stack-free, credential-free rendering of a handler failure.
 * @param {unknown} error
 * @returns {string}
 */
export function safeErrorText(error) {
  const raw = error instanceof Error
    ? error.message
    : typeof error === 'string'
      ? error
      : error && typeof error.message === 'string'
        ? error.message
        : String(error ?? 'unknown failure');
  return raw.replace(/\s+/g, ' ').trim().slice(0, 500);
}

function withTimeout(promise, ms, controller) {
  if (!Number.isFinite(ms) || ms <= 0) return promise;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      controller.abort();
      reject(new LadderError('truffle-processor-timeout', `handler exceeded ${ms} ms`, { timeoutMs: ms }));
    }, ms);
    if (typeof timer.unref === 'function') timer.unref();
    promise.then(
      (value) => { clearTimeout(timer); resolve(value); },
      (err) => { clearTimeout(timer); reject(err); },
    );
  });
}

/**
 * Tiny health endpoint: `GET /healthz` answers with the readiness document
 * (this is what `TRUFFLE_PARTNER_HEALTH_HOST` exists for).
 * @param {object} opts
 * @returns {Promise<{server: import('node:http').Server, port: number, host: string}>}
 */
export async function startHealthServer({
  host = '127.0.0.1',
  port = DEFAULT_HEALTH_PORT,
  readiness,
  state = { ready: false },
} = {}) {
  const server = http.createServer((request, response) => {
    const url = new URL(request.url || '/', 'http://health.invalid');
    const method = request.method || 'GET';
    if (method !== 'GET' && method !== 'HEAD') {
      response.writeHead(405, { 'content-type': 'application/json', allow: 'GET, HEAD' });
      response.end('{"error":"method_not_allowed"}');
      return;
    }
    if (url.pathname !== '/healthz' && url.pathname !== '/health' && url.pathname !== '/live') {
      response.writeHead(404, { 'content-type': 'application/json' });
      response.end('{"error":"not_found"}');
      return;
    }
    const doc = typeof readiness === 'function' ? readiness() : readinessDocument();
    const body = JSON.stringify({
      ok: true,
      ready: Boolean(state.ready),
      capabilities: doc.capabilities ? [...doc.capabilities] : [],
      readiness: doc,
    });
    response.writeHead(200, { 'content-type': 'application/json', 'cache-control': 'no-store' });
    if (method === 'HEAD') response.end();
    else response.end(body);
  });

  await new Promise((resolve, reject) => {
    const failed = (err) => reject(err);
    server.once('error', failed);
    server.listen(port, host, () => {
      server.removeListener('error', failed);
      resolve();
    });
  });
  const address = server.address();
  return { server, host, port: address && typeof address === 'object' ? address.port : port };
}

async function closeServer(server) {
  if (!server || !server.listening) return;
  await new Promise((resolve) => server.close(() => resolve()));
}

/**
 * Polls Truffle for partner tasks and runs them through `handler`.
 * @param {object} opts
 * @param {object} opts.client from `createTruffleClient`
 * @param {(request: object, ctx: {task: object, claim: object, signal: AbortSignal}) => Promise<object|string>} opts.handler
 * @returns {Promise<{processed: number, failed: number, stoppedBy: 'signal'|'maxTasks'}>}
 */
export async function runWorker({
  client,
  processorId = 'kolm-code',
  pollMs = DEFAULT_POLL_MS,
  handler,
  logger = console,
  signal,
  maxTasks = null,
  capacity = 1,
  health = null,
  processorTimeoutMs = DEFAULT_PROCESSOR_TIMEOUT_MS,
  sleep = defaultSleep,
} = {}) {
  if (!client || typeof client.claim !== 'function' || typeof client.settle !== 'function') {
    throw new LadderError('truffle-config', 'runWorker needs a Truffle client with claim and settle');
  }
  if (typeof handler !== 'function') {
    throw new LadderError('truffle-config', 'runWorker needs a handler function');
  }

  const state = { ready: false };
  let healthServer = null;
  if (health && (health.host || health.port !== undefined)) {
    const started = await startHealthServer({
      host: health.host || '127.0.0.1',
      port: health.port ?? DEFAULT_HEALTH_PORT,
      readiness: () => (typeof client.readiness === 'function' ? client.readiness() : readinessDocument({ processorId })),
      state,
    });
    healthServer = started.server;
    if (health && typeof health === 'object' && Object.isExtensible(health)) {
      health.port = started.port;
      health.server = started.server;
    }
    if (typeof health.onListening === 'function') health.onListening(started);
  }

  let processed = 0;
  let failed = 0;
  let stoppedBy = 'signal';
  let backoff = pollMs;

  try {
    for (;;) {
      if (signal?.aborted) { stoppedBy = 'signal'; break; }
      if (maxTasks !== null && processed + failed >= maxTasks) { stoppedBy = 'maxTasks'; break; }

      let claimed = null;
      try {
        claimed = await client.claim({ processorId, capacity, signal });
        state.ready = true;
        backoff = pollMs;
      } catch (err) {
        state.ready = false;
        log(logger, 'warn', `[truffle-worker] claim failed: ${safeErrorText(err)}`);
        if (signal?.aborted) { stoppedBy = 'signal'; break; }
        await sleep(backoff, signal);
        backoff = Math.min(MAX_CLAIM_BACKOFF_MS, Math.max(1, backoff) * 2);
        continue;
      }

      const claim = claimed?.task ?? claimed?.claim ?? null;
      if (!claim) {
        if (signal?.aborted) { stoppedBy = 'signal'; break; }
        await sleep(pollMs, signal);
        continue;
      }

      const request = taskToKolmRequest(claim);
      const controller = new AbortController();
      let result = null;
      let handlerError = null;
      try {
        result = await withTimeout(
          Promise.resolve().then(() => handler(request, { task: claim, claim, signal: controller.signal })),
          processorTimeoutMs,
          controller,
        );
      } catch (err) {
        handlerError = err;
      }

      const status = handlerError ? 'failed' : 'completed';
      const message = handlerError
        ? formatSettlementMessage({
          verdict: 'ERROR',
          error: safeErrorText(handlerError),
          runId: request.taskId,
          title: request.text ? request.text.slice(0, 200) : null,
        })
        : typeof result === 'string'
          ? clampMessage(result)
          : formatSettlementMessage({ runId: request.taskId, ...(result && typeof result === 'object' ? result : {}) });

      try {
        await client.settle({
          task: claim,
          status,
          message,
          error: handlerError ? safeErrorText(handlerError) : null,
          signal,
        });
        if (handlerError) failed += 1;
        else processed += 1;
      } catch (err) {
        failed += 1;
        state.ready = false;
        log(logger, 'error', `[truffle-worker] settle failed: ${safeErrorText(err)}`);
      }

      if (signal?.aborted) { stoppedBy = 'signal'; break; }
      if (maxTasks !== null && processed + failed >= maxTasks) { stoppedBy = 'maxTasks'; break; }
    }
  } finally {
    state.ready = false;
    await closeServer(healthServer);
  }

  return { processed, failed, stoppedBy };
}

function trimmed(value) {
  return typeof value === 'string' && value.trim() !== '' ? value.trim() : null;
}

/**
 * Pure arming report for `kolm code doctor`.
 * @param {NodeJS.ProcessEnv} [env]
 * @returns {{armed: boolean, missing: string[]}}
 */
export function describeTruffleArming(env = process.env) {
  const missing = REQUIRED_TRUFFLE_ENV.filter((name) => !trimmed(env[name]));
  return { armed: missing.length === 0, missing };
}

/**
 * Parses `TRUFFLE_PARTNER_HEALTH_HOST` ("host" or "host:port").
 * @param {string|null|undefined} value
 * @returns {{host: string, port: number}|null}
 */
export function parseHealthHost(value) {
  const raw = trimmed(value);
  if (!raw) return null;
  const match = /^(.*?):(\d{1,5})$/.exec(raw);
  if (match && match[1] !== '') return { host: match[1], port: Number(match[2]) };
  return { host: raw, port: DEFAULT_HEALTH_PORT };
}

/**
 * Thin adapter over owner A's config object: `config.truffle` holds the
 * non-secret fields and `config.getSecret` the two credentials, so this
 * projects them back onto the env shape this module already takes.
 * @param {object} config from `loadConfig()` in `../config.js`
 * @returns {NodeJS.ProcessEnv} a synthetic env with only the Truffle keys set
 */
export function truffleEnvFromConfig(config) {
  const env = {};
  if (!config || typeof config !== 'object') return env;
  const t = config.truffle ?? {};
  const secret = (name) => {
    if (typeof config.getSecret !== 'function') return null;
    try {
      return config.getSecret(name);
    } catch {
      return null;
    }
  };

  if (t.apiUrl) env.TRUFFLE_PARTNER_API_URL = String(t.apiUrl);
  if (t.processorUrl) env.TRUFFLE_PARTNER_PROCESSOR_URL = String(t.processorUrl);
  if (t.workspaceId) env.TRUFFLE_PARTNER_WORKSPACE_ID = String(t.workspaceId);
  if (t.keyId) env.TRUFFLE_PARTNER_PROCESSOR_KEY_ID = String(t.keyId);
  if (t.harnessId) env.KOLM_TRUFFLE_HARNESS_ID = String(t.harnessId);
  if (t.healthHost) env.TRUFFLE_PARTNER_HEALTH_HOST = String(t.healthHost);
  if (t.pollMs) env.TRUFFLE_PARTNER_POLL_MS = String(t.pollMs);

  const serviceToken = secret('truffleServiceToken');
  if (serviceToken) env.TRUFFLE_PARTNER_SERVICE_TOKEN = serviceToken;
  const signingSecret = secret('truffleSigningSecret');
  if (signingSecret) env.TRUFFLE_PARTNER_PROCESSOR_SIGNING_SECRET = signingSecret;
  return env;
}

/**
 * Builds the client from Truffle's own env names and starts the loop.
 * Pass either `env` (the injectable default) or owner A's `config`.
 * Throws `LadderError('not-armed')` listing the MISSING variables.
 * @param {object} opts
 * @returns {Promise<{processed: number, failed: number, stoppedBy: string}>}
 */
export function createWorkerFromEnv({
  config = null,
  env = config ? truffleEnvFromConfig(config) : process.env,
  handler,
  fetchImpl = fetch,
  logger = console,
  signal,
  maxTasks = null,
  health,
  sleep,
  now,
  randomUUID,
  processorTimeoutMs,
} = {}) {
  const { armed, missing } = describeTruffleArming(env);
  if (!armed) {
    throw new LadderError(
      'not-armed',
      `Truffle partner worker needs ${REQUIRED_TRUFFLE_ENV.join(', ')} (missing: ${missing.join(', ')})`,
      { missing },
    );
  }

  const pollMs = Number(trimmed(env.TRUFFLE_PARTNER_POLL_MS)) || DEFAULT_POLL_MS;
  const timeout = Number(trimmed(env.KOLM_CODE_TRUFFLE_PROCESSOR_TIMEOUT_MS))
    || processorTimeoutMs
    || DEFAULT_PROCESSOR_TIMEOUT_MS;
  const processorId = trimmed(env.KOLM_TRUFFLE_HARNESS_ID) || 'kolm-code';

  const client = createTruffleClient({
    apiUrl: env.TRUFFLE_PARTNER_API_URL,
    serviceToken: env.TRUFFLE_PARTNER_SERVICE_TOKEN,
    workspaceId: env.TRUFFLE_PARTNER_WORKSPACE_ID,
    keyId: env.TRUFFLE_PARTNER_PROCESSOR_KEY_ID,
    signingSecret: env.TRUFFLE_PARTNER_PROCESSOR_SIGNING_SECRET,
    processorUrl: trimmed(env.TRUFFLE_PARTNER_PROCESSOR_URL) || undefined,
    releaseSha: trimmed(env.TRUFFLE_RELEASE_SHA) || undefined,
    processorId,
    fetchImpl,
    now,
    randomUUID,
    sleep: sleep ? (ms) => sleep(ms) : undefined,
  });

  return runWorker({
    client,
    processorId,
    pollMs,
    handler,
    logger,
    signal,
    maxTasks,
    processorTimeoutMs: timeout,
    health: health === undefined ? parseHealthHost(env.TRUFFLE_PARTNER_HEALTH_HOST) : health,
    sleep,
  });
}

export { REFERENCE_PROCESSOR_TIMEOUT_MS };

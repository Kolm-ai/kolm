// kolm code — Truffle channel-workflow partner adapter (owner F, spec 9 + 11.2).
//
// Wire-compatible with truffle-chat's
// `src/lib/channel-workflow-partner/contract.ts` (1067 lines) and
// `reference-worker.ts` (872 lines). Nothing is imported from that repo: every
// constant, header name and canonical string below was copied from it, with the
// contract line number that defines it.
//
//   claim   POST {apiUrl}/api/channel-workflow-partner/v1/claim   (reference-worker.ts:411)
//   settle  POST {apiUrl}/api/channel-workflow-partner/v1/settle  (reference-worker.ts:412)
//
// Service auth headers (contract.ts:228-241, reference-worker.ts:248-254):
//   authorization: Bearer <43-char base64url token>
//   x-truffle-service-audience: truffle-collaboration
//   x-truffle-workspace-id: <uuid>
//
// Signature headers (contract.ts:873-903) and canonical string
// (contract.ts:655-674):
//   truffle:partner-processor-request:v1 \n POST \n <target> \n <timestamp>
//   \n <keyId> \n <idempotencyKey> \n <bodyDigest>
// where <target> is `${url.origin}${url.pathname}${url.search}` — the full
// absolute URL without the fragment, NOT the path alone (contract.ts:886,
// contract.ts:743-745) — <timestamp> is whole seconds (contract.ts:884,
// validated by contract.ts:736-741) and <bodyDigest> is the sha256 hex of the
// exact request body bytes (contract.ts:651-653).

import { createHash, createHmac, randomUUID as nodeRandomUUID, timingSafeEqual } from 'node:crypto';
import { setTimeout as delay } from 'node:timers/promises';

import { LadderError } from '../errors.js';

/* ------------------------------------------------------------------ *
 * Contract constants (contract.ts:9-37, reference-worker.ts:34-44)
 * ------------------------------------------------------------------ */

export const PARTNER_AUDIENCE = 'truffle-collaboration';
export const PARTNER_TASK_SCHEMA = 'truffle.channel-partner-task.v1';
export const PARTNER_CLAIM_SCHEMA = 'truffle.partner-work-claim.v1';
export const PARTNER_CLAIM_REQUEST_SCHEMA = 'truffle.partner-work-claim-request.v1';
export const PARTNER_READINESS_SCHEMA = 'truffle.partner-processor-readiness.v1';
export const PARTNER_SETTLEMENT_SCHEMA = 'truffle.partner-work-settlement.v1';
export const PARTNER_SETTLEMENT_RESULT_SCHEMA = 'truffle.partner-work-settlement-result.v1';
export const PARTNER_REQUEST_SCHEMA = 'truffle.partner-work-request.v1';
export const PARTNER_REPLAY_CONTRACT = 'truffle.partner-durable-replay.v1';

export const LEASE_TTL_SECONDS = 120;
export const READINESS_LEASE_TTL_SECONDS = 60;
export const MAX_OUTPUT_BYTES = 32_000;
export const MAX_GATEWAY_RESPONSE_BYTES = 48_000;
export const SIGNATURE_MAX_AGE_SECONDS = 300;
export const REQUEST_TIMEOUT_MS = 10_000;
/** Truffle's own processor budget: 20 s, sized for chat-latency steps. */
export const REFERENCE_PROCESSOR_TIMEOUT_MS = 20_000;
export const DEFAULT_POLL_MS = 1_000;
export const DEFAULT_ATTEMPTS = 3;
export const DEFAULT_BACKOFF_MS = Object.freeze([250, 500, 1000, 2000]);
export const MAX_URL_LENGTH = 2_048;

/** Header names, verbatim from contract.ts:887-902. */
export const HEADER_AUTHORIZATION = 'authorization';
export const HEADER_AUDIENCE = 'x-truffle-service-audience';
export const HEADER_WORKSPACE = 'x-truffle-workspace-id';
export const HEADER_CONTENT_TYPE = 'content-type';
export const HEADER_IDEMPOTENCY_KEY = 'idempotency-key';
export const HEADER_CONTRACT = 'x-truffle-partner-contract';
export const HEADER_CONTENT_SHA256 = 'x-truffle-partner-content-sha256';
export const HEADER_KEY_ID = 'x-truffle-partner-key-id';
export const HEADER_REPLAY_CONTRACT = 'x-truffle-partner-replay-contract';
export const HEADER_TIMESTAMP = 'x-truffle-partner-timestamp';
export const HEADER_SIGNATURE = 'x-truffle-partner-signature';

export const CLAIM_PATH = '/api/channel-workflow-partner/v1/claim';
export const SETTLE_PATH = '/api/channel-workflow-partner/v1/settle';

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const HEX64 = /^[0-9a-f]{64}$/;
const MESSAGE_ID = /^[A-Za-z0-9_-]{1,80}$/;
const KEY_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$/;
const CANONICAL_TOKEN = /^[A-Za-z0-9_-]{43}$/;
const SIGNATURE = /^v1=([0-9a-f]{64})$/;
const CONTROL = /[\u0000\u007f]/g;
const GITHUB_URL = /https:\/\/github\.com\/[\w.-]+\/[\w.-]+/;
const BASE_REF = /(?:^|\s)base:([\w./-]+)/;

/* ------------------------------------------------------------------ *
 * Signing primitives
 * ------------------------------------------------------------------ */

/**
 * Truffle secrets are canonical 32-byte base64url tokens (contract.ts:1032-1041).
 * A raw Buffer/Uint8Array is used as-is; any other string is taken as utf8 so a
 * non-canonical deployment secret still produces a stable signature.
 * @param {string|Uint8Array} secret
 * @returns {Buffer}
 */
export function secretBytes(secret) {
  if (secret instanceof Uint8Array) return Buffer.from(secret);
  if (typeof secret !== 'string' || secret === '') {
    throw new LadderError('truffle-config', 'A partner signing secret is required');
  }
  if (CANONICAL_TOKEN.test(secret)) {
    const bytes = Buffer.from(secret, 'base64url');
    if (bytes.byteLength === 32 && bytes.toString('base64url') === secret) return bytes;
  }
  return Buffer.from(secret, 'utf8');
}

/** sha256 hex of the exact body bytes (contract.ts:651-653). */
export function partnerBodyDigest(body) {
  return createHash('sha256').update(body).digest('hex');
}

/** `${origin}${pathname}${search}` (contract.ts:743-745, contract.ts:886). */
export function processorTarget(url) {
  const parsed = url instanceof URL ? url : new URL(String(url));
  return `${parsed.origin}${parsed.pathname}${parsed.search}`;
}

/** The signed request string, contract.ts:665-673. */
export function canonicalRequestString({ target, timestamp, keyId, idempotencyKey, bodyDigest }) {
  return [
    'truffle:partner-processor-request:v1',
    'POST',
    target,
    String(timestamp),
    keyId,
    idempotencyKey,
    bodyDigest,
  ].join('\n');
}

/** The signed response string, contract.ts:686-693. */
export function canonicalResponseString({ status, timestamp, keyId, idempotencyKey, bodyDigest }) {
  return [
    'truffle:partner-processor-response:v1',
    String(status),
    String(timestamp),
    keyId,
    idempotencyKey,
    bodyDigest,
  ].join('\n');
}

/**
 * `v1=<hex64>` HMAC-SHA256 over the canonical request string.
 * @returns {string}
 */
export function signPartnerRequest({ signingSecret, target, timestamp, keyId, idempotencyKey, bodyDigest }) {
  const canonical = canonicalRequestString({ target, timestamp, keyId, idempotencyKey, bodyDigest });
  return `v1=${createHmac('sha256', secretBytes(signingSecret)).update(canonical, 'utf8').digest('hex')}`;
}

/**
 * The eight signed headers Truffle sends on a partner processor request
 * (contract.ts:887-902), reproduced key for key.
 * @returns {Record<string, string>}
 */
export function signedRequestHeaders({ target, body, idempotencyKey, keyId, signingSecret, now = Date.now }) {
  if (!KEY_ID.test(String(keyId ?? ''))) {
    throw new LadderError('truffle-config', 'Partner processor key id is invalid', { keyId: keyId ?? null });
  }
  if (!HEX64.test(String(idempotencyKey ?? ''))) {
    throw new LadderError('truffle-config', 'Partner idempotency key must be 64 lowercase hex characters');
  }
  const at = typeof now === 'function' ? now() : now;
  const timestamp = String(Math.floor(at / 1000));
  const bodyDigest = partnerBodyDigest(body);
  const canonicalTarget = processorTarget(target);
  return {
    [HEADER_CONTENT_TYPE]: 'application/json',
    [HEADER_IDEMPOTENCY_KEY]: idempotencyKey,
    [HEADER_CONTRACT]: PARTNER_REQUEST_SCHEMA,
    [HEADER_CONTENT_SHA256]: bodyDigest,
    [HEADER_KEY_ID]: keyId,
    [HEADER_REPLAY_CONTRACT]: PARTNER_REPLAY_CONTRACT,
    [HEADER_TIMESTAMP]: timestamp,
    [HEADER_SIGNATURE]: signPartnerRequest({
      signingSecret,
      target: canonicalTarget,
      timestamp,
      keyId,
      idempotencyKey,
      bodyDigest,
    }),
  };
}

function headerReader(headers) {
  if (!headers) return () => null;
  if (typeof headers.get === 'function') return (name) => headers.get(name);
  const lower = new Map();
  for (const [name, value] of Object.entries(headers)) {
    lower.set(String(name).toLowerCase(), Array.isArray(value) ? value[0] : value);
  }
  return (name) => (lower.has(name) ? String(lower.get(name)) : null);
}

function signatureBytes(value) {
  const match = SIGNATURE.exec(String(value ?? ''));
  return match ? Buffer.from(match[1], 'hex') : null;
}

/**
 * The reverse direction: verify a signed partner request we received (and,
 * in tests, prove our own signing is symmetric). Constant-time comparison.
 * @returns {boolean}
 */
export function verifyProcessorSignature({
  headers,
  body,
  target,
  signingSecret,
  now = Date.now,
  skewMs = SIGNATURE_MAX_AGE_SECONDS * 1000,
  expectedKeyId = null,
}) {
  const read = headerReader(headers);
  const timestamp = read(HEADER_TIMESTAMP);
  const keyId = read(HEADER_KEY_ID);
  const idempotencyKey = read(HEADER_IDEMPOTENCY_KEY);
  const presented = read(HEADER_SIGNATURE);
  const digestHeader = read(HEADER_CONTENT_SHA256);

  if (!timestamp || !/^[1-9][0-9]{9}$/.test(String(timestamp))) return false;
  if (!keyId || !KEY_ID.test(keyId)) return false;
  if (expectedKeyId !== null && keyId !== expectedKeyId) return false;
  if (!idempotencyKey || !HEX64.test(idempotencyKey)) return false;

  const at = typeof now === 'function' ? now() : now;
  if (Math.abs(at - Number(timestamp) * 1000) > skewMs) return false;

  const bodyDigest = partnerBodyDigest(body ?? '');
  if (digestHeader !== null && digestHeader !== bodyDigest) return false;

  const expected = signatureBytes(signPartnerRequest({
    signingSecret,
    target: processorTarget(target),
    timestamp,
    keyId,
    idempotencyKey,
    bodyDigest,
  }));
  const candidate = signatureBytes(presented);
  if (!expected || !candidate || expected.byteLength !== candidate.byteLength) return false;
  return timingSafeEqual(candidate, expected);
}

/* ------------------------------------------------------------------ *
 * Readiness attestation
 * ------------------------------------------------------------------ */

function hexFrom(...parts) {
  return createHash('sha256').update(parts.join('\u0000'), 'utf8').digest('hex');
}

function derivedUuid(seed) {
  const hex = hexFrom('truffle:partner-processor-instance', seed);
  const variant = '89ab'[parseInt(hex[16], 16) % 4];
  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    `4${hex.slice(13, 16)}`,
    `${variant}${hex.slice(17, 20)}`,
    hex.slice(20, 32),
  ].join('-');
}

/**
 * `truffle.partner-processor-readiness.v1` (contract.ts:63-75, produced by
 * reference-worker.ts:581-592). Exactly the ten contract keys are enumerable,
 * so `JSON.stringify` yields a document Truffle's `exactRecord` accepts;
 * `capabilities` rides along as a non-enumerable property for local reporting.
 * @returns {object}
 */
export function readinessDocument({
  processorId = 'kolm-code',
  version = '0',
  capabilities = [],
  processorUrl = 'https://kolm.ai/kolm-code/processor',
  keyId,
  releaseSha,
  leaseToken,
  instanceId,
  processStartedAt,
  now = Date.now,
} = {}) {
  const at = typeof now === 'function' ? now() : now;
  const observedAt = new Date(at).toISOString();
  const started = processStartedAt || observedAt;
  const endpointDigest = createHash('sha256')
    .update('truffle:partner-processor-endpoint:v1\u0000', 'utf8')
    .update(String(processorUrl), 'utf8')
    .digest('hex');

  const doc = {
    schema: PARTNER_READINESS_SCHEMA,
    processorInstanceId: instanceId && UUID.test(instanceId)
      ? instanceId.toLowerCase()
      : derivedUuid(`${processorId}@${version}`),
    releaseSha: releaseSha && /^[0-9a-f]{40}$/.test(releaseSha)
      ? releaseSha
      : hexFrom('truffle:partner-release', String(processorId), String(version)).slice(0, 40),
    processStartedAt: started,
    observedAt,
    leaseToken: leaseToken && HEX64.test(leaseToken)
      ? leaseToken
      : hexFrom('truffle:partner-readiness-lease', String(processorId), started),
    leaseTtlSeconds: READINESS_LEASE_TTL_SECONDS,
    processorEndpointDigest: endpointDigest,
    processorKeyId: KEY_ID.test(String(keyId ?? '')) ? String(keyId) : String(processorId),
    partnerAdapterKey: PARTNER_TASK_SCHEMA,
  };
  Object.defineProperty(doc, 'capabilities', {
    value: Object.freeze([...capabilities]),
    enumerable: false,
  });
  return doc;
}

/* ------------------------------------------------------------------ *
 * Messages
 * ------------------------------------------------------------------ */

/**
 * Clamps one thread message to the contract's output budget: at most
 * `maxBytes` utf8 bytes AND at most `maxBytes` characters, no control
 * characters (contract.ts:206-215, contract.ts:33).
 * @returns {string}
 */
export function clampMessage(text, maxBytes = MAX_OUTPUT_BYTES) {
  let out = String(text ?? '').replace(CONTROL, '');
  if (out.length <= maxBytes && Buffer.byteLength(out, 'utf8') <= maxBytes) return out;
  const suffix = ' [truncated]';
  const budget = Math.max(0, maxBytes - suffix.length);
  let low = 0;
  let high = Math.min(out.length, budget);
  while (low < high) {
    const mid = Math.ceil((low + high) / 2);
    if (Buffer.byteLength(out.slice(0, mid), 'utf8') <= budget) low = mid;
    else high = mid - 1;
  }
  out = out.slice(0, low);
  // never end on a lone high surrogate
  if (out.length > 0 && /[\uD800-\uDBFF]$/.test(out)) out = out.slice(0, -1);
  return `${out}${suffix}`;
}

function verdictLabel(verdict, error) {
  if (error) return 'ERROR';
  const value = String(verdict ?? '').toUpperCase();
  if (value === 'PASS' || value === 'PASSED') return 'PASS';
  if (value === 'FAIL' || value === 'FAILED') return 'FAIL';
  if (value === 'JUDGE' || value === 'REVISE') return 'JUDGE';
  return value === '' ? 'ERROR' : value;
}

function summarizeStages(source) {
  if (!source) return '';
  const parts = [];
  if (Array.isArray(source)) {
    for (const entry of source) {
      if (!entry) continue;
      const stage = entry.stage ?? entry.id ?? entry.name;
      const decision = entry.decision ?? entry.verdict;
      if (stage) parts.push(`${stage}: ${decision ?? ''}`.trim());
    }
  } else if (typeof source === 'object') {
    for (const [stage, entry] of Object.entries(source)) {
      const decision = entry && typeof entry === 'object' ? (entry.decision ?? entry.verdict) : entry;
      parts.push(`${stage}: ${decision ?? ''}`.trim());
    }
  }
  return parts.join(', ');
}

function errorText(error) {
  if (!error) return '';
  if (typeof error === 'string') return error;
  if (error instanceof Error) return error.message;
  if (typeof error.message === 'string') return error.message;
  return String(error);
}

/**
 * One plain-text thread message — the only artifact Truffle accepts
 * (spec 11.2). No markdown tables, no stack traces, no credentials.
 * @returns {string} at most 32000 bytes
 */
export function formatSettlementMessage({
  verdict,
  receipt = null,
  prUrl = null,
  costUsd = null,
  summary = null,
  error = null,
  title = null,
  stages = null,
  runId = null,
} = {}) {
  const label = verdictLabel(verdict, error);
  const heading = title
    || (typeof receipt?.task === 'string' ? receipt.task : receipt?.task?.title)
    || receipt?.task?.text
    || 'kolm code run';
  const id = runId || receipt?.run_id || receipt?.runId || null;

  const lines = [`kolm code: ${label} -- ${String(heading).replace(/\s+/g, ' ').trim()}`];
  if (prUrl) lines.push(`- pull request: ${prUrl}`);
  if (typeof costUsd === 'number' && Number.isFinite(costUsd)) {
    lines.push(`- cost: $${costUsd.toFixed(4)}`);
  } else if (typeof receipt?.cost?.total_usd === 'number') {
    lines.push(`- cost: $${receipt.cost.total_usd.toFixed(4)}`);
  }
  const cascade = summarizeStages(stages ?? receipt?.stages ?? null);
  if (cascade) lines.push(`- cascade: ${cascade}`);
  if (summary) lines.push(`- summary: ${String(summary).replace(/\s+/g, ' ').trim()}`);
  if (error) lines.push(`- error: ${errorText(error).replace(/\s+/g, ' ').trim()}`);
  if (id) lines.push(`- receipt run id: ${id}`);
  lines.push(`Verify: kolm code receipt verify ${id || '<run-id>'}`);

  return clampMessage(lines.join('\n'));
}

/* ------------------------------------------------------------------ *
 * Task mapping
 * ------------------------------------------------------------------ */

/**
 * Maps a claimed partner task (the `truffle.partner-work-claim.v1` envelope, or
 * a bare `truffle.channel-partner-task.v1` task) onto the shape `runTask` wants.
 * @param {object} claimed
 * @returns {{taskId: string|null, text: string, repoUrl: string|null, baseRef: string|null, channel: string|null, threadId: string|null, requestedBy: string|null, raw: object}}
 */
export function taskToKolmRequest(claimed) {
  const claim = claimed && typeof claimed === 'object' ? claimed : {};
  const task = claim.task && typeof claim.task === 'object' ? claim.task : claim;
  const triggerText = typeof claim.triggerText === 'string' ? claim.triggerText : '';
  const instructions = typeof task.instructions === 'string' ? task.instructions : '';
  const text = triggerText || instructions;
  const haystack = `${instructions}\n${triggerText}`;

  const repoMatch = GITHUB_URL.exec(haystack);
  const baseMatch = BASE_REF.exec(haystack);

  return {
    taskId: claim.runId ?? task.workflowInstanceId ?? null,
    text,
    instructions,
    repoUrl: repoMatch ? repoMatch[0].replace(/[.,)]+$/, '') : null,
    baseRef: baseMatch ? baseMatch[1] : null,
    channel: task.delivery?.rootMessageId ?? claim.threadRootId ?? null,
    threadId: claim.threadRootId ?? task.output?.rootMessageId ?? null,
    requestedBy: task.role ?? null,
    idempotencyKey: claim.idempotencyKey ?? null,
    stepId: task.stepId ?? null,
    maxOutputBytes: task.output?.maxBytes ?? MAX_OUTPUT_BYTES,
    raw: claimed,
  };
}

/* ------------------------------------------------------------------ *
 * Gateway client
 * ------------------------------------------------------------------ */

function retryableStatus(status) {
  // reference-worker.ts:256-258
  return status === 408 || status === 425 || status === 429 || status >= 500;
}

function excerpt(text, secrets) {
  let out = String(text ?? '').slice(0, 300);
  for (const secret of secrets) {
    if (typeof secret === 'string' && secret.length >= 8) out = out.split(secret).join('[redacted]');
  }
  return out;
}

async function defaultSleep(ms) {
  await delay(ms).catch(() => undefined);
}

function validateClaim(value) {
  const claim = value && typeof value === 'object' ? value : null;
  const task = claim?.task && typeof claim.task === 'object' ? claim.task : null;
  const problems = [];
  if (!claim) problems.push('claim is not an object');
  else {
    if (claim.schema !== PARTNER_CLAIM_SCHEMA) problems.push('schema');
    if (typeof claim.runId !== 'string' || !UUID.test(claim.runId)) problems.push('runId');
    if (!Number.isSafeInteger(claim.version) || claim.version < 1) problems.push('version');
    if (!Number.isSafeInteger(claim.attempt) || claim.attempt < 1) problems.push('attempt');
    if (typeof claim.idempotencyKey !== 'string' || !HEX64.test(claim.idempotencyKey)) {
      problems.push('idempotencyKey');
    }
    if (typeof claim.triggerText !== 'string' || claim.triggerText.length > 32_000) {
      problems.push('triggerText');
    }
    if (typeof claim.threadRootId !== 'string' || !MESSAGE_ID.test(claim.threadRootId)) {
      problems.push('threadRootId');
    }
    if (typeof claim.leaseToken !== 'string' || !HEX64.test(claim.leaseToken)) {
      problems.push('leaseToken');
    }
    if (!task || task.schema !== PARTNER_TASK_SCHEMA) problems.push('task');
    else if (typeof task.stepCapability !== 'string' || !HEX64.test(task.stepCapability)) {
      problems.push('task.stepCapability');
    }
  }
  return problems;
}

/**
 * Creates the Truffle gateway client. Every dependency is injectable so the
 * whole client is testable against an in-process server.
 * @param {object} opts
 * @returns {{claim: Function, settle: Function, readiness: Function, apiUrl: string, keyId: string}}
 */
export function createTruffleClient({
  apiUrl,
  serviceToken,
  workspaceId,
  keyId,
  signingSecret,
  processorId = 'kolm-code',
  processorUrl,
  releaseSha,
  version = '0',
  capabilities = ['kolm-code'],
  fetchImpl = fetch,
  now = () => Date.now(),
  timeoutMs = REQUEST_TIMEOUT_MS,
  retries = DEFAULT_ATTEMPTS,
  backoffMs = DEFAULT_BACKOFF_MS,
  randomUUID = nodeRandomUUID,
  sleep = defaultSleep,
} = {}) {
  const missing = [];
  if (!apiUrl) missing.push('apiUrl');
  if (!serviceToken) missing.push('serviceToken');
  if (!workspaceId) missing.push('workspaceId');
  if (!keyId) missing.push('keyId');
  if (!signingSecret) missing.push('signingSecret');
  if (missing.length > 0) {
    throw new LadderError(
      'not-armed',
      `Truffle partner client needs ${missing.join(', ')}`,
      { missing },
    );
  }
  const root = String(apiUrl).replace(/\/+$/, '');
  if (root.length > MAX_URL_LENGTH) {
    throw new LadderError('truffle-config', 'TRUFFLE_PARTNER_API_URL is too long', { max: MAX_URL_LENGTH });
  }
  const claimUrl = `${root}${CLAIM_PATH}`;
  const settleUrl = `${root}${SETTLE_PATH}`;
  const processStartedAt = new Date(now()).toISOString();
  const attempts = Math.max(1, Number(retries) || 1);
  const backoff = Array.isArray(backoffMs) && backoffMs.length > 0 ? backoffMs : DEFAULT_BACKOFF_MS;

  function freshIdempotencyKey() {
    return createHash('sha256').update(String(randomUUID()), 'utf8').digest('hex');
  }

  function serviceHeaders() {
    return {
      [HEADER_AUTHORIZATION]: `Bearer ${serviceToken}`,
      [HEADER_AUDIENCE]: PARTNER_AUDIENCE,
      [HEADER_WORKSPACE]: String(workspaceId).toLowerCase(),
    };
  }

  async function readBounded(response, label) {
    let text = '';
    try {
      text = typeof response.text === 'function' ? await response.text() : '';
    } catch (err) {
      throw new LadderError('truffle-protocol', `${label} response body failed`, {
        reason: err?.code || err?.name || 'read-failed',
      });
    }
    if (Buffer.byteLength(text, 'utf8') > MAX_GATEWAY_RESPONSE_BYTES) {
      throw new LadderError('truffle-protocol', `${label} response exceeded its byte limit`, {
        max: MAX_GATEWAY_RESPONSE_BYTES,
      });
    }
    return text;
  }

  /**
   * One logical request, retried in place: retries reuse the same idempotency
   * key (so Truffle can dedupe), a fresh logical request always gets a new one.
   */
  async function send({ url, body, idempotencyKey, label, signal }) {
    let attempt = 0;
    for (;;) {
      attempt += 1;
      if (signal?.aborted) {
        throw new LadderError('truffle-aborted', `${label} was aborted`, { label });
      }
      const headers = {
        ...serviceHeaders(),
        ...signedRequestHeaders({
          target: url,
          body,
          idempotencyKey,
          keyId,
          signingSecret,
          now,
        }),
      };

      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      if (typeof timer.unref === 'function') timer.unref();
      const forward = () => controller.abort();
      if (signal) signal.addEventListener('abort', forward, { once: true });

      let response = null;
      let networkError = null;
      try {
        response = await fetchImpl(url, {
          method: 'POST',
          headers,
          body,
          cache: 'no-store',
          redirect: 'error',
          signal: controller.signal,
        });
      } catch (err) {
        networkError = err;
      } finally {
        clearTimeout(timer);
        if (signal) signal.removeEventListener('abort', forward);
      }

      if (networkError) {
        if (signal?.aborted) {
          throw new LadderError('truffle-aborted', `${label} was aborted`, { label });
        }
        if (attempt < attempts) {
          await sleep(backoff[Math.min(attempt - 1, backoff.length - 1)]);
          continue;
        }
        throw new LadderError('truffle-gateway', `${label} request failed`, {
          attempts: attempt,
          reason: excerpt(networkError?.message || networkError?.code || 'network-error', [serviceToken]),
        });
      }

      const status = Number(response?.status ?? 0);
      if (response?.ok) {
        const text = await readBounded(response, label);
        try {
          return JSON.parse(text);
        } catch {
          throw new LadderError('truffle-protocol', `${label} response was not valid JSON`, { status });
        }
      }

      const text = await readBounded(response, label).catch(() => '');
      if (retryableStatus(status) && attempt < attempts) {
        await sleep(backoff[Math.min(attempt - 1, backoff.length - 1)]);
        continue;
      }
      throw new LadderError('truffle-gateway', `${label} failed with HTTP ${status}`, {
        status,
        attempts: attempt,
        retryable: retryableStatus(status),
        body: excerpt(text, [serviceToken, String(signingSecret)]),
      });
    }
  }

  function readiness(overrides = {}) {
    return readinessDocument({
      processorId,
      version,
      capabilities,
      processorUrl: processorUrl || `${root}${CLAIM_PATH}`,
      keyId,
      releaseSha,
      processStartedAt,
      now,
      ...overrides,
    });
  }

  return Object.freeze({
    apiUrl: root,
    claimUrl,
    settleUrl,
    keyId,
    readiness,

    /**
     * `POST /claim`. `capacity` is a local scheduling hint only: the claim body
     * is exactly `{schema, processorReadiness}` (contract.ts:74-77), which
     * Truffle validates with an exact-key record.
     * @returns {Promise<{task: object|null, claim: object|null, raw: object}>}
     */
    async claim({ processorId: overrideId, capacity: _capacity = 1, signal, readiness: doc } = {}) {
      const attestation = doc || readiness(overrideId ? { processorId: overrideId } : {});
      const body = JSON.stringify({
        schema: PARTNER_CLAIM_REQUEST_SCHEMA,
        processorReadiness: attestation,
      });
      const payload = await send({
        url: claimUrl,
        body,
        idempotencyKey: freshIdempotencyKey(),
        label: 'Truffle claim',
        signal,
      });
      if (!payload || typeof payload !== 'object' || !('claim' in payload)) {
        throw new LadderError('truffle-protocol', 'Truffle returned an invalid partner claim', {
          keys: payload && typeof payload === 'object' ? Object.keys(payload).slice(0, 8) : null,
        });
      }
      if (payload.claim === null) return { task: null, claim: null, raw: payload };
      const problems = validateClaim(payload.claim);
      if (problems.length > 0) {
        throw new LadderError('truffle-protocol', 'Truffle returned an invalid partner claim', { problems });
      }
      return { task: payload.claim, claim: payload.claim, raw: payload };
    },

    /**
     * `POST /settle`. The wire body is exactly the contract's five keys
     * (contract.ts:120-131); `status` and `error` shape the thread message,
     * which is the only artifact Truffle accepts.
     * @returns {Promise<{schema: string, runId: string, status: string, messageId: string|null, replayed: boolean, requestedStatus: string}>}
     */
    async settle({
      task = null,
      taskId,
      runVersion,
      leaseToken,
      capability,
      idempotencyKey,
      claimToken,
      status = 'completed',
      message,
      error = null,
      signal,
    } = {}) {
      const claim = task && typeof task === 'object' ? (task.claim ?? task) : null;
      const runId = taskId ?? claim?.runId ?? null;
      const version = runVersion ?? claim?.version ?? 1;
      const lease = leaseToken ?? claimToken ?? claim?.leaseToken ?? null;
      const cap = capability ?? claim?.task?.stepCapability ?? claim?.stepCapability ?? null;
      const key = idempotencyKey ?? claim?.idempotencyKey ?? null;

      const problems = [];
      if (typeof runId !== 'string' || !UUID.test(runId)) problems.push('runId');
      if (typeof lease !== 'string' || !HEX64.test(lease)) problems.push('leaseToken');
      if (typeof cap !== 'string' || !HEX64.test(cap)) problems.push('capability');
      if (typeof key !== 'string' || !HEX64.test(key)) problems.push('idempotencyKey');
      if (problems.length > 0) {
        throw new LadderError('truffle-config', 'Partner settlement is missing claim identity', { problems });
      }

      let text = message ?? formatSettlementMessage({
        verdict: status === 'failed' ? 'ERROR' : 'PASS',
        error,
        runId,
      });
      if (status === 'failed' && error) {
        const detail = errorText(error).replace(/\s+/g, ' ').trim();
        if (detail && !text.includes(detail)) text = `${text}\n- error: ${detail}`;
      }
      text = clampMessage(text);

      const body = JSON.stringify({
        schema: PARTNER_SETTLEMENT_SCHEMA,
        run: { id: runId, version, leaseToken: lease },
        capability: cap,
        operationId: `partner.settle.${key}`,
        output: { kind: 'thread-message', text },
      });

      const payload = await send({
        url: settleUrl,
        body,
        idempotencyKey: key,
        label: 'Truffle settlement',
        signal,
      });

      if (
        !payload || typeof payload !== 'object' ||
        payload.schema !== PARTNER_SETTLEMENT_RESULT_SCHEMA ||
        typeof payload.runId !== 'string' || payload.runId.toLowerCase() !== runId.toLowerCase() ||
        (payload.status !== 'succeeded' && payload.status !== 'rejected') ||
        typeof payload.replayed !== 'boolean'
      ) {
        throw new LadderError('truffle-protocol', 'Truffle returned an invalid partner settlement', {
          runId,
        });
      }
      return {
        schema: payload.schema,
        runId: payload.runId.toLowerCase(),
        status: payload.status,
        messageId: payload.messageId ?? null,
        replayed: payload.replayed,
        requestedStatus: status,
        text,
      };
    },
  });
}

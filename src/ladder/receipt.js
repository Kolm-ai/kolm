// kolm code — the signed run receipt (spec 6.5). Owner D.
//
// The receipt is the ONLY artifact that outlives the run's database, so it has
// two hard constraints:
//
//   1. It must be reproducible-canonical. Signing covers
//      canonicalJson(receipt-without-signature), so field order can never
//      matter and every absent value is an explicit `null` — `undefined`
//      disappears from JSON.stringify and would silently change the payload.
//   2. It must never carry a secret. No tokens, no env values, no raw prompts,
//      no file contents, no diff text. Hashes and counts only. `scrubSecrets`
//      is a second line of defence over the free-text fields (stage reasons,
//      judge rationale) that come from models and from repo contents.

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

import { canonicalJson } from '../cid.js';
import {
  loadOrCreateDefaultSigner,
  buildSignatureBlock,
  verifySignatureBlock,
} from '../ed25519.js';
import { LadderError } from './errors.js';

export const RECEIPT_SCHEMA = 'kolm-code-receipt-v1';
export const RECEIPT_VERSION = 2;
/** Receipt versions this build will verify. v1 carried the full repo path. */
export const SUPPORTED_RECEIPT_VERSIONS = Object.freeze([1, 2]);

const REASON_CAP = 300;

/** Patterns that must never reach a receipt file. */
const SECRET_PATTERNS = [
  /sk-ant-[A-Za-z0-9_-]{8,}/g,
  /sk-[A-Za-z0-9]{20,}/g,
  /gh[pousr]_[A-Za-z0-9]{16,}/g,
  /AKIA[0-9A-Z]{12,}/g,
  /Bearer\s+[A-Za-z0-9._-]{8,}/g,
  /xox[abprs]-[A-Za-z0-9-]{8,}/g,
];

/** sha256 hex of a string (null in -> null out). */
export function sha256(text) {
  if (text === null || text === undefined) return null;
  return crypto.createHash('sha256').update(String(text), 'utf8').digest('hex');
}

/** sha256 of the canonical form of any JSON value. */
export function sha256Canonical(value) {
  if (value === null || value === undefined) return null;
  return sha256(canonicalJson(value));
}

/**
 * Replace anything that looks like a credential with a marker.
 * @param {string} text
 * @returns {string}
 */
export function scrubSecrets(text) {
  let out = String(text ?? '');
  for (const re of SECRET_PATTERNS) out = out.replace(re, '[redacted]');
  return out;
}

/**
 * The last path segment of a local repo path. A receipt is a portable artifact;
 * "/home/alice/work/secret-client" is not something it should carry.
 * @param {string|null} repoPath
 * @returns {string|null}
 */
export function repoBasename(repoPath) {
  if (repoPath === null || repoPath === undefined) return null;
  const text = String(repoPath).trim().replace(/[\\/]+$/, '');
  if (text === '') return null;
  const parts = text.split(/[\\/]/);
  const last = parts[parts.length - 1];
  return last === '' ? null : cleanString(last, 200);
}

function cleanString(value, cap) {
  if (value === null || value === undefined) return null;
  const scrubbed = scrubSecrets(value);
  return scrubbed.length > cap ? `${scrubbed.slice(0, cap)}...` : scrubbed;
}

function cleanReasons(reasons) {
  if (!Array.isArray(reasons)) return [];
  return reasons.slice(0, 12).map((r) => cleanString(r, REASON_CAP)).filter((r) => r !== null);
}

/** Only finite numbers, booleans and short strings survive into metrics. */
function cleanMetrics(metrics) {
  const out = {};
  if (!metrics || typeof metrics !== 'object') return out;
  for (const [key, value] of Object.entries(metrics)) {
    if (typeof value === 'number' && Number.isFinite(value)) out[key] = value;
    else if (typeof value === 'boolean') out[key] = value;
    else if (typeof value === 'string') out[key] = cleanString(value, 120);
  }
  return out;
}

function num(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function pick(obj, ...keys) {
  for (const key of keys) {
    if (obj && obj[key] !== undefined && obj[key] !== null) return obj[key];
  }
  return null;
}

function verdictOf(entry) {
  const v = pick(entry, 'verdict');
  if (typeof v === 'string') return v;
  if (v && typeof v === 'object' && typeof v.verdict === 'string') return v.verdict;
  return null;
}

/**
 * Build an unsigned receipt. Pure: it reads the rows it is given and never
 * touches the store or the filesystem.
 *
 * @param {object} args
 * @param {object} args.run          store run row (id, task, repo_path, base_sha, mode)
 * @param {Array}  [args.candidates] store candidate rows or {candidateId, generatorRef, verdict}
 * @param {Array}  [args.results]    store.resultsFor(runId) rows
 * @param {Array}  [args.judgments]  store.listJudgments(runId) rows
 * @param {object} [args.cost]       store.costSummary(runId)
 * @param {string|null} [args.winner] winning candidate id
 * @param {string|null} [args.branch]
 * @param {string|null} [args.sha]
 * @param {string} [args.mode]
 * @param {object} [args.bands]
 * @param {object} [args.rubric]
 * @param {object} [args.config]
 * @param {object} [args.models]     {orchestrator, judge, coders:[], reviewer}
 * @param {object} [args.calibration] {sampled, agreementRate|agreement_rate}
 * @param {string|null} [args.stopReason]
 * @param {string} [args.verdict]    PASS | FAIL | JUDGE | ERROR
 * @param {Array} [args.subtasks]    subtask winners (spec 4.4); [] for a single-unit run.
 *   A subtask that failed or was SKIPPED (its dependency failed) is listed too,
 *   with null winner/branch/sha: the ledger says what the run attempted, and the
 *   per-subtask verdict lives in the events.
 * @returns {object} receipt (unsigned)
 */
export function buildReceipt({
  run = {},
  candidates = [],
  results = [],
  judgments = [],
  cost = {},
  winner = null,
  branch = null,
  sha = null,
  mode = null,
  bands = null,
  rubric = null,
  config = {},
  models = {},
  calibration = null,
  stopReason = null,
  verdict = null,
  createdAt = null,
  signerKid = null,
  subtasks = [],
} = {}) {
  const task = run.task && typeof run.task === 'object' ? run.task : {};
  const candidateRows = (candidates || []).map((c) => ({
    candidate_id: pick(c, 'candidate_id', 'candidateId'),
    generator: pick(c, 'generator', 'generator_ref', 'generatorRef'),
    verdict: verdictOf(c),
  })).filter((c) => c.candidate_id !== null);

  const stages = (results || []).map((row) => {
    const v = row && row.verdict && typeof row.verdict === 'object' ? row.verdict : row || {};
    return {
      candidate_id: pick(row, 'candidate_id', 'candidateId'),
      stage: pick(row, 'stage') ?? pick(v, 'stage'),
      version: Math.trunc(num(pick(row, 'version') ?? v.version, 1)),
      decision: typeof v.decision === 'string' ? v.decision : 'skip',
      reasons: cleanReasons(v.reasons),
      metrics: cleanMetrics(v.metrics),
    };
  }).filter((s) => s.stage !== null);
  stages.sort((a, b) => (
    String(a.candidate_id).localeCompare(String(b.candidate_id))
    || String(a.stage).localeCompare(String(b.stage))
  ));

  // The receipt records ONE judge block: the verdict-bearing judgment for the
  // winner if there is one, else the last verdict-bearing judgment. Calibration
  // judgments never decide anything, so they are summarised, not embedded.
  const decisive = (judgments || []).filter((j) => (j.kind ?? j.reason) !== 'calibration');
  const chosen = decisive.find((j) => pick(j, 'candidate_id', 'candidateId') === winner)
    ?? decisive[decisive.length - 1]
    ?? null;
  let judge = null;
  if (chosen) {
    const jv = chosen.verdict && typeof chosen.verdict === 'object' ? chosen.verdict : {};
    judge = {
      candidate_id: pick(chosen, 'candidate_id', 'candidateId'),
      decision: typeof jv.decision === 'string' ? jv.decision : null,
      score: jv.score === undefined || jv.score === null ? null : num(jv.score),
      blocking: Array.isArray(jv.blocking) ? jv.blocking.map(String) : [],
      rationale_sha256: sha256(jv.rationale ?? null),
    };
  }

  const winnerRow = winner
    ? (candidates || []).find((c) => pick(c, 'candidate_id', 'candidateId') === winner)
    : null;

  const agreement = calibration
    ? (calibration.agreementRate ?? calibration.agreement_rate ?? null)
    : null;

  return {
    schema: RECEIPT_SCHEMA,
    version: RECEIPT_VERSION,
    run_id: run.id ?? run.run_id ?? null,
    created_at: createdAt ?? new Date().toISOString(),
    task: {
      title: cleanString(task.title ?? null, 200),
      text_sha256: sha256(task.text ?? null),
    },
    repo: {
      // v2: the receipt travels; the operator's directory layout does not. Only
      // the basename and the hash of the full path go in, and the full path
      // stays in the local SQLite run row where it belongs.
      basename: repoBasename(run.repo_path ?? run.repoPath ?? null),
      // Spec 6.5 wants the path hashed as well: a receipt often travels further
      // than the machine it was produced on, and the hash is enough to prove two
      // receipts describe the same checkout without publishing a home directory.
      path_sha256: (run.repo_path ?? run.repoPath) ? sha256(String(run.repo_path ?? run.repoPath)) : null,
      url: task.repoUrl ?? task.repo_url ?? null,
      base_sha: run.base_sha ?? run.baseSha ?? null,
    },
    mode: mode ?? run.mode ?? null,
    verdict: verdict ?? null,
    stop_reason: stopReason ?? null,
    winner: winner
      ? {
        candidate_id: winner,
        // Aliases: owner F's buildPrBody reads winner.candidate / winner.id.
        candidate: winner,
        id: winner,
        branch: branch ?? null,
        sha: sha ?? null,
        commit: sha ?? null,
        diff_sha256: winnerRow ? (pick(winnerRow, 'diff_sha256', 'diffSha256') ?? null) : null,
      }
      : null,
    candidates: candidateRows,
    // Empty for a single-unit run, so today's receipts keep the same shape with
    // one extra (always present, always an array) field. Canonicalisation covers
    // it, so it is signed like everything else.
    subtasks: (subtasks || []).map((s) => ({
      id: pick(s, 'id') ?? null,
      winner: pick(s, 'winner') ?? null,
      branch: pick(s, 'branch') ?? null,
      sha: pick(s, 'sha') ?? null,
    })).filter((s) => s.id !== null),
    stages,
    judge,
    calibration: {
      sampled: Math.trunc(num(calibration?.sampled, 0)),
      agreement_rate: agreement === null || agreement === undefined ? null : num(agreement),
    },
    cost: {
      total_usd: num(cost?.total_usd ?? cost?.totalUsd, 0),
      by_model: cost && cost.by_model && typeof cost.by_model === 'object'
        ? Object.fromEntries(Object.entries(cost.by_model).map(([k, v]) => [k, num(v?.cost_usd ?? v, 0)]))
        : {},
    },
    models: {
      orchestrator: models.orchestrator ?? null,
      judge: models.judge ?? null,
      coders: Array.isArray(models.coders) ? models.coders.map(String) : [],
      reviewer: models.reviewer ?? null,
    },
    ladder_config: {
      candidates: Math.trunc(num(config.candidates, 0)),
      coder_turns: Math.trunc(num(config.coderTurns ?? config.coder_turns, 0)),
      repair_rounds: Math.trunc(num(config.repairRounds ?? config.repair_rounds, 0)),
      repair_turns: Math.trunc(num(config.repairTurns ?? config.repair_turns, 0)),
      max_cost_usd: num(config.maxCostUsd ?? config.max_cost_usd, 0),
      calibration_rate: num(config.calibrationRate ?? config.calibration_rate, 0),
      bands_sha256: sha256Canonical(bands),
      rubric_sha256: sha256Canonical(rubric),
    },
    signer_kid: signerKid,
  };
}

/**
 * Sign a receipt (returns a new object). The signature covers the canonical
 * form of everything except `signature_ed25519` itself, with `signer_kid`
 * already filled in so the key claim is inside the signed payload.
 *
 * @param {object} receipt
 * @param {object|null} [signer] {privateKey, publicKey, key_fingerprint}
 * @returns {object} the signed receipt
 */
export function signReceipt(receipt, signer = loadOrCreateDefaultSigner()) {
  if (!receipt || typeof receipt !== 'object') {
    throw new LadderError('bad-arg', 'signReceipt needs a receipt object');
  }
  if (!signer || !signer.privateKey || !signer.publicKey) {
    // Signing disabled (KOLM_ED25519_DISABLE=1) is a legitimate state; the
    // receipt is still a receipt, it just carries no signature block.
    const unsigned = { ...receipt, signer_kid: receipt.signer_kid ?? null };
    delete unsigned.signature_ed25519;
    return unsigned;
  }
  const payload = { ...receipt, signer_kid: signer.key_fingerprint ?? null };
  delete payload.signature_ed25519;
  const block = buildSignatureBlock({
    privateKey: signer.privateKey,
    publicKey: signer.publicKey,
    key_fingerprint: signer.key_fingerprint,
    payloadCanonical: canonicalJson(payload),
  });
  return { ...payload, signature_ed25519: block };
}

/**
 * Verify a receipt (object or path to a receipt file).
 * @param {object|string} receiptOrPath
 * @param {{publicKey?:string}} [opts] pin the expected public key
 * @returns {{ok:boolean, kid:string|null, reason:string|null}}
 */
export function verifyReceipt(receiptOrPath, { publicKey } = {}) {
  let receipt = receiptOrPath;
  if (typeof receiptOrPath === 'string') {
    try {
      receipt = readReceipt(receiptOrPath);
    } catch (err) {
      return { ok: false, kid: null, reason: `cannot read receipt: ${err.message}` };
    }
  }
  if (!receipt || typeof receipt !== 'object') {
    return { ok: false, kid: null, reason: 'receipt is not an object' };
  }
  if (receipt.schema && receipt.schema !== RECEIPT_SCHEMA) {
    return { ok: false, kid: receipt.signer_kid ?? null, reason: `unknown receipt schema ${receipt.schema}` };
  }
  if (receipt.version !== undefined && !SUPPORTED_RECEIPT_VERSIONS.includes(Number(receipt.version))) {
    return {
      ok: false,
      kid: receipt.signer_kid ?? null,
      reason: `unsupported receipt version ${receipt.version} (supported: ${SUPPORTED_RECEIPT_VERSIONS.join(', ')})`,
    };
  }
  const block = receipt.signature_ed25519;
  if (!block) return { ok: false, kid: receipt.signer_kid ?? null, reason: 'receipt is not signed' };
  if (publicKey && block.public_key !== publicKey) {
    return { ok: false, kid: block.key_fingerprint ?? null, reason: 'public key does not match the pinned key' };
  }
  const payload = { ...receipt };
  delete payload.signature_ed25519;
  const res = verifySignatureBlock(block, canonicalJson(payload));
  if (!res.ok) {
    return { ok: false, kid: block.key_fingerprint ?? null, reason: res.reason ?? 'signature does not verify' };
  }
  if (receipt.signer_kid && receipt.signer_kid !== res.key_fingerprint) {
    return { ok: false, kid: res.key_fingerprint ?? null, reason: 'signer_kid does not match the signing key' };
  }
  return { ok: true, kid: res.key_fingerprint ?? block.key_fingerprint ?? null, reason: null };
}

/** `<config.codeDir>/receipts/<runId>.json` */
export function receiptPathFor(config = {}, runId) {
  const base = config.codeDir ?? config.code_dir ?? path.join(config.dataDir ?? '.', 'code');
  return path.join(base, 'receipts', `${runId}.json`);
}

/**
 * Write a receipt as pretty JSON.
 * @param {object} receipt
 * @param {string} dir directory to write into
 * @returns {string} the path written
 */
export function writeReceipt(receipt, dir) {
  if (!receipt || typeof receipt !== 'object') {
    throw new LadderError('bad-arg', 'writeReceipt needs a receipt object');
  }
  const target = path.join(dir, `${receipt.run_id ?? 'receipt'}.json`);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, `${JSON.stringify(receipt, null, 2)}\n`, 'utf8');
  return target;
}

/** Read a receipt file. */
export function readReceipt(receiptPath) {
  return JSON.parse(fs.readFileSync(receiptPath, 'utf8'));
}

export default {
  buildReceipt,
  signReceipt,
  verifyReceipt,
  writeReceipt,
  readReceipt,
  receiptPathFor,
  scrubSecrets,
  sha256,
  sha256Canonical,
  RECEIPT_SCHEMA,
  RECEIPT_VERSION,
  SUPPORTED_RECEIPT_VERSIONS,
};

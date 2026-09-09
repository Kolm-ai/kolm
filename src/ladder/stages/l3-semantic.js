// kolm code — L3 cheap semantic review (spec 5.6). Owner C.
//
// The first rung that costs money. It sees everything the cheap rungs produced
// (their reasons become part of the prompt) plus the rubric, the plan, a repo
// brief excerpt and the diff, and returns per-item scores through a strict JSON
// schema. Its job is NOT to be right — it is to be confidently right often
// enough that only the residue reaches the expensive judge, so a null/garbled
// response is 'unsure' (escalate), never 'bad' (reject).

import { bandFor, worst } from '../bands.js';
import { modelTimeoutMs } from '../config.js';
import { flowsUp } from '../verdict.js';
import { rubricToPrompt, scoreRubric, blockingFailures } from '../rubric.js';
import { briefToPrompt } from '../context/repo-brief.js';

export const id = 'l3-semantic';
export const level = 3;
export const version = 1;
export const workers = 4;

/** L1..L3 run on anything the level below did not call confidently bad. */
export const gate = flowsUp;

const DIFF_CAP = 60 * 1024;
const BRIEF_CAP = 6 * 1024;
const BLOCKING_THRESHOLD = 0.3;

const SYSTEM = [
  'You are a strict code reviewer inside an automated verification cascade.',
  'You review one patch against one rubric and return JSON only.',
  'Score what the diff actually does. Do not credit intent, comments or TODOs as work.',
  'If you cannot tell from the evidence, say so with a low confidence rather than guessing high.',
].join(' ');

/**
 * Truncate to a byte cap with an explicit marker so the model knows it is partial.
 * @param {string} text
 * @param {number} cap
 * @returns {string}
 */
export function capText(text, cap) {
  const src = String(text ?? '');
  const bytes = Buffer.byteLength(src, 'utf8');
  if (bytes <= cap) return src;
  // -8 of slack: cutting mid-codepoint can re-encode as a 3-byte U+FFFD.
  const head = Buffer.from(src, 'utf8').subarray(0, Math.max(0, cap - 8)).toString('utf8');
  return `${head}\n[truncated ${bytes - Buffer.byteLength(head, 'utf8')} bytes]`;
}

// ------------------------------------------------------- acceptance criteria
//
// Live run r_20260909_18fc043b: the plan said "the regex literal in
// parseDuration matches the character class [dhms]" and "parseDuration('1d')
// returns 86400". The coder added the unit to the table but left the regex at
// [hms] (its edit of that line was refused), no existing test exercised 'd', so
// the dynamic rung passed and the judge scored 0.9 with an empty blocking list.
// A stated criterion was simply never checked, because nothing in the loop ever
// asked about the criteria one at a time.
//
// These helpers are shared by the L3 reviewer and the L4 judge so both graders
// answer the same shape: one attestation per numbered criterion, with the
// evidence that shows it. An unattested or unmet criterion is then a blocking
// failure that flows into repair, not prose at the end of a rationale.

/** One acceptance criterion is rendered at most this long into a prompt. */
const ACCEPTANCE_TEXT_CAP = 400;
/** Per-criterion evidence kept on the result (prompts stay bounded too). */
const ACCEPTANCE_EVIDENCE_CAP = 300;
/** Cap of the "Unmet acceptance: ..." rationale prefix. */
const UNMET_PREFIX_CAP = 600;

/**
 * Optional per-criterion attestation. Shared by REVIEW_SCHEMA and the L4
 * JUDGMENT_SCHEMA so a criterion is answered identically at both rungs.
 * Optional on purpose: a plan with no acceptance list must not force a grader
 * to invent one, and every pre-existing fixture answers without this field.
 */
export const ACCEPTANCE_SCHEMA = Object.freeze({
  type: 'array',
  items: {
    type: 'object',
    required: ['index', 'met', 'evidence'],
    properties: {
      index: { type: 'integer' },
      met: { type: 'boolean' },
      evidence: { type: 'string' },
    },
  },
});

/**
 * The plan's acceptance criteria as a clean list of strings.
 *
 * Tolerates every shape a plan actually arrives in: missing, empty, not an
 * array, or a plan that is itself a free-text string (which has no criteria to
 * attest to). Non-string entries are dropped because a criterion that cannot be
 * printed cannot be quoted back as evidence either.
 *
 * @param {object|string} [plan]
 * @returns {string[]} criterion #n is element n-1
 */
export function acceptanceList(plan) {
  if (!plan || typeof plan !== 'object' || Array.isArray(plan)) return [];
  if (!Array.isArray(plan.acceptance)) return [];
  return plan.acceptance
    .filter((c) => typeof c === 'string' && c.trim() !== '')
    .map((c) => c.trim().replace(/\s+/g, ' ').slice(0, ACCEPTANCE_TEXT_CAP));
}

/**
 * The numbered ACCEPTANCE CRITERIA prompt block, or null when there are none
 * (callers omit the whole section rather than print an empty heading).
 * @param {string[]} criteria acceptanceList() output
 * @returns {string|null}
 */
export function acceptanceBlock(criteria, plan = null) {
  if (!Array.isArray(criteria) || criteria.length === 0) return null;
  const lines = ['ACCEPTANCE CRITERIA (from the plan; attest to each one separately)'];
  criteria.forEach((text, i) => lines.push(`${i + 1}. ${text}`));
  if (plan && typeof plan === 'object' && plan.subtask_id) {
    const files = Array.isArray(plan.files_expected) ? plan.files_expected.join(', ') : '';
    lines.push(
      `This candidate is one subtask of a split plan${files ? ` (its files: ${files})` : ''}. ` +
        'Judge each criterion only as far as those files can satisfy it; work that belongs to ' +
        'another subtask is not missing here and does not make a criterion unmet.',
    );
  }
  return lines.join('\n');
}

/** Output-contract lines that go with the block above. ASCII only. */
export const ACCEPTANCE_CONTRACT = Object.freeze([
  'Return exactly one "acceptance" entry per numbered criterion, with "index" the criterion number (1-based).',
  'Set "met" true only when the diff or the test output shows the criterion is satisfied, and quote that line in "evidence".',
  'A criterion the evidence cannot confirm is met:false. Intent, a comment, a plan to do it later, and an unexercised code path are not evidence.',
]);

/**
 * Normalise a grader's `acceptance` array against the plan's criteria.
 *
 * Drops entries that are not objects, indices outside 1..N and duplicate
 * indices (first entry wins), so a grader cannot pad or double-count its own
 * attestation. `met` is true only when it is literally `true`: a string, a
 * number or a missing field means the grader did not attest, which is the same
 * as "the evidence does not show it".
 *
 * @param {string[]} criteria acceptanceList() output
 * @param {unknown} raw json.acceptance exactly as the grader returned it
 * @returns {Array<{index:number, text:string, met:boolean, evidence:string}>} sorted by index
 */
export function normalizeAcceptance(criteria, raw) {
  if (!Array.isArray(criteria) || criteria.length === 0) return [];
  if (!Array.isArray(raw)) return [];
  const seen = new Set();
  const out = [];
  for (const entry of raw) {
    if (!entry || typeof entry !== 'object') continue;
    if (typeof entry.index !== 'number' && typeof entry.index !== 'string') continue;
    const index = Number(entry.index);
    if (!Number.isInteger(index) || index < 1 || index > criteria.length) continue;
    if (seen.has(index)) continue;
    seen.add(index);
    out.push({
      index,
      text: criteria[index - 1],
      met: entry.met === true,
      evidence: String(entry.evidence ?? '').slice(0, ACCEPTANCE_EVIDENCE_CAP),
    });
  }
  out.sort((a, b) => a.index - b.index);
  return out;
}

/**
 * The criterion numbers a grader did not attest as met.
 * @param {Array<{index:number, met:boolean}>} acceptance normalizeAcceptance() output
 * @returns {number[]} empty when every attested criterion is met
 */
export function unmetIndices(acceptance) {
  return (Array.isArray(acceptance) ? acceptance : []).filter((a) => !a.met).map((a) => a.index);
}

/**
 * The prefix that carries unmet criteria into the repair prompt.
 *
 * The orchestrator turns a judge rejection into repair feedback whose first
 * `reasons` entry is the rationale, and the repair prompt prints `reasons`
 * verbatim; putting the unmet criteria at the front of the rationale is what
 * makes the coder see them.
 *
 * @param {Array<{index:number, text:string, met:boolean}>} acceptance
 * @returns {string} '' when nothing is unmet
 */
export function unmetPrefix(acceptance) {
  const unmet = (Array.isArray(acceptance) ? acceptance : []).filter((a) => !a.met);
  if (unmet.length === 0) return '';
  const body = unmet.map((a) => `#${a.index} ${a.text}`).join('; ');
  return `Unmet acceptance: ${body}`.slice(0, UNMET_PREFIX_CAP);
}

/** JSON schema the reviewer must fill (spec 5.6). */
export const REVIEW_SCHEMA = Object.freeze({
  name: 'l3_review',
  schema: {
    type: 'object',
    required: ['items', 'overall', 'confidence', 'blocking'],
    properties: {
      items: {
        type: 'array',
        items: {
          type: 'object',
          required: ['id', 'score', 'evidence'],
          properties: {
            id: { type: 'string' },
            score: { type: 'number' },
            evidence: { type: 'string' },
          },
        },
      },
      overall: { type: 'number' },
      confidence: { type: 'number' },
      blocking: { type: 'array', items: { type: 'string' } },
      // Optional: absent means "this grader did not attest", which leaves the
      // rung's behaviour exactly as it was before acceptance attestation.
      acceptance: ACCEPTANCE_SCHEMA,
    },
    additionalProperties: false,
  },
});

function clamp01(value, fallback = 0) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(1, Math.max(0, n));
}

/**
 * One-line-per-stage summary of the cheap rungs, for the reviewer prompt.
 * @param {object} priorResults {stage: {decision, reasons}} or store.latestResults() shape
 * @returns {string}
 */
export function priorSummary(priorResults) {
  const lines = [];
  for (const [stage, entry] of Object.entries(priorResults || {})) {
    const verdict = entry && entry.verdict ? entry.verdict : entry;
    if (!verdict) continue;
    const reasons = Array.isArray(verdict.reasons) ? verdict.reasons.slice(0, 4).join('; ') : '';
    lines.push(`- ${stage}: ${verdict.decision || '?'}${reasons ? ` — ${reasons}` : ''}`);
  }
  return lines.length > 0 ? lines.join('\n') : '- (no cheap-rung results recorded)';
}

/**
 * @param {object} ctx spec 5.6 stage context
 * @returns {Promise<object>} stage result
 */
export async function run({
  wt, plan, brief, bands, rubric, providers, config, store, signal,
  runId, candidate, candidateId, priorResults, diffText,
}) {
  const started = Date.now();
  const reviewer = providers?.reviewer || null;
  if (!reviewer) {
    return {
      decision: 'skip',
      reasons: ['no reviewer provider is armed'],
      hits: [],
      metrics: {},
      artifacts: {},
      costUsd: 0,
      latencyMs: Date.now() - started,
    };
  }

  let diff = diffText ?? candidate?.diffText ?? null;
  if (diff == null && wt && typeof wt.diff === 'function') {
    try {
      diff = await wt.diff();
    } catch (err) {
      // A worktree that cannot produce a diff is a tooling failure, not a bad
      // patch; the empty-diff case is already L0's job to call bad.
      return {
        decision: 'unsure',
        reasons: [`could not read the diff: ${err?.message || String(err)}`],
        hits: [],
        metrics: {},
        artifacts: { error: String(err?.message || err) },
        costUsd: 0,
        latencyMs: Date.now() - started,
      };
    }
  }

  const cid = candidateId || candidate?.candidateId || null;
  const priors = priorResults || (store && runId && cid ? store.latestResults(runId, cid) : {});

  const criteria = acceptanceList(plan);
  const criteriaBlock = acceptanceBlock(criteria, plan);

  const parts = [];
  parts.push(rubricToPrompt(rubric));
  parts.push('');
  parts.push('TASK / PLAN');
  parts.push(typeof plan === 'string' ? plan : JSON.stringify(plan ?? {}, null, 2));
  parts.push('');
  if (criteriaBlock) {
    parts.push(criteriaBlock);
    parts.push('');
  }
  parts.push(capText(briefToPrompt(brief), BRIEF_CAP));
  parts.push('');
  parts.push('CHEAP RUNG RESULTS (already computed, do not repeat them)');
  parts.push(priorSummary(priors));
  parts.push('');
  parts.push('PATCH (unified diff)');
  parts.push(capText(diff || '', DIFF_CAP));
  parts.push('');
  parts.push('Return JSON matching the schema. Every rubric item id must appear exactly once in "items".');
  if (criteriaBlock) for (const line of ACCEPTANCE_CONTRACT) parts.push(line);
  const prompt = parts.join('\n');

  const timeoutMs = modelTimeoutMs(config);
  let res;
  try {
    res = await reviewer.chat({
      system: SYSTEM,
      messages: [{ role: 'user', content: prompt }],
      jsonSchema: REVIEW_SCHEMA,
      maxTokens: Number(config?.reviewerMaxTokens) > 0 ? Number(config.reviewerMaxTokens) : 2000,
      temperature: 0,
      effort: config?.reviewerEffort,
      signal,
      timeoutMs,
    });
  } catch (err) {
    return {
      decision: 'unsure',
      reasons: [`reviewer call failed: ${err?.message || String(err)}`],
      hits: [],
      metrics: {},
      artifacts: { error: String(err?.message || err) },
      costUsd: 0,
      latencyMs: Date.now() - started,
      model: reviewer.id ?? null,
    };
  }

  const costUsd = Number(res?.costUsd) || 0;
  if (store && runId) {
    store.addCost({
      runId,
      candidateId: cid,
      stage: id,
      provider: res?.provider ?? reviewer.provider ?? null,
      model: res?.model ?? reviewer.model ?? null,
      inputTokens: res?.usage?.input ?? res?.usage?.input_tokens ?? 0,
      outputTokens: res?.usage?.output ?? res?.usage?.output_tokens ?? 0,
      cacheRead: res?.usage?.cacheRead ?? res?.usage?.cache_read_input_tokens ?? 0,
      costUsd,
    });
  }

  const model = reviewer.id ?? res?.model ?? null;
  if (!res || res.json == null) {
    return {
      decision: 'unsure',
      reasons: ['reviewer returned no JSON'],
      hits: [],
      metrics: {},
      artifacts: {
        text: String(res?.text || '').slice(0, 2048),
        stopReason: res?.stopReason ?? null,
        emptyRetry: res?.meta?.emptyRetry ?? null,
      },
      costUsd,
      latencyMs: Date.now() - started,
      model,
    };
  }

  const json = res.json;
  const rawItems = Array.isArray(json.items) ? json.items : [];
  const items = rawItems
    .filter((it) => it && typeof it.id === 'string')
    .map((it) => ({ id: it.id, score: clamp01(it.score), evidence: String(it.evidence ?? '').slice(0, 400) }));

  // Trust our own weighted arithmetic over the model's `overall` when it gave
  // us items to compute one from; models routinely mis-add weighted means.
  const computed = scoreRubric(rubric, items);
  const overall = items.length > 0 ? computed : clamp01(json.overall);
  const confidence = clamp01(json.confidence);
  const declaredBlocking = Array.isArray(json.blocking) ? json.blocking.filter((b) => typeof b === 'string') : [];
  const scoredBlocking = blockingFailures(rubric, items, BLOCKING_THRESHOLD);
  // A model may declare a blocking id it did not score low; only ids whose
  // score is actually below the threshold reject the patch outright.
  const blocking = scoredBlocking;

  const reasons = [];
  const hits = [];
  const scoreHit = bandFor(bands, id, 'rubric_score', overall);
  const confHit = bandFor(bands, id, 'reviewer_confidence', confidence);
  hits.push(scoreHit, confHit);
  if (scoreHit.band !== 'good') reasons.push(`rubric_score ${overall.toFixed(3)} is ${scoreHit.band}`);
  if (confHit.band !== 'good') reasons.push(`reviewer_confidence ${confidence.toFixed(3)} is ${confHit.band}`);

  // An unmet criterion is a fact about the diff, not a metric reading, so it
  // rejects the patch without fabricating a banded hit: teaching calibration
  // that this rubric_score is 'bad' would move a threshold for the wrong reason.
  const acceptance = normalizeAcceptance(criteria, json.acceptance);
  const unmet = unmetIndices(acceptance);

  let decision = worst(hits);
  if (blocking.length > 0) {
    decision = 'bad';
    reasons.unshift(`blocking rubric item(s) below ${BLOCKING_THRESHOLD}: ${blocking.join(', ')}`);
    hits.push({ key: 'rubric_score', value: overall, band: 'bad' });
  }
  if (unmet.length > 0) {
    decision = 'bad';
    // Newest first, and ahead of everything else: these are the reasons the
    // orchestrator forwards to the repair session.
    for (const entry of [...acceptance].filter((a) => !a.met).reverse()) {
      reasons.unshift(`acceptance #${entry.index} not met: ${entry.text}`);
    }
  }
  if (items.length === 0 && decision !== 'bad') {
    decision = 'unsure';
    reasons.push('reviewer scored no rubric items');
  }
  if (reasons.length === 0) reasons.push(`reviewer scored ${overall.toFixed(3)} at confidence ${confidence.toFixed(3)}`);

  const worstItems = [...items].sort((a, b) => a.score - b.score).slice(0, 4);
  return {
    decision,
    reasons,
    hits,
    acceptance,
    unmet,
    metrics: {
      rubric_score: overall,
      reviewer_confidence: confidence,
      declared_overall: clamp01(json.overall),
      items_scored: items.length,
    },
    artifacts: {
      items: worstItems,
      blocking,
      declaredBlocking,
      stopReason: res.stopReason ?? null,
      // The adapter's one retry after an empty length-capped reply, when it happened.
      emptyRetry: res.meta?.emptyRetry ?? null,
    },
    costUsd,
    latencyMs: Date.now() - started,
    model,
  };
}

export default { id, level, version, workers, gate, run };

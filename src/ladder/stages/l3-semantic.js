// kolm code — L3 cheap semantic review (spec 5.6). Owner C.
//
// The first rung that costs money. It sees everything the cheap rungs produced
// (their reasons become part of the prompt) plus the rubric, the plan, a repo
// brief excerpt and the diff, and returns per-item scores through a strict JSON
// schema. Its job is NOT to be right — it is to be confidently right often
// enough that only the residue reaches the expensive judge, so a null/garbled
// response is 'unsure' (escalate), never 'bad' (reject).

import { bandFor, worst } from '../bands.js';
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

  const parts = [];
  parts.push(rubricToPrompt(rubric));
  parts.push('');
  parts.push('TASK / PLAN');
  parts.push(typeof plan === 'string' ? plan : JSON.stringify(plan ?? {}, null, 2));
  parts.push('');
  parts.push(capText(briefToPrompt(brief), BRIEF_CAP));
  parts.push('');
  parts.push('CHEAP RUNG RESULTS (already computed, do not repeat them)');
  parts.push(priorSummary(priors));
  parts.push('');
  parts.push('PATCH (unified diff)');
  parts.push(capText(diff || '', DIFF_CAP));
  parts.push('');
  parts.push('Return JSON matching the schema. Every rubric item id must appear exactly once in "items".');
  const prompt = parts.join('\n');

  const timeoutMs = Number(config?.stageTimeoutMs) > 0 ? Number(config.stageTimeoutMs) : 600000;
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
      artifacts: { text: String(res?.text || '').slice(0, 2048), stopReason: res?.stopReason ?? null },
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

  let decision = worst(hits);
  if (blocking.length > 0) {
    decision = 'bad';
    reasons.unshift(`blocking rubric item(s) below ${BLOCKING_THRESHOLD}: ${blocking.join(', ')}`);
    hits.push({ key: 'rubric_score', value: overall, band: 'bad' });
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
    },
    costUsd,
    latencyMs: Date.now() - started,
    model,
  };
}

export default { id, level, version, workers, gate, run };

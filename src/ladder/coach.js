// kolm code - the repair coach (spec 4.3). Owner ORCH.
//
// A repair round today hands the coder the same failing feedback it already saw
// and hopes a second pass goes better. The coach spends one cheap orchestrator
// call in between: the smart model reads the failing evidence, the diff the coder
// actually produced and the tail of what its tools returned, and writes a handful
// of concrete hints. That is the whole asymmetry of the ladder applied to repair -
// thinking is cheap, writing is expensive.
//
// Three rules keep this from becoming a liability:
//   1. It is never fatal. Any failure (provider down, bad JSON, abort) returns
//      zero hints and the repair proceeds exactly as it does today.
//   2. It is skipped when it cannot pay for itself: when the orchestrator IS the
//      coder (same ref - the model would be coaching itself), when the operator
//      turned it off, or when the budget is spent.
//   3. It never sees more than a bounded slice of the run: 12 KB of diff and the
//      last three tool results.

import { modelTimeoutMs } from './config.js';

/** Largest slice of the candidate diff the coach is shown. */
export const COACH_DIFF_CAP = 12 * 1024;

/** How many trailing tool results the coach is shown. */
export const COACH_TAIL_RESULTS = 3;

/** Hard cap on hints handed back to the coder. */
export const MAX_HINTS = 5;

/** Per-hint character cap; the coder renders at most 240 chars anyway. */
const HINT_CAP = 240;

const SYSTEM = [
  'You are the orchestrator of a code ladder, coaching a cheaper coder model through one repair round.',
  'You are given the plan, the failing evidence, the diff the coder produced and the tail of its tool output.',
  'Write short, concrete, actionable hints: which file, which function, what to change, what to check.',
  'Do not write the patch and do not restate the failure - the coder already has it.',
  `Return at most ${MAX_HINTS} hints. Fewer is better. Return an empty list if you have nothing useful to add.`,
].join(' ');

/** JSON schema the coach must fill. */
export const COACH_SCHEMA = Object.freeze({
  name: 'kolm_coach_hints',
  schema: {
    type: 'object',
    required: ['hints'],
    properties: {
      hints: { type: 'array', items: { type: 'string' } },
    },
    additionalProperties: false,
  },
});

/**
 * The tail of what this candidate's tools returned, read back out of the stored
 * `coder.turn` events. The orchestrator never holds a transcript (the coder owns
 * that), and the events table is the source of truth for everything else about a
 * run, so it is the right place to read from here too.
 *
 * @param {object} store ladder store (needs `eventsSince`)
 * @param {string} runId
 * @param {string} candidateId candidate whose turns to read
 * @param {number} [limit] how many trailing tool results to keep
 * @returns {Array<{name:string, input:string, result:string}>} oldest first
 */
export function transcriptTailFor(store, runId, candidateId, limit = COACH_TAIL_RESULTS) {
  if (!store || typeof store.eventsSince !== 'function' || !runId) return [];
  let events = [];
  try {
    events = store.eventsSince(runId, 0, 2000);
  } catch {
    return [];
  }
  const out = [];
  for (const ev of events) {
    if (ev.type !== 'coder.turn') continue;
    if (candidateId && ev.payload?.candidateId !== candidateId) continue;
    for (const d of Array.isArray(ev.payload?.detail) ? ev.payload.detail : []) {
      out.push({
        name: String(d?.name ?? '?'),
        input: String(d?.input ?? ''),
        result: String(d?.result ?? ''),
      });
    }
  }
  const n = Math.max(0, Math.trunc(Number(limit)) || 0);
  return n === 0 ? [] : out.slice(-n);
}

/** Render the coach prompt. ASCII only; no key, no path outside the repo. */
function buildPrompt({ plan, feedback, diffText, transcriptTail }) {
  const parts = [];
  parts.push('PLAN');
  parts.push(String(plan?.title ?? '(untitled)'));
  if (plan?.summary) parts.push(String(plan.summary).slice(0, 2000));
  const acceptance = Array.isArray(plan?.acceptance) ? plan.acceptance : [];
  if (acceptance.length > 0) {
    parts.push('ACCEPTANCE');
    for (const a of acceptance.slice(0, 20)) parts.push(`  - ${String(a).slice(0, 300)}`);
  }
  parts.push('');
  parts.push('FAILING EVIDENCE');
  parts.push(String(feedback ?? '(none)').slice(0, 8000));
  parts.push('');
  const diff = String(diffText ?? '');
  parts.push('DIFF THE CODER PRODUCED');
  parts.push(diff.trim() === ''
    ? '(empty diff - the coder produced no change)'
    : (diff.length > COACH_DIFF_CAP ? `${diff.slice(0, COACH_DIFF_CAP)}\n[truncated]` : diff));
  parts.push('');
  const tail = Array.isArray(transcriptTail) ? transcriptTail : [];
  if (tail.length > 0) {
    parts.push('LAST TOOL RESULTS');
    for (const t of tail) {
      parts.push(`  ${t.name} ${t.input}`.trimEnd());
      parts.push(`    ${String(t.result ?? '').slice(0, 1200).replace(/\n/g, '\n    ')}`);
    }
    parts.push('');
  }
  parts.push('Return JSON matching the schema.');
  return parts.join('\n');
}

/**
 * Normalise the model's hints: strings only, trimmed, deduplicated, capped.
 * @param {unknown} value raw `json.hints`
 * @returns {string[]}
 */
export function normalizeHints(value) {
  if (!Array.isArray(value)) return [];
  const seen = new Set();
  const out = [];
  for (const raw of value) {
    if (typeof raw !== 'string') continue;
    const s = raw.trim().replace(/\s+/g, ' ').slice(0, HINT_CAP);
    if (s === '' || seen.has(s)) continue;
    seen.add(s);
    out.push(s);
    if (out.length >= MAX_HINTS) break;
  }
  return out;
}

/**
 * Ask the orchestrator model for repair hints. Never throws.
 *
 * Emits `repair.coach {candidateId, round, hints, costUsd}` when it produced
 * hints, and nothing at all when it was skipped - a skipped coach is the normal
 * single-model configuration, not an event worth logging on every round.
 *
 * @param {object} args
 * @param {object} args.orchestrator orchestrator provider
 * @param {object} [args.coder] the coder provider for this attempt; when its ref
 *   equals the orchestrator's the coach is skipped (self-coaching buys nothing)
 * @param {object} [args.plan]
 * @param {string} [args.feedback] the failing evidence the repair will carry
 * @param {string} [args.diffText] the diff the candidate produced
 * @param {Array<object>} [args.transcriptTail] from transcriptTailFor()
 * @param {object} [args.config] reads `coach`, `modelTimeoutMs`
 * @param {object} [args.store]
 * @param {string} [args.runId]
 * @param {string} [args.candidateId]
 * @param {number} [args.round] repair round number, for the event
 * @param {boolean} [args.budgetExhausted] true when the run has no money left
 * @param {AbortSignal} [args.signal]
 * @returns {Promise<{hints:string[], costUsd:number, skipped:string|null}>}
 */
export async function coachRepair({
  orchestrator,
  coder,
  plan,
  feedback,
  diffText,
  transcriptTail,
  config = {},
  store,
  runId,
  candidateId,
  round = 0,
  budgetExhausted = false,
  signal,
} = {}) {
  if (config.coach === false) return { hints: [], costUsd: 0, skipped: 'disabled' };
  if (budgetExhausted) return { hints: [], costUsd: 0, skipped: 'budget' };
  if (!orchestrator || typeof orchestrator.chat !== 'function' || orchestrator.armed === false) {
    return { hints: [], costUsd: 0, skipped: 'no-orchestrator' };
  }
  const orchRef = orchestrator.id ?? null;
  const coderRef = coder?.id ?? null;
  if (orchRef && coderRef && orchRef === coderRef) {
    return { hints: [], costUsd: 0, skipped: 'same-model' };
  }

  let res = null;
  let costUsd = 0;
  try {
    res = await orchestrator.chat({
      system: SYSTEM,
      messages: [{ role: 'user', content: buildPrompt({ plan, feedback, diffText, transcriptTail }) }],
      jsonSchema: COACH_SCHEMA,
      maxTokens: 600,
      effort: 'low',
      cacheSystem: true,
      signal,
      timeoutMs: modelTimeoutMs(config),
    });
  } catch {
    // Never fatal: a coach that cannot answer costs the run nothing.
    res = null;
  }
  costUsd = Number(res?.costUsd) || 0;
  if (store && runId && costUsd >= 0 && res) {
    try {
      store.addCost({
        runId,
        candidateId: candidateId ?? null,
        stage: 'coach',
        provider: res?.provider ?? orchestrator.provider ?? null,
        model: res?.model ?? orchestrator.model ?? null,
        inputTokens: res?.usage?.input ?? 0,
        outputTokens: res?.usage?.output ?? 0,
        cacheRead: res?.usage?.cacheRead ?? 0,
        costUsd,
      });
    } catch { /* the ledger is best-effort here; the run must not die for a hint */ }
  }

  const hints = normalizeHints(res?.json?.hints);
  if (hints.length > 0 && store && runId) {
    try {
      store.emit(runId, 'repair.coach', {
        candidateId: candidateId ?? null,
        round,
        hints,
        costUsd,
      });
    } catch { /* as above */ }
  }
  return { hints, costUsd, skipped: null };
}

export default { coachRepair, transcriptTailFor, normalizeHints, COACH_SCHEMA };

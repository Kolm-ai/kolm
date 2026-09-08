// kolm code — L4, the expensive judge (spec 6.3). Owner D.
//
// This is the only rung allowed to be slow and expensive, because the cascade
// has already thrown away everything it was confident about. It sees the whole
// evidence pack — plan, brief, full diff, every cheap rung's reasons and hits —
// and returns a rubric-scored verdict.
//
// It is NOT a cascade stage: `runCascade` deliberately stops at l3-semantic and
// the orchestrator calls `runJudge` on the residue. The stage-shaped exports
// (id/level/version/workers) exist so the judge is addressable by the same
// (runId, candidateId, stage, version) key as every other rung.
//
// Failure policy differs from L3 on purpose. L3 answering nothing means
// 'unsure' (escalate to me). Me answering nothing means the candidate cannot be
// resolved at all, so `runJudge` throws LadderError('judge-failed') after one
// retry and the orchestrator records the candidate as JUDGE-unresolved. A parse
// failure must never be read as "the patch is bad".

import { LadderError } from '../errors.js';
import { rubricToPrompt, scoreRubric, blockingFailures } from '../rubric.js';
import { briefToPrompt } from '../context/repo-brief.js';
import { runCommand } from '../context/baseline.js';
import { assembleVerdict } from '../verdict.js';
import { capText } from './l3-semantic.js';

export const id = 'l4-judge';
export const level = 4;
export const version = 1;
export const workers = 2;

const DIFF_CAP = 120 * 1024;
const BRIEF_CAP = 8 * 1024;
const TEST_OUTPUT_CAP = 16 * 1024;
/** A blocking rubric item below this fails the patch at L4 (spec 6.3 + owner brief). */
export const BLOCKING_THRESHOLD = 0.5;

const SYSTEM = [
  'You are the kolm code judge: the last and most expensive rung of a confidence cascade.',
  'Cheaper rungs already decided everything they were confident about; you only ever see the residue,',
  'so "I am not sure" is not an available answer — read the evidence and decide.',
  'Score the patch that is actually in the diff. Intent, comments, TODOs and plans are not work.',
  'Quote the diff in your evidence. Never invent a line that is not there.',
].join(' ');

/** JSON schema the judge must fill. */
export const JUDGMENT_SCHEMA = Object.freeze({
  name: 'l4_judgment',
  schema: {
    type: 'object',
    required: ['items', 'overall', 'blocking', 'rationale', 'decision'],
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
      blocking: { type: 'array', items: { type: 'string' } },
      rationale: { type: 'string' },
      decision: { type: 'string', enum: ['PASS', 'FAIL'] },
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
 * Full evidence dump of the cheap rungs: decision, reasons AND the banded hits
 * that produced them. The judge needs the numbers, not just the verdicts.
 * @param {object} priorResults {stage: result} or store.latestResults() shape
 * @returns {string}
 */
export function evidenceSummary(priorResults) {
  const lines = [];
  for (const [stage, entry] of Object.entries(priorResults || {})) {
    const result = entry && entry.verdict ? entry.verdict : entry;
    if (!result) continue;
    lines.push(`- ${stage}: ${result.decision || '?'}`);
    for (const reason of (Array.isArray(result.reasons) ? result.reasons : []).slice(0, 8)) {
      lines.push(`    reason: ${reason}`);
    }
    for (const hit of (Array.isArray(result.hits) ? result.hits : []).slice(0, 12)) {
      if (!hit || typeof hit !== 'object') continue;
      lines.push(`    metric: ${hit.key} = ${hit.value} -> ${hit.band}`);
    }
  }
  return lines.length > 0 ? lines.join('\n') : '- (no cheap-rung results recorded)';
}

function buildUserPrompt({ plan, brief, diff, priorResults, testOutput }) {
  const parts = [];
  parts.push('TASK / PLAN');
  parts.push(typeof plan === 'string' ? plan : JSON.stringify(plan ?? {}, null, 2));
  parts.push('');
  parts.push(capText(briefToPrompt(brief), BRIEF_CAP));
  parts.push('');
  parts.push('CHEAP RUNG EVIDENCE (already computed; these are the facts, not opinions)');
  parts.push(evidenceSummary(priorResults));
  parts.push('');
  if (testOutput) {
    parts.push('TEST COMMAND RE-RUN (executed just now against this candidate)');
    parts.push(capText(testOutput, TEST_OUTPUT_CAP));
    parts.push('');
  }
  parts.push('PATCH (unified diff)');
  parts.push(capText(diff || '', DIFF_CAP));
  parts.push('');
  parts.push('OUTPUT CONTRACT');
  parts.push('Return JSON matching the schema. Every rubric item id appears exactly once in "items".');
  parts.push(`A BLOCKING item scored below ${BLOCKING_THRESHOLD} fails the patch: put its id in "blocking" and set "decision" to "FAIL".`);
  parts.push('"rationale" is at most six sentences and must name the decisive evidence.');
  return parts.join('\n');
}

async function askJudge({ judge, system, prompt, config, signal }) {
  return judge.chat({
    system,
    messages: [{ role: 'user', content: prompt }],
    jsonSchema: JUDGMENT_SCHEMA,
    maxTokens: Number(config?.judgeMaxTokens) > 0 ? Number(config.judgeMaxTokens) : 4000,
    effort: 'high',
    cacheSystem: true,
    signal,
    timeoutMs: Number(config?.stageTimeoutMs) > 0 ? Number(config.stageTimeoutMs) : 600000,
  });
}

/**
 * Judge one candidate.
 *
 * @param {object} ctx
 * @param {string} ctx.runId
 * @param {string} ctx.candidateId
 * @param {object} [ctx.wt] worktree handle (used for the diff and the optional test re-run)
 * @param {object} [ctx.plan]
 * @param {object} [ctx.brief]
 * @param {object} ctx.rubric
 * @param {object} [ctx.priorResults] {stage: result}
 * @param {object} ctx.judge provider
 * @param {object} ctx.store
 * @param {object} [ctx.config]
 * @param {object} [ctx.commands] detectCommands() output (test re-run)
 * @param {string} [ctx.diffText]
 * @param {AbortSignal} [ctx.signal]
 * @param {'unsure'|'calibration'|'final'} [ctx.reason]
 * @returns {Promise<{decision:'good'|'bad', score:number, items:Array, rationale:string,
 *   blocking:string[], model:string|null, costUsd:number, latencyMs:number}>}
 */
export async function runJudge({
  runId,
  candidateId,
  wt,
  plan,
  brief,
  rubric,
  priorResults,
  judge,
  store,
  config = {},
  commands,
  diffText,
  signal,
  reason = 'unsure',
} = {}) {
  const started = Date.now();
  if (!judge || typeof judge.chat !== 'function') {
    throw new LadderError('judge-failed', 'runJudge needs a judge provider', { runId, candidateId });
  }
  if (judge.armed === false) {
    throw new LadderError('not-armed', `the judge rung ${judge.id ?? '?'} is not armed`, { ref: judge.id ?? null });
  }

  let diff = diffText ?? null;
  if (diff == null && wt && typeof wt.diff === 'function') diff = await wt.diff();

  // Re-running the test suite is off by default: it doubles the wall clock of
  // the most expensive rung and L2 already ran it. `judgeRerunsTests` exists
  // for the case where the coder's own `run` calls polluted the worktree.
  let testOutput = null;
  if (config.judgeRerunsTests === true && wt) {
    const cmd = plan?.test_command || commands?.test || null;
    if (cmd) {
      try {
        const res = await runCommand(wt, cmd, { signal });
        testOutput = [
          `$ ${cmd}`,
          `exit code: ${res?.exitCode ?? '?'}`,
          String(res?.output ?? ''),
        ].join('\n');
      } catch (err) {
        testOutput = `$ ${cmd}\ncould not run: ${err?.message || String(err)}`;
      }
    }
  }

  const system = `${SYSTEM}\n\n${rubricToPrompt(rubric)}`;
  const prompt = buildUserPrompt({ plan, brief, diff, priorResults, testOutput });

  let res = null;
  let costUsd = 0;
  let lastError = null;
  // One retry: a null `json` is a transport/format accident, not a verdict.
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      res = await askJudge({ judge, system, prompt, config, signal });
    } catch (err) {
      lastError = err;
      res = null;
    }
    const attemptCost = Number(res?.costUsd) || 0;
    costUsd += attemptCost;
    if (store && runId) {
      store.addCost({
        runId,
        candidateId,
        stage: id,
        provider: res?.provider ?? judge.provider ?? null,
        model: res?.model ?? judge.model ?? null,
        inputTokens: res?.usage?.input ?? 0,
        outputTokens: res?.usage?.output ?? 0,
        cacheRead: res?.usage?.cacheRead ?? 0,
        costUsd: attemptCost,
      });
    }
    if (res && res.json != null) break;
    res = null;
  }

  if (!res) {
    throw new LadderError(
      'judge-failed',
      `the judge returned no usable judgment for ${candidateId}${lastError ? `: ${lastError.message}` : ''}`,
      { runId, candidateId, reason },
    );
  }

  const json = res.json;
  const items = (Array.isArray(json.items) ? json.items : [])
    .filter((it) => it && typeof it.id === 'string')
    .map((it) => ({ id: it.id, score: clamp01(it.score), evidence: String(it.evidence ?? '').slice(0, 600) }));

  const computed = scoreRubric(rubric, items);
  const score = items.length > 0 ? computed : clamp01(json.overall);
  // Trust scored blocking failures over the model's own list, but keep any id
  // it declared: a judge that says "this is blocking" and forgets to score it
  // low is still telling us something.
  const scoredBlocking = blockingFailures(rubric, items, BLOCKING_THRESHOLD);
  const declared = (Array.isArray(json.blocking) ? json.blocking : []).filter((b) => typeof b === 'string');
  const blocking = [...new Set([...scoredBlocking, ...declared.filter((b) => scoredBlocking.includes(b))])];
  const declaredFail = String(json.decision || '').toUpperCase() === 'FAIL';
  const decision = blocking.length > 0 || declaredFail ? 'bad' : 'good';

  const verdict = {
    stage: id,
    version,
    decision,
    score,
    items,
    blocking,
    rationale: String(json.rationale ?? '').slice(0, 4000),
    declaredDecision: declaredFail ? 'FAIL' : 'PASS',
    declaredBlocking: declared,
    model: judge.id ?? res.model ?? null,
    costUsd,
    latencyMs: Date.now() - started,
    reason,
  };

  if (store && runId) {
    store.putJudgment({ runId, candidateId, kind: reason, verdict });
    store.emit(runId, 'judge', {
      candidateId,
      kind: reason,
      decision,
      score,
      blocking,
      costUsd,
      model: verdict.model,
    });
  }

  return {
    decision,
    score,
    items,
    rationale: verdict.rationale,
    blocking,
    model: verdict.model,
    costUsd,
    latencyMs: verdict.latencyMs,
  };
}

// ---------------------------------------------------------------- calibration
//
// The cascade is only worth running if its confident answers agree with the
// judge. Calibration spends a small, fixed fraction of judge calls on decisions
// the cascade was already sure about, purely to measure that.

/**
 * Pick the candidates to spend a calibration judge call on.
 *
 * Only CONFIDENT cascade verdicts are eligible: a JUDGE verdict is going to the
 * judge anyway, so sampling it would measure nothing.
 *
 * @param {object} args
 * @param {Map<string, object>|object} args.perCandidate runCascade() output
 * @param {number} [args.rate] 0..1
 * @param {() => number} [args.rng] injectable for deterministic tests
 * @returns {string[]} candidate ids
 */
export function selectCalibrationSample({ perCandidate, rate = 0, rng = Math.random } = {}) {
  const r = Number(rate);
  if (!Number.isFinite(r) || r <= 0) return [];
  const entries = perCandidate instanceof Map
    ? [...perCandidate.entries()]
    : Object.entries(perCandidate || {});
  const picked = [];
  for (const [candidateId, state] of entries) {
    const verdict = state?.verdict?.verdict ?? state?.verdict ?? null;
    if (verdict !== 'PASS' && verdict !== 'FAIL') continue;
    if (rng() < r) picked.push(candidateId);
  }
  return picked;
}

/** Cascade verdict for one candidate, rebuilt from the stored stage rows. */
function cascadeVerdictFor(store, runId, candidateId) {
  const latest = store.latestResults(runId, candidateId) || {};
  const decisions = {};
  for (const [stage, entry] of Object.entries(latest)) {
    if (stage === id) continue; // the judge is not a cascade stage
    const result = entry && entry.verdict ? entry.verdict : entry;
    if (result && typeof result.decision === 'string') decisions[stage] = result.decision;
  }
  return assembleVerdict(decisions).verdict;
}

/**
 * Compare the cascade's confident verdicts with the calibration judge calls.
 *
 * @param {object} args
 * @param {object} args.store
 * @param {string} args.runId
 * @param {string[]} [args.kinds] judgment kinds to count (default: calibration only)
 * @returns {{sampled:number, agreements:number,
 *   disagreements:Array<{candidateId:string, cascade:string, judge:string}>, agreementRate:number|null}}
 */
export function calibrationReport({ store, runId, kinds = ['calibration'] } = {}) {
  const wanted = new Set(kinds);
  const rows = (store?.listJudgments(runId) || []).filter((row) => wanted.has(row.kind));
  let agreements = 0;
  const disagreements = [];
  for (const row of rows) {
    const judged = row.verdict?.decision === 'bad' ? 'bad' : 'good';
    const cascade = cascadeVerdictFor(store, runId, row.candidate_id);
    if (cascade !== 'PASS' && cascade !== 'FAIL') continue;
    const agrees = (cascade === 'PASS' && judged === 'good') || (cascade === 'FAIL' && judged === 'bad');
    if (agrees) agreements += 1;
    else disagreements.push({ candidateId: row.candidate_id, cascade, judge: judged });
  }
  const sampled = agreements + disagreements.length;
  return {
    sampled,
    agreements,
    disagreements,
    agreementRate: sampled > 0 ? agreements / sampled : null,
  };
}

/** The next value strictly below `v`, on the granularity `v` looks like. */
function below(v) {
  return Number.isInteger(v) ? v - 1 : Number((v - 0.01).toFixed(4));
}

/** The next value strictly above `v`, on the granularity `v` looks like. */
function above(v) {
  return Number.isInteger(v) ? v + 1 : Number((v + 0.01).toFixed(4));
}

/**
 * Propose band moves from judge/cascade disagreements (spec 6.2).
 *
 * Pure: it reads the store and returns proposals plus a patched copy. It never
 * writes bands.json — `kolm code bands --apply` does that, deliberately, so a
 * threshold change is always a human decision.
 *
 * The rule is one-directional and conservative:
 *   cascade said PASS, judge said bad  -> some stage called a metric 'good' too
 *                                         easily; move the good threshold past
 *                                         the observed value.
 *   cascade said FAIL, judge said good -> some stage called a metric 'bad' too
 *                                         easily; move the bad threshold to
 *                                         include the observed value.
 * Only the offending side of the band moves, and repeated evidence takes the
 * strictest proposal.
 *
 * @param {object} args
 * @param {object} args.store
 * @param {object} args.bands loadBands() output
 * @param {string[]} [args.runIds] default: every run that has judgments
 * @returns {{proposals:Array<{stage:string, key:string, from:number, to:number, evidence:string}>, bandsPatched:object}}
 */
export function suggestBands({ store, bands, runIds } = {}) {
  const patched = JSON.parse(JSON.stringify(bands || {}));
  const proposals = new Map();

  const runs = Array.isArray(runIds) && runIds.length > 0
    ? runIds
    : (store?.listRuns({ limit: 500 }) || []).map((run) => run.id);

  const propose = (stage, key, field, value, evidence) => {
    const spec = patched?.[stage]?.[key];
    if (!spec) return;
    const from = Number(spec[field]);
    if (!Number.isFinite(from) || !Number.isFinite(value)) return;
    // Every rule above moves `lo` down and `hi` up; anything else is a bug in
    // the caller, not a proposal worth printing.
    if (field === 'lo' && !(value < from)) return;
    if (field === 'hi' && !(value > from)) return;
    const mapKey = `${stage}\u0000${key}\u0000${field}`;
    const prev = proposals.get(mapKey);
    let to = value;
    if (prev) to = field === 'lo' ? Math.min(prev.to, value) : Math.max(prev.to, value);
    if (to === from) return;
    if (field === 'lo' && to > Number(spec.hi)) return;
    if (field === 'hi' && to < Number(spec.lo)) return;
    proposals.set(mapKey, { stage, key, field, from, to, evidence });
  };

  for (const runId of runs) {
    const report = calibrationReport({ store, runId });
    for (const dis of report.disagreements) {
      const latest = store.latestResults(runId, dis.candidateId) || {};
      for (const [stage, entry] of Object.entries(latest)) {
        if (stage === id) continue;
        const result = entry && entry.verdict ? entry.verdict : entry;
        const hits = Array.isArray(result?.hits) ? result.hits : [];
        for (const hit of hits) {
          if (!hit || typeof hit !== 'object' || !hit.key) continue;
          const spec = patched?.[stage]?.[hit.key];
          if (!spec) continue;
          const value = Number(hit.value);
          if (!Number.isFinite(value)) continue;
          const lowIsBad = spec.low_is_bad === true;
          const evidence = `${runId}/${dis.candidateId}: cascade ${dis.cascade}, judge ${dis.judge}, ${stage}.${hit.key}=${value} banded ${hit.band}`;
          if (dis.cascade === 'PASS' && dis.judge === 'bad' && hit.band === 'good') {
            if (lowIsBad) propose(stage, hit.key, 'hi', above(value), evidence);
            else propose(stage, hit.key, 'lo', below(value), evidence);
          } else if (dis.cascade === 'FAIL' && dis.judge === 'good' && hit.band === 'bad') {
            if (lowIsBad) propose(stage, hit.key, 'lo', value, evidence);
            else propose(stage, hit.key, 'hi', value, evidence);
          }
        }
      }
    }
  }

  const out = [];
  for (const p of proposals.values()) {
    patched[p.stage][p.key][p.field] = p.to;
    out.push({ stage: p.stage, key: p.key, field: p.field, from: p.from, to: p.to, evidence: p.evidence });
  }
  out.sort((a, b) => (a.stage.localeCompare(b.stage) || a.key.localeCompare(b.key)));
  return { proposals: out, bandsPatched: patched };
}

export default {
  id,
  level,
  version,
  workers,
  runJudge,
  selectCalibrationSample,
  calibrationReport,
  suggestBands,
  JUDGMENT_SCHEMA,
  evidenceSummary,
};

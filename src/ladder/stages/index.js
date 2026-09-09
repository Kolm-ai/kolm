// kolm code — the cascade driver (spec 5.6). Owner C.
//
// Levels run in order; candidates run in parallel inside a level bounded by the
// stage's `workers`. Three invariants make this thing usable in anger:
//   1. RESUMABILITY — a (runId, candidateId, stage, version) row that already
//      exists is reused and the stage function is never invoked.
//   2. NO CRASH PATH — a stage that throws or hangs produces an 'unsure' row,
//      never an exception that loses the other candidates' work.
//   3. FLOW-UP GATING — only a confident 'bad' stops a candidate. Once a
//      candidate fails a gate it stays blocked for every later level (its later
//      stages have no decision at all, which is what assembleVerdict wants).
//
// l4-judge is deliberately NOT here: the orchestrator (owner D) runs it on the
// residue this returns.

import { LadderError } from '../errors.js';
import { assembleVerdict } from '../verdict.js';
import l0 from './l0-meta.js';
import l1 from './l1-static.js';
import l2 from './l2-dynamic.js';
import l3 from './l3-semantic.js';

/** All cascade stages in level order (l4-judge is not a cascade stage). */
export const STAGES = Object.freeze([l0, l1, l2, l3]);

const DEFAULT_STAGE_TIMEOUT_MS = 600000;
// The wrapper always outlives the command it wraps by this much, so a command
// that times out reports its own reason instead of the generic 'stage timed out'.
const STAGE_TIMEOUT_MARGIN_MS = 5000;
const TIMEOUT = Symbol('stage-timeout');

/**
 * The brief without the task-relevant code retrieval (spec 4.6). Returns the
 * SAME object when there is nothing to strip, so a brief that never had a
 * retrieval block reaches the stage byte-identically.
 * @param {object|null|undefined} brief
 * @returns {object|null|undefined}
 */
export function briefWithoutRetrieval(brief) {
  if (!brief || typeof brief !== 'object' || brief.retrieval == null) return brief;
  const rest = { ...brief };
  delete rest.retrieval;
  return rest;
}

/**
 * Run `worker` over `items` with at most `limit` in flight.
 * @param {Array} items
 * @param {number} limit
 * @param {(item:any, index:number)=>Promise<void>} worker
 */
async function pool(items, limit, worker) {
  const width = Math.max(1, Math.trunc(Number(limit)) || 1);
  let next = 0;
  const runners = [];
  for (let k = 0; k < Math.min(width, items.length); k += 1) {
    runners.push((async () => {
      for (;;) {
        const index = next;
        next += 1;
        if (index >= items.length) return;
        await worker(items[index], index);
      }
    })());
  }
  await Promise.all(runners);
}

/**
 * Call `stage.run` with a hard time bound. The stage gets an AbortSignal that
 * fires on either the cascade signal or the timeout, but Promise.race means a
 * stage that ignores its signal still cannot block the cascade.
 * @returns {Promise<object>} the stage result
 */
async function runBounded(stage, args, timeoutMs, signal) {
  const ac = new AbortController();
  const onAbort = () => {
    try {
      ac.abort();
    } catch {
      // an already-aborted controller is fine
    }
  };
  if (signal) {
    if (signal.aborted) onAbort();
    else signal.addEventListener('abort', onAbort, { once: true });
  }
  let timer = null;
  const timeoutPromise = new Promise((_resolve, reject) => {
    timer = setTimeout(() => {
      onAbort();
      reject(TIMEOUT);
    }, timeoutMs);
  });
  timeoutPromise.catch(() => {});
  const started = Promise.resolve().then(() => stage.run({ ...args, signal: ac.signal }));
  // The loser of the race must not surface as an unhandled rejection.
  started.catch(() => {});
  try {
    return await Promise.race([started, timeoutPromise]);
  } finally {
    if (timer) clearTimeout(timer);
    if (signal) signal.removeEventListener('abort', onAbort);
  }
}

function normalizeResult(raw, stage, latencyMs) {
  const result = raw && typeof raw === 'object' ? raw : {};
  const decision = ['good', 'bad', 'unsure', 'skip'].includes(result.decision) ? result.decision : 'unsure';
  return {
    stage: stage.id,
    version: stage.version,
    decision,
    reasons: Array.isArray(result.reasons) ? result.reasons : [],
    hits: Array.isArray(result.hits) ? result.hits : [],
    metrics: result.metrics && typeof result.metrics === 'object' ? result.metrics : {},
    artifacts: result.artifacts && typeof result.artifacts === 'object' ? result.artifacts : {},
    costUsd: Number(result.costUsd) || 0,
    latencyMs: Number.isFinite(Number(result.latencyMs)) ? Number(result.latencyMs) : latencyMs,
    model: result.model ?? null,
  };
}

/**
 * Run the confidence cascade over a set of candidates.
 *
 * @param {object} ctx
 * @param {string} ctx.runId
 * @param {Array<{candidateId:string, wt:object, diffText?:string}>} ctx.candidates
 * @param {object} [ctx.plan] task/plan object handed to L0 and L3
 * @param {object} [ctx.brief] repoBrief() output
 * @param {object} [ctx.baseline] baseline() output
 * @param {object} [ctx.commands] detectCommands() output
 * @param {object} ctx.bands loadBands() output
 * @param {object} [ctx.rubric] loadRubric() output (required when a reviewer is armed)
 * @param {{reviewer?:object}} [ctx.providers]
 * @param {object} [ctx.config] uses config.stageTimeoutMs
 * @param {object} ctx.store openStore() result
 * @param {AbortSignal} [ctx.signal]
 * @param {object} [ctx.logger]
 * @param {string} [ctx.mode] 'direct' skips l3-semantic
 * @returns {Promise<{perCandidate: Map<string, {decisions:object, results:object, verdict:object}>}>}
 */
export async function runCascade(ctx = {}) {
  const {
    runId, candidates = [], store, signal, logger, mode,
    config = {}, plan, brief, baseline, commands, bands, rubric, providers,
  } = ctx;
  if (!runId) throw new LadderError('bad-arg', 'runCascade needs a runId');
  if (!store) throw new LadderError('bad-arg', 'runCascade needs a store');

  // A stage's inner command gets the whole budget, so the wrapper has to be
  // strictly more patient than the command it wraps: otherwise the wrapper wins
  // the race and reports the useless 'stage timed out' instead of the command's
  // own, informative 'test command timed out after N ms'.
  const stageBudget = Number(config.stageTimeoutMs) > 0
    ? Number(config.stageTimeoutMs)
    : DEFAULT_STAGE_TIMEOUT_MS;
  const timeoutMs = stageBudget + STAGE_TIMEOUT_MARGIN_MS;
  const perCandidate = new Map();
  for (const cand of candidates) {
    perCandidate.set(cand.candidateId, { decisions: {}, results: {}, verdict: null, _last: undefined, _blocked: false });
  }

  const levels = [...new Set(STAGES.map((s) => s.level))].sort((a, b) => a - b);
  let abortedDuringLevel = false;

  for (const level of levels) {
    const stagesHere = STAGES.filter((s) => s.level === level);
    for (const stage of stagesHere) {
      const eligible = [];
      for (const cand of candidates) {
        const state = perCandidate.get(cand.candidateId);
        if (state._blocked) continue;
        const gate = typeof stage.gate === 'function' ? stage.gate : () => true;
        if (!gate(state._last)) {
          // A candidate that fails one gate never re-enters at a higher level,
          // even though its missing decision would flow up on its own.
          state._blocked = true;
          continue;
        }
        eligible.push(cand);
      }

      await pool(eligible, stage.workers, async (cand) => {
        const state = perCandidate.get(cand.candidateId);
        if (signal?.aborted) {
          // Write nothing: a half-run stage must not leave a row that a resume
          // would happily reuse.
          abortedDuringLevel = true;
          return;
        }

        const cached = store.getResult(runId, cand.candidateId, stage.id, stage.version);
        if (cached) {
          state.decisions[stage.id] = cached.decision;
          state.results[stage.id] = cached;
          state._last = cached.decision;
          store.emit(runId, 'stage.cached', { candidateId: cand.candidateId, stage: stage.id, version: stage.version, decision: cached.decision });
          return;
        }

        const started = Date.now();
        let result;
        if (mode === 'direct' && stage.id === 'l3-semantic') {
          result = normalizeResult(
            { decision: 'skip', reasons: ['direct mode'], hits: [], metrics: {}, artifacts: {}, costUsd: 0, latencyMs: 0 },
            stage,
            0,
          );
        } else {
          const args = {
            candidate: cand,
            candidateId: cand.candidateId,
            runId,
            wt: cand.wt,
            diffText: cand.diffText,
            priorResults: state.results,
            plan,
            // The reviewer judges the diff against the plan, not against the
            // code the retrieval happened to pull in for the CODER. Handing it
            // that block both wastes the context window and invites it to
            // review code the candidate never touched (spec 4.6).
            brief: stage.id === 'l3-semantic' ? briefWithoutRetrieval(brief) : brief,
            baseline,
            commands,
            bands,
            rubric,
            providers,
            config,
            store,
          };
          try {
            result = normalizeResult(await runBounded(stage, args, timeoutMs, signal), stage, Date.now() - started);
          } catch (err) {
            if (err === TIMEOUT) {
              result = normalizeResult(
                { decision: 'unsure', reasons: ['stage timed out'], artifacts: { timeoutMs } },
                stage,
                Date.now() - started,
              );
            } else if (signal?.aborted) {
              abortedDuringLevel = true;
              return;
            } else {
              const message = err?.message || String(err);
              result = normalizeResult(
                { decision: 'unsure', reasons: [message], artifacts: { error: message } },
                stage,
                Date.now() - started,
              );
              logger?.warn?.(`${stage.id} threw for ${cand.candidateId}: ${message}`);
            }
          }
        }

        state.decisions[stage.id] = result.decision;
        state.results[stage.id] = result;
        state._last = result.decision;
        store.putResult({ runId, candidateId: cand.candidateId, stage: stage.id, version: stage.version, verdict: result });
        store.emit(runId, 'stage.result', {
          candidateId: cand.candidateId,
          stage: stage.id,
          version: stage.version,
          decision: result.decision,
          reasons: result.reasons,
          hits: result.hits,
          costUsd: result.costUsd,
          latencyMs: result.latencyMs,
        });
      });

      if (signal?.aborted || abortedDuringLevel) {
        // Everything already written stays written; a resume picks up here.
        throw new LadderError('aborted', 'cascade aborted', { runId, stage: stage.id });
      }
    }
  }

  for (const [, state] of perCandidate) {
    state.verdict = assembleVerdict(state.decisions);
    delete state._last;
    delete state._blocked;
  }
  return { perCandidate };
}

export default { STAGES, runCascade };

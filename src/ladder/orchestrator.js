// kolm code — the orchestrator (spec 6.4). Owner D.
//
// This is the thing that makes the cascade worth having: it spends the cheapest
// model first, lets the cheap rungs throw away everything they are confident
// about, and only escalates the residue. Every expensive action — a coder turn,
// a cascade, a judge call — is preceded by a budget check, and every state
// transition is an event so the viewer can replay a run it did not watch.
//
// CANDIDATE ID SCHEME (documented once, relied on everywhere):
//
//   candidateId = `c<idx>a<attempt>`  (store.candidateIdFor)
//     idx     = rung * config.candidates + slot
//     attempt = repair round, 0 for the first generation
//
//   So with candidates=3 the first coder rung owns c0a0..c2a0, the second rung
//   owns c3a0..c5a0, and a repair of c1a0 is c1a1. The rung is recoverable from
//   the id (`Math.floor(idx / candidates)`), ids never collide across rungs, and
//   every attempt keeps its own store rows so the history of a repair survives.
//   A repair REUSES the worktree object of the attempt it repairs (state is
//   carried forward) while writing new candidate/result rows.

import path from 'node:path';
import crypto from 'node:crypto';

import { LadderError, isLadderError } from './errors.js';
import { newRunId, candidateIdFor } from './store.js';
import { loadBands } from './bands.js';
import { loadRubric } from './rubric.js';
import { detectCommands } from './context/detect.js';
import { repoBrief, repoKeyFor } from './context/repo-brief.js';
import { loadInstructions } from './context/instructions.js';
import { retrieveContext } from './context/retrieve.js';
import { baseline as computeBaseline } from './context/baseline.js';
import {
  loadLessons,
  saveLessons,
  mergeLessons,
  extractLessonCandidates,
} from './lessons.js';
import { routeRun } from './routing.js';
import { coachRepair, transcriptTailFor } from './coach.js';
import {
  auditCriteria,
  applyPlanCriterionEvents,
  criterionEntries,
  fileFactsFor,
  pathsNamedIn,
  stuckAnalysis,
} from './audit.js';
import {
  validateSubtasks,
  topoLevels,
  subtaskCandidateIdFor,
  subPlanFor,
  conflictFeedback,
  finishedSubtasks,
  priorSubtaskVerdicts,
} from './subtasks.js';
import { runCascade, STAGES, briefWithoutRetrieval } from './stages/index.js';
import { runJudge, selectCalibrationSample, calibrationReport } from './stages/l4-judge.js';
import { plan as makePlan } from './planner.js';
import { runCoder, runRepair } from './coder.js';
import {
  buildReceipt,
  signReceipt,
  writeReceipt,
  receiptPathFor,
  scrubSecrets,
} from './receipt.js';

const JUDGE_WORKERS = 2;

/* ------------------------------------------------------------------ helpers */

/** Run `worker` over `items` with at most `limit` in flight. */
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
 * Deterministic per-run PRNG (xorshift32 seeded from the run id), so a
 * calibration sample is reproducible from the run id alone.
 * @param {string} seedText
 * @returns {() => number}
 */
export function makeRunRng(seedText) {
  const digest = crypto.createHash('sha256').update(String(seedText), 'utf8').digest();
  let state = digest.readUInt32BE(0) || 0x9e3779b9;
  return () => {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return state / 0x100000000;
  };
}

/** A store view whose `emit` also feeds `onEvent`. */
function withEvents(store, onEvent) {
  if (typeof onEvent !== 'function') return store;
  const wrapped = Object.create(store);
  wrapped.emit = (runId, type, payload) => {
    const id = store.emit(runId, type, payload);
    try {
      onEvent({ id, run_id: runId, type, payload, ts: new Date().toISOString() });
    } catch {
      // A listener that throws must not take the run down with it.
    }
    return id;
  };
  return wrapped;
}

/**
 * Rank candidates: fewest reasons, then cheapest, then id (stable).
 * @param {Array<{candidateId:string, reasons?:string[], costUsd?:number}>} list
 * @returns {Array} a sorted copy
 */
export function rankCandidates(list) {
  return [...(list || [])].sort((a, b) => {
    const ra = Array.isArray(a.reasons) ? a.reasons.length : 0;
    const rb = Array.isArray(b.reasons) ? b.reasons.length : 0;
    if (ra !== rb) return ra - rb;
    const ca = Number(a.costUsd) || 0;
    const cb = Number(b.costUsd) || 0;
    if (ca !== cb) return ca - cb;
    return String(a.candidateId).localeCompare(String(b.candidateId));
  });
}

/**
 * The winning candidate, or null.
 * @param {Array<{verdict:string}>} list
 * @returns {object|null}
 */
export function pickWinner(list) {
  const passing = (list || []).filter((c) => c.verdict === 'PASS');
  return rankCandidates(passing)[0] ?? null;
}

/** The stage that rejected a candidate, in level order, with its evidence. */
function failingFeedback(state) {
  const results = state?.results || {};
  for (const stage of STAGES) {
    const result = results[stage.id];
    if (result && result.decision === 'bad') {
      return {
        stage: stage.id,
        reasons: result.reasons || [],
        hits: result.hits || [],
        failing: result.artifacts?.newlyFailing || [],
      };
    }
  }
  for (const stage of STAGES) {
    const result = results[stage.id];
    if (result && result.decision === 'unsure') {
      return {
        stage: stage.id,
        reasons: result.reasons || [],
        hits: result.hits || [],
        failing: result.artifacts?.newlyFailing || [],
      };
    }
  }
  return { stage: 'unknown', reasons: ['no stage produced a decision'], hits: [], failing: [] };
}

/**
 * Error codes that mean "this rung cannot code at all" rather than "this patch
 * is bad". When every attempt of a rung dies with one of these the ladder
 * escalates to the next rung instead of taking the whole run down.
 */
export const RUNG_LEVEL_ERROR_CODES = Object.freeze([
  'provider-bad-request',
  'not-armed',
  'provider-unavailable',
  'provider-timeout',
  'bad-ref',
  'execution',
]);
const RUNG_LEVEL = new Set(RUNG_LEVEL_ERROR_CODES);

/**
 * Reasons that describe the infrastructure rather than the patch. Repair rounds
 * are expensive and a coder cannot fix a timeout, so they are never spent on
 * one of these.
 */
const NON_ACTIONABLE_REASONS = [
  [/timed out|timeout/i, 'timeout'],
  [/provider|not armed|unavailable|bad request|could not produce a verdict/i, 'provider-failure'],
];

/**
 * null when the feedback is actionable, otherwise the reason repair is skipped.
 * @param {{judgeFailed?:boolean}} row
 * @param {{reasons?:string[], hits?:Array, failing?:Array}} feedback
 * @returns {string|null}
 */
export function repairSkipReason(row, feedback) {
  if (row?.judgeFailed) return 'judge-failed';
  const failing = Array.isArray(feedback?.failing) ? feedback.failing : [];
  // Named failing tests are actionable on their own, whatever the prose says.
  if (failing.length > 0) return null;
  const reasons = (feedback?.reasons || [])
    .filter((r) => typeof r === 'string' && r.trim() !== '');
  if (reasons.length === 0) {
    // No prose at all: a banded metric is still something a coder can aim at.
    const hits = Array.isArray(feedback?.hits) ? feedback.hits : [];
    return hits.length > 0 ? null : 'no-actionable-reasons';
  }
  const blocked = reasons.filter((r) => NON_ACTIONABLE_REASONS.some(([re]) => re.test(r)));
  if (blocked.length < reasons.length) return null;
  for (const [re, label] of NON_ACTIONABLE_REASONS) {
    if (reasons.some((r) => re.test(r))) return label;
  }
  return 'no-actionable-reasons';
}

function stageCostOf(state) {
  let sum = 0;
  for (const result of Object.values(state?.results || {})) sum += Number(result?.costUsd) || 0;
  return sum;
}

function reasonsOf(state) {
  const out = [];
  for (const result of Object.values(state?.results || {})) {
    for (const reason of Array.isArray(result?.reasons) ? result.reasons : []) out.push(reason);
  }
  return out;
}

function refOf(provider) {
  if (!provider) return null;
  return provider.id ?? provider.ref ?? provider.model ?? null;
}

/**
 * A store view whose semantic-review results look uncached (spec 4.8).
 *
 * Stage results are keyed by `(runId, candidateId, stage, version)`, which is
 * exactly right until the PLAN changes underneath them: the L3 verdict was an
 * attestation against acceptance criteria the audit has since retired, so
 * replaying it would re-reject a patch for a criterion that no longer exists.
 * Only L3 reads the criteria - L0-L2 are deterministic in the diff and stay
 * cached, and L4 is not cached at all - and `putResult` is INSERT OR REPLACE,
 * so the fresh verdict supersedes the stale one on the way back out.
 *
 * @param {object} store the run's store
 * @returns {object} a prototype-chained view; the underlying store is untouched
 */
function withoutSemanticCache(store) {
  const view = Object.create(store);
  view.getResult = (runId, candidateId, stage, version) => (
    stage === 'l3-semantic' ? null : store.getResult(runId, candidateId, stage, version)
  );
  return view;
}

/* ------------------------------------------------------------------- runTask */

/**
 * Run one coding task end to end.
 *
 * @param {object} opts
 * @param {{text:string, title?:string, repoUrl?:string, baseRef?:string}} opts.task
 * @param {string} [opts.repoPath]
 * @param {string} [opts.repoUrl]
 * @param {string} [opts.baseRef]
 * @param {object} opts.config loadConfig() output
 * @param {object} opts.store openStore() output
 * @param {object} opts.driver worktree driver
 * @param {{orchestrator:object, judge:object, coders:object[], reviewer:object}} opts.providers
 * @param {object} [opts.logger]
 * @param {AbortSignal} [opts.signal]
 * @param {string} [opts.runId]
 * @param {boolean} [opts.resume]
 * @param {'cascade'|'direct'} [opts.mode]
 * @param {(evt:object)=>void} [opts.onEvent]
 * @param {object} [opts.bands]
 * @param {object} [opts.rubric]
 * @param {() => number} [opts.rng] calibration sampler (deterministic by default)
 * @returns {Promise<{runId:string, verdict:'PASS'|'FAIL'|'JUDGE'|'ERROR', winner:string|null,
 *   receipt:object, receiptPath:string|null, branch:string|null, sha:string|null, costUsd:number,
 *   funnel:object, candidates:Array, durationMs:number, stopReason:string|null}>}
 */
export async function runTask(opts = {}) {
  let { task, repoPath, repoUrl, baseRef } = opts;
  const {
    config: callerConfig = {},
    store: rawStore,
    driver,
    providers = {},
    logger,
    signal,
    resume = false,
    onEvent,
    rng,
  } = opts;

  if (!rawStore) throw new LadderError('bad-arg', 'runTask needs a store');
  if (!driver) throw new LadderError('bad-arg', 'runTask needs a worktree driver');
  const runId = opts.runId ?? (resume ? null : newRunId());
  if (!runId) throw new LadderError('bad-arg', 'resume needs a runId');

  // A resume rehydrates whatever the caller left out from the stored run row,
  // so `resumeTask({ runId, store, driver, providers })` is enough.
  let storedRun = null;
  if (resume) {
    storedRun = rawStore.getRun(runId);
    if (!storedRun) throw new LadderError('bad-arg', `cannot resume unknown run ${runId}`, { runId });
    if (!task && storedRun.task && typeof storedRun.task === 'object') task = storedRun.task;
    if (repoPath == null && storedRun.repo_path) repoPath = storedRun.repo_path;
    if (repoUrl == null && task?.repoUrl) repoUrl = task.repoUrl;
    if (baseRef == null && task?.baseRef) baseRef = task.baseRef;
  }
  // The run's shape (candidates per rung, coder/repair turn caps) is fixed at
  // creation and stored on the run row; a resume must not widen it, or the
  // ladder would generate fresh candidates for a run that was one candidate
  // wide. The cost cap deliberately stays caller-controlled so an operator
  // can raise the budget on a run that stopped on 'budget'.
  const storedConfig = resume && storedRun?.config && typeof storedRun.config === 'object'
    ? storedRun.config
    : null;
  const config = storedConfig
    ? {
      ...callerConfig,
      candidates: storedConfig.candidates ?? callerConfig.candidates,
      coderTurns: storedConfig.coderTurns ?? callerConfig.coderTurns,
      repairRounds: storedConfig.repairRounds ?? callerConfig.repairRounds,
      repairTurns: storedConfig.repairTurns ?? callerConfig.repairTurns,
    }
    : callerConfig;
  if (!task || typeof task.text !== 'string' || task.text.trim() === '') {
    throw new LadderError('bad-arg', 'runTask needs a task with non-empty text');
  }

  const store = withEvents(rawStore, onEvent);
  const mode = opts.mode ?? storedRun?.mode ?? config.mode ?? 'cascade';
  const bands = opts.bands ?? loadBands(config.bandsPath);
  const rubric = opts.rubric ?? loadRubric(config.rubricPath);
  const started = Date.now();

  const coderRungs = mode === 'direct'
    ? (providers.coders || []).slice(0, 1)
    : (providers.coders || []);
  if (coderRungs.length === 0) throw new LadderError('bad-arg', 'runTask needs at least one coder provider');

  /**
   * The model a repair round is handed to when this rung's coder never managed a
   * tool call (spec 4.3): the next rung, else the orchestrator when it is a
   * different model. `null` means there is nobody else, so the round runs where
   * it would have run anyway.
   *
   * @param {number} rung index of the rung that produced the failed attempt
   * @param {string|null} coderRef ref of that rung's coder
   * @returns {object|null} provider to write the repair, or null
   */
  const fallbackCoderFor = (rung, coderRef) => {
    const next = coderRungs[rung + 1] ?? null;
    if (next && next.armed !== false && refOf(next) !== coderRef) return next;
    const orch = providers.orchestrator ?? null;
    if (orch && orch.armed !== false && refOf(orch) !== coderRef) return orch;
    return null;
  };
  const perRung = mode === 'direct' ? 1 : Math.max(1, Math.trunc(Number(config.candidates) || 1));
  const repairRounds = Math.max(0, Math.trunc(Number(config.repairRounds ?? 2) || 0));
  const repairTurns = Math.max(1, Math.trunc(Number(config.repairTurns ?? 12) || 12));
  const maxCost = Number(config.maxCostUsd);
  // lessons.js resolves its file under `config.dataDir`, falling back to the
  // process CWD when there is none. A run that was told where its code dir is
  // knows the data dir too, so derive it rather than writing next to whatever
  // directory the caller happened to start in.
  const lessonsConfig = config.dataDir || !config.codeDir
    ? config
    : { ...config, dataDir: path.dirname(config.codeDir) };

  const openWorktrees = new Map(); // candidateId -> wt
  let baseWt = null;
  let stopReason = null;

  /**
   * Record a failed attempt without taking the run down. `aborted` is the one
   * failure that always propagates: the operator asked for the run to stop.
   */
  const noteCandidateError = (candidateId, err) => {
    if (isLadderError(err, 'aborted') || err?.name === 'AbortError') throw err;
    const code = isLadderError(err) ? err.code : 'execution';
    const message = scrubSecrets(err?.message || String(err));
    try {
      store.upsertCandidate({ runId, candidateId, status: 'errored', summary: message });
    } catch {
      // The store may itself be the thing that failed; the event below stands.
    }
    store.emit(runId, 'candidate.error', { candidateId, code, message });
    logger?.warn?.(`candidate ${candidateId} failed (${code}): ${message}`);
    return { code, message };
  };

  const spent = () => Number(store.costSummary(runId).total_usd) || 0;
  const budgetExhausted = () => Number.isFinite(maxCost) && maxCost > 0 && spent() >= maxCost;
  const checkBudget = (where) => {
    if (!budgetExhausted()) return false;
    stopReason = 'budget';
    store.emit(runId, 'run.budget', { where, spentUsd: spent(), maxCostUsd: maxCost });
    logger?.warn?.(`run ${runId} stopped on budget at ${where}`);
    return true;
  };

  try {
    /* ---------------------------------------------------------- 1. the run */
    let run = resume ? storedRun : null;
    // A finished run is either done with or exactly the work
    // `resume --coder <stronger>` exists to redo (spec 4.4). A PASS
    // short-circuits: its receipt is the answer. A FAIL or a JUDGE is REOPENED
    // instead - the row goes back to 'running', the earlier attempt's
    // candidates stay on the record, and the coders on the resume command line
    // take the rungs ABOVE the ones already spent.
    const finishedVerdict = run && run.status === 'done' && run.receipt
      ? (run.receipt.verdict
        ?? ((run.winner_candidate ?? run.receipt.winner?.candidate_id) ? 'PASS' : 'FAIL'))
      : null;
    const reopened = finishedVerdict !== null && finishedVerdict !== 'PASS';
    if (finishedVerdict === 'PASS') {
      // Nothing to continue: the run finished and its receipt is on record.
      const receipt = run.receipt;
      const winner = run.winner_candidate ?? receipt.winner?.candidate_id ?? null;
      const summary = {
        runId,
        verdict: receipt.verdict ?? (winner ? 'PASS' : 'FAIL'),
        winner,
        receipt,
        receiptPath: receiptPathFor(config, runId),
        branch: receipt.winner?.branch ?? null,
        sha: receipt.winner?.sha ?? receipt.winner?.commit ?? null,
        costUsd: Number(run.total_cost_usd) || 0,
        funnel: store.funnel({ runIds: [runId] }),
        candidates: (store.listCandidates(runId) || []).map((row) => ({
          candidateId: row.candidate_id ?? row.id,
          generatorRef: row.generator_ref ?? null,
          verdict: row.verdict ?? null,
          decisions: row.decisions ?? {},
          costUsd: Number(row.cost_usd) || 0,
        })),
        durationMs: Date.now() - started,
        stopReason: 'already-finished',
      };
      store.emit(runId, 'run.resume', { runId, mode, alreadyFinished: true, winner });
      store.emit(runId, 'run.end', {
        verdict: summary.verdict,
        winner,
        costUsd: summary.costUsd,
        durationMs: summary.durationMs,
        stopReason: summary.stopReason,
        receiptPath: summary.receiptPath,
      });
      return summary;
    }
    if (!run) {
      run = store.createRun({
        id: runId,
        task,
        repoPath: repoPath ?? null,
        driver: driver.kind ?? null,
        mode,
        config: {
          candidates: perRung,
          mode,
          coderTurns: config.coderTurns,
          repairRounds,
          repairTurns,
          maxCostUsd: maxCost,
        },
      });
    }
    // What the earlier attempt stopped on. The events are the record; the run
    // row keeps a status and a receipt, not a stop reason.
    let priorStopReason = null;
    if (reopened) {
      for (const ev of store.eventsSince(runId, 0, 5000)) {
        if (ev?.type === 'run.end') priorStopReason = ev.payload?.stopReason ?? null;
      }
      // Only `status` is patched: the stored receipt stays on the row until this
      // attempt writes its own, so a reader that finds the run mid-reopen still
      // sees the outcome the last attempt reached. Nothing reads the
      // "done + receipt" pair except the short-circuit above, which cannot fire
      // while the status says 'running'.
      store.updateRun(runId, { status: 'running' });
    }
    store.emit(runId, resume ? 'run.resume' : 'run.start', {
      runId,
      mode,
      task: { title: task.title ?? null, text: String(task.text).slice(0, 500) },
      repoPath: repoPath ?? null,
      coders: coderRungs.map(refOf),
      ...(reopened ? { reopened: true, priorVerdict: finishedVerdict, priorStopReason } : {}),
    });

    /* ------------------------------------------------ 2. base context pack */
    // A resumed run may find its kept worktrees on disk; reuse them.
    baseWt = await driver.create({ repoPath, repoUrl, baseRef, runId, candidateId: 'base', reuse: resume });
    const baseSha = await baseWt.baseSha();
    store.updateRun(runId, { base_sha: baseSha });
    const commands = await detectCommands(baseWt);
    const brief = await repoBrief(store, baseWt, baseSha, { config });
    const repoKey = repoKeyFor(baseWt, { repoKey: repoPath ?? repoUrl ?? undefined });

    // -- context pack extras (spec 4.5, 4.6) ------------------------------
    // Every one of these is a degradation, never a failure: a run whose
    // retrieval times out is a slightly worse-informed run, not a dead one, so
    // each hook emits `context.warn` and carries on with the plain brief.
    const contextWarn = (kind, err) => {
      store.emit(runId, 'context.warn', {
        kind,
        message: scrubSecrets(err?.message || String(err)).slice(0, 500),
      });
      logger?.warn?.(`context ${kind} degraded: ${scrubSecrets(err?.message || String(err))}`);
    };
    try {
      brief.instructions = await loadInstructions(baseWt, { config });
    } catch (err) {
      contextWarn('instructions', err);
    }
    try {
      brief.lessons = loadLessons(lessonsConfig, repoKey).lessons;
    } catch (err) {
      contextWarn('lessons', err);
    }
    let retrievalCached = false;
    try {
      // The retrieval depends on the task as well as the tree, so the task hash
      // is folded into the cache `kind` - context_cache is keyed by
      // (repo_key, sha, kind) and two different tasks over one commit must not
      // share an entry.
      const taskHash = crypto.createHash('sha256').update(String(task.text)).digest('hex').slice(0, 16);
      const cacheKind = `retrieval:${taskHash}`;
      const cached = store.cacheGet(repoKey, baseSha, cacheKind);
      if (cached) {
        brief.retrieval = cached;
        retrievalCached = true;
      } else {
        const retrieval = await retrieveContext(baseWt, task.text, { config });
        brief.retrieval = retrieval;
        try {
          store.cachePut(repoKey, baseSha, cacheKind, retrieval);
        } catch { /* a cold cache is not a run failure */ }
      }
    } catch (err) {
      contextWarn('retrieval', err);
    }

    // One line per run saying what actually reached the coder's brief. A hook
    // that degraded contributes zeros rather than a missing field, so the shape
    // is the same on every run and `show --events` can render it as a table.
    store.emit(runId, 'context.pack', {
      instructionsBytes: Buffer.byteLength(String(brief.instructions?.text ?? ''), 'utf8'),
      instructionFiles: (Array.isArray(brief.instructions?.files) ? brief.instructions.files : [])
        .map((f) => String(f?.path ?? '')).filter((p) => p !== ''),
      lessons: Array.isArray(brief.lessons) ? brief.lessons.length : 0,
      retrievalFiles: Array.isArray(brief.retrieval?.files) ? brief.retrieval.files.length : 0,
      retrievalBytes: Number(brief.retrieval?.bytes) || 0,
      retrievalCached,
    });

    // The judge scores the diff against the plan; the retrieval block exists to
    // help the CODER find its way around, and feeding it to the judge only
    // spends context on code the candidate never touched (spec 4.6).
    const judgeBrief = briefWithoutRetrieval(brief);

    const baselineData = await computeBaseline(store, baseWt, baseSha, commands, { config, signal });

    /* --------------------------------------------------------- 3. the plan */
    let taskPlan = null;
    if (resume) {
      const events = store.eventsSince(runId, 0, 2000).filter((e) => e.type === 'plan');
      if (events.length > 0) taskPlan = events[events.length - 1].payload;
    }
    if (!taskPlan && !checkBudget('plan')) {
      taskPlan = await makePlan({
        task: { ...task, repoUrl: repoUrl ?? task.repoUrl, baseRef: baseRef ?? task.baseRef },
        brief,
        orchestrator: providers.orchestrator,
        store,
        runId,
        config,
        commands,
        signal,
      });
    }
    if (taskPlan?.test_command) commands.test = taskPlan.test_command;

    /* ------------------------------------------------------- 3b. routing */
    // Routing reads the plan's complexity estimate and decides how much ladder
    // to buy. It is deterministic in (plan, config), so a resume recomputes the
    // identical decision from the stored plan event and the stored run config -
    // which is why the run row keeps the CONFIGURED numbers, not the routed
    // ones. Direct mode is a caller instruction, not a suggestion: no routing.
    const routed = mode === 'direct'
      ? {
        startRung: 0,
        candidates: 1,
        repairRounds,
        reason: 'mode=direct',
        complexity: taskPlan?.complexity ?? 'medium',
        risk: taskPlan?.risk ?? 'medium',
        routing: 'fixed',
      }
      : routeRun({
        plan: taskPlan,
        config: { ...config, candidates: perRung, repairRounds },
        coderRungs,
      });
    const routedCandidates = Math.max(1, Math.trunc(Number(routed.candidates)) || 1);
    const routedRepairRounds = Math.max(0, Math.trunc(Number(routed.repairRounds)) || 0);
    if (taskPlan) {
      store.emit(runId, 'run.route', {
        complexity: routed.complexity,
        risk: routed.risk,
        routing: routed.routing,
        startRung: routed.startRung,
        candidates: routedCandidates,
        repairRounds: routedRepairRounds,
        reason: routed.reason,
      });
    }

    // The base worktree exists only to build the context pack.
    if (config.keepWorktrees !== 'all') {
      await baseWt.destroy().catch(() => {});
      baseWt = null;
    }

    /* -------------------------------------------- 4. the generation ladder */
    const allCandidates = new Map(); // candidateId -> summary row
    // A reopened run's receipt lists what the earlier attempt produced as well
    // as what this one does, so the stored rows are seeded before the ladder
    // adds to them. Their verdict comes from the stored receipt: the hygiene
    // step overwrote the candidate rows' status with 'destroyed' on the way
    // out. They are marked `stored` because they are a record, not work this
    // attempt can win with or be judged on.
    if (reopened) {
      const storedVerdicts = new Map(
        (storedRun.receipt?.candidates || [])
          .filter((c) => typeof c?.candidate_id === 'string' && c.candidate_id !== '')
          .map((c) => [c.candidate_id, c.verdict]),
      );
      for (const row of store.listCandidates(runId) || []) {
        const candidateId = row.candidate_id ?? row.id;
        if (typeof candidateId !== 'string' || candidateId === '') continue;
        allCandidates.set(candidateId, {
          candidateId,
          generatorRef: row.generator_ref ?? null,
          wt: null,
          state: null,
          decisions: {},
          verdict: storedVerdicts.get(candidateId) ?? (row.status === 'passed' ? 'PASS' : 'FAIL'),
          reasons: [],
          costUsd: 0,
          judged: null,
          stored: true,
        });
      }
    }
    let winnerRow = null;
    let judgeUnresolved = false;

    // One calibration RNG per run. Re-seeding it per batch would replay the
    // same first draw for every batch and sample all-or-nothing.
    const calibrationRng = rng ?? makeRunRng(`${runId}:calibration`);

    /**
     * Where a resumed unit picks the ladder back up: the highest rung that
     * already has a candidate row of its own. Units are told apart by the id
     * prefix (`''` for a single-unit run, `s<k>` for a subtask), so one
     * subtask's progress never moves another's starting rung.
     */
    const resumeStartRung = (prefix, width) => {
      if (!resume) return null;
      const re = new RegExp(`^${prefix}c(\\d+)a\\d+$`);
      const idxs = (store.listCandidates(runId) || [])
        .map((c) => re.exec(String(c.candidate_id ?? c.id ?? '')))
        .filter(Boolean)
        .map((m) => Number(m[1]));
      if (idxs.length === 0) return null;
      return Math.floor(Math.max(...idxs) / Math.max(1, width));
    };

    /**
     * How many rungs a unit has already spent, counted from the stored ids. A
     * reopened run adds this to every rung it computes, so the coders on the
     * resume command line sit ABOVE the ones the earlier attempt used: no
     * stored candidate id is written a second time (its worktree is usually
     * gone, destroyed by the hygiene step) and nothing already coded is coded
     * again.
     */
    const spentRungs = (prefix, width) => {
      const highest = resumeStartRung(prefix, width);
      return highest === null ? 0 : highest + 1;
    };

    /**
     * Register the stored, not yet destroyed worktrees of a unit this process
     * never opened (a subtask a resume carries over untouched), so the hygiene
     * step at the end treats them like the ones it opened itself. Best effort:
     * a worktree that is already gone is simply not adopted. Found live in
     * r_20260909_7159a980, where the carried-over subtask's losers stayed on
     * disk after the resumed run had finished.
     * @param {string} prefix candidate id prefix of the unit (`s<k>`)
     */
    const adoptStoredWorktrees = async (prefix) => {
      if (!resume) return;
      const re = new RegExp(`^${prefix}c\\d+a\\d+$`);
      for (const row of store.listCandidates(runId) || []) {
        const candidateId = String(row.candidate_id ?? row.id ?? '');
        if (!re.test(candidateId) || row.status === 'destroyed' || openWorktrees.has(candidateId)) continue;
        try {
          openWorktrees.set(candidateId, await driver.open({ repoPath, repoUrl, runId, candidateId }));
        } catch {
          // Nothing on disk to adopt.
        }
      }
    };

    /**
     * The whole generate -> cascade -> repair -> escalate ladder for ONE unit of
     * work. A single-unit run calls this once with the plan itself; a split run
     * calls it once per subtask with a narrowed plan and an id prefix. Keeping
     * it one closure is what guarantees the single-unit path is literally the
     * same code it was before subtasks existed.
     *
     * @param {object} unit
     * @param {object} unit.unitPlan the plan this unit is written and judged against
     * @param {number|null} [unit.subtaskIndex] null for a single-unit run, else the subtask index
     * @param {string} [unit.unitBaseRef] ref the unit's worktrees start from
     * @param {number} [unit.unitPerRung] candidates per rung
     * @param {number} [unit.unitRepairRounds] repair rounds
     * @param {number} [unit.unitStartRung] first rung to try
     * @returns {Promise<{winnerRow:object|null, rows:Array<object>}>}
     */
    const runLadder = async ({
      unitPlan,
      subtaskIndex = null,
      unitBaseRef = baseRef,
      unitPerRung = routedCandidates,
      unitRepairRounds = routedRepairRounds,
      unitStartRung = routed.startRung,
    }) => {
    const prefix = subtaskIndex === null ? '' : `s${subtaskIndex}`;
    const idFor = (idx, attempt) => (subtaskIndex === null
      ? candidateIdFor(idx, attempt)
      : subtaskCandidateIdFor(subtaskIndex, idx, attempt));
    const unitRows = [];
    // Spec 4.8. `audited` caps the plan audit at one per unit per run, so a
    // wrong `keep` cannot loop; `skipCandidates` holds the candidates that
    // failed a repair round exactly as they failed the round before it.
    let audited = false;
    const skipCandidates = new Set();
    if (resume) {
      // A resumed run must be judged against the plan the audit left behind,
      // not the one the planner wrote: the events are the record.
      const replayed = applyPlanCriterionEvents(unitPlan, store.eventsSince(runId, 0, 5000), subtaskIndex);
      if (replayed.changed) {
        unitPlan = replayed.plan;
        audited = true;
      }
    }

    /**
     * Record a finished cascade + judge pass and return the candidate rows.
     * `freshSemantic` re-grades already-written diffs after the plan changed
     * (spec 4.8); it costs one reviewer call per candidate and no coder turn.
     */
    const settle = async ({ attempts, coderRef, freshSemantic = false }) => {
      const cascade = await runCascade({
        runId,
        candidates: attempts.map((a) => ({ candidateId: a.candidateId, wt: a.wt, diffText: a.diff })),
        store: freshSemantic ? withoutSemanticCache(store) : store,
        plan: unitPlan,
        brief,
        baseline: baselineData,
        commands,
        bands,
        rubric,
        providers: { reviewer: providers.reviewer },
        config,
        signal,
        logger,
        mode,
      });

      const rows = [];
      for (const attempt of attempts) {
        const state = cascade.perCandidate.get(attempt.candidateId) || { decisions: {}, results: {}, verdict: null };
        rows.push({
          candidateId: attempt.candidateId,
          // Normally the rung's ref; a repair handed to a fallback coder carries
          // the ref of the model that actually wrote the diff (spec 4.3).
          generatorRef: attempt.generatorRef ?? coderRef,
          wt: attempt.wt,
          state,
          decisions: state.decisions,
          verdict: state.verdict?.verdict ?? 'JUDGE',
          reasons: reasonsOf(state),
          costUsd: (Number(attempt.costUsd) || 0) + stageCostOf(state),
          judged: null,
        });
      }

      // -- L4 on the residue (and always, in direct mode) -------------------
      const needsJudge = mode === 'direct'
        ? rows
        : rows.filter((row) => row.verdict === 'JUDGE');
      if (needsJudge.length > 0 && !checkBudget('judge')) {
        await pool(needsJudge, JUDGE_WORKERS, async (row) => {
          if (budgetExhausted()) return;
          try {
            const judged = await runJudge({
              runId,
              candidateId: row.candidateId,
              wt: row.wt,
              plan: unitPlan,
              brief: judgeBrief,
              rubric,
              priorResults: row.state.results,
              judge: providers.judge,
              store,
              config,
              commands,
              signal,
              reason: 'unsure',
            });
            row.judged = judged;
            row.costUsd += Number(judged.costUsd) || 0;
            row.verdict = judged.decision === 'good' ? 'PASS' : 'FAIL';
            if (judged.decision === 'bad') {
              row.reasons = [...row.reasons, ...(judged.blocking || []).map((b) => `judge blocked: ${b}`)];
            }
          } catch (err) {
            if (isLadderError(err, 'judge-failed')) {
              // Unresolved: the cascade could not decide and neither could the
              // judge. That is a JUDGE verdict, not a failure of the patch.
              judgeUnresolved = true;
              row.judgeFailed = true;
              row.verdict = 'JUDGE';
              row.reasons = [...row.reasons, 'judge could not produce a verdict'];
              store.emit(runId, 'judge.failed', { candidateId: row.candidateId, message: scrubSecrets(err.message) });
              return;
            }
            throw err;
          }
        });
      }

      // -- calibration (never changes a verdict) ----------------------------
      // An empty diff has nothing for the judge to calibrate against, and a
      // candidate the judge already ruled on in this batch needs no second call.
      const calibratable = rows.filter((row) => {
        if (row.judged) return false;
        const attempt = attempts.find((a) => a.candidateId === row.candidateId);
        return attempt && typeof attempt.diff === 'string' && attempt.diff.trim() !== '';
      });
      const sample = selectCalibrationSample({
        perCandidate: new Map(calibratable.map((row) => [row.candidateId, row.state])),
        rate: config.calibrationRate,
        rng: calibrationRng,
      });
      for (const candidateId of sample) {
        if (budgetExhausted()) break;
        const row = rows.find((r) => r.candidateId === candidateId);
        if (!row) continue;
        try {
          const judged = await runJudge({
            runId,
            candidateId,
            wt: row.wt,
            plan: unitPlan,
            brief: judgeBrief,
            rubric,
            priorResults: row.state.results,
            judge: providers.judge,
            store,
            config,
            commands,
            signal,
            reason: 'calibration',
          });
          row.costUsd += Number(judged.costUsd) || 0;
        } catch (err) {
          if (!isLadderError(err, 'judge-failed')) throw err;
          store.emit(runId, 'judge.failed', { candidateId, kind: 'calibration', message: scrubSecrets(err.message) });
        }
      }

      for (const row of rows) {
        allCandidates.set(row.candidateId, row);
        // A re-settle grades the SAME candidate again (spec 4.8): the fresh row
        // replaces the stale one rather than being counted a second time.
        const seen = unitRows.findIndex((r) => r.candidateId === row.candidateId);
        if (seen >= 0) unitRows[seen] = row;
        else unitRows.push(row);
        store.upsertCandidate({
          runId,
          candidateId: row.candidateId,
          status: row.verdict === 'PASS' ? 'passed' : row.verdict.toLowerCase(),
        });
      }
      return rows;
    };

    let unitWinner = null;
    // A reopened unit starts at the bottom of the ladder it was just given and
    // pushes its ids past the rungs the earlier attempt spent. An interrupted
    // one is resuming INSIDE its own ladder, so it picks that ladder back up
    // where it stopped and offsets nothing.
    const rungOffset = reopened ? spentRungs(prefix, unitPerRung) : 0;
    const firstRung = reopened
      ? 0
      : Math.min(resumeStartRung(prefix, unitPerRung) ?? unitStartRung, coderRungs.length - 1);

    ladder:
    for (let rung = Math.max(0, firstRung); rung < coderRungs.length; rung += 1) {
      if (!unitPlan) break;
      const coder = coderRungs[rung];
      const coderRef = refOf(coder);
      // `rung` indexes the ladder this call was given; `absRung` counts from the
      // first rung the run ever spent on this unit. They differ only on a
      // reopen, and it is `absRung` that ids and event payloads report.
      const absRung = rung + rungOffset;
      if (checkBudget('coder')) break;

      // -- generate this rung's attempts -----------------------------------
      let attempts = [];
      for (let slot = 0; slot < unitPerRung; slot += 1) {
        const idx = absRung * unitPerRung + slot;
        attempts.push({ idx, attempt: 0, candidateId: idFor(idx, 0), wt: null, diff: null, costUsd: 0 });
      }

      await pool(attempts, unitPerRung, async (att) => {
        const prior = resume ? store.getCandidate(runId, att.candidateId) : null;
        if (prior && prior.status !== 'destroyed') {
          att.wt = await driver.open({ repoPath, repoUrl, runId, candidateId: att.candidateId });
          openWorktrees.set(att.candidateId, att.wt);
          if (prior.diff_text) {
            // The coder already finished this attempt in the earlier run.
            att.diff = prior.diff_text;
            return;
          }
        } else {
          att.wt = await driver.create({
            repoPath, repoUrl, baseRef: unitBaseRef, runId, candidateId: att.candidateId, reuse: resume,
          });
          openWorktrees.set(att.candidateId, att.wt);
        }
        store.upsertCandidate({
          runId,
          candidateId: att.candidateId,
          idx: att.idx,
          attempt: att.attempt,
          generatorRef: coderRef,
          worktree: { path: att.wt.path ?? null, branch: att.wt.branch ?? null },
          status: 'coding',
        });
        store.emit(runId, 'candidate.start', { candidateId: att.candidateId, generatorRef: coderRef, rung: absRung });
        if (budgetExhausted()) return;
        try {
          const coded = await runCoder({
            runId,
            candidateId: att.candidateId,
            wt: att.wt,
            plan: unitPlan,
            brief,
            baseline: baselineData,
            commands,
            rubric,
            coder,
            store,
            config,
            signal,
            logger,
          });
          att.diff = coded.diff;
          att.costUsd = coded.costUsd;
          // `reason` is what the repair round reads to decide whether this coder
          // could act at all; `protocol` says whether it had to be moved to the
          // text tool protocol to get there (spec 4.3).
          att.reason = coded.reason;
          store.emit(runId, 'candidate.diff', {
            candidateId: att.candidateId,
            changedFiles: coded.changedFiles.length,
            turns: coded.turns,
            finished: coded.finished,
            reason: coded.reason,
            protocol: coded.protocol,
          });
        } catch (err) {
          att.error = noteCandidateError(att.candidateId, err);
        }
      });

      // A rung whose every attempt died of a rung-level fault (the model does
      // not take tools, the rung is not armed, the endpoint is down) is a
      // failure of the RUNG, not of the run: escalate. Only the last rung
      // throws.
      const erroredAttempts = attempts.filter((a) => a.error);
      if (attempts.length > 0 && erroredAttempts.length === attempts.length) {
        const last = erroredAttempts[erroredAttempts.length - 1].error;
        const rungLevel = erroredAttempts.every((a) => RUNG_LEVEL.has(a.error.code));
        if (!rungLevel || rung === coderRungs.length - 1) {
          throw new LadderError(last.code, last.message, { runId, rung });
        }
        store.emit(runId, 'escalate', {
          from: coderRef,
          to: refOf(coderRungs[rung + 1]),
          rung: absRung + 1,
          reason: 'rung-failed',
        });
        logger?.warn?.(`rung ${coderRef} failed (${last.code}); escalating`);
        continue;
      }
      // A mixed rung carries on with whatever survived.
      attempts = attempts.filter((a) => !a.error);

      if (checkBudget('cascade')) break;
      let rows = await settle({ attempts, coderRef });
      unitWinner = pickWinner(rows);
      if (unitWinner) break ladder;

      // -- repair rounds ----------------------------------------------------
      for (let round = 1; round <= unitRepairRounds; round += 1) {
        if (checkBudget('repair')) break ladder;
        const failing = rankCandidates(rows.filter((row) => row.verdict !== 'PASS'));
        if (failing.length === 0) break;
        const repaired = [];
        for (const row of failing) {
          // Spec 4.8: this candidate came out of its last repair with exactly
          // the failure it went in with, and the plan has not moved since.
          // Another identical round buys another identical rejection.
          if (skipCandidates.has(row.candidateId)) continue;
          const source = attempts.find((a) => a.candidateId === row.candidateId)
            ?? { idx: Number(store.getCandidate(runId, row.candidateId)?.idx ?? 0) };
          const feedback = row.judged && row.judged.decision === 'bad'
            ? {
              stage: 'l4-judge',
              reasons: [row.judged.rationale, ...(row.judged.blocking || []).map((b) => `blocking: ${b}`)]
                .filter(Boolean),
              hits: [],
              failing: [],
            }
            : failingFeedback(row.state);
          // A repair round is a model call. Spending one on "the stage timed
          // out" or "the provider is down" buys nothing: the candidate keeps
          // whatever verdict it has and the run carries on.
          const skip = repairSkipReason(row, feedback);
          if (skip) {
            store.emit(runId, 'repair.skipped', {
              candidateId: row.candidateId,
              round,
              stage: feedback.stage ?? null,
              reason: skip,
            });
            continue;
          }
          // Spec 4.3: an attempt that ended in `no-tool-calls` was written by a
          // model that could not act, not by one that acted wrongly. Repeating
          // the round on the same coder repeats the failure (live run
          // r_20260909_b4d9777c), so the round goes to a fallback coder when
          // there is one to give it to.
          const fallback = source.reason === 'no-tool-calls' ? fallbackCoderFor(rung, coderRef) : null;
          repaired.push({
            idx: source.idx,
            attempt: round,
            candidateId: idFor(source.idx, round),
            sourceCandidateId: row.candidateId,
            // The diff the failing attempt produced, for the coach to read.
            priorDiff: source.diff ?? null,
            wt: row.wt,
            diff: null,
            costUsd: 0,
            feedback,
            hints: null,
            coder: fallback ?? coder,
            generatorRef: fallback ? refOf(fallback) : coderRef,
            fallbackFrom: fallback ? coderRef : null,
          });
        }
        if (repaired.length === 0) break;

        await pool(repaired, unitPerRung, async (att) => {
          if (budgetExhausted()) return;
          store.upsertCandidate({
            runId,
            candidateId: att.candidateId,
            idx: att.idx,
            attempt: att.attempt,
            generatorRef: att.generatorRef,
            worktree: { path: att.wt?.path ?? null, branch: att.wt?.branch ?? null },
            status: 'repairing',
          });
          if (att.fallbackFrom) {
            store.emit(runId, 'coder.fallback', {
              candidateId: att.candidateId,
              from: att.fallbackFrom,
              to: att.generatorRef,
              why: 'no-tool-calls',
            });
          }
          store.emit(runId, 'repair', {
            candidateId: att.candidateId,
            round,
            stage: att.feedback.stage,
            reasons: att.feedback.reasons.slice(0, 6),
          });
          // The coach (spec 4.3): one cheap orchestrator call that turns the
          // failing evidence into concrete hints for the coder. Never fatal,
          // and skipped entirely when the orchestrator IS the coder.
          const coached = await coachRepair({
            orchestrator: providers.orchestrator,
            // The fallback coder, when there is one: the "do not coach yourself"
            // rule has to apply to the model that will read the hints.
            coder: att.coder,
            plan: unitPlan,
            feedback: [
              att.feedback.stage ? `stage: ${att.feedback.stage}` : '',
              ...(att.feedback.reasons || []),
              ...(att.feedback.failing || []).map((f) => `failing test: ${typeof f === 'string' ? f : f?.name ?? ''}`),
            ].filter(Boolean).join('\n'),
            diffText: att.priorDiff,
            transcriptTail: transcriptTailFor(store, runId, att.sourceCandidateId ?? att.candidateId),
            config,
            store,
            runId,
            candidateId: att.candidateId,
            round,
            budgetExhausted: budgetExhausted(),
            signal,
          });
          att.hints = coached.hints.length > 0 ? coached.hints : null;
          try {
            const coded = await runRepair({
              runId,
              candidateId: att.candidateId,
              wt: att.wt,
              plan: unitPlan,
              brief,
              baseline: baselineData,
              commands,
              rubric,
              coder,
              coderOverride: att.coder !== coder ? att.coder : null,
              store,
              config,
              signal,
              logger,
              maxTurns: repairTurns,
              feedback: att.feedback,
              hints: att.hints ?? null,
            });
            att.diff = coded.diff;
            att.costUsd = coded.costUsd;
            att.reason = coded.reason;
          } catch (err) {
            att.error = noteCandidateError(att.candidateId, err);
          }
        });

        if (checkBudget('cascade')) break ladder;
        // Every repair died on the provider: escalate rather than cascade over
        // a set of candidates that never got a second pass.
        if (repaired.every((a) => a.error)) break;
        const priorRows = rows;
        attempts = repaired.filter((a) => !a.error);
        rows = await settle({ attempts, coderRef });
        unitWinner = pickWinner(rows);
        if (unitWinner) break ladder;

        /* -- the plan audit (spec 4.8) -------------------------------------
         * A repair round assumes the plan is right and the patch is wrong.
         * When one acceptance criterion rejected every candidate before the
         * round and every candidate after it, that assumption is the thing to
         * test: in run r_20260909_f02fbbb9 six coder sessions were spent on a
         * criterion ("keep the CRLF line endings") that the LF-only files made
         * unsatisfiable, and nothing in the run ever questioned it. */
        const sourceOf = new Map(repaired.map((a) => [a.candidateId, a.sourceCandidateId ?? null]));
        // The detector reads a row as (rejecting stage, reasons): the same two
        // things a repair round is given to work with.
        const stuckRow = (row) => ({
          candidateId: row.candidateId,
          sourceId: sourceOf.get(row.candidateId) ?? null,
          verdict: row.verdict,
          stage: row.judged && row.judged.decision === 'bad'
            ? 'l4-judge'
            : failingFeedback(row.state).stage,
          reasons: row.reasons,
        });
        const stuck = stuckAnalysis({ before: priorRows.map(stuckRow), after: rows.map(stuckRow) });
        if (stuck.criteria.length > 0 || stuck.candidates.length > 0) {
          store.emit(runId, 'repair.stuck', {
            unit: subtaskIndex,
            round,
            criteria: stuck.criteria,
            candidates: stuck.candidates,
          });
        }
        let planChanged = false;
        if (stuck.criteria.length > 0 && !audited) {
          audited = true;
          const entries = criterionEntries(unitPlan);
          const stuckCriteria = stuck.criteria
            .map((index) => entries.find((e) => e.index === index))
            .filter(Boolean)
            .map((e) => ({ index: e.index, text: e.text }));
          const best = rankCandidates(rows)[0] ?? null;
          const bestDiff = attempts.find((a) => a.candidateId === best?.candidateId)?.diff ?? null;
          const auditResult = await auditCriteria({
            orchestrator: providers.orchestrator,
            plan: unitPlan,
            criteria: stuckCriteria,
            task: task?.text ?? '',
            diffText: bestDiff,
            fileFacts: await fileFactsFor(
              best?.wt ?? null,
              pathsNamedIn(stuckCriteria.map((c) => c.text)),
              { baseRef: unitBaseRef },
            ),
            config,
            store,
            runId,
            unit: subtaskIndex,
            round,
            budgetExhausted: budgetExhausted(),
            signal,
          });
          if (auditResult.changed) {
            unitPlan = auditResult.plan;
            planChanged = true;
            // Re-grade what is already written against the amended plan. The
            // diffs may have been right all along, so this is a cascade with no
            // coder session: L0-L2 stay cached, L3 runs once per candidate.
            if (!checkBudget('cascade')) {
              rows = await settle({ attempts, coderRef, freshSemantic: true });
              unitWinner = pickWinner(rows);
              if (unitWinner) break ladder;
            }
          }
        }
        // Give up on a candidate only when the blockage is a criterion the
        // audit could not move: a candidate that keeps failing the same TEST
        // still has a coach hint and a repair round it can spend. Skipping is
        // also only fair while the plan holds still - an amendment gives every
        // candidate something new to aim at.
        if (!planChanged && stuck.criteria.length > 0) {
          for (const id of stuck.candidates) skipCandidates.add(id);
        }
      }

      if (rung < coderRungs.length - 1) {
        store.emit(runId, 'escalate', {
          from: coderRef,
          to: refOf(coderRungs[rung + 1]),
          rung: absRung + 1,
        });
      }
    }

      return { winnerRow: unitWinner, rows: unitRows };
    };

    /* ------------------------------------------------ 4b. units of work */
    // Subtask coordination (spec 4.4). A split is attempted only when the plan
    // proposed one, the operator did not turn it off, the graph is runnable and
    // the driver can merge. Every other case falls back to the single-unit
    // ladder, which is the path the hardening tests pin byte for byte.
    const subtaskResults = [];
    let integrationWt = null;
    let integrationStop = null;
    let split = null;
    if (taskPlan && config.subtasks !== false
      && Array.isArray(taskPlan.subtasks) && taskPlan.subtasks.length > 0) {
      const check = validateSubtasks(taskPlan);
      if (!check.ok) {
        store.emit(runId, 'subtasks.rejected', {
          reason: check.reason,
          count: taskPlan.subtasks.length,
        });
      } else if (typeof driver.merge !== 'function') {
        store.emit(runId, 'subtasks.rejected', {
          reason: `driver ${driver.kind ?? '?'} cannot merge`,
          count: check.subtasks.length,
        });
      } else {
        split = check.subtasks;
      }
    }

    if (split) {
      const levels = topoLevels(split);
      const indexOf = new Map(split.map((st, i) => [st.id, i]));
      const width = Math.max(1, Math.trunc(Number(config.subtaskConcurrency)) || 2);
      store.emit(runId, 'subtasks.plan', {
        levels: levels.map((level) => level.map((st) => st.id)),
        subtasks: split.map((st) => ({
          id: st.id,
          title: st.title ?? null,
          files: st.files ?? [],
          dependsOn: st.depends_on ?? [],
        })),
        concurrency: width,
      });

      // Resume: subtask progress is reconstructed from subtask.end events. Only
      // a PASS that left a branch is reused; a subtask that failed or ended
      // JUDGE is run again, which is what makes `resume --coder <stronger>` a
      // recovery rather than a replay of the same FAIL.
      const priorEvents = resume ? store.eventsSince(runId, 0, 5000) : [];
      const alreadyDone = finishedSubtasks(priorEvents);
      const priorVerdicts = priorSubtaskVerdicts(priorEvents);

      integrationWt = await driver.create({
        repoPath, repoUrl, baseRef, runId, candidateId: 'integration', reuse: resume,
      });
      openWorktrees.set('integration', integrationWt);
      // Every level starts from the integration commit that carries the levels
      // before it, so a dependent subtask sees its dependencies' work.
      let integrationRef = baseRef;
      // Why the integration stopped, carried into the FAIL verdict so a reader
      // sees what git said instead of a bare 'merge conflict'.
      let integrationFailReason = null;

      /** True when any of `paths` still carries a conflict marker. */
      const markersRemain = async (wt, paths) => {
        for (const p of paths) {
          try {
            const text = await wt.readFile(p);
            if (typeof text === 'string' && text.includes('<<<<<<<')) return true;
          } catch {
            // Unreadable after a merge is not evidence of a conflict.
          }
        }
        return false;
      };

      /** A merge detail, scrubbed and short enough to live in an event row. */
      const mergeDetail = (value) => {
        const text = scrubSecrets(String(value ?? '')).trim();
        return text === '' ? null : text.slice(0, 500);
      };

      /** Merge one finished subtask, repairing a conflict at most once. */
      const mergeSubtask = async (entry) => {
        let res;
        try {
          res = await driver.merge(integrationWt, entry.branch);
        } catch (err) {
          res = { ok: false, conflicts: [], detail: scrubSecrets(err?.message || String(err)) };
        }
        if (res.ok === true) {
          store.emit(runId, 'integration.merge', { id: entry.id, ok: true, conflicts: [] });
          return true;
        }
        const conflicts = res.conflicts ?? [];
        // Without the detail a failed merge is undiagnosable in the log: git
        // says why it refused, and nothing else does (found live).
        const detail = mergeDetail(res.detail);
        store.emit(runId, 'integration.merge', {
          id: entry.id,
          ok: false,
          conflicts,
          detail,
        });
        // No conflicted file means git never started the merge (a merge left in
        // progress, an unknown ref). There is nothing for a coder to resolve,
        // so a repair session would only produce a merge that never happened.
        if (conflicts.length === 0) {
          integrationFailReason = detail
            ? `merge of ${entry.id} failed: ${detail}`
            : `merge of ${entry.id} failed`;
          return false;
        }
        if (budgetExhausted()) return false;
        // One repair-style session in the integration worktree, with the
        // conflict markers as the feedback and the strongest rung driving.
        try {
          await runRepair({
            runId,
            candidateId: 'integration',
            wt: integrationWt,
            plan: taskPlan,
            brief,
            baseline: baselineData,
            commands,
            rubric,
            coder: coderRungs[coderRungs.length - 1],
            store,
            config,
            signal,
            logger,
            maxTurns: repairTurns,
            feedback: {
              stage: 'integration',
              reasons: [conflictFeedback({
                conflicts,
                subtaskId: entry.id,
                detail: res.detail ?? '',
              })],
              hits: [],
              failing: [],
            },
          });
        } catch (err) {
          noteCandidateError('integration', err);
          return false;
        }
        if (await markersRemain(integrationWt, conflicts)) {
          store.emit(runId, 'integration.merge', {
            id: entry.id,
            ok: false,
            conflicts,
            repaired: true,
            detail,
          });
          return false;
        }
        // The markers are gone, so conclude the merge NOW rather than at the
        // end of the level: git refuses to start another merge while MERGE_HEAD
        // is set, and the coder cannot run `git commit` itself. The level-end
        // commit still runs and is a no-op when nothing changed.
        let committed = null;
        try {
          committed = await integrationWt.commit(
            `kolm code: merge ${entry.id} (conflict resolved)\n\nrun: ${runId}`,
          );
        } catch (err) {
          const failed = mergeDetail(err?.message || String(err));
          store.emit(runId, 'integration.merge', {
            id: entry.id,
            ok: false,
            conflicts,
            repaired: true,
            detail: failed,
          });
          integrationFailReason = `merge of ${entry.id} failed: ${failed ?? 'commit failed'}`;
          return false;
        }
        if (committed?.sha) integrationRef = committed.sha;
        // A resolution that keeps HEAD's side leaves a tree that looks finished
        // whether or not the branch landed, so ask the history, not the tree.
        if (typeof integrationWt.isAncestor === 'function') {
          let landed = true;
          try {
            landed = await integrationWt.isAncestor(entry.branch);
          } catch (err) {
            // A driver that cannot answer is not evidence of a bad merge.
            logger?.warn?.(`could not verify the merge of ${entry.id}: ${scrubSecrets(err?.message || String(err))}`);
          }
          if (landed === false) {
            const missing = `branch ${entry.branch} is not in the integration history after repair`;
            store.emit(runId, 'integration.merge', {
              id: entry.id,
              ok: false,
              conflicts,
              repaired: true,
              detail: missing,
            });
            integrationFailReason = `merge of ${entry.id} failed: ${missing}`;
            return false;
          }
        }
        store.emit(runId, 'integration.merge', {
          id: entry.id,
          ok: true,
          conflicts: [],
          repaired: true,
        });
        return true;
      };

      // Subtasks that did not pass, so anything downstream of them has nothing
      // to build on. Levels are topological, so every dependency of a subtask
      // is already decided by the time its own level runs.
      const notPassedIds = new Set();
      // Which failed or skipped dependencies each skipped subtask waited on,
      // for the integration verdict's reasons.
      const skippedBecause = new Map();

      levels:
      for (const level of levels) {
        if (checkBudget('subtask')) { integrationStop = 'budget'; break; }
        const levelOut = [];
        await pool(level, width, async (st) => {
          const k = indexOf.get(st.id) ?? 0;
          const prior = alreadyDone.get(st.id);
          if (prior) {
            await adoptStoredWorktrees(`s${k}`);
            levelOut.push({ ...prior, index: k });
            return;
          }
          // A dependent of a failed or skipped subtask would be coded against an
          // integration ref that lacks the work it depends on, and would burn a
          // whole ladder to produce a diff nobody can use (found live,
          // r_20260909_f02fbbb9). It is skipped instead, loudly, and never gets
          // a subtask.start/subtask.end pair.
          const because = (Array.isArray(st.depends_on) ? st.depends_on : [])
            .filter((dep) => notPassedIds.has(dep));
          if (because.length > 0) {
            skippedBecause.set(st.id, because);
            store.emit(runId, 'subtask.skipped', { id: st.id, index: k, because });
            levelOut.push({
              id: st.id,
              index: k,
              verdict: 'SKIPPED',
              winner: null,
              branch: null,
              sha: null,
              costUsd: 0,
            });
            return;
          }
          if (budgetExhausted()) return;
          const priorVerdict = priorVerdicts.get(st.id);
          if (priorVerdict) {
            store.emit(runId, 'subtask.retry', { id: st.id, index: k, priorVerdict });
          }
          store.emit(runId, 'subtask.start', {
            id: st.id,
            index: k,
            title: st.title ?? null,
            files: st.files ?? [],
            dependsOn: st.depends_on ?? [],
          });
          const unit = await runLadder({
            unitPlan: subPlanFor(taskPlan, st, k),
            subtaskIndex: k,
            unitBaseRef: integrationRef,
          });
          const costUsd = unit.rows.reduce((sum, r) => sum + (Number(r.costUsd) || 0), 0);
          let entry;
          if (unit.winnerRow) {
            const committed = await unit.winnerRow.wt.commit(
              `kolm code: ${st.title ?? st.id}\n\nrun: ${runId}\nsubtask: ${st.id}`,
            );
            entry = {
              id: st.id,
              index: k,
              verdict: 'PASS',
              winner: unit.winnerRow.candidateId,
              branch: committed?.branch ?? unit.winnerRow.wt.branch ?? null,
              sha: committed?.sha ?? null,
              costUsd,
            };
          } else {
            entry = {
              id: st.id,
              index: k,
              verdict: unit.rows.some((r) => r.verdict === 'JUDGE') ? 'JUDGE' : 'FAIL',
              winner: null,
              branch: null,
              sha: null,
              costUsd,
            };
          }
          store.emit(runId, 'subtask.end', entry);
          levelOut.push(entry);
        });

        // Merge in plan order so the integration history is deterministic.
        levelOut.sort((a, b) => a.index - b.index);
        for (const entry of levelOut) {
          subtaskResults.push(entry);
          if (entry.verdict !== 'PASS' || !entry.branch) {
            notPassedIds.add(entry.id);
            integrationStop = integrationStop ?? 'subtask-failed';
            continue;
          }
          const merged = await mergeSubtask(entry);
          if (!merged) { integrationStop = 'integration-conflict'; break levels; }
        }
        if (integrationStop === 'budget') break;
        const committed = await integrationWt.commit(
          `kolm code: integration\n\nrun: ${runId}\nlevel: ${levels.indexOf(level)}`,
        );
        if (committed?.sha) integrationRef = committed.sha;
      }

      // Every subtask that did not produce a mergeable branch, in plan order.
      // The survivors are already merged and committed, so the integration
      // branch holds the partial work for a later resume; what it must not do is
      // pretend that partial work is the change the plan describes.
      const notPassedSubtasks = subtaskResults
        .filter((s) => s.verdict !== 'PASS' || !s.branch)
        .sort((a, b) => a.index - b.index);
      /** One ASCII line saying why this subtask is not in the integration. */
      const missingReason = (s) => {
        if (s.verdict === 'SKIPPED') {
          const because = skippedBecause.get(s.id) ?? [];
          return because.length > 0
            ? `subtask ${s.id} skipped (depends on ${because.join(', ')})`
            : `subtask ${s.id} skipped`;
        }
        if (s.verdict === 'JUDGE') return `subtask ${s.id} was left unresolved`;
        return `subtask ${s.id} failed`;
      };

      if (integrationStop === 'integration-conflict') {
        stopReason = 'integration-conflict';
        store.emit(runId, 'integration.verdict', {
          verdict: 'FAIL',
          reasons: [integrationFailReason ?? 'merge conflict'],
        });
      } else if (notPassedSubtasks.length > 0) {
        /* ----------------------------- a partial diff is not the change --- */
        // Judging the merged survivors as if they were the whole plan grades a
        // change nobody proposed: in r_20260909_f02fbbb9 both subtasks failed,
        // nothing merged, and the cascade dutifully marked the untouched base
        // bad on l0-meta. So no `integration` candidate is drawn and no stage
        // runs; the verdict says which subtask is missing and why.
        store.emit(runId, 'integration.verdict', {
          verdict: 'FAIL',
          winner: null,
          reasons: notPassedSubtasks.map(missingReason).slice(0, 6),
          subtasks: subtaskResults.map((s) => s.id),
        });
        if (!stopReason) stopReason = 'subtask-failed';
      } else {
        /* -------------------------------- the whole change, judged once --- */
        const integrationCommands = { ...commands };
        if (taskPlan.integration_test) integrationCommands.test = taskPlan.integration_test;
        const integrationDiff = await integrationWt.diff();
        store.upsertCandidate({
          runId,
          candidateId: 'integration',
          generatorRef: refOf(providers.orchestrator),
          worktree: { path: integrationWt.path ?? null, branch: integrationWt.branch ?? null },
          status: 'coding',
        });
        const cascade = await runCascade({
          runId,
          candidates: [{ candidateId: 'integration', wt: integrationWt, diffText: integrationDiff }],
          store,
          plan: taskPlan,
          brief,
          baseline: baselineData,
          commands: integrationCommands,
          bands,
          rubric,
          providers: { reviewer: providers.reviewer },
          config,
          signal,
          logger,
          mode,
        });
        const state = cascade.perCandidate.get('integration')
          || { decisions: {}, results: {}, verdict: null };
        let integrationVerdict = state.verdict?.verdict ?? 'JUDGE';
        let judged = null;
        if (integrationVerdict === 'JUDGE' && !checkBudget('judge')) {
          try {
            judged = await runJudge({
              runId,
              candidateId: 'integration',
              wt: integrationWt,
              plan: taskPlan,
              brief: judgeBrief,
              rubric,
              priorResults: state.results,
              judge: providers.judge,
              store,
              config,
              commands: integrationCommands,
              signal,
              reason: 'unsure',
            });
            integrationVerdict = judged.decision === 'good' ? 'PASS' : 'FAIL';
          } catch (err) {
            if (!isLadderError(err, 'judge-failed')) throw err;
            judgeUnresolved = true;
            integrationVerdict = 'JUDGE';
            store.emit(runId, 'judge.failed', {
              candidateId: 'integration',
              message: scrubSecrets(err.message),
            });
          }
        }
        const integrationRow = {
          candidateId: 'integration',
          generatorRef: refOf(providers.orchestrator),
          wt: integrationWt,
          state,
          decisions: state.decisions,
          verdict: integrationVerdict,
          reasons: reasonsOf(state),
          costUsd: stageCostOf(state) + (Number(judged?.costUsd) || 0),
          judged,
        };
        allCandidates.set('integration', integrationRow);
        store.upsertCandidate({
          runId,
          candidateId: 'integration',
          status: integrationVerdict === 'PASS' ? 'passed' : integrationVerdict.toLowerCase(),
        });
        store.emit(runId, 'integration.verdict', {
          verdict: integrationVerdict,
          winner: integrationVerdict === 'PASS' ? 'integration' : null,
          reasons: integrationRow.reasons.slice(0, 6),
          subtasks: subtaskResults.map((s) => s.id),
        });
        // This branch only runs when every subtask passed and merged, so a FAIL
        // here is the merged change losing on its own merits, not a missing
        // subtask: 'subtask-failed' is handled above, before any stage runs.
        if (integrationVerdict === 'PASS') winnerRow = integrationRow;
        else if (integrationVerdict === 'JUDGE') judgeUnresolved = true;
      }
    } else {
      const unit = await runLadder({ unitPlan: taskPlan });
      winnerRow = unit.winnerRow;
    }

    /* ------------------------------------------------------- 5. the winner */
    const candidateList = [...allCandidates.values()];
    let branch = null;
    let commitSha = null;
    let verdict = 'FAIL';
    if (winnerRow) {
      verdict = 'PASS';
      const message = `kolm code: ${taskPlan.title || task.title || 'change'}\n\nrun: ${runId}\ncandidate: ${winnerRow.candidateId}`;
      const committed = await winnerRow.wt.commit(message);
      branch = committed?.branch ?? winnerRow.wt.branch ?? null;
      commitSha = committed?.sha ?? null;
      store.emit(runId, 'winner', { candidateId: winnerRow.candidateId, branch, sha: commitSha });
      if (!stopReason) stopReason = 'winner';
    } else if (judgeUnresolved || candidateList.some((row) => !row.stored && row.verdict === 'JUDGE')) {
      verdict = 'JUDGE';
      if (!stopReason) stopReason = 'judge-unresolved';
    } else if (!stopReason) {
      stopReason = 'ladder-exhausted';
    }

    /* ------------------------------------------------------ 6. the receipt */
    const cost = store.costSummary(runId);
    const calibration = calibrationReport({ store, runId });
    const receipt = signReceipt(buildReceipt({
      run: store.getRun(runId),
      candidates: candidateList.map((row) => ({
        candidate_id: row.candidateId,
        generator: row.generatorRef,
        verdict: row.verdict,
        diff_sha256: store.getCandidate(runId, row.candidateId)?.diff_sha256 ?? null,
      })),
      results: store.resultsFor(runId),
      judgments: store.listJudgments(runId),
      cost,
      winner: winnerRow ? winnerRow.candidateId : null,
      branch,
      sha: commitSha,
      mode,
      bands,
      rubric,
      config,
      models: {
        orchestrator: refOf(providers.orchestrator),
        judge: refOf(providers.judge),
        coders: coderRungs.map(refOf).filter(Boolean),
        reviewer: refOf(providers.reviewer),
      },
      calibration,
      stopReason,
      verdict,
      subtasks: subtaskResults.map((s) => ({
        id: s.id,
        winner: s.winner,
        branch: s.branch,
        sha: s.sha,
      })),
    }));

    let receiptPath = null;
    try {
      receiptPath = writeReceipt(receipt, path.dirname(receiptPathFor(config, runId)));
    } catch (err) {
      logger?.warn?.(`could not write the receipt file: ${scrubSecrets(err.message)}`);
    }

    store.updateRun(runId, {
      status: 'done',
      winner_candidate: winnerRow ? winnerRow.candidateId : null,
      total_cost_usd: cost.total_usd,
      receipt_json: receipt,
    });

    /* -------------------------------------------------- 7. worktree hygiene */
    if (config.keepWorktrees !== 'all') {
      // A repaired attempt shares its predecessor's worktree object, so the
      // winner has to be excluded by IDENTITY, not just by candidate id: the
      // winning worktree is never destroyed.
      const done = new Set();
      for (const [candidateId, wt] of openWorktrees) {
        if (!wt || done.has(wt)) continue;
        if (winnerRow && (wt === winnerRow.wt || candidateId === winnerRow.candidateId)) continue;
        done.add(wt);
        try {
          await wt.destroy();
          // Only a candidate that was drawn has a row to mark; the integration
          // worktree of a run that never judged an integration has none, and
          // must not gain a phantom "destroyed" candidate on the way out.
          if (store.getCandidate(runId, candidateId)) {
            store.upsertCandidate({ runId, candidateId, status: 'destroyed' });
          }
        } catch (err) {
          logger?.warn?.(`could not destroy worktree ${candidateId}: ${scrubSecrets(err.message)}`);
        }
      }
    }

    const out = {
      runId,
      verdict,
      winner: winnerRow ? winnerRow.candidateId : null,
      receipt,
      receiptPath,
      branch,
      sha: commitSha,
      costUsd: cost.total_usd,
      funnel: store.funnel({ runIds: [runId] }),
      candidates: candidateList.map((row) => ({
        candidateId: row.candidateId,
        generatorRef: row.generatorRef,
        verdict: row.verdict,
        decisions: row.decisions,
        costUsd: row.costUsd,
      })),
      durationMs: Date.now() - started,
      stopReason,
      subtasks: subtaskResults.map((s) => ({
        id: s.id,
        verdict: s.verdict,
        winner: s.winner,
        branch: s.branch,
        sha: s.sha,
        costUsd: s.costUsd,
      })),
    };

    /* --------------------------------------------------------- 8. lessons */
    // What this run learned about this repository, for the next run's brief.
    // Best-effort by construction: a lessons file that cannot be written must
    // never turn a finished run into a failed one.
    try {
      if (config.lessons !== false) {
        const proposed = extractLessonCandidates(store.eventsSince(runId, 0, 5000));
        if (proposed.length > 0) {
          const before = loadLessons(lessonsConfig, repoKey);
          const merged = mergeLessons(before, proposed, { runId });
          saveLessons(lessonsConfig, repoKey, merged);
          store.emit(runId, 'run.lessons', { added: proposed.length, total: merged.length });
        }
      }
    } catch (err) {
      logger?.warn?.(`could not record lessons: ${scrubSecrets(err?.message || String(err))}`);
    }

    store.emit(runId, 'run.end', {
      verdict,
      winner: out.winner,
      costUsd: out.costUsd,
      durationMs: out.durationMs,
      stopReason,
      receiptPath,
    });
    return out;
  } catch (err) {
    const message = scrubSecrets(err?.message || String(err));
    const code = isLadderError(err) ? err.code : 'execution';
    try {
      store.updateRun(runId, { status: 'error' });
      store.emit(runId, 'run.error', { message, code });
    } catch {
      // The store may itself be the thing that failed; the throw below stands.
    }
    logger?.error?.(`run ${runId} failed (${code}): ${message}`);
    // The error path used to leak every candidate worktree it had opened. The
    // branch (kolm/code/<runId>/<cid>) survives either way, so the work is not
    // lost; only the checkout goes.
    if (config.keepWorktrees !== 'all') {
      const done = new Set();
      for (const [candidateId, wt] of openWorktrees) {
        if (!wt || done.has(wt)) continue;
        done.add(wt);
        try {
          await wt.destroy();
        } catch (e) {
          logger?.warn?.(`could not destroy worktree ${candidateId}: ${scrubSecrets(e?.message || String(e))}`);
          continue;
        }
        try {
          if (store.getCandidate(runId, candidateId)) {
            store.upsertCandidate({ runId, candidateId, status: 'destroyed' });
          }
        } catch {
          // The store may itself be the thing that failed.
        }
      }
    }
    if (isLadderError(err)) throw err;
    throw new LadderError('execution', message, { runId });
  } finally {
    if (baseWt && config.keepWorktrees !== 'all') {
      await baseWt.destroy().catch(() => {});
    }
  }
}

/**
 * Continue a run that was interrupted. Finished stages are not recomputed:
 * every `(runId, candidateId, stage, version)` row already in the store is
 * reused by `runCascade`, and candidates that already produced a diff skip the
 * coder entirely.
 *
 * A run that already finished is continued too, unless it PASSED (spec 4.4): a
 * stored FAIL or JUDGE is reopened, the coders passed in take the rungs above
 * the ones the earlier attempt spent, and the new receipt replaces the old one
 * while listing both attempts' candidates. A stored PASS still short-circuits
 * to `stopReason: 'already-finished'`.
 *
 * @param {object} opts same as runTask, with a runId
 * @returns {Promise<object>} same shape as runTask
 */
export function resumeTask(opts = {}) {
  return runTask({ ...opts, resume: true });
}

export default { runTask, resumeTask, rankCandidates, pickWinner, makeRunRng };

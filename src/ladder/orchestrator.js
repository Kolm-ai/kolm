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
import { repoBrief } from './context/repo-brief.js';
import { baseline as computeBaseline } from './context/baseline.js';
import { runCascade, STAGES } from './stages/index.js';
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
  const perRung = mode === 'direct' ? 1 : Math.max(1, Math.trunc(Number(config.candidates) || 1));
  const repairRounds = Math.max(0, Math.trunc(Number(config.repairTurns) || 0));
  const maxCost = Number(config.maxCostUsd);

  const openWorktrees = new Map(); // candidateId -> wt
  let baseWt = null;
  let stopReason = null;

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
    if (run && run.status === 'done' && run.receipt) {
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
          repairTurns: repairRounds,
          maxCostUsd: maxCost,
        },
      });
    }
    store.emit(runId, resume ? 'run.resume' : 'run.start', {
      runId,
      mode,
      task: { title: task.title ?? null, text: String(task.text).slice(0, 500) },
      repoPath: repoPath ?? null,
      coders: coderRungs.map(refOf),
    });

    /* ------------------------------------------------ 2. base context pack */
    // A resumed run may find its kept worktrees on disk; reuse them.
    baseWt = await driver.create({ repoPath, repoUrl, baseRef, runId, candidateId: 'base', reuse: resume });
    const baseSha = await baseWt.baseSha();
    store.updateRun(runId, { base_sha: baseSha });
    const commands = await detectCommands(baseWt);
    const brief = await repoBrief(store, baseWt, baseSha, { config });
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

    // The base worktree exists only to build the context pack.
    if (config.keepWorktrees !== 'all') {
      await baseWt.destroy().catch(() => {});
      baseWt = null;
    }

    /* -------------------------------------------- 4. the generation ladder */
    const allCandidates = new Map(); // candidateId -> summary row
    let winnerRow = null;
    let judgeUnresolved = false;

    const existing = resume ? store.listCandidates(runId) : [];
    const startRung = resume && existing.length > 0
      ? Math.max(...existing.map((c) => Math.floor(Number(c.idx || 0) / perRung)))
      : 0;

    // One calibration RNG per run. Re-seeding it per batch would replay the
    // same first draw for every batch and sample all-or-nothing.
    const calibrationRng = rng ?? makeRunRng(`${runId}:calibration`);

    /** Record a finished cascade + judge pass and return the candidate rows. */
    const settle = async ({ attempts, coderRef }) => {
      const cascade = await runCascade({
        runId,
        candidates: attempts.map((a) => ({ candidateId: a.candidateId, wt: a.wt, diffText: a.diff })),
        store,
        plan: taskPlan,
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
          generatorRef: coderRef,
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
              plan: taskPlan,
              brief,
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
            plan: taskPlan,
            brief,
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
        store.upsertCandidate({
          runId,
          candidateId: row.candidateId,
          status: row.verdict === 'PASS' ? 'passed' : row.verdict.toLowerCase(),
        });
      }
      return rows;
    };

    ladder:
    for (let rung = startRung; rung < coderRungs.length; rung += 1) {
      if (!taskPlan) break;
      const coder = coderRungs[rung];
      const coderRef = refOf(coder);
      if (checkBudget('coder')) break;

      // -- generate this rung's attempts -----------------------------------
      let attempts = [];
      for (let slot = 0; slot < perRung; slot += 1) {
        const idx = rung * perRung + slot;
        attempts.push({ idx, attempt: 0, candidateId: candidateIdFor(idx, 0), wt: null, diff: null, costUsd: 0 });
      }

      await pool(attempts, perRung, async (att) => {
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
          att.wt = await driver.create({ repoPath, repoUrl, baseRef, runId, candidateId: att.candidateId, reuse: resume });
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
        store.emit(runId, 'candidate.start', { candidateId: att.candidateId, generatorRef: coderRef, rung });
        if (budgetExhausted()) return;
        const coded = await runCoder({
          runId,
          candidateId: att.candidateId,
          wt: att.wt,
          plan: taskPlan,
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
        store.emit(runId, 'candidate.diff', {
          candidateId: att.candidateId,
          changedFiles: coded.changedFiles.length,
          turns: coded.turns,
          finished: coded.finished,
          reason: coded.reason,
        });
      });

      if (checkBudget('cascade')) break;
      let rows = await settle({ attempts, coderRef });
      winnerRow = pickWinner(rows);
      if (winnerRow) break ladder;

      // -- repair rounds ----------------------------------------------------
      for (let round = 1; round <= repairRounds; round += 1) {
        if (checkBudget('repair')) break ladder;
        const failing = rankCandidates(rows.filter((row) => row.verdict !== 'PASS'));
        if (failing.length === 0) break;
        const repaired = [];
        for (const row of failing) {
          const source = attempts.find((a) => a.candidateId === row.candidateId)
            ?? { idx: Number(store.getCandidate(runId, row.candidateId)?.idx ?? 0) };
          repaired.push({
            idx: source.idx,
            attempt: round,
            candidateId: candidateIdFor(source.idx, round),
            wt: row.wt,
            diff: null,
            costUsd: 0,
            feedback: row.judged && row.judged.decision === 'bad'
              ? {
                stage: 'l4-judge',
                reasons: [row.judged.rationale, ...(row.judged.blocking || []).map((b) => `blocking: ${b}`)]
                  .filter(Boolean),
                hits: [],
                failing: [],
              }
              : failingFeedback(row.state),
          });
        }

        await pool(repaired, perRung, async (att) => {
          if (budgetExhausted()) return;
          store.upsertCandidate({
            runId,
            candidateId: att.candidateId,
            idx: att.idx,
            attempt: att.attempt,
            generatorRef: coderRef,
            worktree: { path: att.wt?.path ?? null, branch: att.wt?.branch ?? null },
            status: 'repairing',
          });
          store.emit(runId, 'repair', {
            candidateId: att.candidateId,
            round,
            stage: att.feedback.stage,
            reasons: att.feedback.reasons.slice(0, 6),
          });
          const coded = await runRepair({
            runId,
            candidateId: att.candidateId,
            wt: att.wt,
            plan: taskPlan,
            brief,
            baseline: baselineData,
            commands,
            rubric,
            coder,
            store,
            config,
            signal,
            logger,
            feedback: att.feedback,
          });
          att.diff = coded.diff;
          att.costUsd = coded.costUsd;
        });

        if (checkBudget('cascade')) break ladder;
        attempts = repaired;
        rows = await settle({ attempts, coderRef });
        winnerRow = pickWinner(rows);
        if (winnerRow) break ladder;
      }

      if (rung < coderRungs.length - 1) {
        store.emit(runId, 'escalate', {
          from: coderRef,
          to: refOf(coderRungs[rung + 1]),
          rung: rung + 1,
        });
      }
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
    } else if (judgeUnresolved || candidateList.some((row) => row.verdict === 'JUDGE')) {
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
          store.upsertCandidate({ runId, candidateId, status: 'destroyed' });
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
    };
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
 * @param {object} opts same as runTask, with a runId
 * @returns {Promise<object>} same shape as runTask
 */
export function resumeTask(opts = {}) {
  return runTask({ ...opts, resume: true });
}

export default { runTask, resumeTask, rankCandidates, pickWinner, makeRunRng };

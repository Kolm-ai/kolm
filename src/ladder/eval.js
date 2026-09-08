// kolm code — offline evaluation over stored runs (spec 6.6). Owner D.
//
// Pure over the store: no providers, no worktrees, no network. It answers the
// only question that justifies the cascade — "do the cheap rungs agree with the
// expensive judge, and what did that agreement cost?" — from rows that are
// already on disk, so it can be run long after the runs themselves.

import { calibrationReport } from './stages/l4-judge.js';
import { LadderError } from './errors.js';

const STAGE_DECISIONS = ['good', 'unsure', 'bad', 'skip'];

function ratio(numerator, denominator) {
  return denominator > 0 ? numerator / denominator : null;
}

function verdictOfRun(run) {
  const fromReceipt = run?.receipt?.verdict;
  if (typeof fromReceipt === 'string') return fromReceipt;
  if (run?.status === 'error') return 'ERROR';
  if (run?.winner_candidate) return 'PASS';
  if (run?.status === 'running') return 'RUNNING';
  return 'FAIL';
}

/**
 * Evaluate a set of runs.
 *
 * @param {object} args
 * @param {object} args.store openStore() output
 * @param {string[]} [args.runIds] default: every run in the store
 * @param {string} [args.since] ISO timestamp filter (applies to the funnel and run list)
 * @returns {{funnel:object, runs:Array, cascade_vs_direct:object, judge_agreement:object, per_stage:object}}
 */
export function evalRuns({ store, runIds, since } = {}) {
  if (!store) throw new LadderError('bad-arg', 'evalRuns needs a store');

  let runs = Array.isArray(runIds) && runIds.length > 0
    ? runIds.map((id) => store.getRun(id)).filter(Boolean)
    : store.listRuns({ limit: 1000 });
  if (since) runs = runs.filter((run) => String(run.created_at || '') >= String(since));
  const ids = runs.map((run) => run.id);

  const perStage = {};
  const buckets = {
    cascade: { runs: 0, passes: 0, cost: 0, judgeCalls: 0 },
    direct: { runs: 0, passes: 0, cost: 0, judgeCalls: 0 },
  };
  const agreement = { sampled: 0, agreements: 0, disagreements: [] };
  const runRows = [];

  for (const run of runs) {
    const judgments = store.listJudgments(run.id) || [];
    const verdict = verdictOfRun(run);
    const candidates = (store.listCandidates(run.id) || []).length;
    const cost = Number(store.costSummary(run.id).total_usd) || 0;

    runRows.push({
      run_id: run.id,
      mode: run.mode || 'cascade',
      verdict,
      cost_usd: cost,
      candidates,
      judged: judgments.length,
    });

    const bucket = run.mode === 'direct' ? buckets.direct : buckets.cascade;
    bucket.runs += 1;
    if (verdict === 'PASS') bucket.passes += 1;
    bucket.cost += cost;
    bucket.judgeCalls += judgments.length;

    // Per-candidate judge decision, for the stage-precision comparison.
    const judgeByCandidate = new Map();
    for (const row of judgments) {
      const decision = row.verdict?.decision === 'bad' ? 'bad' : 'good';
      judgeByCandidate.set(row.candidate_id, decision);
    }

    // resultsFor() returns every version; a repaired candidate re-runs stages,
    // so count only the latest version per (candidate, stage) — otherwise a
    // repair inflates the stage counts with results nobody acted on.
    const latest = new Map();
    for (const row of store.resultsFor(run.id) || []) {
      const key = `${row.candidate_id}::${row.stage}`;
      const prev = latest.get(key);
      if (!prev || row.version >= prev.version) latest.set(key, row);
    }

    for (const row of latest.values()) {
      const stage = row.stage;
      const decision = row.verdict?.decision || 'skip';
      const entry = (perStage[stage] ||= {
        good: 0, unsure: 0, bad: 0, skip: 0, compared: 0, agreed: 0, precision_vs_judge: null,
      });
      if (STAGE_DECISIONS.includes(decision)) entry[decision] += 1;
      const judged = judgeByCandidate.get(row.candidate_id);
      if (judged && (decision === 'good' || decision === 'bad')) {
        entry.compared += 1;
        if (decision === judged) entry.agreed += 1;
      }
    }

    const report = calibrationReport({ store, runId: run.id });
    agreement.sampled += report.sampled;
    agreement.agreements += report.agreements;
    for (const d of report.disagreements) agreement.disagreements.push({ runId: run.id, ...d });
  }

  for (const entry of Object.values(perStage)) {
    entry.precision_vs_judge = ratio(entry.agreed, entry.compared);
  }

  const shapeBucket = (b) => ({
    runs: b.runs,
    pass_rate: ratio(b.passes, b.runs),
    avg_cost_usd: ratio(b.cost, b.runs),
    judge_calls_per_run: ratio(b.judgeCalls, b.runs),
  });

  return {
    funnel: store.funnel({ runIds: ids.length > 0 ? ids : undefined, since }),
    runs: runRows,
    cascade_vs_direct: {
      cascade: shapeBucket(buckets.cascade),
      direct: shapeBucket(buckets.direct),
    },
    judge_agreement: {
      sampled: agreement.sampled,
      agreements: agreement.agreements,
      disagreements: agreement.disagreements,
      agreementRate: ratio(agreement.agreements, agreement.sampled),
    },
    per_stage: perStage,
  };
}

/* ------------------------------------------------------------------- table */

function fmt(value, digits = 3) {
  if (value === null || value === undefined) return '-';
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(digits);
  return String(value);
}

function table(headers, rows) {
  const widths = headers.map((h, i) => Math.max(
    String(h).length,
    ...rows.map((r) => String(r[i] ?? '').length),
    1,
  ));
  const line = (cells) => `| ${cells.map((c, i) => String(c ?? '').padEnd(widths[i])).join(' | ')} |`;
  const rule = `+-${widths.map((w) => '-'.repeat(w)).join('-+-')}-+`;
  return [rule, line(headers), rule, ...rows.map(line), rule].join('\n');
}

/**
 * Render `evalRuns()` output as ASCII. No colour, no unicode boxes: this text
 * ends up in terminals, CI logs and pasted issues.
 * @param {object} result evalRuns() output
 * @returns {string}
 */
export function formatEvalTable(result) {
  if (!result || typeof result !== 'object') return '(no evaluation)';
  const out = [];

  out.push('RUNS');
  out.push(table(
    ['run_id', 'mode', 'verdict', 'cost_usd', 'cands', 'judged'],
    (result.runs || []).map((r) => [
      r.run_id, r.mode, r.verdict, fmt(r.cost_usd, 4), fmt(r.candidates), fmt(r.judged),
    ]),
  ));
  out.push('');

  out.push('CASCADE VS DIRECT');
  const cvd = result.cascade_vs_direct || {};
  out.push(table(
    ['mode', 'runs', 'pass_rate', 'avg_cost_usd', 'judge_calls_per_run'],
    ['cascade', 'direct'].map((mode) => {
      const b = cvd[mode] || {};
      return [mode, fmt(b.runs), fmt(b.pass_rate), fmt(b.avg_cost_usd, 4), fmt(b.judge_calls_per_run)];
    }),
  ));
  out.push('');

  out.push('PER STAGE');
  out.push(table(
    ['stage', 'good', 'unsure', 'bad', 'skip', 'precision_vs_judge'],
    Object.entries(result.per_stage || {}).sort(([a], [b]) => a.localeCompare(b)).map(([stage, s]) => [
      stage, fmt(s.good), fmt(s.unsure), fmt(s.bad), fmt(s.skip), fmt(s.precision_vs_judge),
    ]),
  ));
  out.push('');

  const funnel = result.funnel || {};
  out.push('FUNNEL');
  out.push(table(
    ['candidates', 'cleared_cheaply', 'judged', 'hard_fail'],
    [[fmt(funnel.candidates), fmt(funnel.cleared_cheaply), fmt(funnel.judged), fmt(funnel.hard_fail)]],
  ));
  out.push('');

  const ja = result.judge_agreement || {};
  out.push('JUDGE AGREEMENT (calibration sample)');
  out.push(table(
    ['sampled', 'agreements', 'agreement_rate'],
    [[fmt(ja.sampled), fmt(ja.agreements), fmt(ja.agreementRate)]],
  ));
  if ((ja.disagreements || []).length > 0) {
    out.push('');
    out.push('DISAGREEMENTS');
    out.push(table(
      ['run_id', 'candidate', 'cascade', 'judge'],
      ja.disagreements.map((d) => [d.runId ?? '-', d.candidateId, d.cascade, d.judge]),
    ));
  }

  return out.join('\n');
}

export default { evalRuns, formatEvalTable };

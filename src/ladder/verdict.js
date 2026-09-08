// kolm code — cascade verdict assembly + gates + funnel math (spec 5.4). Owner C.
//
// Ladder semantics (publu/ladder core.py `assemble_verdict`): a rung that was
// CONFIDENT decides. Any confident-bad -> FAIL. Otherwise any unsure -> JUDGE.
// Otherwise PASS.
//
// The one deliberate deviation from ladder is the gate: ladder flows only
// confident-GOOD clips up (`good_from`), we flow everything that is not
// confidently bad. Verification rungs are cheap relative to the judge, so an
// unsure candidate keeps accruing evidence instead of jumping the queue.

/** @typedef {'good'|'bad'|'unsure'|'skip'} Decision */

/**
 * Gate for L1..L3: does the previous level's decision let this candidate up?
 * A missing decision (the stage never ran / was skipped) flows up.
 * @param {Decision|undefined|null} prevDecision
 * @returns {boolean}
 */
export function flowsUp(prevDecision) {
  return prevDecision !== 'bad';
}

/**
 * Collapse a candidate's per-stage decisions into one verdict.
 * @param {Record<string, Decision>} stageDecisions
 * @returns {{verdict:'PASS'|'FAIL'|'JUDGE', reasons:string[], unsure:string[], bad:string[], skipped:string[]}}
 */
export function assembleVerdict(stageDecisions) {
  const bad = [];
  const unsure = [];
  const skipped = [];
  const good = [];
  for (const [stage, decision] of Object.entries(stageDecisions || {})) {
    if (decision === 'bad') bad.push(stage);
    else if (decision === 'unsure') unsure.push(stage);
    else if (decision === 'skip') skipped.push(stage);
    else if (decision === 'good') good.push(stage);
  }
  const reasons = [];
  let verdict;
  if (bad.length > 0) {
    verdict = 'FAIL';
    for (const stage of bad) reasons.push(`${stage} is confidently bad`);
  } else if (unsure.length > 0) {
    verdict = 'JUDGE';
    for (const stage of unsure) reasons.push(`${stage} is unsure`);
  } else {
    verdict = 'PASS';
    reasons.push(
      good.length > 0
        ? `every stage that ran is confidently good (${good.join(', ')})`
        : 'no stage produced a decision',
    );
  }
  if (skipped.length > 0) reasons.push(`skipped: ${skipped.join(', ')}`);
  return { verdict, reasons, unsure, bad, skipped };
}

/**
 * Funnel math over stored rows.
 *
 * @param {Array<{runId:string, candidateId:string, stage:string, decision:Decision,
 *                judged?:boolean, judgeVerdict?:string}>} rows
 *   One row per (run, candidate, stage) — the LATEST version of each.
 * @returns {{stages:Record<string,{good:number,unsure:number,bad:number,skip:number}>,
 *            cleared_cheaply:number, clearedCheaply:number, judged:number,
 *            hard_fail:number, hardFail:number, runs:number, candidates:number,
 *            verdicts:{PASS:number, FAIL:number, JUDGE:number}}}
 */
export function funnel(rows) {
  const stages = {};
  /** @type {Map<string, {decisions:Record<string,Decision>, judged:boolean, runId:string}>} */
  const byCandidate = new Map();
  const runs = new Set();

  for (const row of rows || []) {
    if (!row || !row.stage) continue;
    const bucket = (stages[row.stage] ||= { good: 0, unsure: 0, bad: 0, skip: 0 });
    if (row.decision in bucket) bucket[row.decision] += 1;
    const key = `${row.runId}\u0000${row.candidateId}`;
    runs.add(row.runId);
    let entry = byCandidate.get(key);
    if (!entry) {
      entry = { decisions: {}, judged: false, runId: row.runId };
      byCandidate.set(key, entry);
    }
    entry.decisions[row.stage] = row.decision;
    if (row.judged === true || (row.judgeVerdict !== undefined && row.judgeVerdict !== null)) entry.judged = true;
  }

  const verdicts = { PASS: 0, FAIL: 0, JUDGE: 0 };
  let judged = 0;
  let hardFail = 0;
  let cleared = 0;
  for (const entry of byCandidate.values()) {
    const { verdict } = assembleVerdict(entry.decisions);
    verdicts[verdict] += 1;
    if (verdict === 'FAIL') hardFail += 1;
    const neededJudge = verdict === 'JUDGE' || entry.judged;
    if (entry.judged) judged += 1;
    if (!neededJudge) cleared += 1;
  }
  const candidates = byCandidate.size;
  const clearedCheaply = candidates === 0 ? 0 : cleared / candidates;
  return {
    stages,
    cleared_cheaply: clearedCheaply,
    clearedCheaply,
    judged,
    hard_fail: hardFail,
    hardFail,
    runs: runs.size,
    candidates,
    verdicts,
  };
}

export default { assembleVerdict, flowsUp, funnel };

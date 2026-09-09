// kolm code - complexity routing (spec 4.2). Owner ORCH.
//
// The plan carries a `complexity` estimate from the orchestrator model. Routing
// turns that estimate into the three knobs that actually cost money: which rung
// of the coder ladder to start on, how many candidates to draw per rung, and how
// many repair rounds to allow. The point is asymmetry - a one-line fix should not
// pay for three candidates and two repair rounds on two rungs, and a large change
// should not start on the cheapest model only to burn a full rung discovering it
// is out of its depth.
//
// This module is pure: it reads a plan and a config and returns a decision. The
// orchestrator emits the `run.route` event and applies it. That keeps the policy
// testable offline and keeps `runTask` free of branching on complexity.

/** Complexity values the planner may emit, cheapest first. */
export const COMPLEXITY_LEVELS = Object.freeze(['trivial', 'small', 'medium', 'large']);

/** Hard ceiling on candidates per rung, whatever the config asks for. */
export const MAX_ROUTED_CANDIDATES = 4;

/**
 * Normalise a complexity string. Anything unknown (including undefined, which is
 * what every plan written before Plan v2 has) becomes 'medium', which routes to
 * exactly today's behaviour.
 *
 * @param {unknown} value raw `plan.complexity`
 * @returns {'trivial'|'small'|'medium'|'large'} a level from COMPLEXITY_LEVELS
 */
export function normalizeComplexity(value) {
  const s = typeof value === 'string' ? value.trim().toLowerCase() : '';
  return COMPLEXITY_LEVELS.includes(s) ? /** @type {any} */ (s) : 'medium';
}

/**
 * Decide where on the ladder to start and how wide to search.
 *
 * Policy (config.routing === 'fixed' disables all of it and returns today's
 * numbers unchanged):
 *
 *   trivial -> rung 0, 1 candidate, at most 1 repair round
 *   small   -> rung 0, configured candidates
 *   medium  -> rung 0, configured candidates            (today's behaviour)
 *   large   -> rung min(1, last), configured + 1, capped at 4
 *
 * `risk: 'high'` never lowers candidates below 2, so a risky one-liner still gets
 * a second opinion. Repair rounds are only ever LOWERED by routing - a config
 * that asks for 0 repair rounds means 0, even on a trivial task, because raising
 * a budget the operator set is not routing's job.
 *
 * @param {object} args
 * @param {object|null} args.plan the plan object (may be null on a plan failure)
 * @param {object} args.config ladder config; reads `routing`, `candidates`, `repairRounds`
 * @param {Array<unknown>} args.coderRungs the coder ladder, used only for its length
 * @returns {{startRung:number, candidates:number, repairRounds:number, reason:string,
 *   complexity:string, risk:string, routing:string}} the decision
 */
export function routeRun({ plan, config = {}, coderRungs = [] }) {
  const lastRung = Math.max(0, (Array.isArray(coderRungs) ? coderRungs.length : 1) - 1);
  const configured = Math.max(1, Math.trunc(Number(config.candidates)) || 1);
  const configuredRepairs = Math.max(0, Math.trunc(Number(config.repairRounds)) || 0);
  const complexity = normalizeComplexity(plan?.complexity);
  const risk = typeof plan?.risk === 'string' ? plan.risk : 'medium';
  const routing = config.routing === 'fixed' ? 'fixed' : 'auto';

  if (routing === 'fixed') {
    return {
      startRung: 0,
      candidates: configured,
      repairRounds: configuredRepairs,
      reason: 'routing=fixed',
      complexity,
      risk,
      routing,
    };
  }

  let startRung = 0;
  let candidates = configured;
  let repairRounds = configuredRepairs;
  let reason = `complexity=${complexity}`;

  if (complexity === 'trivial') {
    candidates = 1;
    // Cap, never raise: an operator who configured 0 repair rounds gets 0.
    repairRounds = Math.min(configuredRepairs, 1);
    reason = 'trivial: one candidate on the cheapest rung';
  } else if (complexity === 'large') {
    startRung = Math.min(1, lastRung);
    candidates = Math.min(configured + 1, MAX_ROUTED_CANDIDATES);
    reason = startRung > 0
      ? 'large: start one rung up, widen the search'
      : 'large: widen the search (only one rung available)';
  } else {
    reason = `${complexity}: configured ladder`;
  }

  if (risk === 'high' && candidates < 2) {
    candidates = 2;
    reason += '; risk=high floors candidates at 2';
  }

  return { startRung, candidates, repairRounds, reason, complexity, risk, routing };
}

export default { routeRun, normalizeComplexity, COMPLEXITY_LEVELS, MAX_ROUTED_CANDIDATES };

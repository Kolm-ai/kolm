// kolm code — L1 typecheck + lint delta (spec 5.6). Owner C.
//
// Absolute error counts are meaningless in a repo that already has errors, so
// this rung diffs STABLE ERROR IDS against the per-SHA baseline: only ids that
// the patch introduced are banded. Errors the patch removed are recorded too,
// because a candidate that cleans up is evidence for the judge.

import { bandFor, worst } from '../bands.js';
import { flowsUp } from '../verdict.js';
import { runCommand } from '../context/baseline.js';
import { parseTypecheckErrors, parseLintErrors } from '../context/detect.js';

export const id = 'l1-static';
export const level = 1;
export const version = 1;
export const workers = 4;

/** L1..L3 run on anything the level below did not call confidently bad. */
export const gate = flowsUp;

const OUTPUT_KEEP = 4096;
const ID_CAP = 20;

function toolFor(cmd, kind) {
  const c = String(cmd || '').toLowerCase();
  if (kind === 'typecheck') {
    if (/tsc\b/.test(c)) return 'tsc';
    if (/pyright/.test(c)) return 'pyright';
    if (/mypy/.test(c)) return 'mypy';
    if (/go\s+vet/.test(c)) return 'go';
    if (/cargo\s+check/.test(c)) return 'cargo';
    return undefined;
  }
  if (/eslint/.test(c)) return 'eslint';
  if (/ruff/.test(c)) return 'ruff';
  return undefined;
}

/**
 * @param {object} ctx spec 5.6 stage context
 * @returns {Promise<object>} stage result
 */
export async function run({ wt, baseline, commands = {}, bands, config, signal }) {
  const started = Date.now();
  if (!commands.typecheck && !commands.lint) {
    return {
      decision: 'skip',
      reasons: ['no typecheck or lint command detected for this repo'],
      hits: [],
      metrics: {},
      artifacts: {},
      costUsd: 0,
      latencyMs: Date.now() - started,
    };
  }
  const timeoutMs = Number(config?.stageTimeoutMs) > 0 ? Number(config.stageTimeoutMs) : 600000;
  const root = wt?.path;
  const reasons = [];
  const hits = [];
  const metrics = {};
  const artifacts = {};

  const baselineTypecheckTimedOut = baseline?.typecheck?.timedOut === true;
  const baselineLintTimedOut = baseline?.lint?.timedOut === true;

  if (commands.typecheck && baselineTypecheckTimedOut) {
    // The base commit could not typecheck itself inside the budget.
    reasons.push(`typecheck timed out at baseline too (${timeoutMs} ms)`);
    artifacts.baselineTypecheckTimedOut = true;
  } else if (commands.typecheck) {
    const res = await runCommand(wt, commands.typecheck, { timeoutMs, signal });
    const ids = parseTypecheckErrors(res.output, toolFor(commands.typecheck, 'typecheck'), { root });
    const base = new Set(baseline?.typecheck?.errors || []);
    const introduced = ids.filter((e) => !base.has(e));
    const fixed = [...base].filter((e) => !ids.includes(e));
    metrics.type_errors = ids.length;
    metrics.new_type_errors = introduced.length;
    metrics.fixed_type_errors = fixed.length;
    artifacts.typecheckOutput = res.output.slice(0, OUTPUT_KEEP);
    artifacts.newTypeErrors = introduced.slice(0, ID_CAP);
    if (fixed.length > 0) artifacts.fixedTypeErrors = fixed.slice(0, ID_CAP);
    if (res.timedOut) {
      reasons.push(`typecheck timed out after ${timeoutMs} ms`);
      hits.push({ key: 'new_type_errors', value: introduced.length, band: 'unsure' });
    } else {
      const hit = bandFor(bands, id, 'new_type_errors', introduced.length);
      hits.push(hit);
      if (hit.band !== 'good') {
        reasons.push(`${introduced.length} new type error(s): ${introduced.slice(0, 3).join(' | ') || '(unparsed output)'}`);
      }
      if (fixed.length > 0) reasons.push(`${fixed.length} pre-existing type error(s) fixed`);
    }
  }

  if (commands.lint && baselineLintTimedOut) {
    reasons.push(`lint timed out at baseline too (${timeoutMs} ms)`);
    artifacts.baselineLintTimedOut = true;
  } else if (commands.lint) {
    const res = await runCommand(wt, commands.lint, { timeoutMs, signal });
    const ids = parseLintErrors(res.output, toolFor(commands.lint, 'lint'), { root });
    const base = new Set(baseline?.lint?.errors || []);
    const introduced = ids.filter((e) => !base.has(e));
    const fixed = [...base].filter((e) => !ids.includes(e));
    metrics.lint_errors = ids.length;
    metrics.new_lint_errors = introduced.length;
    metrics.fixed_lint_errors = fixed.length;
    artifacts.lintOutput = res.output.slice(0, OUTPUT_KEEP);
    artifacts.newLintErrors = introduced.slice(0, ID_CAP);
    if (res.timedOut) {
      reasons.push(`lint timed out after ${timeoutMs} ms`);
      hits.push({ key: 'new_lint_errors', value: introduced.length, band: 'unsure' });
    } else {
      const hit = bandFor(bands, id, 'new_lint_errors', introduced.length);
      hits.push(hit);
      if (hit.band !== 'good') {
        reasons.push(`${introduced.length} new lint error(s): ${introduced.slice(0, 3).join(' | ') || '(unparsed output)'}`);
      }
    }
  }

  // Nothing measurable ran: both tools timed out at the base commit already.
  if (hits.length === 0 && (baselineTypecheckTimedOut || baselineLintTimedOut)) {
    return {
      decision: 'skip',
      reasons,
      hits,
      metrics,
      artifacts,
      costUsd: 0,
      latencyMs: Date.now() - started,
    };
  }
  const decision = worst(hits);
  if (decision === 'good' && reasons.length === 0) {
    reasons.push('no new type or lint errors versus the baseline');
  }
  return { decision, reasons, hits, metrics, artifacts, costUsd: 0, latencyMs: Date.now() - started };
}

export default { id, level, version, workers, gate, run };

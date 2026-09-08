// kolm code — L2 tests + changed-line coverage (spec 5.6). Owner C.
//
// The rung that actually executes the candidate's code. Like L1 it works on a
// DELTA: a test that was already red at the base sha is not this candidate's
// fault, and a baseline failure that now passes is recorded as `fixed` (that is
// usually the whole point of the task).
//
// Coverage is strictly optional evidence. Parsing coverage output is fragile
// across c8 / nyc / node --experimental-test-coverage, so a coverage parse
// failure NEVER changes the decision: the metric is simply omitted with a note.

import { bandFor, worst } from '../bands.js';
import { flowsUp } from '../verdict.js';
import { runCommand } from '../context/baseline.js';
import { detectRunner, parseTestOutput } from '../context/detect.js';

export const id = 'l2-dynamic';
export const level = 2;
export const version = 1;
export const workers = 2;

/** L1..L3 run on anything the level below did not call confidently bad. */
export const gate = flowsUp;

const OUTPUT_KEEP = 4096;
const NAME_CAP = 25;

/**
 * Pull a 0..1 line-coverage fraction out of c8/nyc json-summary or the
 * `node --test --experimental-test-coverage` table.
 * @param {string} text
 * @returns {{value:number|null, source:string|null}}
 */
export function parseCoverage(text) {
  const src = String(text ?? '');
  // c8 / nyc --reporter=json-summary
  const idx = src.indexOf('{"total"');
  const idx2 = idx >= 0 ? idx : src.indexOf('{ "total"');
  if (idx2 >= 0) {
    try {
      const doc = JSON.parse(src.slice(idx2));
      const pct = doc?.total?.lines?.pct;
      if (Number.isFinite(Number(pct))) return { value: Number(pct) / 100, source: 'json-summary' };
    } catch {
      // fall through
    }
  }
  // node --test --experimental-test-coverage: "# all files | 84.21 | ..."
  const m = /all files\s*\|\s*([\d.]+)/i.exec(src);
  if (m && Number.isFinite(Number(m[1]))) return { value: Number(m[1]) / 100, source: 'node-coverage' };
  return { value: null, source: null };
}

/**
 * @param {object} ctx spec 5.6 stage context
 * @returns {Promise<object>} stage result
 */
export async function run({ wt, baseline, commands = {}, bands, config, signal }) {
  const started = Date.now();
  if (!commands.test) {
    return {
      decision: 'skip',
      reasons: ['no test command detected for this repo'],
      hits: [],
      metrics: {},
      artifacts: {},
      costUsd: 0,
      latencyMs: Date.now() - started,
    };
  }
  const timeoutMs = Number(config?.stageTimeoutMs) > 0 ? Number(config.stageTimeoutMs) : 600000;
  const runner = detectRunner(commands.test);
  const reasons = [];
  const hits = [];
  const artifacts = {};

  const res = await runCommand(wt, commands.test, { timeoutMs, signal });
  const parsed = parseTestOutput(res.output, runner);
  artifacts.testOutput = res.output.slice(0, OUTPUT_KEEP);
  artifacts.runner = runner;

  const baseFailing = new Set(baseline?.tests?.failing || []);
  const nowFailing = new Set(parsed.failing || []);
  const newlyFailing = [...nowFailing].filter((name) => !baseFailing.has(name));
  const fixed = [...baseFailing].filter((name) => !nowFailing.has(name));
  artifacts.newlyFailing = newlyFailing.slice(0, NAME_CAP);
  artifacts.fixed = fixed.slice(0, NAME_CAP);

  const metrics = {
    passed: parsed.passed,
    failed: parsed.failed,
    newly_failing_tests: newlyFailing.length,
    fixed_tests: fixed.length,
    exit_code: res.exitCode,
  };

  if (res.timedOut) {
    reasons.push(`test command timed out after ${timeoutMs} ms`);
    hits.push({ key: 'newly_failing_tests', value: newlyFailing.length, band: 'unsure' });
  } else if (!parsed.parsed && res.exitCode !== 0) {
    reasons.push(`test command exited ${res.exitCode} and its output could not be parsed`);
    hits.push({ key: 'newly_failing_tests', value: 1, band: 'bad' });
  } else {
    const hit = bandFor(bands, id, 'newly_failing_tests', newlyFailing.length);
    hits.push(hit);
    if (hit.band !== 'good') {
      reasons.push(`${newlyFailing.length} newly failing test(s): ${newlyFailing.slice(0, 5).join(', ')}`);
    }
    // A non-zero exit with nothing parsed as failing (and a green baseline) is a
    // build/collection error, not a clean run — unsure, not good.
    const baseExit = baseline?.tests?.exitCode;
    if (res.exitCode !== 0 && newlyFailing.length === 0 && parsed.failed === 0 && (baseExit === 0 || baseExit === null || baseExit === undefined)) {
      reasons.push(`test command exited ${res.exitCode} with no failing test parsed`);
      hits.push({ key: 'newly_failing_tests', value: 0, band: 'unsure' });
    }
  }
  if (fixed.length > 0) reasons.push(`${fixed.length} baseline failure(s) now pass: ${fixed.slice(0, 5).join(', ')}`);

  // --- optional coverage
  if (commands.coverage) {
    const cov = await runCommand(wt, commands.coverage, { timeoutMs, signal });
    const { value, source } = parseCoverage(cov.output);
    artifacts.coverageOutput = cov.output.slice(0, OUTPUT_KEEP);
    if (value === null) {
      reasons.push('coverage output could not be parsed; the coverage metric is omitted');
      artifacts.coverageNote = 'unparsed';
    } else {
      metrics.changed_line_coverage = value;
      artifacts.coverageSource = source;
      const hit = bandFor(bands, id, 'changed_line_coverage', value);
      hits.push(hit);
      if (hit.band !== 'good') reasons.push(`changed_line_coverage = ${value.toFixed(3)} is ${hit.band}`);
    }
  }

  const decision = worst(hits);
  if (decision === 'good' && reasons.length === 0) {
    reasons.push(`no newly failing tests (${parsed.passed} passed, ${parsed.failed} failed)`);
  }
  return { decision, reasons, hits, metrics, artifacts, costUsd: 0, latencyMs: Date.now() - started };
}

export default { id, level, version, workers, gate, run };

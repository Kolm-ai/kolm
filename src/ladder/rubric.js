// kolm code — rubric loader / renderer / scorer (spec 5.3). Owner C.
//
// rubric.json is the single source of truth for L3 (cheap reviewer) and L4
// (judge). Both graders get the SAME item list, so their scores are directly
// comparable and eval.js can measure the cheap rung against the judge.

import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { LadderError } from './errors.js';

const require = createRequire(import.meta.url);
const HERE = path.dirname(fileURLToPath(import.meta.url));

/** Path of the bundled rubric. */
export const BUNDLED_RUBRIC_PATH = path.join(HERE, 'rubric.json');

/** Items that must exist and must be marked blocking (spec 5.3). */
export const BLOCKING_ITEM_IDS = Object.freeze(['correctness', 'no_secrets', 'spec_compliance', 'no_placeholders']);

const WEIGHT_TOLERANCE = 1e-9;

/**
 * Validate a rubric object (throws LadderError('bad-rubric')).
 * @param {object} rubric
 * @param {string} source
 * @returns {object} the same object
 */
export function validateRubric(rubric, source = '<inline>') {
  if (!rubric || typeof rubric !== 'object' || !Array.isArray(rubric.items)) {
    throw new LadderError('bad-rubric', `rubric must be an object with an items array (${source})`, { source });
  }
  if (rubric.items.length === 0) {
    throw new LadderError('bad-rubric', `rubric has no items (${source})`, { source });
  }
  const seen = new Set();
  let sum = 0;
  for (const item of rubric.items) {
    if (!item || typeof item !== 'object') {
      throw new LadderError('bad-rubric', `rubric item must be an object (${source})`, { source });
    }
    for (const field of ['id', 'title', 'question', 'evidence']) {
      if (typeof item[field] !== 'string' || item[field].length === 0) {
        throw new LadderError('bad-rubric', `rubric item ${item.id || '?'} is missing ${field} (${source})`, { source, id: item.id });
      }
    }
    if (seen.has(item.id)) {
      throw new LadderError('bad-rubric', `duplicate rubric item id ${item.id} (${source})`, { source, id: item.id });
    }
    seen.add(item.id);
    if (typeof item.weight !== 'number' || !Number.isFinite(item.weight) || item.weight < 0) {
      throw new LadderError('bad-rubric', `rubric item ${item.id} needs a non-negative numeric weight (${source})`, { source, id: item.id });
    }
    if (typeof item.blocking !== 'boolean') {
      throw new LadderError('bad-rubric', `rubric item ${item.id}.blocking must be a boolean (${source})`, { source, id: item.id });
    }
    sum += item.weight;
  }
  if (Math.abs(sum - 1) > WEIGHT_TOLERANCE) {
    throw new LadderError('bad-rubric', `rubric weights sum to ${sum}, expected 1 (${source})`, { source, sum });
  }
  for (const id of BLOCKING_ITEM_IDS) {
    const item = rubric.items.find((it) => it.id === id);
    if (!item) {
      throw new LadderError('bad-rubric', `rubric is missing required item ${id} (${source})`, { source, id });
    }
    if (item.blocking !== true) {
      throw new LadderError('bad-rubric', `rubric item ${id} must be blocking (${source})`, { source, id });
    }
  }
  return rubric;
}

/**
 * Load the rubric from `path`, or $KOLM_CODE_RUBRIC_PATH, or the bundled file.
 * @param {string} [rubricPath]
 * @returns {object}
 */
export function loadRubric(rubricPath) {
  const target = rubricPath || process.env.KOLM_CODE_RUBRIC_PATH || BUNDLED_RUBRIC_PATH;
  let raw;
  try {
    raw = require(path.resolve(target));
  } catch (err) {
    throw new LadderError('bad-rubric', `cannot read rubric file ${target}: ${err.message}`, { path: target });
  }
  const rubric = JSON.parse(JSON.stringify(raw));
  return validateRubric(rubric, target);
}

/**
 * Render the rubric as grader instructions. Same text for L3 and L4.
 * @param {object} rubric
 * @returns {string}
 */
export function rubricToPrompt(rubric) {
  validateRubric(rubric, 'rubricToPrompt');
  const lines = [];
  lines.push(`RUBRIC ${rubric.id || 'kolm-code-patch'} — score every item on ${rubric.scale || '0..1'}.`);
  lines.push('For each item return {"id", "score", "evidence"}; evidence must quote the diff, not restate the question.');
  lines.push('');
  for (const item of rubric.items) {
    lines.push(`- ${item.id} (weight ${item.weight}${item.blocking ? ', BLOCKING' : ''}) — ${item.title}`);
    lines.push(`  question: ${item.question}`);
    lines.push(`  look at: ${item.evidence}`);
  }
  lines.push('');
  lines.push('A BLOCKING item scored below 0.3 fails the patch outright; list its id in "blocking".');
  lines.push('"overall" is the weighted mean of the item scores. "confidence" is how sure you are of that overall, 0..1.');
  return lines.join('\n');
}

/**
 * Weighted overall from per-item scores. Unknown ids are ignored; missing items
 * do not drag the score down (the weights of the items actually scored are
 * renormalized), so a reviewer that skips an item is 'unsure', not 'bad'.
 * @param {object} rubric
 * @param {Array<{id:string, score:number}>} items
 * @returns {number} 0..1 (0 when nothing scored)
 */
export function scoreRubric(rubric, items) {
  const byId = new Map();
  for (const item of rubric?.items || []) byId.set(item.id, item);
  let weighted = 0;
  let total = 0;
  for (const scored of items || []) {
    const item = byId.get(scored?.id);
    if (!item) continue;
    const raw = Number(scored.score);
    if (!Number.isFinite(raw)) continue;
    const score = Math.min(1, Math.max(0, raw));
    weighted += score * item.weight;
    total += item.weight;
  }
  if (total <= 0) return 0;
  return weighted / total;
}

/**
 * Blocking item ids scored below `threshold`.
 * @param {object} rubric
 * @param {Array<{id:string, score:number}>} items
 * @param {number} [threshold]
 * @returns {string[]}
 */
export function blockingFailures(rubric, items, threshold = 0.3) {
  const blocking = new Set((rubric?.items || []).filter((it) => it.blocking).map((it) => it.id));
  const out = [];
  for (const scored of items || []) {
    if (!blocking.has(scored?.id)) continue;
    const raw = Number(scored.score);
    if (Number.isFinite(raw) && raw < threshold) out.push(scored.id);
  }
  return out;
}

export default { loadRubric, rubricToPrompt, scoreRubric, blockingFailures, validateRubric, BUNDLED_RUBRIC_PATH };

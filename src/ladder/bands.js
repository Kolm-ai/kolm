// kolm code — confidence bands (spec 5.2). Owner C.
//
// One source of truth for "is this metric confidently bad, confidently good, or
// unsure (defer to the judge)". Two thresholds, ladder's orientation flag:
//
//   low_is_bad === false (a HIGH value is the bad one, e.g. patch_lines):
//       value > hi   -> 'bad'
//       value <= lo  -> 'good'
//       else         -> 'unsure'
//   low_is_bad === true (a LOW value is the bad one, e.g. coverage):
//       value < lo   -> 'bad'
//       value >= hi  -> 'good'
//       else         -> 'unsure'
//
// DELIBERATE DEVIATION from ladder core.py, which is exclusive on BOTH edges
// (`bad if v > hi else good if v < lo else unsure`). The good-side threshold is
// inclusive here, because ladder bands continuous perceptual metrics where an
// exact tie is measure-zero, while ours band integer counts whose *good* value
// is exactly the threshold. Under ladder's rule `new_type_errors {lo:0, hi:3}`
// would call ZERO new type errors 'unsure' (good needs v < 0, impossible), so
// every candidate would escalate to the expensive judge and the cascade would
// never return PASS. Inclusivity also makes spec 5.2's degenerate note ("lo ==
// hi means no unsure band for that key; value <= lo is good, above is bad") a
// consequence of the general rule instead of a special case. The bad-side
// threshold stays exclusive, exactly as ladder has it.

import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { LadderError } from './errors.js';

const require = createRequire(import.meta.url);
const HERE = path.dirname(fileURLToPath(import.meta.url));

/** Path of the bundled defaults. */
export const BUNDLED_BANDS_PATH = path.join(HERE, 'bands.json');

/**
 * Decide one metric against one band spec.
 * @param {number} value
 * @param {{lo:number, hi:number, low_is_bad?:boolean}} spec
 * @returns {'bad'|'good'|'unsure'}
 */
export function band(value, spec) {
  if (!spec || typeof spec !== 'object') {
    throw new LadderError('bad-bands', 'band() needs a {lo, hi, low_is_bad} spec');
  }
  const lo = Number(spec.lo);
  const hi = Number(spec.hi);
  const lowIsBad = spec.low_is_bad === true;
  const v = Number(value);
  if (!Number.isFinite(v)) {
    throw new LadderError('bad-bands', `band() got a non-finite value: ${String(value)}`);
  }
  // lo === hi needs no special case: it collapses the unsure region to nothing,
  // which is exactly spec 5.2's degenerate rule.
  if (lowIsBad) return v < lo ? 'bad' : v >= hi ? 'good' : 'unsure';
  return v > hi ? 'bad' : v <= lo ? 'good' : 'unsure';
}

/**
 * Look a key up in a loaded bands object and decide it.
 * @param {object} bands   loaded bands.json shape
 * @param {string} stageId e.g. 'l0-meta'
 * @param {string} key     e.g. 'patch_lines'
 * @param {number} value
 * @returns {{key:string, value:number, band:'bad'|'good'|'unsure'}}
 */
export function bandFor(bands, stageId, key, value) {
  const forStage = bands && bands[stageId];
  const spec = forStage && forStage[key];
  if (!spec) {
    throw new LadderError('bad-bands', `no band configured for ${stageId}.${key}`, { stageId, key });
  }
  return { key, value: Number(value), band: band(value, spec) };
}

/**
 * ladder's `_worst`: bad beats unsure beats good. 'skip' and unknown entries are
 * ignored; an empty list is 'good'.
 * @param {Array<string|{band?:string, decision?:string}>} decisions
 * @returns {'bad'|'good'|'unsure'}
 */
export function worst(decisions) {
  let sawUnsure = false;
  for (const d of decisions || []) {
    const v = typeof d === 'string' ? d : d && (d.band || d.decision);
    if (v === 'bad') return 'bad';
    if (v === 'unsure') sawUnsure = true;
  }
  return sawUnsure ? 'unsure' : 'good';
}

/**
 * Validate a bands object in place (throws LadderError('bad-bands')).
 * @param {object} bands
 * @param {string} source for the error message
 * @returns {object} the same object
 */
export function validateBands(bands, source = '<inline>') {
  if (!bands || typeof bands !== 'object' || Array.isArray(bands)) {
    throw new LadderError('bad-bands', `bands must be an object (${source})`, { source });
  }
  for (const [stageId, keys] of Object.entries(bands)) {
    if (!keys || typeof keys !== 'object' || Array.isArray(keys)) {
      throw new LadderError('bad-bands', `bands.${stageId} must be an object (${source})`, { source, stageId });
    }
    for (const [key, spec] of Object.entries(keys)) {
      const where = `${stageId}.${key}`;
      if (!spec || typeof spec !== 'object' || Array.isArray(spec)) {
        throw new LadderError('bad-bands', `bands.${where} must be an object (${source})`, { source, stageId, key });
      }
      if (typeof spec.lo !== 'number' || !Number.isFinite(spec.lo)) {
        throw new LadderError('bad-bands', `bands.${where}.lo must be a finite number (${source})`, { source, stageId, key });
      }
      if (typeof spec.hi !== 'number' || !Number.isFinite(spec.hi)) {
        throw new LadderError('bad-bands', `bands.${where}.hi must be a finite number (${source})`, { source, stageId, key });
      }
      if (spec.lo > spec.hi) {
        throw new LadderError('bad-bands', `bands.${where}: lo (${spec.lo}) must be <= hi (${spec.hi}) (${source})`, { source, stageId, key });
      }
      if (typeof spec.low_is_bad !== 'boolean') {
        throw new LadderError('bad-bands', `bands.${where}.low_is_bad must be a boolean (${source})`, { source, stageId, key });
      }
    }
  }
  return bands;
}

/**
 * Load bands from `path`, or $KOLM_CODE_BANDS_PATH, or the bundled bands.json.
 * @param {string} [bandsPath]
 * @returns {object}
 */
export function loadBands(bandsPath) {
  const target = bandsPath || process.env.KOLM_CODE_BANDS_PATH || BUNDLED_BANDS_PATH;
  let raw;
  try {
    // require() gives us a synchronous JSON read that works on every supported
    // Node without the import-attributes syntax churn.
    raw = require(path.resolve(target));
  } catch (err) {
    throw new LadderError('bad-bands', `cannot read bands file ${target}: ${err.message}`, { path: target });
  }
  // require() caches and freezes nothing; clone so callers cannot poison the cache.
  const bands = JSON.parse(JSON.stringify(raw));
  return validateBands(bands, target);
}

export default { band, bandFor, worst, loadBands, validateBands, BUNDLED_BANDS_PATH };

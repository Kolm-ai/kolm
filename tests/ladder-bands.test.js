// Owner C — confidence band semantics (spec 5.2), matched against ladder core.py.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { band, bandFor, worst, loadBands, validateBands, BUNDLED_BANDS_PATH } from '../src/ladder/bands.js';

const HIGH_IS_BAD = { lo: 600, hi: 2500, low_is_bad: false };
const LOW_IS_BAD = { lo: 0.4, hi: 0.8, low_is_bad: true };

test('band with low_is_bad false calls high values bad and low values good', () => {
  assert.equal(band(0, HIGH_IS_BAD), 'good');
  assert.equal(band(599, HIGH_IS_BAD), 'good');
  assert.equal(band(1500, HIGH_IS_BAD), 'unsure');
  assert.equal(band(2501, HIGH_IS_BAD), 'bad');
  assert.equal(band(99999, HIGH_IS_BAD), 'bad');
});

test('band is inclusive on the good edge and exclusive on the bad edge', () => {
  // Deliberate deviation from ladder core.py, which is exclusive on both edges:
  // our metrics are counts whose good value IS the threshold (zero new errors),
  // so lo/hi on the good side must read as good, never unsure.
  assert.equal(band(600, HIGH_IS_BAD), 'good');
  assert.equal(band(601, HIGH_IS_BAD), 'unsure');
  assert.equal(band(2500, HIGH_IS_BAD), 'unsure');
  assert.equal(band(0.8, LOW_IS_BAD), 'good');
  assert.equal(band(0.79, LOW_IS_BAD), 'unsure');
  assert.equal(band(0.4, LOW_IS_BAD), 'unsure');
});

test('band calls a clean count good so the cascade can settle cheaply', () => {
  // With exclusive thresholds `lo: 0` could never be reached from below and
  // every clean candidate would escalate to the judge.
  const typeErrors = { lo: 0, hi: 3, low_is_bad: false };
  assert.equal(band(0, typeErrors), 'good');
  assert.equal(band(1, typeErrors), 'unsure');
  assert.equal(band(3, typeErrors), 'unsure');
  assert.equal(band(4, typeErrors), 'bad');
});

test('band with low_is_bad true inverts the orientation', () => {
  assert.equal(band(0, LOW_IS_BAD), 'bad');
  assert.equal(band(0.39, LOW_IS_BAD), 'bad');
  assert.equal(band(0.6, LOW_IS_BAD), 'unsure');
  assert.equal(band(0.81, LOW_IS_BAD), 'good');
  assert.equal(band(1, LOW_IS_BAD), 'good');
});

test('band collapses the unsure region when lo equals hi', () => {
  const zeroTolerance = { lo: 0, hi: 0, low_is_bad: false };
  assert.equal(band(0, zeroTolerance), 'good');
  assert.equal(band(1, zeroTolerance), 'bad');
  assert.equal(band(50, zeroTolerance), 'bad');
  const mustReach = { lo: 1, hi: 1, low_is_bad: true };
  assert.equal(band(1, mustReach), 'good');
  assert.equal(band(1.5, mustReach), 'good');
  assert.equal(band(0.99, mustReach), 'bad');
});

test('band rejects a missing spec or a non-finite value', () => {
  assert.throws(() => band(1, null), (err) => err.code === 'bad-bands');
  assert.throws(() => band(Number.NaN, HIGH_IS_BAD), (err) => err.code === 'bad-bands');
  assert.throws(() => band('nope', HIGH_IS_BAD), (err) => err.code === 'bad-bands');
});

test('bandFor looks a metric up by stage and key', () => {
  const bands = loadBands();
  assert.deepEqual(bandFor(bands, 'l0-meta', 'patch_lines', 12), { key: 'patch_lines', value: 12, band: 'good' });
  assert.equal(bandFor(bands, 'l2-dynamic', 'newly_failing_tests', 0).band, 'good');
  assert.equal(bandFor(bands, 'l2-dynamic', 'newly_failing_tests', 2).band, 'bad');
  assert.throws(() => bandFor(bands, 'l0-meta', 'not_a_metric', 1), (err) => err.code === 'bad-bands');
  assert.throws(() => bandFor(bands, 'no-such-stage', 'patch_lines', 1), (err) => err.code === 'bad-bands');
});

test('worst prefers bad over unsure over good', () => {
  assert.equal(worst([]), 'good');
  assert.equal(worst(['good', 'good']), 'good');
  assert.equal(worst(['good', 'unsure']), 'unsure');
  assert.equal(worst(['unsure', 'bad', 'good']), 'bad');
  assert.equal(worst(['skip', 'skip']), 'good');
  assert.equal(worst([{ band: 'unsure' }, { band: 'good' }]), 'unsure');
  assert.equal(worst([{ decision: 'bad' }]), 'bad');
  assert.equal(worst(null), 'good');
});

test('validateBands rejects malformed band specs', () => {
  assert.throws(() => validateBands(null), (err) => err.code === 'bad-bands');
  assert.throws(() => validateBands([]), (err) => err.code === 'bad-bands');
  assert.throws(() => validateBands({ stage: 7 }), (err) => err.code === 'bad-bands');
  assert.throws(
    () => validateBands({ stage: { key: { lo: 5, hi: 1, low_is_bad: false } } }),
    (err) => err.code === 'bad-bands' && /lo \(5\) must be <= hi/.test(err.message),
  );
  assert.throws(
    () => validateBands({ stage: { key: { lo: 1, hi: 2 } } }),
    (err) => err.code === 'bad-bands' && /low_is_bad/.test(err.message),
  );
  assert.throws(
    () => validateBands({ stage: { key: { lo: 'x', hi: 2, low_is_bad: true } } }),
    (err) => err.code === 'bad-bands' && /lo must be a finite number/.test(err.message),
  );
});

test('loadBands reads the bundled defaults from spec 5.2', () => {
  const bands = loadBands();
  assert.deepEqual(Object.keys(bands).sort(), ['l0-meta', 'l1-static', 'l2-dynamic', 'l3-semantic']);
  assert.deepEqual(bands['l0-meta'].patch_lines, { lo: 600, hi: 2500, low_is_bad: false });
  assert.deepEqual(bands['l0-meta'].files_touched, { lo: 15, hi: 40, low_is_bad: false });
  assert.deepEqual(bands['l1-static'].new_type_errors, { lo: 0, hi: 3, low_is_bad: false });
  assert.deepEqual(bands['l1-static'].new_lint_errors, { lo: 0, hi: 10, low_is_bad: false });
  assert.deepEqual(bands['l2-dynamic'].newly_failing_tests, { lo: 0, hi: 0, low_is_bad: false });
  assert.deepEqual(bands['l2-dynamic'].changed_line_coverage, { lo: 0.4, hi: 0.8, low_is_bad: true });
  assert.deepEqual(bands['l3-semantic'].rubric_score, { lo: 0.55, hi: 0.85, low_is_bad: true });
  assert.deepEqual(bands['l3-semantic'].reviewer_confidence, { lo: 0.4, hi: 0.75, low_is_bad: true });
  assert.ok(fs.existsSync(BUNDLED_BANDS_PATH));
});

test('loadBands returns a fresh object each call so callers cannot poison it', () => {
  const first = loadBands();
  first['l0-meta'].patch_lines.hi = 1;
  const second = loadBands();
  assert.equal(second['l0-meta'].patch_lines.hi, 2500);
});

test('loadBands honors an override path and reports unreadable files', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kolm-bands-'));
  const override = path.join(dir, 'bands.json');
  fs.writeFileSync(override, JSON.stringify({ 'l0-meta': { patch_lines: { lo: 1, hi: 2, low_is_bad: false } } }));
  try {
    const bands = loadBands(override);
    assert.equal(bands['l0-meta'].patch_lines.hi, 2);
    const previous = process.env.KOLM_CODE_BANDS_PATH;
    process.env.KOLM_CODE_BANDS_PATH = override;
    try {
      assert.equal(loadBands()['l0-meta'].patch_lines.lo, 1);
    } finally {
      if (previous === undefined) delete process.env.KOLM_CODE_BANDS_PATH;
      else process.env.KOLM_CODE_BANDS_PATH = previous;
    }
    assert.throws(() => loadBands(path.join(dir, 'missing.json')), (err) => err.code === 'bad-bands');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

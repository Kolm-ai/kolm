// Owner JUDGE - per-criterion acceptance attestation at L3 and L4.
//
// Regression source: live run r_20260909_18fc043b. The plan listed "the regex
// literal in parseDuration matches the character class [dhms]" and
// "parseDuration('1d') returns 86400". The coder added the unit to the table
// but left the regex at [hms], no existing test exercised 'd', and the judge
// still returned score 0.9, blocking [], decision good. A stated criterion was
// never checked at all. These tests pin the fix: both graders attest to every
// criterion one by one, an unattested-as-met criterion blocks, and a grader
// that answers without the optional field behaves exactly as it did before.
//
// Offline: scripted fake provider only, no worktree, no network, no live model.

import test from 'node:test';
import assert from 'node:assert/strict';

import {
  runJudge,
  JUDGMENT_SCHEMA,
  SPEC_COMPLIANCE_ID,
  ACCEPTANCE_FAIL_SCORE,
} from '../src/ladder/stages/l4-judge.js';
import {
  run as l3Run,
  REVIEW_SCHEMA,
  acceptanceList,
  acceptanceBlock,
  normalizeAcceptance,
  unmetIndices,
  unmetPrefix,
} from '../src/ladder/stages/l3-semantic.js';
import { createFakeProvider } from '../src/ladder/providers/fake.js';
import { loadRubric } from '../src/ladder/rubric.js';
import { loadBands } from '../src/ladder/bands.js';

const RUBRIC = loadRubric();
const RUBRIC_IDS = RUBRIC.items.map((item) => item.id);
const BANDS = loadBands();
const BRIEF = { root: '/repo/demo', files: [] };

// --------------------------------------------------------------- fixtures

const C1 = 'The regex literal in parseDuration matches the character class [dhms]';
const C2 = "parseDuration('1d') returns 86400";

/** The plan whose second criterion the live run silently skipped. */
const PLAN = {
  title: 'parseDuration accepts days',
  summary: 'Teach parseDuration the d unit.',
  acceptance: [C1, C2],
  files_expected: ['src/duration.js'],
  test_command: 'node --test',
};

/** The same task with no acceptance list at all. */
const PLAN_NO_ACCEPTANCE = {
  title: PLAN.title,
  summary: PLAN.summary,
  files_expected: PLAN.files_expected,
  test_command: PLAN.test_command,
};

const DIFF = [
  'diff --git a/src/duration.js b/src/duration.js',
  'index 1111111..2222222 100644',
  '--- a/src/duration.js',
  '+++ b/src/duration.js',
  '@@ -1,6 +1,7 @@',
  "-const UNITS = { s: 1, m: 60, h: 3600 };",
  "+const UNITS = { s: 1, m: 60, h: 3600, d: 86400 };",
  ' export function parseDuration(text) {',
  '   const match = /^(\\d+)([hms])$/.exec(text);',
  '   if (!match) return null;',
  '   return Number(match[1]) * UNITS[match[2]];',
  ' }',
  '',
].join('\n');

/**
 * A judge whose reply is a PASS on every rubric item, optionally carrying an
 * `acceptance` array. `acceptance === null` omits the field entirely, which is
 * how every pre-existing fixture in the suite answers.
 * @param {Array|null} acceptance
 * @returns {object} fake provider
 */
function judgeProvider(acceptance = null) {
  return createFakeProvider({
    ref: 'fake:judge',
    script: () => {
      const json = {
        items: RUBRIC_IDS.map((id) => ({ id, score: 0.95, evidence: 'read the whole diff' })),
        overall: 0.95,
        blocking: [],
        rationale: 'The diff adds the unit table entry.',
        decision: 'PASS',
      };
      if (acceptance !== null) json.acceptance = acceptance;
      return { json, costUsd: 0.01 };
    },
  });
}

/**
 * A reviewer whose reply scores every rubric item at `score`, optionally
 * carrying an `acceptance` array.
 * @param {Array|null} acceptance
 * @param {number} [score]
 * @param {number} [confidence]
 * @returns {object} fake provider
 */
function reviewerProvider(acceptance = null, score = 0.95, confidence = 0.9) {
  return createFakeProvider({
    ref: 'fake:reviewer',
    script: () => {
      const json = {
        items: RUBRIC_IDS.map((id) => ({ id, score, evidence: 'the diff implements the criteria' })),
        overall: score,
        confidence,
        blocking: [],
      };
      if (acceptance !== null) json.acceptance = acceptance;
      return { json, costUsd: 0.001 };
    },
  });
}

/** runJudge with the fixtures above and no store. */
function judge(provider, plan = PLAN) {
  return runJudge({ diffText: DIFF, plan, brief: BRIEF, judge: provider, rubric: RUBRIC });
}

/** l3Run with the fixtures above and no store. */
function review(provider, plan = PLAN) {
  return l3Run({
    diffText: DIFF,
    plan,
    brief: BRIEF,
    providers: { reviewer: provider },
    rubric: RUBRIC,
    bands: BANDS,
    config: {},
  });
}

/** The user prompt the provider actually saw. */
function promptOf(provider) {
  return provider.calls[0].messages[0].content;
}

/** The scored rubric item by id. */
function itemOf(items, id) {
  return items.find((it) => it.id === id) || null;
}

// ------------------------------------------------------------- the schemas

test('both grader schemas carry acceptance as an optional property', () => {
  for (const [label, schema] of [['judge', JUDGMENT_SCHEMA], ['reviewer', REVIEW_SCHEMA]]) {
    const props = schema.schema.properties;
    assert.ok(props.acceptance, `${label} schema exposes acceptance`);
    assert.equal(props.acceptance.type, 'array');
    assert.deepEqual(props.acceptance.items.required, ['index', 'met', 'evidence']);
    assert.equal(props.acceptance.items.properties.index.type, 'integer');
    assert.equal(props.acceptance.items.properties.met.type, 'boolean');
    assert.equal(props.acceptance.items.properties.evidence.type, 'string');
    assert.ok(
      !schema.schema.required.includes('acceptance'),
      `${label} must not require acceptance: a plan may have no criteria`,
    );
    assert.equal(schema.schema.additionalProperties, false, `${label} stays closed`);
  }
});

// ------------------------------------------------------------ pure helpers

test('acceptanceList tolerates every plan shape that reaches a grader', () => {
  assert.deepEqual(acceptanceList(PLAN), [C1, C2]);
  assert.deepEqual(acceptanceList(PLAN_NO_ACCEPTANCE), [], 'missing field');
  assert.deepEqual(acceptanceList({ acceptance: [] }), [], 'empty array');
  assert.deepEqual(acceptanceList({ acceptance: 'do the thing' }), [], 'not an array');
  assert.deepEqual(acceptanceList('a free-text plan'), [], 'plan passed as a string');
  assert.deepEqual(acceptanceList(null), []);
  assert.deepEqual(acceptanceList(undefined), []);
  assert.deepEqual(
    acceptanceList({ acceptance: ['  keeps   one  space ', '', 42, null, 'second'] }),
    ['keeps one space', 'second'],
    'blank and non-string entries are dropped, whitespace is collapsed',
  );
  assert.equal(acceptanceBlock([]), null, 'no criteria means no section');
  assert.equal(
    acceptanceBlock(['a', 'b']),
    'ACCEPTANCE CRITERIA (from the plan; attest to each one separately)\n1. a\n2. b',
  );
  // A whole-task plan adds no scope note; a subtask plan tells the grader that
  // work owned by another subtask does not make a criterion unmet.
  assert.equal(acceptanceBlock(['a'], { title: 'whole' }), acceptanceBlock(['a']));
  const scoped = acceptanceBlock(['a'], { subtask_id: 'st-csv', files_expected: ['src/csv.js'] });
  assert.match(
    scoped,
    /^ACCEPTANCE CRITERIA .*\n1\. a\nThis candidate is one subtask of a split plan \(its files: src\/csv\.js\)\. Judge/,
  );
  assert.match(scoped, /does not make a criterion unmet\.$/);
  assert.match(acceptanceBlock(['a'], { subtask_id: 'x' }), /split plan\. Judge/);
});

test('normalizeAcceptance drops junk and never guesses met', () => {
  const criteria = [C1, C2];
  assert.deepEqual(normalizeAcceptance(criteria, undefined), [], 'absent field');
  assert.deepEqual(normalizeAcceptance(criteria, 'nope'), [], 'not an array');
  assert.deepEqual(normalizeAcceptance([], [{ index: 1, met: true, evidence: 'x' }]), [], 'no criteria');

  const out = normalizeAcceptance(criteria, [
    { index: 2, met: 'true', evidence: 'a string is not an attestation' },
    { index: 1, met: true, evidence: 'quoted the diff' },
  ]);
  assert.deepEqual(out.map((a) => a.index), [1, 2], 'sorted by index');
  assert.equal(out[0].met, true);
  assert.equal(out[1].met, false, 'only a literal true counts as met');
  assert.equal(out[0].text, C1, 'the text comes from the plan, not the model');

  const capped = normalizeAcceptance(criteria, [{ index: 1, met: true, evidence: 'e'.repeat(500) }]);
  assert.equal(capped[0].evidence.length, 300, 'evidence is capped at 300');
});

test('unmetIndices and unmetPrefix describe what was not shown', () => {
  const acceptance = normalizeAcceptance([C1, C2], [
    { index: 1, met: true, evidence: 'ok' },
    { index: 2, met: false, evidence: 'no test exercises 1d' },
  ]);
  assert.deepEqual(unmetIndices(acceptance), [2]);
  assert.equal(unmetPrefix(acceptance), `Unmet acceptance: #2 ${C2}`);
  assert.equal(unmetPrefix([]), '', 'nothing unmet means no prefix');
  const long = normalizeAcceptance(['x'.repeat(400)], [{ index: 1, met: false, evidence: '' }]);
  assert.ok(unmetPrefix(long).length <= 600, 'the prefix is capped at 600 characters');
});

// ------------------------------------------------------------- L4, the judge

test('judge: every criterion met keeps the verdict good with no unmet indices', async () => {
  const provider = judgeProvider([
    { index: 1, met: true, evidence: '+  const match = /^(d+)([dhms])$/.exec(text);' },
    { index: 2, met: true, evidence: '+ ok 3 - parseDuration(1d) returns 86400' },
  ]);
  const out = await judge(provider);
  assert.equal(out.decision, 'good');
  assert.deepEqual(out.unmet, []);
  assert.deepEqual(out.blocking, []);
  assert.deepEqual(out.acceptance.map((a) => [a.index, a.met]), [[1, true], [2, true]]);
  assert.deepEqual(out.acceptance.map((a) => a.text), [C1, C2]);
  assert.equal(out.rationale, 'The diff adds the unit table entry.', 'no prefix when nothing is unmet');
  assert.ok(Math.abs(out.score - 0.95) < 1e-9, 'the rubric score is untouched');
});

test('judge: one unmet criterion blocks the patch and names it first', async () => {
  const provider = judgeProvider([
    { index: 1, met: true, evidence: '+  d: 86400,' },
    { index: 2, met: false, evidence: 'the regex still reads ([hms]); no line shows 1d parsing' },
  ]);
  const out = await judge(provider);

  assert.equal(out.decision, 'bad', 'an unmet criterion is a rejection, not a low score');
  assert.deepEqual(out.unmet, [2]);
  assert.ok(out.blocking.includes(SPEC_COMPLIANCE_ID), 'spec_compliance is the blocking rubric id');
  assert.ok(
    out.rationale.startsWith(`Unmet acceptance: #2 ${C2}`),
    `rationale must lead with the unmet criterion, got: ${out.rationale}`,
  );
  assert.ok(out.rationale.includes('The diff adds the unit table entry.'), 'the judge prose survives');

  const spec = itemOf(out.items, SPEC_COMPLIANCE_ID);
  assert.ok(spec, 'spec_compliance is scored');
  assert.ok(spec.score <= ACCEPTANCE_FAIL_SCORE, `spec_compliance capped, got ${spec.score}`);
  assert.ok(out.score < 0.95, 'the capped item pulls the weighted overall down');
  assert.equal(itemOf(out.items, 'correctness').score, 0.95, 'other items are untouched');
});

test('judge: the cap applies even when the judge omitted spec_compliance', async () => {
  const provider = createFakeProvider({
    ref: 'fake:judge',
    script: () => ({
      json: {
        items: [{ id: 'correctness', score: 0.9, evidence: 'looks fine' }],
        overall: 0.9,
        blocking: [],
        rationale: 'Only scored one item.',
        decision: 'PASS',
        acceptance: [
          { index: 1, met: false, evidence: 'the regex was not changed' },
          { index: 2, met: false, evidence: 'no test covers it' },
        ],
      },
    }),
  });
  const out = await judge(provider);
  assert.equal(out.decision, 'bad');
  assert.deepEqual(out.unmet, [1, 2]);
  const spec = itemOf(out.items, SPEC_COMPLIANCE_ID);
  assert.ok(spec, 'the item is added so scoreRubric can see the miss');
  assert.equal(spec.score, ACCEPTANCE_FAIL_SCORE);
  assert.ok(out.blocking.includes(SPEC_COMPLIANCE_ID));
  assert.ok(out.rationale.startsWith(`Unmet acceptance: #1 ${C1}; #2 ${C2}`), out.rationale);
});

test('judge: a reply without the acceptance field is identical to the control run', async () => {
  const withCriteria = await judge(judgeProvider(null), PLAN);
  const control = await judge(judgeProvider(null), PLAN_NO_ACCEPTANCE);

  assert.deepEqual(withCriteria.acceptance, [], 'no attestation, no entries');
  assert.deepEqual(withCriteria.unmet, []);
  assert.deepEqual(control.acceptance, []);
  assert.deepEqual(control.unmet, []);

  // Everything a caller reads must match the pre-attestation behaviour.
  const strip = (v) => {
    const { latencyMs: _latencyMs, ...rest } = v;
    return rest;
  };
  assert.deepEqual(strip(withCriteria), strip(control));
  assert.equal(withCriteria.decision, 'good');
  assert.equal(withCriteria.rationale, 'The diff adds the unit table entry.');
});

test('judge: out-of-range and duplicate indices are dropped, first entry wins', async () => {
  const provider = judgeProvider([
    { index: 0, met: false, evidence: 'there is no criterion zero' },
    { index: 1, met: true, evidence: 'the regex now matches [dhms]' },
    { index: 1, met: false, evidence: 'a second bite at criterion one' },
    { index: 3, met: false, evidence: 'there is no criterion three' },
    { index: 2, met: true, evidence: 'the test asserts 86400' },
  ]);
  const out = await judge(provider);
  assert.deepEqual(out.acceptance.map((a) => a.index), [1, 2], 'two criteria, two entries');
  assert.equal(out.acceptance[0].met, true, 'the first entry for an index wins');
  assert.equal(out.acceptance[0].evidence, 'the regex now matches [dhms]');
  assert.deepEqual(out.unmet, []);
  assert.equal(out.decision, 'good', 'the dropped junk cannot manufacture a rejection');
});

test('judge prompt: numbered criteria in plan order plus the output contract', async () => {
  const provider = judgeProvider(null);
  await judge(provider);
  const prompt = promptOf(provider);

  assert.ok(prompt.includes('ACCEPTANCE CRITERIA (from the plan; attest to each one separately)'));
  const first = prompt.indexOf(`1. ${C1}`);
  const second = prompt.indexOf(`2. ${C2}`);
  assert.ok(first > 0, 'criterion 1 is numbered');
  assert.ok(second > first, 'criterion 2 follows criterion 1');
  assert.ok(prompt.includes('one "acceptance" entry per numbered criterion'));
  assert.ok(prompt.includes('Set "met" true only when the diff or the test output shows'));
  assert.ok(prompt.includes('is met:false'));
  assert.ok(prompt.includes('quote that line in "evidence"'));
  assert.ok(!/[^ -~\r\n\t]/.test(prompt), 'the prompt stays printable ASCII');
});

test('judge prompt: a plan with no criteria omits the section entirely', async () => {
  const plain = judgeProvider(null);
  await judge(plain, PLAN_NO_ACCEPTANCE);
  assert.ok(!promptOf(plain).includes('ACCEPTANCE CRITERIA'), 'no heading without criteria');
  assert.ok(!promptOf(plain).includes('"acceptance" entry'), 'no contract without criteria');

  const stringPlan = judgeProvider(null);
  await judge(stringPlan, 'just make parseDuration handle days');
  assert.ok(!promptOf(stringPlan).includes('ACCEPTANCE CRITERIA'), 'a string plan has no criteria');
});

// ---------------------------------------------------------- L3, the reviewer

test('reviewer: one unmet criterion turns a good review bad', async () => {
  const provider = reviewerProvider([
    { index: 1, met: true, evidence: '+  d: 86400,' },
    { index: 2, met: false, evidence: 'nothing in the diff parses 1d' },
  ]);
  const out = await review(provider);

  assert.equal(out.decision, 'bad', 'a 0.95 review with an unmet criterion is still a rejection');
  assert.deepEqual(out.unmet, [2]);
  assert.equal(out.reasons[0], `acceptance #2 not met: ${C2}`, 'the reason leads, for the repair prompt');
  assert.deepEqual(out.acceptance.map((a) => [a.index, a.met]), [[1, true], [2, false]]);
});

test('reviewer: several unmet criteria are listed in index order', async () => {
  const provider = reviewerProvider([
    { index: 2, met: false, evidence: 'no' },
    { index: 1, met: false, evidence: 'no' },
  ]);
  const out = await review(provider);
  assert.equal(out.decision, 'bad');
  assert.deepEqual(out.unmet, [1, 2]);
  assert.deepEqual(out.reasons.slice(0, 2), [
    `acceptance #1 not met: ${C1}`,
    `acceptance #2 not met: ${C2}`,
  ]);
});

test('reviewer: a reply without the acceptance field is unchanged', async () => {
  const withCriteria = await review(reviewerProvider(null), PLAN);
  const control = await review(reviewerProvider(null), PLAN_NO_ACCEPTANCE);

  assert.equal(withCriteria.decision, 'good');
  assert.deepEqual(withCriteria.acceptance, []);
  assert.deepEqual(withCriteria.unmet, []);
  assert.deepEqual(withCriteria.reasons, control.reasons, 'the reasons are the pre-attestation ones');
  assert.deepEqual(withCriteria.hits, control.hits, 'no fabricated banded hit');
  assert.equal(control.decision, 'good');
});

test('reviewer prompt: numbered criteria and the contract, or nothing at all', async () => {
  const provider = reviewerProvider(null);
  await review(provider);
  const prompt = promptOf(provider);
  assert.ok(prompt.includes('ACCEPTANCE CRITERIA (from the plan; attest to each one separately)'));
  assert.ok(prompt.indexOf(`1. ${C1}`) > 0);
  assert.ok(prompt.indexOf(`2. ${C2}`) > prompt.indexOf(`1. ${C1}`));
  assert.ok(prompt.includes('one "acceptance" entry per numbered criterion'));

  const plain = reviewerProvider(null);
  await review(plain, PLAN_NO_ACCEPTANCE);
  assert.ok(!promptOf(plain).includes('ACCEPTANCE CRITERIA'));
});

// Owner ORCH - the plan audit (spec 4.8).
//
// Three pure layers and one bounded call:
//   1. fileFacts / renderFileFacts - the measured evidence that settles a claim
//      like "these files use CRLF";
//   2. the reason parser and stuckAnalysis - which criterion blocked every
//      candidate before and after a repair round, and which candidates came out
//      of that round exactly as they went in;
//   3. criterionEntries / amendPlan - the numbering the graders use, and the
//      NEW plan a retire or a rewrite produces;
//   4. auditCriteria - one orchestrator call, never fatal.
//
// Everything is offline: scripted providers, no store outside a temp directory.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import {
  AUDIT_SCHEMA,
  acceptanceIndicesIn,
  acceptanceIndicesOf,
  amendPlan,
  applyPlanCriterionEvents,
  auditCriteria,
  criterionEntries,
  fileFacts,
  fileFactsFor,
  normalizeVerdicts,
  pathsNamedIn,
  reasonSignature,
  renderFileFacts,
  stuckAnalysis,
} from '../src/ladder/audit.js';
import { openStore, sqliteAvailable } from '../src/ladder/store.js';
import { createFakeDriver } from '../src/ladder/worktree/fake.js';
import { createFakeProvider } from '../src/ladder/providers/fake.js';

const HAS_SQLITE = sqliteAvailable();
const options = HAS_SQLITE ? {} : { skip: 'node:sqlite is unavailable on this Node' };

const tempDirs = [];
const openStores = [];

function tempStore(label) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `kolm-audit-${label}-`));
  tempDirs.push(dir);
  const store = openStore(path.join(dir, 'ladder.db'));
  openStores.push(store);
  return { store, dir };
}

test.after(() => {
  for (const store of openStores) {
    try {
      store.close();
    } catch {
      // an already-closed store is fine
    }
  }
  for (const dir of tempDirs) {
    try {
      fs.rmSync(dir, { recursive: true, force: true });
    } catch {
      // a lingering handle on Windows is not a test failure
    }
  }
});

/* --------------------------------------------------------- 1. file facts */

test('fileFacts counts CRLF and bare LF separately', () => {
  // The defect this whole feature exists for: the task said CRLF, the file was
  // LF-only, and nothing in the run ever measured it.
  const lf = fileFacts(`${Array.from({ length: 23 }, (_, i) => `line ${i}`).join('\n')}\n`);
  assert.equal(lf.lines, 23);
  assert.equal(lf.crlf, 0);
  assert.equal(lf.lf, 23);
  assert.equal(lf.bom, false);
  assert.equal(lf.trailingNewline, true);

  const crlf = fileFacts('a\r\nb\r\nc\r\n');
  assert.equal(crlf.lines, 3);
  assert.equal(crlf.crlf, 3, 'three CRLF pairs');
  assert.equal(crlf.lf, 0, 'a CRLF newline is not also a bare LF');

  const mixed = fileFacts('a\r\nb\nc\r\n');
  assert.equal(mixed.crlf, 2);
  assert.equal(mixed.lf, 1);
  assert.equal(mixed.lines, 3);
});

test('fileFacts reports the BOM, the missing trailing newline and the empty file', () => {
  const withBom = '\ufeffexport const a = 1;\n';
  const bom = fileFacts(withBom);
  assert.equal(bom.bom, true);
  assert.equal(bom.lines, 1, 'the BOM is not a line of its own');
  assert.equal(bom.bytes, Buffer.byteLength(withBom, 'utf8'));
  assert.ok(bom.bytes > 'export const a = 1;\n'.length, 'the BOM counts towards the byte size');

  const noNewline = fileFacts('a\nb');
  assert.equal(noNewline.lines, 2, 'the last line counts even without a newline');
  assert.equal(noNewline.trailingNewline, false);

  const empty = fileFacts('');
  assert.deepEqual(empty, { bytes: 0, lines: 0, crlf: 0, lf: 0, bom: false, trailingNewline: false });

  assert.equal(fileFacts(null), null, 'a missing file has no facts');
  assert.equal(fileFacts(undefined), null);
  assert.equal(fileFacts(42), null);
});

test('fileFactsFor reads the worktree and the base, and survives a driver without readFileAt', async () => {
  const driver = createFakeDriver({
    files: { 'src/inventory.js': 'a\nb\nc\n', 'README.md': '# demo\n' },
  });
  const wt = await driver.create({ runId: 'r_audit', candidateId: 'c0a0' });
  await wt.writeFile('src/inventory.js', 'a\r\nb\r\nc\r\n');

  const entries = await fileFactsFor(wt, ['src/inventory.js', 'nope/missing.js']);
  assert.equal(entries.length, 2);
  assert.equal(entries[0].facts.crlf, 3, 'the working tree has been rewritten to CRLF');
  assert.equal(entries[0].base.crlf, 0, 'the base is still LF-only');
  assert.equal(entries[0].base.lf, 3);
  assert.equal(entries[1].facts, null, 'a path that does not exist reports nothing');
  assert.equal(entries[1].base, null);

  const lines = renderFileFacts(entries);
  assert.equal(lines.length, 2, 'the missing path contributes no line');
  assert.match(lines[0], /^src\/inventory\.js: 3 lines, 3 CRLF, 0 LF, no BOM, trailing newline$/);
  assert.match(lines[1], /^src\/inventory\.js \(base\): 3 lines, 0 CRLF, 3 LF/);

  // A driver with no history support contributes working-tree facts only.
  const noHistory = { readFile: async (rel) => (rel === 'a.js' ? 'x\n' : null) };
  const partial = await fileFactsFor(noHistory, ['a.js'], { baseRef: 'HEAD' });
  assert.equal(partial[0].facts.lines, 1);
  assert.equal(partial[0].base, null);

  assert.deepEqual(await fileFactsFor(null, ['a.js']), [], 'no worktree, no facts');
  assert.deepEqual(await fileFactsFor(noHistory, []), []);
  await wt.destroy();
});

test('pathsNamedIn picks file-shaped tokens out of criterion text', () => {
  assert.deepEqual(
    pathsNamedIn([
      'Both src/inventory.js and tests/inventory.test.js use CRLF (\\r\\n) line endings throughout.',
    ]),
    ['src/inventory.js', 'tests/inventory.test.js'],
  );
  // Prose without a file name yields nothing, and a traversal is never a path.
  assert.deepEqual(pathsNamedIn(['total() must sum the quantities']), []);
  assert.deepEqual(pathsNamedIn(['see ../../etc/passwd.js and /abs/x.js']), []);
  assert.deepEqual(pathsNamedIn(undefined), []);
});

/* ----------------------------------------------------- 2. reason parsing */

test('the reason parser reads the real L3 and L4 wording', () => {
  // Verbatim from run r_20260909_f02fbbb9, scoped-acceptance parenthetical and
  // all. Criterion "#2" is index 1: the graders number from 1, the events from 0.
  const live = 'acceptance #2 not met: Both src/inventory.js and tests/inventory.test.js use CRLF '
    + '(\\r\\n) line endings throughout - no bare LF introduced. (only the src/inventory.js part is '
    + "this subtask's; tests/inventory.test.js belongs to another subtask)";
  assert.deepEqual(acceptanceIndicesIn(live), [1]);

  assert.deepEqual(acceptanceIndicesIn('acceptance #1 not met: src/math.js exports add(a, b)'), [0]);
  assert.deepEqual(
    acceptanceIndicesIn('Unmet acceptance: #2 keep CRLF; #3 do not touch package.json'),
    [1, 2],
    'the judge rationale carries several at once',
  );

  // Reasons that name no criterion: they are what makes a round "not stuck".
  assert.deepEqual(acceptanceIndicesIn('judge blocked: spec_compliance'), []);
  assert.deepEqual(acceptanceIndicesIn('rubric_score 0.400 is bad'), []);
  assert.deepEqual(acceptanceIndicesIn('the acceptance criteria are ambiguous'), []);
  assert.deepEqual(acceptanceIndicesIn(''), []);
  assert.deepEqual(acceptanceIndicesIn(null), []);

  assert.deepEqual(
    acceptanceIndicesOf(['rubric_score 0.400 is bad', 'acceptance #2 not met: x', 'acceptance #2 not met: x']),
    [1],
    'deduplicated across reasons',
  );
  assert.deepEqual(acceptanceIndicesOf(null), []);
});

test('reasonSignature is equal exactly when the failure repeated', () => {
  const before = { stage: 'l3-semantic', reasons: ['acceptance #2 not met: keep CRLF', 'rubric_score 0.700 is unsure'] };
  const same = { stage: 'l3-semantic', reasons: ['acceptance #2 not met: keep CRLF', 'rubric_score  0.700 is unsure'] };
  assert.equal(reasonSignature(before), reasonSignature(same), 'whitespace is not a difference');

  const otherCriterion = { stage: 'l3-semantic', reasons: ['acceptance #3 not met: keep CRLF'] };
  assert.notEqual(reasonSignature(before), reasonSignature(otherCriterion));

  const otherStage = { stage: 'l4-judge', reasons: before.reasons };
  assert.notEqual(reasonSignature(before), reasonSignature(otherStage));

  const otherProse = { stage: 'l3-semantic', reasons: ['acceptance #2 not met: keep CRLF', 'tests failing: add()'] };
  assert.notEqual(reasonSignature(before), reasonSignature(otherProse), 'a new non-acceptance reason is progress');
});

test('stuckAnalysis names the criterion that blocked every candidate twice', () => {
  const before = [
    { candidateId: 'c0a0', verdict: 'FAIL', stage: 'l3-semantic', reasons: ['acceptance #2 not met: keep CRLF'] },
    { candidateId: 'c1a0', verdict: 'FAIL', stage: 'l3-semantic', reasons: ['acceptance #2 not met: keep CRLF'] },
  ];
  const after = [
    { candidateId: 'c0a1', sourceId: 'c0a0', verdict: 'FAIL', stage: 'l3-semantic', reasons: ['acceptance #2 not met: keep CRLF'] },
    { candidateId: 'c1a1', sourceId: 'c1a0', verdict: 'FAIL', stage: 'l3-semantic', reasons: ['acceptance #2 not met: keep CRLF'] },
  ];
  const stuck = stuckAnalysis({ before, after });
  assert.deepEqual(stuck.criteria, [1], 'criterion #2 is index 1');
  assert.deepEqual(stuck.candidates, ['c0a1', 'c1a1'], 'neither candidate moved');

  // One candidate that got past the criterion proves it is satisfiable.
  const oneMoved = stuckAnalysis({
    before,
    after: [after[0], { ...after[1], reasons: ['rubric_score 0.700 is unsure'] }],
  });
  assert.deepEqual(oneMoved.criteria, [], 'not every failing row names it');
  assert.deepEqual(oneMoved.candidates, ['c0a1'], 'the one that repeated itself is still no-progress');

  // A PASS anywhere means nothing was stuck.
  assert.deepEqual(
    stuckAnalysis({ before, after: [{ ...after[0], verdict: 'PASS', reasons: [] }, after[1]] }),
    { criteria: [], candidates: [] },
  );
  // A first round with nothing before it, and the empty case.
  assert.deepEqual(stuckAnalysis({ before: [], after }), { criteria: [], candidates: [] });
  assert.deepEqual(stuckAnalysis({}), { criteria: [], candidates: [] });
});

/* --------------------------------------------------------- 3. the plan */

test('criterionEntries numbers exactly what the graders are shown', () => {
  const plan = { acceptance: ['first', '   ', 'second', 42, 'third'] };
  assert.deepEqual(criterionEntries(plan), [
    { index: 0, rawIndex: 0, text: 'first' },
    { index: 1, rawIndex: 2, text: 'second' },
    { index: 2, rawIndex: 4, text: 'third' },
  ]);
  assert.deepEqual(criterionEntries({}), []);
  assert.deepEqual(criterionEntries(null), []);
});

test('amendPlan retires and rewrites without touching the stored plan', () => {
  const plan = Object.freeze({
    title: 'inventory',
    acceptance: Object.freeze(['total() sums the quantities', 'keep the CRLF line endings', 'no new dependency']),
    files_expected: ['src/inventory.js'],
  });

  const retired = amendPlan(plan, [{ index: 1, verdict: 'retire', reason: 'the files are LF-only', rewrite: null }]);
  assert.equal(retired.changed, true);
  assert.deepEqual(retired.plan.acceptance, ['total() sums the quantities', 'no new dependency']);
  assert.equal(retired.plan.title, 'inventory', 'the rest of the plan is carried over');
  assert.deepEqual(plan.acceptance, [
    'total() sums the quantities',
    'keep the CRLF line endings',
    'no new dependency',
  ], 'the plan handed in is never mutated');
  assert.deepEqual(retired.changes, [{
    index: 1,
    verdict: 'retire',
    reason: 'the files are LF-only',
    from: 'keep the CRLF line endings',
    to: null,
  }]);

  const rewritten = amendPlan(plan, [{ index: 1, verdict: 'rewrite', reason: 'unmeasurable as written', rewrite: 'src/inventory.js keeps its LF line endings' }]);
  assert.deepEqual(rewritten.plan.acceptance, [
    'total() sums the quantities',
    'src/inventory.js keeps its LF line endings',
    'no new dependency',
  ]);
  assert.equal(rewritten.changes[0].to, 'src/inventory.js keeps its LF line endings');

  // keep, an empty rewrite and a rewrite that changes nothing are all no-ops.
  for (const verdict of [
    { index: 1, verdict: 'keep', reason: 'a coder could still do this' },
    { index: 1, verdict: 'rewrite', reason: 'no wording given', rewrite: '' },
    { index: 1, verdict: 'rewrite', reason: 'same words', rewrite: 'keep the CRLF line endings' },
  ]) {
    const out = amendPlan(plan, [verdict]);
    assert.equal(out.changed, false);
    assert.equal(out.plan, plan, 'an unchanged plan is the same object');
  }
});

test('amendPlan never retires the last criterion', () => {
  const plan = { acceptance: ['keep the CRLF line endings'] };
  const out = amendPlan(plan, [{ index: 0, verdict: 'retire', reason: 'unsatisfiable' }]);
  assert.equal(out.changed, false, 'a plan with nothing to attest to would pass every candidate');
  assert.deepEqual(out.plan.acceptance, ['keep the CRLF line endings']);

  const two = { acceptance: ['a', 'b'] };
  const both = amendPlan(two, [
    { index: 0, verdict: 'retire', reason: 'x' },
    { index: 1, verdict: 'retire', reason: 'y' },
  ]);
  assert.deepEqual(both.plan.acceptance, ['b'], 'the second retire is refused, not the first');
});

test('normalizeVerdicts drops what the audit was not asked about', () => {
  const raw = [
    { index: 1, verdict: 'RETIRE', reason: '  the files   are LF-only ' },
    { index: 1, verdict: 'keep', reason: 'duplicate index' },
    { index: 4, verdict: 'retire', reason: 'never asked about this one' },
    { index: 'x', verdict: 'retire', reason: 'not an index' },
    { index: 2, verdict: 'maybe', reason: 'not a verdict' },
    { index: 2, verdict: 'rewrite', reason: 'ok', rewrite: '  new   wording ' },
    'not an object',
  ];
  assert.deepEqual(normalizeVerdicts(raw, { allowed: [1, 2] }), [
    { index: 1, verdict: 'retire', reason: 'the files are LF-only', rewrite: null },
    { index: 2, verdict: 'rewrite', reason: 'ok', rewrite: 'new wording' },
  ]);
  assert.deepEqual(normalizeVerdicts(null), []);
});

test('applyPlanCriterionEvents rebuilds an amended plan on resume', () => {
  const plan = { acceptance: ['total() sums the quantities', 'keep the CRLF line endings'] };
  const events = [
    { type: 'repair.stuck', payload: { unit: null, round: 1, criteria: [1], candidates: ['c0a1'] } },
    { type: 'plan.criterion', payload: { unit: null, round: 1, index: 1, verdict: 'retire', reason: 'LF-only', from: 'keep the CRLF line endings', to: null } },
    { type: 'plan.criterion', payload: { unit: 3, round: 1, index: 0, verdict: 'retire', reason: 'another subtask', from: 'total() sums the quantities', to: null } },
  ];
  const out = applyPlanCriterionEvents(plan, events, null);
  assert.equal(out.applied, 1, 'another unit-s amendment is not this unit-s');
  assert.deepEqual(out.plan.acceptance, ['total() sums the quantities']);

  assert.equal(applyPlanCriterionEvents(plan, [], null).changed, false);
  assert.equal(applyPlanCriterionEvents(plan, events, 7).applied, 0);
});

/* ------------------------------------------------------- 4. the call */

const CRITERIA = [{ index: 1, text: 'Both src/inventory.js and tests/inventory.test.js use CRLF (\\r\\n) line endings throughout.' }];
const PLAN = Object.freeze({
  title: 'inventory total',
  summary: 'Make total() sum the quantities.',
  acceptance: Object.freeze([
    'total() returns the sum of quantity * price',
    'Both src/inventory.js and tests/inventory.test.js use CRLF (\\r\\n) line endings throughout.',
  ]),
});

function auditor(json, { costUsd = 0.0004 } = {}) {
  return createFakeProvider({
    ref: 'anthropic:claude-sonnet',
    script: () => ({ json, costUsd }),
  });
}

function harness(label) {
  const { store, dir } = tempStore(label);
  // The run the whole feature exists for: every candidate rejected for the same
  // acceptance criterion, twice over.
  const runId = 'r_20260909_f02fbbb9';
  store.createRun({ id: runId, task: { text: 'inventory total' }, config: {} });
  const events = [];
  const view = Object.create(store);
  view.emit = (id, type, payload) => {
    events.push({ type, payload });
    return store.emit(id, type, payload);
  };
  return { store: view, raw: store, dir, runId, events };
}

test('a retire verdict produces a new plan, one event and a ledger row', options, async () => {
  const { store, raw, runId, events } = harness('retire');
  const orchestrator = auditor({
    verdicts: [{ index: 1, verdict: 'retire', reason: 'both files are LF-only in the base and the task never asks to convert them' }],
  });

  const out = await auditCriteria({
    orchestrator,
    plan: PLAN,
    criteria: CRITERIA,
    task: 'Fix total() in src/inventory.js. The files use CRLF; keep them that way.',
    diffText: '--- a/src/inventory.js\n+++ b/src/inventory.js\n@@\n+  return items.reduce((s, i) => s + i.quantity * i.price, 0);\n',
    fileFacts: [{
      path: 'src/inventory.js',
      facts: { bytes: 400, lines: 23, crlf: 0, lf: 23, bom: false, trailingNewline: true },
      base: null,
    }],
    config: {},
    store,
    runId,
    unit: null,
    round: 1,
  });

  assert.equal(out.changed, true);
  assert.equal(out.skipped, null);
  assert.deepEqual(out.plan.acceptance, ['total() returns the sum of quantity * price']);
  assert.deepEqual(PLAN.acceptance.length, 2, 'the plan handed in is untouched');

  const emitted = events.filter((e) => e.type === 'plan.criterion');
  assert.equal(emitted.length, 1);
  assert.equal(emitted[0].payload.index, 1, 'the event numbers criteria from 0: acceptance #2 is index 1');
  assert.equal(emitted[0].payload.verdict, 'retire');
  assert.equal(emitted[0].payload.unit, null);
  assert.equal(emitted[0].payload.round, 1);
  assert.equal(emitted[0].payload.from, CRITERIA[0].text);
  assert.equal(emitted[0].payload.to, null);
  assert.ok(emitted[0].payload.costUsd > 0, 'the audit call is on the event');

  const cost = raw.costSummary(runId);
  assert.ok(cost.total_usd > 0, 'the audit call is on the ledger');
  assert.ok(cost.by_stage.audit, 'the ledger row carries the audit stage');
  assert.equal(cost.by_stage.audit.calls, 1);

  // The call itself: one bounded, low-effort request with the audit schema.
  const call = orchestrator.calls[0];
  assert.equal(call.jsonSchema.name, AUDIT_SCHEMA.name);
  assert.equal(call.effort, 'low');
  assert.ok(call.maxTokens <= 1000 && call.maxTokens > 0);
  const prompt = String(call.messages[0].content);
  assert.match(prompt, /FILE FACTS/);
  assert.match(prompt, /23 lines, 0 CRLF, 23 LF/, 'the measured evidence reached the prompt');
  assert.match(prompt, /index 1:/, 'the criterion is presented with the index it must answer with');
  assert.match(prompt, /BEST CANDIDATE DIFF/);
  assert.ok(/^[\x20-\x7e\n]*$/.test(prompt), 'the prompt is ASCII');
});

test('a rewrite verdict replaces the wording in place', options, async () => {
  const { store, runId, events } = harness('rewrite');
  const orchestrator = auditor({
    verdicts: [{
      index: 1,
      verdict: 'rewrite',
      reason: 'line endings cannot be checked from a diff; the testable part is that no file is converted',
      rewrite: 'src/inventory.js keeps the line endings it already has.',
    }],
  });

  const out = await auditCriteria({
    orchestrator, plan: PLAN, criteria: CRITERIA, config: {}, store, runId, unit: 2, round: 1,
  });
  assert.equal(out.changed, true);
  assert.deepEqual(out.plan.acceptance, [
    'total() returns the sum of quantity * price',
    'src/inventory.js keeps the line endings it already has.',
  ]);
  const emitted = events.filter((e) => e.type === 'plan.criterion');
  assert.equal(emitted.length, 1);
  assert.equal(emitted[0].payload.verdict, 'rewrite');
  assert.equal(emitted[0].payload.unit, 2);
  assert.equal(emitted[0].payload.to, 'src/inventory.js keeps the line endings it already has.');
});

test('a keep verdict changes nothing and says nothing', options, async () => {
  const { store, runId, events } = harness('keep');
  const orchestrator = auditor({ verdicts: [{ index: 1, verdict: 'keep', reason: 'the coder can still convert the file' }] });
  const out = await auditCriteria({ orchestrator, plan: PLAN, criteria: CRITERIA, config: {}, store, runId });
  assert.equal(out.changed, false);
  assert.equal(out.plan, PLAN, 'the same plan object comes back');
  assert.equal(events.filter((e) => e.type === 'plan.criterion').length, 0);
  assert.equal(out.verdicts.length, 1, 'the verdict was still read');
});

test('an audit that throws, or is turned off, leaves the plan exactly as it was', options, async () => {
  const { store, runId, events } = harness('failure');

  const thrower = createFakeProvider({
    ref: 'anthropic:claude-sonnet',
    script: () => { throw new Error('provider unavailable'); },
  });
  const failed = await auditCriteria({ orchestrator: thrower, plan: PLAN, criteria: CRITERIA, config: {}, store, runId });
  assert.equal(failed.changed, false, 'a failed audit is not a failed run');
  assert.equal(failed.plan, PLAN);
  assert.equal(failed.costUsd, 0);

  // Junk that matches no schema is the same as no answer.
  const junk = auditor({ verdicts: 'not an array' });
  assert.equal((await auditCriteria({ orchestrator: junk, plan: PLAN, criteria: CRITERIA, config: {}, store, runId })).changed, false);

  // The skips, in the order the code checks them.
  const off = auditor({ verdicts: [{ index: 1, verdict: 'retire', reason: 'never asked' }] });
  const disabled = await auditCriteria({ orchestrator: off, plan: PLAN, criteria: CRITERIA, config: { audit: false }, store, runId });
  assert.equal(disabled.skipped, 'disabled');
  assert.equal(off.calls.length, 0, 'no call was spent');

  const broke = await auditCriteria({
    orchestrator: off, plan: PLAN, criteria: CRITERIA, config: {}, store, runId, budgetExhausted: true,
  });
  assert.equal(broke.skipped, 'budget');

  assert.equal((await auditCriteria({ orchestrator: null, plan: PLAN, criteria: CRITERIA, config: {}, store, runId })).skipped, 'no-orchestrator');
  assert.equal((await auditCriteria({ orchestrator: off, plan: PLAN, criteria: [], config: {}, store, runId })).skipped, 'no-criteria');

  // Unlike the coach, the audit has no "same model" skip at all: the question
  // is about the plan, not a second opinion on code, so it is asked even when
  // the orchestrator is also the coder.
  const same = auditor({ verdicts: [{ index: 1, verdict: 'retire', reason: 'LF-only' }] });
  const asked = await auditCriteria({
    orchestrator: same, plan: PLAN, criteria: CRITERIA, config: {}, store, runId,
  });
  assert.equal(asked.changed, true);
  assert.equal(same.calls.length, 1);

  assert.equal(events.filter((e) => e.type === 'plan.criterion').length, 1, 'only the last audit changed anything');
});

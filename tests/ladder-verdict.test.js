// Owner C — cascade verdict assembly, gates and funnel math (spec 5.4).
import test from 'node:test';
import assert from 'node:assert/strict';
import { assembleVerdict, flowsUp, funnel } from '../src/ladder/verdict.js';

test('flowsUp lets everything except a confident bad rise', () => {
  assert.equal(flowsUp('good'), true);
  assert.equal(flowsUp('unsure'), true);
  assert.equal(flowsUp('skip'), true);
  assert.equal(flowsUp(undefined), true);
  assert.equal(flowsUp(null), true);
  assert.equal(flowsUp('bad'), false);
});

test('assembleVerdict truth table over the four decisions', () => {
  const cases = [
    [{}, 'PASS'],
    [{ 'l0-meta': 'good' }, 'PASS'],
    [{ 'l0-meta': 'good', 'l1-static': 'good', 'l2-dynamic': 'good', 'l3-semantic': 'good' }, 'PASS'],
    [{ 'l0-meta': 'good', 'l1-static': 'skip' }, 'PASS'],
    [{ 'l0-meta': 'skip', 'l1-static': 'skip' }, 'PASS'],
    [{ 'l0-meta': 'unsure' }, 'JUDGE'],
    [{ 'l0-meta': 'good', 'l3-semantic': 'unsure' }, 'JUDGE'],
    [{ 'l0-meta': 'unsure', 'l1-static': 'skip' }, 'JUDGE'],
    [{ 'l0-meta': 'bad' }, 'FAIL'],
    [{ 'l0-meta': 'bad', 'l1-static': 'unsure' }, 'FAIL'],
    [{ 'l0-meta': 'good', 'l2-dynamic': 'bad', 'l3-semantic': 'unsure' }, 'FAIL'],
  ];
  for (const [decisions, expected] of cases) {
    assert.equal(assembleVerdict(decisions).verdict, expected, JSON.stringify(decisions));
  }
});

test('assembleVerdict reports which stages were bad unsure and skipped', () => {
  const v = assembleVerdict({
    'l0-meta': 'good',
    'l1-static': 'skip',
    'l2-dynamic': 'unsure',
    'l3-semantic': 'bad',
  });
  assert.equal(v.verdict, 'FAIL');
  assert.deepEqual(v.bad, ['l3-semantic']);
  assert.deepEqual(v.unsure, ['l2-dynamic']);
  assert.deepEqual(v.skipped, ['l1-static']);
  assert.ok(v.reasons.some((r) => r.includes('l3-semantic')));
  assert.ok(v.reasons.some((r) => r.includes('skipped')));
});

test('assembleVerdict tolerates a null or partial decision map', () => {
  const v = assembleVerdict(null);
  assert.equal(v.verdict, 'PASS');
  assert.deepEqual(v.bad, []);
  assert.deepEqual(v.unsure, []);
  assert.deepEqual(v.skipped, []);
});

test('funnel counts decisions per stage across runs', () => {
  const rows = [
    { runId: 'r1', candidateId: 'c0a0', stage: 'l0-meta', decision: 'good' },
    { runId: 'r1', candidateId: 'c0a0', stage: 'l1-static', decision: 'good' },
    { runId: 'r1', candidateId: 'c1a0', stage: 'l0-meta', decision: 'bad' },
    { runId: 'r2', candidateId: 'c0a0', stage: 'l0-meta', decision: 'unsure' },
    { runId: 'r2', candidateId: 'c0a0', stage: 'l1-static', decision: 'skip' },
  ];
  const f = funnel(rows);
  assert.equal(f.runs, 2);
  assert.equal(f.candidates, 3);
  assert.deepEqual(f.stages['l0-meta'], { good: 1, unsure: 1, bad: 1, skip: 0 });
  assert.deepEqual(f.stages['l1-static'], { good: 1, unsure: 0, bad: 0, skip: 1 });
  assert.deepEqual(f.verdicts, { PASS: 1, FAIL: 1, JUDGE: 1 });
  assert.equal(f.hard_fail, 1);
  assert.equal(f.hardFail, 1);
});

test('funnel cleared_cheaply is the share decided without the judge', () => {
  const rows = [
    { runId: 'r1', candidateId: 'a', stage: 'l0-meta', decision: 'good' },
    { runId: 'r1', candidateId: 'b', stage: 'l0-meta', decision: 'bad' },
    { runId: 'r1', candidateId: 'c', stage: 'l0-meta', decision: 'unsure' },
    { runId: 'r1', candidateId: 'd', stage: 'l0-meta', decision: 'unsure' },
  ];
  const f = funnel(rows);
  // a (PASS) and b (FAIL) were settled by the cheap rungs; c and d escalate.
  assert.equal(f.cleared_cheaply, 0.5);
  assert.equal(f.clearedCheaply, 0.5);
  assert.equal(f.judged, 0);
});

test('funnel counts a judged candidate even when every stage was good', () => {
  const rows = [
    { runId: 'r1', candidateId: 'a', stage: 'l0-meta', decision: 'good', judged: true },
    { runId: 'r1', candidateId: 'b', stage: 'l0-meta', decision: 'good' },
    { runId: 'r1', candidateId: 'c', stage: 'l0-meta', decision: 'unsure', judgeVerdict: 'PASS' },
  ];
  const f = funnel(rows);
  assert.equal(f.judged, 2);
  assert.equal(f.candidates, 3);
  // Only b avoided the judge entirely.
  assert.equal(f.cleared_cheaply, 1 / 3);
});

test('funnel on an empty row list returns zeroes not division by zero', () => {
  const f = funnel([]);
  assert.equal(f.candidates, 0);
  assert.equal(f.runs, 0);
  assert.equal(f.cleared_cheaply, 0);
  assert.equal(f.judged, 0);
  assert.deepEqual(f.verdicts, { PASS: 0, FAIL: 0, JUDGE: 0 });
  assert.deepEqual(funnel(null).stages, {});
});

test('funnel keeps candidates from different runs separate', () => {
  const rows = [
    { runId: 'r1', candidateId: 'c0a0', stage: 'l0-meta', decision: 'good' },
    { runId: 'r2', candidateId: 'c0a0', stage: 'l0-meta', decision: 'bad' },
  ];
  const f = funnel(rows);
  assert.equal(f.candidates, 2);
  assert.deepEqual(f.verdicts, { PASS: 1, FAIL: 1, JUDGE: 0 });
});

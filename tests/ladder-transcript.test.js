// kolm code - transcript compaction (spec 2.1). Owner CODER.
//
// Pure functions over message lists: no provider, no worktree, no clock.

import test from 'node:test';
import assert from 'node:assert/strict';

import { estimateTokens, compactTranscript } from '../src/ladder/transcript.js';

const OPENING = 'TASK\nimplement the plan\n\nREPO BRIEF\nroot: /repo/demo\n';

/** One assistant tool_use turn plus the user turn carrying its tool_result. */
function toolTurn(id, name, input, resultText) {
  return [
    { role: 'assistant', content: [{ type: 'tool_use', id, name, input }] },
    { role: 'user', content: [{ type: 'tool_result', tool_use_id: id, content: resultText }] },
  ];
}

/** `n` read_file turns, each returning `chars` characters. */
function readTranscript(n, chars = 2000) {
  const messages = [{ role: 'user', content: OPENING }];
  for (let i = 1; i <= n; i += 1) {
    messages.push(...toolTurn(`t${i}`, 'read_file', { path: `src/a${i}.js` }, 'X'.repeat(chars)));
  }
  return messages;
}

/** The tool_result strings still present in a transcript, in order. */
function toolResults(messages) {
  const out = [];
  for (const m of messages) {
    if (!Array.isArray(m?.content)) continue;
    for (const b of m.content) if (b?.type === 'tool_result') out.push(String(b.content));
  }
  return out;
}

/* ------------------------------------------------------------ estimateTokens */

test('estimateTokens is monotonic: appending a message never lowers the estimate', () => {
  const messages = readTranscript(12, 900);
  assert.equal(estimateTokens([]), 0);
  let previous = 0;
  for (let i = 1; i <= messages.length; i += 1) {
    const now = estimateTokens(messages.slice(0, i));
    assert.ok(now >= previous, `estimate fell at message ${i}: ${now} < ${previous}`);
    previous = now;
  }
  assert.ok(previous > 0);
});

test('estimateTokens counts tool_result strings exactly and charges per block', () => {
  const one = estimateTokens([{ role: 'user', content: 'abcdefghij' }]);
  // 10 chars / 3.6 -> 3 tokens, plus the 12-token block envelope.
  assert.equal(one, 15);

  const big = 'Y'.repeat(3600);
  const withResult = estimateTokens([
    { role: 'user', content: [{ type: 'tool_result', tool_use_id: 't1', content: big }] },
  ]);
  assert.equal(withResult, 1000 + 12);

  // A string content and a single text block cost the same.
  const asBlock = estimateTokens([{ role: 'user', content: [{ type: 'text', text: 'abcdefghij' }] }]);
  assert.equal(asBlock, one);
});

/* ---------------------------------------------------------------- no-op path */

test('compactTranscript is a no-op under budget and returns the caller array untouched', () => {
  const messages = readTranscript(3, 100);
  const result = compactTranscript(messages, { budgetTokens: 100000 });
  assert.equal(result.note, null);
  assert.equal(result.dropped, 0);
  assert.equal(result.savedTokens, 0);
  assert.equal(result.messages, messages, 'the same array comes back');
});

test('compactTranscript refuses to act without a usable budget', () => {
  const messages = readTranscript(20, 2000);
  for (const budgetTokens of [0, -1, Number.NaN, undefined]) {
    const result = compactTranscript(messages, { budgetTokens });
    assert.equal(result.note, null, `budget ${String(budgetTokens)} should be a no-op`);
    assert.equal(result.messages, messages);
  }
});

/* --------------------------------------------------------------- stub policy */

test('compaction keeps the opening and the last 6 turns intact and stubs older tool results', () => {
  const messages = readTranscript(20, 2000);
  const before = estimateTokens(messages);
  const result = compactTranscript(messages, { budgetTokens: 5000 });

  assert.ok(result.note && result.note.includes('stubbed'), result.note);
  assert.ok(!result.note.includes('collapsed'), 'stubbing alone was enough');
  assert.ok(result.dropped > 0);
  assert.ok(result.savedTokens > 0);
  assert.equal(estimateTokens(result.messages), before - result.savedTokens);
  assert.ok(estimateTokens(result.messages) <= 5000, 'compaction met the budget');

  // 1. the opening survives byte for byte, and the caller's array is not mutated.
  assert.equal(result.messages[0].content, OPENING);
  assert.equal(messages[0].content, OPENING);
  assert.equal(result.messages.length, messages.length, 'stubbing drops no messages');

  const kept = toolResults(result.messages);
  assert.equal(kept.length, 20);
  // 2. the last six turns keep their full output...
  for (const text of kept.slice(-6)) assert.equal(text.length, 2000);
  // ...and at least one older turn is a stub that names the tool and its path.
  const stub = kept.find((t) => t.startsWith('[trimmed'));
  assert.equal(stub, '[trimmed read_file src/a1.js: 2000 chars; re-run the tool if you need it]');
});

test('compaction never stubs an edit_file or write_file result', () => {
  const messages = [{ role: 'user', content: OPENING }];
  for (let i = 1; i <= 20; i += 1) {
    const write = i % 2 === 0;
    messages.push(...toolTurn(
      `t${i}`,
      write ? 'write_file' : 'read_file',
      { path: `src/a${i}.js` },
      write ? `wrote src/a${i}.js ${'W'.repeat(2000)}` : 'X'.repeat(2000),
    ));
  }
  const result = compactTranscript(messages, { budgetTokens: 9000 });
  assert.ok(result.note?.includes('stubbed'), result.note);
  const kept = toolResults(result.messages);
  for (let i = 0; i < kept.length; i += 1) {
    if (i % 2 === 1) assert.ok(kept[i].startsWith('wrote src/a'), `write result ${i} was stubbed: ${kept[i].slice(0, 60)}`);
  }
  assert.ok(kept.some((t) => t.startsWith('[trimmed read_file')), 'reads were stubbed instead');
});

test('a tool result under minToolResultChars is left alone', () => {
  const messages = readTranscript(20, 300);
  const result = compactTranscript(messages, { budgetTokens: 200, minToolResultChars: 400 });
  for (const text of toolResults(result.messages)) {
    assert.ok(!text.startsWith('[trimmed'), 'a 300-char result is not worth a stub');
  }
  // Nothing was stubbable, so the collapse is what did the work.
  assert.ok(result.note?.includes('collapsed'), result.note);
});

/* ------------------------------------------------------------ collapse policy */

test('compaction collapses the oldest turns into one mechanical summary when stubbing is not enough', () => {
  const messages = [{ role: 'user', content: OPENING }];
  messages.push(...toolTurn('t1', 'read_file', { path: 'src/index.js' }, 'X'.repeat(3000)));
  messages.push(...toolTurn('t2', 'edit_file', { path: 'src/math.js' }, 'edited src/math.js (1 replacement, +12 chars)'));
  messages.push(...toolTurn('t3', 'run', { cmd: 'npm test' }, 'exit code: 0\nok 1 - add adds numbers\n'));
  messages.push(...toolTurn('t4', 'delete_file', { path: 'src/old.js' }, 'deleted src/old.js'));
  for (let i = 5; i <= 12; i += 1) {
    messages.push(...toolTurn(`t${i}`, 'read_file', { path: `src/a${i}.js` }, 'X'.repeat(3000)));
  }

  const result = compactTranscript(messages, { budgetTokens: 400 });
  assert.ok(result.note?.includes('collapsed'), result.note);
  assert.ok(result.messages.length < messages.length);

  // The opening is still the opening, then one assistant summary, then a user turn.
  assert.equal(result.messages[0].content, OPENING);
  assert.equal(result.messages[1].role, 'assistant');
  const summary = result.messages[1].content[0].text;
  assert.ok(summary.startsWith('[earlier turns: '), summary);
  assert.ok(summary.endsWith(']'), summary);
  assert.ok(summary.includes('read src/index.js'), summary);
  assert.ok(summary.includes('edited src/math.js'), summary);
  assert.ok(summary.includes('deleted src/old.js'), summary);
  assert.ok(summary.includes('ran npm test (exit 0)'), summary);
  assert.equal(result.messages[2].role, 'user');

  // Roles still alternate, so no provider can reject the rebuilt transcript.
  for (let i = 1; i < result.messages.length; i += 1) {
    assert.notEqual(result.messages[i].role, result.messages[i - 1].role, `roles repeat at ${i}`);
  }
  // No orphaned tool_result: every surviving one still has its tool_use.
  const ids = new Set();
  for (const m of result.messages) {
    if (!Array.isArray(m.content)) continue;
    for (const b of m.content) if (b.type === 'tool_use') ids.add(b.id);
  }
  for (const m of result.messages) {
    if (!Array.isArray(m.content)) continue;
    for (const b of m.content) {
      if (b.type === 'tool_result') assert.ok(ids.has(b.tool_use_id), `orphan tool_result ${b.tool_use_id}`);
    }
  }
});

test('the summary lists each file once and caps long lists', () => {
  const messages = [{ role: 'user', content: OPENING }];
  for (let i = 1; i <= 24; i += 1) {
    // Twelve distinct paths, each read twice.
    messages.push(...toolTurn(`t${i}`, 'read_file', { path: `src/a${i % 12}.js` }, 'X'.repeat(3000)));
  }
  const result = compactTranscript(messages, { budgetTokens: 400 });
  const summary = result.messages[1].content[0].text;
  assert.ok(/read src\/a\d+\.js(, src\/a\d+\.js){5}, \+\d+ more/.test(summary), summary);
  assert.ok(summary.length <= 1200);
});

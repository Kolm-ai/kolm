// kolm code - coder runtime: self-review, tool-result hygiene, steering hints
// and transcript compaction inside the loop (spec 2.2-2.5). Owner CODER.
//
// Offline throughout: the in-memory fake driver and the scripted fake provider.
// `tests/ladder-hardening.test.js` pins the tool semantics themselves; this file
// only covers what the SOTA wave added.

import test from 'node:test';
import assert from 'node:assert/strict';

import {
  runCoder,
  runRepair,
  contextWindowFor,
  transcriptBudget,
  MAX_FINISH_REJECTIONS,
} from '../src/ladder/coder.js';
import { createFakeDriver } from '../src/ladder/worktree/fake.js';
import { createFakeProvider } from '../src/ladder/providers/fake.js';

const TEST_CMD = 'npm test --silent';

const BASE_FILES = {
  'package.json': `${JSON.stringify({ name: 'demo', version: '1.0.0', scripts: { test: 'node --test' } }, null, 2)}\n`,
  'README.md': '# demo\n\nA tiny repository for the coder tests.\n',
  'src/index.js': "export const name = 'demo';\n",
};

const GOOD_MATH = 'export function add(a, b) {\n  return a + b;\n}\n';

const PLAN_JSON = {
  title: 'add add(a, b)',
  summary: 'Create src/math.js with an add function.',
  acceptance: ['src/math.js exports add(a, b)'],
  files_expected: ['src/math.js'],
  forbidden_paths: ['package.json'],
  test_command: TEST_CMD,
  risk: 'low',
  budget_hint_usd: 0.25,
  clarifications: [],
};

/** A store stand-in for the coder loop: records events, charges nothing. */
function stubStore(events) {
  return {
    emit(runId, type, payload) {
      events.push({ type, payload });
      return events.length;
    },
    addCost() {},
    upsertCandidate() {},
    costSummary() {
      return { total_usd: 0 };
    },
  };
}

async function coderWith(script, {
  config = {},
  files = BASE_FILES,
  exec = {},
  maxTurns,
  hints,
  provider: injected = null,
} = {}) {
  const driver = createFakeDriver({ files, exec });
  const wt = await driver.create({ repoPath: '/repo/demo', runId: 'r_coder', candidateId: 'c0a0' });
  const events = [];
  const provider = injected ?? createFakeProvider({ ref: 'ollama:qwen3:8b', script });
  const out = await runCoder({
    runId: 'r_coder',
    candidateId: 'c0a0',
    wt,
    plan: PLAN_JSON,
    coder: provider,
    store: stubStore(events),
    config,
    maxTurns,
    hints,
  });
  return { out, events, provider, wt };
}

const WRITE_MATH = { toolCalls: [{ name: 'write_file', input: { path: 'src/math.js', content: GOOD_MATH } }] };
const FINISH = { toolCalls: [{ name: 'finish', input: { summary: 'wrote src/math.js' } }] };

/* =========================================================================
 * 2.1 - transcript budget
 * ========================================================================= */

test('the context window falls back to 32k for a local model the catalog only knows by wildcard', () => {
  assert.equal(contextWindowFor('ollama:qwen3:8b'), 32768);
  assert.equal(contextWindowFor('ollama:deepseek-coder-v2'), 32768);
  assert.equal(contextWindowFor(null), 32768);
  assert.equal(contextWindowFor('anthropic:claude-sonnet-5'), 1000000);
});

test('the transcript budget is 75 percent of the window less the reply cap', () => {
  assert.equal(transcriptBudget('ollama:qwen3:8b', 8000, {}), Math.floor(32768 * 0.75) - 8000);
  assert.equal(transcriptBudget('anthropic:claude-sonnet-5', 8000, {}), Math.floor(1000000 * 0.75) - 8000);
  // KOLM_CODE_CONTEXT_TOKENS only ever lowers the window.
  assert.equal(transcriptBudget('ollama:qwen3:8b', 8000, { contextTokens: 40000 }), Math.floor(32768 * 0.75) - 8000);
  assert.equal(transcriptBudget('ollama:qwen3:8b', 2000, { contextTokens: 12000 }), Math.floor(12000 * 0.75) - 2000);
  // A budget the reply cap would drive negative is floored, not inverted.
  assert.equal(transcriptBudget('ollama:qwen3:8b', 8000, { contextTokens: 4000 }), 1000);
});

test('coder.turn carries the token estimate and whether the turn was compacted', async () => {
  const { events } = await coderWith([FINISH]);
  const turns = events.filter((e) => e.type === 'coder.turn');
  assert.equal(turns.length, 1);
  assert.equal(typeof turns[0].payload.estTokens, 'number');
  assert.ok(turns[0].payload.estTokens > 0, 'the opening message costs tokens');
  assert.equal(turns[0].payload.compacted, false);
});

test('a transcript over the budget is compacted before the call, and the coder still finishes', async () => {
  const big = `${'const filler = 1;'.repeat(12)}\n`.repeat(40); // ~8 KB over 40 lines
  const files = { ...BASE_FILES, 'src/big.js': big };
  const script = (call, index) => {
    if (index < 7) return { toolCalls: [{ name: 'read_file', input: { path: 'src/big.js' } }] };
    if (index === 7) return WRITE_MATH;
    return FINISH;
  };
  const { out, events } = await coderWith(script, {
    files,
    config: { contextTokens: 2000 },
    maxTurns: 20,
  });

  assert.equal(out.reason, 'finish');
  assert.ok(out.compactions > 0, 'compaction ran');
  const compacts = events.filter((e) => e.type === 'coder.compact');
  assert.ok(compacts.length > 0, 'a coder.compact event was emitted');
  const first = compacts[0].payload;
  assert.equal(first.candidateId, 'c0a0');
  assert.ok(first.turn >= 2);
  assert.ok(first.dropped > 0);
  assert.ok(first.savedTokens > 0);
  assert.match(first.note, /^compacted transcript: /);

  const compactedTurn = events.find((e) => e.type === 'coder.turn' && e.payload.compacted === true);
  assert.ok(compactedTurn, 'the turn that compacted says so');
  assert.equal(compactedTurn.payload.turn, first.turn);
});

/* =========================================================================
 * 2.2 - self-review before finish
 * ========================================================================= */

test('self-review happens once: the finish becomes a review turn and the second finish ends it', async () => {
  const { out, events, provider } = await coderWith(
    [WRITE_MATH, FINISH, FINISH],
    { config: { selfReview: true } },
  );

  assert.equal(out.reason, 'finish');
  assert.equal(out.finished, true);
  assert.equal(out.turns, 3, 'the review costs exactly one extra turn');
  assert.equal(out.selfReviewed, true);

  const reviews = events.filter((e) => e.type === 'coder.self_review');
  assert.equal(reviews.length, 1);
  assert.equal(reviews[0].payload.candidateId, 'c0a0');
  assert.equal(reviews[0].payload.turn, 2);
  assert.ok(reviews[0].payload.diffBytes > 0);
  assert.deepEqual(reviews[0].payload.changedFiles, ['src/math.js']);
  assert.equal(reviews[0].payload.skipped, undefined);

  // The review is a real user turn the model sees, carrying the diff and the
  // acceptance criteria.
  const prompt = String(provider.calls[2].messages.at(-1).content);
  assert.ok(prompt.startsWith('SELF-REVIEW before finish. This is your diff:'), prompt.slice(0, 80));
  assert.ok(prompt.includes('diff --git a/src/math.js b/src/math.js'), prompt.slice(0, 400));
  assert.ok(prompt.includes('Check every acceptance criterion:'), prompt);
  assert.ok(prompt.includes('  - src/math.js exports add(a, b)'), prompt);
  assert.ok(prompt.includes('no unrelated files, no leftover debug output'), prompt);
  assert.ok(prompt.includes('call finish again'), prompt);
});

test('the second finish ends the session even when the review changed the diff', async () => {
  const { out, events, wt } = await coderWith(
    [
      WRITE_MATH,
      FINISH,
      {
        toolCalls: [
          { name: 'write_file', input: { path: 'src/extra.js', content: 'export const extra = 1;\n' } },
          { name: 'finish', input: { summary: 'and one more file' } },
        ],
      },
    ],
    { config: { selfReview: true } },
  );
  assert.equal(out.reason, 'finish');
  assert.equal(out.turns, 3);
  assert.equal(events.filter((e) => e.type === 'coder.self_review').length, 1, 'never a second review');
  assert.equal(await wt.readFile('src/extra.js'), 'export const extra = 1;\n');
});

test('self-review is skipped with reason no-turns when the cap leaves no room', async () => {
  const { out, events } = await coderWith(
    [WRITE_MATH, FINISH],
    { config: { selfReview: true }, maxTurns: 2 },
  );
  assert.equal(out.reason, 'finish');
  assert.equal(out.turns, 2);
  assert.equal(out.selfReviewed, false);
  const reviews = events.filter((e) => e.type === 'coder.self_review');
  assert.equal(reviews.length, 1);
  assert.equal(reviews[0].payload.skipped, 'no-turns');
  assert.deepEqual(reviews[0].payload.changedFiles, ['src/math.js']);
});

test('self-review is disabled by config and off unless the operator asks for it', async () => {
  for (const config of [{}, { selfReview: false }]) {
    const { out, events } = await coderWith([WRITE_MATH, FINISH], { config });
    assert.equal(out.reason, 'finish');
    assert.equal(out.turns, 2, `config ${JSON.stringify(config)} spent an extra turn`);
    assert.equal(out.selfReviewed, false);
    assert.equal(events.filter((e) => e.type === 'coder.self_review').length, 0);
  }
});

test('a finish with an empty diff gets one nudge, then the second finish ends the session', async () => {
  // Live run r_20260909_1c67825d: a small model read two files and called
  // finish with nothing changed. One user turn saying so is cheaper than a
  // whole cascade round that ends in "empty diff".
  const { out, events, provider } = await coderWith(
    [{ toolCalls: [{ name: 'read_file', input: { path: 'src/index.js' } }] }, FINISH, FINISH],
    { config: { selfReview: true } },
  );
  assert.equal(out.reason, 'finish');
  assert.equal(out.turns, 3);
  assert.equal(out.selfReviewed, true);
  const reviews = events.filter((e) => e.type === 'coder.self_review');
  assert.equal(reviews.length, 1);
  assert.equal(reviews[0].payload.emptyDiff, true);
  assert.equal(reviews[0].payload.diffBytes, 0);
  assert.deepEqual(reviews[0].payload.changedFiles, []);
  const prompt = String(provider.calls[2].messages.at(-1).content);
  assert.ok(prompt.startsWith('SELF-REVIEW before finish: the worktree has NO changes'), prompt.slice(0, 80));
  assert.ok(prompt.includes('The plan expects changes in: src/math.js'), prompt);
  assert.ok(prompt.includes('If no change is needed, call finish again with a summary that says exactly why.'), prompt);
});

test('a finish in the same reply as a read is answered, not obeyed', async () => {
  // read_file + read_file + finish in one reply means the model wrote finish
  // before it saw a single byte of the files. The reads run, the finish gets
  // a tool_result explaining itself, and the session goes on.
  const { out, events, provider } = await coderWith(
    [
      {
        toolCalls: [
          { name: 'read_file', input: { path: 'src/index.js' } },
          { name: 'read_file', input: { path: 'package.json' } },
          { name: 'finish', input: { summary: 'done' } },
        ],
      },
      WRITE_MATH,
      FINISH,
    ],
  );
  assert.equal(out.reason, 'finish');
  assert.equal(out.turns, 3);
  assert.equal(out.summary, 'wrote src/math.js', 'the summary comes from the finish that was obeyed');
  const results = provider.calls[1].messages.at(-1).content;
  assert.equal(results.length, 3, 'every tool_use still gets its tool_result');
  assert.ok(results[2].content.startsWith('finish ignored: nothing has changed in the worktree'), results[2].content);
  const rejected = events.filter((e) => e.type === 'coder.finish_rejected');
  assert.equal(rejected.length, 1);
  assert.equal(rejected[0].payload.reason, 'with-reads');
  assert.equal(rejected[0].payload.turn, 1);
  assert.equal(rejected[0].payload.count, 1);
  assert.equal(rejected[0].payload.candidateId, 'c0a0');
});

test('a finish riding with a read on top of a real diff is obeyed', async () => {
  // The model already changed the file; the self-review (when on) and the
  // cascade are better judges of that diff than another refusal.
  const { out, events } = await coderWith([
    WRITE_MATH,
    {
      toolCalls: [
        { name: 'run', input: { cmd: 'npm test' } },
        { name: 'finish', input: { summary: 'done' } },
      ],
    },
  ]);
  assert.equal(out.reason, 'finish');
  assert.equal(out.turns, 2);
  assert.equal(events.filter((e) => e.type === 'coder.finish_rejected').length, 0);
});

test('after three rejected finishes the fourth is obeyed', async () => {
  // Live run r_20260909_cb3b87e4: a model that appends finish to every reply
  // must not loop to the turn cap on refusals.
  const premature = {
    toolCalls: [
      { name: 'read_file', input: { path: 'src/index.js' } },
      { name: 'finish', input: { summary: 'done' } },
    ],
  };
  const { out, events } = await coderWith([premature, premature, premature, premature, premature]);
  assert.equal(out.reason, 'finish');
  assert.equal(out.turns, 4);
  const rejected = events.filter((e) => e.type === 'coder.finish_rejected');
  assert.deepEqual(rejected.map((e) => e.payload.count), [1, 2, 3]);
  assert.equal(MAX_FINISH_REJECTIONS, 3);
});

test('a finish in the same reply as a failed edit is answered, not obeyed', async () => {
  const { out, events, provider } = await coderWith(
    [
      {
        toolCalls: [
          { name: 'edit_file', input: { path: 'src/index.js', old_string: 'no such text', new_string: 'x' } },
          { name: 'finish', input: { summary: 'done' } },
        ],
      },
      WRITE_MATH,
      FINISH,
    ],
  );
  assert.equal(out.reason, 'finish');
  assert.equal(out.turns, 3);
  const results = provider.calls[1].messages.at(-1).content;
  assert.ok(results[0].content.startsWith('refused: old_string is not present'), results[0].content);
  assert.ok(results[1].content.startsWith('finish ignored: a tool call in the same reply failed'), results[1].content);
  assert.equal(events.filter((e) => e.type === 'coder.finish_rejected')[0].payload.reason, 'with-failures');
});

test('a finish riding with a successful write still ends the session', async () => {
  // write_file + finish in one reply is fine: the model already knows what it
  // wrote. Only reads, runs and failures make a finish premature.
  const { out, events } = await coderWith([
    {
      toolCalls: [
        { name: 'write_file', input: { path: 'src/math.js', content: 'export const add = (a, b) => a + b;\n' } },
        { name: 'finish', input: { summary: 'done' } },
      ],
    },
  ]);
  assert.equal(out.reason, 'finish');
  assert.equal(out.turns, 1);
  assert.equal(events.filter((e) => e.type === 'coder.finish_rejected').length, 0);
});

test('edit_file and write_file read a double-escaped block as newlines', async () => {
  // Small models on the text protocol send "a\\nb" for a two-line block.
  const { wt, provider } = await coderWith(
    [
      {
        toolCalls: [{
          name: 'edit_file',
          input: {
            path: 'src/index.js',
            old_string: 'export function main() {\\n  return 1;\\n}',
            new_string: 'export function main() {\\n  return 2;\\n}',
          },
        }],
      },
      {
        toolCalls: [{
          name: 'write_file',
          input: { path: 'src/new.js', content: 'const a = 1;\\nconst b = 2;\\nexport { a, b };\\n' },
        }],
      },
      FINISH,
    ],
    { files: { ...BASE_FILES, 'src/index.js': 'export function main() {\n  return 1;\n}\n' } },
  );
  const editResult = provider.calls[1].messages.at(-1).content[0].content;
  assert.ok(editResult.startsWith('edited src/index.js (1 replacement'), editResult);
  assert.ok(editResult.includes('note: old_string matched only after its literal'), editResult);
  assert.equal(await wt.readFile('src/index.js'), 'export function main() {\n  return 2;\n}\n');
  const writeResult = provider.calls[2].messages.at(-1).content[0].content;
  assert.ok(writeResult.includes('note: the content had literal'), writeResult);
  assert.equal(await wt.readFile('src/new.js'), 'const a = 1;\nconst b = 2;\nexport { a, b };\n');
});

test('a one-line write with a single escaped newline is stored as sent', async () => {
  // One "\\n" in a one-line string is not evidence of double escaping.
  const { wt } = await coderWith([
    { toolCalls: [{ name: 'write_file', input: { path: 'src/one.js', content: 'const s = "a\\nb";' } }] },
    FINISH,
  ]);
  assert.equal(await wt.readFile('src/one.js'), 'const s = "a\\nb";');
});

/* =========================================================================
 * 2.3 - tool-result hygiene
 * ========================================================================= */

const BIG_FILE = `${Array.from({ length: 500 }, (_, i) => `line ${i + 1}`).join('\n')}\n`;

test('a whole-file read stops at 400 lines and says how to get the rest', async () => {
  const { provider } = await coderWith(
    [{ toolCalls: [{ name: 'read_file', input: { path: 'src/big.js' } }] }, FINISH],
    { files: { ...BASE_FILES, 'src/big.js': BIG_FILE } },
  );
  const result = provider.calls[1].messages.at(-1).content[0].content;
  const lines = result.split('\n');
  assert.equal(lines[0], 'line 1');
  assert.equal(lines[399], 'line 400');
  assert.equal(lines[400], '[... 100 more lines; pass start_line/end_line]');
  assert.equal(lines.length, 401);
});

test('read_file honours start_line/end_line and marks the window', async () => {
  const { provider } = await coderWith(
    [
      { toolCalls: [{ name: 'read_file', input: { path: 'src/big.js', start_line: 2, end_line: 4 } }] },
      FINISH,
    ],
    { files: { ...BASE_FILES, 'src/big.js': BIG_FILE } },
  );
  const result = provider.calls[1].messages.at(-1).content[0].content;
  assert.equal(result, 'line 2\nline 3\nline 4\n[lines 2-4 of 500; pass start_line/end_line for more]');
});

test('a short whole-file read is still returned byte for byte', async () => {
  const { provider } = await coderWith(
    [{ toolCalls: [{ name: 'read_file', input: { path: 'src/index.js' } }] }, FINISH],
  );
  assert.equal(provider.calls[1].messages.at(-1).content[0].content, "export const name = 'demo';\n");
});

test('search returns at most 40 hits and says the rest were cut', async () => {
  const files = { ...BASE_FILES };
  for (let i = 0; i < 50; i += 1) files[`src/hit-${String(i).padStart(2, '0')}.js`] = 'const needle = 1;\n';
  const { provider } = await coderWith(
    [{ toolCalls: [{ name: 'search', input: { pattern: 'needle' } }] }, FINISH],
    { files },
  );
  const result = provider.calls[1].messages.at(-1).content[0].content;
  const lines = result.split('\n');
  assert.equal(lines.length, 41, 'forty hits plus one marker');
  assert.equal(lines[0], 'src/hit-00.js:1: const needle = 1;');
  assert.equal(lines[40], '[40 hits shown; further hits were cut - narrow the pattern or pass a glob]');
});

test('search under the cap says nothing about cutting', async () => {
  const files = { ...BASE_FILES, 'src/one.js': 'const needle = 1;\n' };
  const { provider } = await coderWith(
    [{ toolCalls: [{ name: 'search', input: { pattern: 'needle' } }] }, FINISH],
    { files },
  );
  const result = provider.calls[1].messages.at(-1).content[0].content;
  assert.equal(result, 'src/one.js:1: const needle = 1;');
});

test('run keeps the head and the tail of a long output, where the summary lives', async () => {
  const stdout = `${'H'.repeat(5000)}${'T'.repeat(5000)}`;
  const { provider } = await coderWith(
    [{ toolCalls: [{ name: 'run', input: { cmd: 'npm test' } }] }, FINISH],
    { exec: { 'npm test': { stdout, exitCode: 0 } } },
  );
  const result = provider.calls[1].messages.at(-1).content[0].content;
  assert.ok(result.startsWith(`exit code: 0\n${'H'.repeat(100)}`), result.slice(0, 60));
  assert.ok(result.includes('[... 1808 bytes omitted from the middle; head and tail kept ...]'), 'marker present');
  assert.ok(result.includes(`${'T'.repeat(100)}`), 'the tail survived');
  assert.ok(Buffer.byteLength(result, 'utf8') < 16 * 1024, 'and it fits the per-result cap');
});

test('every tool result is capped at 16 KB', async () => {
  const huge = `${'a'.repeat(199)}\n`.repeat(100); // 20 000 bytes over 100 lines
  const { provider } = await coderWith(
    [{ toolCalls: [{ name: 'read_file', input: { path: 'src/huge.js' } }] }, FINISH],
    { files: { ...BASE_FILES, 'src/huge.js': huge } },
  );
  const result = provider.calls[1].messages.at(-1).content[0].content;
  assert.ok(result.includes('[truncated '), 'the cap announces itself');
  assert.ok(Buffer.byteLength(result, 'utf8') < 17 * 1024, `result was ${Buffer.byteLength(result, 'utf8')} bytes`);
});

/* =========================================================================
 * 2.4 - steering hints
 * ========================================================================= */

test('hints are rendered in the opening under HINTS FROM THE ORCHESTRATOR', async () => {
  const { provider } = await coderWith([FINISH], {
    hints: ['the failing assertion expects 5400; your UNITS map is missing h', 'do not edit tests/duration.test.js'],
  });
  const opening = String(provider.calls[0].messages[0].content);
  assert.ok(opening.includes('HINTS FROM THE ORCHESTRATOR'), opening.slice(0, 400));
  assert.ok(opening.includes('  - the failing assertion expects 5400; your UNITS map is missing h'), opening);
  assert.ok(opening.includes('  - do not edit tests/duration.test.js'), opening);
  // Hints sit under the task and above everything else, where the coder reads
  // them before it starts opening files.
  assert.ok(opening.indexOf('TASK') < opening.indexOf('HINTS FROM THE ORCHESTRATOR'), opening);
  assert.ok(
    opening.indexOf('HINTS FROM THE ORCHESTRATOR') < opening.indexOf('Start by reading the files'),
    opening,
  );
});

test('at most six hints reach the model, each at most 240 characters', async () => {
  const hints = Array.from({ length: 9 }, (_, i) => `hint ${i} ${'x'.repeat(400)}`);
  const { provider } = await coderWith([FINISH], { hints });
  const opening = String(provider.calls[0].messages[0].content);
  for (let i = 0; i < 6; i += 1) assert.ok(opening.includes(`hint ${i} `), `hint ${i} is missing`);
  for (let i = 6; i < 9; i += 1) assert.ok(!opening.includes(`hint ${i} `), `hint ${i} should have been dropped`);
  const rendered = opening.split('\n').filter((l) => l.startsWith('  - hint '));
  assert.equal(rendered.length, 6);
  for (const line of rendered) assert.equal(line.length, 240 + 4, line.length);
});

test('no hints means no block at all', async () => {
  for (const hints of [undefined, [], ['  ', ''], null]) {
    const { provider } = await coderWith([FINISH], { hints });
    assert.ok(!String(provider.calls[0].messages[0].content).includes('HINTS FROM THE ORCHESTRATOR'));
  }
});

test('runRepair renders the hints under the repair feedback', async () => {
  const driver = createFakeDriver({ files: BASE_FILES });
  const wt = await driver.create({ repoPath: '/repo/demo', runId: 'r_coder', candidateId: 'c0a0' });
  const provider = createFakeProvider({ ref: 'ollama:qwen3:8b', script: [FINISH] });
  const out = await runRepair({
    runId: 'r_coder',
    candidateId: 'c0a0',
    wt,
    plan: PLAN_JSON,
    coder: provider,
    store: stubStore([]),
    config: { repairTurns: 4 },
    feedback: { stage: 'l2-dynamic', reasons: ['1 newly failing test(s): add adds numbers'] },
    hints: ['the UNITS map is missing h'],
  });
  assert.equal(out.reason, 'finish');
  const opening = String(provider.calls[0].messages[0].content);
  assert.ok(opening.startsWith('REPAIR TURN'), opening.slice(0, 60));
  assert.ok(opening.includes('HINTS FROM THE ORCHESTRATOR'), opening);
  assert.ok(opening.includes('  - the UNITS map is missing h'), opening);
  assert.ok(opening.indexOf('REPAIR TURN') < opening.indexOf('HINTS FROM THE ORCHESTRATOR'), 'feedback comes first');
});

/* =========================================================================
 * 4.3 - a coder that describes its tool calls in prose
 * ========================================================================= */

// Live run r_20260909_b4d9777c (bench task fix-duration-units, coder
// ollama:mistral:latest): every turn read "To fix the seconds divisor in
// formatDuration, I will use the `edit_file` function ... Here's the command to
// do that:" followed by a fenced `edit_file src/duration.js ...` line, and the
// recorded events were `coder.turn ... toolCalls=[]` twice, then
// `candidate.diff ... changedFiles=0 turns=2 finished=false reason=no-tool-calls`.
const PROSE_TURN = {
  text: 'To fix the seconds divisor in formatDuration, I will use the `edit_file` function.\n'
    + "Here's the command to do that:\n\nedit_file src/index.js \"demo\" \"demo2\"",
};

const EDIT_INDEX = {
  toolCalls: [{
    name: 'edit_file',
    input: { path: 'src/index.js', old_string: "'demo'", new_string: "'demo2'" },
  }],
};

test('a coder that answers prose is moved to the text tool protocol for the rest of the session', async () => {
  const asked = [];
  const script = (call) => {
    asked.push(call.textTools === true);
    if (call.textTools !== true) return PROSE_TURN; // native tools: prose, every time
    return asked.length === 2 ? EDIT_INDEX : FINISH;
  };
  const { out, events, provider, wt } = await coderWith(script, { maxTurns: 6 });

  assert.equal(out.reason, 'finish');
  assert.equal(out.finished, true);
  assert.equal(out.protocol, 'text');
  assert.ok(out.diff.trim() !== '', 'the text protocol turn produced a real diff');
  assert.equal(await wt.readFile('src/index.js'), "export const name = 'demo2';\n");
  assert.deepEqual(asked, [false, true, true], 'the switch holds for every later turn');

  const switched = events.filter((e) => e.type === 'coder.protocol');
  assert.equal(switched.length, 1, 'the protocol switch is announced exactly once');
  assert.deepEqual(switched[0].payload, {
    candidateId: 'c0a0',
    turn: 1,
    protocol: 'text',
    why: 'no-tool-calls',
  });

  // The nudge that goes with the switch teaches the format instead of repeating
  // "you made no tool call", which is what failed on the live run.
  const nudge = String(provider.calls[1].messages.at(-1).content);
  assert.ok(nudge.includes('<<<tool'), nudge);
  assert.ok(nudge.includes('<<<end>>>'), nudge);
  assert.ok(nudge.startsWith('You made no tool call; the tool call you described was never executed.'), nudge);
});

test('a session that has already used a tool keeps the plain nudge and the native protocol', async () => {
  const asked = [];
  const script = (call, index) => {
    asked.push(call.textTools === true);
    return index === 0 ? WRITE_MATH : PROSE_TURN;
  };
  const { out, events, provider } = await coderWith(script, { maxTurns: 6 });

  assert.equal(out.reason, 'no-tool-calls');
  assert.equal(out.finished, false);
  assert.equal(out.protocol, 'native');
  assert.equal(events.filter((e) => e.type === 'coder.protocol').length, 0, 'nothing to switch');
  assert.deepEqual(asked, [false, false, false], 'the native tool API is never given up');

  const nudge = String(provider.calls[2].messages.at(-1).content);
  assert.ok(nudge.startsWith('You made no tool call. If the change is complete'), nudge);
  assert.ok(!nudge.includes('<<<tool'), nudge);
});

test('a coder already speaking the text protocol is not switched again', async () => {
  const inner = createFakeProvider({ ref: 'ollama:deepseek-coder-v2', script: [PROSE_TURN, PROSE_TURN] });
  // What an adapter latched to the text protocol reports on every reply.
  const provider = {
    ...inner,
    async chat(call) {
      const reply = await inner.chat(call);
      return { ...reply, meta: { textTools: true } };
    },
  };
  const { out, events } = await coderWith(null, { provider, maxTurns: 6 });

  assert.equal(out.reason, 'no-tool-calls');
  assert.equal(out.protocol, 'text', 'the session was in text mode from the first turn');
  assert.equal(events.filter((e) => e.type === 'coder.protocol').length, 0, 'already there: no switch event');
  const nudge = String(inner.calls[1].messages.at(-1).content);
  assert.ok(nudge.startsWith('You made no tool call. If the change is complete'), nudge);
});

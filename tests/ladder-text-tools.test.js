// kolm code - text tool protocol tests (owner PROV, spec 3.1).
//
// No network, no timers: everything here is pure string work.

import test from 'node:test';
import assert from 'node:assert/strict';

import {
  TEXT_TOOL_PROTOCOL_VERSION,
  looksLikeToolsUnsupported,
  parseLenientJson,
  parseTextToolCalls,
  refForcedToTextTools,
  renderJsonOnlyForText,
  renderToolResultsForText,
  renderToolsForText,
  toTextProtocolMessages,
} from '../src/ladder/providers/text-tools.js';

const TOOLS = [
  {
    name: 'read_file',
    description: 'Read a file from the candidate worktree.',
    input_schema: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] },
  },
  { name: 'finish', description: 'End the session.', input_schema: { type: 'object' } },
];

// ===========================================================================
// parsing
// ===========================================================================

test('a clean block parses into one tool call and leaves no text', () => {
  const r = parseTextToolCalls('<<<tool name="read_file">>>\n{"path": "a.js"}\n<<<end>>>');
  assert.equal(r.text, '');
  assert.equal(r.errors.length, 0);
  assert.deepEqual(r.toolCalls, [{ id: 'text_1', name: 'read_file', input: { path: 'a.js' } }]);
});

test('several blocks get ids text_1.. in reply order', () => {
  const raw = [
    'first I look:',
    '<<<tool name="read_file">>>{"path":"a.js"}<<<end>>>',
    'then I write:',
    '<<<tool name=\'write_file\'>>>{"path":"b.js","content":"x"}<<<end>>>',
    '<<<tool name=finish>>>{}<<<end>>>',
  ].join('\n');
  const r = parseTextToolCalls(raw);
  assert.deepEqual(
    r.toolCalls.map((c) => [c.id, c.name]),
    [
      ['text_1', 'read_file'],
      ['text_2', 'write_file'],
      ['text_3', 'finish'],
    ],
  );
  assert.deepEqual(r.toolCalls[1].input, { path: 'b.js', content: 'x' });
  assert.equal(r.text, 'first I look:\n\nthen I write:');
});

test('a block wrapped in a code fence still parses, fence and all', () => {
  const raw = [
    'here you go',
    '',
    '```json',
    '<<<tool name="read_file">>>',
    '{"path": "src/index.js"}',
    '<<<end>>>',
    '```',
  ].join('\n');
  const r = parseTextToolCalls(raw);
  assert.deepEqual(r.toolCalls, [
    { id: 'text_1', name: 'read_file', input: { path: 'src/index.js' } },
  ]);
  assert.equal(r.text, 'here you go');
  assert.equal(r.text.includes('```'), false);
});

test('a fence inside the block body is stripped from the arguments', () => {
  const raw = '<<<tool name="read_file">>>\n```json\n{"path":"a.js"}\n```\n<<<end>>>';
  assert.deepEqual(parseTextToolCalls(raw).toolCalls[0].input, { path: 'a.js' });
});

test('trailing prose after a block is kept as assistant text', () => {
  const r = parseTextToolCalls(
    '<<<tool name="read_file">>>{"path":"a.js"}<<<end>>>\nI will patch it next.',
  );
  assert.equal(r.toolCalls.length, 1);
  assert.equal(r.text, 'I will patch it next.');
});

test('CRLF transports parse and the returned text is normalized to LF', () => {
  const raw = 'looking\r\n\r\n<<<tool name="read_file">>>\r\n{"path": "a.js"}\r\n<<<end>>>\r\nnow patching\r\n';
  const r = parseTextToolCalls(raw);
  assert.deepEqual(r.toolCalls[0].input, { path: 'a.js' });
  assert.equal(r.text, 'looking\n\nnow patching');
  assert.equal(r.text.includes('\r'), false);
});

test('broken JSON becomes a tool call carrying __parse_error and the raw body', () => {
  const r = parseTextToolCalls('<<<tool name="write_file">>>\n{"path": "a.js", oops}\n<<<end>>>');
  assert.equal(r.toolCalls.length, 1);
  assert.equal(r.toolCalls[0].name, 'write_file');
  assert.equal(r.toolCalls[0].input.__parse_error, 'arguments are not valid JSON');
  assert.equal(r.toolCalls[0].input.raw, '{"path": "a.js", oops}');
  assert.deepEqual(r.errors, ['tool "write_file": arguments are not valid JSON']);
});

test('a trailing comma is repaired rather than reported', () => {
  const r = parseTextToolCalls('<<<tool name="read_file">>>{"path":"a.js",}<<<end>>>');
  assert.deepEqual(r.toolCalls[0].input, { path: 'a.js' });
  assert.equal(r.errors.length, 0);
});

test('a JSON array of arguments is a parse error, not a silent object', () => {
  const r = parseTextToolCalls('<<<tool name="read_file">>>[1,2]<<<end>>>');
  assert.equal(r.toolCalls[0].input.__parse_error, 'arguments must be a JSON object');
  assert.equal(r.errors.length, 1);
});

test('an empty body is a legal no-argument call', () => {
  const r = parseTextToolCalls('<<<tool name="finish">>>\n<<<end>>>');
  assert.deepEqual(r.toolCalls, [{ id: 'text_1', name: 'finish', input: {} }]);
  assert.equal(r.errors.length, 0);
});

test('a nameless block is reported instead of being dropped', () => {
  const r = parseTextToolCalls('<<<tool>>>{"path":"a.js"}<<<end>>>');
  assert.equal(r.toolCalls[0].name, '');
  assert.equal(r.toolCalls[0].input.__parse_error, 'tool block has no name');
  assert.deepEqual(r.errors, ['tool block text_1 has no name']);
});

test('marker whitespace and alternative closers are tolerated', () => {
  const r = parseTextToolCalls('<<< tool name = "read_file" >>>{"path":"a.js"}<<< /end >>>');
  assert.deepEqual(r.toolCalls[0], { id: 'text_1', name: 'read_file', input: { path: 'a.js' } });
});

test('a block truncated by max_tokens is still recovered', () => {
  const r = parseTextToolCalls('working\n<<<tool name="read_file">>>\n{"path": "a.js"}');
  assert.deepEqual(r.toolCalls[0], { id: 'text_1', name: 'read_file', input: { path: 'a.js' } });
  assert.equal(r.text, 'working');
});

test('a reply with no block is all text', () => {
  const r = parseTextToolCalls('nothing to do here');
  assert.deepEqual(r.toolCalls, []);
  assert.equal(r.text, 'nothing to do here');
  assert.equal(parseTextToolCalls(null).text, '');
});

// ===========================================================================
// parseLenientJson - the slips a small model actually makes
//
// Source-string note: '\n' below is a REAL U+000A inside the block text (the
// slip), while '\\n' is the two characters backslash-n (a correct escape).
// ===========================================================================

test('a raw newline inside a string value parses and yields a real newline', () => {
  const raw = '{"path":"a.js","old_string":"\n  let total = 0;\n"}';
  const r = parseLenientJson(raw);
  assert.equal(r.ok, true);
  assert.equal(r.repaired, true);
  assert.equal(r.value.path, 'a.js');
  assert.equal(r.value.old_string, '\n  let total = 0;\n');
  assert.equal(r.value.old_string.charCodeAt(0), 10);
});

test('a bare backslash-d from a regex literal parses to backslash-d', () => {
  const r = parseLenientJson('{"re":"/(\\d+)([hms])/g"}');
  assert.equal(r.ok, true);
  assert.equal(r.repaired, true);
  assert.equal(r.value.re, '/(\\d+)([hms])/g');
  assert.equal(r.value.re.charCodeAt(2), 92); // "/(" then a real backslash, then "d"
  assert.equal(r.value.re.charAt(3), 'd');
});

test('the live llama3 block - raw newlines plus backslash-d - parses in one go', () => {
  // Reproduces the probe that failed with "arguments are not valid JSON".
  const body =
    '{"path":"src/duration.js","old_string":"\n  let total = 0;\n' +
    '  const re = /(\\d+)([hms])/g;\n","new_string":"\n  return 0;\n"}';
  const r = parseTextToolCalls(`<<<tool name="edit_file">>>\n${body}\n<<<end>>>`);
  assert.equal(r.errors.length, 0);
  assert.equal(r.repaired, 1);
  assert.equal(r.toolCalls[0].name, 'edit_file');
  assert.equal(r.toolCalls[0].input.path, 'src/duration.js');
  assert.equal(
    r.toolCalls[0].input.old_string,
    '\n  let total = 0;\n  const re = /(\\d+)([hms])/g;\n',
  );
  assert.equal(r.toolCalls[0].input.new_string, '\n  return 0;\n');
});

test('a valid block is byte-identical after parsing and is not marked repaired', () => {
  // "real" carries an escaped newline; "literal" carries the TWO characters
  // backslash-n, which must survive as two characters (deepseek-coder-v2 style).
  const valid = '{"real":"a\\nb","literal":"a\\\\nb"}';
  const r = parseLenientJson(valid);
  assert.equal(r.ok, true);
  assert.equal(r.repaired, false);
  assert.equal(r.value.real, 'a\nb');
  assert.equal(r.value.real.length, 3);
  assert.equal(r.value.literal, 'a\\nb');
  assert.equal(r.value.literal.length, 4);
  assert.equal(r.value.literal.charCodeAt(1), 92);
  assert.equal(r.value.literal.charAt(2), 'n');
  // Round-trips to the exact same bytes: the repair never ran.
  assert.equal(JSON.stringify(r.value), valid);
  const viaBlock = parseTextToolCalls(`<<<tool name="write_file">>>${valid}<<<end>>>`);
  assert.equal(viaBlock.repaired, 0);
  assert.equal(viaBlock.errors.length, 0);
  assert.deepEqual(viaBlock.toolCalls[0].input, r.value);
});

test('a code fence around the arguments is stripped before parsing', () => {
  const fenced = parseLenientJson('```json\n{"path":"a.js"}\n```');
  assert.equal(fenced.ok, true);
  assert.equal(fenced.repaired, true);
  assert.deepEqual(fenced.value, { path: 'a.js' });
  const bare = parseLenientJson('```\n{"path":"a.js"}\n```');
  assert.equal(bare.ok, true);
  assert.deepEqual(bare.value, { path: 'a.js' });
});

test('prose before and after the object is trimmed away', () => {
  const r = parseLenientJson('Sure, here you go:\n{"path":"a.js"}\nLet me know.');
  assert.equal(r.ok, true);
  assert.equal(r.repaired, true);
  assert.deepEqual(r.value, { path: 'a.js' });
});

test('a genuinely broken object reports the original parse error', () => {
  const broken = '{"path": "a.js"';
  let expected = '';
  try {
    JSON.parse(broken);
  } catch (err) {
    expected = err.message;
  }
  const r = parseLenientJson(broken);
  assert.equal(r.ok, false);
  assert.equal(r.error, expected);
  assert.equal(r.repaired, undefined);
  // ... and through the block parser it is still the same __parse_error as before.
  const viaBlock = parseTextToolCalls(`<<<tool name="write_file">>>${broken}<<<end>>>`);
  assert.equal(viaBlock.toolCalls[0].input.__parse_error, 'arguments are not valid JSON');
  assert.equal(viaBlock.toolCalls[0].input.raw, broken);
  assert.deepEqual(viaBlock.errors, ['tool "write_file": arguments are not valid JSON']);
  assert.equal(viaBlock.repaired, 0);
});

test('a raw newline outside a string is left alone because JSON allows it', () => {
  const between = parseLenientJson('{"a":1,\n"b":2}');
  assert.equal(between.ok, true);
  assert.equal(between.repaired, false); // it parsed as written; nothing was touched
  assert.deepEqual(between.value, { a: 1, b: 2 });
  // Mixed: the in-string newline is escaped, the between-keys newline is not.
  const mixed = parseLenientJson('{"a":"x\ny",\n"b":2}');
  assert.equal(mixed.ok, true);
  assert.equal(mixed.repaired, true);
  assert.deepEqual(mixed.value, { a: 'x\ny', b: 2 });
});

test('a backslash-u escape with four hex digits is preserved by the repair', () => {
  const r = parseLenientJson('{"a":"\\u0041\nz"}');
  assert.equal(r.ok, true);
  assert.equal(r.repaired, true);
  assert.equal(r.value.a, 'A\nz');
  // A backslash-u WITHOUT four hex digits is kept literally instead.
  const bad = parseLenientJson('{"a":"\\u12 tail\nx"}');
  assert.equal(bad.ok, true);
  assert.equal(bad.value.a, '\\u12 tail\nx');
});

test('raw control characters inside a string are escaped, not dropped', () => {
  const r = parseLenientJson('{"a":"x\u0001y"}');
  assert.equal(r.ok, true);
  assert.equal(r.repaired, true);
  assert.equal(r.value.a, 'x\u0001y');
  assert.equal(r.value.a.charCodeAt(1), 1);
  assert.equal(JSON.stringify(r.value.a).includes('\\u0001'), true);
  // Tab and carriage return get their short escapes.
  const tabbed = parseLenientJson('{"a":"x\ty\rz"}');
  assert.equal(tabbed.ok, true);
  assert.equal(tabbed.value.a, 'x\ty\rz');
});

test('parseTextToolCalls counts repaired blocks without reporting them as errors', () => {
  const raw = [
    '<<<tool name="write_file">>>{"path":"a.js","body":"one\ntwo"}<<<end>>>',
    '<<<tool name="read_file">>>{"path":"b.js"}<<<end>>>',
  ].join('\n');
  const r = parseTextToolCalls(raw);
  assert.equal(r.repaired, 1); // only the first block needed rescuing
  assert.equal(r.errors.length, 0);
  assert.equal(r.toolCalls.length, 2);
  assert.equal(r.toolCalls[0].input.body, 'one\ntwo');
  assert.deepEqual(r.toolCalls[1].input, { path: 'b.js' });
  // The field is additive: the three original keys are unchanged.
  assert.equal(typeof r.text, 'string');
  assert.equal(Array.isArray(r.toolCalls), true);
  assert.equal(Array.isArray(r.errors), true);
  assert.equal(parseTextToolCalls('nothing to do here').repaired, 0);
  assert.equal(parseTextToolCalls(null).repaired, 0);
});

test('parseLenientJson tolerates non-string input instead of throwing', () => {
  const r = parseLenientJson(undefined);
  assert.equal(r.ok, false);
  assert.equal(typeof r.error, 'string');
});

// ===========================================================================
// rendering
// ===========================================================================

test('renderToolsForText teaches the exact wire format and lists every tool', () => {
  const rendered = renderToolsForText(TOOLS);
  assert.match(rendered, /^TOOL CALLING PROTOCOL \(text mode, v1\)/);
  assert.equal(rendered.includes('<<<tool name="TOOL_NAME">>>'), true);
  assert.equal(rendered.includes('<<<end>>>'), true);
  assert.equal(rendered.includes('1. read_file - Read a file from the candidate worktree.'), true);
  assert.equal(rendered.includes('2. finish - End the session.'), true);
  assert.equal(rendered.includes('"required":["path"]'), true);
  // ASCII only: this string goes straight into a system prompt.
  assert.equal(/^[\x09\x0a\x20-\x7e]*$/.test(rendered), true);
});

test('the rendered protocol block round-trips through the parser', () => {
  // The example in the prompt must itself be a parseable block, otherwise the
  // model is being taught a format the adapter would reject.
  const rendered = renderToolsForText(TOOLS);
  const from = rendered.indexOf('<<<tool');
  const to = rendered.indexOf('<<<end>>>') + '<<<end>>>'.length;
  const example = rendered.slice(from, to);
  const r = parseTextToolCalls(example);
  assert.equal(r.toolCalls.length, 1);
  assert.equal(r.toolCalls[0].name, 'TOOL_NAME');
  assert.deepEqual(r.toolCalls[0].input, { first_argument: 'value' });
});

test('renderToolsForText says so when a turn has no tools', () => {
  assert.equal(renderToolsForText([]).includes('(none available on this turn)'), true);
});

test('renderToolResultsForText labels each result and prompts the next move', () => {
  const rendered = renderToolResultsForText([
    { type: 'tool_result', tool_use_id: 'text_1', name: 'read_file', content: 'file body' },
    { type: 'tool_result', tool_use_id: 'text_2', name: 'run', content: 'boom', is_error: true },
    { type: 'text', text: 'ignored' },
  ]);
  assert.match(rendered, /^TOOL RESULTS/);
  assert.equal(rendered.includes('<<<result id="text_1" tool="read_file">>>'), true);
  assert.equal(rendered.includes('file body'), true);
  assert.equal(rendered.includes('<<<result id="text_2" tool="run" status="error">>>'), true);
  assert.equal(rendered.includes('ignored'), false);
  assert.match(rendered, /Continue: write the next <<<tool \.\.\.>>> block, or give your answer\.$/);
  assert.equal(renderToolResultsForText([]), '');
});

test('renderJsonOnlyForText inlines the schema and forbids prose', () => {
  const rendered = renderJsonOnlyForText({ name: 'verdict', schema: { type: 'object' } });
  assert.match(rendered, /^OUTPUT FORMAT/);
  assert.equal(rendered.includes('{"type":"object"}'), true);
  assert.equal(rendered.includes('no backticks'), true);
});

test('toTextProtocolMessages flattens tool_use and tool_result out of the history', () => {
  const out = toTextProtocolMessages([
    { role: 'user', content: 'fix it' },
    {
      role: 'assistant',
      content: [
        { type: 'text', text: 'reading' },
        { type: 'tool_use', id: 'text_1', name: 'read_file', input: { path: 'a.js' } },
      ],
    },
    { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'text_1', content: 'body' }] },
  ]);
  assert.deepEqual(out[0], { role: 'user', content: 'fix it' });
  assert.equal(out[1].role, 'assistant');
  assert.equal(typeof out[1].content, 'string');
  assert.equal(
    out[1].content,
    'reading\n\n<<<tool name="read_file">>>\n{"path":"a.js"}\n<<<end>>>',
  );
  // The result names the tool even though the block itself did not carry a name.
  assert.equal(out[2].content.includes('<<<result id="text_1" tool="read_file">>>'), true);
  for (const message of out) assert.equal(typeof message.content, 'string');
});

test('an assistant turn rendered to text re-parses to the same call', () => {
  const out = toTextProtocolMessages([
    {
      role: 'assistant',
      content: [
        {
          type: 'tool_use',
          id: 'text_1',
          name: 'write_file',
          input: { path: 'a.js', content: 'line1\nline2\t"q"' },
        },
      ],
    },
  ]);
  const r = parseTextToolCalls(out[0].content);
  assert.deepEqual(r.toolCalls[0].input, { path: 'a.js', content: 'line1\nline2\t"q"' });
});

// ===========================================================================
// selection helpers
// ===========================================================================

test('refForcedToTextTools matches exact refs and provider wildcards', () => {
  assert.equal(refForcedToTextTools('ollama:deepseek-coder-v2', ['ollama:deepseek-coder-v2']), true);
  assert.equal(refForcedToTextTools('OLLAMA:Deepseek-Coder-V2', ['ollama:deepseek-coder-v2']), true);
  assert.equal(refForcedToTextTools('openrouter:x/y', ['openrouter:*']), true);
  assert.equal(refForcedToTextTools('openrouter:x/y', ['ollama:*']), false);
  assert.equal(refForcedToTextTools('openrouter:x/y', []), false);
  assert.equal(refForcedToTextTools('openrouter:x/y', undefined), false);
});

test('looksLikeToolsUnsupported recognizes both 400 wordings and nothing else', () => {
  assert.equal(
    looksLikeToolsUnsupported({ status: 400, body: 'registry.Model "x" does not support tools' }),
    true,
  );
  assert.equal(looksLikeToolsUnsupported({ status: 400, body: 'tools are not supported' }), true);
  assert.equal(looksLikeToolsUnsupported({ status: 400, body: 'tool_choice is not supported' }), true);
  assert.equal(
    looksLikeToolsUnsupported({ detail: { status: 400, body: { error: 'tools are not supported' } } }),
    true,
  );
  assert.equal(looksLikeToolsUnsupported({ status: 400, body: 'strict schemas are not supported' }), false);
  assert.equal(looksLikeToolsUnsupported({ status: 500, body: 'does not support tools' }), false);
  assert.equal(looksLikeToolsUnsupported(null), false);
});

test('the protocol version is pinned', () => {
  assert.equal(TEXT_TOOL_PROTOCOL_VERSION, 1);
});

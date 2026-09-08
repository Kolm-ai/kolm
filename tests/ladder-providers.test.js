// kolm code - tests for src/ladder/providers/* (owner A).
//
// Fully offline: the Anthropic client and `fetch` are injected, so no test
// needs a key, a network or a running Ollama. No timing assertions.

import test from 'node:test';
import assert from 'node:assert/strict';

import { isLadderError } from '../src/ladder/errors.js';
import {
  MODEL_CATALOG,
  ROLES,
  knownRefs,
  listModels,
  describeModel,
  priceFor,
  rolesFor,
  estimateCost,
} from '../src/ladder/providers/registry.js';
import {
  createFakeProvider,
  createMockProvider,
  minimalFromSchema,
} from '../src/ladder/providers/fake.js';
import {
  createAnthropicProvider,
  mapStopReason,
  parseJsonLoose,
} from '../src/ladder/providers/anthropic.js';
import {
  createOpenAICompatibleProvider,
  OPENAI_COMPATIBLE_PROVIDERS,
  mapFinishReason,
  stripThinking,
  toOpenAIMessages,
  toOpenAITools,
} from '../src/ladder/providers/openai-compatible.js';
import {
  createProvider,
  resolveRef,
  PROVIDER_KINDS,
  SUPPORTED_PROVIDERS,
} from '../src/ladder/providers/index.js';

// --- helpers ---------------------------------------------------------------

/** A config shaped like loadConfig()'s output, without touching the filesystem. */
function cfg(secrets = {}, extra = {}) {
  return {
    secrets: {
      anthropic: null,
      cerebras: null,
      deepseek: null,
      openrouter: null,
      ollama: null,
      ...secrets,
    },
    baseUrls: {
      cerebras: 'https://api.cerebras.ai/v1',
      deepseek: 'https://api.deepseek.com/v1',
      openrouter: 'https://openrouter.ai/api/v1',
      ollama: 'http://127.0.0.1:11434/v1',
    },
    hasKey: {
      anthropic: Boolean(secrets.anthropic),
      cerebras: Boolean(secrets.cerebras),
      deepseek: Boolean(secrets.deepseek),
      openrouter: Boolean(secrets.openrouter),
    },
    ...extra,
  };
}

/** Anthropic SDK stand-in. `handler(params, index)` returns a message or throws. */
function fakeAnthropicClient(handler) {
  const calls = [];
  return {
    calls,
    messages: {
      async create(params, options) {
        calls.push({ mode: 'create', params, options });
        return handler(params, calls.length - 1);
      },
      stream(params, options) {
        calls.push({ mode: 'stream', params, options });
        const promise = Promise.resolve().then(() => handler(params, calls.length - 1));
        return { finalMessage: () => promise };
      },
    },
  };
}

function anthropicMessage(over = {}) {
  return {
    id: 'msg_1',
    model: 'claude-fable-5-1',
    stop_reason: 'end_turn',
    content: [{ type: 'text', text: 'hello' }],
    usage: { input_tokens: 1000, output_tokens: 500, cache_read_input_tokens: 2000 },
    ...over,
  };
}

/** `fetch` stand-in returning queued responses and recording every request. */
function fakeFetch(queue) {
  const requests = [];
  const responses = Array.isArray(queue) ? [...queue] : [queue];
  const impl = async (url, init) => {
    requests.push({ url, init, body: init?.body ? JSON.parse(init.body) : null });
    const next = responses.length > 1 ? responses.shift() : responses[0];
    if (typeof next === 'function') return next(url, init, requests.length - 1);
    return next;
  };
  impl.requests = requests;
  return impl;
}

function okResponse(payload) {
  return { status: 200, json: async () => payload, text: async () => JSON.stringify(payload) };
}

function errResponse(status, body = 'boom') {
  return { status, json: async () => ({ error: body }), text: async () => body };
}

function chatCompletion(over = {}) {
  return {
    id: 'cmpl_1',
    model: 'deepseek-v4-flash',
    choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', content: 'hi' } }],
    usage: { prompt_tokens: 1000, completion_tokens: 500 },
    ...over,
  };
}

const noSleep = async () => {};
const fixedRandom = () => 0.5;

// ===========================================================================
// registry
// ===========================================================================

test('the catalog seeds every ref named in the spec', () => {
  for (const ref of [
    'anthropic:claude-fable-5-1',
    'anthropic:claude-opus-5',
    'anthropic:claude-sonnet-5',
    'anthropic:claude-haiku-4-5',
    'deepseek:deepseek-v4-flash',
    'cerebras:qwen-3.8-27b',
  ]) {
    assert.ok(knownRefs().includes(ref), `catalog is missing ${ref}`);
  }
  assert.ok(Object.isFrozen(MODEL_CATALOG));
  assert.deepEqual([...ROLES], ['planner', 'judge', 'coder', 'reviewer']);
  assert.equal(listModels().length, knownRefs().length);
});

test('describeModel returns the sampling and pricing facts an adapter needs', () => {
  const fable = describeModel('anthropic:claude-fable-5-1');
  assert.equal(fable.provider, 'anthropic');
  assert.equal(fable.model, 'claude-fable-5-1');
  assert.equal(fable.priceIn, 10);
  assert.equal(fable.priceOut, 50);
  assert.equal(fable.cacheRead, 0.25);
  assert.equal(fable.cacheWrite, 12.5); // 1.25x the input price
  assert.equal(fable.acceptsSampling, false);
  assert.equal(fable.unknownPricing, false);
  assert.ok(fable.roles.includes('planner'));

  // Only Haiku 4.5 accepts sampling parameters in the seeded Anthropic set.
  assert.equal(describeModel('anthropic:claude-opus-5').acceptsSampling, false);
  assert.equal(describeModel('anthropic:claude-sonnet-5').acceptsSampling, false);
  assert.equal(describeModel('anthropic:claude-haiku-4-5').acceptsSampling, true);
});

test('describeModel handles the ollama, fake and unknown wildcards', () => {
  const local = describeModel('ollama:qwen3.8:latest');
  assert.equal(local.model, 'qwen3.8:latest');
  assert.equal(local.priceIn, 0);
  assert.equal(local.priceOut, 0);
  assert.equal(local.unknownPricing, false);
  assert.deepEqual([...local.roles], ['coder', 'reviewer']);

  const fake = describeModel('fake:whatever');
  assert.equal(fake.priceIn, 0);
  assert.ok(fake.roles.includes('judge'));

  const unknown = describeModel('mistral:big');
  assert.equal(unknown.unknownPricing, true);
  assert.equal(unknown.priceIn, 0);
});

test('estimateCost is exact for the seeded prices', () => {
  assert.equal(
    estimateCost('anthropic:claude-fable-5-1', { input: 1000, output: 500, cacheRead: 2000 }),
    0.0355,
  );
  assert.equal(estimateCost('anthropic:claude-opus-5', { input: 1e6, output: 1e6 }), 30);
  assert.equal(estimateCost('ollama:qwen3.8:latest', { input: 1e6, output: 1e6 }), 0);
  assert.equal(estimateCost('anthropic:claude-fable-5-1', {}), 0);
});

test('the DeepSeek off-peak schedule halves the peak prices', () => {
  assert.deepEqual(priceFor('deepseek:deepseek-v4-flash'), {
    priceIn: 0.44,
    priceOut: 1.32,
    cacheRead: 0.044,
    cacheWrite: 0,
  });
  assert.deepEqual(priceFor('deepseek:deepseek-v4-flash', { offpeak: true }), {
    priceIn: 0.22,
    priceOut: 0.66,
    cacheRead: 0.022,
    cacheWrite: 0,
  });
  const peak = estimateCost('deepseek:deepseek-v4-flash', { input: 1e6, output: 1e6 });
  const off = estimateCost('deepseek:deepseek-v4-flash', { input: 1e6, output: 1e6 }, { offpeak: true });
  assert.equal(peak, 1.76);
  assert.equal(off, 0.88);
  // A model with no off-peak schedule ignores the flag.
  assert.deepEqual(
    priceFor('anthropic:claude-fable-5-1', { offpeak: true }),
    priceFor('anthropic:claude-fable-5-1'),
  );
});

test('rolesFor reports the ladder roles and estimateCost rejects bad usage', () => {
  assert.ok(rolesFor('anthropic:claude-haiku-4-5').includes('reviewer'));
  assert.throws(
    () => estimateCost('anthropic:claude-fable-5-1', { input: -1 }),
    (e) => isLadderError(e, 'bad-config'),
  );
  assert.throws(
    () => estimateCost('anthropic:claude-fable-5-1', { output: 'lots' }),
    (e) => isLadderError(e, 'bad-config'),
  );
});

// ===========================================================================
// fake provider
// ===========================================================================

test('the fake provider answers text and records every call', async () => {
  const p = createFakeProvider({ script: ['first', { text: 'second' }] });
  assert.equal(p.id, 'fake:test');
  assert.equal(p.armed, true);
  assert.equal(p.kind, 'fake');

  const a = await p.chat({ system: 'sys', messages: [{ role: 'user', content: 'hi' }], effort: 'high' });
  assert.equal(a.text, 'first');
  assert.equal(a.stopReason, 'end_turn');
  assert.deepEqual(a.content, [{ type: 'text', text: 'first' }]);
  assert.deepEqual(a.usage, { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });
  assert.equal(a.costUsd, 0);

  const b = await p.chat({ system: 'sys2' });
  assert.equal(b.text, 'second');

  assert.equal(p.calls.length, 2);
  assert.equal(p.calls[0].system, 'sys');
  assert.equal(p.calls[0].effort, 'high');
  assert.deepEqual(p.calls[0].messages, [{ role: 'user', content: 'hi' }]);
  p.reset();
  assert.equal(p.calls.length, 0);
});

test('recorded calls are deep copies, immune to later mutation', async () => {
  const p = createFakeProvider();
  const messages = [{ role: 'user', content: 'original' }];
  await p.chat({ messages });
  messages[0].content = 'mutated';
  assert.equal(p.calls[0].messages[0].content, 'original');
});

test('an exhausted array script fails loudly', async () => {
  const p = createFakeProvider({ script: ['only'] });
  await p.chat({});
  await assert.rejects(() => p.chat({}), /exhausted/);
});

test('a function script sees the call and its index', async () => {
  const seen = [];
  const p = createFakeProvider({
    script: (call, index) => {
      seen.push(index);
      return { text: `turn-${index}-${call.system}` };
    },
  });
  assert.equal((await p.chat({ system: 'a' })).text, 'turn-0-a');
  assert.equal((await p.chat({ system: 'b' })).text, 'turn-1-b');
  assert.deepEqual(seen, [0, 1]);
});

test('the fake provider scripts json, tool calls, refusals and errors', async () => {
  const p = createFakeProvider({
    script: [
      { json: { ok: true } },
      { toolCalls: [{ name: 'write_file', input: { path: 'a.js' } }, { name: 'run', input: {} }] },
      { stopReason: 'refusal' },
      { content: [{ type: 'text', text: 'verbatim' }] },
      { error: Object.assign(new Error('scripted failure'), { status: 500 }) },
    ],
  });

  const json = await p.chat({});
  assert.deepEqual(json.json, { ok: true });
  assert.equal(json.text, '{"ok":true}');

  const tools = await p.chat({});
  assert.equal(tools.stopReason, 'tool_use');
  assert.equal(tools.toolCalls.length, 2);
  assert.equal(tools.toolCalls[0].name, 'write_file');
  assert.deepEqual(tools.toolCalls[0].input, { path: 'a.js' });
  assert.equal(tools.toolCalls[0].id, 'toolu_1');
  assert.equal(tools.toolCalls[1].id, 'toolu_2');
  assert.equal(tools.content[0].type, 'tool_use');

  assert.equal((await p.chat({})).stopReason, 'refusal');
  assert.equal((await p.chat({})).text, 'verbatim');
  await assert.rejects(() => p.chat({}), /scripted failure/);
});

test('an unarmed fake provider throws not-armed naming its env var', async () => {
  const p = createFakeProvider({ ref: 'fake:locked', armed: false, envVar: 'FAKE_KEY' });
  assert.equal(p.armed, false);
  await assert.rejects(
    () => p.chat({}),
    (e) => isLadderError(e, 'not-armed') && e.detail.envVar === 'FAKE_KEY',
  );
});

test('the fake provider reports scripted usage and cost', async () => {
  const p = createFakeProvider({ usage: { input: 10, output: 20 }, costUsd: 0.5 });
  const r = await p.chat({});
  assert.deepEqual(r.usage, { input: 10, output: 20, cacheRead: 0, cacheWrite: 0 });
  assert.equal(r.costUsd, 0.5);
});

test('minimalFromSchema builds a deterministic minimal instance', () => {
  assert.deepEqual(
    minimalFromSchema({
      type: 'object',
      properties: {
        verdict: { type: 'string', enum: ['pass', 'fail'] },
        score: { type: 'number' },
        notes: { type: 'array' },
        nested: { type: 'object', properties: { ok: { type: 'boolean' } }, required: ['ok'] },
        optional: { type: 'string' },
      },
      required: ['verdict', 'score', 'notes', 'nested'],
    }),
    { verdict: 'pass', score: 0, notes: [], nested: { ok: false } },
  );
  assert.equal(minimalFromSchema({ type: 'string' }), '');
  assert.deepEqual(minimalFromSchema(null), {});
});

test('the mock provider answers deterministically for any ref', async () => {
  const m = createMockProvider('anthropic:claude-fable-5-1');
  assert.equal(m.armed, true);
  assert.equal(m.kind, 'mock');
  const plain = await m.chat({ system: 's' });
  assert.equal(plain.text, 'mock');
  assert.equal(plain.costUsd, 0);
  assert.equal(plain.json, null);
  const shaped = await m.chat({
    jsonSchema: { schema: { type: 'object', properties: { ok: { type: 'boolean' } }, required: ['ok'] } },
  });
  assert.deepEqual(shaped.json, { ok: false });
  assert.equal(shaped.text, '{"ok":false}');
  assert.equal(m.calls.length, 2);
});

// ===========================================================================
// anthropic adapter
// ===========================================================================

test('mapStopReason normalizes the Anthropic stop_reason union', () => {
  assert.equal(mapStopReason('end_turn'), 'end_turn');
  assert.equal(mapStopReason('tool_use'), 'tool_use');
  assert.equal(mapStopReason('max_tokens'), 'max_tokens');
  assert.equal(mapStopReason('refusal'), 'refusal');
  assert.equal(mapStopReason('pause_turn'), 'pause_turn');
  assert.equal(mapStopReason('stop_sequence'), 'stop');
  assert.equal(mapStopReason('model_context_window_exceeded'), 'max_tokens');
  assert.equal(mapStopReason(null), 'end_turn');
  assert.equal(mapStopReason(undefined), 'end_turn');
});

test('parseJsonLoose recovers an object from surrounding prose', () => {
  assert.deepEqual(parseJsonLoose('{"a":1}'), { a: 1 });
  assert.deepEqual(parseJsonLoose('sure:\n{"a":{"b":2}}\nthanks'), { a: { b: 2 } });
  assert.deepEqual(parseJsonLoose('text {"s":"} not the end"} tail'), { s: '} not the end' });
  assert.equal(parseJsonLoose('no json here'), null);
  assert.equal(parseJsonLoose(''), null);
  assert.equal(parseJsonLoose('{"broken":'), null);
  assert.equal(parseJsonLoose('42'), null);
});

test('the Anthropic request never carries thinking or tool_choice', async () => {
  const client = fakeAnthropicClient(() => anthropicMessage());
  const p = createAnthropicProvider('anthropic:claude-fable-5-1', cfg(), { client });
  await p.chat({
    system: 'you are a coder',
    messages: [{ role: 'user', content: 'go' }],
    effort: 'high',
    maxTokens: 4096,
    temperature: 0.7,
    tools: [{ name: 'read_file', description: 'read', input_schema: { type: 'object' } }],
  });
  const { params } = client.calls[0];
  assert.equal(client.calls[0].mode, 'create');
  assert.equal('thinking' in params, false);
  assert.equal('tool_choice' in params, false);
  assert.equal('temperature' in params, false); // fable rejects sampling params
  assert.equal(params.model, 'claude-fable-5-1');
  assert.equal(params.max_tokens, 4096);
  assert.deepEqual(params.messages, [{ role: 'user', content: 'go' }]);
  assert.deepEqual(params.output_config, { effort: 'high' });
  assert.deepEqual(params.system, [
    { type: 'text', text: 'you are a coder', cache_control: { type: 'ephemeral' } },
  ]);
  assert.deepEqual(params.tools, [
    { name: 'read_file', description: 'read', input_schema: { type: 'object' } },
  ]);
});

test('cacheSystem:false sends a plain system string and sampling models keep temperature', async () => {
  const client = fakeAnthropicClient(() => anthropicMessage());
  const haiku = createAnthropicProvider('anthropic:claude-haiku-4-5', cfg(), { client });
  await haiku.chat({ system: 'plain', cacheSystem: false, temperature: 0.2 });
  assert.equal(client.calls[0].params.system, 'plain');
  assert.equal(client.calls[0].params.temperature, 0.2);
});

test('a json schema becomes output_config.format and is parsed back out', async () => {
  const schema = { type: 'object', properties: { verdict: { type: 'string' } }, required: ['verdict'] };
  const client = fakeAnthropicClient(() =>
    anthropicMessage({ content: [{ type: 'text', text: '{"verdict":"pass"}' }] }),
  );
  const p = createAnthropicProvider('anthropic:claude-fable-5-1', cfg(), { client });
  const r = await p.chat({ jsonSchema: { schema }, effort: 'medium' });
  assert.deepEqual(client.calls[0].params.output_config, {
    effort: 'medium',
    format: { type: 'json_schema', schema },
  });
  assert.deepEqual(r.json, { verdict: 'pass' });

  // A bare schema object is accepted too.
  const bare = fakeAnthropicClient(() => anthropicMessage());
  await createAnthropicProvider('anthropic:claude-fable-5-1', cfg(), { client: bare }).chat({
    jsonSchema: schema,
  });
  assert.deepEqual(bare.calls[0].params.output_config, { format: { type: 'json_schema', schema } });
});

test('a tool_use response round-trips into toolCalls with exact cost', async () => {
  const client = fakeAnthropicClient(() =>
    anthropicMessage({
      stop_reason: 'tool_use',
      content: [
        { type: 'text', text: 'calling' },
        { type: 'tool_use', id: 'toolu_abc', name: 'write_file', input: { path: 'a.js' } },
      ],
    }),
  );
  const p = createAnthropicProvider('anthropic:claude-fable-5-1', cfg(), { client });
  const r = await p.chat({ messages: [] });
  assert.equal(r.stopReason, 'tool_use');
  assert.equal(r.text, 'calling');
  assert.deepEqual(r.toolCalls, [{ id: 'toolu_abc', name: 'write_file', input: { path: 'a.js' } }]);
  assert.deepEqual(r.usage, { input: 1000, output: 500, cacheRead: 2000, cacheWrite: 0 });
  assert.equal(r.costUsd, 0.0355);
  assert.equal(r.provider, 'anthropic');
  assert.equal(r.model, 'claude-fable-5-1');
});

test('a refusal is returned, never thrown', async () => {
  const client = fakeAnthropicClient(() =>
    anthropicMessage({ stop_reason: 'refusal', content: [] }),
  );
  const p = createAnthropicProvider('anthropic:claude-fable-5-1', cfg(), { client });
  const r = await p.chat({});
  assert.equal(r.stopReason, 'refusal');
  assert.equal(r.text, '');
  assert.deepEqual(r.toolCalls, []);
});

test('max_tokens truncation surfaces as a stop reason', async () => {
  const client = fakeAnthropicClient(() => anthropicMessage({ stop_reason: 'max_tokens' }));
  const p = createAnthropicProvider('anthropic:claude-fable-5-1', cfg(), { client });
  assert.equal((await p.chat({})).stopReason, 'max_tokens');
});

test('max_tokens above 16000 switches to the streaming path', async () => {
  const client = fakeAnthropicClient(() => anthropicMessage());
  const p = createAnthropicProvider('anthropic:claude-fable-5-1', cfg(), { client });
  await p.chat({ maxTokens: 32000 });
  assert.equal(client.calls[0].mode, 'stream');
  assert.equal(client.calls[0].params.max_tokens, 32000);

  await p.chat({ maxTokens: 16000 });
  assert.equal(client.calls[1].mode, 'create');
});

test('overload (529) is retried and then succeeds', async () => {
  let attempts = 0;
  const client = fakeAnthropicClient(() => {
    attempts += 1;
    if (attempts === 1) throw Object.assign(new Error('overloaded'), { status: 529 });
    return anthropicMessage({ content: [{ type: 'text', text: 'recovered' }] });
  });
  const slept = [];
  const p = createAnthropicProvider('anthropic:claude-fable-5-1', cfg(), {
    client,
    sleep: async (ms) => {
      slept.push(ms);
    },
    random: fixedRandom,
  });
  const r = await p.chat({});
  assert.equal(r.text, 'recovered');
  assert.equal(attempts, 2);
  assert.deepEqual(slept, [563]); // 500 + 0.5 * 500 * 0.25
});

test('a persistent 500 gives up after three tries as provider-unavailable', async () => {
  let attempts = 0;
  const client = fakeAnthropicClient(() => {
    attempts += 1;
    throw Object.assign(new Error('server error'), { status: 500 });
  });
  const p = createAnthropicProvider('anthropic:claude-fable-5-1', cfg(), {
    client,
    sleep: noSleep,
    random: fixedRandom,
  });
  await assert.rejects(
    () => p.chat({}),
    (e) => isLadderError(e, 'provider-unavailable') && e.detail.status === 500,
  );
  assert.equal(attempts, 3);
});

test('a 400 is a provider-bad-request and is never retried', async () => {
  let attempts = 0;
  const client = fakeAnthropicClient(() => {
    attempts += 1;
    throw Object.assign(new Error('max_tokens: too large'), {
      status: 400,
      error: { type: 'invalid_request_error' },
    });
  });
  const p = createAnthropicProvider('anthropic:claude-fable-5-1', cfg(), { client, sleep: noSleep });
  await assert.rejects(
    () => p.chat({}),
    (e) =>
      isLadderError(e, 'provider-bad-request') &&
      e.detail.status === 400 &&
      e.detail.body.type === 'invalid_request_error',
  );
  assert.equal(attempts, 1);
});

test('a rejected key is not-armed and a timeout is provider-timeout', async () => {
  const rejected = fakeAnthropicClient(() => {
    throw Object.assign(new Error('invalid x-api-key'), { status: 401 });
  });
  await assert.rejects(
    () => createAnthropicProvider('anthropic:claude-fable-5-1', cfg(), { client: rejected }).chat({}),
    (e) => isLadderError(e, 'not-armed') && e.detail.envVar === 'ANTHROPIC_API_KEY',
  );

  const timedOut = fakeAnthropicClient(() => {
    throw Object.assign(new Error('timed out'), { name: 'APIConnectionTimeoutError' });
  });
  await assert.rejects(
    () => createAnthropicProvider('anthropic:claude-fable-5-1', cfg(), { client: timedOut }).chat({}),
    (e) => isLadderError(e, 'provider-timeout'),
  );
});

test('an Anthropic call aborted by the caller propagates unchanged', async () => {
  const controller = new AbortController();
  const client = fakeAnthropicClient(() => {
    controller.abort();
    throw Object.assign(new Error('aborted'), { name: 'AbortError' });
  });
  const p = createAnthropicProvider('anthropic:claude-fable-5-1', cfg(), { client, sleep: noSleep });
  await assert.rejects(
    () => p.chat({ signal: controller.signal }),
    (e) => e.name === 'AbortError' && !isLadderError(e),
  );
});

test('an unarmed Anthropic rung constructs but refuses to chat', async () => {
  const p = createAnthropicProvider('anthropic:claude-fable-5-1', cfg());
  assert.equal(p.armed, false);
  assert.equal(p.id, 'anthropic:claude-fable-5-1');
  await assert.rejects(
    () => p.chat({}),
    (e) => isLadderError(e, 'not-armed') && /ANTHROPIC_API_KEY/.test(e.message),
  );
  assert.equal(createAnthropicProvider('anthropic:claude-fable-5-1', cfg({ anthropic: 'sk' })).armed, true);
  assert.throws(
    () => createAnthropicProvider('deepseek:deepseek-v4-flash', cfg()),
    (e) => isLadderError(e, 'bad-ref'),
  );
});

test('signal and timeoutMs are forwarded as SDK request options', async () => {
  const client = fakeAnthropicClient(() => anthropicMessage());
  const p = createAnthropicProvider('anthropic:claude-fable-5-1', cfg(), { client });
  const controller = new AbortController();
  await p.chat({ signal: controller.signal, timeoutMs: 30000 });
  assert.equal(client.calls[0].options.timeout, 30000);
  assert.equal(client.calls[0].options.signal, controller.signal);
});

// ===========================================================================
// openai-compatible adapter
// ===========================================================================

test('mapFinishReason normalizes OpenAI finish reasons', () => {
  assert.equal(mapFinishReason('stop'), 'end_turn');
  assert.equal(mapFinishReason('tool_calls'), 'tool_use');
  assert.equal(mapFinishReason('function_call'), 'tool_use');
  assert.equal(mapFinishReason('length'), 'max_tokens');
  assert.equal(mapFinishReason('content_filter'), 'refusal');
  assert.equal(mapFinishReason(null), 'end_turn');
  assert.equal(mapFinishReason('something_new'), 'something_new');
});

test('stripThinking removes reasoning blocks, terminated or not', () => {
  assert.equal(stripThinking('<think>plan</think>answer'), 'answer');
  assert.equal(stripThinking('a<think>x</think>b<think>y</think>c'), 'abc');
  assert.equal(stripThinking('answer<think>truncated...'), 'answer');
  assert.equal(stripThinking('  spaced  '), 'spaced');
  assert.equal(stripThinking(''), '');
  assert.equal(stripThinking(null), '');
});

test('toOpenAIMessages translates system, tool_use and tool_result', () => {
  const messages = [
    { role: 'user', content: 'do it' },
    {
      role: 'assistant',
      content: [
        { type: 'text', text: 'sure' },
        { type: 'tool_use', id: 'toolu_1', name: 'write_file', input: { path: 'a.js' } },
      ],
    },
    {
      role: 'user',
      content: [{ type: 'tool_result', tool_use_id: 'toolu_1', content: 'written' }],
    },
  ];
  assert.deepEqual(toOpenAIMessages('be brief', messages), [
    { role: 'system', content: 'be brief' },
    { role: 'user', content: 'do it' },
    {
      role: 'assistant',
      content: 'sure',
      tool_calls: [
        { id: 'toolu_1', type: 'function', function: { name: 'write_file', arguments: '{"path":"a.js"}' } },
      ],
    },
    { role: 'tool', tool_call_id: 'toolu_1', content: 'written' },
  ]);
});

test('a tool-only assistant turn has null content and block results stringify', () => {
  const out = toOpenAIMessages(null, [
    { role: 'assistant', content: [{ type: 'tool_use', id: 't1', name: 'run', input: {} }] },
    {
      role: 'user',
      content: [
        { type: 'tool_result', tool_use_id: 't1', content: [{ type: 'text', text: 'ok' }] },
        { type: 'text', text: 'and now this' },
      ],
    },
  ]);
  assert.equal(out[0].content, null);
  assert.equal(out[0].tool_calls[0].function.arguments, '{}');
  assert.deepEqual(out[1], { role: 'tool', tool_call_id: 't1', content: 'ok' });
  assert.deepEqual(out[2], { role: 'user', content: 'and now this' });
});

test('toOpenAITools maps input_schema to parameters', () => {
  assert.deepEqual(toOpenAITools([{ name: 'run', description: 'run it', input_schema: { type: 'object' } }]), [
    { type: 'function', function: { name: 'run', description: 'run it', parameters: { type: 'object' } } },
  ]);
  assert.deepEqual(toOpenAITools([{ name: 'bare' }]), [
    {
      type: 'function',
      function: { name: 'bare', description: '', parameters: { type: 'object', properties: {} } },
    },
  ]);
  assert.deepEqual(toOpenAITools(), []);
});

test('the DeepSeek request carries the bearer key, model and messages', async () => {
  const fetchImpl = fakeFetch(okResponse(chatCompletion()));
  const p = createOpenAICompatibleProvider('deepseek:deepseek-v4-flash', cfg({ deepseek: 'sk-ds' }), {
    fetchImpl,
  });
  assert.equal(p.armed, true);
  assert.equal(p.kind, 'openai-compatible');
  const r = await p.chat({ system: 'sys', messages: [{ role: 'user', content: 'hi' }], maxTokens: 2048 });
  const req = fetchImpl.requests[0];
  assert.equal(req.url, 'https://api.deepseek.com/v1/chat/completions');
  assert.equal(req.init.method, 'POST');
  assert.equal(req.init.headers.authorization, 'Bearer sk-ds');
  assert.equal(req.init.headers['content-type'], 'application/json');
  assert.equal(req.body.model, 'deepseek-v4-flash');
  assert.equal(req.body.max_tokens, 2048);
  assert.equal(req.body.stream, false);
  assert.deepEqual(req.body.messages, [
    { role: 'system', content: 'sys' },
    { role: 'user', content: 'hi' },
  ]);
  assert.equal(r.text, 'hi');
  assert.equal(r.stopReason, 'end_turn');
  assert.equal(r.costUsd, estimateCost('deepseek:deepseek-v4-flash', { input: 1000, output: 500 }));
});

test('ollama needs no key and openrouter adds its attribution headers', async () => {
  const ollamaFetch = fakeFetch(okResponse(chatCompletion()));
  const ollama = createOpenAICompatibleProvider('ollama:qwen3.8:latest', cfg(), {
    fetchImpl: ollamaFetch,
  });
  assert.equal(ollama.armed, true);
  assert.equal(ollama.baseUrl, 'http://127.0.0.1:11434/v1');
  await ollama.chat({ messages: [{ role: 'user', content: 'hi' }] });
  assert.equal(ollamaFetch.requests[0].url, 'http://127.0.0.1:11434/v1/chat/completions');
  assert.equal('authorization' in ollamaFetch.requests[0].init.headers, false);
  assert.equal(ollamaFetch.requests[0].body.model, 'qwen3.8:latest');

  const orFetch = fakeFetch(okResponse(chatCompletion()));
  await createOpenAICompatibleProvider(
    'openrouter:deepseek/deepseek-v4-flash',
    cfg({ openrouter: 'sk-or' }),
    { fetchImpl: orFetch },
  ).chat({});
  const headers = orFetch.requests[0].init.headers;
  assert.equal(headers.authorization, 'Bearer sk-or');
  assert.equal(headers['HTTP-Referer'], 'https://kolm.ai');
  assert.equal(headers['X-Title'], 'kolm code');
  assert.equal(orFetch.requests[0].body.model, 'deepseek/deepseek-v4-flash');
});

test('a base URL override wins and an unknown provider has none', async () => {
  const fetchImpl = fakeFetch(okResponse(chatCompletion()));
  const p = createOpenAICompatibleProvider('cerebras:qwen-3.8-27b', cfg({ cerebras: 'csk' }), {
    fetchImpl,
    baseUrl: 'https://proxy.internal/v1/',
  });
  await p.chat({});
  assert.equal(fetchImpl.requests[0].url, 'https://proxy.internal/v1/chat/completions');
  assert.throws(
    () => createOpenAICompatibleProvider('mistral:big', {}, { fetchImpl }),
    (e) => isLadderError(e, 'bad-ref'),
  );
});

test('a json schema is sent strict, then retried without strict after a 400', async () => {
  const schema = { type: 'object', properties: { ok: { type: 'boolean' } }, required: ['ok'] };
  const fetchImpl = fakeFetch([
    errResponse(400, 'strict schemas are not supported'),
    okResponse(chatCompletion({ choices: [{ finish_reason: 'stop', message: { content: '{"ok":true}' } }] })),
  ]);
  const p = createOpenAICompatibleProvider('cerebras:qwen-3.8-27b', cfg({ cerebras: 'csk' }), {
    fetchImpl,
    sleep: noSleep,
    random: fixedRandom,
  });
  const r = await p.chat({ jsonSchema: { name: 'verdict', schema } });
  assert.equal(fetchImpl.requests.length, 2);
  assert.deepEqual(fetchImpl.requests[0].body.response_format, {
    type: 'json_schema',
    json_schema: { name: 'verdict', schema, strict: true },
  });
  assert.deepEqual(fetchImpl.requests[1].body.response_format, {
    type: 'json_schema',
    json_schema: { name: 'verdict', schema },
  });
  assert.deepEqual(r.json, { ok: true });
});

test('tool calls translate back, including unparseable arguments', async () => {
  const fetchImpl = fakeFetch(
    okResponse(
      chatCompletion({
        choices: [
          {
            finish_reason: 'tool_calls',
            message: {
              content: null,
              tool_calls: [
                { id: 'call_1', type: 'function', function: { name: 'write_file', arguments: '{"path":"a.js"}' } },
                { id: 'call_2', type: 'function', function: { name: 'run', arguments: 'not json' } },
              ],
            },
          },
        ],
      }),
    ),
  );
  const p = createOpenAICompatibleProvider('deepseek:deepseek-v4-flash', cfg({ deepseek: 'sk-ds' }), {
    fetchImpl,
  });
  const r = await p.chat({ tools: [{ name: 'write_file', input_schema: { type: 'object' } }] });
  assert.deepEqual(fetchImpl.requests[0].body.tools[0].function.name, 'write_file');
  assert.equal(r.stopReason, 'tool_use');
  assert.equal(r.text, '');
  assert.deepEqual(r.toolCalls[0], { id: 'call_1', name: 'write_file', input: { path: 'a.js' } });
  assert.deepEqual(r.toolCalls[1], { id: 'call_2', name: 'run', input: { _raw: 'not json' } });
  assert.equal(r.content[0].type, 'tool_use');
});

test('reasoning text is stripped and reasoning_content is the fallback', async () => {
  const thinking = fakeFetch(
    okResponse(
      chatCompletion({
        choices: [{ finish_reason: 'stop', message: { content: '<think>plan</think>the answer' } }],
      }),
    ),
  );
  const p = createOpenAICompatibleProvider('deepseek:deepseek-v4-flash', cfg({ deepseek: 'sk-ds' }), {
    fetchImpl: thinking,
  });
  assert.equal((await p.chat({})).text, 'the answer');

  const onlyReasoning = fakeFetch(
    okResponse(
      chatCompletion({
        choices: [{ finish_reason: 'stop', message: { content: '', reasoning_content: 'fallback' } }],
      }),
    ),
  );
  const q = createOpenAICompatibleProvider('deepseek:deepseek-v4-flash', cfg({ deepseek: 'sk-ds' }), {
    fetchImpl: onlyReasoning,
  });
  assert.equal((await q.chat({})).text, 'fallback');
});

test('cache-hit tokens are normalized to Anthropic usage semantics', async () => {
  const hit = fakeFetch(
    okResponse(
      chatCompletion({
        usage: { prompt_tokens: 1000, completion_tokens: 500, prompt_cache_hit_tokens: 400 },
      }),
    ),
  );
  const p = createOpenAICompatibleProvider('deepseek:deepseek-v4-flash', cfg({ deepseek: 'sk-ds' }), {
    fetchImpl: hit,
  });
  const r = await p.chat({});
  assert.deepEqual(r.usage, { input: 600, output: 500, cacheRead: 400, cacheWrite: 0 });
  assert.equal(r.costUsd, estimateCost('deepseek:deepseek-v4-flash', r.usage));

  const details = fakeFetch(
    okResponse(
      chatCompletion({
        usage: {
          prompt_tokens: 1000,
          completion_tokens: 0,
          prompt_tokens_details: { cached_tokens: 250 },
        },
      }),
    ),
  );
  const q = createOpenAICompatibleProvider('deepseek:deepseek-v4-flash', cfg({ deepseek: 'sk-ds' }), {
    fetchImpl: details,
  });
  assert.deepEqual((await q.chat({})).usage, { input: 750, output: 0, cacheRead: 250, cacheWrite: 0 });
});

test('the off-peak flag reaches the cost estimate', async () => {
  const fetchImpl = fakeFetch(okResponse(chatCompletion({ usage: { prompt_tokens: 1e6, completion_tokens: 0 } })));
  const p = createOpenAICompatibleProvider(
    'deepseek:deepseek-v4-flash',
    cfg({ deepseek: 'sk-ds' }, { deepseekOffpeak: true }),
    { fetchImpl },
  );
  assert.equal((await p.chat({})).costUsd, 0.22);
});

test('a 429 is retried and then succeeds', async () => {
  const fetchImpl = fakeFetch([
    errResponse(429, 'rate limited'),
    okResponse(chatCompletion({ choices: [{ finish_reason: 'stop', message: { content: 'ok now' } }] })),
  ]);
  const slept = [];
  const p = createOpenAICompatibleProvider('cerebras:qwen-3.8-27b', cfg({ cerebras: 'csk' }), {
    fetchImpl,
    sleep: async (ms) => {
      slept.push(ms);
    },
    random: fixedRandom,
  });
  assert.equal((await p.chat({})).text, 'ok now');
  assert.equal(fetchImpl.requests.length, 2);
  assert.deepEqual(slept, [563]);
});

test('4xx and 5xx map to the right ladder error codes', async () => {
  const bad = createOpenAICompatibleProvider('cerebras:qwen-3.8-27b', cfg({ cerebras: 'csk' }), {
    fetchImpl: fakeFetch(errResponse(400, 'bad body')),
    sleep: noSleep,
  });
  await assert.rejects(
    () => bad.chat({}),
    (e) => isLadderError(e, 'provider-bad-request') && e.detail.status === 400,
  );

  const unauth = createOpenAICompatibleProvider('cerebras:qwen-3.8-27b', cfg({ cerebras: 'csk' }), {
    fetchImpl: fakeFetch(errResponse(401, 'nope')),
    sleep: noSleep,
  });
  await assert.rejects(
    () => unauth.chat({}),
    (e) => isLadderError(e, 'not-armed') && e.detail.envVar === 'CEREBRAS_API_KEY',
  );

  const down = createOpenAICompatibleProvider('cerebras:qwen-3.8-27b', cfg({ cerebras: 'csk' }), {
    fetchImpl: fakeFetch(errResponse(503, 'down')),
    sleep: noSleep,
    random: fixedRandom,
  });
  await assert.rejects(
    () => down.chat({}),
    (e) => isLadderError(e, 'provider-unavailable') && e.detail.status === 503,
  );
});

test('an exceeded timeout budget is provider-timeout', async () => {
  const fetchImpl = async (url, init) =>
    new Promise((resolve, reject) => {
      init.signal.addEventListener('abort', () =>
        reject(Object.assign(new Error('aborted'), { name: 'AbortError' })),
      );
    });
  const p = createOpenAICompatibleProvider('cerebras:qwen-3.8-27b', cfg({ cerebras: 'csk' }), {
    fetchImpl,
    sleep: noSleep,
  });
  await assert.rejects(
    () => p.chat({ timeoutMs: 1 }),
    (e) => isLadderError(e, 'provider-timeout') && e.detail.timeoutMs === 1,
  );
});

test("the caller's own abort is propagated unchanged", async () => {
  const controller = new AbortController();
  const fetchImpl = async (url, init) =>
    new Promise((resolve, reject) => {
      init.signal.addEventListener('abort', () =>
        reject(Object.assign(new Error('aborted'), { name: 'AbortError' })),
      );
      controller.abort();
    });
  const p = createOpenAICompatibleProvider('cerebras:qwen-3.8-27b', cfg({ cerebras: 'csk' }), {
    fetchImpl,
    sleep: noSleep,
  });
  await assert.rejects(
    () => p.chat({ signal: controller.signal }),
    (e) => e.name === 'AbortError',
  );
});

test('an unarmed OpenAI-compatible rung constructs but refuses to chat', async () => {
  const p = createOpenAICompatibleProvider('cerebras:qwen-3.8-27b', cfg());
  assert.equal(p.armed, false);
  await assert.rejects(
    () => p.chat({}),
    (e) => isLadderError(e, 'not-armed') && e.detail.envVar === 'CEREBRAS_API_KEY',
  );
});

test('OPENAI_COMPATIBLE_PROVIDERS lists exactly the adapter families', () => {
  assert.deepEqual([...OPENAI_COMPATIBLE_PROVIDERS], ['cerebras', 'deepseek', 'openrouter', 'ollama']);
});

// ===========================================================================
// factory
// ===========================================================================

test('resolveRef describes the adapter, key env var and arming for a ref', () => {
  const config = cfg({ deepseek: 'sk-ds' });
  const ds = resolveRef('deepseek:deepseek-v4-flash', config);
  assert.equal(ds.kind, 'openai-compatible');
  assert.equal(ds.keyEnv, 'DEEPSEEK_API_KEY');
  assert.equal(ds.baseUrl, 'https://api.deepseek.com/v1');
  assert.equal(ds.armed, true);
  assert.equal(ds.spec.priceIn, 0.44);

  assert.equal(resolveRef('anthropic:claude-fable-5-1', config).armed, false);
  assert.equal(resolveRef('anthropic:claude-fable-5-1', config).kind, 'anthropic');
  assert.equal(resolveRef('ollama:qwen3.8:latest', config).armed, true);
  assert.equal(resolveRef('fake:x', config).armed, true);
  assert.throws(
    () => resolveRef('mistral:big', config),
    (e) => isLadderError(e, 'bad-ref'),
  );
});

test('createProvider dispatches every supported provider', () => {
  const config = cfg({ anthropic: 'sk-ant', cerebras: 'csk', deepseek: 'sk-ds', openrouter: 'sk-or' });
  assert.deepEqual([...SUPPORTED_PROVIDERS], Object.keys(PROVIDER_KINDS));
  assert.equal(createProvider('anthropic:claude-fable-5-1', config).kind, 'anthropic');
  assert.equal(createProvider('cerebras:qwen-3.8-27b', config).kind, 'openai-compatible');
  assert.equal(createProvider('deepseek:deepseek-v4-flash', config).kind, 'openai-compatible');
  assert.equal(createProvider('openrouter:deepseek/deepseek-v4-flash', config).kind, 'openai-compatible');
  assert.equal(createProvider('ollama:qwen3.8:latest', config).kind, 'openai-compatible');
  assert.equal(createProvider('fake:x', config).kind, 'fake');
  assert.throws(
    () => createProvider('mistral:big', config),
    (e) => isLadderError(e, 'bad-ref'),
  );
});

test('createProvider forwards a script to the fake provider', async () => {
  const p = createProvider('fake:scripted', cfg(), { script: ['scripted answer'] });
  assert.equal(p.id, 'fake:scripted');
  assert.equal((await p.chat({})).text, 'scripted answer');
});

test('the mock switch short-circuits every adapter', async () => {
  assert.equal(createProvider('anthropic:claude-fable-5-1', cfg(), { mock: true }).kind, 'mock');
  assert.equal(createProvider('anthropic:claude-fable-5-1', cfg({}, { mockProvider: true })).kind, 'mock');

  const previous = process.env.KOLM_TEST_MOCK_PROVIDER;
  process.env.KOLM_TEST_MOCK_PROVIDER = '1';
  try {
    const p = createProvider('deepseek:deepseek-v4-flash', cfg());
    assert.equal(p.kind, 'mock');
    assert.equal(p.armed, true);
    assert.equal((await p.chat({})).text, 'mock');
  } finally {
    if (previous === undefined) delete process.env.KOLM_TEST_MOCK_PROVIDER;
    else process.env.KOLM_TEST_MOCK_PROVIDER = previous;
  }
  assert.equal(createProvider('deepseek:deepseek-v4-flash', cfg({ deepseek: 'sk' })).kind, 'openai-compatible');
});

test('an unarmed rung from createProvider still constructs', async () => {
  const p = createProvider('cerebras:qwen-3.8-27b', cfg());
  assert.equal(p.armed, false);
  await assert.rejects(
    () => p.chat({}),
    (e) => isLadderError(e, 'not-armed'),
  );
});

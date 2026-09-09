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
  supportsNativeTools,
} from '../src/ladder/providers/registry.js';
import {
  PROVIDER_CONCURRENCY_DEFAULTS,
  createLimiter,
  limiterFor,
  resetLimiters,
  retryDelayMs,
} from '../src/ladder/providers/limiter.js';
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
  assert.deepEqual(slept, [1000]); // spec 3.3: 1s base, +-20% jitter, random()=0.5 -> no offset
});

test('a persistent 500 gives up after two tries as provider-unavailable', async () => {
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
  assert.equal(attempts, 2); // spec 3.3: a plain 5xx is worth two tries, not three
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

test('an assistant turn with no text and no tool call is sent as an empty string, not null', () => {
  // Ollama's OpenAI layer rejects content:null unless tool_calls is present
  // ("invalid message content type: <nil>"); the coder hits this after a
  // silent turn (live run r_20260909_2fb26394).
  const out = toOpenAIMessages(null, [
    { role: 'assistant', content: [] },
    { role: 'user', content: 'You made no tool call.' },
  ]);
  assert.deepEqual(out[0], { role: 'assistant', content: '' });
  assert.equal('tool_calls' in out[0], false);
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
  assert.deepEqual(slept, [1000]);
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

// ===========================================================================
// limiter and backoff (spec 3.3)
// ===========================================================================

/** A promise plus its resolver, so a test can hold a request open. */
function deferred() {
  let resolve;
  const promise = new Promise((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

test('the limiter runs at most its concurrency in parallel and drains the queue', async () => {
  const limiter = createLimiter({ concurrency: 2 });
  let active = 0;
  let peak = 0;
  const gates = [deferred(), deferred(), deferred(), deferred()];
  const runs = gates.map((gate) =>
    limiter.run(async () => {
      active += 1;
      peak = Math.max(peak, active);
      await gate.promise;
      active -= 1;
      return 'done';
    }),
  );
  await Promise.resolve();
  assert.equal(limiter.stats().active, 2);
  assert.equal(limiter.stats().queued, 2);
  for (const gate of gates) gate.resolve();
  assert.deepEqual(await Promise.all(runs), ['done', 'done', 'done', 'done']);
  assert.equal(peak, 2);
  assert.equal(limiter.stats().active, 0);
  assert.equal(limiter.stats().queued, 0);
});

test('a rejected task still frees its slot', async () => {
  const limiter = createLimiter({ concurrency: 1 });
  await assert.rejects(() => limiter.run(async () => { throw new Error('boom'); }), /boom/);
  assert.equal(await limiter.run(async () => 'after'), 'after');
});

test('an aborted signal rejects a queued task without running it', async () => {
  const limiter = createLimiter({ concurrency: 1 });
  const gate = deferred();
  const first = limiter.run(() => gate.promise);
  const controller = new AbortController();
  let ran = false;
  const second = limiter.run(async () => {
    ran = true;
  }, { signal: controller.signal });
  await Promise.resolve();
  controller.abort();
  await assert.rejects(() => second, (e) => e.name === 'AbortError');
  gate.resolve('ok');
  assert.equal(await first, 'ok');
  assert.equal(ran, false);
});

test('a 429 halves concurrency for the penalty window and never goes below 1', () => {
  let clock = 1000;
  const limiter = createLimiter({ concurrency: 4, now: () => clock });
  assert.equal(limiter.stats().concurrency, 4);
  limiter.penalize({ ms: 60000 });
  assert.equal(limiter.stats().concurrency, 2);
  assert.equal(limiter.stats().throttled, true);
  limiter.penalize({ ms: 60000 });
  assert.equal(limiter.stats().concurrency, 1);
  limiter.penalize({ ms: 60000 });
  assert.equal(limiter.stats().concurrency, 1);
  // The penalty expires on its own, with no timer holding the process open.
  clock += 60001;
  assert.equal(limiter.stats().concurrency, 4);
  assert.equal(limiter.stats().throttled, false);
});

test('limiterFor caches one limiter per provider and honours the config override', () => {
  resetLimiters();
  try {
    assert.equal(limiterFor('ollama', {}).stats().baseConcurrency, PROVIDER_CONCURRENCY_DEFAULTS.ollama);
    assert.equal(limiterFor('anthropic', {}).stats().baseConcurrency, 4);
    assert.equal(limiterFor('fake', {}).stats().baseConcurrency, Infinity);
    assert.equal(limiterFor('mistral', {}).stats().baseConcurrency, 2);
    const first = limiterFor('deepseek', {});
    const again = limiterFor('deepseek', { providerConcurrency: { deepseek: 7 } });
    assert.equal(first, again);
    assert.equal(again.stats().baseConcurrency, 7);
  } finally {
    resetLimiters();
  }
});

test('retryDelayMs honours Retry-After in seconds, as a date, and falls back to backoff', () => {
  const at = Date.parse('2026-09-09T12:00:00Z');
  const now = () => at;
  assert.equal(retryDelayMs(0, { 'retry-after': '2' }, { now }), 2000);
  assert.equal(retryDelayMs(3, { 'Retry-After': '5' }, { now }), 5000);
  // A header-object with .get (fetch Headers) works the same way.
  assert.equal(
    retryDelayMs(0, { get: (name) => (name === 'retry-after' ? '3' : null) }, { now }),
    3000,
  );
  // HTTP-date form.
  assert.equal(
    retryDelayMs(0, { 'retry-after': new Date(at + 4000).toUTCString() }, { now }),
    4000,
  );
  // A date in the past is not a negative wait.
  assert.equal(retryDelayMs(0, { 'retry-after': new Date(at - 9000).toUTCString() }, { now }), 0);
  // Every result is capped at 30 s.
  assert.equal(retryDelayMs(0, { 'retry-after': '600' }, { now }), 30000);
  // No header: 1s, 2s, 4s, 8s with random()=0.5 cancelling the jitter.
  const opts = { now, random: fixedRandom };
  assert.deepEqual([0, 1, 2, 3].map((n) => retryDelayMs(n, null, opts)), [1000, 2000, 4000, 8000]);
  // Jitter stays inside +-20%.
  assert.equal(retryDelayMs(0, null, { now, random: () => 0 }), 800);
  assert.equal(retryDelayMs(0, null, { now, random: () => 1 }), 1200);
  // An unparseable header falls back to backoff rather than throwing.
  assert.equal(retryDelayMs(0, { 'retry-after': 'soon' }, opts), 1000);
});

test('a 429 reaches the reply as meta.throttled and honours Retry-After', async () => {
  const throttled = {
    status: 429,
    headers: { 'retry-after': '2' },
    json: async () => ({ error: 'slow down' }),
    text: async () => 'slow down',
  };
  const fetchImpl = fakeFetch([throttled, okResponse(chatCompletion())]);
  const slept = [];
  const limiter = createLimiter({ concurrency: 4, now: () => 0 });
  const p = createOpenAICompatibleProvider('cerebras:qwen-3.8-27b', cfg({ cerebras: 'csk' }), {
    fetchImpl,
    limiter,
    sleep: async (ms) => {
      slept.push(ms);
    },
    random: fixedRandom,
  });
  const r = await p.chat({});
  assert.equal(r.text, 'hi');
  assert.deepEqual(slept, [2000]);
  assert.equal(r.meta.throttled, true);
  assert.equal(limiter.stats().concurrency, 2);
});

test('a 429 is worth four tries where a plain 500 is worth two', async () => {
  const throttleFetch = fakeFetch(errResponse(429, 'rate limited'));
  const throttled = createOpenAICompatibleProvider('cerebras:qwen-3.8-27b', cfg({ cerebras: 'csk' }), {
    fetchImpl: throttleFetch,
    limiter: createLimiter({ concurrency: 4, now: () => 0 }),
    sleep: noSleep,
    random: fixedRandom,
  });
  await assert.rejects(() => throttled.chat({}), (e) => isLadderError(e, 'provider-unavailable'));
  assert.equal(throttleFetch.requests.length, 4);

  const serverFetch = fakeFetch(errResponse(500, 'server error'));
  const server = createOpenAICompatibleProvider('cerebras:qwen-3.8-27b', cfg({ cerebras: 'csk' }), {
    fetchImpl: serverFetch,
    sleep: noSleep,
    random: fixedRandom,
  });
  await assert.rejects(() => server.chat({}), (e) => isLadderError(e, 'provider-unavailable'));
  assert.equal(serverFetch.requests.length, 2);
});

// ===========================================================================
// text tool protocol in the adapters (spec 3.2)
// ===========================================================================

const TEXT_TOOLS = [
  {
    name: 'read_file',
    description: 'Read a file.',
    input_schema: { type: 'object', properties: { path: { type: 'string' } } },
  },
];

/** A reply written in the text tool protocol. */
const TEXT_REPLY = 'reading it now\n<<<tool name="read_file">>>\n{"path": "a.js"}\n<<<end>>>';

function textCompletion(text) {
  return okResponse(chatCompletion({ choices: [{ finish_reason: 'stop', message: { content: text } }] }));
}

test('a 400 that says tools are unsupported flips the instance and retries once', async () => {
  const fetchImpl = fakeFetch([
    errResponse(400, 'registry.Model "deepseek-coder-v2" does not support tools'),
    textCompletion(TEXT_REPLY),
  ]);
  const p = createOpenAICompatibleProvider('ollama:qwen3.8:latest', cfg(), {
    fetchImpl,
    sleep: noSleep,
    random: fixedRandom,
  });
  assert.equal(p.textTools, false);

  const r = await p.chat({ system: 'be brief', tools: TEXT_TOOLS });
  assert.equal(fetchImpl.requests.length, 2);
  // First attempt: native tools. Second: no tools field, protocol in the system.
  assert.equal(Array.isArray(fetchImpl.requests[0].body.tools), true);
  assert.equal('tools' in fetchImpl.requests[1].body, false);
  const system = fetchImpl.requests[1].body.messages[0];
  assert.equal(system.role, 'system');
  assert.match(system.content, /^be brief\n\nTOOL CALLING PROTOCOL \(text mode, v1\)/);
  assert.equal(system.content.includes('1. read_file - Read a file.'), true);

  assert.equal(r.meta.textTools, true);
  assert.equal(r.text, 'reading it now');
  assert.equal(r.stopReason, 'tool_use');
  assert.deepEqual(r.toolCalls, [{ id: 'text_1', name: 'read_file', input: { path: 'a.js' } }]);
  assert.deepEqual(r.content, [
    { type: 'text', text: 'reading it now' },
    { type: 'tool_use', id: 'text_1', name: 'read_file', input: { path: 'a.js' } },
  ]);

  // The flip is latched for the rest of the process: no second 400 is needed.
  assert.equal(p.textTools, true);
  const next = fakeFetch([textCompletion(TEXT_REPLY)]);
  const q = createOpenAICompatibleProvider('ollama:qwen3.8:latest', cfg(), { fetchImpl: next });
  await q.chat({ tools: TEXT_TOOLS });
  assert.equal(Array.isArray(next.requests[0].body.tools), true); // a fresh instance starts native
});

test('a 400 that is not about tools is not flipped', async () => {
  const fetchImpl = fakeFetch(errResponse(400, 'context length exceeded'));
  const p = createOpenAICompatibleProvider('ollama:qwen3.8:latest', cfg(), {
    fetchImpl,
    sleep: noSleep,
  });
  await assert.rejects(
    () => p.chat({ tools: TEXT_TOOLS }),
    (e) => isLadderError(e, 'provider-bad-request'),
  );
  assert.equal(fetchImpl.requests.length, 1);
  assert.equal(p.textTools, false);
});

test('a registry entry with tools:false starts in text mode', async () => {
  const fetchImpl = fakeFetch([textCompletion(TEXT_REPLY)]);
  const p = createOpenAICompatibleProvider('ollama:deepseek-coder-v2', cfg(), { fetchImpl });
  assert.equal(p.textTools, true);
  const r = await p.chat({ tools: TEXT_TOOLS, messages: [{ role: 'user', content: 'fix it' }] });
  assert.equal(fetchImpl.requests.length, 1);
  assert.equal('tools' in fetchImpl.requests[0].body, false);
  assert.equal(r.meta.textTools, true);
  assert.equal(r.toolCalls[0].id, 'text_1');
});

test('KOLM_CODE_TEXT_TOOLS forces the protocol on any ref, wildcards included', async () => {
  const fetchImpl = fakeFetch([textCompletion(TEXT_REPLY)]);
  const p = createOpenAICompatibleProvider(
    'openrouter:some/free-model',
    cfg({ openrouter: 'sk-or' }, { textTools: ['openrouter:*'] }),
    { fetchImpl },
  );
  assert.equal(p.textTools, true);
  await p.chat({ tools: TEXT_TOOLS });
  assert.equal('tools' in fetchImpl.requests[0].body, false);
});

test('text mode flattens tool_result history into user text', async () => {
  const fetchImpl = fakeFetch([textCompletion('done')]);
  const p = createOpenAICompatibleProvider('ollama:deepseek-coder-v2', cfg(), { fetchImpl });
  await p.chat({
    tools: TEXT_TOOLS,
    messages: [
      { role: 'user', content: 'fix it' },
      {
        role: 'assistant',
        content: [{ type: 'tool_use', id: 'text_1', name: 'read_file', input: { path: 'a.js' } }],
      },
      { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'text_1', content: 'body' }] },
    ],
  });
  const sent = fetchImpl.requests[0].body.messages;
  // No tool role, no tool_calls: everything is plain text on the wire.
  assert.equal(sent.some((m) => m.role === 'tool' || m.tool_calls), false);
  for (const m of sent) assert.equal(typeof m.content, 'string');
  assert.equal(sent.at(-2).content.includes('<<<tool name="read_file">>>'), true);
  assert.equal(sent.at(-1).content.includes('<<<result id="text_1" tool="read_file">>>'), true);
});

test('a jsonSchema call in text mode asks for JSON only and still parses', async () => {
  const schema = { type: 'object', properties: { ok: { type: 'boolean' } } };
  const fetchImpl = fakeFetch([textCompletion('Sure. {"ok": true} hope that helps')]);
  const p = createOpenAICompatibleProvider('ollama:deepseek-coder-v2', cfg(), { fetchImpl });
  const r = await p.chat({ jsonSchema: { name: 'verdict', schema } });
  const body = fetchImpl.requests[0].body;
  assert.equal('response_format' in body, false);
  assert.match(body.messages[0].content, /^OUTPUT FORMAT/);
  assert.equal(body.messages[0].content.includes('{"type":"object"'), true);
  assert.deepEqual(r.json, { ok: true });
});

test('broken tool JSON in text mode still produces a call, with the error attached', async () => {
  const fetchImpl = fakeFetch([
    textCompletion('<<<tool name="write_file">>>\n{"path": "a.js", oops}\n<<<end>>>'),
  ]);
  const p = createOpenAICompatibleProvider('ollama:deepseek-coder-v2', cfg(), { fetchImpl });
  const r = await p.chat({ tools: TEXT_TOOLS });
  assert.equal(r.toolCalls[0].name, 'write_file');
  assert.equal(r.toolCalls[0].input.__parse_error, 'arguments are not valid JSON');
  assert.deepEqual(r.meta.textToolErrors, ['tool "write_file": arguments are not valid JSON']);
});

test('the Anthropic adapter can be forced into text mode too', async () => {
  const client = fakeAnthropicClient(() =>
    anthropicMessage({ content: [{ type: 'text', text: TEXT_REPLY }] }),
  );
  const p = createAnthropicProvider(
    'anthropic:claude-fable-5-1',
    cfg({ anthropic: 'sk-ant' }, { textTools: ['anthropic:*'] }),
    { client },
  );
  assert.equal(p.textTools, true);
  const r = await p.chat({ system: 'be brief', tools: TEXT_TOOLS });
  const params = client.calls[0].params;
  assert.equal('tools' in params, false);
  assert.match(params.system[0].text, /^be brief\n\nTOOL CALLING PROTOCOL/);
  assert.equal(r.meta.textTools, true);
  assert.deepEqual(r.toolCalls, [{ id: 'text_1', name: 'read_file', input: { path: 'a.js' } }]);
  assert.equal(r.stopReason, 'tool_use');
});

// A coder session whose model describes tool calls in prose asks for the text
// protocol per call (spec 4.3, live run r_20260909_b4d9777c). The instance must
// not latch: its other candidates keep the native tool API.
const FINISH_REPLY = '<<<tool name="finish">>>\n{"summary":"fixed the divisor"}\n<<<end>>>';

test('textTools on a single call speaks the protocol for that call only', async () => {
  const fetchImpl = fakeFetch([textCompletion(FINISH_REPLY), okResponse(chatCompletion())]);
  const p = createOpenAICompatibleProvider('ollama:mistral:latest', cfg(), { fetchImpl });

  const r = await p.chat({ system: 'be brief', tools: TEXT_TOOLS, textTools: true });
  const body = fetchImpl.requests[0].body;
  assert.equal('tools' in body, false, 'no native tools go on the wire');
  assert.match(body.messages[0].content, /^be brief\n\nTOOL CALLING PROTOCOL \(text mode, v1\)/);
  assert.equal(body.messages[0].content.includes('1. read_file - Read a file.'), true);
  assert.equal(r.meta.textTools, true);
  assert.equal(r.stopReason, 'tool_use');
  assert.deepEqual(r.toolCalls, [{ id: 'text_1', name: 'finish', input: { summary: 'fixed the divisor' } }]);
  assert.deepEqual(r.content, [
    { type: 'tool_use', id: 'text_1', name: 'finish', input: { summary: 'fixed the divisor' } },
  ]);

  // Nothing latched: the same instance's next call is native again.
  assert.equal(p.textTools, false);
  const next = await p.chat({ system: 'be brief', tools: TEXT_TOOLS });
  assert.equal(Array.isArray(fetchImpl.requests[1].body.tools), true);
  assert.equal(next.meta.textTools, false);
});

test('the Anthropic adapter honours textTools on a single call too', async () => {
  const client = fakeAnthropicClient((params) =>
    anthropicMessage({
      content: [{ type: 'text', text: 'tools' in params ? 'native' : FINISH_REPLY }],
    }),
  );
  const p = createAnthropicProvider('anthropic:claude-fable-5-1', cfg({ anthropic: 'sk-ant' }), { client });

  const r = await p.chat({ system: 'be brief', tools: TEXT_TOOLS, textTools: true });
  assert.equal('tools' in client.calls[0].params, false);
  assert.match(client.calls[0].params.system[0].text, /^be brief\n\nTOOL CALLING PROTOCOL/);
  assert.equal(r.meta.textTools, true);
  assert.deepEqual(r.toolCalls, [{ id: 'text_1', name: 'finish', input: { summary: 'fixed the divisor' } }]);

  assert.equal(p.textTools, false);
  const next = await p.chat({ system: 'be brief', tools: TEXT_TOOLS });
  assert.equal(Array.isArray(client.calls[1].params.tools), true);
  assert.equal(next.meta.textTools, false);
  assert.deepEqual(next.toolCalls, []);
});

// ===========================================================================
// streaming (spec 3.4)
// ===========================================================================

/** A fetch Response whose body is a ReadableStream of SSE chunks. */
function sseResponse(events) {
  const encoder = new TextEncoder();
  const frames = events.map(
    (e) => 'data: ' + (typeof e === 'string' ? e : JSON.stringify(e)) + '\n\n',
  );
  return {
    status: 200,
    body: new ReadableStream({
      start(controller) {
        for (const frame of frames) controller.enqueue(encoder.encode(frame));
        controller.close();
      },
    }),
    json: async () => {
      throw new Error('a streamed response must not be read as JSON');
    },
  };
}

test('a long max_tokens streams and the deltas are reassembled', async () => {
  const fetchImpl = fakeFetch([
    sseResponse([
      { id: 'cmpl_9', model: 'qwen-3.8-27b', choices: [{ index: 0, delta: { role: 'assistant', content: 'Hel' } }] },
      { choices: [{ index: 0, delta: { content: 'lo ' } }] },
      { choices: [{ index: 0, delta: { content: 'there' } }] },
      {
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                { index: 0, id: 'call_1', type: 'function', function: { name: 'write', arguments: '{"pa' } },
              ],
            },
          },
        ],
      },
      {
        choices: [
          { index: 0, delta: { tool_calls: [{ index: 0, function: { name: '_file', arguments: 'th":"a' } }] } },
        ],
      },
      { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '.js"}' } }] } }] },
      { choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] },
      { choices: [], usage: { prompt_tokens: 120, completion_tokens: 34 } },
      '[DONE]',
    ]),
  ]);
  const p = createOpenAICompatibleProvider('cerebras:qwen-3.8-27b', cfg({ cerebras: 'csk' }), {
    fetchImpl,
  });
  const r = await p.chat({ maxTokens: 32000, tools: TEXT_TOOLS });
  const body = fetchImpl.requests[0].body;
  assert.equal(body.stream, true);
  assert.deepEqual(body.stream_options, { include_usage: true });
  assert.equal(r.meta.streamed, true);
  assert.equal(r.text, 'Hello there');
  assert.deepEqual(r.toolCalls, [{ id: 'call_1', name: 'write_file', input: { path: 'a.js' } }]);
  assert.equal(r.stopReason, 'tool_use');
  assert.deepEqual(r.usage, { input: 120, output: 34, cacheRead: 0, cacheWrite: 0 });
  assert.equal(r.model, 'qwen-3.8-27b');
});

test('config.stream forces streaming and a missing usage chunk is estimated', async () => {
  const fetchImpl = fakeFetch([
    sseResponse([
      { choices: [{ index: 0, delta: { content: 'short answer' } }] },
      { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] },
      '[DONE]',
    ]),
  ]);
  const p = createOpenAICompatibleProvider(
    'cerebras:qwen-3.8-27b',
    cfg({ cerebras: 'csk' }, { stream: true }),
    { fetchImpl },
  );
  const r = await p.chat({ messages: [{ role: 'user', content: 'hi' }] });
  assert.equal(fetchImpl.requests[0].body.stream, true);
  assert.equal(r.text, 'short answer');
  assert.equal(r.usage.estimated, true);
  assert.equal(r.usage.output > 0, true);
  assert.equal(r.usage.input > 0, true);
});

test('a partial SSE frame is skipped instead of failing the request', async () => {
  const fetchImpl = fakeFetch([
    sseResponse([
      { choices: [{ index: 0, delta: { content: 'kept' } }] },
      '{"choices":[{"delta":{"content":"lost"',
      { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }], usage: { prompt_tokens: 3, completion_tokens: 1 } },
      '[DONE]',
    ]),
  ]);
  const p = createOpenAICompatibleProvider(
    'cerebras:qwen-3.8-27b',
    cfg({ cerebras: 'csk' }, { stream: true }),
    { fetchImpl },
  );
  const r = await p.chat({});
  assert.equal(r.text, 'kept');
  assert.equal(r.usage.estimated, undefined);
});

test('a streamed reply in text mode is parsed back into tool_use blocks', async () => {
  const fetchImpl = fakeFetch([
    sseResponse([
      { choices: [{ index: 0, delta: { content: '<<<tool name="read_file">>>{"path":' } }] },
      { choices: [{ index: 0, delta: { content: '"a.js"}<<<end>>>' } }] },
      { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }], usage: { prompt_tokens: 5, completion_tokens: 2 } },
      '[DONE]',
    ]),
  ]);
  const p = createOpenAICompatibleProvider(
    'ollama:deepseek-coder-v2',
    cfg({}, { stream: true }),
    { fetchImpl },
  );
  const r = await p.chat({ tools: TEXT_TOOLS });
  assert.deepEqual(r.toolCalls, [{ id: 'text_1', name: 'read_file', input: { path: 'a.js' } }]);
  assert.equal(r.meta.streamed, true);
  assert.equal(r.meta.textTools, true);
});

// ===========================================================================
// registry additions (spec 3.5)
// ===========================================================================

test('the catalog marks the models that have no tool API', () => {
  const noTools = describeModel('ollama:deepseek-coder-v2');
  assert.equal(noTools.tools, false);
  assert.equal(typeof noTools.textToolsHint, 'string');
  assert.equal(supportsNativeTools('ollama:deepseek-coder-v2'), false);

  const withTools = describeModel('anthropic:claude-fable-5-1');
  assert.equal(withTools.tools, true);
  assert.equal(withTools.textToolsHint, null);
  assert.equal(supportsNativeTools('anthropic:claude-fable-5-1'), true);

  // An unlisted Ollama model whose name matches a known tool-less family.
  assert.equal(describeModel('ollama:deepseek-coder-v2:16b').tools, false);
  assert.equal(describeModel('ollama:qwen3.8:latest').tools, true);
});

test('an unknown openrouter ref falls back to the wildcard entry', () => {
  const wildcard = describeModel('openrouter:some/brand-new-model');
  assert.equal(wildcard.unknownPricing, true);
  assert.equal(wildcard.pricingVerified, false);
  assert.equal(wildcard.ctx > 0, true);
  assert.equal(typeof wildcard.textToolsHint, 'string');
  assert.equal(Object.prototype.hasOwnProperty.call(MODEL_CATALOG, 'openrouter:*'), true);
});

// ===========================================================================
// empty-on-length retry (thinking models that spend the budget on reasoning)
// ===========================================================================

test('an empty reply with finish_reason length is retried once with a larger budget', async () => {
  const schema = { type: 'object', properties: { ok: { type: 'boolean' } }, required: ['ok'] };
  const fetchImpl = fakeFetch([
    okResponse(
      chatCompletion({
        choices: [
          {
            index: 0,
            finish_reason: 'length',
            message: { role: 'assistant', content: '', reasoning: 'let me think about this at length' },
          },
        ],
        usage: { prompt_tokens: 300, completion_tokens: 2000 },
      }),
    ),
    // 3000 tokens -> 12000 is above the streaming threshold, so the retry streams.
    sseResponse([
      { id: 'cmpl_2', model: 'qwen3.8:latest', choices: [{ index: 0, delta: { role: 'assistant', reasoning: 'short' } }] },
      { choices: [{ index: 0, delta: { content: '{"ok":' } }] },
      { choices: [{ index: 0, delta: { content: ' true}' } }] },
      { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] },
      { choices: [], usage: { prompt_tokens: 300, completion_tokens: 40 } },
      '[DONE]',
    ]),
  ]);
  const p = createOpenAICompatibleProvider('ollama:qwen3.8:latest', cfg({}), { fetchImpl });
  const r = await p.chat({ jsonSchema: { name: 'verdict', schema }, maxTokens: 3000 });

  assert.equal(fetchImpl.requests.length, 2);
  assert.equal(fetchImpl.requests[0].body.max_tokens, 3000);
  assert.equal(fetchImpl.requests[0].body.stream, false);
  assert.equal(fetchImpl.requests[1].body.max_tokens, 12000);
  assert.equal(fetchImpl.requests[1].body.stream, true);
  assert.equal(r.meta.streamed, true);
  assert.deepEqual(r.meta.emptyRetry, { from: 3000, to: 12000 });
  assert.equal(r.meta.reasoningChars, 'let me think about this at length'.length);
  assert.equal(r.text, '{"ok": true}');
  assert.deepEqual(r.json, { ok: true });
  assert.equal(r.stopReason, 'end_turn');
  // Both calls are billed: the wasted first reply still counts.
  assert.deepEqual(r.usage, { input: 600, output: 2040, cacheRead: 0, cacheWrite: 0 });
});

test('the empty-on-length retry is capped at 16000 tokens and made only once', async () => {
  const empty = () =>
    okResponse(
      chatCompletion({
        choices: [{ index: 0, finish_reason: 'length', message: { role: 'assistant', content: '', reasoning: 'x' } }],
      }),
    );
  const fetchImpl = fakeFetch([empty(), sseResponse([
    { choices: [{ index: 0, delta: { role: 'assistant', reasoning: 'still thinking' } }] },
    { choices: [{ index: 0, delta: {}, finish_reason: 'length' }] },
    { choices: [], usage: { prompt_tokens: 10, completion_tokens: 16000 } },
    '[DONE]',
  ])]);
  const p = createOpenAICompatibleProvider('ollama:qwen3.8:latest', cfg({}), { fetchImpl });
  const r = await p.chat({ jsonSchema: { name: 'verdict', schema: { type: 'object' } }, maxTokens: 6000 });
  assert.equal(fetchImpl.requests.length, 2);
  assert.equal(fetchImpl.requests[1].body.max_tokens, 16000);
  assert.deepEqual(r.meta.emptyRetry, { from: 6000, to: 16000 });
  assert.equal(r.stopReason, 'max_tokens');
  // Reasoning is the fallback text when the retry is empty too; it is not JSON.
  assert.equal(r.text, 'still thinking');
  assert.equal(r.json, null);

  // Already at the cap: no retry at all (16000 streams, so the reply is SSE).
  const capped = fakeFetch([
    sseResponse([
      { choices: [{ index: 0, delta: { role: 'assistant', reasoning: 'x' } }] },
      { choices: [{ index: 0, delta: {}, finish_reason: 'length' }] },
      '[DONE]',
    ]),
  ]);
  const q = createOpenAICompatibleProvider('ollama:qwen3.8:latest', cfg({}), { fetchImpl: capped });
  const s = await q.chat({ maxTokens: 16000 });
  assert.equal(capped.requests.length, 1);
  assert.equal(s.meta.emptyRetry, undefined);
  assert.equal(s.stopReason, 'max_tokens');
});

test('a length-capped reply that carries content or tool calls is not retried', async () => {
  const withText = fakeFetch([
    okResponse(
      chatCompletion({
        choices: [{ index: 0, finish_reason: 'length', message: { role: 'assistant', content: 'partial' } }],
      }),
    ),
  ]);
  const p = createOpenAICompatibleProvider('ollama:qwen3.8:latest', cfg({}), { fetchImpl: withText });
  const r = await p.chat({ maxTokens: 2000 });
  assert.equal(withText.requests.length, 1);
  assert.equal(r.text, 'partial');
  assert.equal(r.stopReason, 'max_tokens');

  const withCall = fakeFetch([
    okResponse(
      chatCompletion({
        choices: [
          {
            index: 0,
            finish_reason: 'length',
            message: {
              role: 'assistant',
              content: '',
              tool_calls: [{ id: 'c1', type: 'function', function: { name: 'read_file', arguments: '{"path":"a"}' } }],
            },
          },
        ],
      }),
    ),
  ]);
  const q = createOpenAICompatibleProvider('ollama:qwen3.8:latest', cfg({}), { fetchImpl: withCall, });
  const s = await q.chat({ maxTokens: 2000, tools: TEXT_TOOLS });
  assert.equal(withCall.requests.length, 1);
  assert.deepEqual(s.toolCalls, [{ id: 'c1', name: 'read_file', input: { path: 'a' } }]);
});

test('the "reasoning" field is read like reasoning_content, on a plain reply and on a stream', async () => {
  const plain = fakeFetch(
    okResponse(
      chatCompletion({
        choices: [{ finish_reason: 'stop', message: { content: '', reasoning: 'only thoughts' } }],
      }),
    ),
  );
  const p = createOpenAICompatibleProvider('ollama:qwen3.8:latest', cfg({}), { fetchImpl: plain });
  assert.equal((await p.chat({})).text, 'only thoughts');

  const streamed = fakeFetch([
    sseResponse([
      { choices: [{ index: 0, delta: { role: 'assistant', reasoning: 'thought ' } }] },
      { choices: [{ index: 0, delta: { reasoning: 'stream' } }] },
      { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] },
      '[DONE]',
    ]),
  ]);
  const q = createOpenAICompatibleProvider('ollama:qwen3.8:latest', cfg({}), { fetchImpl: streamed });
  assert.equal((await q.chat({ maxTokens: 9000 })).text, 'thought stream');
});

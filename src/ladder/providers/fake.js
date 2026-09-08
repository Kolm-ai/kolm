// kolm code — scripted provider for tests (owner A, spec §3).
//
// `createFakeProvider({script})` implements the full provider contract without
// touching a network. `script` is either an array of responses (consumed in
// order) or a function `(call, index) => response`. Every call is recorded as a
// deep copy in `provider.calls`, so a test can assert on prompts, tools and
// schemas after the fact.
//
// A response may be:
//   'plain string'                       -> { text: 'plain string' }
//   { text }                             -> one text block, stopReason 'end_turn'
//   { json }                             -> JSON.stringify'd text + parsed json
//   { toolCalls: [{name, input, id?}] }  -> tool_use blocks, stopReason 'tool_use'
//   { stopReason: 'refusal' }            -> refusal with no content
//   { content: [...] }                   -> raw Anthropic-shaped blocks verbatim
// plus optional `usage`, `costUsd`, `latencyMs`, `model`, `error`.

import { LadderError } from '../errors.js';
import { parseRef } from '../config.js';

const ZERO_USAGE = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };

/** Deep copy of the parts of a chat call that a test may assert on. */
function snapshotCall(call = {}) {
  const picked = {
    system: call.system ?? null,
    messages: call.messages ?? null,
    tools: call.tools ?? null,
    jsonSchema: call.jsonSchema ?? null,
    maxTokens: call.maxTokens ?? null,
    temperature: call.temperature ?? null,
    effort: call.effort ?? null,
    cacheSystem: call.cacheSystem ?? null,
    timeoutMs: call.timeoutMs ?? null,
  };
  return JSON.parse(JSON.stringify(picked));
}

function normalizeUsage(usage) {
  return { ...ZERO_USAGE, ...(usage || {}) };
}

/**
 * @param {{script?: Array|Function, ref?: string, armed?: boolean,
 *   usage?: object, costUsd?: number, envVar?: string}} [opts]
 */
export function createFakeProvider(opts = {}) {
  const ref = opts.ref ?? 'fake:test';
  const { provider, model } = parseRef(ref);
  const armed = opts.armed !== false;
  const script = opts.script;
  const defaultUsage = normalizeUsage(opts.usage);
  const defaultCost = opts.costUsd ?? 0;

  const calls = [];
  let index = 0;
  let toolIdSeq = 0;

  function nextResponse(call) {
    if (typeof script === 'function') return script(call, index);
    if (Array.isArray(script)) {
      if (index >= script.length) {
        throw new Error(
          `fake provider ${ref} exhausted: no scripted response for call ${index + 1} ` +
            `(script has ${script.length})`,
        );
      }
      return script[index];
    }
    if (script === undefined || script === null) return { text: 'fake' };
    return script;
  }

  function build(raw) {
    const response = typeof raw === 'string' ? { text: raw } : (raw ?? {});
    if (response.error) throw response.error;

    const content = [];
    const toolCalls = [];
    let text = '';
    let json = null;

    if (Array.isArray(response.content)) {
      for (const block of response.content) {
        content.push(block);
        if (block?.type === 'text') text += block.text ?? '';
        if (block?.type === 'tool_use') {
          toolCalls.push({ id: block.id, name: block.name, input: block.input });
        }
      }
    } else {
      if (response.json !== undefined && response.json !== null) {
        json = response.json;
        text = response.text ?? JSON.stringify(response.json);
      } else if (response.text !== undefined && response.text !== null) {
        text = String(response.text);
      }
      if (text !== '') content.push({ type: 'text', text });
      for (const tc of response.toolCalls ?? []) {
        toolIdSeq += 1;
        const id = tc.id ?? `toolu_${toolIdSeq}`;
        const input = tc.input ?? {};
        content.push({ type: 'tool_use', id, name: tc.name, input });
        toolCalls.push({ id, name: tc.name, input });
      }
    }

    let stopReason = response.stopReason;
    if (!stopReason) stopReason = toolCalls.length > 0 ? 'tool_use' : 'end_turn';

    return {
      text,
      json,
      toolCalls,
      content,
      stopReason,
      usage: normalizeUsage(response.usage ?? defaultUsage),
      costUsd: response.costUsd ?? defaultCost,
      model: response.model ?? model,
      provider,
      latencyMs: response.latencyMs ?? 0,
    };
  }

  return {
    id: ref,
    provider,
    model,
    armed,
    kind: 'fake',
    calls,
    reset() {
      calls.length = 0;
      index = 0;
      toolIdSeq = 0;
    },
    async chat(call = {}) {
      if (!armed) {
        throw new LadderError(
          'not-armed',
          `rung ${ref} is not armed: set ${opts.envVar ?? 'the provider API key'}`,
          { ref, envVar: opts.envVar ?? null },
        );
      }
      calls.push(snapshotCall(call));
      const raw = nextResponse(call);
      index += 1;
      return build(raw);
    },
  };
}

/**
 * Build a minimal instance of a JSON schema. Deterministic: object properties
 * are emitted in schema order, enums take their first member.
 */
export function minimalFromSchema(schema) {
  if (!schema || typeof schema !== 'object') return {};
  if (Array.isArray(schema.enum) && schema.enum.length > 0) return schema.enum[0];
  const type = Array.isArray(schema.type) ? schema.type[0] : schema.type;
  switch (type) {
    case 'string':
      return '';
    case 'number':
    case 'integer':
      return 0;
    case 'boolean':
      return false;
    case 'null':
      return null;
    case 'array':
      return [];
    case 'object':
    default: {
      const out = {};
      const props = schema.properties ?? {};
      const required = Array.isArray(schema.required) ? schema.required : Object.keys(props);
      for (const key of Object.keys(props)) {
        if (required.includes(key)) out[key] = minimalFromSchema(props[key]);
      }
      return out;
    }
  }
}

/**
 * Deterministic stand-in used when `KOLM_TEST_MOCK_PROVIDER=1`. Always armed,
 * always free, answers `text:'mock'` and a schema-shaped minimal object.
 * @param {string} ref
 */
export function createMockProvider(ref = 'fake:mock') {
  const { provider, model } = parseRef(ref);
  const calls = [];
  return {
    id: ref,
    provider,
    model,
    armed: true,
    kind: 'mock',
    calls,
    async chat(call = {}) {
      calls.push(snapshotCall(call));
      const schema = call.jsonSchema?.schema ?? call.jsonSchema ?? null;
      const json = call.jsonSchema ? minimalFromSchema(schema) : null;
      const text = call.jsonSchema ? JSON.stringify(json) : 'mock';
      return {
        text,
        json,
        toolCalls: [],
        content: [{ type: 'text', text }],
        stopReason: 'end_turn',
        usage: { ...ZERO_USAGE },
        costUsd: 0,
        model,
        provider,
        latencyMs: 0,
      };
    },
  };
}

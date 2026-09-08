// kolm code — Anthropic adapter (owner A, spec §3).
//
// Rules taken verbatim from the claude-api skill, not from memory:
//   * `thinking` is NEVER sent. Fable 5.1 / Opus 5 are adaptive by default and
//     `thinking.budget_tokens` is a 400 on them.
//   * `tool_choice` is NEVER sent. Forced tool use (`any` / `tool`) is a 400 on
//     Fable 5.1.
//   * effort and structured output both live under `output_config`:
//     `{ effort }` and `{ format: { type:'json_schema', schema } }`
//     (verified against @anthropic-ai/sdk 0.124.0 `OutputConfig` /
//     `JSONOutputFormat` in resources/messages/messages.d.ts).
//   * sampling parameters are rejected by Fable 5.x / Opus 5 / Sonnet 5, so
//     `temperature` is forwarded only for models the registry marks
//     `acceptsSampling`.
//   * `stop_reason === 'refusal'` is returned, never thrown.
//   * `max_tokens > 16000` must stream (non-streaming requests hit SDK HTTP
//     timeouts at high max_tokens).

import { LadderError } from '../errors.js';
import { parseRef } from '../config.js';
import { describeModel, estimateCost } from './registry.js';

const STREAM_THRESHOLD_TOKENS = 16000;
const MAX_TRIES = 3;
const BACKOFF_BASE_MS = 500;

/** Anthropic stop_reason -> unified stopReason. */
export function mapStopReason(stopReason) {
  switch (stopReason) {
    case 'stop_sequence':
      return 'stop';
    case 'model_context_window_exceeded':
      return 'max_tokens';
    case null:
    case undefined:
      return 'end_turn';
    default:
      return stopReason; // end_turn | tool_use | max_tokens | refusal | pause_turn
  }
}

/** Parse `text` as JSON; fall back to the first balanced `{...}` run. */
export function parseJsonLoose(text) {
  if (typeof text !== 'string' || text.trim() === '') return null;
  try {
    const direct = JSON.parse(text);
    return direct && typeof direct === 'object' ? direct : null;
  } catch {
    // fall through to brace extraction
  }
  const start = text.indexOf('{');
  if (start === -1) return null;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) {
        try {
          const parsed = JSON.parse(text.slice(start, i + 1));
          return parsed && typeof parsed === 'object' ? parsed : null;
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

function statusOf(err) {
  if (!err || typeof err !== 'object') return null;
  if (typeof err.status === 'number') return err.status;
  if (typeof err.statusCode === 'number') return err.statusCode;
  if (err.response && typeof err.response.status === 'number') return err.response.status;
  return null;
}

function isTimeout(err) {
  if (!err || typeof err !== 'object') return false;
  const name = String(err.name || '');
  return (
    name === 'APIConnectionTimeoutError' ||
    name === 'AbortError' ||
    name === 'TimeoutError' ||
    err.code === 'ETIMEDOUT'
  );
}

function isConnection(err) {
  if (!err || typeof err !== 'object') return false;
  const name = String(err.name || '');
  return (
    name === 'APIConnectionError' ||
    name === 'APIConnectionTimeoutError' ||
    ['ECONNRESET', 'ECONNREFUSED', 'EPIPE', 'ENOTFOUND', 'EAI_AGAIN'].includes(err.code)
  );
}

function isRetryable(err) {
  const status = statusOf(err);
  if (status === 429 || status === 529) return true;
  if (status !== null && status >= 500) return true;
  return isConnection(err) && !isTimeout(err);
}

function bodyOf(err) {
  if (!err || typeof err !== 'object') return null;
  return err.error ?? err.body ?? null;
}

function secretOf(config, name) {
  if (config && typeof config.getSecret === 'function') {
    try {
      return config.getSecret(name);
    } catch {
      return null; // deliberate: an unknown secret name simply does not arm the rung
    }
  }
  if (config && config.secrets && Object.prototype.hasOwnProperty.call(config.secrets, name)) {
    return config.secrets[name];
  }
  return null;
}

const defaultSleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * @param {string} ref e.g. 'anthropic:claude-fable-5-1'
 * @param {object} [config] from loadConfig()
 * @param {{client?: object, sleep?: Function, random?: Function, AnthropicCtor?: Function}} [opts]
 */
export function createAnthropicProvider(ref, config = {}, opts = {}) {
  const { provider, model, ref: normalized } = parseRef(ref);
  if (provider !== 'anthropic') {
    throw new LadderError('bad-ref', `anthropic adapter cannot serve ref "${ref}"`, { ref });
  }
  const apiKey = secretOf(config, 'anthropic');
  const injected = opts.client ?? null;
  const armed = Boolean(apiKey) || Boolean(injected);
  const sleep = opts.sleep ?? defaultSleep;
  const random = opts.random ?? Math.random;
  const spec = describeModel(normalized);

  let client = injected;
  async function getClient() {
    if (client) return client;
    const mod = await import('@anthropic-ai/sdk');
    const Ctor = opts.AnthropicCtor ?? mod.default;
    client = new Ctor({ apiKey });
    return client;
  }

  function buildParams(call) {
    const maxTokens = call.maxTokens ?? 8192;
    const params = {
      model,
      max_tokens: maxTokens,
      messages: call.messages ?? [],
    };

    if (call.system) {
      params.system =
        call.cacheSystem === false
          ? call.system
          : [{ type: 'text', text: call.system, cache_control: { type: 'ephemeral' } }];
    }

    // Sampling params are rejected by the Fable 5.x / Opus 5 / Sonnet 5 family.
    if (spec.acceptsSampling && call.temperature !== undefined && call.temperature !== null) {
      params.temperature = call.temperature;
    }

    const outputConfig = {};
    if (call.effort) outputConfig.effort = call.effort;
    if (call.jsonSchema) {
      const schema = call.jsonSchema.schema ?? call.jsonSchema;
      outputConfig.format = { type: 'json_schema', schema };
    }
    if (Object.keys(outputConfig).length > 0) params.output_config = outputConfig;

    if (Array.isArray(call.tools) && call.tools.length > 0) {
      params.tools = call.tools.map((t) => ({
        name: t.name,
        description: t.description,
        input_schema: t.input_schema ?? t.inputSchema,
      }));
    }
    // Deliberately absent: `thinking` (adaptive by default) and `tool_choice`
    // (`any`/`tool` are a 400 on Fable 5.1).
    return params;
  }

  function mapError(err) {
    const status = statusOf(err);
    const message = String(err?.message ?? err);
    if (isTimeout(err)) {
      return new LadderError('provider-timeout', `${normalized}: ${message}`, { ref: normalized });
    }
    if (status === 400 || status === 422) {
      return new LadderError('provider-bad-request', `${normalized}: ${message}`, {
        ref: normalized,
        status,
        body: bodyOf(err),
      });
    }
    if (status === 401 || status === 403) {
      return new LadderError(
        'not-armed',
        `${normalized}: ANTHROPIC_API_KEY rejected (HTTP ${status})`,
        { ref: normalized, status, envVar: 'ANTHROPIC_API_KEY' },
      );
    }
    return new LadderError('provider-unavailable', `${normalized}: ${message}`, {
      ref: normalized,
      status,
      body: bodyOf(err),
    });
  }

  async function send(params, requestOptions, useStream) {
    const c = await getClient();
    let lastErr = null;
    for (let attempt = 0; attempt < MAX_TRIES; attempt += 1) {
      try {
        if (useStream) {
          const stream = requestOptions
            ? c.messages.stream(params, requestOptions)
            : c.messages.stream(params);
          return await stream.finalMessage();
        }
        return requestOptions
          ? await c.messages.create(params, requestOptions)
          : await c.messages.create(params);
      } catch (err) {
        // A cancellation the caller asked for is not a provider fault: propagate
        // it unchanged so the orchestrator can tell abort from failure.
        if (requestOptions?.signal?.aborted && isTimeout(err)) throw err;
        lastErr = err;
        if (attempt === MAX_TRIES - 1 || !isRetryable(err)) break;
        const delay = BACKOFF_BASE_MS * 2 ** attempt;
        await sleep(Math.round(delay + random() * delay * 0.25));
      }
    }
    throw mapError(lastErr);
  }

  return {
    id: normalized,
    provider,
    model,
    armed,
    kind: 'anthropic',
    spec,
    async chat(call = {}) {
      if (!armed) {
        throw new LadderError(
          'not-armed',
          `rung ${normalized} is not armed: set ANTHROPIC_API_KEY`,
          { ref: normalized, envVar: 'ANTHROPIC_API_KEY' },
        );
      }
      const started = Date.now();
      const params = buildParams(call);
      const requestOptions = {};
      if (call.signal) requestOptions.signal = call.signal;
      if (call.timeoutMs) requestOptions.timeout = call.timeoutMs;
      const useStream = (call.maxTokens ?? 8192) > STREAM_THRESHOLD_TOKENS;

      const message = await send(
        params,
        Object.keys(requestOptions).length > 0 ? requestOptions : null,
        useStream,
      );

      const content = Array.isArray(message?.content) ? message.content : [];
      let text = '';
      const toolCalls = [];
      for (const block of content) {
        if (block?.type === 'text') text += block.text ?? '';
        else if (block?.type === 'tool_use') {
          toolCalls.push({ id: block.id, name: block.name, input: block.input });
        }
      }

      const rawUsage = message?.usage ?? {};
      const usage = {
        input: rawUsage.input_tokens ?? 0,
        output: rawUsage.output_tokens ?? 0,
        cacheRead: rawUsage.cache_read_input_tokens ?? 0,
        cacheWrite: rawUsage.cache_creation_input_tokens ?? 0,
      };

      return {
        text,
        json: call.jsonSchema ? parseJsonLoose(text) : null,
        toolCalls,
        content,
        stopReason: mapStopReason(message?.stop_reason),
        stopDetails: message?.stop_details ?? null,
        usage,
        costUsd: estimateCost(normalized, usage, { offpeak: Boolean(config.deepseekOffpeak) }),
        model: message?.model ?? model,
        provider,
        latencyMs: Date.now() - started,
      };
    },
  };
}

// kolm code - Anthropic adapter (owner PROV, spec section 3).
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
//
// Spec section 3.3 additions: every request goes through the shared per-provider
// limiter, retries honour `Retry-After`, and a 429 halves concurrency for a
// minute. Spec section 3.2: the text tool protocol can be forced onto any ref through
// KOLM_CODE_TEXT_TOOLS, and a 400 that says tools are unsupported latches it on.
// Spec 4.3: a single call can ask for it with `textTools: true`, which speaks the
// protocol for that call only and leaves the instance latch alone.

import { LadderError } from '../errors.js';
import { parseRef } from '../config.js';
import { describeModel, estimateCost } from './registry.js';
import { limiterFor, retryDelayMs, THROTTLE_PENALTY_MS } from './limiter.js';
import {
  looksLikeToolsUnsupported,
  parseTextToolCalls,
  refForcedToTextTools,
  renderJsonOnlyForText,
  renderToolsForText,
  toTextProtocolMessages,
} from './text-tools.js';

const STREAM_THRESHOLD_TOKENS = 16000;

// Spec section 3.3: throttling and overload are worth four tries, any other failure two.
const MAX_TRIES_THROTTLED = 4;
const MAX_TRIES_OTHER = 2;
const ABSOLUTE_MAX_TRIES = MAX_TRIES_THROTTLED;

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

function headersOf(err) {
  if (!err || typeof err !== 'object') return null;
  return err.headers ?? err.response?.headers ?? null;
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

function triesFor(status) {
  return status === 429 || status === 529 || status === 503 ? MAX_TRIES_THROTTLED : MAX_TRIES_OTHER;
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
 * @param {object} [config] from loadConfig(); reads `textTools`,
 *   `providerConcurrency` and `stream` defensively (all optional)
 * @param {{client?: object, sleep?: Function, random?: Function, now?: Function,
 *   AnthropicCtor?: Function, limiter?: object}} [opts]
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
  const now = opts.now ?? Date.now;
  const spec = describeModel(normalized);
  const limiter = opts.limiter ?? limiterFor(provider, config, { sleep, now });

  // Anthropic models all have a tool API, so this is only ever on because the
  // operator forced it (KOLM_CODE_TEXT_TOOLS) or a gateway in front of the API
  // rejected `tools`. Supporting it here keeps one code path across adapters.
  let textTools = spec.tools === false || refForcedToTextTools(normalized, config.textTools);

  let client = injected;
  async function getClient() {
    if (client) return client;
    const mod = await import('@anthropic-ai/sdk');
    const Ctor = opts.AnthropicCtor ?? mod.default;
    client = new Ctor({ apiKey });
    return client;
  }

  // `useText` is decided per call by `chat`: the instance latch OR this call's
  // own `textTools: true`, which is how the coder moves ONE session to the text
  // protocol without dragging the shared instance with it (spec 4.3).
  function buildParams(call, useText = false) {
    const maxTokens = call.maxTokens ?? 8192;
    const tools = Array.isArray(call.tools) ? call.tools : [];
    const params = {
      model,
      max_tokens: maxTokens,
      messages: useText ? toTextProtocolMessages(call.messages ?? []) : (call.messages ?? []),
    };

    let system = call.system ?? null;
    if (useText) {
      // In text mode the protocol, the tool list and any JSON-only rule all ride
      // in the system prompt; nothing tool-shaped reaches the wire.
      const extra = [];
      if (tools.length > 0) extra.push(renderToolsForText(tools));
      if (call.jsonSchema) extra.push(renderJsonOnlyForText(call.jsonSchema));
      if (extra.length > 0) system = [system, ...extra].filter(Boolean).join('\n\n');
    }
    if (system) {
      params.system =
        call.cacheSystem === false
          ? system
          : [{ type: 'text', text: system, cache_control: { type: 'ephemeral' } }];
    }

    // Sampling params are rejected by the Fable 5.x / Opus 5 / Sonnet 5 family.
    if (spec.acceptsSampling && call.temperature !== undefined && call.temperature !== null) {
      params.temperature = call.temperature;
    }

    const outputConfig = {};
    if (call.effort) outputConfig.effort = call.effort;
    if (call.jsonSchema && !useText) {
      const schema = call.jsonSchema.schema ?? call.jsonSchema;
      outputConfig.format = { type: 'json_schema', schema };
    }
    if (Object.keys(outputConfig).length > 0) params.output_config = outputConfig;

    if (!useText && tools.length > 0) {
      params.tools = tools.map((t) => ({
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

  async function send(params, requestOptions, useStream, meta) {
    const c = await getClient();
    // The gate wraps one attempt, not the retry loop: a request sleeping out a
    // backoff must not hold a concurrency slot the whole time.
    const attemptOnce = () =>
      limiter.run(
        () => {
          if (useStream) {
            const stream = requestOptions
              ? c.messages.stream(params, requestOptions)
              : c.messages.stream(params);
            return stream.finalMessage();
          }
          return requestOptions
            ? c.messages.create(params, requestOptions)
            : c.messages.create(params);
        },
        { signal: requestOptions?.signal },
      );

    let lastErr = null;
    for (let attempt = 0; attempt < ABSOLUTE_MAX_TRIES; attempt += 1) {
      try {
        return await attemptOnce();
      } catch (err) {
        // A cancellation the caller asked for is not a provider fault: propagate
        // it unchanged so the orchestrator can tell abort from failure.
        if (requestOptions?.signal?.aborted && isTimeout(err)) throw err;
        lastErr = err;
        const status = statusOf(err);
        if (status === 429) {
          meta.throttled = true;
          limiter.penalize({ ms: THROTTLE_PENALTY_MS });
        }
        if (!isRetryable(err) || attempt + 1 >= triesFor(status)) break;
        await sleep(retryDelayMs(attempt, headersOf(err), { random, now }));
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
    /** True once this instance is speaking the text tool protocol. */
    get textTools() {
      return textTools;
    },
    async chat(call = {}) {
      if (!armed) {
        throw new LadderError(
          'not-armed',
          `rung ${normalized} is not armed: set ANTHROPIC_API_KEY`,
          { ref: normalized, envVar: 'ANTHROPIC_API_KEY' },
        );
      }
      const started = Date.now();
      const requestOptions = {};
      if (call.signal) requestOptions.signal = call.signal;
      if (call.timeoutMs) requestOptions.timeout = call.timeoutMs;
      const options = Object.keys(requestOptions).length > 0 ? requestOptions : null;
      const useStream = (call.maxTokens ?? 8192) > STREAM_THRESHOLD_TOKENS || config.stream === true;
      // Per-call request for the text protocol (spec 4.3). It never latches the
      // instance: the next call without the flag speaks native tools again.
      let useText = textTools || call.textTools === true;
      const meta = { textTools: useText, throttled: false, streamed: useStream };
      const hasTools = Array.isArray(call.tools) && call.tools.length > 0;

      let message;
      try {
        message = await send(buildParams(call, useText), options, useStream, meta);
      } catch (err) {
        if (!useText && hasTools && looksLikeToolsUnsupported(err)) {
          // Latch the text protocol on for the life of this instance and retry this
          // same call once; providers keep no store the caller can read, so the only
          // signal out is meta.textTools.
          textTools = true;
          useText = true;
          meta.textTools = true;
          message = await send(buildParams(call, useText), options, useStream, meta);
        } else {
          throw err;
        }
      }

      const blocks = Array.isArray(message?.content) ? message.content : [];
      let text = '';
      const toolCalls = [];
      let content = blocks;
      for (const block of blocks) {
        if (block?.type === 'text') text += block.text ?? '';
        else if (block?.type === 'tool_use') {
          toolCalls.push({ id: block.id, name: block.name, input: block.input });
        }
      }

      if (meta.textTools) {
        // Text mode: the calls are in the text. Parse them out and rebuild the
        // tool_use blocks the coder's transcript expects.
        const parsed = parseTextToolCalls(text);
        text = parsed.text;
        content = [];
        if (text !== '') content.push({ type: 'text', text });
        for (const tc of parsed.toolCalls) {
          content.push({ type: 'tool_use', id: tc.id, name: tc.name, input: tc.input });
          toolCalls.push({ id: tc.id, name: tc.name, input: tc.input });
        }
        if (parsed.errors.length > 0) meta.textToolErrors = parsed.errors;
      }

      const rawUsage = message?.usage ?? {};
      const usage = {
        input: rawUsage.input_tokens ?? 0,
        output: rawUsage.output_tokens ?? 0,
        cacheRead: rawUsage.cache_read_input_tokens ?? 0,
        cacheWrite: rawUsage.cache_creation_input_tokens ?? 0,
      };

      let stopReason = mapStopReason(message?.stop_reason);
      if (meta.textTools && toolCalls.length > 0 && stopReason === 'end_turn') {
        stopReason = 'tool_use';
      }

      return {
        text,
        json: call.jsonSchema ? parseJsonLoose(text) : null,
        toolCalls,
        content,
        stopReason,
        stopDetails: message?.stop_details ?? null,
        usage,
        costUsd: estimateCost(normalized, usage, { offpeak: Boolean(config.deepseekOffpeak) }),
        model: message?.model ?? model,
        provider,
        latencyMs: Date.now() - started,
        meta,
      };
    },
  };
}

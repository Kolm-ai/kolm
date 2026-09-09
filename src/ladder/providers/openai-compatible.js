// kolm code - OpenAI-compatible adapter (owner PROV, spec section 3).
//
// Serves cerebras / deepseek / openrouter / ollama through
// `POST {base}/chat/completions`. Translates the unified Anthropic-shaped block
// vocabulary in both directions:
//   assistant tool_use  <-> `tool_calls[{id, type:'function', function:{name, arguments}}]`
//   user tool_result    <-> separate `{role:'tool', tool_call_id, content}` messages
//   tools               <-> `[{type:'function', function:{name, description, parameters}}]`
//
// Three behaviours the ladder depends on live here:
//   * text tool protocol (spec section 3.1/section 3.2) for endpoints with no tool API, chosen
//     at construction from the registry or forced by config, flipped to at
//     runtime when a 400 says tools are unsupported, and requestable for a single
//     call with `textTools: true` (spec 4.3) without latching the instance;
//   * one limiter per provider (spec section 3.3), with Retry-After-aware backoff and a
//     429 that halves concurrency for a minute;
//   * streaming (spec section 3.4) for long outputs, where a non-streaming request is
//     what hits gateway timeouts.
//
// `fetch` is injectable so every test runs offline.

import { LadderError, isLadderError } from '../errors.js';
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

// Spec section 3.3: a 429/529/503 is worth four tries, any other 5xx two. Nothing gets
// more than four, so a wedged endpoint cannot hold a candidate hostage.
const MAX_TRIES_THROTTLED = 4;
const MAX_TRIES_OTHER = 2;
const ABSOLUTE_MAX_TRIES = MAX_TRIES_THROTTLED;

/** Above this many max_tokens a non-streaming request risks a gateway timeout. */
const STREAM_THRESHOLD_TOKENS = 8000;

// Deliberately equal to the threshold and to the coder's own default: a caller
// that names no cap is not asking for a long output, so it must not stream.
const DEFAULT_MAX_TOKENS = 8000;

/** Characters per token used only when an endpoint omits usage on a stream. */
const CHARS_PER_TOKEN = 3.6;

/**
 * Ceiling for the one retry made when a thinking model spends its whole
 * budget on reasoning and returns an empty message with finish_reason
 * "length". Four times the original budget, never above this.
 */
const EMPTY_RETRY_MAX_TOKENS = 16000;

const DEFAULT_BASE_URLS = {
  cerebras: 'https://api.cerebras.ai/v1',
  deepseek: 'https://api.deepseek.com/v1',
  openrouter: 'https://openrouter.ai/api/v1',
  ollama: 'http://127.0.0.1:11434/v1',
};

const KEY_ENV = {
  cerebras: 'CEREBRAS_API_KEY',
  deepseek: 'DEEPSEEK_API_KEY',
  openrouter: 'OPENROUTER_API_KEY',
};

/** Providers this adapter can serve. */
export const OPENAI_COMPATIBLE_PROVIDERS = Object.freeze(['cerebras', 'deepseek', 'openrouter', 'ollama']);

/** OpenAI finish_reason -> unified stopReason. */
export function mapFinishReason(finishReason) {
  switch (finishReason) {
    case 'stop':
      return 'end_turn';
    case 'tool_calls':
    case 'function_call':
      return 'tool_use';
    case 'length':
      return 'max_tokens';
    case 'content_filter':
      return 'refusal';
    case null:
    case undefined:
      return 'end_turn';
    default:
      return finishReason;
  }
}

/** Remove DeepSeek / Qwen `<think>...</think>` reasoning from visible text. */
export function stripThinking(text) {
  if (typeof text !== 'string' || text === '') return '';
  return text
    .replace(/<think>[\s\S]*?<\/think>/g, '')
    .replace(/<think>[\s\S]*$/, '')
    .trim();
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

function blockText(content) {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .filter((b) => b?.type === 'text')
    .map((b) => b.text ?? '')
    .join('');
}

/**
 * Reasoning text of a chat message. DeepSeek and vLLM use "reasoning_content";
 * Ollama's OpenAI layer (and OpenRouter) use "reasoning". Empty string when
 * the message carries neither.
 */
export function reasoningTextOf(message) {
  if (!message || typeof message !== 'object') return '';
  if (typeof message.reasoning_content === 'string') return message.reasoning_content;
  if (typeof message.reasoning === 'string') return message.reasoning;
  return '';
}

/**
 * True when the reply hit its token cap with nothing usable in it: no visible
 * content and no tool calls. Thinking models do this when the whole budget
 * went to reasoning; the caller gets one retry with a larger budget.
 */
export function isEmptyOnLength(data) {
  const choice = data?.choices?.[0];
  if (!choice || choice.finish_reason !== 'length') return false;
  const message = choice.message ?? {};
  if (Array.isArray(message.tool_calls) && message.tool_calls.length > 0) return false;
  const raw = typeof message.content === 'string' ? message.content : blockText(message.content);
  return stripThinking(raw).trim() === '';
}

/** Budget for the empty-on-length retry: 4x the original, capped. */
export function emptyRetryBudget(from) {
  const base = Number.isFinite(from) && from > 0 ? from : DEFAULT_MAX_TOKENS;
  return Math.min(Math.max(base * 4, DEFAULT_MAX_TOKENS), EMPTY_RETRY_MAX_TOKENS);
}

/**
 * Usage of two calls that served one chat() (the empty-on-length retry). The
 * cache-hit fields are kept from the later call; the counts are summed so the
 * cost ledger sees every token the provider billed.
 */
export function sumUsage(prior, current) {
  if (!prior || typeof prior !== 'object') return current ?? {};
  const cur = current ?? {};
  return {
    ...cur,
    prompt_tokens: (prior.prompt_tokens ?? 0) + (cur.prompt_tokens ?? 0),
    completion_tokens: (prior.completion_tokens ?? 0) + (cur.completion_tokens ?? 0),
  };
}

function toolResultToString(content) {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blockText(content);
    return text === '' ? JSON.stringify(content) : text;
  }
  if (content === undefined || content === null) return '';
  return JSON.stringify(content);
}

/**
 * Anthropic-shaped history -> OpenAI chat messages.
 * @param {string|null} system
 * @param {Array} messages
 */
export function toOpenAIMessages(system, messages = []) {
  const out = [];
  if (system) out.push({ role: 'system', content: system });

  for (const message of messages) {
    const { role, content } = message ?? {};
    if (typeof content === 'string') {
      out.push({ role, content });
      continue;
    }
    const blocks = Array.isArray(content) ? content : [];

    if (role === 'assistant') {
      const toolCalls = [];
      for (const block of blocks) {
        if (block?.type === 'tool_use') {
          toolCalls.push({
            id: block.id,
            type: 'function',
            function: { name: block.name, arguments: JSON.stringify(block.input ?? {}) },
          });
        }
      }
      const text = blockText(blocks);
      // Null content is only legal next to tool_calls: Ollama's OpenAI layer
      // rejects a bare null ("invalid message content type: <nil>"), so an
      // assistant turn that produced neither text nor a call goes out as ''.
      const msg = { role: 'assistant', content: text === '' && toolCalls.length > 0 ? null : text };
      if (toolCalls.length > 0) msg.tool_calls = toolCalls;
      out.push(msg);
      continue;
    }

    // user (or any non-assistant role): tool_result blocks become tool messages
    for (const block of blocks) {
      if (block?.type === 'tool_result') {
        out.push({
          role: 'tool',
          tool_call_id: block.tool_use_id,
          content: toolResultToString(block.content),
        });
      }
    }
    const text = blockText(blocks);
    if (text !== '') out.push({ role: role ?? 'user', content: text });
  }
  return out;
}

/** Unified tool definitions -> OpenAI function tools. */
export function toOpenAITools(tools = []) {
  return tools.map((t) => ({
    type: 'function',
    function: {
      name: t.name,
      description: t.description ?? '',
      parameters: t.input_schema ?? t.inputSchema ?? { type: 'object', properties: {} },
    },
  }));
}

function isAbortError(err) {
  return Boolean(err) && (err.name === 'AbortError' || err.name === 'TimeoutError');
}

function isRetryableStatus(status) {
  return status === 429 || status === 529 || (typeof status === 'number' && status >= 500);
}

function triesFor(status) {
  return status === 429 || status === 529 || status === 503 ? MAX_TRIES_THROTTLED : MAX_TRIES_OTHER;
}

function secretOf(config, name) {
  if (config && typeof config.getSecret === 'function') {
    try {
      return config.getSecret(name);
    } catch {
      return null; // deliberate: unknown secret name simply leaves the rung unarmed
    }
  }
  if (config && config.secrets && Object.prototype.hasOwnProperty.call(config.secrets, name)) {
    return config.secrets[name];
  }
  return null;
}

const defaultSleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// ---------------------------------------------------------------------------
// streaming
// ---------------------------------------------------------------------------

/** Read a fetch Response body to a string, whatever stream flavour it exposes. */
async function readBodyText(res) {
  const body = res?.body;
  const decoder = new TextDecoder();
  const decode = (chunk) =>
    typeof chunk === 'string' ? chunk : decoder.decode(chunk, { stream: true });

  if (body && typeof body.getReader === 'function') {
    const reader = body.getReader();
    let text = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      if (value !== undefined && value !== null) text += decode(value);
    }
    return text + decoder.decode();
  }
  if (body && typeof body[Symbol.asyncIterator] === 'function') {
    let text = '';
    for await (const chunk of body) text += decode(chunk);
    return text + decoder.decode();
  }
  if (typeof res?.text === 'function') return await res.text();
  return '';
}

/** SSE text -> the JSON payloads it carried, `[DONE]` and partial frames skipped. */
function parseSseEvents(raw) {
  const events = [];
  for (const line of String(raw ?? '').split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) continue;
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') continue;
    try {
      events.push(JSON.parse(payload));
    } catch {
      // deliberate: a truncated frame is not a failed request
    }
  }
  return events;
}

/**
 * Fold streamed deltas back into the non-streaming completion shape, so the
 * reply-building code below has exactly one input format.
 */
function assembleStreamedCompletion(events, body) {
  let id = null;
  let model = null;
  let finishReason = null;
  let content = '';
  let reasoning = '';
  let usage = null;
  const byIndex = new Map();

  for (const event of events) {
    if (event?.id) id = event.id;
    if (event?.model) model = event.model;
    if (event?.usage) usage = event.usage;
    const choice = event?.choices?.[0];
    if (!choice) continue;
    if (choice.finish_reason) finishReason = choice.finish_reason;
    const delta = choice.delta ?? choice.message ?? {};
    if (typeof delta.content === 'string') content += delta.content;
    if (typeof delta.reasoning_content === 'string') reasoning += delta.reasoning_content;
    else if (typeof delta.reasoning === 'string') reasoning += delta.reasoning;
    for (const tc of delta.tool_calls ?? []) {
      const index = Number.isInteger(tc?.index) ? tc.index : byIndex.size;
      const current = byIndex.get(index) ?? { id: null, name: null, args: '' };
      if (tc?.id) current.id = tc.id;
      const namePart = tc?.function?.name;
      if (namePart) {
        // Most endpoints send the whole name in the first delta; a few split it.
        if (!current.name) current.name = namePart;
        else if (!current.name.endsWith(namePart)) current.name += namePart;
      }
      if (typeof tc?.function?.arguments === 'string') current.args += tc.function.arguments;
      byIndex.set(index, current);
    }
  }

  const toolCalls = [...byIndex.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, tc], i) => ({
      id: tc.id ?? `call_${i + 1}`,
      type: 'function',
      function: { name: tc.name ?? '', arguments: tc.args },
    }));

  const message = { role: 'assistant', content: content === '' ? null : content };
  if (reasoning !== '') message.reasoning_content = reasoning;
  if (toolCalls.length > 0) message.tool_calls = toolCalls;

  const payload = {
    id,
    model,
    choices: [
      {
        index: 0,
        finish_reason: finishReason ?? (toolCalls.length > 0 ? 'tool_calls' : 'stop'),
        message,
      },
    ],
    usage,
  };

  if (!usage) {
    // `stream_options.include_usage` is not universal; a rough count beats a
    // zero that would silently under-report the run's cost.
    const promptChars = JSON.stringify(body?.messages ?? []).length;
    const outChars =
      content.length +
      toolCalls.reduce((n, tc) => n + tc.function.arguments.length + tc.function.name.length, 0);
    payload.usage = {
      prompt_tokens: Math.ceil(promptChars / CHARS_PER_TOKEN),
      completion_tokens: Math.ceil(outChars / CHARS_PER_TOKEN),
    };
    payload.__estimatedUsage = true;
  }
  return payload;
}

// ---------------------------------------------------------------------------
// adapter
// ---------------------------------------------------------------------------

/**
 * @param {string} ref e.g. 'deepseek:deepseek-v4-flash'
 * @param {object} [config] from loadConfig(); reads `textTools`, `providerConcurrency`
 *   and `stream` defensively (all optional)
 * @param {{fetchImpl?: Function, sleep?: Function, random?: Function, now?: Function,
 *   baseUrl?: string, limiter?: object}} [opts]
 */
export function createOpenAICompatibleProvider(ref, config = {}, opts = {}) {
  const { provider, model, ref: normalized } = parseRef(ref);
  const baseUrl = String(
    opts.baseUrl ?? config.baseUrls?.[provider] ?? DEFAULT_BASE_URLS[provider] ?? '',
  ).replace(/\/+$/, '');
  if (!baseUrl) {
    throw new LadderError('bad-ref', `no base URL known for provider "${provider}"`, { ref });
  }
  const needsKey = provider !== 'ollama';
  const apiKey = needsKey ? secretOf(config, provider) : null;
  const envVar = KEY_ENV[provider] ?? null;
  const armed = needsKey ? Boolean(apiKey) : true;
  const fetchImpl = opts.fetchImpl ?? globalThis.fetch;
  const sleep = opts.sleep ?? defaultSleep;
  const random = opts.random ?? Math.random;
  const now = opts.now ?? Date.now;
  const spec = describeModel(normalized);
  const limiter = opts.limiter ?? limiterFor(provider, config, { sleep, now });

  // Text tool protocol: chosen at construction from the catalog or from
  // KOLM_CODE_TEXT_TOOLS, and latched on for the rest of the process if the
  // endpoint answers 400 to `tools`.
  let textTools = spec.tools === false || refForcedToTextTools(normalized, config.textTools);

  function headers() {
    const h = { 'content-type': 'application/json' };
    if (apiKey) h.authorization = `Bearer ${apiKey}`;
    if (provider === 'openrouter') {
      h['HTTP-Referer'] = 'https://kolm.ai';
      h['X-Title'] = 'kolm code';
    }
    return h;
  }

  function wantsStream(call) {
    return (call.maxTokens ?? DEFAULT_MAX_TOKENS) > STREAM_THRESHOLD_TOKENS || config.stream === true;
  }

  // `useText` is decided per call by `chat`: the instance latch OR this call's
  // own `textTools: true`, which is how the coder moves ONE session to the text
  // protocol without dragging the shared instance with it (spec 4.3).
  function buildBody(call, { strict, useText = false }) {
    const tools = Array.isArray(call.tools) ? call.tools : [];
    const stream = wantsStream(call);
    let system = call.system ?? null;
    let messages = call.messages ?? [];

    if (useText) {
      // The protocol and the tool list ride in the system prompt, and the
      // history is flattened so no tool_use / tool_result block reaches the wire.
      messages = toTextProtocolMessages(messages);
      const extra = [];
      if (tools.length > 0) extra.push(renderToolsForText(tools));
      if (call.jsonSchema) extra.push(renderJsonOnlyForText(call.jsonSchema));
      if (extra.length > 0) system = [system, ...extra].filter(Boolean).join('\n\n');
    }

    const body = {
      model,
      messages: toOpenAIMessages(system, messages),
      max_tokens: call.maxTokens ?? DEFAULT_MAX_TOKENS,
      stream,
    };
    if (stream) body.stream_options = { include_usage: true };
    if (call.temperature !== undefined && call.temperature !== null) {
      body.temperature = call.temperature;
    }
    if (!useText && tools.length > 0) {
      body.tools = toOpenAITools(tools);
    }
    if (call.jsonSchema && !useText) {
      const schema = call.jsonSchema.schema ?? call.jsonSchema;
      const name = call.jsonSchema.name ?? 'response';
      body.response_format = strict
        ? { type: 'json_schema', json_schema: { name, schema, strict: true } }
        : { type: 'json_schema', json_schema: { name, schema } };
    }
    return body;
  }

  function mapError(err) {
    if (isLadderError(err)) return err;
    const status = err?.status ?? null;
    const message = String(err?.message ?? err);
    if (status === 400 || status === 422) {
      return new LadderError('provider-bad-request', `${normalized}: ${message}`, {
        ref: normalized,
        status,
        body: err?.body ?? null,
      });
    }
    if (status === 401 || status === 403) {
      return new LadderError(
        'not-armed',
        `${normalized}: ${envVar ?? 'API key'} rejected (HTTP ${status})`,
        { ref: normalized, status, envVar },
      );
    }
    return new LadderError('provider-unavailable', `${normalized}: ${message}`, {
      ref: normalized,
      status,
      body: err?.body ?? null,
    });
  }

  async function request(body, call) {
    if (typeof fetchImpl !== 'function') {
      throw new LadderError('provider-unavailable', `${normalized}: fetch is unavailable`, {
        ref: normalized,
      });
    }
    const controller = new AbortController();
    let timedOut = false;
    const timeoutMs = call.timeoutMs ?? 0;
    const timer =
      timeoutMs > 0
        ? setTimeout(() => {
            timedOut = true;
            controller.abort();
          }, timeoutMs)
        : null;
    const forward = () => controller.abort();
    if (call.signal) {
      if (call.signal.aborted) controller.abort();
      else call.signal.addEventListener('abort', forward);
    }
    try {
      const res = await fetchImpl(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers: headers(),
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!res || typeof res.status !== 'number') {
        const err = new Error('provider returned no response');
        err.status = null;
        throw err;
      }
      if (res.status < 200 || res.status >= 300) {
        let text = '';
        try {
          text = await res.text();
        } catch {
          text = ''; // deliberate: an unreadable error body is not itself an error
        }
        const err = new Error(`HTTP ${res.status}${text ? `: ${text.slice(0, 500)}` : ''}`);
        err.status = res.status;
        err.body = text;
        err.headers = res.headers ?? null; // Retry-After lives here
        throw err;
      }
      if (body.stream === true) {
        return assembleStreamedCompletion(parseSseEvents(await readBodyText(res)), body);
      }
      return await res.json();
    } catch (err) {
      if (isAbortError(err) || timedOut) {
        if (call.signal?.aborted && !timedOut) throw err;
        throw new LadderError(
          'provider-timeout',
          `${normalized}: request exceeded ${timeoutMs || 'the'} ms budget`,
          { ref: normalized, timeoutMs },
        );
      }
      throw err;
    } finally {
      if (timer) clearTimeout(timer);
      if (call.signal) call.signal.removeEventListener('abort', forward);
    }
  }

  // The gate is around one HTTP attempt, not around the retry loop: a request
  // waiting out a backoff must not hold a concurrency slot.
  function once(body, call) {
    return limiter.run(() => request(body, call), { signal: call.signal });
  }

  async function sendWithRetries(body, call, meta) {
    let lastErr = null;
    for (let attempt = 0; attempt < ABSOLUTE_MAX_TRIES; attempt += 1) {
      try {
        return await once(body, call);
      } catch (err) {
        // A cancellation the caller asked for is not a provider fault: propagate
        // it unchanged so the orchestrator can tell abort from failure.
        if (isAbortError(err) && call.signal?.aborted) throw err;
        lastErr = err;
        const status = err?.status ?? null;
        const noStatus = status === undefined || status === null;
        const retryable =
          !isLadderError(err) && (isRetryableStatus(status) || (noStatus && !isAbortError(err)));
        if (status === 429) {
          meta.throttled = true;
          limiter.penalize({ ms: THROTTLE_PENALTY_MS });
        }
        if (!retryable || attempt + 1 >= triesFor(status)) break;
        await sleep(retryDelayMs(attempt, err?.headers ?? null, { random, now }));
      }
    }
    throw mapError(lastErr);
  }

  return {
    id: normalized,
    provider,
    model,
    armed,
    kind: 'openai-compatible',
    baseUrl,
    spec,
    /** True once this instance is speaking the text tool protocol. */
    get textTools() {
      return textTools;
    },
    async chat(call = {}) {
      if (!armed) {
        throw new LadderError('not-armed', `rung ${normalized} is not armed: set ${envVar}`, {
          ref: normalized,
          envVar,
        });
      }
      const started = Date.now();
      // Per-call request for the text protocol (spec 4.3). It never latches the
      // instance: the next call without the flag speaks native tools again.
      let useText = textTools || call.textTools === true;
      const meta = { textTools: useText, throttled: false, streamed: wantsStream(call) };
      const hasTools = Array.isArray(call.tools) && call.tools.length > 0;
      let strictUsed = true;
      const send = (strict) => {
        strictUsed = strict;
        return sendWithRetries(buildBody(call, { strict, useText }), call, meta);
      };

      let data;
      try {
        data = await send(true);
      } catch (err) {
        if (!useText && hasTools && looksLikeToolsUnsupported(err)) {
          // The endpoint has no tool API. Latch the text protocol on for the life
          // of this instance and retry this same call once; providers keep no store
          // the caller can read, so the only signal out is meta.textTools.
          textTools = true;
          useText = true;
          meta.textTools = true;
          data = await send(true);
        } else if (call.jsonSchema && isLadderError(err, 'provider-bad-request')) {
          // Some gateways reject `strict:true` schemas; retry once without it.
          data = await send(false);
        } else {
          throw err;
        }
      }

      // A thinking model can spend the whole budget on reasoning and hand back
      // an empty message with finish_reason "length". One retry with a larger
      // budget turns that into an answer; the first call's tokens still count.
      let priorUsage = null;
      if (isEmptyOnLength(data)) {
        const from = call.maxTokens ?? DEFAULT_MAX_TOKENS;
        const to = emptyRetryBudget(from);
        if (to > from) {
          meta.emptyRetry = { from, to };
          meta.reasoningChars = reasoningTextOf(data.choices[0].message).length;
          priorUsage = data?.usage ?? null;
          const bigger = { ...call, maxTokens: to };
          meta.streamed = wantsStream(bigger);
          data = await sendWithRetries(buildBody(bigger, { strict: strictUsed, useText }), bigger, meta);
        }
      }

      const choice = data?.choices?.[0] ?? null;
      const message = choice?.message ?? {};
      const rawContent = typeof message.content === 'string' ? message.content : blockText(message.content);
      let text = stripThinking(rawContent);
      if (text === '') {
        const reasoning = reasoningTextOf(message);
        if (reasoning !== '') text = stripThinking(reasoning);
      }

      const toolCalls = [];
      const content = [];

      if (meta.textTools) {
        // Text mode: the calls are in the text. Parse them out, then rebuild the
        // exact tool_use blocks the coder's transcript expects.
        const parsed = parseTextToolCalls(text);
        text = parsed.text;
        if (text !== '') content.push({ type: 'text', text });
        for (const tc of parsed.toolCalls) {
          content.push({ type: 'tool_use', id: tc.id, name: tc.name, input: tc.input });
          toolCalls.push({ id: tc.id, name: tc.name, input: tc.input });
        }
        if (parsed.errors.length > 0) meta.textToolErrors = parsed.errors;
      } else {
        if (text !== '') content.push({ type: 'text', text });
        for (const tc of message.tool_calls ?? []) {
          const rawArgs = tc?.function?.arguments;
          let input;
          try {
            input = typeof rawArgs === 'string' ? JSON.parse(rawArgs) : (rawArgs ?? {});
            if (input === null || typeof input !== 'object') input = { _raw: String(rawArgs) };
          } catch {
            input = { _raw: String(rawArgs) };
          }
          const id = tc?.id ?? `toolu_${toolCalls.length + 1}`;
          const name = tc?.function?.name;
          content.push({ type: 'tool_use', id, name, input });
          toolCalls.push({ id, name, input });
        }
      }

      const rawUsage = sumUsage(priorUsage, data?.usage ?? {});
      const promptTokens = rawUsage.prompt_tokens ?? 0;
      const cacheRead =
        rawUsage.prompt_cache_hit_tokens ?? rawUsage.prompt_tokens_details?.cached_tokens ?? 0;
      // Anthropic reports uncached input separately; normalize to that shape so
      // cost math is uniform across adapters.
      const usage = {
        input: cacheRead > 0 && cacheRead <= promptTokens ? promptTokens - cacheRead : promptTokens,
        output: rawUsage.completion_tokens ?? 0,
        cacheRead,
        cacheWrite: 0,
      };
      if (data?.__estimatedUsage) usage.estimated = true;

      let stopReason = mapFinishReason(choice?.finish_reason);
      if (toolCalls.length > 0 && stopReason === 'end_turn') stopReason = 'tool_use';

      return {
        text,
        json: call.jsonSchema ? parseJsonLoose(text) : null,
        toolCalls,
        content,
        stopReason,
        usage,
        costUsd: estimateCost(normalized, usage, { offpeak: Boolean(config.deepseekOffpeak) }),
        model: data?.model ?? model,
        provider,
        latencyMs: Date.now() - started,
        meta,
      };
    },
  };
}

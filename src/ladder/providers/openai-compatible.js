// kolm code — OpenAI-compatible adapter (owner A, spec §3).
//
// Serves cerebras / deepseek / openrouter / ollama through
// `POST {base}/chat/completions`. Translates the unified Anthropic-shaped block
// vocabulary in both directions:
//   assistant tool_use  <-> `tool_calls[{id, type:'function', function:{name, arguments}}]`
//   user tool_result    <-> separate `{role:'tool', tool_call_id, content}` messages
//   tools               <-> `[{type:'function', function:{name, description, parameters}}]`
//
// `fetch` is injectable so every test runs offline.

import { LadderError, isLadderError } from '../errors.js';
import { parseRef } from '../config.js';
import { describeModel, estimateCost } from './registry.js';

const MAX_TRIES = 3;
const BACKOFF_BASE_MS = 500;

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

/**
 * @param {string} ref e.g. 'deepseek:deepseek-v4-flash'
 * @param {object} [config] from loadConfig()
 * @param {{fetchImpl?: Function, sleep?: Function, random?: Function, baseUrl?: string}} [opts]
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
  const spec = describeModel(normalized);

  function headers() {
    const h = { 'content-type': 'application/json' };
    if (apiKey) h.authorization = `Bearer ${apiKey}`;
    if (provider === 'openrouter') {
      h['HTTP-Referer'] = 'https://kolm.ai';
      h['X-Title'] = 'kolm code';
    }
    return h;
  }

  function buildBody(call, { strict }) {
    const body = {
      model,
      messages: toOpenAIMessages(call.system ?? null, call.messages ?? []),
      max_tokens: call.maxTokens ?? 8192,
      stream: false,
    };
    if (call.temperature !== undefined && call.temperature !== null) {
      body.temperature = call.temperature;
    }
    if (Array.isArray(call.tools) && call.tools.length > 0) {
      body.tools = toOpenAITools(call.tools);
    }
    if (call.jsonSchema) {
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

  async function once(body, call) {
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
        throw err;
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

  async function sendWithRetries(body, call) {
    let lastErr = null;
    for (let attempt = 0; attempt < MAX_TRIES; attempt += 1) {
      try {
        return await once(body, call);
      } catch (err) {
        // A cancellation the caller asked for is not a provider fault: propagate
        // it unchanged so the orchestrator can tell abort from failure.
        if (isAbortError(err) && call.signal?.aborted) throw err;
        lastErr = err;
        const noStatus = err?.status === undefined || err?.status === null;
        const retryable =
          !isLadderError(err) && (isRetryableStatus(err?.status) || (noStatus && !isAbortError(err)));
        if (attempt === MAX_TRIES - 1 || !retryable) break;
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
    kind: 'openai-compatible',
    baseUrl,
    spec,
    async chat(call = {}) {
      if (!armed) {
        throw new LadderError('not-armed', `rung ${normalized} is not armed: set ${envVar}`, {
          ref: normalized,
          envVar,
        });
      }
      const started = Date.now();
      let data;
      try {
        data = await sendWithRetries(buildBody(call, { strict: true }), call);
      } catch (err) {
        // Some gateways reject `strict:true` schemas; retry once without it.
        if (call.jsonSchema && isLadderError(err, 'provider-bad-request')) {
          data = await sendWithRetries(buildBody(call, { strict: false }), call);
        } else {
          throw err;
        }
      }

      const choice = data?.choices?.[0] ?? null;
      const message = choice?.message ?? {};
      const rawContent = typeof message.content === 'string' ? message.content : blockText(message.content);
      let text = stripThinking(rawContent);
      if (text === '' && typeof message.reasoning_content === 'string') {
        text = stripThinking(message.reasoning_content);
      }

      const toolCalls = [];
      const content = [];
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

      const rawUsage = data?.usage ?? {};
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
      };
    },
  };
}

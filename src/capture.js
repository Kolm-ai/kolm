// Capture proxy - drop-in replacement for Anthropic / OpenAI APIs that
// records (input, output, latency_us, model, namespace, tenant) tuples on
// every call. The customer points OPENAI_BASE_URL, ANTHROPIC_BASE_URL, or
// OPENROUTER_BASE_URL at `https://kolm.ai/v1/capture/<provider>` and passes
// their own provider key in the `x-upstream-api-key` header (we strip +
// forward it).
//
// The captured corpus is queryable via `/v1/labels/synthesize-corpus` as
// JSONL or parquet, then promoted to a recipe via the existing
// `/v1/bridges/auto-synthesize` path or distilled to a local LoRA via
// `/v1/specialists/auto-distill` (kolm trainer bridge).
//
// We never train on the customer's data without consent. The artifact
// produced by distill ships to the customer; no copy is retained.

import crypto from 'node:crypto';
// W-N - shared fetch hardening: 429+backoff retries (max 3), AbortController
// timeouts (default 60s, clamped 1-300s), malformed-JSON envelopes. Used by
// forwardAnthropic / forwardOpenAI / forwardOpenRouter below alongside the
// W-M Vercel teacher-chat proxy fallback (which is in this file).
import {
  hardenedFetch as _w_N_hardenedFetch,
  buildOpenAICompatBody as _w_N_buildOpenAICompatBody,
  buildAnthropicBody as _w_N_buildAnthropicBody,
  DEFAULT_TIMEOUT_MS as _W_N_DEFAULT_TIMEOUT_MS,
} from './providers/_shared.js';
// W784 - Capture-processor plugin discovery. captureProcessorPlugins() returns
// plugins of kind "capture-processor" so the gateway can fold third-party
// transformers (e.g. domain-specific PII redactors) into the capture pipeline.
// Discovery only at this layer - the plugin's entry script does the actual
// transform per row. See src/plugins.js + /docs/plugins.html.
import { captureProcessorPlugins as _w784CaptureProcessorPlugins } from './plugins.js';
export function listW784CaptureProcessorPlugins() {
  try { return _w784CaptureProcessorPlugins(); } catch (_) {
    return { ok: true, kind: 'capture-processor', total: 0, plugins: [], errors: [] };
  }
}

const ANTHROPIC_DEFAULT = 'https://api.anthropic.com/v1/messages';
const OPENAI_DEFAULT = 'https://api.openai.com/v1/chat/completions';
const OPENROUTER_DEFAULT = 'https://openrouter.ai/api/v1/chat/completions';

export function pickAnthropicUpstream() {
  return process.env.ANTHROPIC_UPSTREAM_URL || ANTHROPIC_DEFAULT;
}

export function pickOpenAIUpstream() {
  return process.env.OPENAI_UPSTREAM_URL || OPENAI_DEFAULT;
}

export function pickOpenRouterUpstream() {
  return process.env.OPENROUTER_UPSTREAM_URL || OPENROUTER_DEFAULT;
}

// Sanitize the namespace label. We allow a-z, 0-9, dash, dot, underscore;
// disallow consecutive dots and leading/trailing dots so a namespace can't
// look like a path-traversal token. Empty -> 'default'.
export function sanitizeNamespace(raw) {
  let s = String(raw || '').toLowerCase().replace(/[^a-z0-9._-]/g, '');
  s = s.replace(/\.+/g, '').slice(0, 64);
  return s || 'default';
}

// Distill the inbound request body into a single canonical "prompt" string we
// hash for clustering. We deliberately drop config fields (temperature, max
// tokens, etc.) so identical user intent across different sampling configs
// clusters together.
export function extractPromptForCapture(body, provider) {
  if (!body || typeof body !== 'object') return '';
  if (provider === 'anthropic') {
    const sys = typeof body.system === 'string' ? body.system : '';
    const messages = Array.isArray(body.messages) ? body.messages : [];
    const turns = messages.map(m => {
      if (!m) return '';
      const role = m.role || 'user';
      const content = typeof m.content === 'string'
        ? m.content
        : Array.isArray(m.content)
          ? m.content.map(c => (c && (c.text || c.content)) || '').join('\n')
          : '';
      return `${role}: ${content}`;
    }).filter(Boolean).join('\n\n');
    return [sys ? `system: ${sys}` : '', turns].filter(Boolean).join('\n\n');
  }
  if (provider === 'openai' || provider === 'openrouter') {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    return messages.map(m => {
      const role = m && m.role || 'user';
      const content = typeof m.content === 'string'
        ? m.content
        : Array.isArray(m.content)
          ? m.content.map(c => (c && (c.text || c.content)) || '').join('\n')
          : '';
      return `${role}: ${content}`;
    }).join('\n\n');
  }
  return '';
}

// Pull the model output text out of a provider response.
export function extractCompletionText(json, provider) {
  if (!json || typeof json !== 'object') return '';
  if (provider === 'anthropic') {
    const blocks = Array.isArray(json.content) ? json.content : [];
    return blocks.map(b => (b && b.text) || '').join('').trim();
  }
  if (provider === 'openai' || provider === 'openrouter') {
    const choices = Array.isArray(json.choices) ? json.choices : [];
    const first = choices[0] || {};
    const msg = first.message || {};
    if (typeof msg.content === 'string') return msg.content.trim();
    if (Array.isArray(msg.content)) {
      return msg.content.map(c => (c && c.text) || '').join('').trim();
    }
    return String(first.text || '').trim();
  }
  return '';
}

export function modelFromBody(body, provider) {
  if (!body || typeof body !== 'object') return '';
  if (provider === 'anthropic') return String(body.model || '').slice(0, 128);
  if (provider === 'openai' || provider === 'openrouter') return String(body.model || '').slice(0, 128);
  return '';
}

// --------------------------------------------------------------------------
// Vercel teacher-chat fallback
//
// W-M wave / wrapper-completion. The AWS service that hosts /v1/* does
// NOT see Vercel env vars - vendor API keys (ANTHROPIC_API_KEY, OPENAI_*,
// GOOGLE_*, XAI_*) live on the kolm.ai Vercel deployment because that's where
// /v1/teacher/chat already runs and accepts a kolm bearer. When the gateway
// dispatch path lands on the API host and finds NO local provider key, it should
// transparently proxy upstream calls through Vercel's /v1/teacher/chat
// instead of returning no_upstream_key. The user's gateway request already
// carries a valid kolm bearer; we re-use it as the proxy bearer.
//
// This helper is called by forwardAnthropic / forwardOpenAI / forwardOpenRouter
// when upstreamKey is null but proxyBearer is set. It speaks the teacher-chat
// request/response shape and back-translates to the vendor's native envelope
// so the gateway's downstream code (output PII scan, receipt cost estimator,
// observation insert) keeps working with no changes.
//
// Returns the same {status, json, elapsed_us} shape as a native forward call.
// --------------------------------------------------------------------------

function _resolveProxyBase(explicit) {
  if (explicit && typeof explicit === 'string') return explicit.replace(/\/+$/, '');
  const fromEnv = process.env.KOLM_BASE_URL
    || process.env.KOLM_PROXY_BASE
    || process.env.KOLM_VERCEL_BASE;
  if (fromEnv && typeof fromEnv === 'string') return fromEnv.replace(/\/+$/, '');
  return 'https://kolm.ai';
}

function _extractMessages(body) {
  if (Array.isArray(body && body.messages) && body.messages.length > 0) {
    return body.messages.map((m) => {
      const role = String(m.role || 'user').slice(0, 16);
      const c = typeof m.content === 'string'
        ? m.content
        : Array.isArray(m.content)
          ? m.content.map((x) => (x && (x.text || x.content)) || '').join('\n')
          : '';
      return { role, content: String(c).slice(0, 16000) };
    });
  }
  if (typeof (body && body.input) === 'string' && body.input) {
    return [{ role: 'user', content: body.input.slice(0, 16000) }];
  }
  return [];
}

async function _proxyViaTeacherChat({ vendor, body, proxyBearer, proxyBase }) {
  const base = _resolveProxyBase(proxyBase);
  const url = base + '/v1/teacher/chat';
  const messages = _extractMessages(body);
  const proxyBody = {
    vendor,
    model:      String((body && body.model) || ''),
    messages,
    system:     typeof (body && body.system) === 'string' ? body.system : '',
    max_tokens: Number((body && body.max_tokens) || 1024),
  };
  const t0 = process.hrtime.bigint();
  let res;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers: {
        'authorization': 'Bearer ' + proxyBearer,
        'content-type':  'application/json',
        'x-kolm-via':    'gateway-dispatch-proxy',
      },
      body: JSON.stringify(proxyBody),
    });
  } catch (e) {
    const elapsed_us = Math.round(Number(process.hrtime.bigint() - t0) / 1000);
    return {
      status: 0,
      json: { error: { type: 'proxy_transport_error', message: String(e && e.message || e) } },
      elapsed_us,
      kolm_proxy: { path: 'vercel-teacher-chat', base, ok: false },
    };
  }
  const text = await res.text();
  let json;
  try { json = JSON.parse(text); } catch (_) { json = { _raw: text }; }
  const elapsed_us = Math.round(Number(process.hrtime.bigint() - t0) / 1000);

  // On success, normalize to the vendor's native envelope so downstream
  // readers (gateway-receipt cost estimator, output PII scan) find the
  // shape they already understand.
  if (res.ok && json && json.ok) {
    const txt = (json.choices && json.choices[0] && json.choices[0].message && json.choices[0].message.content) || '';
    const usage = json.usage || {};
    if (vendor === 'anthropic') {
      return {
        status: 200,
        json: {
          id:   json.upstream_id || ('msg_proxy_' + Date.now().toString(36)),
          type: 'message',
          role: 'assistant',
          model: json.model,
          content: [{ type: 'text', text: txt }],
          // Also keep OpenAI-compat keys so the dispatch output-text
          // extractor (which checks choices[0].message.content FIRST) hits
          // immediately without traversing content[].
          choices: json.choices,
          usage: {
            // Native Anthropic keys for the receipt reader's primary lookup.
            input_tokens:  usage.input_tokens  || usage.prompt_tokens     || 0,
            output_tokens: usage.output_tokens || usage.completion_tokens || 0,
            // Pass through char counts too - useful for debugging.
            input_chars:   usage.input_chars   || 0,
            output_chars: usage.output_chars   || 0,
          },
          kolm_proxy: { path: 'vercel-teacher-chat', base, key_source: json.proxy_key_source || null },
        },
        elapsed_us,
      };
    }
    // openai / openrouter - already openai-compat.
    return {
      status: 200,
      json: {
        ...json,
        usage: {
          prompt_tokens:     usage.prompt_tokens     || usage.input_tokens   || 0,
          completion_tokens: usage.completion_tokens || usage.output_tokens  || 0,
          total_tokens:      usage.total_tokens      || ((usage.prompt_tokens || 0) + (usage.completion_tokens || 0)),
          input_chars:       usage.input_chars       || 0,
          output_chars:      usage.output_chars      || 0,
        },
        kolm_proxy: { path: 'vercel-teacher-chat', base, key_source: json.proxy_key_source || null },
      },
      elapsed_us,
    };
  }

  // Failure path - surface the proxy's status so shouldFallback() in
  // gateway-router can advance the chain (503 from the Vercel side when
  // its keys are missing too is 5xx, which IS fallback-eligible).
  return {
    status: res.status || 502,
    json: {
      error: {
        type: (json && json.error) || 'proxy_upstream_error',
        message: (json && json.detail) || (json && json._raw) || ('teacher-chat returned ' + res.status),
        proxy_status: res.status,
      },
    },
    elapsed_us,
    kolm_proxy: { path: 'vercel-teacher-chat', base, ok: false },
  };
}

// Forward to the upstream provider. The customer's own provider key
// arrives in `x-upstream-api-key`; we use it and never log it. When no
// key is configured AND a proxyBearer is provided, fall back to the
// Vercel /v1/teacher/chat function which has the keys (see W-M above).
export async function forwardAnthropic({ url, body, upstreamKey, anthropicVersion, proxyBearer, proxyBase, timeoutMs }) {
  if (!upstreamKey) {
    if (proxyBearer) return _proxyViaTeacherChat({ vendor: 'anthropic', body, proxyBearer, proxyBase });
    return { status: 401, json: { error: { type: 'no_upstream_key', message: 'pass your Anthropic key in x-upstream-api-key' } } };
  }
  // W-N: normalize string `content` to content-block form per Anthropic
  // Messages API docs ([{type:'text', text}]). Pass-through temperature,
  // top_p, max_tokens, stop_sequences, tools, system. String `stop` maps
  // to `stop_sequences` so OpenAI-style callers keep working.
  const shapedBody = _w_N_buildAnthropicBody(body);
  return _w_N_hardenedFetch({
    url,
    method: 'POST',
    headers: {
      'x-api-key': upstreamKey,
      'anthropic-version': anthropicVersion || '2023-06-01',
      'content-type': 'application/json',
    },
    body: JSON.stringify(shapedBody),
    timeoutMs: timeoutMs || _W_N_DEFAULT_TIMEOUT_MS,
    requireJson: true,
  });
}

export async function forwardOpenAI({ url, body, upstreamKey, proxyBearer, proxyBase, timeoutMs }) {
  if (!upstreamKey) {
    if (proxyBearer) return _proxyViaTeacherChat({ vendor: 'openai', body, proxyBearer, proxyBase });
    return { status: 401, json: { error: { type: 'no_upstream_key', message: 'pass your OpenAI key in x-upstream-api-key' } } };
  }
  // W-N: pass-through temperature / top_p / stop / tools / response_format
  // and map `max_tokens` -> `max_completion_tokens` for the o-series + gpt-5
  // reasoning models. Other model families keep `max_tokens` unchanged.
  const shapedBody = _w_N_buildOpenAICompatBody(body);
  return _w_N_hardenedFetch({
    url,
    method: 'POST',
    headers: {
      'authorization': `Bearer ${upstreamKey}`,
      'content-type': 'application/json',
    },
    body: JSON.stringify(shapedBody),
    timeoutMs: timeoutMs || _W_N_DEFAULT_TIMEOUT_MS,
    requireJson: true,
  });
}

export async function forwardOpenRouter({ url, body, upstreamKey, referer = 'https://kolm.ai', title = 'kolm.ai', categories = '', proxyBearer, proxyBase, timeoutMs }) {
  if (!upstreamKey) {
    // OpenRouter isn't in the Vercel VENDOR_KEYS table, so the proxy
    // fallback routes through the openai vendor (compatible API shape).
    // If the Vercel side has no openai key either, the proxy returns
    // 503 and the gateway falls back to the next chain entry.
    if (proxyBearer) return _proxyViaTeacherChat({ vendor: 'openai', body, proxyBearer, proxyBase });
    return { status: 401, json: { error: { type: 'no_upstream_key', message: 'pass your OpenRouter key in x-upstream-api-key' } } };
  }
  // W-N: OpenRouter speaks OpenAI-compat. Same body normalizer fires here
  // so max_tokens -> max_completion_tokens for the o-series, plus the
  // standard temperature / top_p / stop / tools / response_format pass-through.
  const shapedBody = _w_N_buildOpenAICompatBody(body);
  const headers = {
    'authorization': `Bearer ${upstreamKey}`,
    'content-type': 'application/json',
    'http-referer': referer,
    'x-title': title,
  };
  if (title) headers['x-openrouter-title'] = title;
  if (categories) headers['x-openrouter-categories'] = categories;
  return _w_N_hardenedFetch({
    url,
    method: 'POST',
    headers,
    body: JSON.stringify(shapedBody),
    timeoutMs: timeoutMs || _W_N_DEFAULT_TIMEOUT_MS,
    requireJson: true,
  });
}

// Hash the prompt to detect duplicate captures (the customer hitting "send"
// twice on the same input in two seconds).
export function promptHash(prompt) {
  return crypto.createHash('sha256').update(prompt || '', 'utf8').digest('hex').slice(0, 16);
}

// W713-1 - reasoning-trace extraction.
//
// When the teacher is a reasoning model (Claude with thinking blocks, OpenAI
// o1/o3 with reasoning_tokens, DeepSeek-R1 with <think>...</think>), we want
// to capture the chain-of-thought, not just the final answer. The student
// then learns to reproduce the reasoning process - see apps/trainer/distill_cot.py
// for the training-side formatter and src/chat-templates.js (kolm-think) for
// the chat-template wrapper that bytes-match this envelope.
//
// Honesty contract: returns null when no reasoning is detected (NOT {} - null
// is the honest "no trace present" signal; {} would be confusable with "empty
// trace recorded"). Never throws; malformed inputs return null.
//
// Output envelope shape (stable across providers - distill_cot.py reads this):
//   {
//     provider: 'anthropic' | 'openai' | 'generic',
//     blocks?: [{ type: 'thinking', text: '...' }, { type: 'text', text: '...' }],
//     reasoning_tokens?: number,            // OpenAI usage hint
//     reasoning_text_if_present?: string,   // OpenAI o-series can return text
//     total_thinking_chars: number,         // always present, may be 0
//   }
export function extractReasoningTrace(response, provider) {
  if (!response || typeof response !== 'object') return null;
  try {
    if (provider === 'anthropic') return _extractAnthropicReasoning(response);
    if (provider === 'openai' || provider === 'openrouter') return _extractOpenAIReasoning(response);
    if (provider === 'generic' || provider === 'deepseek' || provider === 'ollama') {
      return _extractGenericReasoning(response);
    }
    // Unknown provider - try generic as the best-effort fallback, never throw.
    return _extractGenericReasoning(response);
  } catch (_) {
    return null;
  }
}

function _extractAnthropicReasoning(response) {
  const content = Array.isArray(response.content) ? response.content : null;
  if (!content) return null;
  const blocks = [];
  let totalThinking = 0;
  let sawThinking = false;
  for (const b of content) {
    if (!b || typeof b !== 'object') continue;
    if (b.type === 'thinking') {
      const text = typeof b.thinking === 'string' ? b.thinking
        : (typeof b.text === 'string' ? b.text : '');
      blocks.push({ type: 'thinking', text });
      totalThinking += text.length;
      sawThinking = true;
    } else if (b.type === 'text') {
      const text = typeof b.text === 'string' ? b.text : '';
      blocks.push({ type: 'text', text });
    }
  }
  if (!sawThinking) return null;
  return {
    provider: 'anthropic',
    blocks,
    total_thinking_chars: totalThinking,
  };
}

function _extractOpenAIReasoning(response) {
  const choices = Array.isArray(response.choices) ? response.choices : [];
  const first = choices[0] || {};
  const msg = first.message || {};
  const usage = response.usage || {};
  const ctd = usage.completion_tokens_details || {};
  const reasoningTokens = Number(ctd.reasoning_tokens) || 0;
  // o1/o3 sometimes attaches the reasoning text on message.reasoning (preview
  // SDKs) or message.reasoning_content (DeepSeek-OpenAI-compatible adapter).
  let reasoningText = '';
  if (typeof msg.reasoning === 'string') reasoningText = msg.reasoning;
  else if (msg.reasoning && typeof msg.reasoning.content === 'string') reasoningText = msg.reasoning.content;
  else if (typeof msg.reasoning_content === 'string') reasoningText = msg.reasoning_content;
  // No reasoning tokens AND no inline reasoning text - honest null.
  if (reasoningTokens === 0 && !reasoningText) return null;
  const out = {
    provider: 'openai',
    reasoning_tokens: reasoningTokens,
    total_thinking_chars: reasoningText.length,
  };
  if (reasoningText) out.reasoning_text_if_present = reasoningText;
  return out;
}

// Permissive <think>...</think> parser. Takes the FIRST </think> as the end
// of the thinking block (DeepSeek-R1 emits exactly one). Unbalanced (no
// closing tag) → returns null gracefully.
export function parseThinkBlocks(text) {
  if (typeof text !== 'string') return null;
  const openIdx = text.indexOf('<think>');
  if (openIdx === -1) return null;
  const closeIdx = text.indexOf('</think>', openIdx);
  if (closeIdx === -1) return null;  // unbalanced - honest null
  const thinking = text.slice(openIdx + '<think>'.length, closeIdx);
  const answer = text.slice(closeIdx + '</think>'.length);
  return {
    thinking,
    answer,
  };
}

function _extractGenericReasoning(response) {
  // Three accepted shapes for generic / DeepSeek-R1:
  //   1) { text: '<think>...</think>final' }
  //   2) { content: '<think>...</think>final' } (Ollama-ish)
  //   3) OpenAI-style choices[0].message.content with <think> inside
  let raw = '';
  if (typeof response.text === 'string') raw = response.text;
  else if (typeof response.content === 'string') raw = response.content;
  else if (Array.isArray(response.choices) && response.choices[0]) {
    const m = response.choices[0].message || {};
    if (typeof m.content === 'string') raw = m.content;
  }
  if (!raw) return null;
  const parsed = parseThinkBlocks(raw);
  if (!parsed) return null;
  return {
    provider: 'generic',
    blocks: [
      { type: 'thinking', text: parsed.thinking },
      { type: 'text', text: parsed.answer },
    ],
    total_thinking_chars: parsed.thinking.length,
  };
}

// W828-1 - AUTO-DETECT reasoning capability from response shape.
//
// W713 required the caller to pass provider:'anthropic'|'openai'|'generic'.
// W828 sniffs the response shape itself so the capture path doesn't need
// per-call config to know which extractor to run.
//
// Detection rules (additive, never throws):
//   * Anthropic extended-thinking: response.content[*].type === 'thinking'
//     → { has_traces:true, format:'anthropic_thinking', provider:'anthropic' }
//   * OpenAI o1/o3: response.usage.completion_tokens_details.reasoning_tokens
//     > 0 OR (legacy) response.usage.reasoning_tokens > 0
//     → { has_traces:true, format:'openai_reasoning_tokens', provider:'openai' }
//   * DeepSeek-R1 OpenAI-compatible: response.choices[0].message.reasoning_content
//     → { has_traces:true, format:'deepseek_reasoning', provider:'deepseek' }
//   * Gemini thinking: response.candidates[0].content.parts[*].thinking
//     → { has_traces:true, format:'gemini_thinking', provider:'gemini' }
//   * else → { has_traces:false }
//
// Honesty contract: never throws on malformed input; missing/null/non-object →
// { has_traces:false } (NOT null - callers branch on has_traces directly).
// Order: Anthropic shape first (richest signal), then OpenAI reasoning_tokens,
// then DeepSeek reasoning_content (shares OpenAI choices[] shape but field is
// distinct), then Gemini. A response cannot be more than one of these in
// practice; first match wins.
export function autoDetectReasoningCapability(response) {
  if (!response || typeof response !== 'object') {
    return { has_traces: false };
  }
  try {
    // Anthropic extended-thinking - content[].type === 'thinking'.
    if (Array.isArray(response.content)) {
      for (const block of response.content) {
        if (block && typeof block === 'object' && block.type === 'thinking') {
          return {
            has_traces: true,
            format: 'anthropic_thinking',
            provider: 'anthropic',
          };
        }
      }
    }
    // OpenAI o-series reasoning tokens - usage.completion_tokens_details.reasoning_tokens
    // (current shape) or usage.reasoning_tokens (early preview / spec stub).
    const usage = response.usage;
    if (usage && typeof usage === 'object') {
      const ctd = usage.completion_tokens_details;
      const ctdTokens = ctd && typeof ctd === 'object'
        ? Number(ctd.reasoning_tokens) || 0
        : 0;
      const flatTokens = Number(usage.reasoning_tokens) || 0;
      if (ctdTokens > 0 || flatTokens > 0) {
        return {
          has_traces: true,
          format: 'openai_reasoning_tokens',
          provider: 'openai',
        };
      }
    }
    // DeepSeek-R1 OpenAI-compatible adapter - message.reasoning_content.
    if (Array.isArray(response.choices) && response.choices[0]) {
      const msg = response.choices[0].message;
      if (msg && typeof msg === 'object' && typeof msg.reasoning_content === 'string'
          && msg.reasoning_content.length > 0) {
        return {
          has_traces: true,
          format: 'deepseek_reasoning',
          provider: 'deepseek',
        };
      }
    }
    // Gemini thinking - candidates[0].content.parts[*].thinking.
    if (Array.isArray(response.candidates) && response.candidates[0]) {
      const cand = response.candidates[0];
      const content = cand && cand.content;
      const parts = content && Array.isArray(content.parts) ? content.parts : null;
      if (parts) {
        for (const p of parts) {
          if (p && typeof p === 'object' && p.thinking) {
            return {
              has_traces: true,
              format: 'gemini_thinking',
              provider: 'gemini',
            };
          }
        }
      }
    }
  } catch (_) {
    // Defense-in-depth - malformed shapes should never throw.
    return { has_traces: false };
  }
  return { has_traces: false };
}

// W828-1 - autoExtractReasoningTrace: thin wrapper around extractReasoningTrace
// that uses autoDetectReasoningCapability to pick the provider so callers in
// the capture path (src/router.js, src/vision-capture.js) don't have to
// hardcode the provider when the response shape already tells the story.
//
// When auto-detect returns has_traces:false, we still try the caller-supplied
// hint (if any) as a fallback so existing call sites that DO pass provider:
// keep working unchanged (W713 lock-in tests expect that).
export function autoExtractReasoningTrace(response, hintProvider) {
  const cap = autoDetectReasoningCapability(response);
  if (cap.has_traces) {
    return extractReasoningTrace(response, cap.provider);
  }
  // Fallback: honour the caller's explicit hint (W713 contract preservation).
  if (hintProvider) return extractReasoningTrace(response, hintProvider);
  return null;
}

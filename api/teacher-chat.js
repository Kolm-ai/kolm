// Vercel serverless function for /v1/teacher/chat (POST relay) and
// /v1/teacher/chat/health (GET status). Lives on Vercel rather than the
// API router so it can read sensitive env vars (anthropic_api_key,
// openai_key, google_api_key, xai_api_key) that exist only at the kolm.ai
// Vercel runtime. The user wired the Trinity distill worker to fall back
// to ${KOLM_BASE_URL}/v1/teacher/chat when the local env lacks teacher
// keys - this function is the server-side half of that path.
//
// Vercel env vars are case-sensitive and operators have been adding the
// keys with mixed casing. Read every reasonable form so the function
// never silently misses a configured key.

import crypto from 'node:crypto';

export const TEACHER_CHAT_VERSION = 'w692-v1';

// Origin of the kolm API (/v1/whoami) used to validate bearer keys. Set
// KOLM_API_ORIGIN on Vercel; the fallback is the current production origin and
// goes away at the AWS cutover (docs/operations/aws-migration.md, Phase 3).
export const KOLM_API_ORIGIN = String(process.env.KOLM_API_ORIGIN || 'https://kolmogorov-stack-production.up.railway.app').replace(/\/+$/, '');

export const TEACHER_CHAT_LIMITS = Object.freeze({
  AUTH_TIMEOUT_MS: 4000,
  UPSTREAM_TIMEOUT_MS: 60_000,
  MAX_MODEL_CHARS: 128,
  MAX_SYSTEM_CHARS: 8000,
  MAX_MESSAGE_CHARS: 16_000,
  MAX_TOTAL_MESSAGE_CHARS: 32_000,
  MAX_UPSTREAM_ERROR_CHARS: 400,
});

export const VENDOR_KEYS = Object.freeze({
  anthropic: ['ANTHROPIC_API_KEY', 'anthropic_api_key', 'ANTHROPIC_KEY', 'anthropic_key'],
  openai:    ['OPENAI_API_KEY', 'openai_api_key', 'OPENAI_KEY', 'openai_key'],
  google:    ['GOOGLE_API_KEY', 'google_api_key', 'GOOGLE_KEY', 'google_key', 'GEMINI_API_KEY', 'gemini_api_key'],
  xai:       ['XAI_API_KEY', 'xai_api_key', 'XAI_KEY', 'xai_key'],
});

function sha256Hex(value) {
  return crypto.createHash('sha256').update(String(value), 'utf8').digest('hex');
}

function safeText(value, maxChars = TEACHER_CHAT_LIMITS.MAX_UPSTREAM_ERROR_CHARS) {
  const s = String(value == null ? '' : value)
    .replace(/[\u0000-\u001f\u007f]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g, '[redacted_email]')
    .replace(/\b(?:sk|ghp|gho|ghs|ghu|ghr|xai|ya29|AIza)[_-][A-Za-z0-9_.-]{12,}\b/g, '[redacted_secret]');
  return s.length > maxChars ? s.slice(0, maxChars) : s;
}

function setSecurityHeaders(res) {
  try {
    res.setHeader('cache-control', 'no-store');
    res.setHeader('x-content-type-options', 'nosniff');
  } catch (_) {}
}

async function fetchWithTimeout(url, opts, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...(opts || {}), signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

function readBody(req) {
  if (req && req.body && typeof req.body === 'object' && !Array.isArray(req.body)) return req.body;
  if (req && typeof req.body === 'string' && req.body.trim()) {
    try {
      const parsed = JSON.parse(req.body);
      return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
    } catch (_) {
      return {};
    }
  }
  return {};
}

function firstEnv(envs) {
  for (const k of envs) {
    const v = process.env[k];
    if (v && String(v).trim().length > 0) return { key: v, var: k };
  }
  return null;
}

// Bearer-token authn against the kolm API router. Caches good keys
// for 5 minutes per cold start so we don't burn a /v1/whoami round trip
// on every teacher call.
const _keyCache = new Map();
function _cacheKey(bearer) {
  return sha256Hex('teacher-chat-auth:' + String(bearer));
}

async function authenticate(req) {
  const auth = String(req.headers.authorization || '');
  const m = auth.match(/^Bearer\s+(\S+)$/i);
  const bearer = m ? m[1] : (req.headers['x-api-key'] || '');
  if (!bearer) return { ok: false, status: 401, error: 'auth_required', detail: 'set Authorization: Bearer <kolm-key>' };
  if (!/^(ks_|kao_)[A-Za-z0-9_-]{6,256}$/.test(String(bearer))) {
    return { ok: false, status: 401, error: 'auth_invalid', detail: 'expected ks_ or kao_ prefix' };
  }
  const cacheKey = _cacheKey(bearer);
  const cached = _keyCache.get(cacheKey);
  if (cached && cached.expires > Date.now()) return { ok: true, tenant: cached.tenant };
  try {
    const r = await fetchWithTimeout(KOLM_API_ORIGIN + '/v1/whoami', {
      headers: { authorization: 'Bearer ' + bearer },
    }, TEACHER_CHAT_LIMITS.AUTH_TIMEOUT_MS);
    if (!r.ok) {
      return { ok: false, status: 401, error: 'auth_invalid', detail: 'kolm key rejected by tenant store' };
    }
    const j = await r.json();
    const tenant = (j && j.tenant) || (j && j.tenant_id) || 'unknown';
    _keyCache.set(cacheKey, { tenant, expires: Date.now() + 5 * 60 * 1000 });
    return { ok: true, tenant };
  } catch (e) {
    return { ok: false, status: 502, error: 'auth_upstream_failed', detail: safeText(e && e.message || e) };
  }
}

async function relayAnthropic({ key, model, system, messages, maxTokens }) {
  const r = await fetchWithTimeout('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'x-api-key': key,
      'anthropic-version': '2023-06-01',
      'content-type': 'application/json',
    },
    body: JSON.stringify({
      model,
      max_tokens: maxTokens,
      system: system || undefined,
      messages,
    }),
  }, TEACHER_CHAT_LIMITS.UPSTREAM_TIMEOUT_MS);
  if (!r.ok) {
    const t = await r.text();
    return {
      ok: false,
      status: 502,
      error: 'upstream_error',
      upstream_status: r.status,
      upstream_body_excerpt: safeText(t),
      upstream_body_sha256: sha256Hex(t),
    };
  }
  const j = await r.json();
  const block = (j.content || []).find(b => b && b.type === 'text');
  // Anthropic returns usage = {input_tokens, output_tokens}; preserve so the
  // The API gateway's cost_usd estimate is non-zero when the proxy is in
  // play. Without this the receipt would always show 0 tokens for the
  // Vercel-fallback path.
  return { ok: true, text: block ? block.text : '', usage: j.usage || {}, upstream_id: j.id || null };
}

async function relayOpenAILike({ url, key, model, system, messages, maxTokens }) {
  const oaMessages = system ? [{ role: 'system', content: system }, ...messages] : messages;
  const r = await fetchWithTimeout(url, {
    method: 'POST',
    headers: { authorization: 'Bearer ' + key, 'content-type': 'application/json' },
    body: JSON.stringify({ model, messages: oaMessages, max_tokens: maxTokens }),
  }, TEACHER_CHAT_LIMITS.UPSTREAM_TIMEOUT_MS);
  if (!r.ok) {
    const t = await r.text();
    return {
      ok: false,
      status: 502,
      error: 'upstream_error',
      upstream_status: r.status,
      upstream_body_excerpt: safeText(t),
      upstream_body_sha256: sha256Hex(t),
    };
  }
  const j = await r.json();
  const text = (j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content) || '';
  // OpenAI returns usage = {prompt_tokens, completion_tokens, total_tokens}.
  return { ok: true, text, usage: j.usage || {}, upstream_id: j.id || null };
}

async function relayGoogle({ key, model, system, messages, maxTokens }) {
  const url = 'https://generativelanguage.googleapis.com/v1beta/models/'
    + encodeURIComponent(model) + ':generateContent';
  const gBody = {
    contents: messages.map(m => ({
      role: m.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: m.content }],
    })),
    generationConfig: { maxOutputTokens: maxTokens },
  };
  if (system) gBody.systemInstruction = { role: 'system', parts: [{ text: system }] };
  const r = await fetchWithTimeout(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'x-goog-api-key': key },
    body: JSON.stringify(gBody),
  }, TEACHER_CHAT_LIMITS.UPSTREAM_TIMEOUT_MS);
  if (!r.ok) {
    const t = await r.text();
    return {
      ok: false,
      status: 502,
      error: 'upstream_error',
      upstream_status: r.status,
      upstream_body_excerpt: safeText(t),
      upstream_body_sha256: sha256Hex(t),
    };
  }
  const j = await r.json();
  const parts = (j.candidates && j.candidates[0] && j.candidates[0].content && j.candidates[0].content.parts) || [];
  // Google returns usageMetadata = {promptTokenCount, candidatesTokenCount,
  // totalTokenCount}; normalize to the OpenAI-compat shape the gateway
  // receipt reader looks for so cost math works for all 4 vendors.
  const um = j.usageMetadata || {};
  const usage = {
    prompt_tokens:     um.promptTokenCount || 0,
    completion_tokens: um.candidatesTokenCount || 0,
    total_tokens:      um.totalTokenCount || 0,
  };
  return { ok: true, text: parts.map(p => p.text || '').join(''), usage };
}

export default async function handler(req, res) {
  setSecurityHeaders(res);

  // GET -> health probe. No auth (publishes only booleans).
  if (req.method === 'GET') {
    const status = {};
    for (const [v, envs] of Object.entries(VENDOR_KEYS)) {
      const hit = firstEnv(envs);
      status[v] = !!hit;
    }
    return res.status(200).json({
      ok: true,
      version: TEACHER_CHAT_VERSION,
      vendors: status,
      any_configured: Object.values(status).some(Boolean),
      served_by: 'vercel-function',
    });
  }

  if (req.method !== 'POST') {
    res.setHeader('allow', 'GET, POST');
    return res.status(405).json({ ok: false, error: 'method_not_allowed' });
  }

  // POST -> relay. Auth required.
  const a = await authenticate(req);
  if (!a.ok) return res.status(a.status).json({ ok: false, error: a.error, detail: a.detail });

  const body = readBody(req);
  const vendor = String(body.vendor || '').toLowerCase();
  const envs = VENDOR_KEYS[vendor];
  if (!envs) {
    return res.status(400).json({
      ok: false, error: 'unknown_vendor',
      detail: 'vendor must be one of: ' + Object.keys(VENDOR_KEYS).join(', '),
    });
  }
  const rawModel = String(body.model || '').trim();
  if (/[\u0000-\u001f\u007f]/.test(rawModel)) {
    return res.status(400).json({ ok: false, error: 'model_control_chars', version: TEACHER_CHAT_VERSION });
  }
  const model = rawModel.slice(0, TEACHER_CHAT_LIMITS.MAX_MODEL_CHARS);
  if (!model) return res.status(400).json({ ok: false, error: 'missing_model' });
  const messagesIn = Array.isArray(body.messages) ? body.messages : null;
  const inputText = typeof body.input === 'string' ? body.input : '';
  if (!messagesIn && !inputText) {
    return res.status(400).json({ ok: false, error: 'missing_messages_or_input' });
  }
  const system = typeof body.system === 'string' ? safeText(body.system, TEACHER_CHAT_LIMITS.MAX_SYSTEM_CHARS) : '';
  let maxTokens = Number(body.max_tokens);
  if (!Number.isFinite(maxTokens) || maxTokens <= 0) maxTokens = 1024;
  if (maxTokens > 4096) maxTokens = 4096;

  let messages = [];
  if (messagesIn) {
    for (const m of messagesIn) {
      if (!m || typeof m !== 'object') continue;
      const roleRaw = String(m.role || 'user').toLowerCase();
      const role = roleRaw === 'assistant' ? 'assistant' : 'user';
      const c = typeof m.content === 'string' ? m.content
        : Array.isArray(m.content)
          ? m.content.map(x => (x && (x.text || x.content)) || '').join('\n')
          : '';
      if (!c) continue;
      messages.push({ role, content: String(c).slice(0, TEACHER_CHAT_LIMITS.MAX_MESSAGE_CHARS) });
    }
  } else {
    messages.push({ role: 'user', content: inputText.slice(0, TEACHER_CHAT_LIMITS.MAX_MESSAGE_CHARS) });
  }
  const totalChars = messages.reduce((a, m) => a + m.content.length, 0);
  if (totalChars > TEACHER_CHAT_LIMITS.MAX_TOTAL_MESSAGE_CHARS) {
    return res.status(413).json({
      ok: false,
      error: 'messages_too_large',
      total_chars: totalChars,
      max_total_chars: TEACHER_CHAT_LIMITS.MAX_TOTAL_MESSAGE_CHARS,
      version: TEACHER_CHAT_VERSION,
    });
  }
  const request_sha256 = sha256Hex(JSON.stringify({ vendor, model, system, messages, max_tokens: maxTokens }));

  const hit = firstEnv(envs);
  if (!hit) {
    return res.status(503).json({
      ok: false, error: 'key_not_configured',
      vendor,
      detail: `none of ${envs.join(' / ')} is set on this kolm.ai Vercel instance; ask the operator to add one (or use vendor=local).`,
    });
  }

  let result;
  try {
    if (vendor === 'anthropic') {
      result = await relayAnthropic({ key: hit.key, model, system, messages, maxTokens });
    } else if (vendor === 'openai') {
      result = await relayOpenAILike({ url: 'https://api.openai.com/v1/chat/completions', key: hit.key, model, system, messages, maxTokens });
    } else if (vendor === 'xai') {
      result = await relayOpenAILike({ url: 'https://api.x.ai/v1/chat/completions', key: hit.key, model, system, messages, maxTokens });
    } else if (vendor === 'google') {
      result = await relayGoogle({ key: hit.key, model, system, messages, maxTokens });
    }
  } catch (e) {
    return res.status(500).json({ ok: false, error: 'proxy_failed', detail: safeText(e && e.message || e), version: TEACHER_CHAT_VERSION });
  }
  if (!result.ok) return res.status(result.status || 502).json(result);

  return res.status(200).json({
    ok: true,
    version: TEACHER_CHAT_VERSION,
    vendor, model,
    choices: [{ message: { role: 'assistant', content: result.text } }],
    usage: {
      // Merge upstream token counts (when present) with always-known char
      // counts. Token counts feed the receipt's cost_usd estimate;
      // char counts are the no-key-required floor.
      ...(result.usage || {}),
      input_chars:  totalChars + system.length,
      output_chars: result.text.length,
    },
    request_sha256,
    output_sha256: sha256Hex(result.text),
    upstream_id: result.upstream_id || null,
    proxy_key_configured: true,
    tenant: a.tenant,
    served_by: 'vercel-function',
  });
}

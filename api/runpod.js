// Vercel serverless function: /v1/runpod/graphql (POST proxy) + /v1/runpod/health (GET).
//
// Lives on Vercel — like api/teacher-chat.js — so it can read the runpod_api_key
// that operators keep in the kolm.ai Vercel runtime. The API router and the
// CLI cannot see Vercel env vars, so the RunPod dispatch (src/cloud/runpod.js)
// falls back to this proxy when it has no local RUNPOD_API_KEY. Net effect: an
// operator who keeps their RunPod key in Vercel gets a working frontier->RunPod
// path without duplicating the key onto the API host.
//
// Vercel env is case-sensitive + operators add keys with mixed casing, so read
// every reasonable form (same policy as teacher-chat).

const RUNPOD_KEY_ALIASES = ['RUNPOD_API_KEY', 'runpod_api_key', 'KOLM_RUNPOD_TOKEN', 'runpod_token', 'RUNPOD_TOKEN'];
const RUNPOD_GRAPHQL = 'https://api.runpod.io/graphql';
// Origin of the kolm API used to validate bearer keys. Set KOLM_API_ORIGIN on
// Vercel; the fallback is the current production origin and goes away at the
// AWS cutover (docs/operations/aws-migration.md, Phase 3).
const KOLM_API_ORIGIN = String(process.env.KOLM_API_ORIGIN || 'https://kolmogorov-stack-production.up.railway.app').replace(/\/+$/, '');
const WHOAMI = KOLM_API_ORIGIN + '/v1/whoami';

function firstEnv(envs) {
  for (const k of envs) {
    const v = process.env[k];
    if (v && String(v).trim().length > 0) return { key: v, var: k };
  }
  return null;
}

const _keyCache = new Map();
async function authenticate(req) {
  const auth = String(req.headers.authorization || '');
  const m = auth.match(/^Bearer\s+(\S+)$/i);
  const bearer = m ? m[1] : (req.headers['x-api-key'] || '');
  if (!bearer) return { ok: false, status: 401, error: 'auth_required', detail: 'set Authorization: Bearer <kolm-key>' };
  if (!/^(ks_|kao_)[A-Za-z0-9_-]{6,256}$/.test(String(bearer))) {
    return { ok: false, status: 401, error: 'auth_invalid', detail: 'expected ks_ or kao_ prefix' };
  }
  const cached = _keyCache.get(bearer);
  if (cached && cached.expires > Date.now()) return { ok: true, tenant: cached.tenant };
  try {
    const r = await fetch(WHOAMI, { headers: { authorization: 'Bearer ' + bearer } });
    if (!r.ok) return { ok: false, status: 401, error: 'auth_invalid', detail: 'kolm key rejected by tenant store' };
    const j = await r.json();
    const tenant = (j && (j.tenant || j.tenant_id)) || 'unknown';
    _keyCache.set(bearer, { tenant, expires: Date.now() + 5 * 60 * 1000 });
    return { ok: true, tenant };
  } catch (e) {
    return { ok: false, status: 502, error: 'auth_upstream_failed', detail: String(e && e.message || e) };
  }
}

export default async function handler(req, res) {
  // GET → health probe. No auth (publishes only a boolean + which var matched).
  if (req.method === 'GET') {
    const hit = firstEnv(RUNPOD_KEY_ALIASES);
    return res.status(200).json({
      ok: true,
      provider: 'runpod',
      configured: !!hit,
      source: hit ? hit.var : null,
      served_by: 'vercel-function',
    });
  }
  if (req.method !== 'POST') {
    res.setHeader('allow', 'GET, POST');
    return res.status(405).json({ ok: false, error: 'method_not_allowed' });
  }

  // POST → proxy a RunPod GraphQL call. Auth required (an authed kolm tenant).
  const a = await authenticate(req);
  if (!a.ok) return res.status(a.status).json({ ok: false, error: a.error, detail: a.detail });

  const hit = firstEnv(RUNPOD_KEY_ALIASES);
  if (!hit) {
    return res.status(503).json({
      ok: false, error: 'runpod_key_not_configured',
      detail: `none of ${RUNPOD_KEY_ALIASES.join(' / ')} is set on this kolm.ai Vercel instance; add a RunPod key (https://runpod.io/console/user/settings).`,
    });
  }

  const body = (req.body && typeof req.body === 'object') ? req.body : {};

  // Serverless REST branch: proxy /v2/{endpointId}/{run|status/jobId} (api.runpod.ai)
  // for the training path (compute/backends/runpod.js) using the Vercel key.
  if (body.serverless && typeof body.serverless === 'object') {
    const sv = body.serverless;
    const endpointId = String(sv.endpointId || sv.endpoint_id || '').replace(/[^A-Za-z0-9_-]/g, '');
    const op = sv.op === 'status' ? 'status' : 'run';
    if (!endpointId) return res.status(400).json({ ok: false, error: 'missing_endpointId' });
    const jobId = String(sv.jobId || sv.job_id || '').replace(/[^A-Za-z0-9_-]/g, '');
    if (op === 'status' && !jobId) return res.status(400).json({ ok: false, error: 'missing_jobId_for_status' });
    const url = op === 'status'
      ? `https://api.runpod.ai/v2/${endpointId}/status/${jobId}`
      : `https://api.runpod.ai/v2/${endpointId}/run`;
    try {
      const r = await fetch(url, {
        method: op === 'status' ? 'GET' : 'POST',
        headers: { 'content-type': 'application/json', authorization: 'Bearer ' + hit.key },
        body: op === 'status' ? undefined : JSON.stringify(sv.payload || { input: sv.input || {} }),
      });
      const text = await r.text();
      let json = null; try { json = JSON.parse(text); } catch { /* raw */ }
      return res.status(r.ok ? 200 : 502).json({
        ok: r.ok, status: r.status, result: json, raw: r.ok ? undefined : text.slice(0, 1000),
        proxy_key_source: hit.var, tenant: a.tenant, served_by: 'vercel-function',
      });
    } catch (e) {
      return res.status(502).json({ ok: false, error: 'runpod_serverless_proxy_failed', detail: String(e && e.message || e) });
    }
  }

  const query = typeof body.query === 'string' ? body.query : '';
  if (!query) return res.status(400).json({ ok: false, error: 'missing_query', detail: 'POST { query, variables? } (GraphQL) or { serverless:{endpointId,op,...} }' });
  const variables = (body.variables && typeof body.variables === 'object') ? body.variables : {};

  try {
    const r = await fetch(RUNPOD_GRAPHQL, {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: 'Bearer ' + hit.key },
      body: JSON.stringify({ query, variables }),
    });
    const text = await r.text();
    let json = null;
    try { json = JSON.parse(text); } catch { /* leave raw */ }
    return res.status(r.ok ? 200 : 502).json({
      ok: r.ok && !(json && json.errors),
      status: r.status,
      data: json ? json.data : null,
      errors: json ? json.errors : null,
      raw: r.ok ? undefined : text.slice(0, 1000),
      proxy_key_source: hit.var,
      tenant: a.tenant,
      served_by: 'vercel-function',
    });
  } catch (e) {
    return res.status(502).json({ ok: false, error: 'runpod_proxy_failed', detail: String(e && e.message || e) });
  }
}

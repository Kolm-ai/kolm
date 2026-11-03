// Cloudflare zone API client - WAF custom rules, rate-limit rules, Email Routing.
// Uses the same account-scoped Bearer API token as src/r2.js. Adds the ability
// to discover zones for kolm.ai and apply hardening rules idempotently.
//
// Env required:
//   CLOUDFLARE_ACCOUNT_ID  (alias: cloudflare_account_id)
//   CLOUDFLARE_API_TOKEN   (alias: Cloudflare_api_token)
// Optional:
//   CLOUDFLARE_ZONE_ID     (alias: cloudflare_zone_id) - skips zone discovery
//   KOLM_DOMAIN            (default: kolm.ai)
//
// Token scopes needed: Zone.Zone (read), Zone.Firewall Services (edit),
// Zone.Email Routing Rules (edit). Rate-limit rules ride on the Firewall scope.
//
// Reference:
//   https://developers.cloudflare.com/api/operations/zone-list-zones
//   https://developers.cloudflare.com/waf/custom-rules/
//   https://developers.cloudflare.com/email-routing/setup/email-routing-rules/

import net from 'node:net';

function accountId() {
  return process.env.CLOUDFLARE_ACCOUNT_ID || process.env.cloudflare_account_id || '';
}

function apiToken() {
  return process.env.CLOUDFLARE_API_TOKEN || process.env.Cloudflare_api_token || '';
}

function defaultDomain() {
  return process.env.KOLM_DOMAIN || 'kolm.ai';
}

function authHeaders(extra = {}) {
  const token = apiToken();
  if (!token) throw new Error('CLOUDFLARE_API_TOKEN not set');
  return { Authorization: `Bearer ${token}`, ...extra };
}

const API = 'https://api.cloudflare.com/client/v4';

export function cloudflareConfigured() {
  return Boolean(accountId() && apiToken());
}

function fetchTimeoutMs() {
  const n = Number(process.env.KOLM_CLOUDFLARE_FETCH_TIMEOUT_MS || 10_000);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : 10_000;
}

async function cfFetch(url, init = {}) {
  const signal = init.signal || (typeof AbortSignal !== 'undefined' && AbortSignal.timeout
    ? AbortSignal.timeout(fetchTimeoutMs())
    : undefined);
  const r = await fetch(url, { ...init, signal, headers: { ...authHeaders(), ...(init.headers || {}) } });
  const j = await r.json().catch(() => ({}));
  if (!j.success) {
    const errs = JSON.stringify(j.errors || j);
    throw new Error(`cloudflare ${init.method || 'GET'} ${url} failed: ${errs}`);
  }
  return j;
}

export async function listZones() {
  const j = await cfFetch(`${API}/zones?account.id=${encodeURIComponent(accountId())}&per_page=50`);
  return j.result || [];
}

export async function discoverZoneId(domain = defaultDomain()) {
  if (process.env.CLOUDFLARE_ZONE_ID || process.env.cloudflare_zone_id) {
    return process.env.CLOUDFLARE_ZONE_ID || process.env.cloudflare_zone_id;
  }
  const zones = await listZones();
  const z = zones.find(z => z.name === domain);
  if (!z) throw new Error(`zone ${domain} not found on account ${accountId()}`);
  return z.id;
}

// ---- WAF custom rules (Rulesets API) -----------------------------------

// Returns the per-zone http_request_firewall_custom ruleset id (creating
// it implicitly via the rules endpoint if none exists yet).
async function customRulesetId(zone_id) {
  const j = await cfFetch(`${API}/zones/${zone_id}/rulesets?phase=http_request_firewall_custom`);
  const entry = (j.result || []).find(rs => rs.phase === 'http_request_firewall_custom');
  return entry ? entry.id : null;
}

export async function listCustomRules(zone_id) {
  const rid = await customRulesetId(zone_id);
  if (!rid) return [];
  const j = await cfFetch(`${API}/zones/${zone_id}/rulesets/${rid}`);
  return j.result?.rules || [];
}

// Replace the entire custom ruleset atomically. Cloudflare's API treats
// the rules list as the source of truth, so we PUT the full set on every
// apply. Idempotent - passing the same rules twice produces the same state.
export async function putCustomRules(zone_id, rules) {
  const rid = await customRulesetId(zone_id);
  if (rid) {
    return cfFetch(`${API}/zones/${zone_id}/rulesets/${rid}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rules }),
    });
  }
  // No ruleset exists yet - create one with the rules attached.
  return cfFetch(`${API}/zones/${zone_id}/rulesets`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      name: 'kolm WAF custom rules',
      kind: 'zone',
      phase: 'http_request_firewall_custom',
      rules,
    }),
  });
}

// Default rule set: block traversal probes, drop obvious credential-stuffing
// patterns on /v1/signin, gate WordPress probes, gate /v1/admin to ADMIN_ALLOW_CIDR.
function normalizeCidr(value) {
  const raw = String(value || '').trim();
  if (!raw) return null;
  const [addr, bits, extra] = raw.split('/');
  if (extra !== undefined || !net.isIP(addr)) {
    throw new Error(`invalid ADMIN_ALLOW_CIDR entry: ${raw}`);
  }
  if (bits === undefined) return addr;
  if (!/^\d+$/.test(bits)) throw new Error(`invalid ADMIN_ALLOW_CIDR entry: ${raw}`);
  const n = Number(bits);
  const max = net.isIP(addr) === 4 ? 32 : 128;
  if (!Number.isInteger(n) || n < 0 || n > max) {
    throw new Error(`invalid ADMIN_ALLOW_CIDR entry: ${raw}`);
  }
  return `${addr}/${n}`;
}

function adminAllowCidrs() {
  return (process.env.ADMIN_ALLOW_CIDR || '')
    .split(',')
    .map(normalizeCidr)
    .filter(Boolean);
}

export function defaultWafRules() {
  const adminAllow = adminAllowCidrs();
  const rules = [
    {
      description: 'block path traversal probes',
      expression: '(http.request.uri.path contains "../") or (http.request.uri.path contains "%2e%2e%2f")',
      action: 'block',
    },
    {
      description: 'block WordPress / PHP probes',
      expression: '(http.request.uri.path contains "/wp-admin") or (http.request.uri.path contains "/wp-login") or (http.request.uri.path contains "/xmlrpc.php") or (http.request.uri.path eq "/phpmyadmin")',
      action: 'block',
    },
    {
      description: 'managed-challenge unknown UAs on POST /v1/signin',
      expression: '(http.request.method eq "POST" and http.request.uri.path eq "/v1/signin" and http.user_agent eq "")',
      action: 'managed_challenge',
    },
    {
      description: 'block /.env and other secret-file probes',
      expression: '(http.request.uri.path in {"/.env" "/.git/config" "/.aws/credentials" "/wp-config.php"})',
      action: 'block',
    },
  ];
  if (adminAllow.length) {
    rules.push({
      description: 'gate /v1/admin to ADMIN_ALLOW_CIDR',
      expression: `(starts_with(http.request.uri.path, "/v1/admin") and not ip.src in {${adminAllow.join(' ')}})`,
      action: 'block',
    });
  }
  return rules;
}

// ---- Rate-limit rules (Rulesets API, phase=http_ratelimit) -------------

async function ratelimitRulesetId(zone_id) {
  const j = await cfFetch(`${API}/zones/${zone_id}/rulesets?phase=http_ratelimit`);
  const entry = (j.result || []).find(rs => rs.phase === 'http_ratelimit');
  return entry ? entry.id : null;
}

export async function listRateLimitRules(zone_id) {
  const rid = await ratelimitRulesetId(zone_id);
  if (!rid) return [];
  const j = await cfFetch(`${API}/zones/${zone_id}/rulesets/${rid}`);
  return j.result?.rules || [];
}

export async function putRateLimitRules(zone_id, rules) {
  const rid = await ratelimitRulesetId(zone_id);
  if (rid) {
    return cfFetch(`${API}/zones/${zone_id}/rulesets/${rid}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rules }),
    });
  }
  return cfFetch(`${API}/zones/${zone_id}/rulesets`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      name: 'kolm rate-limit rules',
      kind: 'zone',
      phase: 'http_ratelimit',
      rules,
    }),
  });
}

// Defense-in-depth on top of in-process rate limiting in src/router.js. These
// cut traffic at the edge before it ever reaches the API host.
export function defaultRateLimitRules() {
  return [
    {
      description: 'signin: 10 per minute per IP',
      expression: '(http.request.uri.path eq "/v1/signin" and http.request.method eq "POST")',
      action: 'block',
      ratelimit: { characteristics: ['ip.src'], period: 60, requests_per_period: 10, mitigation_timeout: 600 },
    },
    {
      description: 'signup: 5 per hour per IP',
      expression: '(http.request.uri.path eq "/v1/signup" and http.request.method eq "POST")',
      action: 'block',
      ratelimit: { characteristics: ['ip.src'], period: 3600, requests_per_period: 5, mitigation_timeout: 3600 },
    },
    {
      description: 'verify endpoint: 300 per minute per IP',
      expression: '(starts_with(http.request.uri.path, "/v1/receipts/verify"))',
      action: 'block',
      ratelimit: { characteristics: ['ip.src'], period: 60, requests_per_period: 300, mitigation_timeout: 60 },
    },
    {
      description: 'global anon API ceiling: 600 per minute per IP',
      expression: '(starts_with(http.request.uri.path, "/v1/"))',
      action: 'block',
      ratelimit: { characteristics: ['ip.src'], period: 60, requests_per_period: 600, mitigation_timeout: 60 },
    },
  ];
}

// ---- Email Routing ------------------------------------------------------

export async function emailRoutingStatus(zone_id) {
  return cfFetch(`${API}/zones/${zone_id}/email/routing`);
}

export async function enableEmailRouting(zone_id) {
  return cfFetch(`${API}/zones/${zone_id}/email/routing/enable`, { method: 'POST' });
}

export async function listEmailRules(zone_id) {
  const j = await cfFetch(`${API}/zones/${zone_id}/email/routing/rules`);
  return j.result || [];
}

export async function putEmailRule(zone_id, rule) {
  // Cloudflare Email Routing rules are appended, not replaced. We dedupe by
  // matcher value (the inbound address) so re-applying is idempotent.
  const existing = await listEmailRules(zone_id);
  const inbound = rule.matchers?.[0]?.value || '';
  const dupe = existing.find(r => r.matchers?.[0]?.value === inbound);
  if (dupe) {
    return cfFetch(`${API}/zones/${zone_id}/email/routing/rules/${dupe.tag}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(rule),
    });
  }
  return cfFetch(`${API}/zones/${zone_id}/email/routing/rules`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(rule),
  });
}

// Default forward destination. Operators override with KOLM_EMAIL_FORWARD.
function forwardTo() {
  return process.env.KOLM_EMAIL_FORWARD || 'dev@kolm.ai';
}

function normalizeDomain(value) {
  const domain = String(value || defaultDomain()).trim().toLowerCase().replace(/\.$/, '');
  const labels = domain.split('.');
  if (
    domain.length < 3
    || domain.length > 253
    || labels.length < 2
    || labels.some((label) => !/^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/.test(label))
  ) {
    throw new Error(`invalid Cloudflare zone domain: ${value}`);
  }
  return domain;
}

function normalizeEmail(value) {
  const email = String(value || '').trim().toLowerCase();
  if (!/^[^\s@<>"]+@[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+$/.test(email)) {
    throw new Error(`invalid KOLM_EMAIL_FORWARD: ${value}`);
  }
  return email;
}

export function defaultEmailRules(options = {}) {
  const opts = typeof options === 'string' ? { domain: options } : (options || {});
  const domain = normalizeDomain(opts.domain || defaultDomain());
  const dest = normalizeEmail(opts.forwardTo || forwardTo());
  const mkRule = (local, label) => ({
    actions: [{ type: 'forward', value: [dest] }],
    matchers: [{ field: 'to', type: 'literal', value: `${local}@${domain}` }],
    enabled: true,
    name: label,
    priority: 0,
  });
  return [
    mkRule('sales', 'sales owner'),
    mkRule('dev', 'dev owner'),
    mkRule('support', 'support owner'),
    mkRule('security', 'security owner'),
    mkRule('legal', 'legal owner'),
    mkRule('compliance', 'compliance owner'),
    mkRule('hi', 'hi owner'),
    mkRule('contact', 'contact owner'),
  ];
}

// ---- One-shot bootstrap -------------------------------------------------

export async function bootstrapZoneHardening({ domain = defaultDomain() } = {}) {
  if (!cloudflareConfigured()) throw new Error('Cloudflare not configured (need account_id + api_token)');
  const zone_id = await discoverZoneId(domain);
  const out = { zone_id, domain, applied: {} };

  // WAF
  const wafRules = defaultWafRules();
  await putCustomRules(zone_id, wafRules);
  out.applied.waf_rules = wafRules.length;

  // Rate-limit
  const rlRules = defaultRateLimitRules();
  await putRateLimitRules(zone_id, rlRules);
  out.applied.rate_limit_rules = rlRules.length;

  // Email Routing - soft enable. If the API rejects (token lacks scope, or
  // the zone hasn't accepted the DNS records yet), we still surface the WAF
  // + rate-limit work as applied.
  try {
    await enableEmailRouting(zone_id).catch(() => {}); // idempotent
    const emailRules = defaultEmailRules({ domain });
    for (const r of emailRules) await putEmailRule(zone_id, r);
    out.applied.email_rules = emailRules.length;
  } catch (e) {
    out.email_error = String(e.message || e);
  }

  return out;
}

// Provider-key vault - per-employee and per-team upstream provider keys
// (OpenAI / Anthropic / Gemini / OpenRouter / ...), encrypted at rest.
//
// This is the core of the "every employee's AI use, in one place you control"
// product: a team member stores their own provider key here, the gateway routes
// their traffic under it, and the call lands in the team lake attributed to them.
//
// Storage split (deliberate):
//   - ciphertext + metadata rows live in the multi-tenant `provider_keys` table
//     via store.js, so they persist on the deployed DB (SQLite (production) /
//     JSON (dev)) and are tenant/team scoped like every other row. Transactional
//     guarantees (atomic upsert/cascade) hold in SQLite mode only; the JSON dev
//     backend is a single-process file store with no real transaction boundary.
//   - the AES-256-GCM key lives in the secrets-vault.key file (under
//     KOLM_DATA_DIR - the EFS volume), reused from secrets-vault.js.
//
// Scopes:
//   - 'member': usable only by the member who stored it (actor_id match).
//   - 'team':   usable by any active team member; only admins rotate/delete.
// Resolution precedence inside the gateway: member key -> team key -> (caller
// falls through to env / proxy in the router).

import crypto from 'node:crypto';
import { insert, update, remove, findByField } from './store.js';
import { encrypt, decrypt } from './secrets-vault.js';
import { membershipOf } from './teams.js';

const TABLE = 'provider_keys';
const PROVIDERS = ['openai', 'anthropic', 'google', 'openrouter', 'groq', 'together', 'fireworks', 'deepseek', 'mistral', 'cohere'];

function cleanProvider(p) {
  const s = String(p || '').trim().toLowerCase();
  if (!s) throw Object.assign(new Error('provider required'), { code: 'bad_provider' });
  return s;
}

function nowIso() { return new Date().toISOString(); }
function genId() { return 'pk_' + crypto.randomBytes(9).toString('hex'); }

function keyPrefix(value) {
  const s = String(value || '');
  // Show enough to recognize (sk-..., ks_...) without leaking the secret.
  return s.length <= 10 ? s.slice(0, 3) + '...' : s.slice(0, 6) + '...' + s.slice(-2);
}
function keySha256(value) {
  return crypto.createHash('sha256').update(String(value || ''), 'utf8').digest('hex');
}

// Redacted view - never returns plaintext, ciphertext, iv, or tag.
export function redactProviderKey(row) {
  if (!row || typeof row !== 'object') return row;
  return {
    id: row.id,
    tenant_id: row.tenant_id,
    team_id: row.team_id || null,
    actor_id: row.actor_id || null,
    provider: row.provider,
    scope: row.scope,
    label: row.label || '',
    key_prefix: row.key_prefix || '',
    key_sha256: row.key_sha256 || '',
    created_at: row.created_at,
    updated_at: row.updated_at,
    last_used_at: row.last_used_at || null,
    value_included: false,
    encrypted_at_rest: !!(row.enc && row.enc.ciphertext),
  };
}

function rowsForTenant(tenantId) {
  if (!tenantId) return [];
  return findByField(TABLE, 'tenant_id', tenantId).filter((r) => r && !r._deleted);
}

// Team keys are SHARED across a team's members, but they must never leak to a
// FOREIGN tenant that simply supplies the same team_id in a request body
// (W936 #5 cross-tenant isolation). A caller may read a team's keys only when it
// is genuinely associated with that team - which is true in exactly two ways:
//
//   1. the team key was stored under the caller's OWN tenant_id (the owner who
//      put the key, and the single-tenant case where no membership rows exist), OR
//   2. the caller's tenant is an ACTIVE member of the team (membershipOf), which
//      is how a member on a different tenant legitimately shares the owner's key.
//
// A tenant that is neither the storer nor an active member resolves nothing,
// even with the right team_id. This is the data-boundary fence; the route still
// authorizes membership before passing teamId (defense in depth).
function rowsForTeam(teamId, tenantId) {
  if (!teamId || !tenantId) return [];
  const isActiveMember = (() => {
    try { return !!membershipOf(teamId, tenantId); }
    catch (_) { return false; } // never let a teams lookup failure open the fence
  })();
  return findByField(TABLE, 'team_id', teamId).filter((r) =>
    r && !r._deleted && r.scope === 'team' && (r.tenant_id === tenantId || isActiveMember));
}

// Store (upsert) a provider key. One row per (tenant, team|null, actor|scope, provider).
export function putProviderKey({ tenantId, teamId = null, actorId = null, provider, scope = 'member', value, label = '' } = {}) {
  if (!tenantId) throw Object.assign(new Error('tenant required'), { code: 'no_tenant' });
  if (value == null || value === '') throw Object.assign(new Error('provider key value required'), { code: 'bad_value' });
  const prov = cleanProvider(provider);
  const sc = scope === 'team' ? 'team' : 'member';
  if (sc === 'team' && !teamId) throw Object.assign(new Error('team scope requires a team'), { code: 'no_team' });
  const enc = encrypt(value);
  const now = nowIso();
  const match = (r) => r.provider === prov && r.scope === sc &&
    (sc === 'team' ? r.team_id === teamId : (r.actor_id === actorId && (r.team_id || null) === (teamId || null)));
  const existing = rowsForTenant(tenantId).find(match);
  const patch = {
    enc, key_prefix: keyPrefix(value), key_sha256: keySha256(value),
    label: String(label || '').slice(0, 80), updated_at: now,
  };
  if (existing) {
    update(TABLE, (r) => r.id === existing.id, patch);
    return redactProviderKey({ ...existing, ...patch });
  }
  const row = {
    id: genId(), tenant_id: tenantId, team_id: teamId || null, actor_id: actorId || null,
    provider: prov, scope: sc, created_at: now, last_used_at: null, ...patch,
  };
  insert(TABLE, row);
  return redactProviderKey(row);
}

// Resolve plaintext for the gateway ONLY. Precedence: this member's key, then
// the team key. Returns null if neither is configured. Stamps last_used_at.
//
// memberOf (optional) - a defense-in-depth predicate ({tenantId,teamId}) => bool
// the gateway can pass to assert the caller is genuinely a member of the team
// BEFORE a team-scoped key is resolved. The router already nulls teamId for a
// non-member, so this is belt-and-suspenders: even if a teamId leaks into this
// call for a non-member, a provided memberOf that returns falsy forces the
// team branch to resolve nothing. Omitting memberOf preserves the prior
// behavior exactly (rowsForTeam's own tenant/membership fence still applies).
export function resolveProviderKey({ tenantId, teamId = null, actorId = null, provider, memberOf = null } = {}) {
  if (!tenantId || !provider) return null;
  const prov = cleanProvider(provider);
  // Member key: the caller's own, scoped to their tenant + actor.
  const member = actorId
    ? rowsForTenant(tenantId).find((r) => r.provider === prov && r.scope === 'member' && r.actor_id === actorId)
    : null;
  // Team-membership gate (defense-in-depth). When memberOf is supplied and
  // reports the caller is NOT a member, the team branch resolves nothing even
  // if teamId leaked in. A throwing predicate fails CLOSED (no team key).
  let teamAllowed = true;
  if (teamId && typeof memberOf === 'function') {
    try { teamAllowed = !!memberOf({ tenantId, teamId }); }
    catch (_) { teamAllowed = false; }
  }
  // Team key: shared across the team's members, fenced to the caller's tenant
  // (a team belongs to one tenant; a foreign tenant supplying the same team_id
  // must never resolve it - W936 #5).
  const team = (teamId && teamAllowed)
    ? rowsForTeam(teamId, tenantId).find((r) => r.provider === prov && r.scope === 'team')
    : null;
  const pick = member || team;
  if (!pick || !pick.enc) return null;
  try {
    const value = decrypt(pick.enc);
    update(TABLE, (r) => r.id === pick.id, { last_used_at: nowIso() });
    return value;
  } catch (_) { return null; } // deliberate: a corrupt row must not break dispatch
}

// List redacted keys visible to the caller. A member sees their own member-keys
// plus the team's team-keys; an admin sees all team-keys for the team.
export function listProviderKeys({ tenantId, teamId = null, actorId = null, isAdmin = false } = {}) {
  // Member keys: the caller's own (or all member-keys for an admin), scoped to
  // the caller's tenant. Team keys: shared, looked up by team_id across tenants.
  const memberRows = rowsForTenant(tenantId).filter((r) => r.scope === 'member' && (r.actor_id === actorId || isAdmin));
  const teamRows = teamId ? rowsForTeam(teamId, tenantId).filter((r) => r.scope === 'team') : [];
  const visible = [...memberRows, ...teamRows];
  return visible.map(redactProviderKey).sort((a, b) => String(a.provider).localeCompare(String(b.provider)));
}

export function deleteProviderKey({ tenantId, id, actorId = null, isAdmin = false } = {}) {
  if (!tenantId || !id) return { ok: false, deleted: false, reason: 'bad_request' };
  const row = rowsForTenant(tenantId).find((r) => r.id === id);
  if (!row) return { ok: true, deleted: false };
  // member keys: owner or admin; team keys: admin only.
  const allowed = row.scope === 'team' ? isAdmin : (row.actor_id === actorId || isAdmin);
  if (!allowed) return { ok: false, deleted: false, reason: 'forbidden' };
  remove(TABLE, (r) => r.id === id && r.tenant_id === tenantId);
  return { ok: true, deleted: true, id };
}

export function supportedProviders() { return PROVIDERS.slice(); }

// Cascade: soft-delete every team-scoped provider key when its team is deleted.
// Called by teams.deleteTeam so a deleted team leaves no resolvable shared keys
// behind (resolveProviderKey/listProviderKeys/rowsForTeam all filter !_deleted).
// Soft-delete (not hard remove) keeps the ciphertext row for audit/forensics
// while making it unresolvable, matching how deleteTeam soft-deletes the team
// and its invites. Returns the count of keys cascaded. Idempotent: a second
// call after the keys are already deleted is a no-op (returns 0).
export function deleteTeamProviderKeys(teamId) {
  if (!teamId) return 0;
  const now = nowIso();
  return update(
    TABLE,
    (r) => r && r.team_id === teamId && r.scope === 'team' && !r._deleted,
    { _deleted: true, deleted_at: now, updated_at: now, deleted_reason: 'team_deleted' },
  );
}

export default { putProviderKey, resolveProviderKey, listProviderKeys, deleteProviderKey, deleteTeamProviderKeys, redactProviderKey, supportedProviders };

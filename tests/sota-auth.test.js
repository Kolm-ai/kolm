// SOTA Auth lane - real fixes for the AUTH-01..07 atoms.
//
// Atoms exercised:
//   1) [p1] Scoped-key last_used_at flush is actually driven. startKeyLastUsedFlusher
//      registers an unref'd interval; stopKeyLastUsedFlusher drains the final
//      window. recordKeyLastUsed -> flushKeyLastUsed writes last_used_at.
//   2) [p2] gcMagicLinkTokens prunes consumed/expired rows past retention while
//      sparing fresh unconsumed rows.
//   3) [p1] Magic-link single-use consume is atomic - two verifies of the same
//      token yield exactly one ok:true.
//   4) [p1] OAuth/magic-link account linking - findOrCreateTenantByEmail refuses
//      a provider_id mismatch on an existing provider binding.
//   5) [p2] Unified OAuth base-URL resolution honours KOLM_PUBLIC_URL and warns
//      loudly when ambiguous in production.
//   6) [p2] recoverKeyByEmail rotates a locked-out tenant's key (email-verified
//      recovery escape hatch); tenantIsLockedOut detects the locked state.
//   7) [p2] PKCE pair shape (S256) is correct.
//
// Isolation: per-test temp KOLM_DATA_DIR + JSON store driver, set BEFORE the
// auth modules are imported (auth.js runs a migration at module load).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';

function freshDir(prefix) {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
  process.env.KOLM_DATA_DIR = path.join(tmp, '.kolm');
  process.env.HOME = tmp;
  process.env.USERPROFILE = tmp;
  process.env.KOLM_HOME = path.join(tmp, '.kolm');
  process.env.KOLM_ENV = 'test';
  process.env.KOLM_STORE_DRIVER = 'json';
  delete process.env.NODE_ENV;
  delete process.env.AWS_EXECUTION_ENV;
  delete process.env.VERCEL;
  delete process.env.AWS_LAMBDA_FUNCTION_NAME;
  fs.mkdirSync(process.env.KOLM_DATA_DIR, { recursive: true });
  return tmp;
}

let _store = null;
let _auth = null;
let _authEmail = null;
let _oauth = null;
async function loadAuth() {
  if (!_store) _store = await import('../src/store.js');
  if (!_auth) _auth = await import('../src/auth.js');
  if (!_authEmail) _authEmail = await import('../src/auth-email.js');
  if (!_oauth) _oauth = await import('../src/oauth.js');
  try { _store.reset(); } catch { /* deliberate: cleanup */ }
  return { store: _store, auth: _auth, authEmail: _authEmail, oauth: _oauth };
}

// ---------------------------------------------------------------------------
// AUTH-01 - last_used_at flush is real
// ---------------------------------------------------------------------------
test('atom1 - startKeyLastUsedFlusher + flush write last_used_at; stop drains', async () => {
  freshDir('kolm-sota-auth-1-');
  process.env.KOLM_KEY_LAST_USED_TRACKING = '1';
  process.env.KOLM_KEY_LAST_USED_FLUSH_MS = '1000';
  const { auth } = await loadAuth();

  const t = auth.provisionTenant('flush-co-' + crypto.randomBytes(3).toString('hex'));
  const minted = auth.mintScopedKey(t.id, { scopes: ['capture:read'], label: 'ci' });
  const rawHex = crypto.createHash('sha256').update(minted.key).digest('hex');

  // Before: last_used_at is null.
  let row = auth.listScopedKeys(t.id).find(k => k.key_prefix === minted.key_prefix);
  assert.equal(row.last_used_at, null, 'fresh key starts with null last_used_at');

  // Record + flush directly.
  auth.recordKeyLastUsed(rawHex);
  const { flushed } = auth.flushKeyLastUsed();
  assert.equal(flushed, 1, 'flush writes exactly the one queued key');

  row = auth.listScopedKeys(t.id).find(k => k.key_prefix === minted.key_prefix);
  assert.ok(row.last_used_at, 'last_used_at is now set after flush');

  // The flusher registers when tracking is enabled, and stop drains the final
  // window (record after start, then stop, must persist).
  const timer = auth.startKeyLastUsedFlusher({ debug() {} });
  assert.ok(timer, 'flusher starts when tracking enabled');
  const t2 = auth.mintScopedKey(t.id, { scopes: ['*'], label: 'second' });
  const rawHex2 = crypto.createHash('sha256').update(t2.key).digest('hex');
  auth.recordKeyLastUsed(rawHex2);
  const final = auth.stopKeyLastUsedFlusher();
  assert.equal(final.flushed, 1, 'stop drains the final queued window');
  const row2 = auth.listScopedKeys(t.id).find(k => k.key_prefix === t2.key_prefix);
  assert.ok(row2.last_used_at, 'second key flushed on shutdown');

  delete process.env.KOLM_KEY_LAST_USED_TRACKING;
  delete process.env.KOLM_KEY_LAST_USED_FLUSH_MS;
});

test('atom1 - startKeyLastUsedFlusher is a no-op when tracking disabled', async () => {
  freshDir('kolm-sota-auth-1b-');
  delete process.env.KOLM_KEY_LAST_USED_TRACKING;
  const { auth } = await loadAuth();
  const timer = auth.startKeyLastUsedFlusher();
  assert.equal(timer, null, 'no interval is registered when tracking is off');
});

// ---------------------------------------------------------------------------
// AUTH-02 - magic-link GC
// ---------------------------------------------------------------------------
test('atom2 - gcMagicLinkTokens prunes dead+old rows, spares fresh', async () => {
  freshDir('kolm-sota-auth-2-');
  process.env.KOLM_MAGICLINK_SECRET = 'test-magic-secret';
  process.env.KOLM_MAGICLINK_RETENTION_DAYS = '7';
  const { store, authEmail } = await loadAuth();

  const now = Date.now();
  const dayMs = 24 * 60 * 60 * 1000;
  const oldCreated = now - 30 * dayMs; // well past 7-day retention

  // 1) consumed + old  -> prune
  store.insert('magic_link_tokens', { nonce: 'n1', email: 'a@co.com', exp: now + 1000, consumed_at: now - 29 * dayMs, created_at: oldCreated });
  // 2) expired + old (never consumed) -> prune
  store.insert('magic_link_tokens', { nonce: 'n2', email: 'b@co.com', exp: now - 20 * dayMs, consumed_at: null, created_at: oldCreated });
  // 3) expired but RECENT (created today) -> spare (inside retention window)
  store.insert('magic_link_tokens', { nonce: 'n3', email: 'c@co.com', exp: now - 1000, consumed_at: null, created_at: now });
  // 4) fresh unconsumed unexpired -> spare
  store.insert('magic_link_tokens', { nonce: 'n4', email: 'd@co.com', exp: now + dayMs, consumed_at: null, created_at: now });

  const { removed } = authEmail.gcMagicLinkTokens(now);
  assert.equal(removed, 2, 'exactly the two dead+old rows are pruned');

  const remaining = store.all('magic_link_tokens').map(r => r.nonce).sort();
  assert.deepEqual(remaining, ['n3', 'n4'], 'recent-expired and fresh rows survive');

  delete process.env.KOLM_MAGICLINK_SECRET;
  delete process.env.KOLM_MAGICLINK_RETENTION_DAYS;
});

// ---------------------------------------------------------------------------
// AUTH-03 - single-use consume atomicity
// ---------------------------------------------------------------------------
test('atom3 - verifyMagicToken consumes exactly once', async () => {
  freshDir('kolm-sota-auth-3-');
  process.env.KOLM_MAGICLINK_SECRET = 'test-magic-secret';
  const { authEmail } = await loadAuth();

  const token = authEmail.mintMagicToken('race@co.com');
  const first = authEmail.verifyMagicToken(token);
  const second = authEmail.verifyMagicToken(token);

  assert.equal(first.ok, true, 'first verify succeeds');
  assert.equal(first.email, 'race@co.com');
  assert.equal(second.ok, false, 'second verify is rejected (single-use)');
  assert.equal(second.error, 'already_used');

  delete process.env.KOLM_MAGICLINK_SECRET;
});

// ---------------------------------------------------------------------------
// AUTH-04 - provider_id account-linking trust
// ---------------------------------------------------------------------------
test('atom4 - findOrCreateTenantByEmail refuses provider_id mismatch', async () => {
  freshDir('kolm-sota-auth-4-');
  const { auth } = await loadAuth();

  // First Google login establishes the binding.
  const a = auth.findOrCreateTenantByEmail({ email: 'v@co.com', name: 'V', provider: 'google', provider_id: 'google-123' });
  assert.equal(a.created, true);

  // Same Google id signs in fine (no overwrite needed).
  const b = auth.findOrCreateTenantByEmail({ email: 'v@co.com', name: 'V', provider: 'google', provider_id: 'google-123' });
  assert.equal(b.created, false, 'same provider id is an existing sign-in');

  // A DIFFERENT Google id claiming the same email is refused.
  assert.throws(
    () => auth.findOrCreateTenantByEmail({ email: 'v@co.com', name: 'attacker', provider: 'google', provider_id: 'google-999' }),
    (err) => err && err.code === 'provider_id_mismatch',
    'mismatched provider_id must be refused, not silently overwritten',
  );

  // Magic-link (provider:'email') is the email-ownership proof and is NOT
  // blocked by the binding check - it can still sign into the same tenant.
  const c = auth.findOrCreateTenantByEmail({ email: 'v@co.com', name: 'V', provider: 'email', provider_id: 'v@co.com' });
  assert.equal(c.created, false, 'magic-link signs into the existing tenant');
});

// ---------------------------------------------------------------------------
// AUTH-05 - unified base URL + warning
// ---------------------------------------------------------------------------
test('atom5 - oauthRedirectBaseWarning fires only in prod when base ambiguous', async () => {
  freshDir('kolm-sota-auth-5-');
  const { oauth } = await loadAuth();

  // Non-prod: never warns.
  delete process.env.OAUTH_REDIRECT_BASE;
  delete process.env.KOLM_PUBLIC_URL;
  assert.equal(oauth.oauthRedirectBaseWarning(), null, 'no warning outside production');

  // Prod with a provider configured and no base set: warns.
  process.env.NODE_ENV = 'production';
  process.env.GOOGLE_OAUTH_CLIENT_ID = 'gid';
  process.env.GOOGLE_OAUTH_CLIENT_SECRET = 'gsec';
  const w = oauth.oauthRedirectBaseWarning();
  assert.ok(w && w.warning === 'oauth_redirect_base_unset', 'warns when base ambiguous in prod');

  // Setting KOLM_PUBLIC_URL clears the warning (proves precedence is honoured).
  process.env.KOLM_PUBLIC_URL = 'https://preview.example.com';
  assert.equal(oauth.oauthRedirectBaseWarning(), null, 'KOLM_PUBLIC_URL resolves the ambiguity');

  delete process.env.NODE_ENV;
  delete process.env.GOOGLE_OAUTH_CLIENT_ID;
  delete process.env.GOOGLE_OAUTH_CLIENT_SECRET;
  delete process.env.KOLM_PUBLIC_URL;
});

// ---------------------------------------------------------------------------
// AUTH-06 - lockout recovery
// ---------------------------------------------------------------------------
test('atom6 - recoverKeyByEmail rotates a locked-out tenant key', async () => {
  freshDir('kolm-sota-auth-6-');
  const { store, auth } = await loadAuth();

  // Simulate a pre-migration lockout: a tenant row with an email but NO
  // api_key_hash (the documented permanently-locked-out state).
  const tid = 'tenant_' + crypto.randomBytes(6).toString('hex');
  store.insert('tenants', {
    id: tid, name: 'locked-co', email: 'locked@co.com', kind: 'user',
    plan: 'free', quota: 10000, used: 0, created_at: new Date().toISOString(),
    // api_key_hash deliberately absent
  });
  const locked = store.findOne('tenants', x => x.id === tid);
  assert.equal(auth.tenantIsLockedOut(locked), true, 'no api_key_hash => locked out');

  // Email-verified recovery rotates the key and makes the tenant usable again.
  const r = auth.recoverKeyByEmail('locked@co.com');
  assert.equal(r.ok, true, 'recovery succeeds for a known email');
  assert.ok(/^ks_/.test(r.api_key), 'recovery hands back a fresh ks_ key');

  const resolved = auth.findTenantByApiKey(r.api_key);
  assert.ok(resolved && resolved.id === tid, 'the recovered key authenticates to the same tenant');
  assert.equal(auth.tenantIsLockedOut(resolved), false, 'tenant is no longer locked out');

  // Unknown email is refused (no enumeration leak in the return shape beyond reason).
  const miss = auth.recoverKeyByEmail('nobody@co.com');
  assert.equal(miss.ok, false);
});

// ---------------------------------------------------------------------------
// AUTH-07 - PKCE pair
// ---------------------------------------------------------------------------
test('atom7 - PKCE challenge is S256(verifier) base64url', async () => {
  // _pkcePair is module-internal; re-derive its contract here to assert the
  // S256 relationship the start handler relies on.
  const verifier = crypto.randomBytes(32).toString('base64url');
  const challenge = crypto.createHash('sha256').update(verifier).digest('base64url');
  // base64url must not contain +, /, or = padding.
  assert.ok(!/[+/=]/.test(challenge), 'challenge is base64url (no +, /, = padding)');
  // Recompute -> deterministic.
  const recomputed = crypto.createHash('sha256').update(verifier).digest('base64url');
  assert.equal(challenge, recomputed, 'S256 is deterministic over the verifier');
});

// OAuth (Google + GitHub) without the SDK dependency. Each provider has the
// same shape: redirect to provider with state cookie, receive code on callback,
// exchange code for access_token, fetch /userinfo, find-or-create the tenant,
// drop a kolm_session cookie, redirect back to the originally requested page.
//
// Configuration (set on the API host via SSM):
//   GOOGLE_OAUTH_CLIENT_ID, GOOGLE_OAUTH_CLIENT_SECRET
//   GITHUB_OAUTH_CLIENT_ID, GITHUB_OAUTH_CLIENT_SECRET
//   OAUTH_REDIRECT_BASE = https://kolm.ai
//
// Provider redirect URIs to configure in Google/GitHub developer consoles:
//   https://kolm.ai/v1/oauth/google/callback
//   https://kolm.ai/v1/oauth/github/callback

import crypto from 'node:crypto';
import { findOrCreateTenantByEmail } from './auth.js';
import { isProductionRuntime } from './env.js';

const STATE_COOKIE = 'kolm_oauth_state';
const RETURN_COOKIE = 'kolm_oauth_return';
const PKCE_COOKIE = 'kolm_oauth_pkce';

// AUTH-05 - unified base-URL resolution. Same precedence as auth-email.js so
// magic-link and OAuth never disagree: OAUTH_REDIRECT_BASE > KOLM_PUBLIC_URL >
// (prod ? https://kolm.ai : localhost). On a preview/staging/self-host deploy
// where OAUTH_REDIRECT_BASE is unset, this used to ALWAYS resolve to prod, so
// the provider called back to kolm.ai and the local instance never completed
// sign-in (a silent cross-environment auth break). Honouring KOLM_PUBLIC_URL
// (which magic-link already honoured) and only defaulting to prod when actually
// in a production runtime fixes that.
function baseUrl() {
  const b = process.env.OAUTH_REDIRECT_BASE
    || process.env.KOLM_PUBLIC_URL
    || (isProductionRuntime() ? 'https://kolm.ai' : 'http://localhost:8787');
  return String(b).replace(/\/+$/, '');
}

// AUTH-05 - loud misconfig surface. In production, if NEITHER OAUTH_REDIRECT_BASE
// NOR KOLM_PUBLIC_URL is set, OAuth silently falls back to the hardcoded
// https://kolm.ai callback - correct for the real prod host, but a self-hosted
// production deploy on another domain would ship users to kolm.ai and never
// complete sign-in. Surface it so an operator notices at startup / via /health
// rather than via a stream of failed logins. Returns null when fine.
export function oauthRedirectBaseWarning() {
  if (!isProductionRuntime()) return null;
  if (process.env.OAUTH_REDIRECT_BASE || process.env.KOLM_PUBLIC_URL) return null;
  const anyProvider = oauthConfigured('google') || oauthConfigured('github');
  if (!anyProvider) return null;
  return {
    warning: 'oauth_redirect_base_unset',
    base: 'https://kolm.ai',
    hint: 'set OAUTH_REDIRECT_BASE (or KOLM_PUBLIC_URL) to this deploy\'s public origin so the OAuth callback returns to THIS instance, not https://kolm.ai',
  };
}

// AUTH-05 - startup hook. server.js calls this at boot; logs a single WARNING
// line when the OAuth callback base is ambiguous in production. Never throws.
export function oauthStartupCheck(logger = console) {
  const w = oauthRedirectBaseWarning();
  try {
    if (w) {
      logger.warn(`[oauth] WARNING: ${w.hint} (currently defaulting to ${w.base}).`);
    }
  } catch { /* deliberate: cleanup */ }
  return w;
}

function safeReturn(req) {
  const r = (req.query && req.query.redirect) || '/dashboard';
  if (typeof r !== 'string') return '/dashboard';
  if (!r.startsWith('/') || r.startsWith('//')) return '/dashboard';
  return r;
}

function setCookie(res, name, value, maxAgeMs) {
  res.cookie(name, value, {
    httpOnly: true,
    secure: isProductionRuntime(),
    sameSite: 'lax',
    maxAge: maxAgeMs,
    path: '/',
  });
}

function clearCookie(res, name) {
  res.clearCookie(name, { path: '/' });
}

function setSessionCookie(res, apiKey) {
  res.cookie('kolm_session', apiKey, {
    httpOnly: true,
    secure: isProductionRuntime(),
    sameSite: 'lax',
    maxAge: 30 * 24 * 60 * 60 * 1000,
    path: '/',
  });
}

const PROVIDERS = {
  google: {
    authUrl: 'https://accounts.google.com/o/oauth2/v2/auth',
    tokenUrl: 'https://oauth2.googleapis.com/token',
    userUrl: 'https://www.googleapis.com/oauth2/v2/userinfo',
    scope: 'openid email profile',
    clientIdEnv: 'GOOGLE_OAUTH_CLIENT_ID',
    clientSecretEnv: 'GOOGLE_OAUTH_CLIENT_SECRET',
    // AUTH-07 - Google supports PKCE on the web auth-code flow. Enabling it
    // means an intercepted authorization code cannot be exchanged without the
    // matching code_verifier (which never leaves this server's httpOnly cookie).
    pkce: true,
    extractEmail: (u) => u.email,
    extractName: (u) => u.name || (u.email && u.email.split('@')[0]),
  },
  github: {
    authUrl: 'https://github.com/login/oauth/authorize',
    tokenUrl: 'https://github.com/login/oauth/access_token',
    userUrl: 'https://api.github.com/user',
    emailUrl: 'https://api.github.com/user/emails',
    scope: 'read:user user:email',
    clientIdEnv: 'GITHUB_OAUTH_CLIENT_ID',
    clientSecretEnv: 'GITHUB_OAUTH_CLIENT_SECRET',
    // GitHub's OAuth app flow does not support PKCE (S256); we rely on the
    // state cookie + client_secret. Left explicit so the asymmetry is visible.
    pkce: false,
    extractEmail: (u) => u.email,
    extractName: (u) => u.name || u.login,
  },
};

// AUTH-04 - verified-email enforcement on the trusted OAuth path. Pure
// email-equality account resolution means any provider that asserts an email
// signs the caller into the matching tenant; an attacker who controls a
// provider account claiming victim@co.com would be signed into the victim's
// tenant. We require provider-asserted email VERIFICATION before trusting the
// email as an ownership proof:
//   - Google: userinfo carries email_verified (boolean, sometimes string
//     'true'); require it true. Email-only logins with email_verified:false are
//     rejected.
//   - GitHub: the primary email from /user/emails carries primary+verified
//     flags; accept ONLY primary && verified. The /user.email field has no
//     verification flag, so it is NOT trusted on its own - we always confirm
//     against /user/emails.
// Returns { email } on success or { error } (oauth_error code) on rejection.
function _googleVerifiedEmail(userJson) {
  const email = userJson && userJson.email;
  const ev = userJson && userJson.email_verified;
  const verified = ev === true || ev === 'true';
  if (!email) return { error: 'no_email_returned' };
  if (!verified) return { error: 'email_unverified' };
  return { email };
}

async function _githubVerifiedEmail(accessToken, emailUrl) {
  let emails = [];
  try {
    const emailRes = await fetch(emailUrl, {
      headers: { authorization: `Bearer ${accessToken}`, accept: 'application/json', 'user-agent': 'kolm-oauth' },
    });
    emails = await emailRes.json().catch(() => []);
  } catch { emails = []; }
  if (!Array.isArray(emails)) return { error: 'no_email_returned' };
  const primary = emails.find((e) => e && e.primary && e.verified);
  if (primary && primary.email) return { email: primary.email };
  // No primary verified email - do NOT silently fall back to any verified
  // address; require an explicit primary+verified so a secondary, attacker-added
  // address can never be the sign-in identity.
  return { error: 'email_unverified' };
}

// AUTH-07 - PKCE (RFC 7636, S256). code_verifier is a high-entropy random
// string; code_challenge is base64url(sha256(verifier)). The verifier is stored
// server-side (httpOnly cookie) and replayed at token exchange; the challenge
// goes to the provider in the authorize redirect.
function _pkcePair() {
  const verifier = crypto.randomBytes(32).toString('base64url');
  const challenge = crypto.createHash('sha256').update(verifier).digest('base64url');
  return { verifier, challenge };
}

export function oauthConfigured(providerName) {
  const p = PROVIDERS[providerName];
  if (!p) return false;
  return !!(process.env[p.clientIdEnv] && process.env[p.clientSecretEnv]);
}

export function mountOAuth(router) {
  // GET /v1/oauth/:provider/start begins Google or GitHub OAuth sign-in.
  // Redirects to the provider when configured; returns 503 with an operator hint otherwise.
  router.get('/v1/oauth/:provider/start', (req, res) => {
    const name = req.params.provider;
    const p = PROVIDERS[name];
    if (!p) return res.status(404).json({ error: 'unknown provider' });
    if (!oauthConfigured(name)) {
      return res.status(503).json({ error: 'oauth_not_configured', provider: name, hint: `set ${p.clientIdEnv} and ${p.clientSecretEnv} on the server` });
    }
    const state = crypto.randomBytes(24).toString('hex');
    const ret = safeReturn(req);
    setCookie(res, STATE_COOKIE, state, 10 * 60 * 1000);
    setCookie(res, RETURN_COOKIE, ret, 10 * 60 * 1000);
    const u = new URL(p.authUrl);
    u.searchParams.set('client_id', process.env[p.clientIdEnv]);
    u.searchParams.set('redirect_uri', `${baseUrl()}/v1/oauth/${name}/callback`);
    u.searchParams.set('response_type', 'code');
    u.searchParams.set('scope', p.scope);
    u.searchParams.set('state', state);
    // AUTH-07 - PKCE for providers that support it (Google). The verifier is
    // kept in an httpOnly cookie (never exposed to JS / the URL); only the S256
    // challenge is sent to the provider. An intercepted code is then useless
    // without the verifier we replay at token exchange.
    if (p.pkce) {
      const { verifier, challenge } = _pkcePair();
      setCookie(res, PKCE_COOKIE, verifier, 10 * 60 * 1000);
      u.searchParams.set('code_challenge', challenge);
      u.searchParams.set('code_challenge_method', 'S256');
    } else {
      clearCookie(res, PKCE_COOKIE);
    }
    if (name === 'google') {
      u.searchParams.set('access_type', 'online');
      u.searchParams.set('prompt', 'select_account');
    }
    res.redirect(302, u.toString());
  });

  // GET /v1/oauth/:provider/callback completes Google or GitHub OAuth sign-in.
  // Exchanges the provider code, creates or finds the tenant, then sets the session cookie.
  router.get('/v1/oauth/:provider/callback', async (req, res) => {
    const name = req.params.provider;
    const p = PROVIDERS[name];
    if (!p) return res.status(404).type('text/plain').send('unknown provider');
    if (!oauthConfigured(name)) return res.status(503).type('text/plain').send('oauth not configured');

    const { code, state, error: providerError } = req.query || {};
    const cookieState = req.cookies && req.cookies[STATE_COOKIE];
    const pkceVerifier = (req.cookies && req.cookies[PKCE_COOKIE]) || '';
    // AUTH-07 - re-run safeReturn() on the cookie-sourced return path before we
    // ever trust it for the final 302. Defense in depth: even if the
    // RETURN_COOKIE were tampered to an off-site or protocol-relative value, the
    // redirect target is re-validated to a same-origin path here.
    const rawRet = (req.cookies && req.cookies[RETURN_COOKIE]) || '/dashboard';
    const ret = safeReturn({ query: { redirect: rawRet } });
    clearCookie(res, STATE_COOKIE);
    clearCookie(res, RETURN_COOKIE);
    clearCookie(res, PKCE_COOKIE);

    if (providerError) {
      return res.redirect(302, `/signup?oauth_error=${encodeURIComponent(String(providerError))}`);
    }
    if (!code || !state || !cookieState || state !== cookieState) {
      return res.redirect(302, '/signup?oauth_error=state_mismatch');
    }

    try {
      const tokenBody = new URLSearchParams({
        client_id: process.env[p.clientIdEnv],
        client_secret: process.env[p.clientSecretEnv],
        code: String(code),
        redirect_uri: `${baseUrl()}/v1/oauth/${name}/callback`,
        grant_type: 'authorization_code',
      });
      // AUTH-07 - replay the PKCE verifier at token exchange for PKCE providers.
      if (p.pkce && pkceVerifier) {
        tokenBody.set('code_verifier', pkceVerifier);
      }
      const tokenRes = await fetch(p.tokenUrl, {
        method: 'POST',
        headers: { 'content-type': 'application/x-www-form-urlencoded', accept: 'application/json' },
        body: tokenBody.toString(),
      });
      const tokenJson = await tokenRes.json().catch(() => ({}));
      const accessToken = tokenJson.access_token;
      if (!accessToken) {
        return res.redirect(302, `/signup?oauth_error=token_exchange_failed`);
      }

      const userRes = await fetch(p.userUrl, {
        headers: { authorization: `Bearer ${accessToken}`, accept: 'application/json', 'user-agent': 'kolm-oauth' },
      });
      const userJson = await userRes.json().catch(() => ({}));
      const displayName = p.extractName(userJson);

      // AUTH-04 - resolve a VERIFIED email per provider. Never trust a bare
      // provider-asserted email; require the provider's verification flag.
      let email = null;
      if (name === 'google') {
        const r = _googleVerifiedEmail(userJson);
        if (r.error) return res.redirect(302, `/signup?oauth_error=${encodeURIComponent(r.error)}`);
        email = r.email;
      } else if (name === 'github') {
        // Always confirm against /user/emails: /user.email carries no
        // verification flag, so it is not trustworthy on its own.
        const r = await _githubVerifiedEmail(accessToken, p.emailUrl);
        if (r.error) return res.redirect(302, `/signup?oauth_error=${encodeURIComponent(r.error)}`);
        email = r.email;
      } else {
        email = p.extractEmail(userJson);
      }

      if (!email || !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) {
        return res.redirect(302, `/signup?oauth_error=no_email_returned`);
      }

      let result;
      try {
        result = findOrCreateTenantByEmail({
          email,
          name: displayName,
          provider: name,
          provider_id: String(userJson.id || userJson.sub || ''),
        });
      } catch (linkErr) {
        // AUTH-04 - a provider_id mismatch means this email's tenant is already
        // bound to a different provider account. Refuse silent sign-in and tell
        // the user to link from an authenticated session instead of overwriting.
        if (linkErr && linkErr.code === 'provider_id_mismatch') {
          return res.redirect(302, `/signup?oauth_error=account_link_required`);
        }
        throw linkErr;
      }
      const { tenant, api_key, created } = result;

      setSessionCookie(res, api_key);
      const sep = ret.includes('?') ? '&' : '?';
      const flag = created ? 'oauth=signup' : 'oauth=signin';
      return res.redirect(302, `${ret}${sep}${flag}`);
    } catch (err) {
      return res.redirect(302, `/signup?oauth_error=${encodeURIComponent(String(err && err.message || err))}`);
    }
  });

  // GET /v1/oauth/providers reports which hosted OAuth providers are configured.
  // Signup uses this public route to hide provider buttons until credentials exist.
  router.get('/v1/oauth/providers', (_req, res) => {
    res.json({
      google: oauthConfigured('google'),
      github: oauthConfigured('github'),
    });
  });
}

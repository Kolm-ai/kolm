// kolm code — GitHub App authentication (owner F, spec sections 9 + 11.2).
//
// Resolves a token that can push a branch and open a pull request. Truffle's
// own App is a read + check-run + merge gateway (no contents:write), so the
// App installation token is only one of four sources; the resolution order is
// fixed by spec 11.2:
//
//   KOLM_GITHUB_TOKEN -> GITHUB_TOKEN -> GitHub App installation token -> gh auth token
//
// Rules that this module never breaks:
//   * no npm dependencies (node:crypto only),
//   * key material never reaches a log line, an Error message or an Error
//     detail — not the PEM, not the signed JWT, not the installation token,
//   * every export takes its inputs through an options object so callers and
//     tests never have to construct the real config.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';

import { LadderError } from '../errors.js';

/** Truffle's PR-authority key, relative to the user's home directory. */
export const DEFAULT_APP_KEY_RELATIVE_PATH = '.truffle/truffle-pr-authority.private-key.pem';

export const GITHUB_API_BASE = 'https://api.github.com';
export const GITHUB_API_VERSION = '2022-11-28';
export const USER_AGENT = 'kolm-code';

/** Default installation-token JWT lifetime (GitHub caps App JWTs at 10 min). */
export const DEFAULT_JWT_TTL_SECONDS = 540;

/** Refresh an installation token this long before GitHub expires it. */
export const TOKEN_REFRESH_MARGIN_MS = 60_000;

const PEM_MARKER = /-----BEGIN [A-Z ]*PRIVATE KEY-----/;

/** Module-level installation-token cache keyed by `${appId}\u0000${installationId}`. */
const tokenCache = new Map();

/** Drops every cached installation token. Tests call this between cases. */
export function _clearTokenCache() {
  tokenCache.clear();
}

function nowMs(now) {
  if (typeof now === 'function') return now();
  if (typeof now === 'number') return now;
  return Date.now();
}

function trimmed(value) {
  return typeof value === 'string' && value.trim() !== '' ? value.trim() : null;
}

function base64url(input) {
  return Buffer.from(input).toString('base64url');
}

/**
 * Removes anything that looks like credential material from a string that is
 * about to be attached to an error. Cheap belt-and-braces: the callers already
 * avoid passing secrets, this makes an accidental echo (a server that repeats
 * the Authorization header back in its error body) non-fatal.
 * @param {string} text
 * @param {string[]} [secrets] exact values to blank out
 * @returns {string}
 */
export function redactSecrets(text, secrets = []) {
  let out = typeof text === 'string' ? text : String(text ?? '');
  for (const secret of secrets) {
    if (typeof secret === 'string' && secret.length >= 8) {
      out = out.split(secret).join('[redacted]');
    }
  }
  return out
    .replace(/-----BEGIN[\s\S]*?-----END[^-]*-----/g, '[redacted pem]')
    .replace(/gh[pousr]_[A-Za-z0-9]{16,}/g, '[redacted token]')
    .replace(/github_pat_[A-Za-z0-9_]{20,}/g, '[redacted token]')
    .replace(/\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b/g, '[redacted jwt]');
}

function unescapePem(value) {
  // Truffle stores the App key as an inline env var whose newlines are the two
  // characters `\` `n` (src/lib/github/runtime.ts:154-157 does the same swap).
  return !value.includes('\n') && value.includes('\\n') ? value.replaceAll('\\n', '\n') : value;
}

function assertPem(pem, source) {
  if (typeof pem !== 'string' || !PEM_MARKER.test(pem)) {
    throw new LadderError(
      'github-app-key',
      `GitHub App private key from ${source} is not a PEM private key`,
      { source },
    );
  }
  return pem;
}

/**
 * Resolves the GitHub App private key without ever returning its origin bytes
 * in an error. Order (spec 11.2 + owner F brief):
 *   KOLM_GITHUB_APP_PRIVATE_KEY (inline) -> TRUFFLE_GITHUB_PRIVATE_KEY (inline)
 *   -> KOLM_GITHUB_APP_PRIVATE_KEY_PATH -> TRUFFLE_GITHUB_PRIVATE_KEY_PATH
 *   -> ~/.truffle/truffle-pr-authority.private-key.pem when that file exists.
 * @param {object} [opts]
 * @param {NodeJS.ProcessEnv} [opts.env]
 * @param {(p: string, enc: string) => string} [opts.readFile]
 * @param {(p: string) => boolean} [opts.exists]
 * @param {string} [opts.home]
 * @returns {{ pem: string, source: string, path: string|null }|null}
 */
export function resolvePrivateKeySource({
  env = process.env,
  readFile = fs.readFileSync,
  exists = fs.existsSync,
  home = os.homedir(),
} = {}) {
  const inlineKolm = trimmed(env.KOLM_GITHUB_APP_PRIVATE_KEY);
  if (inlineKolm) {
    return {
      pem: assertPem(unescapePem(inlineKolm), 'KOLM_GITHUB_APP_PRIVATE_KEY'),
      source: 'KOLM_GITHUB_APP_PRIVATE_KEY',
      path: null,
    };
  }
  const inlineTruffle = trimmed(env.TRUFFLE_GITHUB_PRIVATE_KEY);
  if (inlineTruffle) {
    return {
      pem: assertPem(unescapePem(inlineTruffle), 'TRUFFLE_GITHUB_PRIVATE_KEY'),
      source: 'TRUFFLE_GITHUB_PRIVATE_KEY',
      path: null,
    };
  }

  const candidates = [
    ['KOLM_GITHUB_APP_PRIVATE_KEY_PATH', trimmed(env.KOLM_GITHUB_APP_PRIVATE_KEY_PATH)],
    ['TRUFFLE_GITHUB_PRIVATE_KEY_PATH', trimmed(env.TRUFFLE_GITHUB_PRIVATE_KEY_PATH)],
  ];
  for (const [source, raw] of candidates) {
    if (!raw) continue;
    const file = expandHome(raw, home);
    return { pem: assertPem(readPem(readFile, file, source), source), source, path: file };
  }

  const fallback = path.join(home, ...DEFAULT_APP_KEY_RELATIVE_PATH.split('/'));
  if (exists(fallback)) {
    return {
      pem: assertPem(readPem(readFile, fallback, 'default-truffle-path'), 'default-truffle-path'),
      source: 'default-truffle-path',
      path: fallback,
    };
  }
  return null;
}

function expandHome(value, home) {
  if (value === '~') return home;
  if (value.startsWith('~/') || value.startsWith('~\\')) return path.join(home, value.slice(2));
  return path.resolve(value);
}

function readPem(readFile, file, source) {
  let raw;
  try {
    raw = readFile(file, 'utf8');
  } catch (err) {
    throw new LadderError(
      'github-app-key',
      `GitHub App private key at ${file} could not be read`,
      { source, path: file, cause: err?.code || err?.name || 'read-failed' },
    );
  }
  return unescapePem(String(raw));
}

/**
 * The PEM only, or null when no source is configured.
 * @param {object} [opts] see {@link resolvePrivateKeySource}
 * @returns {string|null}
 */
export function loadPrivateKey(opts = {}) {
  const resolved = resolvePrivateKeySource(opts);
  return resolved ? resolved.pem : null;
}

/**
 * Builds the RS256 App JWT GitHub wants on `POST /app/installations/:id/access_tokens`.
 * @param {object} opts
 * @param {string|number} opts.appId
 * @param {string} opts.privateKey PEM
 * @param {(() => number)|number} [opts.now]
 * @param {number} [opts.ttlSeconds]
 * @returns {string} compact JWS
 */
export function buildAppJwt({ appId, privateKey, now = Date.now, ttlSeconds = DEFAULT_JWT_TTL_SECONDS }) {
  if (!appId) throw new LadderError('github-app-auth', 'GitHub App id is required to sign a JWT');
  if (typeof privateKey !== 'string' || privateKey === '') {
    throw new LadderError('github-app-auth', 'GitHub App private key is required to sign a JWT');
  }
  const seconds = Math.floor(nowMs(now) / 1000);
  const header = base64url(JSON.stringify({ alg: 'RS256', typ: 'JWT' }));
  const claims = base64url(JSON.stringify({
    iat: seconds - 60,
    exp: seconds + ttlSeconds,
    iss: String(appId),
  }));
  const signingInput = `${header}.${claims}`;
  let signature;
  try {
    signature = crypto.createSign('RSA-SHA256').update(signingInput).sign(privateKey);
  } catch (err) {
    // err.message can quote the offending PEM; only the code ever escapes.
    throw new LadderError('github-app-auth', 'GitHub App JWT signing failed', {
      reason: err?.code || err?.name || 'sign-failed',
    });
  }
  return `${signingInput}.${signature.toString('base64url')}`;
}

async function readBodyExcerpt(response, secrets) {
  let text = '';
  try {
    text = typeof response.text === 'function' ? await response.text() : '';
  } catch {
    text = '';
  }
  return redactSecrets(String(text).slice(0, 300), secrets);
}

/**
 * Exchanges an App JWT for an installation token, cached per
 * (appId, installationId) until 60 s before expiry.
 * @param {object} opts
 * @returns {Promise<{token: string, expiresAt: string, permissions: object}>}
 */
export async function getInstallationToken({
  appId,
  installationId,
  privateKey,
  fetchImpl = fetch,
  now = Date.now,
  apiBase = GITHUB_API_BASE,
  ttlSeconds = DEFAULT_JWT_TTL_SECONDS,
}) {
  if (!appId || !installationId) {
    throw new LadderError(
      'github-app-auth',
      'GitHub App id and installation id are both required',
      { hasAppId: Boolean(appId), hasInstallationId: Boolean(installationId) },
    );
  }
  const key = `${appId}\u0000${installationId}`;
  const at = nowMs(now);
  const cached = tokenCache.get(key);
  if (cached && cached.expiresAtMs - TOKEN_REFRESH_MARGIN_MS > at) {
    return { token: cached.token, expiresAt: cached.expiresAt, permissions: cached.permissions };
  }

  const jwt = buildAppJwt({ appId, privateKey, now: at, ttlSeconds });
  const url = `${String(apiBase).replace(/\/+$/, '')}/app/installations/${encodeURIComponent(String(installationId))}/access_tokens`;
  let response;
  try {
    response = await fetchImpl(url, {
      method: 'POST',
      headers: {
        authorization: `Bearer ${jwt}`,
        accept: 'application/vnd.github+json',
        'x-github-api-version': GITHUB_API_VERSION,
        'user-agent': USER_AGENT,
        'content-type': 'application/json',
      },
      body: '{}',
    });
  } catch (err) {
    throw new LadderError('github-app-auth', 'GitHub App token request failed', {
      reason: redactSecrets(String(err?.message || err?.code || 'network-error'), [jwt]),
    });
  }

  const status = Number(response?.status ?? 0);
  if (!response?.ok) {
    throw new LadderError('github-app-auth', `GitHub App token request failed: ${status}`, {
      status,
      body: await readBodyExcerpt(response, [jwt]),
    });
  }

  let payload;
  try {
    payload = JSON.parse(await response.text());
  } catch {
    throw new LadderError('github-app-auth', 'GitHub App token response was not JSON', { status });
  }
  const token = typeof payload?.token === 'string' ? payload.token : null;
  if (!token) {
    throw new LadderError('github-app-auth', 'GitHub App token response carried no token', { status });
  }
  const expiresAt = typeof payload.expires_at === 'string' ? payload.expires_at : null;
  const expiresAtMs = expiresAt ? Date.parse(expiresAt) : NaN;
  const entry = {
    token,
    expiresAt,
    expiresAtMs: Number.isFinite(expiresAtMs) ? expiresAtMs : at + 3_600_000,
    permissions: payload.permissions ?? {},
  };
  tokenCache.set(key, entry);
  return { token: entry.token, expiresAt: entry.expiresAt, permissions: entry.permissions };
}

function defaultExec(file, args) {
  return execFileSync(file, args, {
    encoding: 'utf8',
    timeout: 5000,
    stdio: ['ignore', 'pipe', 'ignore'],
  });
}

function ghCliToken(execImpl) {
  const run = typeof execImpl === 'function' ? execImpl : defaultExec;
  // execFileSync resolves `gh` through PATHEXT on Windows; `gh.cmd` is the
  // fallback for shims that PATHEXT does not cover.
  for (const file of ['gh', 'gh.cmd']) {
    try {
      const out = run(file, ['auth', 'token']);
      const token = trimmed(typeof out === 'string' ? out : String(out ?? ''));
      if (token) return token;
    } catch {
      // try the next spelling
    }
  }
  return null;
}

function appTriple(env) {
  return {
    appId: trimmed(env.KOLM_GITHUB_APP_ID) || trimmed(env.TRUFFLE_GITHUB_APP_ID),
    installationId:
      trimmed(env.KOLM_GITHUB_APP_INSTALLATION_ID) || trimmed(env.TRUFFLE_GITHUB_INSTALLATION_ID),
  };
}

/**
 * Thin adapter over owner A's config object: it already resolved the token and
 * the private key source, so this projects those values back onto the env
 * shape every function here takes. Values stay in memory; nothing is logged.
 * @param {object} config from `loadConfig()` in `../config.js`
 * @returns {NodeJS.ProcessEnv} a synthetic env with only the GitHub keys set
 */
export function githubEnvFromConfig(config) {
  const env = {};
  if (!config || typeof config !== 'object') return env;
  const gh = config.github ?? {};

  let token = null;
  if (typeof config.getSecret === 'function') {
    try {
      token = config.getSecret('githubToken');
    } catch {
      token = null;
    }
  }
  if (token) {
    if (gh.tokenSource === 'GITHUB_TOKEN') env.GITHUB_TOKEN = token;
    else env.KOLM_GITHUB_TOKEN = token;
  }

  if (gh.appId) env.KOLM_GITHUB_APP_ID = String(gh.appId);
  if (gh.installationId) env.KOLM_GITHUB_APP_INSTALLATION_ID = String(gh.installationId);

  if (typeof config.githubPrivateKeyPem === 'function') {
    let pem = null;
    try {
      pem = config.githubPrivateKeyPem();
    } catch {
      pem = null;
    }
    if (pem) env.KOLM_GITHUB_APP_PRIVATE_KEY = pem;
  } else if (gh.privateKeyPath) {
    env.KOLM_GITHUB_APP_PRIVATE_KEY_PATH = gh.privateKeyPath;
  }
  return env;
}

/**
 * Resolves the push/PR token. Never throws for a missing source; an App
 * exchange that fails is recorded in `reasons` and the next source is tried.
 * Pass either `env` (the injectable default) or owner A's `config`.
 * @param {object} [opts]
 * @returns {Promise<{token: string, source: string, reasons: string[], expiresAt?: string}|null>}
 */
export async function resolveGitHubToken({
  config = null,
  env = config ? githubEnvFromConfig(config) : process.env,
  fetchImpl = fetch,
  execImpl,
  readFile = fs.readFileSync,
  exists = fs.existsSync,
  home = os.homedir(),
  now = Date.now,
  apiBase = GITHUB_API_BASE,
} = {}) {
  const reasons = [];

  const direct = trimmed(env.KOLM_GITHUB_TOKEN);
  if (direct) return { token: direct, source: 'KOLM_GITHUB_TOKEN', reasons };

  const fallback = trimmed(env.GITHUB_TOKEN);
  if (fallback) return { token: fallback, source: 'GITHUB_TOKEN', reasons };

  const { appId, installationId } = appTriple(env);
  if (appId && installationId) {
    let privateKey = null;
    try {
      privateKey = loadPrivateKey({ env, readFile, exists, home });
      if (!privateKey) reasons.push('github-app: no private key configured');
    } catch (err) {
      reasons.push(`github-app: ${err?.message || 'private key unavailable'}`);
    }
    if (privateKey) {
      try {
        const issued = await getInstallationToken({
          appId,
          installationId,
          privateKey,
          fetchImpl,
          now,
          apiBase,
        });
        return {
          token: issued.token,
          source: 'github-app',
          reasons,
          expiresAt: issued.expiresAt,
          permissions: issued.permissions,
        };
      } catch (err) {
        reasons.push(`github-app: ${err?.message || 'installation token exchange failed'}`);
      }
    }
  } else {
    reasons.push('github-app: app id and installation id are not both set');
  }

  const cli = ghCliToken(execImpl);
  if (cli) return { token: cli, source: 'gh-cli', reasons };
  reasons.push('gh-cli: `gh auth token` produced no token');

  return null;
}

/**
 * Pure arming report for `kolm code doctor`: no network, no child processes.
 * gh-cli availability is deliberately reported as unknown via the ghCli field;
 * sources lists only credential sources that are actually present.
 * @param {object} [opts]
 * @returns {{armed: boolean, sources: string[], missing: string[], ghCli: 'unknown', privateKeySource: string|null}}
 */
export function describeGitHubArming({
  config = null,
  env = config ? githubEnvFromConfig(config) : process.env,
  readFile = fs.readFileSync,
  exists = fs.existsSync,
  home = os.homedir(),
} = {}) {
  const sources = [];
  const missing = [];

  const direct = trimmed(env.KOLM_GITHUB_TOKEN);
  const fallback = trimmed(env.GITHUB_TOKEN);
  if (direct) sources.push('KOLM_GITHUB_TOKEN');
  else missing.push('KOLM_GITHUB_TOKEN');
  if (fallback) sources.push('GITHUB_TOKEN');
  else missing.push('GITHUB_TOKEN');

  const { appId, installationId } = appTriple(env);
  if (!appId) missing.push('KOLM_GITHUB_APP_ID');
  if (!installationId) missing.push('KOLM_GITHUB_APP_INSTALLATION_ID');

  let privateKeySource = null;
  try {
    const resolved = resolvePrivateKeySource({ env, readFile, exists, home });
    privateKeySource = resolved ? resolved.source : null;
  } catch (err) {
    privateKeySource = null;
    missing.push(`KOLM_GITHUB_APP_PRIVATE_KEY_PATH (${err?.code || 'unreadable'})`);
  }
  if (!privateKeySource) missing.push('KOLM_GITHUB_APP_PRIVATE_KEY_PATH');

  if (appId && installationId && privateKeySource) sources.push('github-app');

  return {
    armed: Boolean(direct || fallback || (appId && installationId && privateKeySource)),
    sources,
    missing,
    ghCli: 'unknown',
    privateKeySource,
  };
}


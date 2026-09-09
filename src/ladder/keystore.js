// kolm code — bring-your-own-key keystore (spec §2 addendum).
//
// Provider keys used to come only from environment variables, which meant
// editing a shell profile before `kolm code` could do anything. This module
// adds a small JSON file that `kolm code keys set` writes and every verb reads:
//
//   <KOLM_DATA_DIR or ~/.kolm>/code/keys.json
//
//   { "version": 1,
//     "keys": {
//       "openrouter": { "value": "...", "set_at": "<iso>", "fingerprint": "sha256:<12 hex>" }
//     } }
//
// Discipline, identical to config.js: a secret value is never an enumerable
// own property of anything this module returns. `JSON.stringify(readKeystore(env))`
// is safe to print, log and paste into an issue. Only `.secrets` (non-enumerable)
// and the values handed to `setKey` carry the plaintext.
//
// This module owns the filesystem half; `loadConfig(env, { fileSecrets })` stays
// pure and is handed the secrets object by the caller (the CLI).

import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { LadderError } from './errors.js';
import { FILE_SECRET_NAMES, expandHome } from './config.js';

/** Keystore schema version written by `setKey`. */
export const KEYSTORE_VERSION = 1;

/** The exact names a keystore entry may carry. Canonical list lives in config.js. */
export const KEY_NAMES = FILE_SECRET_NAMES;

/**
 * Environment variables that supply the same secret. The first name is the one
 * the operator guide leads with; every name in the list wins over the keystore.
 */
export const KEY_ENV_NAMES = Object.freeze({
  anthropic: Object.freeze(['ANTHROPIC_API_KEY']),
  cerebras: Object.freeze(['CEREBRAS_API_KEY', 'KOLM_CEREBRAS_TOKEN']),
  deepseek: Object.freeze(['DEEPSEEK_API_KEY']),
  openrouter: Object.freeze(['OPENROUTER_API_KEY']),
  github: Object.freeze(['KOLM_GITHUB_TOKEN', 'GITHUB_TOKEN']),
  sandbox: Object.freeze(['KOLM_CODE_SANDBOX_TOKEN']),
});

// ---------------------------------------------------------------------------
// env helpers (a deliberate copy of the config.js reader: the keystore must
// resolve its own path even when some unrelated variable would fail loadConfig)
// ---------------------------------------------------------------------------

function str(env, key, fallback = null) {
  const raw = env?.[key];
  if (raw === undefined || raw === null) return fallback;
  const value = String(raw).replace(/^\uFEFF+/, '').trim();
  return value === '' ? fallback : value;
}

function resolveHome(env) {
  return (
    str(env, 'KOLM_CODE_HOME', null) ||
    str(env, 'HOME', null) ||
    str(env, 'USERPROFILE', null) ||
    os.homedir()
  );
}

/**
 * True when `KOLM_CODE_NO_KEYSTORE` is set to a truthy flag. CI and the Truffle
 * worker set it so a developer machine's keys can never leak into a shared run.
 * @param {Record<string,string|undefined>} [env]
 * @returns {boolean}
 */
export function keystoreDisabled(env = process.env) {
  const raw = str(env, 'KOLM_CODE_NO_KEYSTORE', null);
  if (raw === null) return false;
  return ['1', 'true', 'yes', 'on'].includes(raw.toLowerCase());
}

/**
 * Absolute path of the keystore file. `KOLM_CODE_KEYS_PATH` overrides it
 * (`~` expands); otherwise it sits next to the ladder store in `codeDir`.
 * @param {Record<string,string|undefined>} [env]
 * @returns {string}
 */
export function keystorePath(env = process.env) {
  const home = resolveHome(env);
  const override = str(env, 'KOLM_CODE_KEYS_PATH', null);
  if (override) return path.resolve(expandHome(override, home));
  const dataDir = path.resolve(expandHome(str(env, 'KOLM_DATA_DIR', '~/.kolm'), home));
  return path.join(dataDir, 'code', 'keys.json');
}

// ---------------------------------------------------------------------------
// validation
// ---------------------------------------------------------------------------

/**
 * @param {string} name
 * @returns {string} the validated name
 */
export function assertKeyName(name) {
  const value = typeof name === 'string' ? name.trim().toLowerCase() : '';
  if (!KEY_NAMES.includes(value)) {
    throw new LadderError(
      'bad-args',
      `unknown key name "${name}"; supported names are ${KEY_NAMES.join(', ')}`,
      { name, supported: [...KEY_NAMES] },
    );
  }
  return value;
}

/**
 * Trim the value and refuse anything that cannot be sent as a header: an empty
 * string, or a value with whitespace or a newline inside it (the usual symptom
 * of a copied line wrap or a shell quoting slip).
 * @param {string} name
 * @param {string} value
 * @returns {string}
 */
export function assertKeyValue(name, value) {
  if (typeof value !== 'string') {
    throw new LadderError('bad-args', `the ${name} key must be a string, got ${typeof value}`, {
      name,
    });
  }
  const trimmed = value.replace(/^\uFEFF+/, '').trim();
  if (trimmed === '') {
    throw new LadderError('bad-args', `the ${name} key is empty; nothing was written`, { name });
  }
  if (/\s/.test(trimmed)) {
    throw new LadderError(
      'bad-args',
      `the ${name} key contains whitespace or a line break; copy it again without wrapping`,
      { name },
    );
  }
  return trimmed;
}

/**
 * Short, non-reversible identity for a secret: the first 12 hex characters of
 * its SHA-256. Safe to print, store and compare across machines.
 * @param {string} value
 * @returns {string} `sha256:<12 hex>`
 */
export function fingerprint(value) {
  const material = typeof value === 'string' ? value : String(value ?? '');
  return `sha256:${crypto.createHash('sha256').update(material, 'utf8').digest('hex').slice(0, 12)}`;
}

// ---------------------------------------------------------------------------
// file io
// ---------------------------------------------------------------------------

function readDocument(file) {
  let text;
  try {
    text = fs.readFileSync(file, 'utf8');
  } catch (err) {
    if (err && err.code === 'ENOENT') return null; // a missing keystore is not an error
    throw new LadderError('bad-config', `keystore at ${file} could not be read (${err?.code ?? err?.message})`, {
      path: file,
      cause: err?.code ?? null,
    });
  }
  let parsed;
  try {
    parsed = JSON.parse(text.replace(/^\uFEFF/, ''));
  } catch {
    throw new LadderError(
      'bad-config',
      `keystore at ${file} is not valid JSON; fix it or delete it and set the keys again`,
      { path: file },
    );
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new LadderError('bad-config', `keystore at ${file} must be a JSON object`, { path: file });
  }
  const keys = parsed.keys ?? {};
  if (!keys || typeof keys !== 'object' || Array.isArray(keys)) {
    throw new LadderError('bad-config', `keystore at ${file} must carry an object at .keys`, {
      path: file,
    });
  }
  return { version: Number(parsed.version) || KEYSTORE_VERSION, keys };
}

/**
 * Temp file + rename, so a reader never sees a half-written keystore, and 0600
 * so a shared POSIX box cannot read it. Windows ignores the mode; the ACL of
 * the containing profile directory is what protects the file there.
 * @param {string} file
 * @param {object} document
 */
/**
 * Replace `file` with `tmp`. On Windows a scanner or indexer can hold the
 * target open for a moment and the rename reports EPERM or EBUSY; a few short
 * retries turn that race into a successful write (seen once in the test suite).
 * @param {string} tmp
 * @param {string} file
 */
function renameWithRetry(tmp, file) {
  const RETRIES = 5;
  for (let attempt = 0; ; attempt += 1) {
    try {
      fs.renameSync(tmp, file);
      return;
    } catch (err) {
      const transient = err?.code === 'EPERM' || err?.code === 'EBUSY';
      if (!transient || attempt >= RETRIES) throw err;
      const until = Date.now() + 20 * (attempt + 1);
      while (Date.now() < until) { /* brief synchronous wait */ }
    }
  }
}

function writeDocument(file, document) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.tmp.${process.pid}.${crypto.randomBytes(4).toString('hex')}`;
  const body = `${JSON.stringify(document, null, 2)}\n`;
  try {
    fs.writeFileSync(tmp, body, { encoding: 'utf8', mode: 0o600 });
    try {
      fs.chmodSync(tmp, 0o600);
    } catch {
      // deliberate: chmod is a no-op on Windows, the write already succeeded
    }
    renameWithRetry(tmp, file);
  } catch (err) {
    try {
      fs.rmSync(tmp, { force: true });
    } catch {
      // deliberate: cleanup only
    }
    throw new LadderError('bad-config', `keystore at ${file} could not be written (${err?.code ?? err?.message})`, {
      path: file,
      cause: err?.code ?? null,
    });
  }
}

// ---------------------------------------------------------------------------
// read / write API
// ---------------------------------------------------------------------------

/**
 * Read the keystore.
 *
 * Entry names this build does not know are ignored rather than fatal, so a file
 * written by a newer kolm never bricks an older one.
 *
 * @param {Record<string,string|undefined>} [env]
 * @returns {{path: string, enabled: boolean, keys: Record<string,{fingerprint:string,set_at:string|null}>}}
 *   plus a NON-enumerable `secrets` map of name -> plaintext value.
 */
export function readKeystore(env = process.env) {
  const file = keystorePath(env);
  const enabled = !keystoreDisabled(env);
  const keys = {};
  const secrets = {};

  if (enabled) {
    const document = readDocument(file);
    for (const [rawName, entry] of Object.entries(document?.keys ?? {})) {
      const name = String(rawName).trim().toLowerCase();
      if (!KEY_NAMES.includes(name)) continue;
      if (!entry || typeof entry !== 'object' || typeof entry.value !== 'string') {
        throw new LadderError(
          'bad-config',
          `keystore at ${file} has a malformed entry for "${name}"; delete it and set the key again`,
          { path: file, name },
        );
      }
      const value = entry.value.trim();
      if (value === '') continue; // an emptied entry is the same as an absent one
      keys[name] = {
        fingerprint: typeof entry.fingerprint === 'string' && entry.fingerprint
          ? entry.fingerprint
          : fingerprint(value),
        set_at: typeof entry.set_at === 'string' ? entry.set_at : null,
      };
      secrets[name] = value;
    }
  }

  const result = { path: file, enabled, keys };
  Object.defineProperty(result, 'secrets', {
    value: secrets,
    enumerable: false,
    writable: false,
    configurable: false,
  });
  return result;
}

/**
 * Write one key. Other entries in the file are preserved untouched.
 * @param {Record<string,string|undefined>} env
 * @param {string} name
 * @param {string} value
 * @returns {{path: string, name: string, fingerprint: string, set_at: string}}
 */
export function setKey(env, name, value) {
  const key = assertKeyName(name);
  const secret = assertKeyValue(key, value);
  const file = keystorePath(env);
  const document = readDocument(file) ?? { version: KEYSTORE_VERSION, keys: {} };
  const setAt = new Date().toISOString();
  const next = {
    version: KEYSTORE_VERSION,
    keys: { ...document.keys, [key]: { value: secret, set_at: setAt, fingerprint: fingerprint(secret) } },
  };
  writeDocument(file, next);
  return { path: file, name: key, fingerprint: fingerprint(secret), set_at: setAt };
}

/**
 * Remove one key. Removing a key that was never set is not an error.
 * @param {Record<string,string|undefined>} env
 * @param {string} name
 * @returns {{path: string, name: string, removed: boolean}}
 */
export function unsetKey(env, name) {
  const key = assertKeyName(name);
  const file = keystorePath(env);
  const document = readDocument(file);
  if (!document || !Object.prototype.hasOwnProperty.call(document.keys, key)) {
    return { path: file, name: key, removed: false };
  }
  const keys = { ...document.keys };
  delete keys[key];
  writeDocument(file, { version: KEYSTORE_VERSION, keys });
  return { path: file, name: key, removed: true };
}

/**
 * One row per supported name: where the secret would come from, and the
 * fingerprint of the value that actually wins. Never carries a value.
 * @param {Record<string,string|undefined>} [env]
 * @returns {{path: string, enabled: boolean, rows: Array<{name:string,source:string,fingerprint:string|null,set_at:string|null,envNames:string[]}>}}
 */
export function describeKeys(env = process.env) {
  const store = readKeystore(env);
  const rows = KEY_NAMES.map((name) => {
    const envNames = KEY_ENV_NAMES[name] ?? [];
    const fromEnv = envNames.map((varName) => str(env, varName, null)).find(Boolean) ?? null;
    const inStore = store.keys[name] ?? null;
    let source = 'none';
    if (fromEnv && inStore) source = 'env+keystore(env wins)';
    else if (fromEnv) source = 'env';
    else if (inStore) source = 'keystore';
    const winner = fromEnv ?? store.secrets[name] ?? null;
    return {
      name,
      source,
      fingerprint: winner ? fingerprint(winner) : null,
      set_at: inStore ? inStore.set_at : null,
      envNames: [...envNames],
    };
  });
  return { path: store.path, enabled: store.enabled, rows };
}

// ---------------------------------------------------------------------------
// live probe
// ---------------------------------------------------------------------------

/** The cheapest authenticated GET each provider answers. */
function probeRequestFor(name, secret, config) {
  const base = (url) => String(url ?? '').replace(/\/+$/, '');
  switch (name) {
    case 'anthropic':
      return {
        url: `${base(config?.baseUrls?.anthropic ?? 'https://api.anthropic.com/v1')}/models`,
        headers: { 'x-api-key': secret, 'anthropic-version': '2023-06-01' },
      };
    case 'cerebras':
    case 'deepseek':
    case 'openrouter': {
      const url = base(config?.baseUrls?.[name]);
      if (!url) return { skip: `no base url configured for ${name}` };
      return { url: `${url}/models`, headers: { authorization: `Bearer ${secret}` } };
    }
    case 'github':
      return { url: 'https://api.github.com/user', headers: { authorization: `Bearer ${secret}`, accept: 'application/vnd.github+json' } };
    case 'sandbox': {
      const url = base(config?.sandboxUrl);
      if (!url) return { skip: 'sandbox url not configured' };
      return { url: `${url}/health`, headers: { authorization: `Bearer ${secret}` } };
    }
    default:
      return { skip: `no probe for ${name}` };
  }
}

/**
 * Validate one key with a single authenticated GET. Never prints, logs or
 * returns the secret; the caller injects `fetchImpl` in tests so no test opens
 * a socket.
 *
 * @param {{name: string, secret: string, config?: object, fetchImpl?: Function, timeoutMs?: number}} opts
 * @returns {Promise<{name:string, status:'ok'|'bad-key'|'error'|'skipped', httpStatus:number|null, detail:string, url:string|null}>}
 */
export async function probeKey({ name, secret, config = {}, fetchImpl, timeoutMs = 10000 } = {}) {
  const key = assertKeyName(name);
  if (typeof secret !== 'string' || secret.trim() === '') {
    return { name: key, status: 'skipped', httpStatus: null, detail: 'not set', url: null };
  }
  const request = probeRequestFor(key, secret.trim(), config);
  if (request.skip) {
    return { name: key, status: 'skipped', httpStatus: null, detail: request.skip, url: null };
  }
  const doFetch = fetchImpl || globalThis.fetch;
  if (typeof doFetch !== 'function') {
    return { name: key, status: 'error', httpStatus: null, detail: 'fetch unavailable', url: request.url };
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await doFetch(request.url, {
      method: 'GET',
      headers: request.headers,
      signal: controller.signal,
    });
    const httpStatus = Number(res?.status ?? 0) || null;
    if (httpStatus === 200) {
      return { name: key, status: 'ok', httpStatus, detail: 'ok', url: request.url };
    }
    if (httpStatus === 401 || httpStatus === 403) {
      return { name: key, status: 'bad-key', httpStatus, detail: `rejected with HTTP ${httpStatus}`, url: request.url };
    }
    return {
      name: key,
      status: 'error',
      httpStatus,
      detail: `HTTP ${httpStatus ?? 'no-response'}`,
      url: request.url,
    };
  } catch (err) {
    const timedOut = err?.name === 'AbortError' || err?.code === 'ABORT_ERR';
    return {
      name: key,
      status: 'error',
      httpStatus: null,
      detail: timedOut ? 'timeout' : String(err?.code ?? err?.message ?? err),
      url: request.url,
    };
  } finally {
    clearTimeout(timer);
  }
}

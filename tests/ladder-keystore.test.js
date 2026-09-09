// kolm code - tests for src/ladder/keystore.js (bring-your-own-key store).
//
// Offline and deterministic: every probe gets an injected fetch, so nothing
// here opens a socket. Every filesystem touch happens inside a per-test temp
// dir under os.tmpdir(), and no fixture is a static secret-shaped string -
// values are assembled at runtime.

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { isLadderError } from '../src/ladder/errors.js';
import {
  KEY_NAMES,
  KEY_ENV_NAMES,
  KEYSTORE_VERSION,
  assertKeyName,
  assertKeyValue,
  describeKeys,
  fingerprint,
  keystoreDisabled,
  keystorePath,
  probeKey,
  readKeystore,
  setKey,
  unsetKey,
} from '../src/ladder/keystore.js';

// --- helpers ---------------------------------------------------------------

const tempDirs = [];

function makeTempDir() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kolm-keys-'));
  tempDirs.push(dir);
  return dir;
}

test.after(() => {
  for (const dir of tempDirs) {
    try {
      fs.rmSync(dir, { recursive: true, force: true });
    } catch {
      // deliberate: the OS reclaims a directory a handle still holds on Windows
    }
  }
});

/** A hermetic env whose keystore lives in its own temp dir. */
function envWithStore(extra = {}) {
  const dir = makeTempDir();
  return { dir, env: { KOLM_CODE_KEYS_PATH: path.join(dir, 'keys.json'), ...extra } };
}

/** Synthetic, never a real credential: a prefix plus a run of filler. */
function synthetic(prefix, length = 20) {
  return prefix + 'x'.repeat(length);
}

function fakeFetch(responder) {
  const calls = [];
  const impl = async (url, init) => {
    calls.push({ url, init });
    return responder(url, init);
  };
  impl.calls = calls;
  return impl;
}

const okResponse = { status: 200 };

// --- paths -----------------------------------------------------------------

test('keystorePath defaults to <data dir>/code/keys.json', () => {
  const home = makeTempDir();
  const dataDir = path.join(home, 'data');
  const file = keystorePath({ KOLM_CODE_HOME: home, KOLM_DATA_DIR: dataDir });
  assert.equal(file, path.join(dataDir, 'code', 'keys.json'));
});

test('keystorePath follows KOLM_DATA_DIR through a tilde and KOLM_CODE_KEYS_PATH over both', () => {
  const home = makeTempDir();
  const viaTilde = keystorePath({ KOLM_CODE_HOME: home, KOLM_DATA_DIR: '~/.kolm' });
  assert.equal(viaTilde, path.join(home, '.kolm', 'code', 'keys.json'));

  const override = keystorePath({
    KOLM_CODE_HOME: home,
    KOLM_DATA_DIR: '~/.kolm',
    KOLM_CODE_KEYS_PATH: '~/elsewhere/mine.json',
  });
  assert.equal(override, path.join(home, 'elsewhere', 'mine.json'));
  assert.ok(path.isAbsolute(override));
});

test('KOLM_CODE_NO_KEYSTORE flags are read like every other boolean', () => {
  assert.equal(keystoreDisabled({}), false);
  assert.equal(keystoreDisabled({ KOLM_CODE_NO_KEYSTORE: '0' }), false);
  for (const flag of ['1', 'true', 'YES', 'on']) {
    assert.equal(keystoreDisabled({ KOLM_CODE_NO_KEYSTORE: flag }), true, `${flag} disables`);
  }
});

// --- read / write ----------------------------------------------------------

test('a missing keystore reads as empty rather than failing', () => {
  const { env } = envWithStore();
  const store = readKeystore(env);
  assert.deepEqual(store.keys, {});
  assert.equal(store.enabled, true);
  assert.equal(store.path, env.KOLM_CODE_KEYS_PATH);
  assert.deepEqual(store.secrets, {});
});

test('setKey writes the documented shape and readKeystore returns it', () => {
  const { env } = envWithStore();
  const value = synthetic('sk-or-');
  const written = setKey(env, 'openrouter', value);
  assert.equal(written.name, 'openrouter');
  assert.equal(written.fingerprint, fingerprint(value));
  assert.match(written.set_at, /^\d{4}-\d{2}-\d{2}T/);

  const document = JSON.parse(fs.readFileSync(env.KOLM_CODE_KEYS_PATH, 'utf8'));
  assert.equal(document.version, KEYSTORE_VERSION);
  assert.deepEqual(Object.keys(document.keys), ['openrouter']);
  assert.equal(document.keys.openrouter.value, value);
  assert.equal(document.keys.openrouter.fingerprint, fingerprint(value));

  const store = readKeystore(env);
  assert.deepEqual(store.keys.openrouter, {
    fingerprint: fingerprint(value),
    set_at: written.set_at,
  });
  assert.equal(store.secrets.openrouter, value);
});

test('the secrets map is non-enumerable, so a stringified keystore carries no value', () => {
  const { env } = envWithStore();
  const value = synthetic('sk-ds-');
  setKey(env, 'deepseek', value);
  const store = readKeystore(env);

  const dump = JSON.stringify(store);
  assert.equal(dump.includes(value), false, 'the value leaked into JSON');
  assert.equal(dump.includes('secrets'), false, 'the secrets map leaked into JSON');
  assert.equal(Object.keys(store).includes('secrets'), false);
  assert.deepEqual(Object.keys(store), ['path', 'enabled', 'keys']);
  assert.equal(store.secrets.deepseek, value, 'still reachable through the accessor');
});

test('setKey leaves the other entries alone and unsetKey removes exactly one', () => {
  const { env } = envWithStore();
  const openrouter = synthetic('sk-or-');
  const deepseek = synthetic('sk-ds-');
  setKey(env, 'openrouter', openrouter);
  setKey(env, 'deepseek', deepseek);
  assert.deepEqual(Object.keys(readKeystore(env).keys).sort(), ['deepseek', 'openrouter']);

  const removed = unsetKey(env, 'openrouter');
  assert.equal(removed.removed, true);
  const after = readKeystore(env);
  assert.deepEqual(Object.keys(after.keys), ['deepseek']);
  assert.equal(after.secrets.deepseek, deepseek);

  const again = unsetKey(env, 'openrouter');
  assert.equal(again.removed, false, 'removing an absent key is not an error');
});

test('setKey rewrites an existing entry with a new fingerprint and timestamp', () => {
  const { env } = envWithStore();
  const first = setKey(env, 'cerebras', synthetic('csk-'));
  const second = setKey(env, 'cerebras', synthetic('csk-', 24));
  assert.notEqual(first.fingerprint, second.fingerprint);
  assert.equal(readKeystore(env).keys.cerebras.fingerprint, second.fingerprint);
  assert.equal(Object.keys(readKeystore(env).keys).length, 1);
});

test('the write is atomic: no temp file survives and the mode is 0600 on posix', () => {
  const { dir, env } = envWithStore();
  setKey(env, 'github', synthetic('ghp_', 30));
  const entries = fs.readdirSync(dir);
  assert.deepEqual(entries, ['keys.json'], `a temp file survived: ${entries.join(', ')}`);
  if (process.platform !== 'win32') {
    const mode = fs.statSync(env.KOLM_CODE_KEYS_PATH).mode & 0o777;
    assert.equal(mode, 0o600, 'the keystore is readable only by its owner');
  }
});

test('the directory is created when it does not exist yet', () => {
  const dir = makeTempDir();
  const env = { KOLM_CODE_KEYS_PATH: path.join(dir, 'deep', 'nested', 'keys.json') };
  setKey(env, 'anthropic', synthetic('sk-ant-'));
  assert.ok(fs.existsSync(env.KOLM_CODE_KEYS_PATH));
});

// --- failure modes ---------------------------------------------------------

test('a corrupt keystore raises a LadderError naming the path, never a SyntaxError', () => {
  const { env } = envWithStore();
  fs.writeFileSync(env.KOLM_CODE_KEYS_PATH, '{ this is not json', 'utf8');
  assert.throws(
    () => readKeystore(env),
    (err) => {
      assert.ok(isLadderError(err), `expected a LadderError, got ${err?.name}`);
      assert.equal(err.name, 'LadderError');
      assert.ok(err.message.includes(env.KOLM_CODE_KEYS_PATH), 'the message names the file');
      assert.equal(err.detail.path, env.KOLM_CODE_KEYS_PATH);
      return true;
    },
  );
});

test('a keystore that is not an object, or has no .keys object, fails loud', () => {
  const { env } = envWithStore();
  fs.writeFileSync(env.KOLM_CODE_KEYS_PATH, '[1,2,3]', 'utf8');
  assert.throws(() => readKeystore(env), (err) => isLadderError(err));
  fs.writeFileSync(env.KOLM_CODE_KEYS_PATH, '{"version":1,"keys":"nope"}', 'utf8');
  assert.throws(() => readKeystore(env), (err) => isLadderError(err));
});

test('an entry with no string value fails loud, an unknown entry name is ignored', () => {
  const { env } = envWithStore();
  fs.writeFileSync(
    env.KOLM_CODE_KEYS_PATH,
    JSON.stringify({ version: 1, keys: { anthropic: { set_at: 'x' } } }),
    'utf8',
  );
  assert.throws(
    () => readKeystore(env),
    (err) => isLadderError(err) && err.message.includes('anthropic'),
  );

  const value = synthetic('sk-ant-');
  fs.writeFileSync(
    env.KOLM_CODE_KEYS_PATH,
    JSON.stringify({
      version: 99,
      keys: { anthropic: { value }, 'some-future-provider': { value: synthetic('zz-') } },
    }),
    'utf8',
  );
  const store = readKeystore(env);
  assert.deepEqual(Object.keys(store.keys), ['anthropic'], 'a name this build does not know is skipped');
  assert.equal(store.keys.anthropic.fingerprint, fingerprint(value), 'a missing fingerprint is recomputed');
  assert.equal(store.keys.anthropic.set_at, null);
});

test('only the six documented names are accepted', () => {
  assert.deepEqual([...KEY_NAMES], [
    'anthropic',
    'cerebras',
    'deepseek',
    'openrouter',
    'github',
    'sandbox',
  ]);
  for (const name of KEY_NAMES) assert.equal(assertKeyName(name), name);
  assert.equal(assertKeyName('  Anthropic '), 'anthropic', 'names are trimmed and lowercased');

  const { env } = envWithStore();
  for (const bad of ['openai', 'gemini', '', 'anthropic-2', null]) {
    assert.throws(
      () => setKey(env, bad, synthetic('sk-')),
      (err) => isLadderError(err, 'bad-args'),
      `expected bad-args for ${JSON.stringify(bad)}`,
    );
  }
  assert.throws(() => unsetKey(env, 'openai'), (err) => isLadderError(err, 'bad-args'));
  assert.equal(fs.existsSync(env.KOLM_CODE_KEYS_PATH), false, 'a rejected name writes nothing');
});

test('an empty or whitespace-bearing value is refused with a legible message', () => {
  const { env } = envWithStore();
  for (const bad of ['', '   ', '\n', '\t']) {
    assert.throws(
      () => setKey(env, 'openrouter', bad),
      (err) => isLadderError(err, 'bad-args') && /empty/.test(err.message),
      `expected an empty-value error for ${JSON.stringify(bad)}`,
    );
  }
  for (const bad of ['sk-or wrapped', `sk-or-${'x'.repeat(8)}\nmore`, 'sk-or\tx']) {
    assert.throws(
      () => setKey(env, 'openrouter', bad),
      (err) => isLadderError(err, 'bad-args') && /whitespace|line break/.test(err.message),
      `expected a whitespace error for ${JSON.stringify(bad)}`,
    );
  }
  assert.throws(() => setKey(env, 'openrouter', 42), (err) => isLadderError(err, 'bad-args'));
  assert.equal(fs.existsSync(env.KOLM_CODE_KEYS_PATH), false, 'a rejected value writes nothing');
});

test('the stored value is trimmed', () => {
  const { env } = envWithStore();
  const value = synthetic('sk-or-');
  setKey(env, 'openrouter', `  ${value}\n`);
  assert.equal(readKeystore(env).secrets.openrouter, value);
  assert.equal(assertKeyValue('openrouter', ` ${value} `), value);
});

test('KOLM_CODE_NO_KEYSTORE=1 stops the file from being read at all', () => {
  const { env } = envWithStore();
  const value = synthetic('sk-ant-');
  setKey(env, 'anthropic', value);
  assert.equal(readKeystore(env).secrets.anthropic, value);

  const off = readKeystore({ ...env, KOLM_CODE_NO_KEYSTORE: '1' });
  assert.equal(off.enabled, false);
  assert.deepEqual(off.keys, {});
  assert.deepEqual(off.secrets, {});
  assert.equal(off.path, env.KOLM_CODE_KEYS_PATH, 'the path is still reported');

  // Even a corrupt file is inert while the keystore is disabled.
  fs.writeFileSync(env.KOLM_CODE_KEYS_PATH, 'not json at all', 'utf8');
  assert.deepEqual(readKeystore({ ...env, KOLM_CODE_NO_KEYSTORE: '1' }).keys, {});
});

// --- fingerprint -----------------------------------------------------------

test('fingerprint is a stable, short, non-reversible sha256 prefix', () => {
  const value = synthetic('sk-or-');
  const fp = fingerprint(value);
  assert.match(fp, /^sha256:[0-9a-f]{12}$/);
  assert.equal(fp, fingerprint(value), 'stable across calls');
  assert.notEqual(fp, fingerprint(`${value}z`));
  assert.equal(fp.includes(value.slice(4)), false, 'the fingerprint carries no part of the value');
});

// --- describeKeys ----------------------------------------------------------

test('describeKeys names the winning source without printing a value', () => {
  const { env } = envWithStore();
  const stored = synthetic('sk-ant-');
  const fromEnv = synthetic('sk-ant-', 24);
  setKey(env, 'anthropic', stored);
  setKey(env, 'openrouter', synthetic('sk-or-'));

  const described = describeKeys({ ...env, ANTHROPIC_API_KEY: fromEnv, GITHUB_TOKEN: synthetic('ghp_', 30) });
  const row = (name) => described.rows.find((r) => r.name === name);
  assert.equal(row('anthropic').source, 'env+keystore(env wins)');
  assert.equal(row('anthropic').fingerprint, fingerprint(fromEnv), 'the fingerprint is the winner');
  assert.equal(row('openrouter').source, 'keystore');
  assert.equal(row('github').source, 'env');
  assert.equal(row('deepseek').source, 'none');
  assert.equal(row('deepseek').fingerprint, null);
  assert.deepEqual(row('cerebras').envNames, [...KEY_ENV_NAMES.cerebras]);

  const dump = JSON.stringify(described);
  assert.equal(dump.includes(stored), false);
  assert.equal(dump.includes(fromEnv), false);
});

test('describeKeys reads the cerebras fallback variable too', () => {
  const { env } = envWithStore();
  const described = describeKeys({ ...env, KOLM_CEREBRAS_TOKEN: synthetic('csk-') });
  assert.equal(described.rows.find((r) => r.name === 'cerebras').source, 'env');
});

// --- probeKey --------------------------------------------------------------

const probeConfig = {
  baseUrls: {
    cerebras: 'https://cerebras.example/v1',
    deepseek: 'https://deepseek.example/v1',
    openrouter: 'https://openrouter.example/api/v1',
  },
  sandboxUrl: 'https://sandbox.example',
};

test('probeKey uses the documented endpoint and headers per provider', async () => {
  const secret = synthetic('sk-');

  const anthropic = fakeFetch(() => okResponse);
  assert.equal((await probeKey({ name: 'anthropic', secret, config: probeConfig, fetchImpl: anthropic })).status, 'ok');
  assert.equal(anthropic.calls[0].url, 'https://api.anthropic.com/v1/models');
  assert.equal(anthropic.calls[0].init.method, 'GET');
  assert.equal(anthropic.calls[0].init.headers['x-api-key'], secret);
  assert.equal(anthropic.calls[0].init.headers['anthropic-version'], '2023-06-01');
  assert.equal(anthropic.calls[0].init.headers.authorization, undefined);

  for (const name of ['cerebras', 'deepseek', 'openrouter']) {
    const impl = fakeFetch(() => okResponse);
    const result = await probeKey({ name, secret, config: probeConfig, fetchImpl: impl });
    assert.equal(result.status, 'ok');
    assert.equal(impl.calls[0].url, `${probeConfig.baseUrls[name]}/models`);
    assert.equal(impl.calls[0].init.headers.authorization, `Bearer ${secret}`);
  }

  const github = fakeFetch(() => okResponse);
  assert.equal((await probeKey({ name: 'github', secret, config: probeConfig, fetchImpl: github })).status, 'ok');
  assert.equal(github.calls[0].url, 'https://api.github.com/user');
  assert.equal(github.calls[0].init.headers.authorization, `Bearer ${secret}`);

  const sandbox = fakeFetch(() => okResponse);
  assert.equal((await probeKey({ name: 'sandbox', secret, config: probeConfig, fetchImpl: sandbox })).status, 'ok');
  assert.equal(sandbox.calls[0].url, 'https://sandbox.example/health');
  assert.equal(sandbox.calls[0].init.headers.authorization, `Bearer ${secret}`);
});

test('probeKey classifies 200, 401, 403, 404 and 500', async () => {
  const secret = synthetic('sk-');
  const cases = [
    [200, 'ok', null],
    [401, 'bad-key', 401],
    [403, 'bad-key', 403],
    [404, 'error', 404],
    [500, 'error', 500],
  ];
  for (const [status, expected, expectedStatus] of cases) {
    const result = await probeKey({
      name: 'openrouter',
      secret,
      config: probeConfig,
      fetchImpl: fakeFetch(() => ({ status })),
    });
    assert.equal(result.status, expected, `HTTP ${status} -> ${expected}`);
    assert.equal(result.httpStatus, status);
    if (expectedStatus !== null) assert.ok(result.detail.includes(String(expectedStatus)));
  }
});

test('probeKey turns a transport failure into an error result, never a throw', async () => {
  const failing = async () => {
    const err = new Error('getaddrinfo ENOTFOUND openrouter.example');
    err.code = 'ENOTFOUND';
    throw err;
  };
  const result = await probeKey({
    name: 'openrouter',
    secret: synthetic('sk-or-'),
    config: probeConfig,
    fetchImpl: failing,
  });
  assert.equal(result.status, 'error');
  assert.equal(result.httpStatus, null);
  assert.equal(result.detail, 'ENOTFOUND');
});

test('probeKey aborts on its own timeout and reports it as a timeout', async () => {
  const hanging = (_url, init) =>
    new Promise((_resolve, reject) => {
      init.signal.addEventListener('abort', () => {
        const err = new Error('The operation was aborted');
        err.name = 'AbortError';
        reject(err);
      });
    });
  const started = Date.now();
  const result = await probeKey({
    name: 'github',
    secret: synthetic('ghp_', 30),
    config: probeConfig,
    fetchImpl: hanging,
    timeoutMs: 25,
  });
  assert.equal(result.status, 'error');
  assert.equal(result.detail, 'timeout');
  assert.ok(Date.now() - started < 5000, 'the abort fired well inside the default budget');
});

test('probeKey skips what it cannot reach and refuses a name it does not know', async () => {
  const noUrl = await probeKey({
    name: 'sandbox',
    secret: synthetic('sbx-'),
    config: { baseUrls: {} },
    fetchImpl: fakeFetch(() => okResponse),
  });
  assert.equal(noUrl.status, 'skipped');
  assert.equal(noUrl.detail, 'sandbox url not configured');

  const unset = await probeKey({ name: 'deepseek', secret: '', config: probeConfig });
  assert.equal(unset.status, 'skipped');
  assert.equal(unset.detail, 'not set');

  await assert.rejects(
    () => probeKey({ name: 'openai', secret: synthetic('sk-'), config: probeConfig }),
    (err) => isLadderError(err, 'bad-args'),
  );
});

test('a probe result never carries the secret it tested', async () => {
  const secret = synthetic('sk-or-');
  for (const name of KEY_NAMES) {
    const result = await probeKey({
      name,
      secret,
      config: probeConfig,
      fetchImpl: fakeFetch(() => ({ status: 401 })),
    });
    assert.equal(JSON.stringify(result).includes(secret), false, `${name} leaked the secret`);
  }
});

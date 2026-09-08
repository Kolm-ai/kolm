// kolm code - tests for src/ladder/config.js and src/ladder/errors.js (owner A).
//
// Offline and deterministic: no network, no API keys, no spawned processes.
// Every filesystem touch happens inside a per-suite temp dir under os.tmpdir().

import test from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { LadderError, LADDER_ERROR_CODES, isLadderError } from '../src/ladder/errors.js';
import {
  DEFAULT_ORCHESTRATOR,
  DEFAULT_CODER_LADDER,
  DEFAULT_REVIEWER_LADDER,
  DEFAULT_OLLAMA_URL,
  DEFAULT_TRUFFLE_KEY_FILE,
  PROVIDER_KEY_ENV,
  parseRef,
  expandHome,
  loadConfig,
  withOllamaReachable,
  armingReport,
  effectiveLadders,
  ollamaApiBase,
  probeOllama,
  probeGhCli,
} from '../src/ladder/config.js';

// --- temp homes ------------------------------------------------------------

const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'kolm-code-cfg-'));
const emptyHome = path.join(tmpRoot, 'empty-home');
const truffleHome = path.join(tmpRoot, 'truffle-home');
fs.mkdirSync(emptyHome, { recursive: true });
fs.mkdirSync(path.join(truffleHome, '.truffle'), { recursive: true });
const truffleKeyPath = path.join(truffleHome, ...DEFAULT_TRUFFLE_KEY_FILE.split('/'));
// A throwaway key generated at test startup (no static PEM fixtures in the
// repository). Its base64 body is the marker the leak assertions look for.
const FAKE_PEM = crypto.generateKeyPairSync('ed25519', {
  privateKeyEncoding: { type: 'pkcs8', format: 'pem' },
  publicKeyEncoding: { type: 'spki', format: 'pem' },
}).privateKey.trim();
const FAKE_PEM_BODY = FAKE_PEM.split('\n').filter((line) => line && !line.startsWith('-----')).join('');
fs.writeFileSync(truffleKeyPath, FAKE_PEM, 'utf8');

test.after(() => {
  fs.rmSync(tmpRoot, { recursive: true, force: true });
});

/** A hermetic env: nothing inherited from process.env. */
function env(extra = {}) {
  return { KOLM_CODE_HOME: emptyHome, ...extra };
}

// --- errors ----------------------------------------------------------------

test('LadderError carries a code, serializes, and is recognized by isLadderError', () => {
  const err = new LadderError('bad-ref', 'nope', { ref: 'x' });
  assert.equal(err.name, 'LadderError');
  assert.equal(err.code, 'bad-ref');
  assert.deepEqual(err.detail, { ref: 'x' });
  assert.ok(err instanceof Error);
  assert.ok(isLadderError(err));
  assert.ok(isLadderError(err, 'bad-ref'));
  assert.equal(isLadderError(err, 'bad-config'), false);
  assert.equal(isLadderError(new Error('plain')), false);
  const json = JSON.parse(JSON.stringify(err));
  assert.equal(json.code, 'bad-ref');
  assert.equal(json.message, 'nope');
});

test('LADDER_ERROR_CODES is the frozen closed set from the spec', () => {
  assert.ok(Object.isFrozen(LADDER_ERROR_CODES));
  for (const code of [
    'not-armed',
    'provider-bad-request',
    'provider-unavailable',
    'provider-timeout',
    'budget-exceeded',
    'sqlite-unavailable',
    'bad-ref',
    'bad-config',
  ]) {
    assert.ok(LADDER_ERROR_CODES.includes(code), `missing ${code}`);
  }
});

// --- ref grammar -----------------------------------------------------------

test('parseRef splits on the first colon only', () => {
  assert.deepEqual(parseRef('anthropic:claude-fable-5-1'), {
    provider: 'anthropic',
    model: 'claude-fable-5-1',
    ref: 'anthropic:claude-fable-5-1',
  });
  assert.deepEqual(parseRef('ollama:qwen3.8:latest'), {
    provider: 'ollama',
    model: 'qwen3.8:latest',
    ref: 'ollama:qwen3.8:latest',
  });
  assert.equal(
    parseRef('openrouter:deepseek/deepseek-v4-flash').model,
    'deepseek/deepseek-v4-flash',
  );
  assert.equal(parseRef('  deepseek:deepseek-v4-flash  ').ref, 'deepseek:deepseek-v4-flash');
  assert.equal(parseRef('Anthropic:claude-opus-5').provider, 'anthropic');
});

test('parseRef rejects malformed refs with LadderError bad-ref', () => {
  for (const bad of ['', 'anthropic', ':model', 'anthropic:', 'has space:model', 'a/b:model']) {
    assert.throws(
      () => parseRef(bad),
      (e) => isLadderError(e, 'bad-ref'),
      `expected bad-ref for ${JSON.stringify(bad)}`,
    );
  }
  assert.throws(
    () => parseRef(null),
    (e) => isLadderError(e, 'bad-ref'),
  );
  assert.throws(
    () => parseRef(42),
    (e) => isLadderError(e, 'bad-ref'),
  );
});

test('expandHome expands ~ against the supplied home only', () => {
  assert.equal(expandHome('~', '/h'), '/h');
  assert.equal(expandHome('~/x', '/h'), path.join('/h', 'x'));
  assert.equal(expandHome('~\\x', '/h'), path.join('/h', 'x'));
  assert.equal(expandHome('/abs', '/h'), '/abs');
  assert.equal(expandHome('', '/h'), '');
  assert.equal(expandHome(null, '/h'), null);
});

// --- defaults --------------------------------------------------------------

test('loadConfig defaults match the spec', () => {
  const c = loadConfig(env());
  assert.equal(c.orchestratorRef, DEFAULT_ORCHESTRATOR);
  assert.equal(c.judgeRef, DEFAULT_ORCHESTRATOR);
  assert.equal(c.mode, 'cascade');
  assert.deepEqual(c.coderLadder, DEFAULT_CODER_LADDER.split(','));
  assert.deepEqual(c.reviewerLadder, DEFAULT_REVIEWER_LADDER.split(','));
  assert.equal(c.candidates, 3);
  assert.equal(c.coderTurns, 24);
  assert.equal(c.repairTurns, 2);
  assert.equal(c.maxCostUsd, 5);
  assert.equal(c.calibrationRate, 0.1);
  assert.equal(c.stageTimeoutMs, 600000);
  assert.equal(c.keepWorktrees, 'losers-destroyed');
  assert.equal(c.driver, 'local');
  assert.equal(c.ollamaUrl, DEFAULT_OLLAMA_URL);
  assert.equal(c.ollamaReachable, null);
  assert.equal(c.deepseekOffpeak, false);
  assert.equal(c.mockProvider, false);
  assert.equal(c.sandboxUrl, null);
  assert.equal(c.hasSandboxToken, false);
  assert.equal(c.bandsPath, null);
  assert.deepEqual(c.hasKey, {
    anthropic: false,
    cerebras: false,
    deepseek: false,
    openrouter: false,
  });
  assert.equal(c.home, emptyHome);
  assert.equal(c.dataDir, path.resolve(path.join(emptyHome, '.kolm')));
  assert.equal(c.codeDir, path.join(c.dataDir, 'code'));
  assert.equal(c.dbPath, path.resolve(path.join(c.codeDir, 'ladder.db')));
  assert.equal(c.worktreesDir, path.join(c.codeDir, 'worktrees'));
  assert.equal(c.runsDir, path.join(c.codeDir, 'runs'));
  assert.equal(c.baseUrls.cerebras, 'https://api.cerebras.ai/v1');
  assert.equal(c.baseUrls.deepseek, 'https://api.deepseek.com/v1');
  assert.equal(c.baseUrls.openrouter, 'https://openrouter.ai/api/v1');
  assert.equal(c.baseUrls.ollama, DEFAULT_OLLAMA_URL);
  assert.equal(c.truffle.harnessId, 'kolm-code');
  assert.equal(c.truffle.pollMs, 1000);
  assert.equal(c.github.hasPrivateKey, false);
  assert.equal(c.github.privateKeySource, null);
  assert.ok(Object.isFrozen(c));
});

test('loadConfig rejects a non-object env', () => {
  assert.throws(
    () => loadConfig(null),
    (e) => isLadderError(e, 'bad-config'),
  );
  assert.throws(
    () => loadConfig('HOME=/x'),
    (e) => isLadderError(e, 'bad-config'),
  );
});

test('PROVIDER_KEY_ENV names the env var for every keyed provider', () => {
  assert.deepEqual(PROVIDER_KEY_ENV, {
    anthropic: 'ANTHROPIC_API_KEY',
    cerebras: 'CEREBRAS_API_KEY',
    deepseek: 'DEEPSEEK_API_KEY',
    openrouter: 'OPENROUTER_API_KEY',
  });
});

// --- every env override ----------------------------------------------------

test('loadConfig honours every documented env override', () => {
  const dataDir = path.join(tmpRoot, 'data');
  const c = loadConfig(
    env({
      KOLM_CODE_ORCHESTRATOR: 'anthropic:claude-opus-5',
      KOLM_CODE_JUDGE: 'anthropic:claude-sonnet-5',
      KOLM_CODE_MODE: 'direct',
      KOLM_CODE_CODER_LADDER: 'fake:a, fake:b',
      KOLM_CODE_REVIEWER_LADDER: 'ollama:qwen3.8:latest',
      KOLM_CODE_CANDIDATES: '5',
      KOLM_CODE_CODER_TURNS: '40',
      KOLM_CODE_REPAIR_TURNS: '0',
      KOLM_CODE_MAX_COST_USD: '12.5',
      KOLM_CODE_CALIBRATION_RATE: '0.25',
      KOLM_CODE_STAGE_TIMEOUT_MS: '90000',
      KOLM_CODE_KEEP_WORKTREES: 'all',
      KOLM_CODE_DRIVER: 'cloudflare',
      KOLM_DATA_DIR: dataDir,
      KOLM_CODE_DB_PATH: path.join(dataDir, 'custom.db'),
      KOLM_CODE_SANDBOX_URL: 'https://sandbox.example',
      KOLM_CODE_SANDBOX_TOKEN: 'sbx-secret-token',
      KOLM_CODE_BANDS_PATH: '/etc/bands.json',
      KOLM_CODE_OLLAMA_URL: 'http://127.0.0.1:9999/v1',
      KOLM_CODE_DEEPSEEK_OFFPEAK: 'true',
      KOLM_TEST_MOCK_PROVIDER: '1',
      KOLM_CEREBRAS_URL: 'https://cerebras.internal/v1',
      KOLM_CODE_DEEPSEEK_URL: 'https://deepseek.internal/v1',
      KOLM_CODE_OPENROUTER_URL: 'https://openrouter.internal/api/v1',
      KOLM_TRUFFLE_HARNESS_ID: 'kolm-code-dev',
      TRUFFLE_PARTNER_HEALTH_HOST: '127.0.0.1',
      TRUFFLE_PARTNER_POLL_MS: '250',
    }),
  );
  assert.equal(c.orchestratorRef, 'anthropic:claude-opus-5');
  assert.equal(c.judgeRef, 'anthropic:claude-sonnet-5');
  assert.equal(c.mode, 'direct');
  assert.deepEqual(c.coderLadder, ['fake:a', 'fake:b']);
  assert.deepEqual(c.reviewerLadder, ['ollama:qwen3.8:latest']);
  assert.equal(c.candidates, 5);
  assert.equal(c.coderTurns, 40);
  assert.equal(c.repairTurns, 0);
  assert.equal(c.maxCostUsd, 12.5);
  assert.equal(c.calibrationRate, 0.25);
  assert.equal(c.stageTimeoutMs, 90000);
  assert.equal(c.keepWorktrees, 'all');
  assert.equal(c.driver, 'cloudflare');
  assert.equal(c.dataDir, path.resolve(dataDir));
  assert.equal(c.dbPath, path.resolve(path.join(dataDir, 'custom.db')));
  assert.equal(c.sandboxUrl, 'https://sandbox.example');
  assert.equal(c.hasSandboxToken, true);
  assert.equal(c.bandsPath, '/etc/bands.json');
  assert.equal(c.ollamaUrl, 'http://127.0.0.1:9999/v1');
  assert.equal(c.baseUrls.ollama, 'http://127.0.0.1:9999/v1');
  assert.equal(c.deepseekOffpeak, true);
  assert.equal(c.mockProvider, true);
  assert.equal(c.baseUrls.cerebras, 'https://cerebras.internal/v1');
  assert.equal(c.baseUrls.deepseek, 'https://deepseek.internal/v1');
  assert.equal(c.baseUrls.openrouter, 'https://openrouter.internal/api/v1');
  assert.equal(c.truffle.harnessId, 'kolm-code-dev');
  assert.equal(c.truffle.healthHost, '127.0.0.1');
  assert.equal(c.truffle.pollMs, 250);
});

test('provider keys set hasKey, including the Cerebras alias', () => {
  const c = loadConfig(
    env({
      ANTHROPIC_API_KEY: 'sk-ant-test',
      KOLM_CEREBRAS_TOKEN: 'csk-alias-test',
      DEEPSEEK_API_KEY: 'sk-ds-test',
      OPENROUTER_API_KEY: 'sk-or-test',
    }),
  );
  assert.deepEqual(c.hasKey, {
    anthropic: true,
    cerebras: true,
    deepseek: true,
    openrouter: true,
  });
  assert.equal(c.getSecret('cerebras'), 'csk-alias-test');
  const direct = loadConfig(
    env({ CEREBRAS_API_KEY: 'csk-direct', KOLM_CEREBRAS_TOKEN: 'csk-alias' }),
  );
  assert.equal(direct.getSecret('cerebras'), 'csk-direct');
});

test('values are trimmed, BOM-stripped, and empty strings fall back to defaults', () => {
  const c = loadConfig(env({ KOLM_CODE_MODE: '﻿  direct  ', KOLM_CODE_ORCHESTRATOR: '   ' }));
  assert.equal(c.mode, 'direct');
  assert.equal(c.orchestratorRef, DEFAULT_ORCHESTRATOR);
});

// --- bad config ------------------------------------------------------------

test('bad numbers, enums, booleans and refs raise LadderError bad-config', () => {
  const cases = [
    { KOLM_CODE_CANDIDATES: 'three' },
    { KOLM_CODE_CANDIDATES: '2.5' },
    { KOLM_CODE_CANDIDATES: '0' },
    { KOLM_CODE_CANDIDATES: '999' },
    { KOLM_CODE_MAX_COST_USD: '-1' },
    { KOLM_CODE_CALIBRATION_RATE: '2' },
    { KOLM_CODE_STAGE_TIMEOUT_MS: '10' },
    { KOLM_CODE_MODE: 'turbo' },
    { KOLM_CODE_DRIVER: 'kubernetes' },
    { KOLM_CODE_KEEP_WORKTREES: 'maybe' },
    { KOLM_CODE_DEEPSEEK_OFFPEAK: 'sometimes' },
    { TRUFFLE_PARTNER_POLL_MS: '1' },
  ];
  for (const extra of cases) {
    assert.throws(
      () => loadConfig(env(extra)),
      (e) => isLadderError(e, 'bad-config'),
      `expected bad-config for ${JSON.stringify(extra)}`,
    );
  }
});

test('a malformed ref anywhere in the ladder raises bad-ref', () => {
  assert.throws(
    () => loadConfig(env({ KOLM_CODE_CODER_LADDER: 'fake:a,notaref' })),
    (e) => isLadderError(e, 'bad-ref'),
  );
  assert.throws(
    () => loadConfig(env({ KOLM_CODE_ORCHESTRATOR: 'notaref' })),
    (e) => isLadderError(e, 'bad-ref'),
  );
  assert.throws(
    () => loadConfig(env({ KOLM_CODE_JUDGE: 'notaref' })),
    (e) => isLadderError(e, 'bad-ref'),
  );
});

// --- secret discipline -----------------------------------------------------

test('no secret value ever appears in JSON.stringify(config)', () => {
  const secretValues = [
    'sk-ant-SECRETA',
    'csk-SECRETB',
    'sk-ds-SECRETC',
    'sk-or-SECRETD',
    'sbx-SECRETE',
    'ghp-SECRETF',
    'trf-SECRETG',
    'sig-SECRETH',
  ];
  const c = loadConfig(
    env({
      ANTHROPIC_API_KEY: secretValues[0],
      CEREBRAS_API_KEY: secretValues[1],
      DEEPSEEK_API_KEY: secretValues[2],
      OPENROUTER_API_KEY: secretValues[3],
      KOLM_CODE_SANDBOX_TOKEN: secretValues[4],
      KOLM_GITHUB_TOKEN: secretValues[5],
      TRUFFLE_PARTNER_SERVICE_TOKEN: secretValues[6],
      TRUFFLE_PARTNER_PROCESSOR_SIGNING_SECRET: secretValues[7],
      KOLM_GITHUB_APP_PRIVATE_KEY: FAKE_PEM,
    }),
  );
  const dump = JSON.stringify(c);
  for (const value of secretValues) {
    assert.equal(dump.includes(value), false, `config leaked ${value}`);
  }
  assert.equal(dump.includes(FAKE_PEM_BODY), false, 'config leaked the inline PEM');
  assert.equal(dump.includes('PRIVATE KEY'), false, 'config leaked a PEM block');
  assert.equal(dump.includes('secrets'), false, 'config exposed a secrets key');

  // The values are still reachable through the non-enumerable accessors.
  assert.equal(c.getSecret('anthropic'), secretValues[0]);
  assert.equal(c.getSecret('githubToken'), secretValues[5]);
  assert.equal(c.sandboxToken, secretValues[4]);
  assert.equal(c.githubPrivateKeyPem(), FAKE_PEM);
  assert.equal(Object.keys(c).includes('secrets'), false);
  assert.equal(Object.keys(c).includes('sandboxToken'), false);
  assert.equal(Object.keys(c).includes('getSecret'), false);
  assert.equal(Object.keys(c).includes('githubPrivateKeyPem'), false);

  // Booleans stand in for the values in the enumerable surface.
  assert.equal(c.hasSandboxToken, true);
  assert.equal(c.github.token, true);
  assert.equal(c.github.privateKeyInline, true);
  assert.equal(c.truffle.serviceToken, true);
  assert.equal(c.truffle.signingSecret, true);
  assert.throws(
    () => c.getSecret('nope'),
    (e) => isLadderError(e, 'bad-config'),
  );
});

// --- GitHub / Truffle key resolution --------------------------------------

test('GitHub App key resolution order: inline kolm > inline truffle > path > default', () => {
  const inlineKolm = loadConfig(
    env({ KOLM_GITHUB_APP_PRIVATE_KEY: 'A-PEM', TRUFFLE_GITHUB_PRIVATE_KEY: 'B-PEM' }),
  );
  assert.equal(inlineKolm.github.privateKeySource, 'KOLM_GITHUB_APP_PRIVATE_KEY');
  assert.equal(inlineKolm.github.privateKeyPath, null);
  assert.equal(inlineKolm.githubPrivateKeyPem(), 'A-PEM');

  const inlineTruffle = loadConfig(env({ TRUFFLE_GITHUB_PRIVATE_KEY: 'B-PEM' }));
  assert.equal(inlineTruffle.github.privateKeySource, 'TRUFFLE_GITHUB_PRIVATE_KEY');
  assert.equal(inlineTruffle.githubPrivateKeyPem(), 'B-PEM');

  const explicit = loadConfig(env({ KOLM_GITHUB_APP_PRIVATE_KEY_PATH: truffleKeyPath }));
  assert.equal(explicit.github.privateKeySource, 'KOLM_GITHUB_APP_PRIVATE_KEY_PATH');
  assert.equal(explicit.github.privateKeyPath, path.resolve(truffleKeyPath));
  assert.equal(explicit.github.privateKeyInline, false);
  assert.equal(explicit.githubPrivateKeyPem(), FAKE_PEM);
});

test('the default ~/.truffle key path is used when and only when the file exists', () => {
  const present = loadConfig({ KOLM_CODE_HOME: truffleHome });
  assert.equal(present.github.privateKeySource, 'default-truffle-path');
  assert.equal(present.github.privateKeyPath, truffleKeyPath);
  assert.equal(present.github.hasPrivateKey, true);
  assert.equal(present.githubPrivateKeyPem(), FAKE_PEM);

  const absent = loadConfig({ KOLM_CODE_HOME: emptyHome });
  assert.equal(absent.github.privateKeySource, null);
  assert.equal(absent.github.privateKeyPath, null);
  assert.equal(absent.github.hasPrivateKey, false);
  assert.equal(absent.githubPrivateKeyPem(), null);
});

test('home resolves KOLM_CODE_HOME then HOME then USERPROFILE', () => {
  assert.equal(loadConfig({ KOLM_CODE_HOME: truffleHome, HOME: emptyHome }).home, truffleHome);
  assert.equal(loadConfig({ HOME: truffleHome, USERPROFILE: emptyHome }).home, truffleHome);
  assert.equal(loadConfig({ USERPROFILE: truffleHome }).home, truffleHome);
});

test('a missing private key file raises bad-config only when the PEM is actually read', () => {
  const missing = path.join(tmpRoot, 'nope', 'key.pem');
  const c = loadConfig(env({ KOLM_GITHUB_APP_PRIVATE_KEY_PATH: missing }));
  assert.equal(c.github.hasPrivateKey, true); // the path was configured explicitly
  assert.throws(
    () => c.githubPrivateKeyPem(),
    (e) => isLadderError(e, 'bad-config'),
  );
});

test('the GitHub App triple and token sources are reported without values', () => {
  const app = loadConfig(
    env({
      TRUFFLE_GITHUB_APP_ID: '123',
      TRUFFLE_GITHUB_INSTALLATION_ID: '456',
      KOLM_GITHUB_APP_PRIVATE_KEY: FAKE_PEM,
    }),
  );
  assert.equal(app.github.appId, '123');
  assert.equal(app.github.installationId, '456');
  assert.equal(app.github.app, true);
  assert.equal(app.github.token, false);
  assert.equal(app.github.tokenSource, null);

  const kolmToken = loadConfig(env({ KOLM_GITHUB_TOKEN: 't1', GITHUB_TOKEN: 't2' }));
  assert.equal(kolmToken.github.tokenSource, 'KOLM_GITHUB_TOKEN');
  assert.equal(kolmToken.getSecret('githubToken'), 't1');
  const ghToken = loadConfig(env({ GITHUB_TOKEN: 't2' }));
  assert.equal(ghToken.github.tokenSource, 'GITHUB_TOKEN');
});

// --- arming ----------------------------------------------------------------

function rung(report, id) {
  return report.rungs.find((r) => r.id === id);
}

test('armingReport marks every rung unarmed when nothing is configured', () => {
  const report = armingReport(loadConfig(env()));
  assert.equal(rung(report, 'orchestrator').armed, false);
  assert.equal(rung(report, 'judge').armed, false);
  assert.equal(rung(report, 'coder0').armed, false);
  assert.equal(rung(report, 'coder1').armed, false);
  assert.equal(rung(report, 'coder2').armed, false);
  assert.equal(rung(report, 'reviewer0').armed, false); // ollama reachability unknown
  assert.match(rung(report, 'reviewer0').reason, /unknown/);
  assert.equal(rung(report, 'driver').armed, true); // local git
  assert.equal(rung(report, 'github').armed, false);
  assert.equal(rung(report, 'truffle').armed, false);
  assert.match(rung(report, 'truffle').reason, /TRUFFLE_PARTNER_API_URL/);
});

test('armingReport never spawns or fetches: it only reports gh availability', () => {
  const report = armingReport(loadConfig(env()));
  assert.equal(rung(report, 'github').reason, 'no token; available if gh on PATH (gh auth token)');
});

test('armingReport arms rungs whose keys are present', () => {
  const c = loadConfig(
    env({
      ANTHROPIC_API_KEY: 'sk-ant',
      CEREBRAS_API_KEY: 'csk',
      DEEPSEEK_API_KEY: 'sk-ds',
      KOLM_GITHUB_TOKEN: 'ghp',
    }),
  );
  const report = armingReport(c);
  assert.equal(rung(report, 'orchestrator').armed, true);
  assert.equal(rung(report, 'judge').armed, true);
  assert.equal(rung(report, 'coder0').armed, true); // cerebras
  assert.equal(rung(report, 'coder1').armed, true); // deepseek
  assert.equal(rung(report, 'coder2').armed, true); // anthropic opus
  assert.equal(rung(report, 'github').armed, true);
  assert.equal(rung(report, 'github').reason, 'KOLM_GITHUB_TOKEN');
  assert.match(rung(report, 'coder0').reason, /CEREBRAS_API_KEY/);
});

test('ollama arming follows the injected probe or config.ollamaReachable', () => {
  const c = loadConfig(env());
  assert.equal(
    rung(armingReport(c, { probe: { ollama: { reachable: true } } }), 'reviewer0').armed,
    true,
  );
  assert.equal(
    rung(armingReport(c, { probe: { ollama: { reachable: false } } }), 'reviewer0').armed,
    false,
  );
  assert.equal(rung(armingReport(c, { ollama: { reachable: true } }), 'reviewer0').armed, true);

  const reachable = withOllamaReachable(c, true);
  assert.equal(reachable.ollamaReachable, true);
  assert.equal(rung(armingReport(reachable), 'reviewer0').armed, true);
  assert.match(rung(armingReport(reachable), 'reviewer0').reason, /reachable/);
});

test('withOllamaReachable preserves the non-enumerable secret carriers', () => {
  const c = loadConfig(env({ ANTHROPIC_API_KEY: 'sk-ant-keep' }));
  const next = withOllamaReachable(c, false);
  assert.equal(next.ollamaReachable, false);
  assert.equal(next.getSecret('anthropic'), 'sk-ant-keep');
  assert.equal(JSON.stringify(next).includes('sk-ant-keep'), false);
  assert.ok(Object.isFrozen(next));
});

test('fake refs are always armed and unknown providers never are', () => {
  const c = loadConfig(env({ KOLM_CODE_CODER_LADDER: 'fake:a,mistral:big' }));
  const report = armingReport(c);
  assert.equal(rung(report, 'coder0').armed, true);
  assert.equal(rung(report, 'coder1').armed, false);
  assert.match(rung(report, 'coder1').reason, /unknown provider/);
});

test('the cloudflare driver rung needs both the sandbox URL and its token', () => {
  const partial = armingReport(
    loadConfig(env({ KOLM_CODE_DRIVER: 'cloudflare', KOLM_CODE_SANDBOX_URL: 'https://s' })),
  );
  assert.equal(rung(partial, 'driver').armed, false);
  const full = armingReport(
    loadConfig(
      env({
        KOLM_CODE_DRIVER: 'cloudflare',
        KOLM_CODE_SANDBOX_URL: 'https://s',
        KOLM_CODE_SANDBOX_TOKEN: 'tok',
      }),
    ),
  );
  assert.equal(rung(full, 'driver').armed, true);
  const fake = armingReport(loadConfig(env({ KOLM_CODE_DRIVER: 'fake' })));
  assert.equal(rung(fake, 'driver').armed, true);
});

test('the truffle rung arms only when the whole partner contract is present', () => {
  const c = loadConfig(
    env({
      TRUFFLE_PARTNER_API_URL: 'https://truffle.example',
      TRUFFLE_PARTNER_SERVICE_TOKEN: 'tok',
      TRUFFLE_PARTNER_WORKSPACE_ID: 'ws',
      TRUFFLE_PARTNER_PROCESSOR_URL: 'https://proc.example',
      TRUFFLE_PARTNER_PROCESSOR_KEY_ID: 'kid',
      TRUFFLE_PARTNER_PROCESSOR_SIGNING_SECRET: 'sig',
    }),
  );
  assert.equal(rung(armingReport(c), 'truffle').armed, true);
  assert.equal(rung(armingReport(c), 'truffle').reason, 'partner contract vars set');
});

// --- effective ladders -----------------------------------------------------

test('effectiveLadders keeps armed coders in order and picks the first armed reviewer', () => {
  const c = loadConfig(
    env({
      ANTHROPIC_API_KEY: 'sk-ant',
      DEEPSEEK_API_KEY: 'sk-ds',
      KOLM_CODE_REVIEWER_LADDER: 'ollama:qwen3.8:latest,deepseek:deepseek-v4-flash',
    }),
  );
  const ladders = effectiveLadders(c);
  assert.deepEqual(ladders.coders, ['deepseek:deepseek-v4-flash', 'anthropic:claude-opus-5']);
  assert.equal(ladders.reviewer, 'deepseek:deepseek-v4-flash');
});

test('effectiveLadders falls back to the orchestrator when no coder is armed', () => {
  const c = loadConfig(
    env({ ANTHROPIC_API_KEY: 'sk-ant', KOLM_CODE_CODER_LADDER: 'cerebras:qwen-3.8-27b' }),
  );
  const ladders = effectiveLadders(c);
  assert.deepEqual(ladders.coders, [c.orchestratorRef]);
  assert.equal(ladders.reviewer, null);
});

test('effectiveLadders accepts a precomputed arming report', () => {
  const c = loadConfig(env());
  const arming = armingReport(c, { probe: { ollama: { reachable: true } } });
  assert.equal(effectiveLadders(c, arming).reviewer, 'ollama:qwen3.8:latest');
});

// --- probes ----------------------------------------------------------------

test('ollamaApiBase strips a trailing /v1 and slashes', () => {
  assert.equal(ollamaApiBase('http://127.0.0.1:11434/v1'), 'http://127.0.0.1:11434');
  assert.equal(ollamaApiBase('http://127.0.0.1:11434/v1/'), 'http://127.0.0.1:11434');
  assert.equal(ollamaApiBase('http://host:1/'), 'http://host:1');
  assert.equal(ollamaApiBase(undefined), 'http://127.0.0.1:11434');
});

test('probeOllama reports models from an injected fetch', async () => {
  const seen = [];
  const fetchImpl = async (url) => {
    seen.push(url);
    return {
      ok: true,
      status: 200,
      json: async () => ({ models: [{ name: 'qwen3.8:latest' }, { model: 'gemma:2b' }, {}] }),
    };
  };
  const result = await probeOllama(loadConfig(env()), { fetchImpl });
  assert.deepEqual(result, { reachable: true, models: ['qwen3.8:latest', 'gemma:2b'] });
  assert.deepEqual(seen, ['http://127.0.0.1:11434/api/tags']);
});

test('probeOllama reports unreachable on a non-2xx, a throw, and a missing fetch', async () => {
  const c = loadConfig(env());
  const notOk = await probeOllama(c, { fetchImpl: async () => ({ ok: false, status: 500 }) });
  assert.equal(notOk.reachable, false);
  assert.equal(notOk.error, 'HTTP 500');

  const threw = await probeOllama(c, {
    fetchImpl: async () => {
      throw Object.assign(new Error('connect ECONNREFUSED'), { code: 'ECONNREFUSED' });
    },
  });
  assert.equal(threw.reachable, false);
  assert.match(threw.error, /ECONNREFUSED/);

  const aborted = await probeOllama(c, {
    fetchImpl: async () => {
      throw Object.assign(new Error('aborted'), { name: 'AbortError' });
    },
  });
  assert.deepEqual(aborted, { reachable: false, models: [], error: 'timeout' });

  // A non-callable fetch is reported, never fallen back to the global one.
  const noFetch = await probeOllama(c, { fetchImpl: {} });
  assert.deepEqual(noFetch, { reachable: false, models: [], error: 'fetch unavailable' });
});

test('probeGhCli reports availability without ever returning the token', async () => {
  const authed = await probeGhCli({ run: async () => ({ code: 0, stdout: 'gho_SECRET\n' }) });
  assert.deepEqual(authed, { available: true, authenticated: true, reason: 'gh-cli' });
  assert.equal(JSON.stringify(authed).includes('gho_SECRET'), false);

  const loggedOut = await probeGhCli({ run: async () => ({ code: 1, stdout: '' }) });
  assert.equal(loggedOut.available, true);
  assert.equal(loggedOut.authenticated, false);

  const missing = await probeGhCli({
    run: async () => {
      throw Object.assign(new Error('spawn gh ENOENT'), { code: 'ENOENT' });
    },
  });
  assert.deepEqual(missing, {
    available: false,
    authenticated: false,
    reason: 'gh not usable: ENOENT',
  });
});

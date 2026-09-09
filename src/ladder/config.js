// kolm code — configuration, ref grammar and arming report (owner A, spec §2 + §11).
//
// `loadConfig(env)` is pure with respect to the environment object it is given:
// it reads only that object plus the filesystem existence check for the default
// Truffle GitHub App key path. It never reads a credential from a file, never
// spawns a process and never touches the network. Keys an operator stored with
// `kolm code keys set` reach it as `loadConfig(env, { fileSecrets })` — the
// caller (cli/kolm-code.mjs) does the reading, src/ladder/keystore.js owns it.
//
// Secrets discipline: no secret value is ever an enumerable own property of the
// returned config. `JSON.stringify(config)` is safe to log, embed in a receipt
// and print in `kolm code doctor`. Adapters read secrets through
// `config.getSecret(name)` (or the non-enumerable `config.secrets`).

import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs';

import { LadderError } from './errors.js';

/** Default model ref for planner + judge. */
export const DEFAULT_ORCHESTRATOR = 'anthropic:claude-fable-5-1';
export const DEFAULT_CODER_LADDER =
  'cerebras:qwen-3.8-27b,deepseek:deepseek-v4-flash,anthropic:claude-opus-5';
export const DEFAULT_REVIEWER_LADDER =
  'ollama:qwen3.8:latest,cerebras:gpt-oss-120b,deepseek:deepseek-v4-flash';
export const DEFAULT_OLLAMA_URL = 'http://127.0.0.1:11434/v1';
export const DEFAULT_TRUFFLE_KEY_FILE = '.truffle/truffle-pr-authority.private-key.pem';

const MODES = ['cascade', 'direct'];
const DRIVERS = ['local', 'cloudflare', 'fake'];
const KEEP_WORKTREES = ['losers-destroyed', 'all', 'none'];

/** Providers that need an API key, and the env var that supplies it. */
export const PROVIDER_KEY_ENV = Object.freeze({
  anthropic: 'ANTHROPIC_API_KEY',
  cerebras: 'CEREBRAS_API_KEY',
  deepseek: 'DEEPSEEK_API_KEY',
  openrouter: 'OPENROUTER_API_KEY',
});

/**
 * Names `loadConfig(env, { fileSecrets })` accepts. This is the canonical list
 * the keystore re-exports as KEY_NAMES; it lives here so config.js never has to
 * import the keystore (which would be a cycle: the keystore imports expandHome).
 */
export const FILE_SECRET_NAMES = Object.freeze([
  'anthropic',
  'cerebras',
  'deepseek',
  'openrouter',
  'github',
  'sandbox',
]);

// ---------------------------------------------------------------------------
// ref grammar
// ---------------------------------------------------------------------------

/**
 * Split a model ref on the FIRST colon only.
 * `ollama:qwen3.8:latest` -> { provider:'ollama', model:'qwen3.8:latest' }.
 * @param {string} ref
 * @returns {{provider: string, model: string, ref: string}}
 */
export function parseRef(ref) {
  if (typeof ref !== 'string') {
    throw new LadderError('bad-ref', `model ref must be a string, got ${typeof ref}`, { ref });
  }
  const trimmed = ref.trim();
  const idx = trimmed.indexOf(':');
  if (idx <= 0 || idx === trimmed.length - 1) {
    throw new LadderError(
      'bad-ref',
      `model ref must look like <provider>:<model>, got "${ref}"`,
      { ref },
    );
  }
  const provider = trimmed.slice(0, idx);
  const model = trimmed.slice(idx + 1);
  if (!/^[a-z0-9_-]+$/i.test(provider)) {
    throw new LadderError('bad-ref', `provider "${provider}" in ref "${ref}" is not a bare name`, {
      ref,
      provider,
    });
  }
  return { provider: provider.toLowerCase(), model, ref: trimmed };
}

// ---------------------------------------------------------------------------
// small parsing helpers
// ---------------------------------------------------------------------------

function str(env, key, fallback = null) {
  const raw = env[key];
  if (raw === undefined || raw === null) return fallback;
  // Env values on Windows can carry a BOM when written by some editors.
  const value = String(raw).replace(/^﻿+/, '').trim();
  return value === '' ? fallback : value;
}

function enumValue(env, key, allowed, fallback) {
  const value = str(env, key, fallback);
  if (!allowed.includes(value)) {
    throw new LadderError(
      'bad-config',
      `${key} must be one of ${allowed.join(' | ')}, got "${value}"`,
      { key, value, allowed },
    );
  }
  return value;
}

function num(env, key, fallback, { min = 0, max = Number.MAX_SAFE_INTEGER, integer = false } = {}) {
  const raw = str(env, key, null);
  if (raw === null) return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value)) {
    throw new LadderError('bad-config', `${key} must be a number, got "${raw}"`, { key, raw });
  }
  if (integer && !Number.isInteger(value)) {
    throw new LadderError('bad-config', `${key} must be an integer, got "${raw}"`, { key, raw });
  }
  if (value < min || value > max) {
    throw new LadderError(
      'bad-config',
      `${key} must be between ${min} and ${max}, got ${value}`,
      { key, value, min, max },
    );
  }
  return value;
}

function boolValue(env, key, fallback = false) {
  const raw = str(env, key, null);
  if (raw === null) return fallback;
  const lowered = raw.toLowerCase();
  if (['1', 'true', 'yes', 'on'].includes(lowered)) return true;
  if (['0', 'false', 'no', 'off'].includes(lowered)) return false;
  throw new LadderError('bad-config', `${key} must be a boolean flag, got "${raw}"`, { key, raw });
}

function refList(env, key, fallback) {
  const raw = str(env, key, fallback);
  const parts = String(raw)
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  for (const part of parts) parseRef(part); // fail loud on a malformed entry
  return parts;
}

/** `~` and `~/x` expand against the resolved home directory. */
export function expandHome(p, home) {
  if (!p) return p;
  if (p === '~') return home;
  if (p.startsWith('~/') || p.startsWith('~\\')) return path.join(home, p.slice(2));
  return p;
}

function resolveHome(env) {
  return (
    str(env, 'KOLM_CODE_HOME', null) ||
    str(env, 'HOME', null) ||
    str(env, 'USERPROFILE', null) ||
    os.homedir()
  );
}

function fileExists(p) {
  try {
    return fs.statSync(p).isFile();
  } catch {
    return false; // deliberate: a missing/unreadable path simply does not arm anything
  }
}

function deepFreeze(value) {
  if (value === null) return value;
  const t = typeof value;
  if (t !== 'object' && t !== 'function') return value;
  if (Object.isFrozen(value)) return value;
  Object.freeze(value);
  if (t === 'function') return value;
  for (const key of Object.getOwnPropertyNames(value)) {
    const desc = Object.getOwnPropertyDescriptor(value, key);
    if (desc && 'value' in desc) deepFreeze(desc.value);
  }
  return value;
}

// ---------------------------------------------------------------------------
// loadConfig
// ---------------------------------------------------------------------------

/**
 * Read one name out of an `opts.fileSecrets` bag with the same trimming the env
 * reader uses. Returns null for anything absent, blank or not a string.
 * @param {object} bag
 * @param {string} name
 * @returns {string|null}
 */
function fileSecret(bag, name) {
  const raw = bag[name];
  if (raw === undefined || raw === null) return null;
  const value = String(raw).replace(/^\uFEFF+/, '').trim();
  return value === '' ? null : value;
}

/**
 * `loadConfig` stays pure: it reads the env object it is given plus one
 * existence check for the default Truffle key path, and NEVER opens the
 * keystore file. The CLI reads `<codeDir>/keys.json` (src/ladder/keystore.js)
 * and hands the result in as `opts.fileSecrets`.
 *
 * Precedence is env-wins-over-file, matching `gh` and the AWS CLI: a variable
 * exported for one command, or injected by CI, must be able to override a key
 * an operator once stored on the machine without editing that file.
 *
 * @param {Record<string,string|undefined>} [env]
 * @param {{fileSecrets?: Record<string,string|null|undefined>}} [opts]
 * @returns {object} deep-frozen config
 */
export function loadConfig(env = process.env, opts = {}) {
  if (!env || typeof env !== 'object') {
    throw new LadderError('bad-config', 'loadConfig expects an environment object', {});
  }
  if (opts === null || typeof opts !== 'object') {
    throw new LadderError('bad-config', 'loadConfig options must be an object', {});
  }
  const rawFileSecrets = opts.fileSecrets;
  if (rawFileSecrets !== undefined && rawFileSecrets !== null && typeof rawFileSecrets !== 'object') {
    throw new LadderError(
      'bad-config',
      `loadConfig fileSecrets must be an object, got ${typeof rawFileSecrets}`,
      {},
    );
  }
  const bag = rawFileSecrets ?? {};
  const fromFile = (name) => (FILE_SECRET_NAMES.includes(name) ? fileSecret(bag, name) : null);
  const home = resolveHome(env);

  // --- orchestration -------------------------------------------------------
  const orchestratorRef = str(env, 'KOLM_CODE_ORCHESTRATOR', DEFAULT_ORCHESTRATOR);
  parseRef(orchestratorRef);
  const judgeRef = str(env, 'KOLM_CODE_JUDGE', orchestratorRef);
  parseRef(judgeRef);
  const mode = enumValue(env, 'KOLM_CODE_MODE', MODES, 'cascade');
  const coderLadder = refList(env, 'KOLM_CODE_CODER_LADDER', DEFAULT_CODER_LADDER);
  const reviewerLadder = refList(env, 'KOLM_CODE_REVIEWER_LADDER', DEFAULT_REVIEWER_LADDER);

  // --- budgets and shape ---------------------------------------------------
  const candidates = num(env, 'KOLM_CODE_CANDIDATES', 3, { min: 1, max: 64, integer: true });
  const coderTurns = num(env, 'KOLM_CODE_CODER_TURNS', 24, { min: 1, max: 500, integer: true });
  const repairTurns = num(env, 'KOLM_CODE_REPAIR_TURNS', 2, { min: 0, max: 50, integer: true });
  const maxCostUsd = num(env, 'KOLM_CODE_MAX_COST_USD', 5, { min: 0, max: 100000 });
  const calibrationRate = num(env, 'KOLM_CODE_CALIBRATION_RATE', 0.1, { min: 0, max: 1 });
  const stageTimeoutMs = num(env, 'KOLM_CODE_STAGE_TIMEOUT_MS', 600000, {
    min: 1000,
    max: 24 * 60 * 60 * 1000,
    integer: true,
  });
  const keepWorktrees = enumValue(
    env,
    'KOLM_CODE_KEEP_WORKTREES',
    KEEP_WORKTREES,
    'losers-destroyed',
  );

  // --- worktrees and data --------------------------------------------------
  const driver = enumValue(env, 'KOLM_CODE_DRIVER', DRIVERS, 'local');
  const dataDir = path.resolve(expandHome(str(env, 'KOLM_DATA_DIR', '~/.kolm'), home));
  const codeDir = path.join(dataDir, 'code');
  const dbPath = path.resolve(
    expandHome(str(env, 'KOLM_CODE_DB_PATH', path.join(codeDir, 'ladder.db')), home),
  );
  const worktreesDir = path.join(codeDir, 'worktrees');
  const runsDir = path.join(codeDir, 'runs');
  const sandboxUrl = str(env, 'KOLM_CODE_SANDBOX_URL', null);
  const sandboxTokenEnv = str(env, 'KOLM_CODE_SANDBOX_TOKEN', null);
  const sandboxTokenValue = sandboxTokenEnv || fromFile('sandbox');
  const sandboxTokenSource = sandboxTokenEnv
    ? 'KOLM_CODE_SANDBOX_TOKEN'
    : fromFile('sandbox')
      ? 'keystore'
      : null;
  const bandsPath = str(env, 'KOLM_CODE_BANDS_PATH', null);

  // --- providers -----------------------------------------------------------
  // Each key: the environment first, the keystore file second (see the note on
  // loadConfig), and `keySource` records which of the two actually won.
  const anthropicEnvKey = str(env, 'ANTHROPIC_API_KEY', null);
  const cerebrasEnvKey =
    str(env, 'CEREBRAS_API_KEY', null) || str(env, 'KOLM_CEREBRAS_TOKEN', null);
  const deepseekEnvKey = str(env, 'DEEPSEEK_API_KEY', null);
  const openrouterEnvKey = str(env, 'OPENROUTER_API_KEY', null);
  const anthropicKey = anthropicEnvKey || fromFile('anthropic');
  const cerebrasKey = cerebrasEnvKey || fromFile('cerebras');
  const deepseekKey = deepseekEnvKey || fromFile('deepseek');
  const openrouterKey = openrouterEnvKey || fromFile('openrouter');
  const ollamaUrl = str(env, 'KOLM_CODE_OLLAMA_URL', DEFAULT_OLLAMA_URL);
  const deepseekOffpeak = boolValue(env, 'KOLM_CODE_DEEPSEEK_OFFPEAK', false);
  const mockProvider = boolValue(env, 'KOLM_TEST_MOCK_PROVIDER', false);

  const baseUrls = {
    cerebras: str(env, 'KOLM_CEREBRAS_URL', 'https://api.cerebras.ai/v1'),
    deepseek: str(env, 'KOLM_CODE_DEEPSEEK_URL', 'https://api.deepseek.com/v1'),
    openrouter: str(env, 'KOLM_CODE_OPENROUTER_URL', 'https://openrouter.ai/api/v1'),
    ollama: ollamaUrl,
  };

  // --- github --------------------------------------------------------------
  const ghAppId = str(env, 'KOLM_GITHUB_APP_ID', null) || str(env, 'TRUFFLE_GITHUB_APP_ID', null);
  const ghInstallationId =
    str(env, 'KOLM_GITHUB_APP_INSTALLATION_ID', null) ||
    str(env, 'TRUFFLE_GITHUB_INSTALLATION_ID', null);

  const inlineKolmPem = str(env, 'KOLM_GITHUB_APP_PRIVATE_KEY', null);
  const inlineTrufflePem = str(env, 'TRUFFLE_GITHUB_PRIVATE_KEY', null);
  const explicitKeyPath = str(env, 'KOLM_GITHUB_APP_PRIVATE_KEY_PATH', null);
  const defaultKeyPath = path.join(home, ...DEFAULT_TRUFFLE_KEY_FILE.split('/'));

  let privateKeyInline = null;
  let privateKeyPath = null;
  let privateKeySource = null;
  if (inlineKolmPem) {
    privateKeyInline = inlineKolmPem;
    privateKeySource = 'KOLM_GITHUB_APP_PRIVATE_KEY';
  } else if (inlineTrufflePem) {
    privateKeyInline = inlineTrufflePem;
    privateKeySource = 'TRUFFLE_GITHUB_PRIVATE_KEY';
  } else if (explicitKeyPath) {
    privateKeyPath = path.resolve(expandHome(explicitKeyPath, home));
    privateKeySource = 'KOLM_GITHUB_APP_PRIVATE_KEY_PATH';
  } else if (fileExists(defaultKeyPath)) {
    privateKeyPath = defaultKeyPath;
    privateKeySource = 'default-truffle-path';
  }

  const ghTokenEnv = str(env, 'KOLM_GITHUB_TOKEN', null) || str(env, 'GITHUB_TOKEN', null);
  const ghToken = ghTokenEnv || fromFile('github');
  const ghTokenSource = str(env, 'KOLM_GITHUB_TOKEN', null)
    ? 'KOLM_GITHUB_TOKEN'
    : str(env, 'GITHUB_TOKEN', null)
      ? 'GITHUB_TOKEN'
      : fromFile('github')
        ? 'keystore'
        : null;

  const github = {
    appId: ghAppId,
    installationId: ghInstallationId,
    privateKeyPath,
    privateKeyInline: Boolean(privateKeyInline),
    privateKeySource,
    hasPrivateKey: Boolean(privateKeyInline || privateKeyPath),
    token: Boolean(ghToken),
    tokenSource: ghTokenSource,
    app: Boolean(ghAppId && ghInstallationId && (privateKeyInline || privateKeyPath)),
  };

  // --- truffle -------------------------------------------------------------
  const truffleApiUrl =
    str(env, 'TRUFFLE_PARTNER_API_URL', null) || str(env, 'KOLM_TRUFFLE_BASE_URL', null);
  const truffleProcessorUrl = str(env, 'TRUFFLE_PARTNER_PROCESSOR_URL', null);
  const truffleServiceToken =
    str(env, 'TRUFFLE_PARTNER_SERVICE_TOKEN', null) ||
    str(env, 'KOLM_TRUFFLE_PARTNER_TOKEN', null);
  const truffleWorkspaceId = str(env, 'TRUFFLE_PARTNER_WORKSPACE_ID', null);
  const truffleKeyId = str(env, 'TRUFFLE_PARTNER_PROCESSOR_KEY_ID', null);
  const truffleSigningSecret = str(env, 'TRUFFLE_PARTNER_PROCESSOR_SIGNING_SECRET', null);

  const truffle = {
    apiUrl: truffleApiUrl,
    processorUrl: truffleProcessorUrl,
    workspaceId: truffleWorkspaceId,
    keyId: truffleKeyId,
    harnessId: str(env, 'KOLM_TRUFFLE_HARNESS_ID', 'kolm-code'),
    healthHost: str(env, 'TRUFFLE_PARTNER_HEALTH_HOST', '0.0.0.0'),
    pollMs: num(env, 'TRUFFLE_PARTNER_POLL_MS', 1000, { min: 100, max: 600000, integer: true }),
    serviceToken: Boolean(truffleServiceToken),
    signingSecret: Boolean(truffleSigningSecret),
  };

  const secrets = {
    anthropic: anthropicKey,
    cerebras: cerebrasKey,
    deepseek: deepseekKey,
    openrouter: openrouterKey,
    ollama: null,
    sandbox: sandboxTokenValue,
    githubToken: ghToken,
    githubPrivateKey: privateKeyInline,
    truffleServiceToken,
    truffleSigningSecret,
  };

  const config = {
    // orchestration
    orchestratorRef,
    judgeRef,
    mode,
    coderLadder,
    reviewerLadder,
    // budgets
    candidates,
    coderTurns,
    repairTurns,
    maxCostUsd,
    calibrationRate,
    stageTimeoutMs,
    keepWorktrees,
    // worktrees / data
    driver,
    dataDir,
    codeDir,
    dbPath,
    worktreesDir,
    runsDir,
    sandboxUrl,
    hasSandboxToken: Boolean(sandboxTokenValue),
    sandboxTokenSource,
    bandsPath,
    // providers
    ollamaUrl,
    ollamaReachable: null,
    deepseekOffpeak,
    mockProvider,
    baseUrls,
    hasKey: {
      anthropic: Boolean(anthropicKey),
      cerebras: Boolean(cerebrasKey),
      deepseek: Boolean(deepseekKey),
      openrouter: Boolean(openrouterKey),
    },
    keyEnvNames: {
      anthropic: 'ANTHROPIC_API_KEY',
      cerebras: 'CEREBRAS_API_KEY (or KOLM_CEREBRAS_TOKEN)',
      deepseek: 'DEEPSEEK_API_KEY',
      openrouter: 'OPENROUTER_API_KEY',
    },
    // 'env' | 'keystore' | null per name. Enumerable on purpose: it says where
    // a secret came from, never what it is, so it is safe in doctor and JSON.
    keySource: {
      anthropic: anthropicEnvKey ? 'env' : anthropicKey ? 'keystore' : null,
      cerebras: cerebrasEnvKey ? 'env' : cerebrasKey ? 'keystore' : null,
      deepseek: deepseekEnvKey ? 'env' : deepseekKey ? 'keystore' : null,
      openrouter: openrouterEnvKey ? 'env' : openrouterKey ? 'keystore' : null,
      github: ghTokenEnv ? 'env' : ghToken ? 'keystore' : null,
      sandbox: sandboxTokenEnv ? 'env' : sandboxTokenValue ? 'keystore' : null,
    },
    // integrations
    github,
    truffle,
    home,
  };

  // Secret-bearing members are non-enumerable so JSON.stringify(config) is safe.
  Object.defineProperty(config, 'secrets', {
    value: secrets,
    enumerable: false,
    writable: false,
    configurable: false,
  });
  Object.defineProperty(config, 'sandboxToken', {
    value: sandboxTokenValue,
    enumerable: false,
    writable: false,
    configurable: false,
  });
  Object.defineProperty(config, 'getSecret', {
    value: function getSecret(name) {
      if (!Object.prototype.hasOwnProperty.call(secrets, name)) {
        throw new LadderError('bad-config', `unknown secret "${name}"`, { name });
      }
      return secrets[name];
    },
    enumerable: false,
    writable: false,
    configurable: false,
  });
  Object.defineProperty(config, 'githubPrivateKeyPem', {
    value: function githubPrivateKeyPem() {
      if (privateKeyInline) return privateKeyInline;
      if (!privateKeyPath) return null;
      try {
        return fs.readFileSync(privateKeyPath, 'utf8');
      } catch (err) {
        throw new LadderError(
          'bad-config',
          `GitHub App private key at ${privateKeyPath} could not be read`,
          { path: privateKeyPath, cause: err.code || err.name },
        );
      }
    },
    enumerable: false,
    writable: false,
    configurable: false,
  });

  return deepFreeze(config);
}

/**
 * Return a copy of `config` with `ollamaReachable` replaced. Config objects are
 * frozen, so the doctor uses this rather than mutating.
 * @param {object} config
 * @param {boolean|null} reachable
 */
export function withOllamaReachable(config, reachable) {
  const next = { ...config, ollamaReachable: reachable };
  for (const key of ['secrets', 'sandboxToken', 'getSecret', 'githubPrivateKeyPem']) {
    const desc = Object.getOwnPropertyDescriptor(config, key);
    if (desc) Object.defineProperty(next, key, { ...desc, configurable: false });
  }
  return deepFreeze(next);
}

// ---------------------------------------------------------------------------
// arming
// ---------------------------------------------------------------------------

function armingForRef(config, ref, ollamaProbe) {
  let parsed;
  try {
    parsed = parseRef(ref);
  } catch {
    return { armed: false, reason: `malformed ref "${ref}"` };
  }
  const { provider } = parsed;
  if (provider === 'fake') return { armed: true, reason: 'scripted fake provider' };
  if (provider === 'ollama') {
    const reachable = ollamaProbe ? ollamaProbe.reachable : config.ollamaReachable;
    if (reachable === true) return { armed: true, reason: `ollama reachable at ${config.ollamaUrl}` };
    if (reachable === false) {
      return { armed: false, reason: `ollama unreachable at ${config.ollamaUrl}` };
    }
    return {
      armed: false,
      reason: `ollama reachability unknown (kolm code doctor probes ${config.ollamaUrl})`,
    };
  }
  const envName = config.keyEnvNames?.[provider];
  if (!envName) {
    return { armed: false, reason: `unknown provider "${provider}"` };
  }
  const has = Boolean(config.hasKey?.[provider]);
  if (!has) {
    return {
      armed: false,
      reason: `${envName} missing (set it, or: kolm code keys set ${provider})`,
    };
  }
  const source = config.keySource?.[provider] ?? null;
  return { armed: true, reason: source === 'keystore' ? `${envName} set (keystore)` : `${envName} set` };
}

/**
 * Synchronous, pure arming report. Never spawns a process, never opens a socket.
 * @param {object} config
 * @param {{probe?: {ollama?: {reachable: boolean|null, models?: string[]}}, ollama?: object}} [opts]
 * @returns {{rungs: Array<{id:string,kind:string,ref:string|null,armed:boolean,reason:string}>}}
 */
export function armingReport(config, opts = {}) {
  const ollamaProbe = opts?.probe?.ollama ?? opts?.ollama ?? null;
  const rungs = [];

  const orchestrator = armingForRef(config, config.orchestratorRef, ollamaProbe);
  rungs.push({
    id: 'orchestrator',
    kind: 'orchestrator',
    ref: config.orchestratorRef,
    armed: orchestrator.armed,
    reason: orchestrator.reason,
  });

  const judge = armingForRef(config, config.judgeRef, ollamaProbe);
  rungs.push({
    id: 'judge',
    kind: 'judge',
    ref: config.judgeRef,
    armed: judge.armed,
    reason: judge.reason,
  });

  config.coderLadder.forEach((ref, i) => {
    const a = armingForRef(config, ref, ollamaProbe);
    rungs.push({ id: `coder${i}`, kind: 'coder', ref, armed: a.armed, reason: a.reason });
  });

  config.reviewerLadder.forEach((ref, i) => {
    const a = armingForRef(config, ref, ollamaProbe);
    rungs.push({ id: `reviewer${i}`, kind: 'reviewer', ref, armed: a.armed, reason: a.reason });
  });

  // driver — no live checks here; `kolm code doctor` runs `git --version`.
  let driverArmed = true;
  let driverReason = 'local git';
  if (config.driver === 'cloudflare') {
    driverArmed = Boolean(config.sandboxUrl && config.hasSandboxToken);
    driverReason = driverArmed
      ? `cloudflare sandbox at ${config.sandboxUrl}`
      : 'KOLM_CODE_SANDBOX_URL + KOLM_CODE_SANDBOX_TOKEN required';
  } else if (config.driver === 'fake') {
    driverReason = 'in-memory fake driver';
  }
  rungs.push({
    id: 'driver',
    kind: 'driver',
    ref: config.driver,
    armed: driverArmed,
    reason: driverReason,
  });

  // github — token or the full app triple; gh-cli is reported, never probed here.
  const gh = config.github;
  let githubArmed = false;
  let githubReason = 'no token; available if gh on PATH (gh auth token)';
  if (gh.token) {
    githubArmed = true;
    githubReason = gh.tokenSource;
  } else if (gh.app) {
    githubArmed = true;
    githubReason = 'github-app';
  }
  rungs.push({ id: 'github', kind: 'github', ref: null, armed: githubArmed, reason: githubReason });

  // truffle — every leg of the partner contract must be present.
  const t = config.truffle;
  const truffleParts = {
    TRUFFLE_PARTNER_API_URL: Boolean(t.apiUrl),
    TRUFFLE_PARTNER_SERVICE_TOKEN: t.serviceToken,
    TRUFFLE_PARTNER_WORKSPACE_ID: Boolean(t.workspaceId),
    TRUFFLE_PARTNER_PROCESSOR_URL: Boolean(t.processorUrl),
    TRUFFLE_PARTNER_PROCESSOR_KEY_ID: Boolean(t.keyId),
    TRUFFLE_PARTNER_PROCESSOR_SIGNING_SECRET: t.signingSecret,
  };
  const missing = Object.keys(truffleParts).filter((k) => !truffleParts[k]);
  rungs.push({
    id: 'truffle',
    kind: 'truffle',
    ref: t.harnessId,
    armed: missing.length === 0,
    reason: missing.length === 0 ? 'partner contract vars set' : `missing ${missing.join(', ')}`,
  });

  return { rungs };
}

/**
 * Coder ladder filtered to armed rungs (falls back to the orchestrator), plus
 * the first armed reviewer.
 * @param {object} config
 * @param {{rungs: Array}} [arming]
 * @returns {{coders: string[], reviewer: string|null}}
 */
export function effectiveLadders(config, arming = armingReport(config)) {
  const rungs = arming?.rungs ?? [];
  const coders = rungs.filter((r) => r.kind === 'coder' && r.armed).map((r) => r.ref);
  const reviewer = rungs.find((r) => r.kind === 'reviewer' && r.armed)?.ref ?? null;
  return { coders: coders.length ? coders : [config.orchestratorRef], reviewer };
}

// ---------------------------------------------------------------------------
// probes (async, called by `kolm code doctor` — never by armingReport)
// ---------------------------------------------------------------------------

/** Strip a trailing `/v1` so the native Ollama API path can be built. */
export function ollamaApiBase(url) {
  return String(url || DEFAULT_OLLAMA_URL).replace(/\/+$/, '').replace(/\/v1$/, '');
}

/**
 * `GET {ollamaUrl minus /v1}/api/tags` with a 1500 ms budget.
 * @param {object} config
 * @param {{fetchImpl?: Function, timeoutMs?: number}} [opts]
 * @returns {Promise<{reachable: boolean, models: string[], error?: string}>}
 */
export async function probeOllama(config, opts = {}) {
  const fetchImpl = opts.fetchImpl || globalThis.fetch;
  const timeoutMs = opts.timeoutMs ?? 1500;
  const url = `${ollamaApiBase(config?.ollamaUrl)}/api/tags`;
  if (typeof fetchImpl !== 'function') {
    return { reachable: false, models: [], error: 'fetch unavailable' };
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetchImpl(url, { method: 'GET', signal: controller.signal });
    if (!res || !res.ok) {
      return { reachable: false, models: [], error: `HTTP ${res ? res.status : 'no-response'}` };
    }
    const body = await res.json();
    const models = Array.isArray(body?.models)
      ? body.models.map((m) => m?.name ?? m?.model).filter((n) => typeof n === 'string')
      : [];
    return { reachable: true, models };
  } catch (err) {
    return { reachable: false, models: [], error: err?.name === 'AbortError' ? 'timeout' : String(err?.message || err) };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Report whether the `gh` CLI can mint a token. Never returns the token itself.
 * @param {{run?: Function, timeoutMs?: number}} [opts] `run(cmd)` -> {code, stdout}
 * @returns {Promise<{available: boolean, authenticated: boolean, reason: string}>}
 */
export async function probeGhCli(opts = {}) {
  const run = opts.run || defaultRunner(opts.timeoutMs ?? 5000);
  try {
    const result = await run('gh auth token');
    const code = result?.code ?? result?.exitCode ?? 1;
    const stdout = String(result?.stdout ?? '');
    if (code === 0 && stdout.trim().length > 0) {
      return { available: true, authenticated: true, reason: 'gh-cli' };
    }
    return {
      available: true,
      authenticated: false,
      reason: 'gh on PATH but not authenticated (gh auth login)',
    };
  } catch (err) {
    return { available: false, authenticated: false, reason: `gh not usable: ${err?.code || err?.message || 'spawn failed'}` };
  }
}

function defaultRunner(timeoutMs) {
  return async (cmd) => {
    const { spawn } = await import('node:child_process');
    return await new Promise((resolve, reject) => {
      const child = spawn(cmd, { shell: true });
      let stdout = '';
      let settled = false;
      const timer = setTimeout(() => {
        if (settled) return;
        settled = true;
        child.kill();
        reject(new LadderError('provider-timeout', `${cmd} timed out`, { cmd }));
      }, timeoutMs);
      child.stdout?.on('data', (d) => {
        stdout += String(d);
      });
      child.on('error', (err) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        reject(err);
      });
      child.on('close', (code) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        resolve({ code, stdout });
      });
    });
  };
}

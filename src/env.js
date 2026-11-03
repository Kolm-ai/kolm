import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);

export const DEV_RECEIPT_SECRET = 'ks_receipt_dev_secret_change_in_prod';

// W481 - known-public HMAC secret used by in-repo marketplace seed artifacts
// (e.g. public/registry-pack/qwen-distill-classifier.kolm). This is NOT a
// trust gate - Ed25519 (signed over the canonical receipt body) and Sigstore
// (transparency log) are the trust gates. The fixture secret only seals the
// HMAC integrity layer so verifiers on any host can re-derive the same hex
// without requiring env setup. Tampering with artifact bytes still breaks
// manifest_hash / artifact_hash / chain step inputs and so still breaks
// every signature down the chain regardless of HMAC secret publicity.
export const MARKETPLACE_FIXTURE_SECRET = 'kolm-public-fixture-v0-1-0';

// W481 - list of secrets a verifier may try in order. The runtime-effective
// secret comes first (the user's local_receipt_secret, RECIPE_RECEIPT_SECRET
// env, or DEV_RECEIPT_SECRET dev fallback); the marketplace fixture secret is
// appended so in-repo seed artifacts verify on any fresh checkout. Callers
// that need to walk multiple verification keys (binder audit chain, provenance
// credential) use this instead of a single effectiveReceiptSecret() call.
export function verificationSecrets({ includeLegacyArtifactSecret = false } = {}) {
  const seen = new Set();
  const out = [];
  const primary = effectiveReceiptSecret({ includeLegacyArtifactSecret });
  if (primary && !seen.has(primary)) { seen.add(primary); out.push(primary); }
  if (!seen.has(MARKETPLACE_FIXTURE_SECRET)) {
    seen.add(MARKETPLACE_FIXTURE_SECRET);
    out.push(MARKETPLACE_FIXTURE_SECRET);
  }
  // In dev mode, the explicit DEV_RECEIPT_SECRET path is already returned by
  // effectiveReceiptSecret when no env is set - but if the user has set a
  // RECIPE_RECEIPT_SECRET env, DEV_RECEIPT_SECRET is no longer primary. Append
  // it so historic in-tree dev-secret artifacts still verify regardless of
  // current env. In production-like hosts, DEV_RECEIPT_SECRET is intentionally
  // omitted to avoid silently accepting dev-signed artifacts.
  if (!isProductionRuntime() && !seen.has(DEV_RECEIPT_SECRET)) {
    seen.add(DEV_RECEIPT_SECRET);
    out.push(DEV_RECEIPT_SECRET);
  }
  return out;
}

// In production-like hosts (Vercel/AWS), KOLM_ARTIFACT_DIR and
// KOLM_DATA_DIR may be unset. /ready used to fail-closed 503 in that case.
// Instead, resolve sane writable defaults under os.tmpdir() and create them
// on demand so /ready is green out of the box. Operators can still override
// with explicit env vars when they wire durable storage.
function resolveDefaultDir(envKey, suffix) {
  const explicit = process.env[envKey];
  if (explicit && explicit.trim()) return explicit.trim();
  return path.join(os.tmpdir(), suffix);
}
function ensureDirSync(dir) {
  if (!dir) return false;
  try {
    fs.mkdirSync(dir, { recursive: true });
    return true;
  } catch {
    return false;
  }
}
export function resolveArtifactDir() {
  const dir = resolveDefaultDir('KOLM_ARTIFACT_DIR', 'kolm-artifacts');
  ensureDirSync(dir);
  return dir;
}
export function resolveDataDir() {
  const dir = resolveDefaultDir('KOLM_DATA_DIR', 'kolm-data');
  ensureDirSync(dir);
  return dir;
}

// Production-like hosts: explicit NODE_ENV, Vercel, or any AWS runtime
// (ECS/Fargate sets AWS_EXECUTION_ENV and ECS_CONTAINER_METADATA_URI_V4,
// Lambda sets AWS_LAMBDA_FUNCTION_NAME). Railway is no longer a target.
export function isProductionRuntime() {
  return process.env.NODE_ENV === 'production' ||
    !!process.env.VERCEL ||
    !!process.env.AWS_EXECUTION_ENV ||
    !!process.env.ECS_CONTAINER_METADATA_URI_V4 ||
    !!process.env.AWS_LAMBDA_FUNCTION_NAME;
}

export function effectiveReceiptSecret({ includeLegacyArtifactSecret = false } = {}) {
  const secret = process.env.RECIPE_RECEIPT_SECRET ||
    (includeLegacyArtifactSecret ? process.env.KOLM_ARTIFACT_SECRET : '');
  if (secret) {
    if (isProductionRuntime() && !receiptSecretLooksProductionSafe(secret)) return null;
    return secret;
  }
  return isProductionRuntime() ? null : DEV_RECEIPT_SECRET;
}

// Per-tenant receipt secret. If the tenant carries its own receipt_secret +
// receipt_key_id, return those. Otherwise fall back to the global secret with
// the implicit key_id "global". Callers should persist the key_id alongside
// every signed row so future verification can dispatch on it after a tenant
// rotates their key.
//
// Schema (tenants table): tenant.receipt_secret (string), tenant.receipt_key_id
// (string, e.g. "tk_<tenant>_<short-hex>"), tenant.receipt_rotated_at (iso).
// Rotation appends to tenant.previous_receipt_secrets[] so receipts signed
// with the previous key still verify until the operator chooses to drop them.
export function effectiveReceiptSecretForTenant(tenant, { includeLegacyArtifactSecret = false } = {}) {
  if (tenant && typeof tenant.receipt_secret === 'string' && tenant.receipt_secret.length >= 32) {
    return { secret: tenant.receipt_secret, key_id: tenant.receipt_key_id || `tk_${tenant.id || 'unknown'}_1` };
  }
  const globalSecret = effectiveReceiptSecret({ includeLegacyArtifactSecret });
  return { secret: globalSecret, key_id: 'global' };
}

// All known verification keys for a tenant - current + previous (for rotation).
// Verifiers walk this list and accept the first match.
export function tenantReceiptVerificationKeys(tenant) {
  const out = [];
  if (tenant && typeof tenant.receipt_secret === 'string' && tenant.receipt_secret.length >= 32) {
    out.push({ secret: tenant.receipt_secret, key_id: tenant.receipt_key_id || `tk_${tenant.id || 'unknown'}_1` });
  }
  if (tenant && Array.isArray(tenant.previous_receipt_secrets)) {
    for (const prev of tenant.previous_receipt_secrets) {
      if (prev && typeof prev.secret === 'string' && prev.secret.length >= 32) {
        out.push({ secret: prev.secret, key_id: prev.key_id || 'previous' });
      }
    }
  }
  const globalSecret = effectiveReceiptSecret({ includeLegacyArtifactSecret: true });
  if (globalSecret) out.push({ secret: globalSecret, key_id: 'global' });
  return out;
}

export function runtimeReadiness() {
  const productionLike = isProductionRuntime();
  const receiptSecret = process.env.RECIPE_RECEIPT_SECRET || '';
  const receiptSecretStrong = receiptSecretLooksProductionSafe(receiptSecret);
  const receiptSecretOk = !productionLike ||
    (!!receiptSecret && receiptSecretStrong);
  // Auto-bootstrap writable defaults so /ready does not fail-closed in a
  // production host where the operator hasn't pre-set KOLM_*_DIR. Explicit
  // env vars still win; this just removes the cold-start 503.
  const dataDir = resolveDataDir();
  const artifactDir = resolveArtifactDir();
  const storeDriver = configuredStoreDriver();
  const dataDirOk = !productionLike || directoryWritable(dataDir);
  const artifactDirOk = !productionLike || directoryWritable(artifactDir);
  const sqliteDir = path.dirname(path.resolve(process.env.KOLM_DB_PATH || path.join(dataDir || '.', 'kolm.sqlite')));
  const sqliteAvailable = storeDriver !== 'sqlite' || nodeSqliteAvailable();
  const sqlitePathOk = storeDriver !== 'sqlite' || directoryWritable(sqliteDir);
  const storeDriverOk = !productionLike ||
    (storeDriver === 'sqlite' && sqliteAvailable && sqlitePathOk) ||
    process.env.KOLM_ALLOW_JSON_STORE === 'true';

  const checks = [
    {
      name: 'receipt_secret',
      ok: receiptSecretOk,
      required: productionLike,
      public: 'receipt signing secret configured',
      hint: productionLike
        ? 'set RECIPE_RECEIPT_SECRET to a stable production-only value of at least 32 characters'
        : 'development uses an in-process fallback receipt secret',
    },
    {
      name: 'admin_key',
      ok: !productionLike || !!process.env.ADMIN_KEY,
      required: false,
      public: 'admin key configured',
      hint: productionLike
        ? 'set ADMIN_KEY if staff-only admin endpoints are needed'
        : 'development may use the fallback admin key',
    },
    {
      name: 'model_provider',
      // A frontier teacher is configured via a direct Anthropic key OR fal
      // (fal-ai/any-llm serves Claude). Either powers real LLM synthesis.
      ok: !!(process.env.ANTHROPIC_API_KEY || process.env.FAL_KEY || process.env.KOLM_FAL_TOKEN),
      required: false,
      public: 'frontier model provider configured',
      hint: 'set ANTHROPIC_API_KEY or FAL_KEY (fal-ai/any-llm serves Claude) for LLM-authored synthesis + cloud teacher calls',
    },
    {
      name: 'store_driver',
      ok: storeDriverOk,
      required: productionLike,
      public: 'database-backed store configured',
      hint: productionLike
        ? 'set KOLM_STORE_DRIVER=sqlite and KOLM_DB_PATH, or explicitly set KOLM_ALLOW_JSON_STORE=true for a temporary single-node deployment'
        : `using ${storeDriver} store driver`,
    },
    {
      name: 'data_dir',
      ok: dataDirOk,
      required: productionLike,
      public: 'data directory writable',
      hint: productionLike
        ? (process.env.KOLM_DATA_DIR ? `KOLM_DATA_DIR=${process.env.KOLM_DATA_DIR}` : `auto-bootstrapped under ${dataDir} (override with KOLM_DATA_DIR for durable storage)`)
        : (process.env.KOLM_DATA_DIR ? 'KOLM_DATA_DIR is set' : 'using the default ./data directory'),
    },
    {
      name: 'artifact_dir',
      ok: artifactDirOk,
      required: productionLike,
      public: 'artifact directory writable',
      hint: productionLike
        ? (process.env.KOLM_ARTIFACT_DIR ? `KOLM_ARTIFACT_DIR=${process.env.KOLM_ARTIFACT_DIR}` : `auto-bootstrapped under ${artifactDir} (override with KOLM_ARTIFACT_DIR for durable storage)`)
        : (process.env.KOLM_ARTIFACT_DIR ? 'KOLM_ARTIFACT_DIR is set' : 'using temporary artifact output by default'),
    },
  ];

  const blocking = checks.filter(check => check.required && !check.ok);
  return {
    status: blocking.length ? 'not_ready' : 'ready',
    production_like: productionLike,
    checks,
  };
}

function configuredStoreDriver() {
  if (process.env.KOLM_STORE_DRIVER) {
    return process.env.KOLM_STORE_DRIVER.toLowerCase();
  }
  // Mirror src/store.js auto-detection: in production-like environments
  // SQLite is the default whenever node:sqlite is available.
  if (isProductionRuntime() && nodeSqliteAvailable()) return 'sqlite';
  return 'json';
}

function nodeSqliteAvailable() {
  try {
    require('node:sqlite');
    return true;
  } catch {
    return false;
  }
}

function directoryWritable(dir) {
  if (!dir) return false;
  try {
    const stat = fs.statSync(dir);
    if (!stat.isDirectory()) return false;
    fs.accessSync(dir, fs.constants.R_OK | fs.constants.W_OK);
    return true;
  } catch {
    return false;
  }
}

function receiptSecretLooksProductionSafe(secret) {
  return typeof secret === 'string' &&
    secret.length >= 32 &&
    secret !== DEV_RECEIPT_SECRET &&
    !/^ks_receipt_(live|test|dev|change)/i.test(secret);
}

// WC07 - type-safe env readers. Audit found 10 sites where bare
// `process.env.FOO` reads silently corrupt:
//   * `!!process.env.KOLM_LOCAL_DAEMON` returns TRUE for `KOLM_LOCAL_DAEMON=false`
//     because non-empty strings are truthy. So `=false` does the OPPOSITE of
//     what every operator expects.
//   * `process.env.KOLM_SIGNING_KEY` with no fallback signs with the string
//     `"undefined"` when unset, silently producing forgeable signatures.
// envBool / envSecret are the two helpers all such reads should route through.

// True iff env var is set to a truthy string. Treats '0', 'false', 'no', 'off'
// (case-insensitive) as FALSE. Treats unset/empty as the provided fallback
// (default: false). Unrecognized non-empty strings (e.g. 'maybe') return the
// fallback rather than silently coercing - this prevents 'KOLM_FOO=disabled'
// from being read as `true` just because the string is non-empty.
export function envBool(name, fallback = false) {
  const v = process.env[name];
  if (v === undefined || v === null || v === '') return fallback;
  const s = String(v).trim().toLowerCase();
  if (s === '0' || s === 'false' || s === 'no' || s === 'off') return false;
  if (s === '1' || s === 'true' || s === 'yes' || s === 'on') return true;
  return fallback;
}

// Returns env var value iff it's non-empty after trim, else null. NEVER
// returns ''. Caller MUST handle null (e.g. throw / disable feature / log).
// Use this for SECRETS and other "must be present and meaningful" reads - 
// the bare `process.env.FOO || ''` pattern lets `FOO=""` flow through as a
// "configured" value, which then collides with omitted-header empty-string
// comparisons in constant-time auth paths.
export function envSecret(name) {
  const v = process.env[name];
  if (v === undefined || v === null) return null;
  const s = String(v).trim();
  return s.length > 0 ? s : null;
}

// GW - documented operator env catalog. A single machine-readable registry of
// the optional env vars the gateway + trainer + webhook surfaces read, so an
// operator (and the /v1/diagnostics surface) has one place to discover them
// instead of grepping the source. This is documentation-as-data: it does NOT set
// defaults (each call site still owns its own default), it only describes them.
// ASCII-only by repo convention.
export const KNOWN_ENV_DOCS = Object.freeze({
  // --- auth / credential lifecycle ---
  KOLM_KEY_LAST_USED_FLUSH_MS: 'cadence (ms) for the scoped-key last_used_at flush interval (default 30000)',
  KOLM_KEY_LAST_USED_TRACKING: 'set 1 to track scoped-key last_used_at (opt-in; off by default for hot-path cost)',
  KOLM_MAGICLINK_GC_MS: 'cadence (ms) for the magic-link dead-token GC sweep (default 3600000 = hourly)',
  KOLM_MAGICLINK_RETENTION_DAYS: 'audit-retention window (days) before a dead magic-link row is GC\'d (default 7)',
  KOLM_MAGICLINK_IP_LIMIT: 'magic-link start/recover per-IP cap per 15min (default 5)',
  KOLM_MAGICLINK_EMAIL_LIMIT: 'magic-link start/recover per-email cap per hour (default 5)',
  KOLM_MAGICLINK_COOLDOWN_S: 'per-email cooldown (s) before a second magic link is minted (default 45; 0 disables)',
  KOLM_INVITE_TOKEN_LIMIT: 'team invite preview/accept per-IP cap per 15min (default 60)',
  // --- teams / store ---
  KOLM_ALLOW_NONTXN_TEAMS: "set 'true' to permit a teams-enabled prod deploy on the json core driver instead of hard-failing; leave unset in prod",
  // --- compute / DoS guards ---
  KOLM_COMPUTE_SPAWN_PER_MIN: 'per-tenant compute-job starts allowed per minute across auto-distill/from-captures/compile (default 12)',
  KOLM_MAX_CONCURRENT_DISTILL: 'override the plan-derived running distill/compile job ceiling (applies as the floor for every plan)',
  // --- backups / vault ---
  KOLM_BACKUP_INCLUDE_VAULT_KEY: "default on; set '0'/'false' to exclude the raw secrets-vault.key from co-located backup snapshots",
  // --- event store ---
  KOLM_EVENT_STORE_NO_AUTOCOMPACT: "set '1' to disable opportunistic JSONL compaction in the append hot path",
  // --- finalized-c6: eval / holdout / leakage / K-score overlays ---
  KOLM_KSCORE_CONFORMAL: "set 1 to arm the conformal-bounded ship-gate overlay (stricter-only; requires a calibration-residual pool; an undercalibrated pool fails closed to abstain)",
  KOLM_KSCORE_CONTAM_IMPACT: "set 1 to arm the contamination-impact gate overlay (stricter-only). Clamps the corrected accuracy to min(reported, clean) at the gate so it can only LOWER ships; emits a contamination_impact manifest block bound into the artifact hash. Off by default (output byte-identical)",
  KOLM_DISABLE_DISJOINTNESS_LEDGER: "set 1 to disable the default-on transitive cross-corpus holdout-disjointness ledger and fall back to the legacy per-pair row-hash overlap probe (the ledger is default-on whenever both a train-side and holdout-side corpus are present)",
  KOLM_EVAL_GATE_SIGNIFICANCE: "set 1 to route the eval promotion/regression gate through the significance-bounded, multiplicity-controlled path when the candidate/baseline carry per_case score vectors (BH/Holm + paired bootstrap + mSPRT). Falls back to the point-delta gate when no per_case vectors exist or the flag is unset",
  // --- distill ordering / sampling ---
  KOLM_DISTILL_CURRICULUM: 'ascending|descending|1 - activates curriculum ordering on the default distill path (off by default)',
  KOLM_DISTILL_IMPORTANCE: 'set 1 to activate importance-weighted sampling on the default distill path',
  // --- trainer opt-out seams (test/operator) ---
  KOLM_PREFERENCE_NO_TRAINER: 'set 1 to force the durable no-tool path for preference training',
  KOLM_ONPOLICY_NO_TRAINER: 'set 1 to force the no-trainer path for white-box on-policy (GKD)',
  KOLM_ONPOLICY_TEACHER: 'local teacher model path/id for white-box GKD (required for the in_repo on-policy path)',
  KOLM_SPECDECODE_NO_TRAINER: 'set 1 to force the no-trainer path for the speculative-decoding draft-head trainer',
  KOLM_PREFERENCE_HANDROLLED: 'set 1 to force the hand-rolled in-repo preference loop over trl',
  KOLM_GKD_HANDROLLED: 'set 1 to force the hand-rolled in-repo GKD loop over trl',
  // --- spec-decode backends (OpenAI-compatible) ---
  KOLM_SPEC_DECODE_BACKEND: 'selects the speculative-decoding backend (llama|vllm|sglang|tgi) for acceleratedChatCompletion',
  KOLM_LLAMA_DRAFT_URL: 'llama.cpp draft (student) OpenAI-compatible base URL',
  KOLM_LLAMA_TARGET_URL: 'llama.cpp target (teacher) OpenAI-compatible base URL',
  KOLM_LLAMA_DRAFT_MODEL: 'optional llama.cpp draft model id',
  KOLM_LLAMA_TARGET_MODEL: 'optional llama.cpp target model id',
  KOLM_LLAMA_API_KEY: 'optional bearer for both llama-server endpoints',
  KOLM_LLAMA_TIMEOUT_MS: 'optional per-call timeout for llama-server (default 60000)',
  KOLM_VLLM_URL: 'vLLM OpenAI-compatible spec-decode endpoint',
  KOLM_SGLANG_URL: 'SGLang OpenAI-compatible spec-decode endpoint',
  KOLM_TGI_URL: 'TGI OpenAI-compatible spec-decode endpoint',
  KOLM_SGLANG_API_KEY: 'optional bearer for the SGLang endpoint',
  KOLM_TGI_API_KEY: 'optional bearer for the TGI endpoint',
  KOLM_SPEC_DECODE_TIMEOUT_MS: 'optional timeout (ms) for the vllm/sglang/tgi spec-decode bridge',
  KOLM_SPEC_EAGLE_TOPK: 'resolved EAGLE tree top-k sidecar for serve runtime visibility; SGLang enforces directly, vLLM reports only until upstream exposes supported keys',
  KOLM_SPEC_NUM_STEPS: 'resolved EAGLE tree step count sidecar for serve runtime visibility; SGLang enforces directly, vLLM reports only until upstream exposes supported keys',
  KOLM_SPEC_NUM_DRAFT_TOKENS: 'resolved EAGLE draft-token budget sidecar for serve runtime visibility; SGLang enforces directly, vLLM reports only until upstream exposes supported keys',
  // --- quantization / compile ---
  KOLM_QUANT_OPTIMIZERS: 'opt-in flag for installing workers/quantize/requirements-optimizers.txt (aqlm/quip/qat pinned optimizers)',
  KOLM_COMPILE_STREAM_DEMO: "set 1 to re-export the demo-only fabricated-metric compile stream for UI prototyping; UNSET in all production/CI",
  KOLM_ED25519_DISABLE: 'set 1 to disable the compile-pipeline Ed25519 sidecar (ships HMAC-only); see also KOLM_ED25519_PRIVATE_KEY / _PATH / _KEY_STORE',
  // --- webhooks ---
  KOLM_WEBHOOKS_ALLOW_LOCAL: 'set 1 to permit webhook delivery to private/loopback URLs (dev/test escape hatch; never set in prod)',
});

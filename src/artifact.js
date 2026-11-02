// .kolm artifact packager.
//
// A `.kolm` is a signed zip containing:
//   manifest.json     - task descriptor, hashes, training stats, tier
//   recipes.json      - deterministic recipe pack (executed in a vm sandbox)
//   evals.json        - eval cases that ship inside the artifact
//   model.gguf        - base-model pointer record. v0.1 is recipe-tier so this
//                       is metadata only; the LoRA tier will resolve it to
//                       real weights at first launch.
//   lora.bin          - artifact-bound binary slot. v0.1 carries an optional
//                       behaviour pack here (KOLMPACK\x01 magic + length-
//                       prefixed UTF-8 JSON body - patterns, lookup tables,
//                       rule packs that recipes call into via `lib.pack`).
//                       The LoRA tier (v0.2+) will swap this for a real
//                       weight delta. Empty buffer when no pack is supplied.
//   index.sqlite-vec  - artifact-bound lookup slot. v0.1 carries an optional
//                       JSON lookup index (KOLMIDX\x01 magic + length-prefixed
//                       UTF-8 JSON body - keyword→recipe maps, embedded
//                       lookup tables that recipes call via `lib.index`).
//                       The retrieval tier (v0.3+) will swap this for a real
//                       sqlite-vec database. Empty buffer when no index supplied.
//   signature.sig     - HMAC chain bound to the artifact receipt
//   receipt.json      - 5-step HMAC chain, body sig, anchor list
//
// Tenant-runtime customisation: callers of runArtifact can supply a `params`
// object that recipes read via `lib.params`. The artifact does NOT embed
// tenant data - params are passed at run time, never re-signed, never
// persisted by the runtime. This lets any buyer customise an artifact for
// their use case (extra patterns, vertical-specific rules, allowlists) while
// the signed artifact stays the same byte-exact bundle the issuer published.

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import archiver from 'archiver';
import { effectiveReceiptSecret, isProductionRuntime, verificationSecrets } from './env.js';
import { cidFromManifestHashes } from './cid.js';
import { buildArtifactCredential } from './provenance.js';
import { validateCapability, validateLineage } from './artifact-lineage.js';
import { validateExportBlock, EXPORT_SPEC_VERSION } from './export-provenance.js';
// R-1 - Runtime passport. Per (runtime, target_id) capability fingerprint
// rides inside the manifest as `runtime_passports: []`. Empty array when no
// export targets were probed; otherwise one row per format that ExportForge
// produced (status='tested' if a real probe ran, 'estimated' if the row was
// synthesized at compile time, 'unsupported' for incompatible combinations).
// validatePassports throws on schema violations so a hand-rolled bad row is
// caught at build time, not at first /v1/inspect call.
import { validatePassports as validateRuntimePassports, RUNTIME_PASSPORT_SCHEMA_VERSION } from './runtime-passport.js';
// R-5 - Evidence DAG. The provenance graph that explains where the artifact's
// inputs (captures, evals, teacher rollouts, signature events, policy gates,
// rights checks) came from. Validated at build time so a malformed or cyclic
// graph is caught before the receipt is signed. NOT bound into
// artifact_hash_input - same operational-fingerprint pattern as
// runtime_passports (R-1): a tenant can legitimately re-walk the graph and
// emit a new derived_from edge after an eval re-run without invalidating the
// receipt chain. The artifact's actual bytes (recipes, weights, evals) are
// anchored in artifact_hash; the DAG explains where those bytes came from but
// does not gain authority over them.
import { validateDagInput as validateEvidenceDagInput, toJSON as evidenceDagToJSON, buildDag as buildEvidenceDag, EVIDENCE_DAG_SCHEMA_VERSION } from './evidence-dag.js';
import { validateMoeBlock, MOE_SPEC_VERSION } from './moe-provenance.js';
import { validatePretokenizeBlock, PRETOKENIZE_SPEC_VERSION } from './pretokenize-provenance.js';
import { validateExternalHoldoutBlock, EXTERNAL_HOLDOUT_SPEC_VERSION } from './external-holdout.js';
import { validateTenantShadowBlock, TENANT_SHADOW_SPEC_VERSION } from './tenant-holdout.js';
import { validateAuditorAttestationBlock, AUDITOR_ATTESTATION_SPEC_VERSION } from './auditor-attestation.js';
import { validateSupersessionBlock, validateDriftReport, buildSupersessionBlock, buildDriftReport, SUPERSESSION_SPEC_VERSION, DRIFT_REPORT_SPEC_VERSION } from './drift-supersession.js';
import { RECIPE_CLASSES, validateRecipeClass, rollupArtifactClass, validateArtifactClass, CLASS_DESCRIPTIONS, RECIPE_SOURCE_TYPES, inferSourceType, validateRecipeSourceType } from './recipe-class.js';
import { hashIr } from './workflow-ir.js';
import { computeKScore as computeKScoreFromKscoreModule } from './kscore.js';
// finalized-c1 — conformal-bounded ship gate overlay (opt-in, KOLM_KSCORE_CONFORMAL=1).
// Stricter-only + fail-closed: can never flip ships=false -> true.
import { conformalBoundedGate } from './kscore-gate-harness.js';
import { verifyAttestation, manifestBlock as ccManifestBlock, STATES as CC_STATES } from './confidential-compute.js';
import { loadOrCreateDefaultSigner as loadEd25519DefaultSigner, buildSignatureBlock as buildEd25519Block } from './ed25519.js';
import {
  MODEL_WEIGHT_ARTIFACT_MANIFEST_FILENAME,
  buildModelWeightArtifactManifest,
  hashModelWeightArtifactManifest,
  serializeModelWeightArtifactManifest,
  signModelWeightArtifactManifest,
  verifyModelWeightArtifactManifest,
} from './model-weights-manifest.js';
// Model-signing sidecars (model-signing-standards). emitArtifactAttestation
// writes a signed SLSA Provenance v1 DSSE envelope; toOmsArtifactManifest writes
// an OpenSSF Model-Signing (OMS) file manifest. Both seal over the ACTUAL
// bundled member bytes and are EMITTED AFTER artifact_hash (excluded from it,
// like signature.sig), gated behind the Ed25519 signer.
import { emitArtifactAttestation, inferModelDocumentationRole } from './intoto-slsa.js';
import { toOmsArtifactManifest } from './intoto-receipt.js';
import { buildSigstoreBundle, isDisabled as isSigstoreDisabled, attestArtifactWithRekor, rekorUrl as sigstoreRekorUrl } from './sigstore.js';
import { canonicalizeOutputSchemaSpec, validateOutputSchemaSpec, OUTPUT_SCHEMA_VERSION } from './output-schema.js';
// W736 - Guardrail Compilation. Hard-constraint rules ride INSIDE the .kolm
// manifest (NOT as training signal) so brand-safety policy survives every
// runtime invocation + every re-distill. hashGuardrails feeds the
// conditional `guardrails_hash` slot inside artifact_hash_input so any
// post-build tamper of the rules breaks the receipt chain; the W460
// byte-stability pattern (absent vs null vs []==null collapse) is
// preserved so pre-W736 artifacts rebuilt without a guardrails block
// remain byte-identical to their old artifact_hash.
import { hashGuardrails as hashGuardrailsW736, validateGuardrailRules as validateGuardrailRulesW736 } from './guardrails.js';
// W786 - Carbon footprint / sustainability badge. badgeFor produces a small
// stable structure for the manifest's `sustainability_badge` field. Stamped
// POST artifact_hash via the W460 conditional-spread pattern (badge absent
// → key not present → pre-W786 artifacts remain byte-identical when
// rebuilt). The badge is hygiene metadata, NOT provenance - a tamperer
// flipping it does NOT break receipt.json (badge is not bound into
// artifact_hash_input).
import { badgeFor as badgeForW786, CARBON_VERSION as CARBON_VERSION_W786 } from './carbon-estimator.js';

const ARTIFACT_SPEC = 'kolm-1';
const PACK_MAGIC = 'KOLMPACK\x01';
const INDEX_MAGIC = 'KOLMIDX\x01';

// Artifact classes - see Wave 144 user redirect.
//   'rule' - deterministic JS/rule artifact. No model.gguf / lora.bin
//                      / index.sqlite-vec padding when no real pack/index is
//                      supplied. This is the only class that ships today.
//   'compiled_rule' - generated C/Rust/WASM artifact from a constrained rule
//                      AST (Wave F). Adds target/target_source_hash/target_binary_hash
//                      manifest fields.
//   'distilled_model'- real teacher->student model artifact with LoRA/quantization
//                      metadata + real weights (Wave J/K). Re-introduces
//                      model.gguf / lora.bin slots with real bytes.
// Wave 151 - RECIPE_CLASSES is the new canonical list (adds 'synthesized_rule').
// ARTIFACT_CLASSES stays as the historical export for backward compat; it now
// re-exports the full RECIPE_CLASSES list. Validators import RECIPE_CLASSES.
export const ARTIFACT_CLASSES = RECIPE_CLASSES;
const EMPTY_BUF = Buffer.alloc(0);
const EMPTY_SHA = crypto.createHash('sha256').update(EMPTY_BUF).digest('hex');

// Encode an optional behaviour pack for the lora.bin slot. Returns an empty
// Buffer when no pack is supplied so v0.1 artifacts that don't ship one are
// byte-stable with prior releases.
function encodePack(pack) {
  if (!pack || (typeof pack === 'object' && Object.keys(pack).length === 0)) return Buffer.alloc(0);
  const body = Buffer.from(JSON.stringify(pack), 'utf8');
  const head = Buffer.from(PACK_MAGIC, 'binary');
  const len = Buffer.alloc(4);
  len.writeUInt32LE(body.length, 0);
  return Buffer.concat([head, len, body]);
}

function encodeIndex(index) {
  if (!index || (typeof index === 'object' && Object.keys(index).length === 0)) return Buffer.alloc(0);
  const body = Buffer.from(JSON.stringify(index), 'utf8');
  const head = Buffer.from(INDEX_MAGIC, 'binary');
  const len = Buffer.alloc(4);
  len.writeUInt32LE(body.length, 0);
  return Buffer.concat([head, len, body]);
}

// Decode a pack or index buffer. Tolerates empty buffers (returns null) and
// throws on magic mismatch so a corrupt slot doesn't silently expose
// arbitrary bytes to recipes.
export function decodePack(buf) { return decodeContainer(buf, PACK_MAGIC); }
export function decodeIndex(buf) { return decodeContainer(buf, INDEX_MAGIC); }
function decodeContainer(buf, magic) {
  if (!buf || !buf.length) return null;
  if (buf.length < magic.length + 4) throw new Error('container too short');
  const head = buf.slice(0, magic.length).toString('binary');
  if (head !== magic) throw new Error(`container magic mismatch: expected ${JSON.stringify(magic)}`);
  const len = buf.readUInt32LE(magic.length);
  const body = buf.slice(magic.length + 4, magic.length + 4 + len);
  if (body.length !== len) throw new Error('container length mismatch');
  return JSON.parse(body.toString('utf8'));
}
// IMPORTANT: keep this in lock-step with router.js's RECEIPT_SECRET. The
// receipt the artifact builder seals here is verified by /v1/receipts/verify
// using that same secret - a mismatch produces "signature mismatch" + "chain
// hmac mismatch" failures even though both sides are byte-identical canonical
// JSON. The legacy KOLM_ARTIFACT_SECRET env name is still honoured for
// back-compat, but the default must match router.js's default.
let warnedMissingSignSecret = false;

function signSecret() {
  const secret = effectiveReceiptSecret({ includeLegacyArtifactSecret: true });
  if (secret) return secret;
  if (isProductionRuntime() && !warnedMissingSignSecret) {
    console.error('[artifact] WARNING: RECIPE_RECEIPT_SECRET not set - /v1/compile will 503. Set it in the API host env (SSM parameter).');
    warnedMissingSignSecret = true;
  }
  return null;
}

function requireSignSecret() {
  const secret = signSecret();
  if (secret) return secret;
  const e = new Error('cannot build .kolm: RECIPE_RECEIPT_SECRET not set on server');
  e.statusCode = 503;
  throw e;
}

function sha256(buf) {
  return crypto.createHash('sha256').update(buf).digest('hex');
}

function blake2b(buf) {
  return crypto.createHash('blake2b512').update(buf).digest('hex');
}

// Chunked sync hash for files that exceed Node's 2 GiB readFileSync limit
// (large GGUF exports). Same semantics as sha256(Buffer) but on a path.
function sha256File(absPath) {
  const h = crypto.createHash('sha256');
  const fd = fs.openSync(absPath, 'r');
  try {
    const CHUNK = 1024 * 1024;
    const buf = Buffer.alloc(CHUNK);
    while (true) {
      const read = fs.readSync(fd, buf, 0, CHUNK, null);
      if (read <= 0) break;
      h.update(buf.subarray(0, read));
    }
    return h.digest('hex');
  } finally {
    fs.closeSync(fd);
  }
}

function digestPair(buf) {
  return {
    sha256: sha256(buf),
    blake2b: blake2b(buf),
  };
}

function digestFilePair(absPath) {
  const sha = crypto.createHash('sha256');
  const b2 = crypto.createHash('blake2b512');
  const fd = fs.openSync(absPath, 'r');
  try {
    const CHUNK = 1024 * 1024;
    const buf = Buffer.alloc(CHUNK);
    while (true) {
      const read = fs.readSync(fd, buf, 0, CHUNK, null);
      if (read <= 0) break;
      const chunk = buf.subarray(0, read);
      sha.update(chunk);
      b2.update(chunk);
    }
    return { sha256: sha.digest('hex'), blake2b: b2.digest('hex') };
  } finally {
    fs.closeSync(fd);
  }
}

function modelWeightRuntimeTargets(runtimeTarget) {
  if (runtimeTarget === 'gguf') return ['llama.cpp', 'llama.cpp-webgpu'];
  if (runtimeTarget === 'onnx') return ['onnxruntime-node', 'onnxruntime-web'];
  if (runtimeTarget === 'wasm') return ['webgpu'];
  if (runtimeTarget === 'native') return ['native'];
  return [];
}

// W367 - recipe.bundle.mjs builder. Wraps each rule recipe's source body
// (already a pure `function generate(input, lib){...}` per the sandbox guard)
// in an isolating IIFE so multiple recipes can coexist in one file without
// shadowing each other's `generate` declaration. Exports a single default
// dispatcher that walks the recipes in order and returns the first output that
// does not throw - matching runJsTarget's semantics in artifact-runner.js.
//
// Heavy-deps rule: this builder uses ONLY string concatenation. No esbuild,
// no rollup, no parser. The rule-class recipe shape (single top-level
// `function generate(input, lib){...}`, no imports/requires/process by
// sandbox contract) makes a real bundler unnecessary.
//
// Output shape: a regular ESM module. Default export is
//   async function run(input, opts) -> { output, recipe_id, recipe_name, latency_us }
// The opts shape is { params, pack, index } - matching the lib slots that
// artifact-runner.js exposes via the sandbox `lib` global. Callers that load
// this file directly (host runtime, edge worker, mobile runtime) get the same
// contract as the in-process JS runner without needing the .kolm machinery.
export function buildRecipeBundleMjs(recipes, { spec, job_id, generated_at } = {}) {
  const generatedAt = generated_at || new Date().toISOString();
  const headerLines = [
    '// recipe.bundle.mjs - self-contained ESM bundle generated by src/artifact.js',
    `// spec: ${spec || ARTIFACT_SPEC}`,
    `// job_id: ${job_id || 'unknown'}`,
    `// generated_at: ${generatedAt}`,
    '//',
    '// Default export is an async dispatcher: run(input, { params, pack, index })',
    '// that walks the bundled recipes in declaration order and returns the first',
    '// one that does not throw. Same semantics as src/artifact-runner.js',
    '// runJsTarget - the artifact runs without needing the kolm runtime.',
    '',
  ];
  const recipeLoaders = recipes.map((r, idx) => {
    const id = JSON.stringify(r.id || `recipe_${idx + 1}`);
    const name = JSON.stringify(r.name || r.id || `recipe_${idx + 1}`);
    // Wrap each recipe in an IIFE that returns its `generate` binding. This
    // isolates declarations between recipes (multiple recipes can each declare
    // `function generate(...)` in their own scope without collision) and
    // mirrors what compileJs does in vm-context form.
    const body = String(r.source);
    return [
      `function __loadRecipe_${idx}() {`,
      `  ${body}`,
      `  return generate;`,
      `}`,
      `const __recipes_${idx} = { id: ${id}, name: ${name}, generate: __loadRecipe_${idx}() };`,
    ].join('\n');
  });
  const registryLines = [
    'const RECIPES = [',
    ...recipes.map((_r, idx) => `  __recipes_${idx},`),
    '];',
    '',
    'export { RECIPES };',
  ];
  const dispatcher = [
    'export default async function run(input, opts = {}) {',
    '  const params = opts && opts.params != null ? opts.params : null;',
    '  const pack = opts && opts.pack != null ? opts.pack : null;',
    '  const index = opts && opts.index != null ? opts.index : null;',
    '  const lib = Object.freeze({ params, pack, index });',
    '  const tried = [];',
    '  let lastError = null;',
    '  const t0 = (typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now();',
    '  for (const r of RECIPES) {',
    '    try {',
    '      const output = r.generate(input, lib);',
    '      const t1 = (typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now();',
    '      return { output, recipe_id: r.id, recipe_name: r.name, latency_us: Math.round((t1 - t0) * 1000) };',
    '    } catch (e) {',
    '      tried.push({ id: r.id, error: String(e && e.message || e) });',
    '      lastError = e;',
    '    }',
    '  }',
    '  const err = new Error("no recipe handled the input. tried " + tried.length + (lastError ? "; last: " + (lastError.message || lastError) : ""));',
    '  err.code = "KOLM_E_NO_RECIPE_HANDLED";',
    '  err.tried = tried;',
    '  throw err;',
    '}',
    '',
  ];
  return [
    ...headerLines,
    ...recipeLoaders,
    '',
    ...registryLines,
    '',
    ...dispatcher,
  ].join('\n');
}

function canonicalJson(v) {
  if (v === null || typeof v !== 'object') return JSON.stringify(v);
  if (Array.isArray(v)) return '[' + v.map(canonicalJson).join(',') + ']';
  const k = Object.keys(v).sort();
  return '{' + k.map(x => JSON.stringify(x) + ':' + canonicalJson(v[x])).join(',') + '}';
}

function hasSourceDateEpochValue(v) {
  return v !== undefined && v !== null && String(v).trim() !== '';
}

function parseSourceDateEpoch(v) {
  const raw = String(v).trim();
  const asNumber = Number(raw);
  let ms;
  let epochSeconds;
  if (Number.isFinite(asNumber) && raw !== '') {
    epochSeconds = Math.trunc(asNumber);
    if (epochSeconds < 0) {
      throw new Error(`source_date_epoch must be >= 0 (got ${raw})`);
    }
    ms = epochSeconds * 1000;
  } else {
    ms = Date.parse(raw);
    if (!Number.isFinite(ms)) {
      throw new Error(`source_date_epoch must be unix seconds or an ISO timestamp (got ${raw})`);
    }
    epochSeconds = Math.trunc(ms / 1000);
  }
  const date = new Date(ms);
  const timestamp = date.toISOString();
  return { epoch: epochSeconds, timestamp, date };
}

function reproducibleBuildContext({ source_date_epoch, reproducible_build } = {}) {
  const envEpoch = hasSourceDateEpochValue(process.env.KOLM_SOURCE_DATE_EPOCH)
    ? process.env.KOLM_SOURCE_DATE_EPOCH
    : process.env.SOURCE_DATE_EPOCH;
  const chosen = hasSourceDateEpochValue(source_date_epoch) ? source_date_epoch : envEpoch;
  if (hasSourceDateEpochValue(chosen)) {
    return { enabled: true, ...parseSourceDateEpoch(chosen) };
  }
  if (reproducible_build === true || String(process.env.KOLM_REPRODUCIBLE_BUILD || '') === '1') {
    return { enabled: true, ...parseSourceDateEpoch(0) };
  }
  return { enabled: false, epoch: null, timestamp: null, date: null };
}

function deterministicReceiptId({ job_id, cid, artifact_hash, eval_set_hash, issued_at }) {
  return 'rcpt_' + sha256(canonicalJson({
    spec: 'kolm-receipt-id-v1',
    job_id,
    cid,
    artifact_hash,
    eval_set_hash,
    issued_at,
  })).slice(0, 32);
}

// Compute the K-score - the visible scoreboard for "smallest artifact that
// still passes the tests wins." Implements the documented formula at
// /k-score: K = 0.40·A + 0.15·S + 0.15·L + 0.15·C + 0.15·V, on [0..1].
// Ship gate is 0.85; below that, kolm compile fails closed.
//
// Raw axes (kept on the manifest for downstream tooling):
//   accuracy:          verifier pass-rate on the training positives [0..1]
//   coverage:          fraction of declared task surface handled [0..1]
//   p50_latency_us:    median run-time per call
//   cost_usd_per_call: marginal $ at run-time (0 for pure-recipe)
//   size_bytes:        seal-time probe zip size (the zip before the manifest
//                      embeds the K-score itself); typically 64-100 bytes
//                      less than the final on-disk size. Stored on the
//                      manifest so K-score is deterministically recomputable
//                      from artifact bytes alone.
//
// Each non-fractional axis (S, L, C) is normalized to [0..1] via a smooth
// curve calibrated so that typical recipe-mode artifacts (~10KB, ~100us, $0)
// score near 1, and pathological cases asymptote toward 0:
//   S = max(0, 1 - log2(max(size_kb, 1)) / 30)   // 10KB->0.89, 1MB->0.67, 1GB->0.33
//   L = 1 / (1 + p50_us / 100000)                // 100us->1.0, 100ms->0.50
//   C = 1 / (1 + cost_per_call * 1000)           // $0->1.0, $0.001->0.50
// A and V are already on [0..1].
// Wave 145 - delegates to src/kscore.js so V2 axes (R/F/E/Z/T) are available
// to any caller that supplies them. V1-only callers continue to receive a v1
// envelope (auto-detected by kscore.js when no V2 inputs are present). The
// gate (0.85) and v1 weights (0.40/0.15/0.15/0.15/0.15) are unchanged, so
// every artifact built before wave 145 verifies identically.
export function computeKScore(input) {
  return computeKScoreFromKscoreModule(input);
}

// Build the artifact payload (the parts that end up *inside* the zip).
// Returns a list of {filename, content} entries plus the manifest.
//
// New in v0.1: a receipt.json file ships alongside signature.sig. The
// receipt binds (artifact_hash, eval_set_hash, eval_score, judge_id) via an
// HMAC chain so any third party can re-verify offline without trusting the
// runtime that produced the artifact.
function normalizeLicense(license) {
  if (!license) {
    return {
      id: 'LicenseRef-kolm-default-1.0',
      name: 'kolm default artifact license (1.0)',
      url: 'https://kolm.ai/license#artifact-default-1-0',
      allows: ['inference', 'evaluation', 'redistribution-with-attribution'],
      requires: ['preserve-receipt', 'preserve-attribution'],
      forbids: [],
    };
  }
  if (typeof license === 'string') {
    return { id: license, name: license, url: null, allows: [], requires: [], forbids: [] };
  }
  return {
    id: String(license.id || license.spdx || 'LicenseRef-unknown'),
    name: license.name ? String(license.name) : String(license.id || 'unknown'),
    url: license.url ? String(license.url) : null,
    allows: Array.isArray(license.allows) ? license.allows.map(String) : [],
    requires: Array.isArray(license.requires) ? license.requires.map(String) : [],
    forbids: Array.isArray(license.forbids) ? license.forbids.map(String) : [],
  };
}

export function buildPayload({ job_id, task, base_model, recipes, lora_pointer, recall_namespace, training_stats, evals, k_score, judge_id, eval_score, tier, pack, index, target_device, train_device, license, artifact_class, seed_provenance, compiled_targets, capability, lineage, workflow_ir, attestation_report, confidential_compute, extra_files, export: exportInput, moe: moeInput, pretokenize: pretokenizeInput, external_holdout: externalHoldoutInput, tenant_shadow_corpus: tenantShadowInput, auditor_attestation: auditorAttestationInput, supersession: supersessionInput, drift_report: driftReportInput, contamination_impact: contaminationImpactInput, allow_below_gate, binaries, compiled_binary, native_skip_reasons, runtime_target, runtime_target_config, model_weights, entrypoint, daq_profile, mixed_precision_proof, importance_signal, calibration_provenance, sparsity_profile, kv_profile, output_schema, guardrails, parent_cid, region, runtime_passports, evidence_dag, speculative_decoding, prompt_cache, continuous_batching, source_date_epoch, reproducible_build, receipt_id: receiptIdInput }) {
  const secret = requireSignSecret();
  const repro = reproducibleBuildContext({ source_date_epoch, reproducible_build });
  const buildTimestamp = repro.enabled ? repro.timestamp : new Date().toISOString();
  let ed25519Signer = null;
  try {
    ed25519Signer = loadEd25519DefaultSigner();
  } catch (e) {
    console.error(`[artifact] WARNING: ed25519 signer load skipped: ${e.message}`);
  }
  // W252 - K-score ship gate is load-bearing. If a K-score is supplied AND
  // it says ships=false, the builder must refuse unless the caller explicitly
  // passes allow_below_gate=true (which gets stamped on the manifest so the
  // verifier and downstream procurement gates can flag it). Without this
  // check, the gate is a comment, not a contract.
  // finalized-c1 — OPT-IN conformal-bounded ship gate overlay (KOLM_KSCORE_CONFORMAL=1).
  // The kscore-gate-harness can ONLY make the gate STRICTER and FAILS CLOSED:
  // when a calibration-residual pool is supplied it requires the conformal LOWER
  // bound (not just the point estimate) to clear the gate; an undercalibrated /
  // missing pool yields 'abstain' which we treat as a NON-ship. This never flips
  // a ships=false to true, so the moat is preserved. The decision is stamped on
  // the k_score so the manifest/receipt records the bounded verdict.
  let ship_decision = null;
  if (k_score && String(process.env.KOLM_KSCORE_CONFORMAL || '') === '1') {
    try {
      const g = typeof k_score.gate === 'number' ? k_score.gate : 0.85;
      const cg = conformalBoundedGate({
        precomputed: { composite: k_score.composite },
        calibrationResiduals: Array.isArray(k_score.calibration_residuals) ? k_score.calibration_residuals : [],
        gate: g,
      });
      if (cg.ok) {
        ship_decision = {
          decision: cg.decision, point_ships: cg.point_ships,
          interval: cg.interval, n_cal: cg.n_cal,
          confidence_bounded: cg.confidence_bounded, version: cg.version,
        };
        // Stricter-only: if the point gate passed but the bounded gate does not
        // SHIP, downgrade ships to false (fail-closed) unless overridden.
        if (k_score.ships !== false && cg.decision !== 'ship') {
          k_score = { ...k_score, ships: false, conformal_abstain_reason: cg.decision };
        }
      }
    } catch (e) {
      // Fail-closed: a harness error must not silently ship. Downgrade to a block.
      ship_decision = { decision: 'abstain', error: String((e && e.message) || e) };
      if (k_score.ships !== false) k_score = { ...k_score, ships: false, conformal_abstain_reason: 'harness_error' };
    }
  }
  // finalized-c6 - OPT-IN contamination-impact gate overlay (KOLM_KSCORE_CONTAM_IMPACT=1).
  // Decomposes holdout accuracy into clean vs contaminated and recomputes a
  // contamination-corrected K-score. It is wired into the SAME stricter-only
  // overlay chain as the conformal gate: it can ONLY lower a ship to false,
  // NEVER widen.
  //
  // SAFETY CLAMP (load-bearing): the contamination-impact module's verify pass
  // flagged that its raw correctedA = accuracy_clean is NOT downward-only - when
  // contaminated rows score LOWER than clean rows, accuracy_clean > reported and
  // the correction is UPWARD, which could FLIP a non-shipping model to ship
  // (more lenient). To preserve the fail-closed moat regardless of that atom's
  // direction bug, we clamp the corrected accuracy to min(reported, clean) here
  // at the integration site and re-derive the gate decision from the CLAMPED
  // number. The overlay therefore can never raise the score, so the ship gate
  // only ever gets stricter. (Follow-up: fix the atom to clamp internally.)
  // Default OFF => byte-identical output. The block, when emitted, is bound into
  // the artifact hash via the contamination_impact manifest slot below.
  let contamination_impact_block = null;
  if (k_score && contaminationImpactInput && String(process.env.KOLM_KSCORE_CONTAM_IMPACT || '') === '1') {
    try {
      const ci = contaminationImpactInput;
      // The caller (compile/binder) supplies the already-built block OR the raw
      // inputs. We re-derive the clamped corrected composite defensively.
      const reportedA = (ci.decomposition && typeof ci.decomposition.accuracy_reported === 'number')
        ? ci.decomposition.accuracy_reported
        : (typeof k_score.accuracy === 'number' ? k_score.accuracy : null);
      const cleanA = (ci.decomposition && typeof ci.decomposition.accuracy_clean === 'number')
        ? ci.decomposition.accuracy_clean : null;
      // Genuinely downward-only: never let the corrected number exceed reported.
      const safeCorrectedA = (reportedA != null && cleanA != null)
        ? Math.min(reportedA, cleanA)
        : reportedA;
      // Re-derive a corrected composite via the same kscore module, clamped.
      let correctedComposite = null;
      let correctedShips = null;
      if (safeCorrectedA != null && typeof k_score.composite === 'number') {
        try {
          const kc = computeKScoreFromKscoreModule({
            accuracy: safeCorrectedA,
            holdout_accuracy: safeCorrectedA,
            calibration_disabled: true,
          });
          correctedComposite = typeof kc.composite === 'number' ? kc.composite : null;
          correctedShips = kc.ships === true;
        } catch { /* fall through: no recompute => no downgrade */ }
      }
      contamination_impact_block = {
        spec: 'contamination-impact-gate-v1',
        accuracy_reported: reportedA,
        accuracy_clean: cleanA,
        accuracy_corrected_clamped: safeCorrectedA,
        corrected_composite: correctedComposite,
        corrected_ships: correctedShips,
        // pass the source block hash through for the verifier to re-anchor.
        source_block_hash: (ci && typeof ci.content_hash === 'string') ? ci.content_hash : null,
        source_block: (ci && ci.spec) ? ci : null,
      };
      // Stricter-only: if the CLAMPED corrected gate does NOT ship, downgrade.
      if (k_score.ships !== false && correctedShips === false) {
        k_score = { ...k_score, ships: false, contamination_abstain_reason: 'corrected_below_gate' };
      }
    } catch (e) {
      // Fail-closed: a gate error must not silently ship when the operator
      // explicitly armed the contamination overlay.
      contamination_impact_block = { spec: 'contamination-impact-gate-v1', error: String((e && e.message) || e) };
      if (k_score && k_score.ships !== false) {
        k_score = { ...k_score, ships: false, contamination_abstain_reason: 'gate_error' };
      }
    }
  }
  if (k_score && k_score.ships === false && !allow_below_gate) {
    const composite = typeof k_score.composite === 'number' ? k_score.composite : 0;
    const gate = typeof k_score.gate === 'number' ? k_score.gate : 0.85;
    const why = k_score.conformal_abstain_reason ? ` (conformal gate: ${k_score.conformal_abstain_reason})`
      : (k_score.contamination_abstain_reason ? ` (contamination gate: ${k_score.contamination_abstain_reason})` : '');
    throw new Error(`k_score below ship gate: composite=${composite}, gate=${gate}${why}. ` +
      `Pass allow_below_gate=true to override; the manifest will record ship_gate_overridden=true.`);
  }
  // Default class is 'rule' - the floor. Wave 151 adds 'synthesized_rule' and
  // keeps 'compiled_rule' / 'distilled_model'. The class is load-bearing: the
  // verifier rejects any artifact whose class doesn't match what's in the zip.
  // Callers can pass artifact_class explicitly OR let the builder roll up from
  // the per-recipe classes (computed below from recipes[].class).
  const _class = artifact_class && ARTIFACT_CLASSES.includes(artifact_class) ? artifact_class : null;
  if (_class === 'compiled_rule' && !compiled_targets) {
    throw new Error('compiled_rule artifact requires compiled_targets (call spec-compile with artifact_class=compiled_rule)');
  }

  // Wave V - capability contract, lineage, workflow IR, and attestation report
  // are optional manifest blocks. Validation runs at build time so a malformed
  // block is caught here instead of at verify time. workflow_ir and
  // attestation_report ride along inside the zip as separate files so the
  // verifier can replay hashIr() / verifyAttestation() and confirm the claims.
  let capability_block = null;
  if (capability) {
    capability_block = validateCapability(capability);
  }
  let lineage_block = null;
  if (lineage) {
    lineage_block = validateLineage(lineage);
  }
  // Wave 146 - export block. The src/export-provenance.js bridge builds and
  // validates this; we re-validate here (cheap) so a caller that constructed
  // an export block by hand still gets schema-checked. The block's own short
  // hash is folded into artifact_hash_input below so any post-build mutation
  // breaks the receipt chain.
  let export_block = null;
  if (exportInput) {
    export_block = validateExportBlock(exportInput);
  }
  // R-1 - runtime_passports. Always an array; empty when no targets were
  // exported. Each entry is one (runtime, target_id) capability fingerprint
  // produced by ExportForge after every format export. Validation runs at
  // build time so a malformed row (unknown runtime, missing measurement on a
  // status='tested' row) is caught here, before the receipt is signed. NOT
  // bound into artifact_hash_input below: the passport is an OPERATIONAL
  // fingerprint, not provenance - a tenant who re-probes the same artifact
  // on a different host can update the array without breaking the receipt
  // chain. The export_block already binds the bytes of every target file
  // into artifact_hash, so the passport never has authority over the bytes.
  let _runtime_passports_canon = [];
  if (runtime_passports != null) {
    if (!Array.isArray(runtime_passports)) {
      throw new Error('runtime_passports must be array');
    }
    const v = validateRuntimePassports(runtime_passports);
    if (!v.ok) {
      throw new Error(`runtime_passports[${v.index}] invalid: ${v.reason}`);
    }
    _runtime_passports_canon = runtime_passports.slice();
  }
  // R-5 - evidence_dag. The provenance graph that explains where the
  // artifact's upstream inputs (captures, evals, teacher rollouts,
  // signature events, policy gates, rights checks) came from. Validated
  // here so a hand-rolled malformed graph (unknown kind, cycle, missing
  // endpoint) is caught at build time, not at first /v1/evidence/* call.
  // NOT bound into artifact_hash_input below: the DAG is OPERATIONAL
  // fingerprint, not provenance over bytes - a tenant who re-walks the
  // graph (e.g. to add a new validated_by edge after an eval re-run) can
  // update the field without breaking the receipt chain. The artifact's
  // actual bytes (recipes, weights, evals) are anchored in artifact_hash;
  // the DAG explains where those bytes came from but does not gain
  // authority over them. Conditional spread (per W460 byte-stability law)
  // keeps pre-R5 artifacts byte-identical when rebuilt without an
  // evidence_dag argument.
  let _evidence_dag_canon = null;
  if (evidence_dag != null) {
    const v = validateEvidenceDagInput(evidence_dag);
    if (!v.ok) {
      throw new Error(`evidence_dag invalid: ${v.reason}`);
    }
    _evidence_dag_canon = evidenceDagToJSON(buildEvidenceDag(evidence_dag));
  }
  // Wave 147 - moe block. Same pattern as export: src/moe-provenance.js
  // bridge builds + validates; we re-validate here so a hand-rolled block
  // still gets schema-checked. The block's short hash folds into
  // artifact_hash_input below.
  let moe_block = null;
  if (moeInput) {
    moe_block = validateMoeBlock(moeInput);
  }
  // W809 - output_schema spec. Validate up front so a bad spec is caught at
  // build time (not first runtime invocation). Absence / null / {} all collapse
  // to canon === null via canonicalizeOutputSchemaSpec, and the chain slot
  // below is keyed only when canon != null - pre-W809 artifacts remain
  // byte-identical when rebuilt. Validation throws on a bad spec, mirroring
  // the export/moe pattern above.
  let _output_schema_canon = null;
  if (output_schema != null) {
    const _osv = validateOutputSchemaSpec(output_schema);
    if (!_osv.ok) {
      throw new Error('invalid output_schema: ' + _osv.errors.join(','));
    }
    _output_schema_canon = canonicalizeOutputSchemaSpec(output_schema);
  }
  // W736 - guardrails are validated at build time so a bad rule is caught
  // here instead of at the first runtime hit. Absent / null / [] all
  // collapse to null via _guardrails_canon, mirroring the output_schema
  // collapse rule one block above. The byte-stability contract: pre-W736
  // artifacts rebuilt without a guardrails block MUST hash identically;
  // the conditional slot in artifact_hash_input below keys ONLY when
  // canon is non-null + non-empty, never unconditionally.
  let _guardrails_canon = null;
  if (Array.isArray(guardrails) && guardrails.length > 0) {
    const _gv = validateGuardrailRulesW736(guardrails);
    if (!_gv.ok) {
      throw new Error('invalid guardrails: ' + _gv.errors.map(e => `${e.path}=${e.error}`).join(','));
    }
    _guardrails_canon = guardrails;
  }
  // W739 - Model lineage tracking. parent_cid is OPTIONAL. Validated up
  // front so a malformed pointer is caught at build time instead of at the
  // first `kolm lineage <cid>` walk on a deployed artifact. The W460
  // byte-stability law applies: absent / null / empty-string all collapse
  // to null (the slot is omitted from artifact_hash_input below) so
  // pre-W739 artifacts rebuilt without a parent_cid remain byte-identical
  // to their original artifact_hash. A non-null parent_cid MUST be either
  // bare sha256-hex (64 lowercase hex) OR the canonical `cidv1:sha256:<hex>`
  // form emitted by our own manifest.cid field. We PRESERVE the caller's
  // verbatim form in the manifest + the hash slot so `B.parent_cid` round-
  // trips byte-identical to whatever cid the caller pinned (a downstream
  // walker shouldn't have to know about both hex + cidv1 spellings to
  // string-compare lineage rows).
  let _parent_cid_canon = null;
  if (parent_cid !== null && parent_cid !== undefined && parent_cid !== '') {
    if (typeof parent_cid !== 'string') {
      throw new Error(
        'invalid parent_cid: must be a string (sha256-hex or cidv1:sha256:<hex>) or null; got ' +
        JSON.stringify(parent_cid),
      );
    }
    const _pcPrefix = 'cidv1:sha256:';
    const _pcTail = parent_cid.startsWith(_pcPrefix)
      ? parent_cid.slice(_pcPrefix.length)
      : parent_cid;
    if (!/^[0-9a-f]{64}$/.test(_pcTail)) {
      throw new Error(
        'invalid parent_cid: must be sha256-hex (64 lowercase hex chars) or cidv1:sha256:<hex>; got ' +
        JSON.stringify(parent_cid),
      );
    }
    _parent_cid_canon = parent_cid; // preserve verbatim form (bare hex or cidv1:sha256:<hex>)
  }
  // W769 - Data residency region pin. region is OPTIONAL. The W460
  // byte-stability law applies VERBATIM: absent / null / '' all collapse
  // to null (the conditional manifest spread + conditional hash slot
  // below are skipped) so pre-W769 artifacts rebuilt without a region
  // remain byte-identical to their original artifact_hash. Mirror of the
  // W721 sparsity_profile / W722 kv_profile / W739 parent_cid conditional
  // pattern.
  //
  // Validation: when non-null, region MUST be a non-empty string. We do
  // NOT validate against the REGIONS taxonomy here because (a) the
  // taxonomy is an evolving compliance contract and we don't want a
  // taxonomy bump to invalidate every shipped .kolm, and (b) the
  // residency UI and CLI gate on REGIONS at write time, so reaching
  // buildPayload with an unknown region requires a deliberate caller
  // bypass.
  let _region_canon = null;
  if (region !== null && region !== undefined && region !== '') {
    if (typeof region !== 'string') {
      throw new Error(
        'invalid region: must be a string (REGIONS taxonomy id) or null; got ' +
        JSON.stringify(region),
      );
    }
    _region_canon = region;
  }
  // W786 - Sustainability badge. Computed from training_stats {gpu, gpu_hours,
  // region, utilization} via badgeFor (a pure function in src/carbon-estimator.js).
  // The badge is POST-HASH metadata: it's NOT bound into artifact_hash_input
  // so legacy artifacts (pre-W786) rebuilt without gpu_hours remain byte-
  // identical to their original artifact_hash. The W460 byte-stability
  // contract: when training_stats.gpu_hours is absent, _w786_badge stays
  // null and the matching manifest field collapses to an empty spread (the
  // `sustainability_badge` key is OMITTED from the manifest entirely so
  // JSON.stringify produces byte-identical output to pre-W786 builds).
  // The badge is hygiene/audit signal only - a tamperer flipping
  // co2_kg_estimate after build does NOT break receipt.json because the
  // field is not in artifact_hash_input. This is intentional: the
  // estimator is MODELED (methodology='public-research-estimate'), not
  // measured, so binding it would imply a fidelity we do not have.
  let _w786_badge = null;
  if (training_stats && (
       Number.isFinite(Number(training_stats.gpu_hours))
    || Number.isFinite(Number(training_stats.gpuHours))
  )) {
    const ts = {
      gpu: training_stats.gpu || training_stats.gpu_class || null,
      gpu_hours: Number(training_stats.gpu_hours ?? training_stats.gpuHours),
      region: training_stats.region || _region_canon || null,
      utilization: training_stats.utilization,
    };
    try {
      _w786_badge = badgeForW786({ training_stats: ts });
    } catch (e) {
      // Never let a badge-compute failure abort the build. The badge is
      // hygiene metadata; an honest "could not compute" envelope is better
      // than a refusal.
      _w786_badge = {
        ok: true,
        version: CARBON_VERSION_W786,
        co2_kg_estimate: null,
        kwh: null,
        estimate_quality: 'invalid_inputs',
        methodology: 'public-research-estimate',
        methodology_version: CARBON_VERSION_W786,
        honest_caveat: 'estimate_not_measured',
        error_bar_pct: 30,
        note: 'badge compute threw: ' + (e && e.message),
      };
    }
  }
  // Wave 148 - pretokenize block. Same shape as moe: bridge builds + validates;
  // we re-validate so a hand-rolled block still gets schema-checked. Drift in
  // either tokens.idx or tokens.pack changes idx_file.sha256/pack_file.sha256
  // → block.hash → artifact_hash. The bundled binary files also fold into
  // extra_files_hash below for double anchoring.
  let pretokenize_block = null;
  if (pretokenizeInput) {
    pretokenize_block = validatePretokenizeBlock(pretokenizeInput);
  }
  // Wave 164 - external + adversarial holdout block. Bridge built + validated;
  // we re-validate here so hand-rolled blocks still get schema-checked. Drift
  // in any holdout's file_sha256 or recorded accuracy changes block.hash →
  // artifact_hash → every signature.
  let external_holdout_block = null;
  if (externalHoldoutInput) {
    external_holdout_block = validateExternalHoldoutBlock(externalHoldoutInput);
  }
  // Wave 165 (N+5) - tenant shadow corpus provenance. Unlike external_holdout
  // (one block per recipe, holding many holdouts), tenant_shadow is one block
  // per tenant-corpus pair, and the caller may name multiple. Stored as an
  // array of validated blocks so the verifier can re-anchor each independently.
  // The corpus bytes themselves are NEVER bundled into the .kolm (HIPAA
  // data-never-leaves-tenant) - only the {tenant_id, corpus_id, corpus_sha256,
  // accuracy, ...} fingerprint rides in the manifest.
  let tenant_shadow_blocks = null;
  if (tenantShadowInput) {
    const arr = Array.isArray(tenantShadowInput) ? tenantShadowInput : [tenantShadowInput];
    tenant_shadow_blocks = arr.map(b => validateTenantShadowBlock(b));
  }
  // Wave 166 (N+7) - third-party auditor attestation blocks. Same array shape
  // as tenant_shadow (an artifact may carry multiple auditor signatures - 
  // e.g., Deloitte signed at issue time, AICPA member re-signed at procurement
  // gate). Each block is validated standalone here (schema + Ed25519 signature
  // self-consistency); cross-checking the signed claims against this artifact's
  // own manifest values happens in binder check #22 at verify time so we don't
  // need access to the just-built manifest fields during construction.
  let auditor_attestation_blocks = null;
  if (auditorAttestationInput) {
    const arr = Array.isArray(auditorAttestationInput) ? auditorAttestationInput : [auditorAttestationInput];
    auditor_attestation_blocks = arr.map(b => validateAuditorAttestationBlock(b));
  }
  // Wave 167 (M+4) - supersession block. Exactly one predecessor per artifact
  // (chains form by walking predecessor_artifact_hash recursively). Validated
  // here so a hand-rolled block still gets schema-checked; the block's short
  // hash folds into artifact_hash_input below so any post-build mutation
  // breaks the receipt chain.
  let supersession_block = null;
  if (supersessionInput) {
    // Accept either a raw input (no spec/hash - CLI passes this shape) or a
    // pre-built block (spec/hash present - programmatic callers). buildSupersessionBlock
    // is idempotent on a raw input and validates required fields; validateSupersessionBlock
    // then re-checks schema + hash so the resulting object is canonical.
    const block = supersessionInput.spec === SUPERSESSION_SPEC_VERSION
      ? supersessionInput
      : buildSupersessionBlock(supersessionInput);
    supersession_block = validateSupersessionBlock(block);
  }
  // Wave 167 (M+3) - optional embedded drift report. When present the verifier
  // re-checks its schema + hash and surfaces the verdict (within / drift /
  // breach). Embedding is opt-in: most tenants will ship drift reports as
  // sibling files rather than baking them into the manifest, but compliance-
  // sensitive deployments may want the verdict cryptographically bound.
  let drift_report_block = null;
  if (driftReportInput) {
    // Same pattern as supersession: accept raw input or pre-built block.
    // buildDriftReport demands baseline_snapshot + current_snapshot + signals
    // so callers must already have those; raw shape here means "spec/hash not yet set".
    const block = driftReportInput.spec === DRIFT_REPORT_SPEC_VERSION
      ? driftReportInput
      : buildDriftReport(driftReportInput);
    drift_report_block = validateDriftReport(block);
  }
  const workflow_ir_json = workflow_ir ? JSON.stringify(workflow_ir, null, 2) : null;
  const attestation_report_json = attestation_report ? JSON.stringify(attestation_report, null, 2) : null;
  let confidential_compute_block = null;
  if (confidential_compute) {
    if (typeof confidential_compute !== 'object') {
      throw new Error('confidential_compute must be an object (the precomputed state from verifyAttestation)');
    }
    confidential_compute_block = ccManifestBlock(confidential_compute.kind, confidential_compute);
  } else if (capability_block && capability_block.requires_confidential_compute && !attestation_report_json) {
    // Honest default: contract demands TEE but no report supplied. Emit a
    // visibly UNVERIFIED block so the verifier can fail loudly instead of
    // silently shipping an artifact with no attestation state at all.
    confidential_compute_block = ccManifestBlock(capability_block.attestation, null);
  }
  if (lineage_block && lineage_block.workflow_ir_hash && !workflow_ir_json) {
    throw new Error(`lineage claims workflow_ir_hash=${lineage_block.workflow_ir_hash} but no workflow_ir was supplied to buildPayload; the artifact would fail verification.`);
  }
  if (workflow_ir_json && lineage_block && lineage_block.workflow_ir_hash) {
    const recomputed = hashIr(workflow_ir);
    if (recomputed !== lineage_block.workflow_ir_hash) {
      throw new Error(`workflow_ir hash mismatch: lineage claims ${lineage_block.workflow_ir_hash}, supplied IR hashes to ${recomputed}`);
    }
  }
  // Wave 151 - validate each recipe's declared class (or infer if omitted).
  // The per-recipe class lives inside recipes.json so a verifier can re-check
  // it without trusting the manifest. The artifact-level class is the
  // most-permissive of the per-recipe classes (see rollupArtifactClass).
  const per_recipe_classes = recipes.map(r => {
    try {
      return validateRecipeClass(r);
    } catch (err) {
      throw new Error(`recipe ${JSON.stringify(r.id)} failed class validation: ${err.message}`);
    }
  });
  // Wave 285 - every recipe carries an honest source_type declaring HOW the
  // source was produced (hand_written / pattern_generated / llm_emitted /
  // distilled / compiled_from_dsl). The verifier rejects any class/source_type
  // mismatch at build time so a `rule` artifact can never silently ship LLM-
  // emitted source.
  const per_recipe_source_types = recipes.map((r, i) => {
    const inferred = r.source_type || inferSourceType({ ...r, class: per_recipe_classes[i] });
    const stamped = { ...r, class: per_recipe_classes[i], source_type: inferred };
    try {
      validateRecipeSourceType(stamped);
    } catch (err) {
      throw new Error(`recipe ${JSON.stringify(r.id)} failed source_type validation: ${err.message}`);
    }
    return inferred;
  });
  const recipes_json = JSON.stringify({
    spec: 'rs-1',
    n: recipes.length,
    recipes: recipes.map((r, i) => ({
      id: r.id,
      name: r.name,
      source: r.source,
      source_hash: r.source_hash,
      version_id: r.version_id,
      tags: r.tags || [],
      schema: r.schema || null,
      // Wave 151 - honest per-recipe class. One of rule / synthesized_rule /
      // compiled_rule / distilled_model. The artifact-level artifact_class
      // is the max of these. See src/recipe-class.js for definitions.
      class: per_recipe_classes[i],
      // Wave 285 - honest per-recipe source_type. One of hand_written /
      // pattern_generated / llm_emitted / distilled / compiled_from_dsl.
      // Verifiers reject any class/source_type mismatch (rule + llm_emitted
      // is rejected; synthesized_rule + pattern_generated is rejected; etc.).
      source_type: per_recipe_source_types[i],
      // Wave F - when the recipe was authored as a DSL, ship the DSL block
      // inside recipes.json so an external verifier can recompute the JS
      // source (via emitJs) AND the native.c / native.rs source (via
      // emitCompiledTargets) and confirm every hash in manifest.compiled_targets
      // matches. Recipes that arrived as raw JS get null here.
      dsl: r.dsl || null,
      // Wave 151 - teacher attribution for synthesized_rule and distilled_model.
      // null when not applicable. Verifiers cross-check against artifact_class.
      teacher_vendor: r.teacher_vendor || null,
      teacher_model: r.teacher_model || null,
      synthesized_by: r.synthesized_by || null,
    })),
  }, null, 2);

  // W367 - recipe.bundle.mjs: self-contained ESM bundle so the .kolm runs on
  // any Node 18+ / Bun 1+ / Deno 1.40+ host without re-reading recipes.json.
  // The recipe source is already a pure function on (input, lib) thanks to the
  // sandbox guard in src/verifier.js (no require/import/process/etc.). We
  // concatenate the source bodies of every rule/synthesized_rule recipe and
  // export a single dispatcher function that walks them in declaration order,
  // returning the first one that does not throw. Same semantics as runJsTarget
  // in artifact-runner.js, but the artifact ships as a runnable file the host
  // can `import` directly instead of having to spin up a vm sandbox.
  //
  // We only emit the bundle when every per-recipe class is in BUNDLEABLE
  // (rule | synthesized_rule | compiled_rule). For distilled_model artifacts
  // the bundle is skipped because the executable artifact is the model itself,
  // not a JS function.
  const BUNDLEABLE_CLASSES = new Set(['rule', 'synthesized_rule', 'compiled_rule']);
  const wantsBundle = recipes.length > 0
    && per_recipe_classes.every((c) => BUNDLEABLE_CLASSES.has(c))
    && recipes.every((r) => typeof r.source === 'string' && r.source.length > 0);
  const bundle_filename = 'recipe.bundle.mjs';
  let recipe_bundle_mjs = null;
  if (wantsBundle) {
    recipe_bundle_mjs = buildRecipeBundleMjs(recipes, { spec: ARTIFACT_SPEC, job_id, generated_at: buildTimestamp });
  }

  // Pack + index slots carry optional real bytes. For 'rule' class with no
  // pack/index supplied we drop the file from the zip entirely (instead of
  // emitting an empty placeholder that pretends to be a LoRA/vector slot).
  const lora_bin = encodePack(pack);
  const index_bin = encodeIndex(index);
  const has_pack = lora_bin.length > 0;
  const has_index = index_bin.length > 0;

  // model.gguf was historically a JSON pointer record padding the zip. For
  // 'rule' class we drop it. For 'distilled_model' (future) it will hold
  // real quantized weights. For backward compat with the four shipped
  // fixtures, when base_model is set to a non-'none' value we still emit
  // the pointer record so existing artifacts that pin a base_model keep
  // their model.gguf entry on disk.
  // Wave 151 - roll up per-recipe classes into the artifact-level class.
  // The artifact_class is the MAX of recipes' classes under CLASS_RANK. An
  // explicit artifact_class arg from the caller takes precedence (allows
  // callers to pin a higher class for cross-compatibility) but must not be
  // LOWER than the rolled-up class.
  const _rolledUpClass = rollupArtifactClass(per_recipe_classes);
  const _finalClass = _class || _rolledUpClass;
  // (We don't reject downgrades here - buildPayload still produces the
  // artifact, but validateArtifactClass below would reject a misdeclared one
  // before the receipt is signed.)
  // W457 - runtime_target is the single source of truth (resolved early so
  // model_pointer + manifest both reference the same value, never the
  // legacy hardcoded 'cloud' that diverged from receipt.runtime_target).
  // Values: js | wasm | native | gguf | onnx | cloud. Default 'js' for the
  // rule-class path; weight-class callers pass runtime_target='gguf' (or
  // 'onnx'/'wasm'/'native') plus the matching model_weights blob.
  const _supportedTargets = new Set(['js', 'wasm', 'native', 'gguf', 'onnx', 'cloud']);
  const _runtimeTargetDeclared = (typeof runtime_target === 'string' && _supportedTargets.has(runtime_target))
    ? runtime_target
    : 'js';
  const _runtimeTargetConfig = (runtime_target_config && typeof runtime_target_config === 'object')
    ? { ...runtime_target_config }
    : null;
  const _entrypoint = (entrypoint && typeof entrypoint === 'object')
    ? { ...entrypoint }
    : null;
  // model_weights: optional { filename:string, content:Buffer } record. When
  // present, gets bundled into the zip at the declared filename + folded into
  // manifest.hashes + artifact_hash so any post-build tampering breaks the
  // signature chain. The verifier (binder.js rtCheck for 'gguf'/'onnx'/'wasm')
  // re-opens the zip and confirms the bytes match the declared sha256.
  let _modelWeightsRecord = null;
  if (model_weights) {
    if (!model_weights.filename || typeof model_weights.filename !== 'string') {
      throw new Error('model_weights.filename must be a string');
    }
    if (!Buffer.isBuffer(model_weights.content)) {
      throw new Error('model_weights.content must be a Buffer');
    }
    _modelWeightsRecord = {
      filename: model_weights.filename,
      content: model_weights.content,
      sha256: sha256(model_weights.content),
      bytes: model_weights.content.length,
    };
  }
  let _modelWeightArtifactManifest = null;
  let _modelWeightArtifactManifestJson = null;
  if (_modelWeightsRecord && ['gguf', 'onnx', 'wasm', 'native'].includes(_runtimeTargetDeclared)) {
    const unsignedWeightManifest = buildModelWeightArtifactManifest({
      artifact_id: `${job_id}:${_modelWeightsRecord.filename}`,
      model_id: base_model || 'unknown',
      variant: _runtimeTargetDeclared,
      runtime_targets: modelWeightRuntimeTargets(_runtimeTargetDeclared),
      files: [{
        path: _modelWeightsRecord.filename,
        bytes: _modelWeightsRecord.bytes,
        sha256: _modelWeightsRecord.sha256,
        format: _runtimeTargetDeclared,
        role: _runtimeTargetDeclared === 'wasm' ? 'model_lib' : 'weights',
      }],
      source: {
        kind: 'kolm-artifact',
        job_id,
        runtime_target: _runtimeTargetDeclared,
        declared_path: _modelWeightsRecord.filename,
      },
      created_at: buildTimestamp,
      policy: {
        require_signature: process.env.KOLM_ED25519_DISABLE !== '1',
        no_unsigned_auto_fetch: true,
      },
    });
    _modelWeightArtifactManifest = ed25519Signer
      ? signModelWeightArtifactManifest(unsignedWeightManifest, ed25519Signer, { signed_at: buildTimestamp })
      : unsignedWeightManifest;
    const weightManifestCheck = verifyModelWeightArtifactManifest(_modelWeightArtifactManifest, {
      require_signature: process.env.KOLM_ED25519_DISABLE !== '1',
    });
    if (!weightManifestCheck.ok) {
      throw new Error(`model weight artifact manifest invalid: ${weightManifestCheck.reason}`);
    }
    _modelWeightArtifactManifestJson = serializeModelWeightArtifactManifest(_modelWeightArtifactManifest);
  }
  // W457 - model_pointer is the legacy pointer-only document (no real
  // weights). Suppressed entirely when a real model_weights bundle was
  // supplied - the bundled weights are the source of truth, the pointer
  // would just be dead bytes that the verifier would have to skip. The
  // pointer's `runtime` field used to be a hardcoded 'cloud' lie; now it
  // mirrors `runtime_target` so a reader cannot get a divergent answer.
  const want_model_pointer = !_modelWeightsRecord && ((_finalClass === 'distilled_model') || (base_model && base_model !== 'none'));
  const model_pointer = want_model_pointer ? JSON.stringify({
    spec: ARTIFACT_SPEC,
    base_model: base_model || 'Qwen/Qwen2.5-3B-Instruct',
    runtime: _runtimeTargetDeclared,
    note: 'pointer-only artifact; weights resolved on `kolm run` first launch.',
  }, null, 2) : null;

  // evals.json - the "no eval, no compile" gate. Synthesized from the
  // user's positives at compile time; surfaced in the artifact so anyone
  // can recompute K-score by re-running them.
  const evals_obj = evals && evals.cases ? evals : {
    spec: 'rs-1-evals',
    n: 0,
    cases: [],
    notes: 'compile-time evals were not supplied; K-score uses synthesizer pass-rate only',
  };
  const evals_json = JSON.stringify(evals_obj, null, 2);
  const eval_set_hash = sha256(Buffer.from(evals_json));
  const _evalScore = (typeof eval_score === 'number')
    ? eval_score
    : (typeof evals_obj.coverage === 'number' ? evals_obj.coverage
      : (typeof training_stats?.pass_rate_positive === 'number' ? training_stats.pass_rate_positive : 0));
  const _judgeId = judge_id || process.env.KOLM_JUDGE_ID || 'kolm-pattern-synth-1';
  const _tier = tier || 'recipe';

  // CID parts always include the five canonical slots so existing verifiers
  // and the cid schema keep working. When a file is physically absent from
  // the zip (rule-class with no pack/index/model), the slot hash is the
  // sha256 of an empty buffer - an explicit "this slot intentionally has no
  // content" sentinel rather than a fake byte payload.
  // W739 - fold parent_cid into the model_pointer hash WHEN PRESENT so the
  // CID itself (not just the receipt chain) is sensitive to lineage. Absent
  // parent_cid (null/undefined/'' all canonicalise to null in
  // _parent_cid_canon above) leaves model_pointer hashing untouched - pre-
  // W739 artifacts rebuilt without a parent_cid remain CID-byte-identical
  // (W460 byte-stability law). The suffix is `\x00parent_cid:<cid>` (NUL
  // separator + ASCII tag) so it cannot collide with any valid model_pointer
  // payload, which is plain bytes by contract upstream. The matching
  // artifact_hash_input.parent_cid slot below is preserved so a tamperer
  // who rewrites manifest.parent_cid AND the model_pointer hash still has
  // to break the receipt chain too - defense in depth, not redundancy.
  let _modelPointerHash;
  if (model_pointer) {
    if (_parent_cid_canon) {
      _modelPointerHash = sha256(Buffer.concat([
        Buffer.from(model_pointer),
        Buffer.from('\x00parent_cid:' + _parent_cid_canon, 'utf8'),
      ]));
    } else {
      _modelPointerHash = sha256(Buffer.from(model_pointer));
    }
  } else if (_parent_cid_canon) {
    // Edge case: no model_pointer, but parent_cid present. We MUST still bind
    // parent_cid into the CID; the bare NUL+tag+cid buffer is unambiguous
    // because legacy empty model_pointer used EMPTY_SHA (sha256 of zero
    // bytes), which can never collide with a non-empty buffer's hash.
    _modelPointerHash = sha256(Buffer.from('\x00parent_cid:' + _parent_cid_canon, 'utf8'));
  } else {
    _modelPointerHash = EMPTY_SHA;
  }
  const hashes = {
    model_pointer: _modelPointerHash,
    recipes_json: sha256(Buffer.from(recipes_json)),
    lora_bin: has_pack ? sha256(lora_bin) : EMPTY_SHA,
    index_bin: has_index ? sha256(index_bin) : EMPTY_SHA,
    evals_json: eval_set_hash,
  };
  // Wave V - optional per-file hashes for new bundled blocks. We add them only
  // when the block is present so legacy CIDs stay byte-stable.
  if (workflow_ir_json) hashes.workflow_ir = sha256(Buffer.from(workflow_ir_json));
  if (attestation_report_json) hashes.attestation_report = sha256(Buffer.from(attestation_report_json));
  // W367 - bundle hash. Only present for rule/synthesized_rule/compiled_rule
  // artifacts; absent for distilled_model so legacy CIDs stay byte-stable.
  if (recipe_bundle_mjs) hashes.recipe_bundle_mjs = sha256(Buffer.from(recipe_bundle_mjs));
  // W457 - model_weights hash. Present whenever the caller bundled real
  // weights (gguf/onnx/wasm/native). The verifier (binder.js rtCheck) reads
  // manifest.runtime_target + runtime_target_config[<target>_path] to locate
  // the entry, hashes the bytes, and refuses to pass when sha256 drifts.
  if (_modelWeightsRecord) hashes.model_weights = _modelWeightsRecord.sha256;
  if (_modelWeightArtifactManifest) {
    hashes.model_weight_artifact_manifest = hashModelWeightArtifactManifest(_modelWeightArtifactManifest);
  }
  // Wave 144 - extra files (e.g. tokenizer.json) ride inside the .kolm zip.
  // Each gets a hash in manifest.hashes.extra_files keyed by filename, and the
  // canonical hash-of-extra-files folds into artifact_hash_input so tampering
  // breaks the receipt chain. Filenames sort for determinism.
  const extra_files_list = Array.isArray(extra_files) ? extra_files.slice() : [];
  if (extra_files_list.length) {
    const map = {};
    const sorted = extra_files_list.slice().sort((a, b) => a.filename.localeCompare(b.filename));
    for (const f of sorted) {
      if (!f || typeof f.filename !== 'string') {
        throw new Error('extra_files entries must be { filename:string, content:Buffer } or { filename:string, absPath:string }');
      }
      if (Buffer.isBuffer(f.content)) {
        map[f.filename] = sha256(f.content);
      } else if (typeof f.absPath === 'string' && fs.existsSync(f.absPath)) {
        // Path-backed entry: hash directly from disk to support files
        // larger than Node's 2 GiB readFileSync limit (e.g. Trinity GGUF).
        map[f.filename] = sha256File(f.absPath);
      } else {
        throw new Error(`extra_files[${f.filename}]: needs content:Buffer or absPath:string pointing to an existing file`);
      }
    }
    hashes.extra_files = map;
  }
  // Deterministic content-id over the per-file hashes - independent of the
  // K-score, signature, or receipt. Same content always produces the same
  // CID, even across signing key rotations.
  const cid = cidFromManifestHashes(hashes);

  // Honest seed provenance block. Always present in the manifest so the
  // verifier can branch on shape. When the compile path did NOT split seeds
  // (legacy/hardcoded-evals path) `eval_source` is 'self_generated' and the
  // verifier downgrades the artifact to 'sample_check' regardless of the
  // K-score - see verifier.js (Wave D).
  const seed_provenance_block = seed_provenance ? {
    seeds_hash: seed_provenance.seeds_hash,
    split_seed: seed_provenance.split_seed,
    holdout_ratio: seed_provenance.holdout_ratio,
    train_hash: seed_provenance.train_hash,
    holdout_hash: seed_provenance.holdout_hash,
    train_count: seed_provenance.train_count,
    holdout_count: seed_provenance.holdout_count,
    eval_source: seed_provenance.eval_source || 'unknown',
    leakage_report_hash: seed_provenance.leakage_report_hash || null,
    comparator: seed_provenance.comparator || 'exact',
    source_format_mix: seed_provenance.source_format_mix || null,
    seeds_path_basename: seed_provenance.seeds_path_basename || null,
    production_ready: seed_provenance.production_ready === true,
    min_train: typeof seed_provenance.min_train === 'number' ? seed_provenance.min_train : null,
    min_holdout: typeof seed_provenance.min_holdout === 'number' ? seed_provenance.min_holdout : null,
    input_overlap_count: typeof seed_provenance.input_overlap_count === 'number' ? seed_provenance.input_overlap_count : null,
    output_overlap_count: typeof seed_provenance.output_overlap_count === 'number' ? seed_provenance.output_overlap_count : null,
    near_duplicate_count: typeof seed_provenance.near_duplicate_count === 'number' ? seed_provenance.near_duplicate_count : null,
    grouped_overlap_count: typeof seed_provenance.grouped_overlap_count === 'number' ? seed_provenance.grouped_overlap_count : null,
    // Wave 283 - hash of the rows the teacher actually received. When the
    // policy held (train-only synthesis) this equals train_hash; the
    // verifier (and an external auditor) reads this to confirm no holdout
    // leaked into recipe construction.
    synthesis_input_hash: typeof seed_provenance.synthesis_input_hash === 'string' ? seed_provenance.synthesis_input_hash : null,
    // Wave 284 - when group-aware splitting was requested, this records
    // which row-metadata key was used to define a "group" so two rows
    // about the same member / claim / case never straddle the split.
    group_key: typeof seed_provenance.group_key === 'string' ? seed_provenance.group_key : null,
    // Wave 409c - auditor mandate. Honest counts of where seeds came from,
    // and an explicit "eval_provenance" flag distinguishing a real eval-run
    // verifier output from a placeholder (synthetic / hard-coded number).
    //   eval_provenance ∈ { 'real_eval' | 'placeholder' | 'unknown' }
    // The productionReady() gate rejects 'placeholder' unconditionally so a
    // pipeline that never actually ran the eval loop cannot ship an artifact
    // with production_ready:true. source_seed_count / approved_count /
    // synthetic_count let a downstream auditor reconstruct the data lineage
    // without re-reading the on-disk dataset record.
    source_seed_count: typeof seed_provenance.source_seed_count === 'number' ? seed_provenance.source_seed_count : null,
    approved_count: typeof seed_provenance.approved_count === 'number' ? seed_provenance.approved_count : null,
    synthetic_count: typeof seed_provenance.synthetic_count === 'number' ? seed_provenance.synthetic_count : null,
    eval_provenance: typeof seed_provenance.eval_provenance === 'string' ? seed_provenance.eval_provenance : 'unknown',
  } : {
    seeds_hash: null,
    split_seed: null,
    holdout_ratio: null,
    train_hash: null,
    holdout_hash: null,
    train_count: 0,
    holdout_count: 0,
    eval_source: 'self_generated',
    leakage_report_hash: null,
    comparator: 'exact',
    source_format_mix: null,
    seeds_path_basename: null,
    production_ready: false,
    min_train: null,
    min_holdout: null,
    input_overlap_count: null,
    output_overlap_count: null,
    near_duplicate_count: null,
    grouped_overlap_count: null,
    synthesis_input_hash: null,
    group_key: null,
    // Wave 409c - see above.
    source_seed_count: null,
    approved_count: null,
    synthetic_count: null,
    eval_provenance: 'unknown',
  };

  // finalized-c6 - transitive cross-corpus holdout-disjointness ledger.
  // ADDITIVE + CONDITIONAL: attached ONLY when the compile path actually built a
  // ledger (both train + holdout corpora present). Default output is byte-
  // identical to pre-c6 manifests (no field added), so existing artifact hashes
  // and golden fixtures are unchanged. When present, the full signed block + its
  // hash + verdict ride inside seed_provenance, which is hashed into
  // manifest_hash and thus the artifact_hash chain - any tamper with a matrix
  // cell, tolerance, or the disjoint verdict drifts the ledger hash -> manifest
  // hash -> the receipt chain breaks. A verifier re-runs validateDisjointness
  // Ledger(seed_provenance.disjointness_ledger) and (when commitments published)
  // reVerifyFromCommitments to confirm the fail-closed verdict from Merkle roots
  // without any plaintext.
  if (seed_provenance && seed_provenance.disjointness_ledger
    && typeof seed_provenance.disjointness_ledger === 'object'
    && typeof seed_provenance.disjointness_ledger_hash === 'string') {
    seed_provenance_block.disjointness_ledger = seed_provenance.disjointness_ledger;
    seed_provenance_block.disjointness_ledger_hash = seed_provenance.disjointness_ledger_hash;
    seed_provenance_block.disjointness_disjoint = (seed_provenance.disjointness_disjoint === true);
    seed_provenance_block.disjointness_n_violations =
      typeof seed_provenance.disjointness_n_violations === 'number'
        ? seed_provenance.disjointness_n_violations : null;
  }

  // Compiled-targets manifest block. Only present for compiled_rule artifacts.
  // The source bodies themselves live on disk as native.c / native.rs entries
  // in the zip (added to `files` below). The manifest carries the hashes so a
  // verifier can recompute them from recipes_json.dsl + emitCompiledTargets
  // and confirm every byte matches.
  //
  // Wave G - when the caller also supplies `compiled_targets.native`
  // (produced by src/native-compile.js when a toolchain is present), each
  // recipe entry gains a `c.bin` / `rust.bin` sub-block recording compiler
  // version, flags, and bin_hash; the binary files are appended to the zip.
  // Absence of `.native` is the JS-rule fallback - manifest stays
  // source-only and verification works without any toolchain.
  let compiled_targets_block = null;
  let compiled_target_files = [];
  if (_finalClass === 'compiled_rule' && compiled_targets) {
    const native = compiled_targets.native || null;
    const out = {
      spec: compiled_targets.spec,
      single_recipe: compiled_targets.single_recipe,
      targets: compiled_targets.targets,
      recipes: {},
    };
    if (native) {
      out.native_spec = native.bundle.spec;
      out.host_triple = native.bundle.host_triple;
      // Wave 153 - bundle-level toolchain pin record. One pin per kind (c,
      // rust) capturing compiler + version + shim hash + source_date_epoch.
      // The receipt chain absorbs this via compiled_targets_hash; product
      // surfaces (binder PDF, /spec/rs-1, /how-it-works) read it to display
      // "this compiled_rule artifact was built with rustc X / gcc Y."
      if (native.bundle.target_toolchain_pin && Object.keys(native.bundle.target_toolchain_pin).length) {
        out.target_toolchain_pin = native.bundle.target_toolchain_pin;
      }
      if (native.bundle.skipped && Object.keys(native.bundle.skipped).length) {
        out.native_skipped = native.bundle.skipped;
      }
    }
    for (const rid of Object.keys(compiled_targets.recipes)) {
      const t = compiled_targets.recipes[rid];
      const recipeBlock = {
        c: { filename: t.c.filename, source_hash: t.c.source_hash, bytes: t.c.bytes },
        rust: { filename: t.rust.filename, source_hash: t.rust.source_hash, bytes: t.rust.bytes },
      };
      if (native && native.bundle.recipes[rid]) {
        const nr = native.bundle.recipes[rid];
        if (nr.c) recipeBlock.c.bin = nr.c;
        else if (nr.c_error) recipeBlock.c.bin_error = nr.c_error;
        if (nr.rust) recipeBlock.rust.bin = nr.rust;
        else if (nr.rust_error) recipeBlock.rust.bin_error = nr.rust_error;
        // Wave 155 §P+3 - WASM lands as its own sub-block. Unlike c/rust the
        // wasm block does not carry its own source file (it reuses whichever
        // c/rust source was used); the bin is the only new artifact.
        if (nr.wasm) recipeBlock.wasm = { bin: nr.wasm };
        else if (nr.wasm_error) recipeBlock.wasm = { bin_error: nr.wasm_error };
      }
      out.recipes[rid] = recipeBlock;
      compiled_target_files.push({ filename: t.c.filename, content: Buffer.from(t.c.source, 'utf8') });
      compiled_target_files.push({ filename: t.rust.filename, content: Buffer.from(t.rust.source, 'utf8') });
    }
    if (native) {
      for (const f of native.files) compiled_target_files.push(f);
    }
    compiled_targets_block = out;
  }

  const manifest = {
    spec: ARTIFACT_SPEC,
    job_id,
    task,
    created_at: buildTimestamp,
    ...(repro.enabled ? { reproducible_build: {
      spec: 'kolm-reproducible-build-v1',
      source_date_epoch: repro.epoch,
      timestamp: buildTimestamp,
    } } : {}),
    // W457 - alias of runtime_target so the two fields can never diverge.
    // Legacy readers (binder HTML, intent ranking, marketplace fingerprint)
    // consume manifest.runtime; the verifier + dispatchRuntime + receipt
    // consume manifest.runtime_target. Both now report the same string.
    runtime: _runtimeTargetDeclared,
    runtime_target: _runtimeTargetDeclared,
    runtime_target_config: _runtimeTargetConfig,
    entrypoint: _entrypoint,
    ...(_modelWeightArtifactManifest ? { model_weight_artifact_manifest: _modelWeightArtifactManifest } : {}),
    artifact_class: _finalClass,
    // Wave 151 - per-class recipe count surfaces "we have 6 rule recipes and
    // 1 distilled-model recipe" without forcing readers to parse recipes.json.
    artifact_class_breakdown: per_recipe_classes.reduce((acc, c) => { acc[c] = (acc[c] || 0) + 1; return acc; }, {}),
    base_model: base_model || 'Qwen/Qwen2.5-3B-Instruct',
    target_device: target_device || null,
    train_device: train_device || null,
    // W409s - device-fit hints carried at the top level. memory_requirement_mb
    // is the artifact's working-set; offline_capable says "this artifact can
    // run without network egress". Both are read by devices.recommendForProfile()
    // to gate target/quant picks. Default to honest minimums (5 MB / true) when
    // a caller doesn't set them; both can be overridden via target_device.
    memory_requirement_mb: (target_device && typeof target_device.memory_requirement_mb === 'number')
      ? target_device.memory_requirement_mb : 5,
    offline_capable: (target_device && target_device.offline_capable != null)
      ? !!target_device.offline_capable : true,
    tier: _tier,
    judge_id: _judgeId,
    eval_score: Number(Math.max(0, Math.min(1, _evalScore)).toFixed(4)),
    recipes: {
      n: recipes.length,
      registry_hash: sha256(canonicalJson(recipes.map(r => ({ id: r.id, hash: r.source_hash })))),
    },
    lora: lora_pointer || null,
    recall: recall_namespace ? { namespace: recall_namespace } : null,
    // Wave 151 - when the rolled-up class is synthesized_rule or distilled_model,
    // teacher attribution from the per-recipe shape must be promoted into the
    // manifest.training block so validateArtifactClass passes. The verifier
    // reads training.teacher_vendor / teacher_model / synthesized_by; a tenant
    // who only set them per-recipe gets them rolled up automatically here so
    // they do not have to repeat themselves in `training_stats`.
    training: (() => {
      const t = { ...(training_stats || { distilled_pairs: 0, accuracy: null }) };
      if (!t.teacher_vendor) {
        const tv = recipes.find(r => r.teacher_vendor)?.teacher_vendor;
        if (tv) t.teacher_vendor = tv;
      }
      if (!t.teacher_model) {
        const tm = recipes.find(r => r.teacher_model)?.teacher_model;
        if (tm) t.teacher_model = tm;
      }
      if (!t.synthesized_by) {
        const sb = recipes.find(r => r.synthesized_by)?.synthesized_by;
        if (sb) t.synthesized_by = sb;
      }
      return t;
    })(),
    evals: { n: evals_obj.n || (evals_obj.cases?.length || 0), spec: evals_obj.spec, hash: eval_set_hash },
    seed_provenance: seed_provenance_block,
    // Wave 409q - honest top-level binaries[] manifest. Always present (empty
    // array when no target was requested). Each entry pins a target
    // ('native' | 'wasm'), kind (c | rust), filename, sha256, size, and
    // compiler identity for an ACTUALLY produced binary. The verifier
    // (binder check #binaries-integrity) re-opens the zip, finds each
    // entry's filename, re-hashes the bytes, and confirms the match - 
    // surfacing `native_binary_missing` when the file is absent and
    // `native_binary_hash_mismatch` (or wasm_*) when the sha256 drifts.
    //
    // The honest auditor signal: when a tenant asked for target=c/rust/wasm
    // but no toolchain was present, this array is [] AND compiled_binary
    // is false AND production_ready is false - the manifest never claims a
    // compile happened that did not.
    binaries: Array.isArray(binaries) ? binaries : [],
    // Wave 409q - top-level compiled_binary verdict. null = no native/wasm
    // target was requested (the rule-only path); true = at least one binary
    // was produced; false = a target was requested but toolchain or compile
    // failed and the artifact ships source-only. The matching CLI/UI copy
    // says "source generated" not "compiled" when this is false.
    compiled_binary: typeof compiled_binary === 'boolean' ? compiled_binary : null,
    // Wave 409q - top-level production_ready. ANDs the seed_provenance.
    // production_ready signal (already computed earlier) with the
    // compiled_binary signal: if the tenant asked for native/wasm and the
    // build did NOT produce a binary, production_ready is false even when
    // the seeds split was clean.
    production_ready: (() => {
      const seedReady = seed_provenance_block.production_ready === true;
      if (compiled_binary === false) return false;
      return seedReady;
    })(),
    // Wave 409q - surface toolchain skip reasons at the top level so a
    // tenant reading the manifest sees "no clang on host: source-only" with
    // no need to walk compiled_targets.native_skipped. Empty/null when no
    // skips occurred OR no target was requested.
    native_skip_reasons: native_skip_reasons || null,
    compiled_targets: compiled_targets_block,
    capability: capability_block,
    lineage: lineage_block,
    export: export_block,
    // R-1 - runtime passports. Conditional spread per the W460 byte-
    // stability law: when the caller did not pass a runtime_passports
    // argument the key is OMITTED from the manifest entirely so pre-R-1
    // artifacts rebuilt without the field remain byte-identical to their
    // original manifest_hash + cid. When the caller did pass [] (an
    // explicit "no targets probed" signal) the key is present with an
    // empty array. Schema version stamp lets verifiers detect spec
    // migrations.
    ...(runtime_passports != null ? {
      runtime_passports: _runtime_passports_canon,
      runtime_passports_spec_version: RUNTIME_PASSPORT_SCHEMA_VERSION,
    } : {}),
    // W916-I1 - speculative decoding. Conditional spread per the W460
    // byte-stability law: when the caller did not pass a speculative_decoding
    // block the key is OMITTED entirely so pre-W916 artifacts rebuilt without
    // the field remain byte-identical to their original manifest_hash + cid.
    // NOT bound into artifact_hash_input below - speculative_decoding is an
    // OPERATIONAL fingerprint (target/draft pair + measured acceptance_rate),
    // not a property of the artifact bytes. A verifier on a different host
    // can legitimately re-probe with the same draft model and surface a
    // different acceptance_rate without breaking the receipt chain.
    // Schema lives in src/speculative-decoding.js (speculativePassportEntry).
    // Read at serve-time by apps/runtime/serve.py:build_engine to choose
    // draft_model + num_speculative_tokens before falling back to env
    // (KOLM_SERVE_SPECULATIVE_DRAFT) or auto-pairing.
    ...(speculative_decoding && typeof speculative_decoding === 'object'
        && Object.keys(speculative_decoding).length > 0
      ? { speculative_decoding } : {}),
    // W916-I3 - prompt cache (vLLM enable_prefix_caching / llama.cpp
    // --prompt-cache-all). Conditional spread; operational fingerprint
    // only - not bound into artifact_hash. Read at serve-time by
    // apps/runtime/serve.py via KOLM_PROMPT_CACHE env override → manifest.
    ...(prompt_cache && typeof prompt_cache === 'object'
        && Object.keys(prompt_cache).length > 0
      ? { prompt_cache } : {}),
    // W916-I4 - continuous batching (vLLM max_num_seqs / llama.cpp
    // --parallel N). Conditional spread; operational fingerprint only - 
    // not bound into artifact_hash. Read at serve-time by
    // apps/runtime/serve.py via KOLM_MAX_NUM_SEQS env override → manifest.
    ...(continuous_batching && typeof continuous_batching === 'object'
        && Object.keys(continuous_batching).length > 0
      ? { continuous_batching } : {}),
    // R-5 - evidence DAG. Conditional spread per the W460 byte-stability
    // law: when the caller did not pass an evidence_dag argument the key
    // is OMITTED from the manifest entirely so pre-R-5 artifacts rebuilt
    // without the field remain byte-identical to their original
    // manifest_hash + cid. When the caller did pass one (including an
    // empty {nodes:[],edges:[]}) the key is present + the spec version
    // stamp lets verifiers detect spec migrations.
    ...(_evidence_dag_canon != null ? {
      evidence_dag: _evidence_dag_canon,
      evidence_dag_spec_version: EVIDENCE_DAG_SCHEMA_VERSION,
    } : {}),
    moe: moe_block,
    pretokenize: pretokenize_block,
    external_holdout_provenance: external_holdout_block,
    tenant_shadow_corpus_provenance: tenant_shadow_blocks,
    auditor_attestation_provenance: auditor_attestation_blocks,
    supersession_provenance: supersession_block,
    drift_report: drift_report_block,
    // finalized-c6 - contamination-impact gate block. CONDITIONAL: only present
    // when the operator armed KOLM_KSCORE_CONTAM_IMPACT=1 (else byte-identical
    // manifest). Bound into artifact_hash_input below so any raw->corrected
    // tamper drifts the artifact hash.
    ...(contamination_impact_block ? { contamination_impact: contamination_impact_block } : {}),
    confidential_compute: confidential_compute_block,
    // W719 - Distillation-Aware Quantization (DAQ) per-layer bit budget.
    // When the build was driven by a DAQ profile (kolm distill / kolm
    // quantize --mixed-precision), the resolved per-layer profile array
    // rides here so a verifier replaying the quantize step can re-apply
    // the same per-layer schedule and reproduce the quantized weights.
    // Binds into artifact_hash below via mixed_precision_profile_hash so
    // any post-build profile mutation breaks the receipt chain.
    mixed_precision_profile: Array.isArray(daq_profile) && daq_profile.length > 0
      ? daq_profile : null,
    // finalized-c5 (layer-importance atom) - the falsifiable "applied == requested"
    // schedule read-back proof (src/layer-sensitivity-allocator.js
    // buildScheduleReceipt) and the REAL per-layer importance signal that drove the
    // allocation (computeCohortSensitivities). Conditional spread per the W460
    // byte-stability law: absent / null / {} all collapse to the empty spread so
    // pre-finalized-c5 artifacts rebuilt without these blocks remain byte-identical
    // to their original manifest_hash + artifact_hash. Bound into artifact_hash
    // below via mixed_precision_proof_hash / importance_signal_hash so any post-build
    // tamper of the read-back proof or the importance signal breaks the receipt chain.
    ...(mixed_precision_proof && typeof mixed_precision_proof === 'object'
        && Object.keys(mixed_precision_proof).length > 0
      ? { mixed_precision_proof } : {}),
    ...(importance_signal && typeof importance_signal === 'object'
        && Object.keys(importance_signal).length > 0
      ? { importance_signal } : {}),
    // finalized-c5 (calibration-set atom) - the reproducible calibration regime
    // (src/calibration-set.js calibrationReceiptBlock): method/regime/seqlen/rows,
    // provenance_hash, source fingerprints, dedup counts, language/kind distribution,
    // ordered window hashes - NO raw tenant text. Conditional spread per W460 so a
    // verifier can re-derive the exact calibration corpus from (seeds, params)
    // without changing hashes of artifacts that omit it.
    ...(calibration_provenance && typeof calibration_provenance === 'object'
        && Object.keys(calibration_provenance).length > 0
      ? { calibration_provenance } : {}),
    // W721 - Task-Specific Attention Compiler (TSAC) per-(layer,head)
    // sparsity profile. When the build was driven by `kolm distill
    // sparse-attention compile`, the resolved profile rides here so a
    // verifier (or a serve-time kernel selector) can re-derive the same
    // per-head kernel dispatch. Schema lives in src/tsac-profile.js;
    // builder in src/tsac-compiler.js. Bound into artifact_hash below via
    // sparsity_profile_hash with the W460 conditional-slot pattern so
    // existing (no-TSAC) artifacts stay byte-identical.
    sparsity_profile: sparsity_profile && typeof sparsity_profile === 'object'
        && Object.keys(sparsity_profile).length > 0
      ? sparsity_profile
      : null,
    // W722 - Importance-Tiered KV Cache (ITKV) per-artifact profile. When
    // the build was driven by `kolm distill itkv build`, the resolved
    // token-class + precision-tier profile rides here so a runtime KV
    // scheduler (PagedAttention / radix cache) can apply the same per-class
    // precision schedule. Schema lives in src/itkv-profile.js. Bound into
    // artifact_hash below via kv_profile_hash with the W460 conditional-slot
    // pattern so existing (no-ITKV) artifacts stay byte-identical.
    kv_profile: kv_profile && typeof kv_profile === 'object'
        && Object.keys(kv_profile).length > 0
      ? kv_profile
      : null,
    // W809 - output_schema spec (canonicalized). Surface the spec a runtime
    // wrapper or constrained decoder uses to enforce structured output. Schema
    // lives in src/output-schema.js (canonicalizer + validator + parser).
    // Bound into artifact_hash below via output_schema_hash with the W460
    // conditional-slot pattern: absence / null / {} all canonicalize to null
    // and skip the slot, so pre-W809 artifacts stay byte-identical when
    // rebuilt. Schema version stamp lets verifiers detect spec migrations.
    output_schema: _output_schema_canon,
    output_schema_spec_version: _output_schema_canon ? OUTPUT_SCHEMA_VERSION : undefined,
    // W736 - Guardrail Compilation. Brand-safety rules bake into the manifest
    // as HARD constraints (NOT training signal). Each entry is
    // { name, pattern, action }; pattern accepts keyword:/glob:/raw regex,
    // action is one of block|warn|rewrite. Runtime enforcement lives in
    // src/router.js (the /v1/chat/completions wrapper calls enforceGuardrails
    // against this exact array). Verify-time replay lives in
    // verifyGuardrailsAgainstTraces. The W460 byte-stability rule: absent,
    // null, and [] all collapse to null here so pre-W736 artifacts rebuilt
    // without a guardrails block remain byte-identical to their original
    // artifact_hash. _guardrails_canon is null in that path; the conditional
    // slot in artifact_hash_input below is skipped to match.
    guardrails: _guardrails_canon,
    // W739 - Model lineage tracking. Each .kolm references its parent artifact
    // via this field. The W460 byte-stability law: when parent_cid is absent
    // the manifest MUST not carry the key at all (so manifest_hash + cid stay
    // byte-identical to pre-W739 artifacts). We use a conditional spread
    // (...(value ? {key:value} : {})) instead of `parent_cid: null` so a
    // JSON.stringify replayed against a pre-W739 artifact rebuilt without
    // a parent_cid produces the exact same bytes the original did. Walked
    // at runtime by walkLineage in src/artifact-lineage.js; diffed by
    // diffArtifacts in src/kolm-diff.js. Tested by W739 #9 byte-stability
    // lock-in.
    ...(_parent_cid_canon ? { parent_cid: _parent_cid_canon } : {}),
    // W769 - Data residency region. The W460 byte-stability law: when region
    // is absent the manifest MUST NOT carry the key at all (so manifest_hash
    // + cid + artifact_hash stay byte-identical to pre-W769 artifacts).
    // Conditional spread (...(value ? {key:value} : {})) mirrors the W739
    // parent_cid pattern verbatim - absent / null / '' all collapse to the
    // empty spread, and the matching artifact_hash_input slot below is
    // skipped in lockstep. Schema lives in src/data-residency.js (REGIONS
    // taxonomy + DEFAULT_REGION). Bound into artifact_hash below via
    // region_hash with the W460 conditional-slot pattern so any post-build
    // residency tamper (swap from EU_WEST → US_EAST after sign-time) breaks
    // the receipt chain.
    ...(_region_canon ? { region: _region_canon } : {}),
    // W786 - Sustainability badge. Conditional-spread per the W460 byte-
    // stability law: absent (training_stats.gpu_hours not supplied) →
    // _w786_badge is null → key is OMITTED from the manifest entirely so
    // pre-W786 artifacts rebuilt without gpu_hours remain byte-identical
    // to their original manifest_hash + artifact_hash. NOT bound into
    // artifact_hash_input below: badge is hygiene metadata (MODELED
    // estimate, never measured) so receipt.json does not seal it.
    ...(_w786_badge ? { sustainability_badge: _w786_badge } : {}),
    k_score: k_score || null,  // patched after zipping for the size_bytes axis
    ship_gate_overridden: allow_below_gate === true ? true : undefined,
    // finalized-c1 — conformal ship-gate verdict. Conditional spread (W460
    // pattern): absent when the overlay is off so legacy artifacts stay
    // byte-identical; present only when KOLM_KSCORE_CONFORMAL=1 produced a
    // bounded decision. Operational metadata, not folded into artifact_hash.
    ...(ship_decision ? { ship_decision } : {}),
    license: normalizeLicense(license),
    // Wave 161 (Q+8) - signature policy. Every modern artifact ships with
    // Ed25519 by default (Wave 149); the policy field records this stance so
    // a verifier (or a downstream tenant procurement gate) can REJECT an
    // HMAC-only re-issuance of the same task. Default true unless Ed25519 is
    // explicitly disabled at build time (KOLM_ED25519_DISABLE=1) or the
    // tenant has explicitly opted out of policy enforcement
    // (KOLM_POLICY_OPT_OUT=1). The matching binder check #17 reads this
    // field PLUS the verifier-side env KOLM_REQUIRE_ED25519=1 - either side
    // can demand Ed25519; both sides false means HMAC-only is acceptable.
    //
    // Wave 162 (Q+9) - Rekor transparency policy. Sigstore is dry-run by
    // default (Wave 150 emits a structurally-valid bundle that verifies
    // offline, but is not pinned to a public transparency log). Setting
    // KOLM_REKOR_REQUIRE=1 at build time flips this to a CONTRACT - the
    // build will fail unless KOLM_SIGSTORE_REKOR_URL is set AND the Rekor
    // submission succeeds. Default false because Rekor pinning needs
    // network egress; most builds are offline by design. The matching
    // binder check #18 reads this field PLUS env KOLM_REQUIRE_REKOR=1 - 
    // either side can demand a pinned bundle; both sides false means a
    // dry-run sigstore block is acceptable.
    policy: {
      require_ed25519: process.env.KOLM_ED25519_DISABLE !== '1'
        && process.env.KOLM_POLICY_OPT_OUT !== '1',
      require_rekor: process.env.KOLM_REKOR_REQUIRE === '1'
        && process.env.KOLM_POLICY_OPT_OUT !== '1',
    },
    cid,
    hashes,
    // W367 - entry block. Names the executable file inside the zip plus its
    // sha256 + the host runtimes it targets. Verifiers reject any rule-class
    // artifact whose entry file is missing or whose entry_sha256 drifts
    // (artifact.no_executable_bundle). Absent (null) for distilled_model
    // artifacts whose executable IS the model weights themselves.
    entry: recipe_bundle_mjs ? {
      file: bundle_filename,
      sha256: sha256(Buffer.from(recipe_bundle_mjs)),
      runtime: 'node>=18 | bun>=1 | deno>=1.40',
      class: _finalClass,
      export: 'default',
    } : null,
  };
  // Wave 151 - validate the rolled-up artifact_class against the manifest
  // contents. A `distilled_model` claim with no weights, a `compiled_rule`
  // claim with no compiled_targets, or a `synthesized_rule` claim with no
  // teacher attribution all throw here, before the receipt is signed.
  const classCheck = validateArtifactClass(manifest);
  if (!classCheck.ok) {
    throw new Error(`artifact class validation failed: ${classCheck.reason}`);
  }
  const manifest_json = JSON.stringify(manifest, null, 2);
  const manifest_hash = sha256(Buffer.from(manifest_json));

  // The artifact_hash is the sha256 of the canonical join of every file
  // we are about to put in the zip (excluding signature.sig and receipt.json
  // which seal it). Computing it here lets the receipt bind to the artifact
  // *before* the zip is finalised, while the manifest_hash anchors the
  // legacy signature.sig.
  //
  // Wave F - compiled_rule artifacts add a `compiled_targets_hash` field over
  // the canonical compiled_targets manifest block (per-recipe filename +
  // source_hash + bytes). The field is omitted for non-compiled artifacts so
  // existing CIDs and artifact_hashes stay byte-stable.
  const artifact_hash_input = {
    manifest_hash,
    model_pointer_hash: manifest.hashes.model_pointer,
    recipes_json_hash: manifest.hashes.recipes_json,
    lora_bin_hash: manifest.hashes.lora_bin,
    index_bin_hash: manifest.hashes.index_bin,
    evals_json_hash: eval_set_hash,
  };
  if (compiled_targets_block) {
    artifact_hash_input.compiled_targets_hash = sha256(canonicalJson(compiled_targets_block));
  }
  // Wave 409q - bind honest binaries[] into artifact_hash. Empty array hashes
  // to a stable canonical "[]" so the chain is byte-stable for non-compiled
  // artifacts; any post-build mutation (added entry, dropped one, swapped
  // sha256, swapped filename) breaks the receipt chain.
  if (Array.isArray(binaries) && binaries.length > 0) {
    artifact_hash_input.binaries_hash = sha256(canonicalJson(
      binaries.map(b => ({
        target: b.target, kind: b.kind, recipe_id: b.recipe_id,
        filename: b.filename, sha256: b.sha256, size: b.size,
      }))
    ));
  }
  // Wave V - bind capability/lineage/IR/attestation into the artifact hash so
  // tampering with any of them after seal-time breaks the receipt chain.
  if (capability_block) {
    artifact_hash_input.capability_hash = capability_block.hash;
  }
  if (lineage_block) {
    artifact_hash_input.lineage_hash = lineage_block.hash;
  }
  // Wave 146 - bind export block into artifact_hash. The block's hash is the
  // short hash over its own canonical contents (see export-provenance.js
  // buildExportBlock); tamper with any target sha256, the block hash drifts,
  // the artifact hash drifts, the receipt chain breaks.
  if (export_block) {
    artifact_hash_input.export_hash = export_block.hash;
  }
  // Wave 147 - bind moe block into artifact_hash. Same drift-propagation as
  // export: tamper any expert sha256, the moe block hash drifts, the
  // artifact hash drifts, the receipt chain breaks. The bundled expert
  // files themselves also folded into extra_files_hash below for double
  // anchoring.
  if (moe_block) {
    artifact_hash_input.moe_hash = moe_block.hash;
  }
  // Wave 148 - bind pretokenize block into artifact_hash. Same drift-
  // propagation as moe/export: tamper either binary, the file's sha256 drifts,
  // the block hash drifts, the artifact hash drifts, the receipt chain breaks.
  if (pretokenize_block) {
    artifact_hash_input.pretokenize_hash = pretokenize_block.hash;
  }
  // Wave 164 - bind external/adversarial holdout block into artifact_hash.
  // Tamper with any holdout's file_sha256, normalized_hash, or recorded
  // accuracy: the block hash drifts, the artifact hash drifts, every
  // signature breaks. The holdout JSONLs themselves live under repo root
  // (holdouts/<kind>/<name>.jsonl), not inside the .kolm zip, because they
  // are independent corpora that a third party can re-download and re-anchor
  // against catalog.json's expected_sha256.
  if (external_holdout_block) {
    artifact_hash_input.external_holdout_hash = external_holdout_block.hash;
  }
  // Wave 165 - bind tenant shadow corpus blocks into artifact_hash. Hash
  // over the canonical ordered array of per-corpus block hashes so any
  // post-build mutation (added corpus, dropped corpus, swapped corpus_sha256,
  // edited accuracy) breaks the receipt chain. The hash binds only the
  // fingerprint, never the corpus bytes - those stay on tenant storage.
  if (tenant_shadow_blocks && tenant_shadow_blocks.length > 0) {
    artifact_hash_input.tenant_shadow_corpus_hash = sha256(canonicalJson(
      tenant_shadow_blocks.map(b => ({ tenant_id: b.tenant_id, corpus_id: b.corpus_id, hash: b.hash }))
    ));
  }
  // Wave 166 (N+7) - bind auditor attestation blocks into artifact_hash. Hash
  // over the canonical ordered array of per-block {auditor_id, key_fingerprint,
  // hash} tuples so any post-build mutation (added attestation, dropped one,
  // swapped signature, edited claimed eval_score) breaks the receipt chain.
  // The auditor's Ed25519 signature inside each block is the third-party
  // anchor; THIS artifact_hash binding is what stops a tamperer from quietly
  // removing an attestation after the fact.
  if (auditor_attestation_blocks && auditor_attestation_blocks.length > 0) {
    artifact_hash_input.auditor_attestation_hash = sha256(canonicalJson(
      auditor_attestation_blocks.map(b => ({ auditor_id: b.auditor_id, key_fingerprint: b.key_fingerprint, hash: b.hash }))
    ));
  }
  // Wave 167 (M+4) - bind supersession block into artifact_hash via the
  // block's own short hash. Tamper with predecessor_artifact_hash, reason,
  // supersession_date, drift_signals, etc.: the block hash drifts → artifact
  // hash drifts → every downstream signature breaks. This is what makes the
  // supersession chain auditable end-to-end: a verifier walking from any
  // artifact back to its genesis can confirm every successor was legitimately
  // signed against the predecessor it claims to replace.
  if (supersession_block) {
    artifact_hash_input.supersession_hash = supersession_block.hash;
  }
  // Wave 167 (M+3) - bind drift report into artifact_hash. Same pattern as
  // supersession: tamper the verdict, the breach_count, any signal value, and
  // the block hash drifts → artifact hash drifts → all signatures break.
  if (drift_report_block) {
    artifact_hash_input.drift_report_hash = drift_report_block.hash;
  }
  // finalized-c6 - bind the contamination-impact gate block into artifact_hash.
  // Conditional: only when the overlay was armed AND produced a block, so the
  // default chain is byte-identical. Tamper with the corrected accuracy/composite
  // and the canonical hash drifts -> artifact hash drifts -> signatures break.
  if (contamination_impact_block) {
    artifact_hash_input.contamination_impact_hash = sha256(canonicalJson(contamination_impact_block));
  }
  if (workflow_ir_json) {
    artifact_hash_input.workflow_ir_hash = manifest.hashes.workflow_ir;
  }
  if (attestation_report_json) {
    artifact_hash_input.attestation_report_hash = manifest.hashes.attestation_report;
  }
  if (confidential_compute_block) {
    artifact_hash_input.confidential_compute_hash = sha256(canonicalJson(confidential_compute_block));
  }
  // W719 - bind the DAQ mixed-precision profile into artifact_hash so any
  // post-build tamper of the per-layer schedule (added layer, dropped one,
  // changed weight_bits, swapped scale_mode, etc.) breaks the receipt
  // chain. Same pattern as confidential_compute_hash (W460) - keyed only
  // when the field is actually present so existing artifacts that did not
  // ship a DAQ profile remain byte-stable.
  if (Array.isArray(daq_profile) && daq_profile.length > 0) {
    artifact_hash_input.mixed_precision_profile_hash = sha256(canonicalJson(daq_profile));
  }
  // finalized-c5 - bind the mixed-precision read-back proof + importance signal +
  // calibration provenance into artifact_hash. Same W460 conditional-slot pattern
  // as mixed_precision_profile_hash: keyed ONLY when the corresponding block is a
  // non-empty object so pre-finalized-c5 artifacts that never carried these blocks
  // remain byte-identical when rebuilt. This extends the moat chain to cover the
  // applied==requested schedule proof, the per-layer importance signal that drove
  // the allocation, and the reproducible calibration regime - so a tamper of any
  // of them (e.g. flipping schedule_honored, swapping a sensitivity, editing the
  // calibration provenance_hash) breaks every signature down the chain.
  if (mixed_precision_proof && typeof mixed_precision_proof === 'object'
      && Object.keys(mixed_precision_proof).length > 0) {
    artifact_hash_input.mixed_precision_proof_hash = sha256(canonicalJson(mixed_precision_proof));
  }
  if (importance_signal && typeof importance_signal === 'object'
      && Object.keys(importance_signal).length > 0) {
    artifact_hash_input.importance_signal_hash = sha256(canonicalJson(importance_signal));
  }
  if (calibration_provenance && typeof calibration_provenance === 'object'
      && Object.keys(calibration_provenance).length > 0) {
    artifact_hash_input.calibration_provenance_hash = sha256(canonicalJson(calibration_provenance));
  }
  // W721 - bind the TSAC sparsity_profile into artifact_hash so any
  // post-build tamper of the per-(layer,head) kernel selection (added
  // entry, dropped one, swapped prefill_pattern, swapped decode_policy,
  // tweaked page_topk/sink_keep/dense_fallback_threshold, etc.) breaks
  // the receipt chain. Mirrors the W460 confidential_compute_hash slot:
  // keyed ONLY when the field is non-null AND non-empty so existing
  // artifacts (which never carried a sparsity_profile) remain
  // byte-stable. CRITICAL - never unconditionally add to the hash chain;
  // that would re-hash every legacy .kolm at the next verify pass.
  if (sparsity_profile && typeof sparsity_profile === 'object'
      && Object.keys(sparsity_profile).length > 0) {
    artifact_hash_input.sparsity_profile_hash = sha256(canonicalJson(sparsity_profile));
  }
  // W722 - bind the ITKV kv_profile into artifact_hash so any post-build
  // mutation of the per-token-class precision schedule breaks the receipt
  // chain. Mirrors the W460 confidential_compute_hash + W721
  // sparsity_profile_hash conditional-slot pattern: keyed only when a
  // non-empty profile is present, so pre-W722 artifacts (which never
  // carried a kv_profile) remain byte-identical when rebuilt.
  if (kv_profile && typeof kv_profile === 'object'
      && Object.keys(kv_profile).length > 0) {
    artifact_hash_input.kv_profile_hash = sha256(canonicalJson(kv_profile));
  }
  // W809 - bind the structured-output schema spec into artifact_hash so any
  // post-build swap of the canonical output_schema breaks the receipt chain.
  // Mirrors the W460 confidential_compute_hash + W721 sparsity_profile_hash +
  // W722 kv_profile_hash conditional-slot pattern: keyed only when canonical
  // spec is non-null (absent / null / {} all collapse to null in
  // canonicalizeOutputSchemaSpec), so pre-W809 artifacts remain byte-identical
  // when rebuilt.
  if (_output_schema_canon !== null) {
    artifact_hash_input.output_schema_hash = sha256(canonicalJson(_output_schema_canon));
  }
  // W736 - bind the guardrails block into artifact_hash so any post-build
  // mutation of a rule (added entry, dropped one, swapped action/pattern,
  // edited name, replacement-string drift) breaks the receipt chain. Mirrors
  // the W460 confidential_compute_hash + W721 sparsity_profile_hash + W722
  // kv_profile_hash + W809 output_schema_hash conditional-slot pattern: the
  // slot is keyed ONLY when the canonical block is non-null + non-empty,
  // so pre-W736 artifacts (which never carried guardrails) rebuilt without
  // a guardrails block remain byte-identical to their original artifact_hash.
  // hashGuardrailsW736 returns null on empty input - we re-check Array.isArray
  // + length here as defense-in-depth so a future caller passing {} or 0
  // does not silently key the slot.
  if (Array.isArray(_guardrails_canon) && _guardrails_canon.length > 0) {
    const gh = hashGuardrailsW736(_guardrails_canon);
    if (gh) artifact_hash_input.guardrails_hash = gh;
  }
  // W739 - bind parent_cid into artifact_hash so a tamperer cannot rewrite
  // lineage history without invalidating every descendant's receipt chain.
  // Mirrors the W460 confidential_compute_hash + W721 sparsity_profile_hash
  // + W722 kv_profile_hash + W736 guardrails_hash conditional-slot pattern:
  // keyed ONLY when _parent_cid_canon is a non-empty hex string so pre-W739
  // artifacts (which never carried a parent_cid) rebuilt without one remain
  // byte-identical to their original artifact_hash. The hash position is
  // FIXED immediately after guardrails_hash per the W739 spec so the
  // canonical key order in artifact_hash_input stays predictable.
  if (_parent_cid_canon) {
    artifact_hash_input.parent_cid = _parent_cid_canon;
  }
  // W769 - bind the residency region into artifact_hash so a tamperer
  // cannot rewrite the residency claim on a signed artifact without
  // invalidating the receipt chain. Mirrors the W460 confidential_compute_hash
  // + W721 sparsity_profile_hash + W722 kv_profile_hash + W736 guardrails_hash
  // + W739 parent_cid conditional-slot pattern: keyed ONLY when _region_canon
  // is a non-empty string so pre-W769 artifacts (which never carried a
  // region) rebuilt without one remain byte-identical to their original
  // artifact_hash. CRITICAL: this slot's omission for absent/null/empty
  // region inputs is the W460 byte-stability contract; never key it
  // unconditionally.
  if (_region_canon) {
    artifact_hash_input.region_hash = sha256(Buffer.from(_region_canon));
  }
  if (hashes.extra_files) {
    artifact_hash_input.extra_files_hash = sha256(canonicalJson(hashes.extra_files));
  }
  // W367 - bind recipe.bundle.mjs into artifact_hash so a tamperer can't swap
  // the executable bundle without breaking every signature down the chain.
  if (recipe_bundle_mjs) {
    artifact_hash_input.recipe_bundle_mjs_hash = hashes.recipe_bundle_mjs;
  }
  // W457 - bind model_weights into artifact_hash so swapping weight bytes
  // (e.g. replacing a vetted Qwen 0.5B GGUF with a tampered build) breaks
  // every signature down the chain. The matching verifier check rehashes the
  // declared file and refuses on mismatch.
  if (_modelWeightsRecord) {
    artifact_hash_input.model_weights_hash = _modelWeightsRecord.sha256;
  }
  if (_modelWeightArtifactManifest) {
    artifact_hash_input.model_weight_artifact_manifest_hash = hashes.model_weight_artifact_manifest;
  }
  const artifact_hash = sha256(canonicalJson(artifact_hash_input));

  // Build the HMAC chain. Each step seals the previous step's output.
  const stepSeal = (step, input_hash, output_hash) => {
    const hmac = crypto.createHmac('sha256', secret)
      .update(canonicalJson({ step, input_hash, output_hash }))
      .digest('hex');
    return { step, input_hash, output_hash, hmac };
  };
  const taskHash = sha256(canonicalJson({ task: task || '' }));
  // The 'seeds' step in the chain MUST bind to a real seeds_hash when a seed
  // split was performed. Falling back to a hash of training_stats (the
  // pre-Wave-144 behavior) made the step content-free and broke the receipt
  // chain's purpose. Honest fallback when no seed split happened: hash of
  // the (now explicit) self_generated provenance block, so a verifier can
  // tell at-a-glance that the artifact didn't use a real seed gate.
  const seedsHash = seed_provenance && seed_provenance.seeds_hash
    ? seed_provenance.seeds_hash
    : sha256(canonicalJson({ eval_source: 'self_generated', training: training_stats || null }));
  const recipesHash = manifest.hashes.recipes_json;
  const evalsHash = eval_set_hash;
  const chain = [
    stepSeal('task',    sha256(canonicalJson({ spec: ARTIFACT_SPEC })), taskHash),
    stepSeal('seeds',   taskHash,    seedsHash),
    stepSeal('recipes', seedsHash,   recipesHash),
    stepSeal('evals',   recipesHash, evalsHash),
    stepSeal('package', evalsHash,   artifact_hash),
  ];

  // Receipt body - bound by the HMAC over (artifact_hash, eval_set_hash,
  // eval_score, judge_id, chain). The signed_by field identifies the current
  // HMAC key namespace; future asymmetric signatures should use a new
  // signature_alg value. Verifiers re-check both the chain and the body HMAC.
  const issued_at = buildTimestamp;
  const receipt_id = receiptIdInput || (repro.enabled
    ? deterministicReceiptId({ job_id, cid, artifact_hash, eval_set_hash, issued_at })
    : crypto.randomUUID());

  // Wave 409aa - auditor mandate: receipts MUST surface the dataset/holdout
  // hashes + split seed + the source-event hashes the dataset rolled up from
  // + a build-toolchain block. These fields ride INSIDE the receipt body so
  // the HMAC + Ed25519 + Sigstore signatures all cover them - tampering with
  // any one breaks every signature down the chain.
  //
  //   event_source_hashes[] : per-source-event sha256 the seed split rolled up
  //                           from. When seed_provenance supplies the array we
  //                           use it as-is; otherwise we surface an empty array
  //                           (the legacy/no-seeds path).
  //   dataset_hash          : alias of seed_provenance.seeds_hash. Auditors
  //                           reading the receipt expect the dataset's own hash
  //                           at the receipt top-level.
  //   train_hash            : alias of seed_provenance.train_hash.
  //   holdout_hash          : alias of seed_provenance.holdout_hash.
  //   split_seed            : alias of seed_provenance.split_seed.
  //   runtime_target        : echoes manifest.runtime_target so a verifier can
  //                           cross-check that the receipt was issued for the
  //                           SAME runtime the dispatcher will pick.
  //   artifact_files[]      : canonical-sorted [{filename, sha256}] over every
  //                           file the .kolm bundles. Lets a verifier diff the
  //                           opened zip against the receipt without re-parsing
  //                           manifest.hashes (which has a mixed object shape).
  //   build_toolchain       : {node_version, platform, arch, kolm_version,
  //                           runtime_target, signed_at}. Identifies the
  //                           machine that signed the artifact. Reproducibility
  //                           audits compare this to their own toolchain.
  const eventSourceHashes = Array.isArray(seed_provenance?.event_source_hashes)
    ? seed_provenance.event_source_hashes.map((h) => String(h))
    : [];
  const artifactFiles = (() => {
    const rows = [];
    // Flat hash slots first (model_pointer, recipes_json, lora_bin, index_bin,
    // evals_json, recipe_bundle_mjs, workflow_ir, attestation_report).
    for (const k of Object.keys(hashes).sort()) {
      if (k === 'extra_files') continue;
      const v = hashes[k];
      if (typeof v === 'string') rows.push({ filename: k, sha256: v });
    }
    // Extra files (sorted by filename for canonical order).
    if (hashes.extra_files && typeof hashes.extra_files === 'object') {
      for (const fn of Object.keys(hashes.extra_files).sort()) {
        rows.push({ filename: fn, sha256: hashes.extra_files[fn] });
      }
    }
    return rows;
  })();
  const _runtime_target = manifest.runtime_target || 'js';
  const build_toolchain = {
    node_version: process.version,
    platform: process.platform,
    arch: process.arch,
    kolm_version: '0.1',
    runtime_target: _runtime_target,
    signed_at: issued_at,
  };

  const receiptBody = {
    kolm_version: '0.1',
    receipt_id,
    cid,
    artifact_hash,
    eval_set_hash,
    eval_score: manifest.eval_score,
    judge_id: _judgeId,
    tier: _tier,
    chain,
    anchors: [],
    // W409aa receipt-hardening fields. Added INSIDE the receipt body so every
    // signature scheme (HMAC, Ed25519, Sigstore) covers them; mutation breaks
    // verification.
    event_source_hashes: eventSourceHashes,
    dataset_hash: seed_provenance?.seeds_hash || null,
    train_hash: seed_provenance?.train_hash || null,
    holdout_hash: seed_provenance?.holdout_hash || null,
    split_seed: seed_provenance?.split_seed || null,
    runtime_target: _runtime_target,
    artifact_files: artifactFiles,
    build_toolchain,
  };

  // Wave 149 - Ed25519 is now the DEFAULT signature alg.
  //
  // Prior to Wave 149 the Ed25519 block only appeared when
  // `KOLM_ED25519_PRIVATE_KEY` was explicitly set. Per the Wave 144 plan
  // §Q+8 the HMAC-only path was downgraded to an integrity check; the
  // third-party-verifiable signature has to be the default.
  // `loadEd25519DefaultSigner()` (a) prefers the env-var key when present,
  // (b) falls back to a per-machine cached key at ~/.kolm/signing-key.pem
  // (generated on first build, persists across subsequent builds so the
  // fingerprint is stable). Set `KOLM_ED25519_DISABLE=1` for legacy
  // HMAC-only signing.
  //
  // Field order MUST be: load signer → set signature_alg + signed_by →
  // compute HMAC over the upgraded body → add HMAC `signature` field →
  // compute Ed25519 over canonical(body INCLUDING HMAC signature) → add
  // `signature_ed25519` block. The verifier strips `signature_ed25519`
  // and `signature` independently and re-canonicalizes both.
  // Wave 150 - sigstore (cosign-compatible) bundle is layered on top of
  // Ed25519 when both the Ed25519 signer is present AND sigstore is not
  // explicitly disabled. signature_alg upgrades to reflect every active
  // signature scheme so verifiers can quickly decide which checks apply.
  const sigstoreEnabled = ed25519Signer && !isSigstoreDisabled();
  if (sigstoreEnabled) {
    receiptBody.signature_alg = 'sigstore+ed25519+hmac-sha256';
    receiptBody.signed_at = issued_at;
    receiptBody.signed_by = `ed25519:${ed25519Signer.key_fingerprint}`;
  } else if (ed25519Signer) {
    receiptBody.signature_alg = 'ed25519+hmac-sha256';
    receiptBody.signed_at = issued_at;
    receiptBody.signed_by = `ed25519:${ed25519Signer.key_fingerprint}`;
  } else {
    receiptBody.signature_alg = 'hmac-sha256';
    receiptBody.signed_at = issued_at;
    receiptBody.signed_by = 'kolm-dev-hmac-1';
  }

  const bodyCanon = canonicalJson(receiptBody);
  const bodySig = crypto.createHmac('sha256', secret).update(bodyCanon).digest('hex');
  receiptBody.signature = bodySig;

  if (ed25519Signer) {
    try {
      const ed25519Payload = canonicalJson(receiptBody);
      receiptBody.signature_ed25519 = buildEd25519Block({
        privateKey: ed25519Signer.privateKey,
        publicKey: ed25519Signer.publicKey,
        key_fingerprint: ed25519Signer.key_fingerprint,
        payloadCanonical: ed25519Payload,
        signed_at: issued_at,
      });
    } catch (e) {
      console.error(`[artifact] WARNING: ed25519 sign skipped: ${e.message}`);
    }
  }

  // Wave 150 - sigstore layer. Always dry-run-by-default at build time
  // (KOLM_SIGSTORE_REKOR_URL not consulted here because that's an async
  // network call; `kolm sigstore-attest` upgrades to a Rekor-pinned bundle
  // post-build). The bundle still verifies offline against the embedded
  // Ed25519 public key, so even a dry-run block is structurally useful.
  if (sigstoreEnabled) {
    try {
      const sigstorePayload = canonicalJson(receiptBody);
      receiptBody.signature_sigstore = buildSigstoreBundle({
        privateKey: ed25519Signer.privateKey,
        publicKey: ed25519Signer.publicKey,
        key_fingerprint: ed25519Signer.key_fingerprint,
        payloadCanonical: sigstorePayload,
        signed_at: issued_at,
      });
    } catch (e) {
      console.error(`[artifact] WARNING: sigstore bundle skipped: ${e.message}`);
    }
  }

  const receipt_json = JSON.stringify(receiptBody, null, 2);

  // Legacy signature.sig - kept for back-compat with v0 verifiers. The new
  // receipt.json supersedes it.
  const sig_payload = canonicalJson({
    spec: ARTIFACT_SPEC,
    manifest_hash,
    job_id,
    artifact_hash,
    eval_set_hash,
    eval_score: manifest.eval_score,
    judge_id: _judgeId,
  });
  const hmac = crypto.createHmac('sha256', secret).update(sig_payload).digest('hex');
  const signature = JSON.stringify({
    spec: ARTIFACT_SPEC,
    job_id,
    manifest_hash,
    artifact_hash,
    eval_set_hash,
    eval_score: manifest.eval_score,
    judge_id: _judgeId,
    hmac_alg: 'HMAC-SHA256',
    hmac,
    issued_at,
  }, null, 2);

  // Build an artifact-scoped provenance credential. Signed with the same
  // secret as the receipt chain. Shipped as a sidecar `credential.json` in
  // the zip (not embedded in receipt.json - receipt.json is already signed,
  // and we don't want to invalidate that signature).
  const credential = buildArtifactCredential({
    secret,
    artifact_hash,
    cid,
    k_score: manifest.k_score ?? null,
    base_model,
    signed_at: issued_at,
    judge_id: _judgeId,
    tier: _tier,
    ingredients: [],
  });
  const credential_json = JSON.stringify(credential, null, 2);

  // Physically drop padding entries when there is no real content. Loader
  // (artifact-runner.js) already tolerates missing optional files. The
  // manifest.hashes object still records EMPTY_SHA for absent slots so the
  // CID computation and any external integrity scanner sees a consistent
  // schema. Honest "this slot is empty" beats fake "this slot contains an
  // empty placeholder pretending to be a model."
  const files = [
    { filename: 'manifest.json',    content: Buffer.from(manifest_json) },
    { filename: 'recipes.json',     content: Buffer.from(recipes_json) },
    { filename: 'evals.json',       content: Buffer.from(evals_json) },
    { filename: 'signature.sig',    content: Buffer.from(signature) },
    { filename: 'receipt.json',     content: Buffer.from(receipt_json) },
    { filename: 'credential.json',  content: Buffer.from(credential_json) },
  ];
  if (model_pointer != null) files.push({ filename: 'model.gguf', content: Buffer.from(model_pointer) });
  if (has_pack) files.push({ filename: 'lora.bin', content: lora_bin });
  if (has_index) files.push({ filename: 'index.sqlite-vec', content: index_bin });
  // W367 - emit the executable bundle. Without this, the artifact is metadata
  // only and the homepage hero claim "same file runs on a laptop, a phone, or
  // an air-gapped server" is a lie. The bundle is a self-contained ESM module
  // any Node 18+ / Bun 1+ / Deno 1.40+ host can `import` directly.
  if (recipe_bundle_mjs) files.push({ filename: bundle_filename, content: Buffer.from(recipe_bundle_mjs) });
  // W457 - emit bundled model weights (gguf/onnx/wasm/native). The verifier
  // refuses any manifest whose runtime_target is a weight class but whose
  // declared *_path entry is absent from the zip, so PATH B (rule-class) and
  // PATH A (weight-class) cannot be mixed up. The runtime_target_config path
  // must match the bundled filename - we don't blindly trust the caller; the
  // check below errors at build time if they drift.
  if (_modelWeightsRecord) {
    const _isWeightClass = ['gguf', 'onnx', 'wasm', 'native'].includes(_runtimeTargetDeclared);
    if (!_isWeightClass) {
      throw new Error(`model_weights supplied but runtime_target=${JSON.stringify(_runtimeTargetDeclared)} is not a weight class (gguf|onnx|wasm|native)`);
    }
    const _expectedPath = _runtimeTargetDeclared === 'gguf' ? _runtimeTargetConfig?.gguf_path
      : _runtimeTargetDeclared === 'onnx' ? _runtimeTargetConfig?.onnx_path
      : _runtimeTargetDeclared === 'wasm' ? 'target.wasm'
      : _entrypoint?.binary;
    if (!_expectedPath) {
      throw new Error(`runtime_target=${_runtimeTargetDeclared} requires a path in runtime_target_config (or entrypoint.binary for native)`);
    }
    if (_expectedPath !== _modelWeightsRecord.filename) {
      throw new Error(`model_weights.filename=${JSON.stringify(_modelWeightsRecord.filename)} does not match declared path ${JSON.stringify(_expectedPath)}; the verifier would refuse the bundle`);
    }
    files.push({ filename: _modelWeightsRecord.filename, content: _modelWeightsRecord.content });
    files.push({ filename: MODEL_WEIGHT_ARTIFACT_MANIFEST_FILENAME, content: Buffer.from(_modelWeightArtifactManifestJson, 'utf8') });
  } else if (['gguf', 'onnx', 'wasm', 'native'].includes(_runtimeTargetDeclared)) {
    // W457 - runtime_target declared a weight class but no model_weights was
    // supplied. This is the honest-failure path: the verifier would refuse the
    // bundle (rtCheck would see a missing entry), so refuse at build time
    // instead of shipping a known-broken artifact. Callers can pass
    // runtime_target='js' (the default) to ship a rule-class artifact.
    throw new Error(`runtime_target=${_runtimeTargetDeclared} requires model_weights={filename,content:Buffer} to bundle the matching weights; got none`);
  }
  // Wave F - emit the C and Rust sources for compiled_rule artifacts. They
  // are the source-of-truth the verifier rebuilds against. Wave G adds the
  // compiled binary alongside (target binary hash also enters the manifest).
  for (const f of compiled_target_files) files.push(f);
  // Wave V - emit workflow_ir.json and attestation_report.json so the verifier
  // can replay hashIr() / verifyAttestation() instead of trusting the manifest
  // claim. A claim without bundled evidence is treated as fail by binder.js.
  if (workflow_ir_json) files.push({ filename: 'workflow_ir.json', content: Buffer.from(workflow_ir_json) });
  if (attestation_report_json) files.push({ filename: 'attestation_report.json', content: Buffer.from(attestation_report_json) });
  // Wave 144 - append extra files (e.g. tokenizer.json) last so they don't
  // shift offsets of the load-bearing files above. Filename collisions with
  // the reserved set would silently shadow; we guard here.
  const RESERVED_FILENAMES = new Set(['manifest.json', 'recipes.json', 'signature.sig', 'evals.json', 'receipt.json', 'credential.json', 'model.gguf', 'lora.bin', 'index.sqlite-vec', 'workflow_ir.json', 'attestation_report.json', 'recipe.bundle.mjs', MODEL_WEIGHT_ARTIFACT_MANIFEST_FILENAME, 'provenance.intoto.dsse.json', 'model.sig.bundle']);
  // W457 - also reserve the bundled model_weights filename so an extra_files
  // entry can't silently shadow real weights with a tampered payload.
  if (_modelWeightsRecord) RESERVED_FILENAMES.add(_modelWeightsRecord.filename);
  for (const f of extra_files_list) {
    if (RESERVED_FILENAMES.has(f.filename)) {
      throw new Error(`extra_files: filename '${f.filename}' is reserved`);
    }
    // Pass through absPath when content is path-backed so packageArtifact()
    // can stream directly from disk. Hash already in hashes.extra_files.
    if (f.absPath && !Buffer.isBuffer(f.content)) {
      files.push({ filename: f.filename, absPath: f.absPath });
    } else {
      files.push({ filename: f.filename, content: f.content });
    }
  }

  // -------------------------------------------------------------------------
  // Model-signing sidecars (model-signing-standards). EMITTED LAST, over the
  // ACTUAL bundled member bytes, gated behind the Ed25519 signer:
  //
  //   provenance.intoto.dsse.json - a signed SLSA Provenance v1 DSSE envelope
  //     (cosign verify-attestation / slsa-verifier consume it offline).
  //   model.sig.bundle            - an OpenSSF Model-Signing (OMS) file manifest
  //     (`model-signing verify` accepts a kolm artifact).
  //
  // Both are SEALS over the bytes, exactly like signature.sig: they are added
  // AFTER artifact_hash and are NOT folded into manifest.hashes nor into
  // receipt.artifact_files, so they do NOT change artifact_hash / the CID. Their
  // subjects are the (member path, digest-over-real-bytes) pairs of the members
  // already in `files` - NOT the lineage-folded manifest slots - so a model
  // verifier pins the runnable artifact's actual bytes. signature.sig /
  // receipt.json / credential.json are themselves seals, so they are not
  // subjects of the seal.
  if (ed25519Signer) {
    const SEAL_FILES = new Set(['signature.sig', 'receipt.json', 'credential.json', 'provenance.intoto.dsse.json', 'model.sig.bundle']);
    const memberDigests = {};
    for (const f of files) {
      if (!f || typeof f.filename !== 'string' || SEAL_FILES.has(f.filename)) continue;
      let digest = null;
      try {
        if (Buffer.isBuffer(f.content)) digest = digestPair(f.content);
        else if (f.absPath) digest = digestFilePair(f.absPath);
      } catch { digest = null; }
      if (digest && digest.sha256) memberDigests[f.filename] = digest;
    }
    const memberList = Object.keys(memberDigests).sort().map((name) => ({ name, ...memberDigests[name] }));
    const documentationDigests = Object.keys(memberDigests).sort()
      .map((name) => {
        const role = inferModelDocumentationRole(name);
        return role ? { name, digest: memberDigests[name], role } : null;
      })
      .filter(Boolean);

    // SLSA Provenance v1 DSSE sidecar over the real member byte digests.
    try {
      const dsseJson = emitArtifactAttestation({
        ed25519Signer,
        manifest,
        hashes,
        lineage: lineage || null,
        artifact_hash,
        cid,
        jobId: job_id,
        issued_at,
        subjectDigests: memberDigests,
        documentationDigests,
      });
      files.push({ filename: 'provenance.intoto.dsse.json', content: Buffer.from(dsseJson) });
    } catch (e) {
      console.error(`[artifact] WARNING: SLSA DSSE sidecar skipped: ${e.message}`);
    }

    // OMS file-manifest bundle over the same real member byte digests.
    try {
      const omsBundle = toOmsArtifactManifest(memberList, ed25519Signer);
      files.push({ filename: 'model.sig.bundle', content: Buffer.from(JSON.stringify(omsBundle, null, 2)) });
    } catch (e) {
      console.error(`[artifact] WARNING: OMS bundle sidecar skipped: ${e.message}`);
    }
  }

  return {
    manifest,
    receipt: receiptBody,
    credential,
    artifact_hash,
    cid,
    eval_set_hash,
    files,
  };
}

// Stream the .kolm zip to a writable target (file path or HTTP response).
export function packageArtifact({ job_id, payload, outPath, source_date_epoch, reproducible_build }) {
  return new Promise((resolve, reject) => {
    const repro = reproducibleBuildContext({ source_date_epoch, reproducible_build });
    const zipDate = repro.enabled ? repro.date : null;
    const payloadFiles = repro.enabled
      ? payload.files.slice().sort((a, b) => String(a.filename).localeCompare(String(b.filename)))
      : payload.files;
    const target = outPath
      ? fs.createWriteStream(outPath)
      : null;
    const z = archiver('zip', { zlib: { level: 9 } });
    if (target) {
      z.pipe(target);
      target.on('close', () => resolve({ bytes: z.pointer() }));
    }
    z.on('warning', (e) => { if (e.code !== 'ENOENT') reject(e); });
    z.on('error', reject);
    for (const f of payloadFiles) {
      // Path-backed entries stream directly from disk so files larger than
      // Node's 2 GiB Buffer limit (e.g. Trinity Q4_K_M ~4 GiB) still pack.
      const source = (f.absPath && !f.content) ? fs.createReadStream(f.absPath) : f.content;
      z.append(source, zipDate ? { name: f.filename, date: zipDate } : { name: f.filename });
    }
    z.finalize();
    if (!target) {
      // caller will pipe z elsewhere
      resolve({ archive: z });
    }
  });
}

// Convenience: build + zip in one step. Returns the zip path.
//
// We zip twice when k_score is requested: once to measure size, then again
// with the size-aware K-score patched into the manifest. The double-zip is
// cheap (≤10ms for 5KB artifacts) and keeps the K-score honest - the size
// axis includes the K-score bytes themselves.
export async function buildAndZip({ job_id, task, base_model, recipes, lora_pointer, recall_namespace, training_stats, evals, outDir, outPath: outPathOverride, judge_id, tier, pack, index, target_device, train_device, license, artifact_class, seed_provenance, compiled_targets, capability, lineage, workflow_ir, attestation_report, extra_files, export: exportInput, moe: moeInput, pretokenize: pretokenizeInput, external_holdout: externalHoldoutInput, tenant_shadow_corpus: tenantShadowInput, auditor_attestation: auditorAttestationInput, supersession: supersessionInput, drift_report: driftReportInput, contamination_impact: contaminationImpactInput, allow_below_gate, binaries, compiled_binary, native_skip_reasons, runtime_target, runtime_target_config, model_weights, entrypoint, daq_profile, sparsity_profile, kv_profile, guardrails, parent_cid, region, runtime_passports, speculative_decoding, prompt_cache, continuous_batching, mixed_precision_proof, importance_signal, source_date_epoch, reproducible_build, receipt_id }) {
  requireSignSecret();
  // W457b (build-honors-out) - when an explicit outPath is supplied, write
  // the .kolm directly at the user-requested filename. Otherwise fall back
  // to the legacy `outDir/${job_id}.kolm` path. The override removes the
  // copy-rename step in spec-compile.js that (a) leaked `<job_id>.kolm` into
  // error messages and (b) was the actual line that raised EBUSY/EPERM on
  // Windows when the target was locked by another process.
  const dir = outPathOverride ? path.dirname(outPathOverride) : (outDir || path.join(os.tmpdir(), 'kolm-artifacts'));
  fs.mkdirSync(dir, { recursive: true });

  // Derive eval_score from the synthesis result. Pattern-mode synthesis
  // returns pass_rate_positive in [0..1]; the artifact tier defaults to
  // "recipe" (the only tier the Sprint-1 toolchain produces today - 
  // adapter/specialist/bundle land in later sprints).
  const accuracy = training_stats?.pass_rate_positive ?? (training_stats?.verifier_accepted ? 1.0 : 0.0);
  const coverage = evals && evals.coverage != null ? evals.coverage : accuracy;
  const eval_score = (evals && typeof evals.coverage === 'number') ? evals.coverage : accuracy;
  const _tier = tier || 'recipe';
  const _judgeId = judge_id || process.env.KOLM_JUDGE_ID || 'kolm-pattern-synth-1';

  // Wave V - when an attestation_report is supplied, run the verifier here
  // (async) and pass the resulting state into the sync buildPayload as
  // confidential_compute. The kind comes from the capability block when the
  // contract demands TEE; otherwise the caller must pre-supply a kind via
  // attestation_report.kind (used only as a hint to verifyAttestation).
  let confidential_compute = null;
  if (attestation_report) {
    const kind = capability?.attestation || attestation_report._kind || attestation_report.kind || null;
    if (!kind) {
      throw new Error('attestation_report supplied without an attestation kind (set capability.attestation or attestation_report._kind)');
    }
    confidential_compute = await verifyAttestation(kind, attestation_report);
  }

  const sharedBlocks = { capability, lineage, workflow_ir, attestation_report, confidential_compute, extra_files, export: exportInput, moe: moeInput, pretokenize: pretokenizeInput, external_holdout: externalHoldoutInput, tenant_shadow_corpus: tenantShadowInput, auditor_attestation: auditorAttestationInput, supersession: supersessionInput, drift_report: driftReportInput, contamination_impact: contaminationImpactInput, allow_below_gate, binaries, compiled_binary, native_skip_reasons, runtime_target, runtime_target_config, model_weights, entrypoint, daq_profile, sparsity_profile, kv_profile, guardrails, parent_cid, region, runtime_passports, speculative_decoding, prompt_cache, continuous_batching, mixed_precision_proof, importance_signal, source_date_epoch, reproducible_build, receipt_id };

  // W350 - temp-file cleanup registry. The two-pass build writes a probe zip
  // to measure its size before the K-score is embedded; on success the probe
  // is overwritten in place by the final zip (same outPath). On FAILURE
  // (ship-gate throw, Rekor pinning failure, anything else between Pass 1 and
  // the return) the probe zip used to leak - a half-baked .kolm with no
  // K-score in its manifest would sit in ~/.kolm/artifacts/. Track every
  // temp file we create and unlink them in the finally if `success` never
  // flips true.
  const cleanupOnFail = [];
  let success = false;
  // W457b - honor explicit outPath override so the on-disk artifact is named
  // exactly what the user asked for (no `<job_id>.kolm` intermediate). Wrap
  // the pre-flight write probe so a locked/permission-denied target produces
  // a clean, actionable error instead of an unhandled EPERM/EBUSY crash with
  // the intermediate job_id in the message.
  const outPath = outPathOverride || path.join(dir, `${job_id}.kolm`);
  if (outPathOverride) {
    try {
      // Probe the destination by opening write-only. fs.openSync returns
      // immediately on success and throws synchronously on permission /
      // lock / read-only-file-system errors so we can attach actionable
      // hints before the archiver pipe hits the same failure deep inside
      // async code (where the original stack trace was useless).
      const _fd = fs.openSync(outPath, 'w');
      fs.closeSync(_fd);
    } catch (e) {
      const code = (e && e.code) || 'UNKNOWN';
      if (code === 'EPERM' || code === 'EACCES' || code === 'EBUSY' || code === 'EROFS') {
        const hint = `${code} opening ${outPath}: is the file open in another process? Try a different --out or close the previous build.`;
        const err = new Error(hint);
        err.code = code;
        err.path = outPath;
        throw err;
      }
      throw e;
    }
  }
  try {
  // Pass 1 - zip to measure size.
  const probePayload = buildPayload({ job_id, task, base_model, recipes, lora_pointer, recall_namespace, training_stats, evals, judge_id: _judgeId, eval_score, tier: _tier, pack, index, target_device, train_device, license, artifact_class, seed_provenance, compiled_targets, ...sharedBlocks });
  cleanupOnFail.push(outPath);
  await packageArtifact({ job_id, payload: probePayload, outPath, source_date_epoch, reproducible_build });
  const probeBytes = fs.statSync(outPath).size;

  // K-score: derive accuracy/coverage/latency/cost from training stats and
  // any supplied evals. For Sprint 1 stub: pure-recipe artifacts have
  // cost=0 (no run-time API calls), latency = compiled-fn p50 ~50us, and
  // accuracy = synthesizer pass-rate. Coverage starts at the eval count
  // ratio; if no evals supplied, it equals accuracy (best-effort).
  //
  // Wave 145 - also passes optional V2 axes when training_stats carries them.
  // The distill-provenance bridge surfaces teacher_holdout_accuracy +
  // holdout_accuracy from the worker manifest, which makes the K-score
  // emit a v2 envelope with R + T axes (student-on-holdout / teacher-on-
  // holdout fidelity). v1-only callers continue to get a v1 envelope.
  const k_score = computeKScore({
    size_bytes: probeBytes,
    accuracy,
    coverage,
    p50_latency_us: training_stats?.latency_p50_us ?? 50,
    cost_usd_per_call: training_stats?.cost_usd_per_call ?? 0,
    holdout_accuracy: training_stats?.holdout_accuracy ?? null,
    teacher_holdout_accuracy: training_stats?.teacher_holdout_accuracy ?? null,
    subgroup_min_accuracy: training_stats?.subgroup_min_accuracy ?? null,
    joules_per_call: training_stats?.joules_per_call ?? null,
    eval_set_drift: training_stats?.eval_set_drift ?? null,
  });

  // Pass 2 - repackage with the K-score in the manifest. The K-score size
  // axis reflects the probe zip size (Pass 1); the final zip is typically
  // 64-100 bytes larger because the manifest now embeds the K-score JSON.
  // We do NOT mutate the manifest after writing - the returned manifest is
  // exactly what's inside the on-disk artifact, so a verifier recomputing
  // K-score from the artifact bytes will reproduce manifest.k_score
  // deterministically (size_bytes axis matches the embedded value).
  const finalPayload = buildPayload({ job_id, task, base_model, recipes, lora_pointer, recall_namespace, training_stats, evals, k_score, judge_id: _judgeId, eval_score, tier: _tier, pack, index, target_device, train_device, license, artifact_class, seed_provenance, compiled_targets, ...sharedBlocks });
  await packageArtifact({ job_id, payload: finalPayload, outPath, source_date_epoch, reproducible_build });

  // Wave 162 (Q+9) - opportunistic Rekor pinning. The build is sync; the
  // sigstore block emitted inside buildPayload is dry-run by design. When
  // KOLM_SIGSTORE_REKOR_URL is set, post the bundle's digest+sig+pubkey to
  // that Rekor instance now (async, post-zip) and rewrite the artifact in
  // place with the pinned bundle. If manifest.policy.require_rekor=true and
  // the submission fails, the build fails - that's the contract. Otherwise
  // log a warning and proceed with the dry-run artifact (the artifact is
  // still structurally valid + locally verifiable; the user can rerun
  // `kolm sigstore-attest <artifact>` later).
  let rekorAttestation = null;
  const requiresRekor = !!finalPayload.manifest.policy?.require_rekor;
  const hasRekorUrl = !!sigstoreRekorUrl();
  const sigstorePresent = !!finalPayload.receipt.signature_sigstore;
  if (sigstorePresent && (hasRekorUrl || requiresRekor)) {
    if (!hasRekorUrl && requiresRekor) {
      throw new Error('policy.require_rekor=true but KOLM_SIGSTORE_REKOR_URL is unset - cannot pin sigstore bundle to a transparency log');
    }
    try {
      rekorAttestation = await attestArtifactWithRekor(outPath);
      finalPayload.receipt.signature_sigstore = {
        ...finalPayload.receipt.signature_sigstore,
        rekor_log_entry: {
          uuid: rekorAttestation.rekor_uuid,
          logIndex: rekorAttestation.rekor_log_index,
          integratedTime: rekorAttestation.integrated_time,
          logID: rekorAttestation.log_id,
          rekor_url: rekorAttestation.rekor_url,
        },
        dry_run: false,
      };
    } catch (e) {
      if (requiresRekor) {
        throw new Error(`policy.require_rekor=true but Rekor pinning failed: ${e.message}`);
      }
      console.error(`[artifact] WARNING: sigstore Rekor pinning skipped: ${e.message}`);
    }
  }
  const stat = fs.statSync(outPath);

  success = true;
  return {
    outPath,
    manifest: finalPayload.manifest,
    receipt: finalPayload.receipt,
    credential: finalPayload.credential,
    artifact_hash: finalPayload.artifact_hash,
    cid: finalPayload.cid,
    eval_set_hash: finalPayload.eval_set_hash,
    bytes: stat.size,
    k_score: finalPayload.manifest.k_score,
    rekor_attestation: rekorAttestation,
  };
  } finally {
    // W350 - only clean up on failure. On success the same outPath is the
    // canonical artifact and must remain on disk; tracking it in cleanupOnFail
    // is harmless because `success === true` short-circuits the unlink loop.
    if (!success) {
      for (const p of cleanupOnFail) {
        try { fs.unlinkSync(p); } catch { /* file may not exist if Pass 1 failed before packageArtifact resolved */ }
      }
    }
  }
}

export function verifyManifestSignature(manifest_json, signature) {
  // W481 - try every candidate verification secret in order so in-repo
  // marketplace seed artifacts verify on a fresh checkout AND user-compiled
  // artifacts verify on the user's own machine. The candidate list is built
  // by verificationSecrets() in env.js: env RECIPE_RECEIPT_SECRET first
  // (legacy KOLM_ARTIFACT_SECRET if requested), then MARKETPLACE_FIXTURE_SECRET,
  // then DEV_RECEIPT_SECRET (in dev mode only - never in production-like).
  const candidates = verificationSecrets({ includeLegacyArtifactSecret: true });
  if (candidates.length === 0) return { valid: false, reason: 'sign secret unavailable on server' };
  try {
    const sig = typeof signature === 'string' ? JSON.parse(signature) : signature;
    if (!sig || sig.spec !== ARTIFACT_SPEC || !sig.hmac) return { valid: false, reason: 'bad signature shape' };
    const manifest_hash = sha256(Buffer.from(manifest_json));
    if (manifest_hash !== sig.manifest_hash) return { valid: false, reason: 'manifest_hash mismatch' };
    const payloads = [];
    if (
      sig.artifact_hash &&
      sig.eval_set_hash &&
      typeof sig.eval_score === 'number' &&
      sig.judge_id
    ) {
      payloads.push({
        spec: ARTIFACT_SPEC,
        manifest_hash,
        job_id: sig.job_id,
        artifact_hash: sig.artifact_hash,
        eval_set_hash: sig.eval_set_hash,
        eval_score: sig.eval_score,
        judge_id: sig.judge_id,
      });
    }
    payloads.push({ spec: ARTIFACT_SPEC, manifest_hash, job_id: sig.job_id });

    for (const candidate of candidates) {
      for (const payload of payloads) {
        const expected = crypto.createHmac('sha256', candidate).update(canonicalJson(payload)).digest('hex');
        if (constantTimeEqualHex(sig.hmac, expected)) return { valid: true };
      }
    }
    return { valid: false, reason: 'hmac mismatch' };
  } catch (e) {
    return { valid: false, reason: String(e.message || e) };
  }
}

function constantTimeEqualHex(a, b) {
  if (typeof a !== 'string' || typeof b !== 'string') return false;
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

// Device-fit verification. Given a manifest carrying target_device and the
// current host's detected device, return {ok, reason}. Two failure modes:
//   1. Hard fail: the model in this artifact cannot physically fit on the
//      host (vram too small, arch wrong).
//   2. Soft warn: the artifact was compiled for a different device of the
//      same class; performance won't match the K-score baseline.
export async function verifyDeviceFit(manifest, hostDeviceId) {
  if (!manifest) return { ok: false, reason: 'no manifest' };
  const target = manifest.target_device;
  if (!target) {
    return { ok: true, reason: 'no target_device pinned in manifest', soft: true };
  }
  if (!hostDeviceId) {
    return { ok: false, reason: 'host device could not be detected' };
  }
  if (target === hostDeviceId) {
    return { ok: true, reason: `exact match: ${target}` };
  }
  // Cross-device: load the device registry and check vram/arch.
  const D = await import('./devices.js');
  const tgtDev = D.info(target);
  const hostDev = D.info(hostDeviceId);
  if (!tgtDev || !hostDev) {
    return { ok: false, reason: `unknown device: target=${target} host=${hostDeviceId}` };
  }
  // Same class + host has >= vram of target -> ok with soft warn.
  if (hostDev.class === tgtDev.class &&
      (hostDev.vram_gb || 0) >= (tgtDev.vram_gb || 0)) {
    return {
      ok: true,
      reason: `host ${hostDeviceId} can run an artifact compiled for ${target} (same class, sufficient vram)`,
      soft: true,
    };
  }
  // Host has less vram than target -> hard fail.
  if ((hostDev.vram_gb || 0) < (tgtDev.vram_gb || 0)) {
    return {
      ok: false,
      reason: `host ${hostDeviceId} has ${hostDev.vram_gb}GB vram; artifact was compiled for ${target} (${tgtDev.vram_gb}GB)`,
    };
  }
  return {
    ok: true,
    reason: `host ${hostDeviceId} differs from compile target ${target}; proceeding`,
    soft: true,
  };
}

// W829-2 - Heterogeneous weights extension.
//
// VLM-class artifacts ship three weight families inside one .kolm zip:
//   weights/text/                -> the language-model backbone
//   weights/vision-encoder/      -> the W462-style image encoder
//   weights/tool-use-head/       -> the W735-style tool-call head
//
// This helper does NOT replace the existing model_weights / runtime_target
// pipeline (PATH A in buildAndZip). It augments a `builder` envelope shaped
// as { files:[{filename,content}], manifest:{...} } so a downstream wave can
// fold heterogeneous weights into a builder envelope BEFORE the zip is
// finalised - or onto a freshly-decoded artifact for re-emission.
//
// Returns the SAME builder envelope, mutated in place AND returned, so the
// helper composes nicely with chained transformations.
//
// Honesty contract:
//   - missing modalities are simply absent from present_modalities[] (the
//     manifest never claims a vision encoder shipped when none did).
//   - `*_kind` fields are pinned to a closed set ('clip-vit-b32', 'siglip',
//     'tool-use-head-v1', etc.) so a downstream verifier can refuse an
//     unknown family rather than silently trusting it.
//   - filename collisions with the W457 reserved set throw at build time
//     (we add the new files into the same files[] the buildAndZip writer
//     iterates, so the same RESERVED_FILENAMES guard would catch a
//     collision - but the helper checks up front for a sharper error).
export const HETEROGENEOUS_WEIGHTS_VERSION = 'w829-v1';

const VISION_ENCODER_KINDS = new Set([
  'clip-vit-b32',
  'clip-vit-l14',
  'siglip',
  'siglip2',
  'eva-clip',
  'dinov2',
  'custom',
]);

const TOOL_USE_HEAD_KINDS = new Set([
  'tool-use-head-v1',
  'function-calling-head-v1',
  'json-schema-head-v1',
  'custom',
]);

function _looksLikeWeightBuffer(v) {
  if (v == null) return false;
  if (Buffer.isBuffer(v)) return true;
  if (v instanceof Uint8Array) return true;
  if (typeof v === 'string') return v.length > 0;
  return false;
}

function _toBuffer(v) {
  if (Buffer.isBuffer(v)) return v;
  if (v instanceof Uint8Array) return Buffer.from(v);
  if (typeof v === 'string') return Buffer.from(v, 'utf8');
  throw new Error('addHeterogeneousWeights: weights must be Buffer | Uint8Array | string');
}

export function addHeterogeneousWeights(builder, { text_weights, vision_encoder, tool_use_head } = {}) {
  if (!builder || typeof builder !== 'object') {
    throw new Error('addHeterogeneousWeights: builder must be an object with {files,manifest}');
  }
  if (!Array.isArray(builder.files)) builder.files = [];
  if (!builder.manifest || typeof builder.manifest !== 'object') builder.manifest = {};

  const present_modalities = [];
  const block = {
    spec_version: HETEROGENEOUS_WEIGHTS_VERSION,
    present_modalities,
    vision_encoder_kind: null,
    tool_use_head_kind: null,
    text_weights_present: false,
    vision_encoder_present: false,
    tool_use_head_present: false,
  };

  // weights/text/ - the language-model backbone. Accepts a single buffer
  // OR { filename, content } record OR an array of records for a sharded
  // backbone.
  if (text_weights != null) {
    const records = _normalizeWeightRecords(text_weights, 'weights/text/');
    for (const r of records) builder.files.push(r);
    block.text_weights_present = records.length > 0;
    if (records.length > 0) present_modalities.push('text');
    block.text_weights_files = records.map((r) => r.filename);
  }

  // weights/vision-encoder/ - the image encoder weights + its kind tag.
  if (vision_encoder != null) {
    if (!vision_encoder || typeof vision_encoder !== 'object') {
      throw new Error('addHeterogeneousWeights: vision_encoder must be { kind, content | files }');
    }
    const kind = String(vision_encoder.kind || 'custom');
    if (!VISION_ENCODER_KINDS.has(kind)) {
      throw new Error(`addHeterogeneousWeights: vision_encoder.kind ${JSON.stringify(kind)} not in ${[...VISION_ENCODER_KINDS].join('|')}`);
    }
    const payload = vision_encoder.files != null ? vision_encoder.files
      : vision_encoder.content != null ? vision_encoder.content
      : null;
    if (!payload && !_looksLikeWeightBuffer(payload)) {
      throw new Error('addHeterogeneousWeights: vision_encoder requires { content } or { files }');
    }
    const records = _normalizeWeightRecords(payload, 'weights/vision-encoder/');
    for (const r of records) builder.files.push(r);
    block.vision_encoder_present = records.length > 0;
    block.vision_encoder_kind = kind;
    if (records.length > 0) present_modalities.push('vision');
    block.vision_encoder_files = records.map((r) => r.filename);
  }

  // weights/tool-use-head/ - the tool-call head.
  if (tool_use_head != null) {
    if (!tool_use_head || typeof tool_use_head !== 'object') {
      throw new Error('addHeterogeneousWeights: tool_use_head must be { kind, content | files }');
    }
    const kind = String(tool_use_head.kind || 'custom');
    if (!TOOL_USE_HEAD_KINDS.has(kind)) {
      throw new Error(`addHeterogeneousWeights: tool_use_head.kind ${JSON.stringify(kind)} not in ${[...TOOL_USE_HEAD_KINDS].join('|')}`);
    }
    const payload = tool_use_head.files != null ? tool_use_head.files
      : tool_use_head.content != null ? tool_use_head.content
      : null;
    if (!payload && !_looksLikeWeightBuffer(payload)) {
      throw new Error('addHeterogeneousWeights: tool_use_head requires { content } or { files }');
    }
    const records = _normalizeWeightRecords(payload, 'weights/tool-use-head/');
    for (const r of records) builder.files.push(r);
    block.tool_use_head_present = records.length > 0;
    block.tool_use_head_kind = kind;
    if (records.length > 0) present_modalities.push('tool_use');
    block.tool_use_head_files = records.map((r) => r.filename);
  }

  builder.manifest.heterogeneous_weights = block;
  return builder;
}

// Normalize various input shapes into {filename, content:Buffer} records
// under the requested prefix. Accepts:
//   - Buffer | Uint8Array | string                       -> [{filename: prefix+'weights.bin', content: Buffer(...)}]
//   - { filename, content }                              -> [{filename: prefix+filename, content: Buffer(...)}]
//   - [{filename, content}, ...] (sharded multi-file)     -> [{filename: prefix+each, content: Buffer(each)}, ...]
function _normalizeWeightRecords(input, prefix) {
  if (input == null) return [];
  if (_looksLikeWeightBuffer(input)) {
    return [{ filename: `${prefix}weights.bin`, content: _toBuffer(input) }];
  }
  if (Array.isArray(input)) {
    return input.map((r, i) => {
      if (!r || typeof r !== 'object') {
        throw new Error(`addHeterogeneousWeights: weights[${i}] must be { filename, content }`);
      }
      const filename = String(r.filename || `weights-${i}.bin`).replace(/^[\\/]+/, '');
      if (filename.includes('..')) throw new Error('addHeterogeneousWeights: filename cannot contain ".."');
      return {
        filename: `${prefix}${filename}`,
        content: _toBuffer(r.content),
      };
    });
  }
  if (typeof input === 'object') {
    const filename = String(input.filename || 'weights.bin').replace(/^[\\/]+/, '');
    if (filename.includes('..')) throw new Error('addHeterogeneousWeights: filename cannot contain ".."');
    return [{
      filename: `${prefix}${filename}`,
      content: _toBuffer(input.content),
    }];
  }
  throw new Error('addHeterogeneousWeights: unrecognized weights shape');
}

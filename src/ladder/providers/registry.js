// kolm code — model catalog and pricing (owner A, spec §3).
//
// Prices are USD per 1,000,000 tokens. `estimateCost` divides once at the end so
// a cost is the nearest double to the exact decimal, which keeps test
// assertions exact.
//
// Provenance of the seeded numbers:
//   * anthropic:* context/output/prices — claude-api skill `shared/models.md`
//     and `shared/model-migration.md` (Fable 5.1 $10/$50 with $0.25 cache reads;
//     Opus 5 $5/$25; Sonnet 5 $2/$10; Haiku 4.5 $1/$5 at the standard 0.1x
//     cache-read multiple).
//   * deepseek:deepseek-v4-flash — KOLM_CODE_BUILD_SPEC §3 (peak $0.44/$1.32,
//     off-peak $0.22/$0.66).
//   * every other cheap rung — the spec seeds the ref but not the price; the
//     numbers below are the vendors' published list rates at spec time and are
//     marked `pricingVerified:false` so `kolm code funnel` can flag them.

import { LadderError } from '../errors.js';
import { parseRef } from '../config.js';

/** Roles a rung may be used for. */
export const ROLES = Object.freeze(['planner', 'judge', 'coder', 'reviewer']);

const CACHE_WRITE_MULTIPLE = 1.25; // Anthropic 5-minute cache writes

function entry(o) {
  return Object.freeze({
    ctx: 131072,
    maxOut: 8192,
    priceIn: 0,
    priceOut: 0,
    cacheRead: 0,
    cacheWrite: null, // null -> derived from priceIn
    offpeak: null,
    acceptsSampling: true,
    pricingVerified: true,
    // Native tool calling. `false` sends the rung through the text tool protocol
    // (providers/text-tools.js) from construction instead of waiting for a 400.
    tools: true,
    textToolsHint: null,
    ...o,
    roles: Object.freeze(o.roles),
  });
}

/** ref -> catalog entry. Frozen. */
export const MODEL_CATALOG = Object.freeze({
  'anthropic:claude-fable-5-1': entry({
    ctx: 1000000,
    maxOut: 128000,
    priceIn: 10,
    priceOut: 50,
    cacheRead: 0.25,
    acceptsSampling: false,
    roles: ['planner', 'judge', 'coder', 'reviewer'],
  }),
  'anthropic:claude-opus-5': entry({
    ctx: 1000000,
    maxOut: 128000,
    priceIn: 5,
    priceOut: 25,
    cacheRead: 0.5,
    acceptsSampling: false,
    roles: ['planner', 'judge', 'coder', 'reviewer'],
  }),
  'anthropic:claude-sonnet-5': entry({
    ctx: 1000000,
    maxOut: 128000,
    priceIn: 2,
    priceOut: 10,
    cacheRead: 0.2,
    acceptsSampling: false,
    roles: ['coder', 'reviewer', 'judge'],
  }),
  'anthropic:claude-haiku-4-5': entry({
    ctx: 200000,
    maxOut: 64000,
    priceIn: 1,
    priceOut: 5,
    cacheRead: 0.1,
    acceptsSampling: true,
    roles: ['reviewer'],
  }),
  'deepseek:deepseek-v4-flash': entry({
    ctx: 1000000,
    maxOut: 384000,
    priceIn: 0.44,
    priceOut: 1.32,
    cacheRead: 0.044,
    cacheWrite: 0,
    offpeak: { priceIn: 0.22, priceOut: 0.66, cacheRead: 0.022, cacheWrite: 0 },
    roles: ['coder', 'reviewer'],
  }),
  'deepseek:deepseek-v4-pro': entry({
    ctx: 1000000,
    maxOut: 384000,
    priceIn: 1.1,
    priceOut: 4.4,
    cacheRead: 0.11,
    cacheWrite: 0,
    offpeak: { priceIn: 0.55, priceOut: 2.2, cacheRead: 0.055, cacheWrite: 0 },
    pricingVerified: false,
    roles: ['coder', 'reviewer', 'judge'],
  }),
  'cerebras:qwen-3.8-27b': entry({
    ctx: 131072,
    maxOut: 32768,
    priceIn: 0.6,
    priceOut: 1.2,
    cacheRead: 0,
    cacheWrite: 0,
    pricingVerified: false,
    roles: ['coder', 'reviewer'],
  }),
  'cerebras:kimi-k2.7-code': entry({
    ctx: 262144,
    maxOut: 32768,
    priceIn: 1,
    priceOut: 3,
    cacheRead: 0,
    cacheWrite: 0,
    pricingVerified: false,
    roles: ['coder', 'reviewer'],
  }),
  'cerebras:gpt-oss-120b': entry({
    ctx: 131072,
    maxOut: 32768,
    priceIn: 0.25,
    priceOut: 0.69,
    cacheRead: 0,
    cacheWrite: 0,
    pricingVerified: false,
    roles: ['reviewer', 'coder'],
  }),
  'cerebras:gemma-4-31b': entry({
    ctx: 131072,
    maxOut: 32768,
    priceIn: 0.2,
    priceOut: 0.4,
    cacheRead: 0,
    cacheWrite: 0,
    pricingVerified: false,
    roles: ['reviewer', 'coder'],
  }),
  'openrouter:deepseek/deepseek-v4-flash': entry({
    ctx: 1000000,
    maxOut: 384000,
    priceIn: 0.44,
    priceOut: 1.32,
    cacheRead: 0.044,
    cacheWrite: 0,
    offpeak: { priceIn: 0.22, priceOut: 0.66, cacheRead: 0.022, cacheWrite: 0 },
    pricingVerified: false,
    roles: ['coder', 'reviewer'],
  }),
  'openrouter:deepseek/deepseek-v4-pro': entry({
    ctx: 1000000,
    maxOut: 384000,
    priceIn: 1.1,
    priceOut: 4.4,
    cacheRead: 0.11,
    cacheWrite: 0,
    offpeak: { priceIn: 0.55, priceOut: 2.2, cacheRead: 0.055, cacheWrite: 0 },
    pricingVerified: false,
    roles: ['coder', 'reviewer'],
  }),
  // Ollama's OpenAI layer answers 400 "does not support tools" for this model,
  // so it starts in the text protocol instead of burning a turn discovering it.
  'ollama:deepseek-coder-v2': entry({
    ctx: 131072,
    maxOut: 32768,
    priceIn: 0,
    priceOut: 0,
    cacheRead: 0,
    cacheWrite: 0,
    tools: false,
    textToolsHint: 'ollama deepseek-coder-v2 has no tool API; kolm code uses the text tool protocol',
    roles: ['coder', 'reviewer'],
  }),
  // Wildcard for the long tail of OpenRouter routes. Pricing varies per route
  // and per day, so it is reported unknown rather than guessed.
  'openrouter:*': entry({
    ctx: 131072,
    maxOut: 8192,
    priceIn: 0,
    priceOut: 0,
    cacheRead: 0,
    cacheWrite: 0,
    pricingVerified: false,
    textToolsHint: 'many free OpenRouter routes reject tools; the adapter flips to the text protocol',
    roles: ['coder', 'reviewer'],
  }),
});

/** Local models: every `ollama:*` ref is free and usable as coder or reviewer. */
const OLLAMA_ENTRY = entry({
  ctx: 131072,
  maxOut: 32768,
  priceIn: 0,
  priceOut: 0,
  cacheRead: 0,
  cacheWrite: 0,
  roles: ['coder', 'reviewer'],
});

const UNKNOWN_ENTRY = entry({
  ctx: 131072,
  maxOut: 8192,
  priceIn: 0,
  priceOut: 0,
  cacheRead: 0,
  cacheWrite: 0,
  roles: ['coder', 'reviewer'],
});

const FAKE_ENTRY = entry({
  ctx: 131072,
  maxOut: 8192,
  priceIn: 0,
  priceOut: 0,
  cacheRead: 0,
  cacheWrite: 0,
  roles: ['planner', 'judge', 'coder', 'reviewer'],
});

/**
 * Local model families with no tool API in Ollama's OpenAI layer, matched on the
 * model-name prefix so tags (`:16b`, `:latest`) resolve too. Kept short on
 * purpose: everything else discovers the limit from the 400 and flips at runtime.
 */
const NO_NATIVE_TOOLS_PREFIXES = Object.freeze({
  ollama: Object.freeze(['deepseek-coder-v2', 'codellama']),
});

/** Wildcard catalog entries consulted when an exact ref is unknown. */
const PROVIDER_WILDCARDS = Object.freeze({ openrouter: 'openrouter:*' });

/** Every ref the catalog knows by name (excludes the `ollama:*` wildcard). */
export function knownRefs() {
  return Object.keys(MODEL_CATALOG);
}

/** Full descriptions for every named catalog entry. */
export function listModels() {
  return knownRefs().map((ref) => describeModel(ref));
}

/**
 * @param {string} ref e.g. 'deepseek:deepseek-v4-flash'
 * @returns {{ref:string, provider:string, model:string, ctx:number, maxOut:number,
 *   priceIn:number, priceOut:number, cacheRead:number, cacheWrite:number,
 *   roles:string[], acceptsSampling:boolean, unknownPricing:boolean,
 *   pricingVerified:boolean, offpeak:object|null, tools:boolean,
 *   textToolsHint:string|null}}
 */
export function describeModel(ref) {
  const { provider, model, ref: normalized } = parseRef(ref);
  let base = MODEL_CATALOG[normalized];
  let unknownPricing = false;
  if (!base) {
    const wildcard = PROVIDER_WILDCARDS[provider];
    if (wildcard && MODEL_CATALOG[wildcard]) {
      base = MODEL_CATALOG[wildcard];
      unknownPricing = true; // the route exists; its price does not follow from the catalog
    } else if (provider === 'ollama') base = OLLAMA_ENTRY;
    else if (provider === 'fake') base = FAKE_ENTRY;
    else {
      base = UNKNOWN_ENTRY;
      unknownPricing = true;
    }
  }
  const cacheWrite = base.cacheWrite === null ? base.priceIn * CACHE_WRITE_MULTIPLE : base.cacheWrite;
  const prefixes = NO_NATIVE_TOOLS_PREFIXES[provider] ?? [];
  const noTools = prefixes.some((p) => model.toLowerCase().startsWith(p));
  return {
    ref: normalized,
    provider,
    model,
    ctx: base.ctx,
    maxOut: base.maxOut,
    priceIn: base.priceIn,
    priceOut: base.priceOut,
    cacheRead: base.cacheRead,
    cacheWrite,
    offpeak: base.offpeak,
    roles: base.roles,
    acceptsSampling: base.acceptsSampling,
    pricingVerified: base.pricingVerified,
    unknownPricing,
    tools: noTools ? false : base.tools !== false,
    textToolsHint: noTools
      ? (base.textToolsHint ?? `${normalized} has no tool API; kolm code uses the text tool protocol`)
      : base.textToolsHint,
  };
}

/**
 * Does this ref accept native tool definitions? `false` sends the adapter
 * through the text tool protocol from construction.
 * @param {string} ref
 * @returns {boolean}
 */
export function supportsNativeTools(ref) {
  return describeModel(ref).tools !== false;
}

/**
 * Effective per-MTok prices, applying the DeepSeek off-peak schedule.
 * @param {string} ref
 * @param {{offpeak?: boolean}} [opts]
 */
export function priceFor(ref, opts = {}) {
  const d = describeModel(ref);
  if (opts.offpeak && d.offpeak) {
    const cw = d.offpeak.cacheWrite ?? d.offpeak.priceIn * CACHE_WRITE_MULTIPLE;
    return {
      priceIn: d.offpeak.priceIn,
      priceOut: d.offpeak.priceOut,
      cacheRead: d.offpeak.cacheRead ?? 0,
      cacheWrite: cw,
    };
  }
  return { priceIn: d.priceIn, priceOut: d.priceOut, cacheRead: d.cacheRead, cacheWrite: d.cacheWrite };
}

/** Roles this ref may serve. */
export function rolesFor(ref) {
  return describeModel(ref).roles;
}

/**
 * @param {string} ref
 * @param {{input?:number, output?:number, cacheRead?:number, cacheWrite?:number}} usage
 * @param {{offpeak?: boolean}} [opts]
 * @returns {number} USD
 */
export function estimateCost(ref, usage = {}, opts = {}) {
  const p = priceFor(ref, opts);
  const input = toCount(usage.input, 'input');
  const output = toCount(usage.output, 'output');
  const cacheRead = toCount(usage.cacheRead, 'cacheRead');
  const cacheWrite = toCount(usage.cacheWrite, 'cacheWrite');
  const micro =
    input * p.priceIn + output * p.priceOut + cacheRead * p.cacheRead + cacheWrite * p.cacheWrite;
  return Math.round((micro / 1e6) * 1e10) / 1e10;
}

function toCount(v, field) {
  if (v === undefined || v === null) return 0;
  const n = Number(v);
  if (!Number.isFinite(n) || n < 0) {
    throw new LadderError('bad-config', `usage.${field} must be a non-negative number`, { field, value: v });
  }
  return n;
}

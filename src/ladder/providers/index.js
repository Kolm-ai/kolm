// kolm code — provider factory (owner A, spec §3).
//
// `createProvider(ref, config)` returns a provider object for any ref in the
// ladder. Construction never touches the network and never throws for a missing
// key: an unarmed rung still produces an object with `armed:false` whose
// `chat()` throws `LadderError('not-armed')` naming the env var that would arm
// it. Failing at use, not at construction, is what lets `kolm code doctor`
// render a full arming table.

import { LadderError } from '../errors.js';
import { parseRef, PROVIDER_KEY_ENV } from '../config.js';
import { describeModel } from './registry.js';
import { createAnthropicProvider } from './anthropic.js';
import { createOpenAICompatibleProvider } from './openai-compatible.js';
import { createFakeProvider, createMockProvider } from './fake.js';

/** provider name -> adapter family. */
export const PROVIDER_KINDS = Object.freeze({
  anthropic: 'anthropic',
  cerebras: 'openai-compatible',
  deepseek: 'openai-compatible',
  openrouter: 'openai-compatible',
  ollama: 'openai-compatible',
  fake: 'fake',
});

/** Every provider name `createProvider` can dispatch. */
export const SUPPORTED_PROVIDERS = Object.freeze(Object.keys(PROVIDER_KINDS));

/**
 * Static description of a ref: which adapter serves it, from which base URL,
 * which env var arms it, and whether it is armed under `config`.
 * @param {string} ref
 * @param {object} [config]
 */
export function resolveRef(ref, config = {}) {
  const { provider, model, ref: normalized } = parseRef(ref);
  const kind = PROVIDER_KINDS[provider];
  if (!kind) {
    throw new LadderError(
      'bad-ref',
      `unsupported provider "${provider}" in ref "${ref}" (known: ${SUPPORTED_PROVIDERS.join(', ')})`,
      { ref, provider },
    );
  }
  const keyEnv = PROVIDER_KEY_ENV[provider] ?? null;
  let armed;
  if (provider === 'fake') armed = true;
  else if (provider === 'ollama') armed = true; // no key; reachability is the doctor's job
  else armed = Boolean(config.hasKey?.[provider]);
  return {
    ref: normalized,
    provider,
    model,
    kind,
    keyEnv,
    baseUrl: kind === 'openai-compatible' ? (config.baseUrls?.[provider] ?? null) : null,
    armed,
    spec: describeModel(normalized),
  };
}

/**
 * @param {string} ref e.g. 'deepseek:deepseek-v4-flash'
 * @param {object} [config] from loadConfig()
 * @param {{client?: object, fetchImpl?: Function, sleep?: Function, random?: Function,
 *   script?: Array|Function, baseUrl?: string, mock?: boolean}} [opts]
 * @returns {{id:string, provider:string, model:string, armed:boolean, chat:Function}}
 */
export function createProvider(ref, config = {}, opts = {}) {
  const { provider } = parseRef(ref);

  // House switch: KOLM_TEST_MOCK_PROVIDER=1 short-circuits every adapter.
  const mock =
    opts.mock === true ||
    config.mockProvider === true ||
    (typeof process !== 'undefined' && process.env?.KOLM_TEST_MOCK_PROVIDER === '1');
  if (mock) return createMockProvider(ref);

  switch (PROVIDER_KINDS[provider]) {
    case 'anthropic':
      return createAnthropicProvider(ref, config, opts);
    case 'openai-compatible':
      return createOpenAICompatibleProvider(ref, config, opts);
    case 'fake':
      return createFakeProvider({ ...opts, ref });
    default:
      throw new LadderError(
        'bad-ref',
        `unsupported provider "${provider}" in ref "${ref}" (known: ${SUPPORTED_PROVIDERS.join(', ')})`,
        { ref, provider },
      );
  }
}

export { parseRef, describeModel };
export { createAnthropicProvider } from './anthropic.js';
export { createOpenAICompatibleProvider, OPENAI_COMPATIBLE_PROVIDERS } from './openai-compatible.js';
export { createFakeProvider, createMockProvider } from './fake.js';
export { estimateCost, listModels, knownRefs, rolesFor, priceFor, MODEL_CATALOG } from './registry.js';

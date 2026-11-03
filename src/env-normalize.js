// Provider-key env-var name normalization.
//
// Operators keep their keys in Vercel/AWS under whatever casing the dashboard
// gave them - `runpod_api_key`, `cerebras_api`, `anthropic_api_key`,
// `Cloudflare_api_token`, `google_api_key`, ... - but the code reads canonical
// UPPER_SNAKE names (`RUNPOD_API_KEY`, `CEREBRAS_API_KEY`, ...). Without this,
// a key the operator configured is silently never found (the exact bug that made
// "test frontier with runpod" fail: the key was in Vercel as `runpod_api_key`,
// the code read `RUNPOD_API_KEY`).
//
// Call normalizeEnv() once at the top of every process entry point (server.js,
// the Vercel api/* functions). Idempotent + side-effect-only on process.env.

// Canonical UPPER name  <-  accepted aliases (matched case-insensitively).
// First non-empty alias wins. Covers suffix differences (cerebras_api ->
// CEREBRAS_API_KEY) that a plain case-fold would miss.
const ALIASES = {
  STRIPE_SECRET_KEY:    ['STRIPE_API_KEY', 'stripe_api_key', 'STRIPE_KEY', 'stripe_secret_key'],
  RUNPOD_API_KEY:       ['runpod_api_key', 'KOLM_RUNPOD_TOKEN', 'RUNPOD_TOKEN', 'runpod_token'],
  CEREBRAS_API_KEY:     ['cerebras_api', 'cerebras_api_key', 'CEREBRAS_API'],
  ANTHROPIC_API_KEY:    ['anthropic_api_key', 'CLAUDE_API_KEY', 'claude_api_key'],
  OPENAI_API_KEY:       ['openai_api_key', 'OPENAI_KEY', 'openai_key'],
  GOOGLE_API_KEY:       ['google_api_key', 'GEMINI_API_KEY', 'gemini_api_key'],
  XAI_API_KEY:          ['xai_api_key', 'GROK_API_KEY', 'grok_api_key'],
  CLOUDFLARE_API_TOKEN: ['Cloudflare_api_token', 'cloudflare_api_token', 'CF_API_TOKEN', 'cf_api_token'],
  MODAL_TOKEN_ID:       ['modal_token_id'],
  MODAL_TOKEN_SECRET:   ['modal_token_secret'],
  VAST_API_KEY:         ['vast_api_key'],
  LAMBDA_API_KEY:       ['lambda_api_key', 'LAMBDA_CLOUD_API_KEY'],
  TOGETHER_API_KEY:     ['together_api_key'],
  REPLICATE_API_TOKEN:  ['replicate_api_token', 'REPLICATE_API_KEY'],
  HF_TOKEN:             ['hf_token', 'HUGGINGFACE_TOKEN', 'huggingface_token', 'HUGGING_FACE_HUB_TOKEN'],
  OPENROUTER_API_KEY:   ['openrouter_api_key'],
};

// Build a lowercase -> actualKey index once per call so case-insensitive lookups
// don't rescan the whole env per alias.
function _index(env) {
  const idx = {};
  for (const k of Object.keys(env)) idx[k.toLowerCase()] = k;
  return idx;
}

export function normalizeEnv(env = process.env) {
  const applied = [];
  const idx = _index(env);

  // 1) Specific aliases (handles suffix/name differences).
  for (const [canon, aliases] of Object.entries(ALIASES)) {
    if (env[canon]) continue;
    for (const a of aliases) {
      const hit = idx[a.toLowerCase()];
      if (hit && env[hit]) { env[canon] = env[hit]; applied.push(canon); break; }
    }
  }

  // 2) Generic case-fold: for any lower/mixed-case key with a value, set its
  //    UPPER form when the UPPER form is unset. Covers anything not in ALIASES
  //    (e.g. runpod_api_key -> RUNPOD_API_KEY, google_api_key -> GOOGLE_API_KEY).
  for (const k of Object.keys(env)) {
    const up = k.toUpperCase();
    if (up !== k && env[k] && !env[up]) { env[up] = env[k]; applied.push(up); }
  }
  return applied;
}

export default normalizeEnv;

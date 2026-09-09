// kolm code - per-provider rate limiting and backoff (owner PROV, spec 3.3).
//
// Every adapter request runs inside a limiter that belongs to its provider, not
// to the provider instance: a run with three candidates builds three provider
// objects for the same endpoint, and it is the endpoint that rate-limits. One
// limiter per provider name per process is therefore the correct unit.
//
// Two properties matter for tests and for Windows: no timers are used to expire
// a throttle penalty (it is re-checked lazily against an injectable `now`), and
// both `sleep` and `now` are injectable, so a test never waits a real second.

/** Concurrency each provider gets when nothing overrides it. */
export const PROVIDER_CONCURRENCY_DEFAULTS = Object.freeze({
  anthropic: 4,
  cerebras: 2,
  deepseek: 4,
  openrouter: 4,
  ollama: 1, // a local GPU serialises anyway; queueing here keeps latency predictable
  fake: Infinity,
});

/** Concurrency for a provider the table does not name. */
const FALLBACK_CONCURRENCY = 2;

/** How long a 429 halves a provider's concurrency for. */
export const THROTTLE_PENALTY_MS = 60000;

const DEFAULT_BASE_DELAY_MS = 1000;
const DEFAULT_MAX_DELAY_MS = 30000;
const DEFAULT_JITTER = 0.2;

const defaultSleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function abortError() {
  return Object.assign(new Error('request aborted before it started'), { name: 'AbortError' });
}

function normalizeConcurrency(value, fallback) {
  if (value === Infinity) return Infinity;
  const n = Number(value);
  if (!Number.isFinite(n) || n < 1) return fallback;
  return Math.max(1, Math.floor(n));
}

/**
 * A concurrency gate with an optional minimum gap between request starts.
 *
 * @param {{concurrency?: number, minIntervalMs?: number, sleep?: Function, now?: Function}} [opts]
 * @returns {{run: Function, stats: Function, setConcurrency: Function, penalize: Function}}
 */
export function createLimiter(opts = {}) {
  const sleep = opts.sleep ?? defaultSleep;
  const now = opts.now ?? Date.now;
  const minIntervalMs = Number(opts.minIntervalMs) > 0 ? Number(opts.minIntervalMs) : 0;
  let base = normalizeConcurrency(opts.concurrency, 1);
  let penaltyConcurrency = base;
  let penaltyUntil = 0;
  let active = 0;
  let lastStartAt = -Infinity;
  const queue = [];

  function capacity() {
    return penaltyUntil > now() ? penaltyConcurrency : base;
  }

  function pump() {
    while (queue.length > 0 && active < capacity()) {
      const waiter = queue.shift();
      if (waiter.cancelled) continue;
      waiter.detach();
      active += 1;
      waiter.start();
    }
  }

  function run(fn, { signal } = {}) {
    return new Promise((resolve, reject) => {
      if (signal?.aborted) {
        reject(abortError());
        return;
      }
      const waiter = { cancelled: false, detach: () => {}, start: null };
      const onAbort = () => {
        waiter.cancelled = true;
        waiter.detach();
        reject(abortError());
      };
      if (signal) {
        signal.addEventListener('abort', onAbort);
        waiter.detach = () => signal.removeEventListener('abort', onAbort);
      }
      waiter.start = async () => {
        try {
          if (minIntervalMs > 0) {
            const wait = lastStartAt + minIntervalMs - now();
            if (wait > 0) await sleep(wait);
          }
          lastStartAt = now();
          resolve(await fn());
        } catch (err) {
          reject(err);
        } finally {
          active -= 1;
          pump();
        }
      };
      queue.push(waiter);
      pump();
    });
  }

  return {
    run,
    /** Current gate state; `concurrency` is what the next acquire will see. */
    stats() {
      return {
        concurrency: capacity(),
        baseConcurrency: base,
        active,
        queued: queue.length,
        throttled: penaltyUntil > now(),
        penaltyUntil,
      };
    },
    /** Replace the steady-state concurrency (a config override, not a penalty). */
    setConcurrency(value) {
      base = normalizeConcurrency(value, base);
      if (penaltyConcurrency > base) penaltyConcurrency = base;
      pump();
    },
    /**
     * A 429 halves the effective concurrency for `ms`, never below 1. Repeated
     * 429s halve what is already halved, so a provider that keeps refusing ends
     * up serialised instead of hammered.
     */
    penalize({ ms = THROTTLE_PENALTY_MS } = {}) {
      const current = capacity();
      if (Number.isFinite(current)) penaltyConcurrency = Math.max(1, Math.floor(current / 2));
      penaltyUntil = now() + (Number(ms) > 0 ? Number(ms) : THROTTLE_PENALTY_MS);
      return penaltyConcurrency;
    },
  };
}

const registry = new Map();

/**
 * The process-wide limiter for a provider name, created on first use.
 * `config.providerConcurrency` (a `{name: n}` object) overrides the defaults.
 *
 * @param {string} provider e.g. 'deepseek'
 * @param {object} [config] from loadConfig()
 * @param {{sleep?: Function, now?: Function, minIntervalMs?: number}} [opts] test seams
 */
export function limiterFor(provider, config = {}, opts = {}) {
  const name = String(provider || 'unknown');
  const override = config?.providerConcurrency?.[name];
  const fallback = PROVIDER_CONCURRENCY_DEFAULTS[name] ?? FALLBACK_CONCURRENCY;
  const desired = override === undefined || override === null
    ? fallback
    : normalizeConcurrency(override, fallback);
  let limiter = registry.get(name);
  if (!limiter) {
    limiter = createLimiter({ concurrency: desired, ...opts });
    registry.set(name, limiter);
  } else if (limiter.stats().baseConcurrency !== desired) {
    // A later config may raise or lower the gate; the instance stays the same so
    // in-flight requests keep queueing behind it.
    limiter.setConcurrency(desired);
  }
  return limiter;
}

/** Drop every cached limiter. Tests only: it discards queued work. */
export function resetLimiters() {
  registry.clear();
}

/** Case-insensitive header lookup across Headers, Map and plain objects. */
function headerValue(headers, name) {
  if (!headers) return null;
  if (typeof headers.get === 'function') {
    const via = headers.get(name);
    if (via !== undefined && via !== null && String(via) !== '') return String(via);
    return null;
  }
  for (const key of Object.keys(headers)) {
    if (key.toLowerCase() === name) {
      const value = headers[key];
      if (value !== undefined && value !== null && String(value) !== '') return String(value);
    }
  }
  return null;
}

/**
 * How long to wait before retry `attempt` (0-based).
 *
 * `Retry-After` wins when the response carries one, in either legal form
 * (delta-seconds or an HTTP date). Otherwise exponential backoff 1s, 2s, 4s, 8s
 * with +-20% jitter so a fleet of candidates does not retry in lockstep. Every
 * result is capped at 30 s: a longer wait is a stall, not a retry.
 *
 * @param {number} attempt 0 for the first retry
 * @param {object|Headers|null} [headers] response headers
 * @param {{random?: Function, now?: Function, baseMs?: number, maxMs?: number, jitter?: number}} [opts]
 * @returns {number} milliseconds
 */
export function retryDelayMs(attempt = 0, headers = null, opts = {}) {
  const random = opts.random ?? Math.random;
  const now = opts.now ?? Date.now;
  const baseMs = Number(opts.baseMs) > 0 ? Number(opts.baseMs) : DEFAULT_BASE_DELAY_MS;
  const maxMs = Number(opts.maxMs) > 0 ? Number(opts.maxMs) : DEFAULT_MAX_DELAY_MS;
  const jitter = opts.jitter === undefined ? DEFAULT_JITTER : Number(opts.jitter);

  const retryAfter = headerValue(headers, 'retry-after');
  if (retryAfter !== null) {
    const seconds = Number(retryAfter);
    if (Number.isFinite(seconds) && seconds >= 0) {
      return Math.min(Math.round(seconds * 1000), maxMs);
    }
    const when = Date.parse(retryAfter);
    if (Number.isFinite(when)) {
      return Math.min(Math.max(0, when - now()), maxMs);
    }
  }

  const step = Math.max(0, Math.floor(Number(attempt) || 0));
  const exp = baseMs * 2 ** step;
  const spread = Number.isFinite(jitter) ? jitter : DEFAULT_JITTER;
  const jittered = exp * (1 + (random() * 2 - 1) * spread);
  return Math.min(Math.max(0, Math.round(jittered)), maxMs);
}

export default {
  createLimiter,
  limiterFor,
  resetLimiters,
  retryDelayMs,
  PROVIDER_CONCURRENCY_DEFAULTS,
  THROTTLE_PENALTY_MS,
};

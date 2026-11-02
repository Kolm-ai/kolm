import 'dotenv/config';
import express from 'express';
import compression from 'compression';
import helmet from 'helmet';
import cookieParser from 'cookie-parser';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildRouter } from './src/router.js';
import { provisionTenant, keyLastUsedTrackingEnabled, startKeyLastUsedFlusher, stopKeyLastUsedFlusher } from './src/auth.js';
import { startMagicLinkGc, stopMagicLinkGc } from './src/auth-email.js';
import { oauthStartupCheck } from './src/oauth.js';
import { isProductionRuntime } from './src/env.js';
import { init as initOtel, expressMiddleware as otelMiddleware } from './src/otel.js';
import { initSentry } from './src/sentry-init.js';
import { synthesize } from './src/synthesis.js';
import { createConcept, publishVersion } from './src/registry.js';
import { all, findOne } from './src/store.js';
import { backupNow, pruneBackups, backupDir } from './src/store-backup.js';
import { runDueReattestations, resignPendingReports } from './src/asr-fulfillment.js';
import { runDueDunning, tEmailDunning } from './src/dunning.js';
import { sendEmail } from './src/email.js';

// W922 — normalize provider-key env-var names at startup. Operators keep keys in
// Vercel/AWS under varied casings (runpod_api_key, cerebras_api,
// anthropic_api_key, Cloudflare_api_token, stripe_api_key, ...); the code reads
// canonical UPPER_SNAKE names. Map them once so every provider (Stripe, RunPod,
// Cerebras, Anthropic, Google, ...) is picked up regardless of the operator's
// casing. (Supersedes the Stripe-only normalization.)
import { normalizeEnv } from './src/env-normalize.js';
normalizeEnv();

// Trust moat: guarantee a persistent Ed25519 signing key on boot so kolm never ships
// unsigned artifacts/receipts. Persists to the durable data volume (KOLM_DATA_DIR/keys),
// not ephemeral ~/.kolm, and points the key store there so the signer + /health agree.
import { ensureSigningKey } from './src/ensure-signing-key.js';
ensureSigningKey();

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const app = express();
app.disable('x-powered-by');
// Two-hop proxy chain in prod: Client → Vercel CDN → AWS ALB → Express.
// `trust proxy: 2` makes express read X-Forwarded-For from the right hop so
// the rate-limit keyGenerator sees the real client IP, not the ALB's IP.
app.set('trust proxy', isProductionRuntime() ? 2 : false);

const cspConnectSrc = [
  "'self'",
  'https://api.anthropic.com',
  'https://kolm.ai',
  'https://*.vercel-insights.com',
  'https://api.stripe.com',
];
for (const origin of String(process.env.KOLM_CSP_CONNECT_SRC || '').split(',')) {
  const trimmed = origin.trim();
  if (trimmed && !cspConnectSrc.includes(trimmed)) cspConnectSrc.push(trimmed);
}

// Security headers (S3, S4) — mounted BEFORE express.static so static
// assets get HSTS, CSP, nosniff, etc. CSP allows 'unsafe-inline' for now
// because every page still has inline <script> blocks; Sprint 1 moves
// inline scripts to /js/<page>.js and tightens CSP. 'wasm-unsafe-eval' is
// required by the on-device runtime (wllama, sqlite-vec).
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'", "'unsafe-inline'", "'wasm-unsafe-eval'", 'https://js.stripe.com', 'https://*.vercel-insights.com'],
      styleSrc: ["'self'", "'unsafe-inline'", 'https://fonts.googleapis.com'],
      imgSrc: ["'self'", 'data:', 'blob:', 'https:'],
      fontSrc: ["'self'", 'data:', 'https://fonts.gstatic.com'],
      connectSrc: cspConnectSrc,
      frameSrc: ['https://js.stripe.com'],
      workerSrc: ["'self'", 'blob:'],
      frameAncestors: ["'none'"],
      objectSrc: ["'none'"],
      baseUri: ["'self'"],
      formAction: ["'self'"],
    },
  },
  crossOriginEmbedderPolicy: false,
  crossOriginResourcePolicy: { policy: 'cross-origin' },
  strictTransportSecurity: { maxAge: 63072000, includeSubDomains: true, preload: true },
  referrerPolicy: { policy: 'strict-origin-when-cross-origin' },
  frameguard: { action: 'deny' },
  noSniff: true,
}));

// gzip everything except SSE streams (compression breaks event delivery).
app.use(compression({ filter: (req, res) => res.getHeader('Content-Type') !== 'text/event-stream' && compression.filter(req, res) }));
app.use(cookieParser());
// Stripe webhook signature verification needs the raw request body — JSON
// reparse reorders keys and breaks the HMAC. Mount express.raw() for the
// webhook route ahead of express.json() so req.body is a Buffer there only.
app.use((req, res, next) => {
  if (req.path === '/v1/stripe/webhook') {
    return express.raw({ type: '*/*', limit: '4mb' })(req, res, next);
  }
  return express.json({ limit: '4mb' })(req, res, next);
});
app.use(express.urlencoded({ extended: true }));
if (initOtel()) app.use(otelMiddleware());

function isAuditHost(req) {
  const host = String(req.headers.host || '').split(':')[0].toLowerCase();
  return host === 'audit.kolm.ai' || host === 'www.audit.kolm.ai';
}

function sendPublicHtml(res, file) {
  const f = path.join(__dirname, 'public', file);
  if (!fs.existsSync(f)) return false;
  res.set('Cache-Control', 'public, max-age=60, must-revalidate');
  res.sendFile(f);
  return true;
}

const COMPILER_PRODUCT_ROUTES = [
  '/how-it-works',
];

const AUDIT_HOST_ONLY_ROUTES = [
  '/verify',
  '/checks',
  '/report',
  '/report-viewer',
  '/badge',
  '/roi',
  '/regulatory-clock',
  '/transparency-log',
  '/trust-center',
  '/buyer',
  '/spec',
  '/security/threat-model',
  '/solutions/ai-vendors',
  '/solutions/enterprise-buyers',
];
const COMPILER_PRODUCT_ROUTE_PATTERNS = COMPILER_PRODUCT_ROUTES.concat(
  COMPILER_PRODUCT_ROUTES.map(route => `${route}.html`),
);
const AUDIT_HOST_ONLY_ROUTE_PATTERNS = AUDIT_HOST_ONLY_ROUTES.concat(
  AUDIT_HOST_ONLY_ROUTES.map(route => `${route}.html`),
);

const COMPILER_COMPAT_REDIRECTS = new Map([
  ['/product', '/compiler-product'],
  ['/models', '/platform'],
  ['/api', '/docs/api'],
  ['/api-routes.json', '/docs/api-routes.json'],
  ['/quickstart', '/docs#quickstart'],
  ['/captures', '/compiler-product#pipeline'],
  ['/training', '/compiler-product#pipeline'],
  ['/distill', '/compiler-product#pipeline'],
  ['/tui', '/account/overview'],
  ['/control-center', '/account/api-control-center'],
  ['/api-control-center', '/account/api-control-center'],
  ['/enterprise-control', '/account/api-control-center'],
  ['/self-host', '/security'],
  ['/airgap', '/security'],
]);

for (const [source, destination] of COMPILER_COMPAT_REDIRECTS) {
  app.get([source, `${source}.html`], (req, res, next) => {
    if (isAuditHost(req)) return next();
    res.redirect(302, destination);
  });
}

app.get('/', (req, res, next) => {
  if (isAuditHost(req)) {
    if (sendPublicHtml(res, 'audit.html')) return;
  }
  return next();
});

app.get(['/audit', '/audit.html'], (req, res, next) => {
  if (isAuditHost(req)) {
    if (sendPublicHtml(res, 'audit.html')) return;
    return next();
  }
  res.redirect(302, 'https://audit.kolm.ai/');
});

app.get(['/docs', '/pricing'], (req, res, next) => {
  if (!isAuditHost(req)) return next();
  const file = req.path === '/docs' ? 'audit-docs.html' : 'audit-pricing.html';
  if (sendPublicHtml(res, file)) return;
  return next();
});

app.get('/account', (_req, res) => {
  res.redirect(302, '/account/overview');
});

app.get(['/dashboard', '/dashboard.html'], (req, res, next) => {
  if (isAuditHost(req)) {
    if (sendPublicHtml(res, 'dashboard.html')) return;
    return next();
  }
  res.redirect(302, '/account/overview');
});

app.get(['/terms', '/terms.html'], (req, res, next) => {
  if (isAuditHost(req)) return next();
  if (sendPublicHtml(res, 'compiler-terms.html')) return;
  return next();
});

app.get(COMPILER_PRODUCT_ROUTE_PATTERNS, (req, res, next) => {
  if (isAuditHost(req)) return next();
  if (sendPublicHtml(res, 'compiler-product.html')) return;
  return next();
});

app.get(AUDIT_HOST_ONLY_ROUTE_PATTERNS, (req, res, next) => {
  if (isAuditHost(req)) return next();
  const suffix = req.originalUrl || req.url || req.path;
  res.redirect(302, `https://audit.kolm.ai${suffix}`);
});

// --- Static site (main compiler site plus audit subdomain) ---
// The public surface is a small, flat set of pages plus the verification
// assets. Two page names also exist as directories (public/docs/ holds the API
// schema + reference; public/security/ holds the Halborn report + threat
// model), so an explicit handler serves the page before express.static can
// collide with the directory. Every other clean URL resolves through
// express.static's `extensions:['html']` fallback and the generic catch-all
// below — matching the Vercel deploy's rewrite behaviour.
for (const [route, file] of [['/docs', 'docs.html'], ['/security', 'security.html']]) {
  app.get(route, (_req, res, next) => {
    if (sendPublicHtml(res, file)) return;
    return next();
  });
}

// Static dashboard with strong caching for hashed assets, weak for HTML.
app.use(express.static(path.join(__dirname, 'public'), {
  // extensions: ['html'] gives extensionless URL serving so /how-it-works,
  // /security/threat-model, /solutions/ai-vendors etc. resolve to their .html
  // siblings — matching the Vercel deploy where vercel.json rewrites do the
  // same fallback.
  // redirect: false avoids directory-collision 301s and lets the extensions
  // fallback fire when a name exists as both a directory and a .html file.
  extensions: ['html'],
  redirect: false,
  setHeaders: (res, filePath) => {
    if (filePath.endsWith('.html')) res.setHeader('Cache-Control', 'public, max-age=60, must-revalidate');
    // Images, fonts, and WASM rarely change between deploys — cache 1 day.
    else if (filePath.match(/\.(svg|png|jpg|jpeg|webp|gif|ico|woff2?|wasm)$/)) res.setHeader('Cache-Control', 'public, max-age=86400, must-revalidate');
    // CSS/JS change with deploys — 1 hour with revalidate keeps deploys fresh.
    else if (filePath.match(/\.(css|js|map)$/)) res.setHeader('Cache-Control', 'public, max-age=3600, must-revalidate');
  },
}));

app.use('/', buildRouter());

// Generic extensionless static fallback — mirrors Vercel's "try public/<path>.html"
// rewrite so self-host / Docker / ALB-direct serves the same routes the live
// Vercel deploy does. Conservative: only GET, only extension-less paths, rejects
// traversal (..). Resolves /<page> to public/<page>.html and /<dir> to
// public/<dir>/index.html.
app.get('*', (req, res, next) => {
  if (req.method !== 'GET') return next();
  const p = req.path;
  if (!p || p === '/' || p.startsWith('/v1') || p === '/health' || p === '/ready' || p.includes('..')) return next();
  if (/\.[a-z][a-z0-9]*$/i.test(p)) return next();
  const rel = p.slice(1);
  if (!/^[a-z0-9][a-z0-9_\-\/\.]*$/i.test(rel)) return next();
  const direct = path.join(__dirname, 'public', rel + '.html');
  if (fs.existsSync(direct)) {
    res.set('Cache-Control', 'public, max-age=60, must-revalidate');
    return res.sendFile(direct);
  }
  const indexed = path.join(__dirname, 'public', rel, 'index.html');
  if (fs.existsSync(indexed)) {
    res.set('Cache-Control', 'public, max-age=60, must-revalidate');
    return res.sendFile(indexed);
  }
  next();
});

// 404 fallback for unknown HTML routes — branded page from /public/404.html if it exists.
const _404Path = path.join(__dirname, 'public', '404.html');
app.use((req, res, next) => {
  if (req.method === 'GET' && req.accepts('html') && !req.path.startsWith('/v1') && !req.path.startsWith('/health') && req.path !== '/ready' && req.path !== '/404') {
    if (fs.existsSync(_404Path)) return res.status(404).sendFile(_404Path);
    return res.status(404).type('html').send(`<!DOCTYPE html><html><head><title>404 - kolm</title></head><body style="padding:48px;text-align:center;font-family:system-ui;color:#131713;background:#F5F3EC;min-height:100vh;"><h1 style="font-size:48px;margin:0;letter-spacing:-0.02em;">404</h1><p style="color:#5f675f;margin-top:8px">That compiler route does not exist.</p><p style="margin-top:24px;"><a href="/" style="color:#176b4d;">Home</a> &middot; <a href="/docs" style="color:#176b4d;">Docs</a> &middot; <a href="https://audit.kolm.ai/" style="color:#176b4d;">Audit module</a></p></body></html>`);
  }
  next();
});

// Generic 500 — catches any unhandled error in routes. Standardized shape:
// `{ error, detail?, error_id }` where `detail` is omitted in production to
// avoid leaking stack-tail strings or internal paths and `error_id` is a
// short opaque id the operator can grep for in logs and Sentry.
app.use((err, req, res, _next) => {
  // Client errors from body parsing/limits — return a clean, self-explanatory
  // 400/413 instead of a scary 500. (First-touch DX: malformed JSON and
  // oversized bodies are the most common early failures.)
  if (err && (err.type === 'entity.parse.failed' || err.status === 400 || err.statusCode === 400)) {
    return res.status(400).json({ ok: false, error: 'invalid_json', detail: 'Request body is not valid JSON.' });
  }
  if (err && (err.type === 'entity.too.large' || err.status === 413 || err.statusCode === 413)) {
    return res.status(413).json({ ok: false, error: 'payload_too_large', detail: 'Request body exceeds the size limit.' });
  }
  // Short error id — 12 hex chars from a timestamped random source. Stable
  // for the lifetime of the response; surfaced to the client AND logged so
  // an operator can correlate the user's report with the server log/Sentry.
  const errorId = `e_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
  console.error(`[500] error_id=${errorId} path=${req && req.path} method=${req && req.method}`, err);
  // W890-3 — every 500 is reported to Sentry with context (request path,
  // method, tenant if attached). No-op when SENTRY_DSN is unset.
  try {
    if (globalThis.__kolmSentry && typeof globalThis.__kolmSentry.captureException === 'function') {
      globalThis.__kolmSentry.captureException(err, {
        tags: { kind: 'http_500', method: req && req.method, error_id: errorId },
        extra: { path: req && req.path, query: req && req.query, tenant: req && req.tenant && req.tenant.id },
      });
    }
  } catch { /* deliberate: cleanup */ }
  if (req.accepts('html')) {
    const _500Path = path.join(__dirname, 'public', '500.html');
    if (fs.existsSync(_500Path)) {
      res.set('X-Kolm-Error-Id', errorId);
      return res.status(500).sendFile(_500Path);
    }
  }
  const body = { error: 'internal server error', error_id: errorId };
  if (process.env.NODE_ENV !== 'production' || process.env.KOLM_DEBUG) {
    body.detail = String(err.message || err);
  }
  res.set('X-Kolm-Error-Id', errorId);
  res.status(500).json(body);
});

const PORT = parseInt(process.env.PORT || '8787');

async function bootSeedDemoConcepts(tenant) {
  const dir = path.resolve('examples');
  if (!fs.existsSync(dir)) return { added: 0, skipped: 0 };
  const files = fs.readdirSync(dir).filter(f => f.endsWith('.json'));
  const existing = new Set(all('concepts').filter(c => c.tenant === tenant).map(c => c.name));
  let added = 0, skipped = 0;
  for (const file of files) {
    try {
      const ex = JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8'));
      if (existing.has(ex.name)) { skipped++; continue; }
      const r = await synthesize({ positives: ex.positives, negatives: ex.negatives || [], output_spec: ex.output_spec, priors: ex.priors || {} });
      if (!r.accepted) { skipped++; continue; }
      const concept = createConcept({
        name: ex.name, description: ex.description || ex.name, tenant,
        schema: ex.output_spec || null, tags: ex.tags || [], visibility: ex.visibility || 'public',
      });
      publishVersion({
        concept_id: concept.id, source: r.source,
        evaluation: { quality_score: r.quality_score, pass_rate_positive: r.pass_rate_positive, reject_rate_negative: r.reject_rate_negative, latency_p50_us: r.latency_p50_us, size_bytes: r.size_bytes, source_hash: r.source_hash, strategy: r.strategy, trace: r.test_trace },
        lineage: { synthesized_from_n: ex.positives.length + (ex.negatives?.length || 0), attempts_n: r.attempts_n },
      });
      added++;
    } catch { skipped++; }
  }
  return { added, skipped };
}

if (process.argv[1] && process.argv[1].endsWith('server.js')) {
  // Opt-in crash reporting. No-op when SENTRY_DSN is unset OR @sentry/node
  // is not installed in this deploy.
  const sentry = await initSentry();
  // Make the Sentry handle reachable from the 500 middleware below so the
  // generic error handler can capture context (request URL, status code,
  // tenant if known) before responding to the client.
  if (sentry) globalThis.__kolmSentry = sentry;

  // W890-3 — process-level guards. Every entry point registers BOTH handlers
  // before listen() so a stray throw or unhandled promise can't take the
  // server down without a structured trail. uncaughtException triggers a
  // graceful shutdown (drain in-flight requests via server.close()), while
  // unhandledRejection logs + reports without crashing — the same Node 20
  // default behaviour, but with Sentry context attached.
  process.on('unhandledRejection', (reason, promise) => {
    console.error('[unhandledRejection]', reason);
    try {
      if (globalThis.__kolmSentry && typeof globalThis.__kolmSentry.captureException === 'function') {
        globalThis.__kolmSentry.captureException(reason, { tags: { kind: 'unhandledRejection' } });
      }
    } catch { /* deliberate: cleanup */ }
  });
  process.on('uncaughtException', (err) => {
    console.error('[uncaughtException]', err);
    try {
      if (globalThis.__kolmSentry && typeof globalThis.__kolmSentry.captureException === 'function') {
        globalThis.__kolmSentry.captureException(err, { tags: { kind: 'uncaughtException' } });
      }
    } catch { /* deliberate: cleanup */ }
    // Graceful shutdown: stop accepting new connections, let in-flight finish,
    // then exit. setTimeout fallback guarantees we don't hang on a stuck conn.
    try {
      if (globalThis.__kolmServer && typeof globalThis.__kolmServer.close === 'function') {
        globalThis.__kolmServer.close(() => process.exit(1));
        setTimeout(() => process.exit(1), 10_000).unref();
        return;
      }
    } catch { /* deliberate: cleanup */ }
    process.exit(1);
  });
  // SIGTERM / SIGINT — ECS sends SIGTERM on deploys; honour it by closing
  // the listening socket and exiting cleanly so we don't drop in-flight
  // requests on rolling restarts.
  const onSig = (sig) => () => {
    console.log(`[${sig}] graceful shutdown initiated`);
    // AUTH-03 - stop the background timers and drain their final windows before
    // we begin closing connections, so the last scoped-key last_used batch and a
    // pending magic-link GC tick are not lost on a rolling restart. Both are
    // synchronous + no-throw and no-ops when never started.
    try { stopKeyLastUsedFlusher(); } catch (e) { console.error('[auth] stopKeyLastUsedFlusher error:', e && e.message); }
    try { stopMagicLinkGc(); } catch (e) { console.error('[auth] stopMagicLinkGc error:', e && e.message); }
    // Best-effort durable snapshot before exit so every deploy / rolling
    // restart leaves behind a fresh, consistent recovery point. backupNow() is
    // synchronous and never throws (store-backup contract), so it completes
    // before we begin draining connections without risking the clean exit.
    if (process.env.KOLM_BACKUP_DISABLE !== '1') {
      try {
        const b = backupNow();
        if (b && b.ok) console.log(`[backup] shutdown snapshot -> ${b.path}`);
        else if (b) console.error(`[backup] shutdown snapshot failed: ${b.error}`);
      } catch (e) { console.error('[backup] shutdown error:', e && e.message); }
    }
    try {
      if (globalThis.__kolmServer && typeof globalThis.__kolmServer.close === 'function') {
        globalThis.__kolmServer.close(() => process.exit(0));
        setTimeout(() => process.exit(0), 10_000).unref();
        return;
      }
    } catch { /* deliberate: cleanup */ }
    process.exit(0);
  };
  process.on('SIGTERM', onSig('SIGTERM'));
  process.on('SIGINT', onSig('SIGINT'));

  // Auto-provision the demo tenant. Wrapped: if the durable store is
  // unrecoverable (corrupt JSON + EACCES on backup + EACCES on writes), we
  // STILL want the HTTP server to start so /health responds and operators
  // can introspect — auth-bearing routes degrade to 503 downstream rather
  // than crash-looping the whole container.
  const demoName = process.env.DEFAULT_TENANT || 'demo';
  let demo;
  try {
    demo = provisionTenant(demoName);
  } catch (err) {
    console.error('[boot] provisionTenant failed; continuing in degraded mode:', err && err.message);
    demo = { name: demoName, api_key: null, plan: 'free', degraded: true };
  }

  // Idempotent seed: synthesizes any missing example/*.json concepts.
  // Skip if we couldn't provision the tenant — seeding without a real
  // tenant row would write orphan concepts.
  if (!demo.degraded) {
    try {
      const { added, skipped } = await bootSeedDemoConcepts(demo.name);
      if (added > 0 || skipped > 0) console.log(`  seed: +${added} added, ${skipped} skipped`);
    } catch (err) {
      console.error('[boot] bootSeedDemoConcepts failed; continuing:', err && err.message);
    }
  } else {
    console.log('  seed: skipped (tenant provision degraded)');
  }

  // Stripe configuration sanity check. Counts how many of the 5 payment links
  // are wired so a misconfigured deploy logs a single line at boot rather than
  // surfacing as a 503 the first time a customer clicks Upgrade. Webhook secret
  // is required for paid plans to flip; surface the gap explicitly.
  const stripeLinks = [
    'STRIPE_PAYMENT_LINK_STARTER', 'STRIPE_PAYMENT_LINK_PRO',
    'STRIPE_PAYMENT_LINK_TEAMS', 'STRIPE_PAYMENT_LINK_BUSINESS',
    'STRIPE_PAYMENT_LINK_ENT',
  ];
  const stripePresent = stripeLinks.filter(v => !!process.env[v]).length;
  const webhookOk = !!process.env.STRIPE_WEBHOOK_SECRET;
  const stripeStatus = stripePresent === 5 && webhookOk ? 'wired' : `degraded (${stripePresent}/5 links, webhook ${webhookOk ? 'ok' : 'missing'})`;

  // AUTH-05 - OAuth callback-base sanity check. Logs a single WARNING line at
  // boot when a production deploy on a non-kolm.ai host has OAuth configured but
  // no OAUTH_REDIRECT_BASE / KOLM_PUBLIC_URL set (so callbacks would return to
  // https://kolm.ai instead of this instance). No-op + null in dev. Never throws.
  try { oauthStartupCheck(console); } catch (e) { console.error('[oauth] startup check error:', e && e.message); }

  // finalized-c1 — NVIDIA NRAS confidential-compute attestation verifier.
  // Env-gated (KOLM_NRAS_VERIFIER=1 + KOLM_NRAS_ROOT_CERT). When OFF this is a
  // no-op and confidential_compute attestations stay shape-only (verified=false).
  // When ON but misconfigured, registerNrasVerifier() throws a LOUD install hint
  // (pinned NVIDIA root cert + python nv-attestation-sdk/cryptography/PyJWT); we
  // FAIL the boot rather than ship a deploy that claims CC-verified without a
  // registered crypto chain. The privacy boundary holds: only digests + the NRAS
  // token cross the worker, never plaintext.
  try {
    const { registerNrasVerifier } = await import('./src/nras-verifier.js');
    const reg = registerNrasVerifier({});
    if (reg.registered) console.log(`  nras:       registered (${reg.version}, root=${reg.root_cert})`);
  } catch (e) {
    console.error('[nras] FATAL: KOLM_NRAS_VERIFIER=1 but verifier could not register:', e && e.message);
    throw e;
  }

  // W890-3 — keep a handle to the http.Server so graceful-shutdown hooks
  // above can call .close() and let in-flight requests drain on SIGTERM /
  // uncaughtException.
  const httpServer = app.listen(PORT, () => {
    console.log('\nkolm server');
    console.log(`  home:       http://localhost:${PORT}`);
    console.log(`  workspace:  http://localhost:${PORT}/account/overview`);
    console.log(`  docs:       http://localhost:${PORT}/docs`);
    console.log(`  demo key:   ${!!demo.api_key ? 'configured' : 'missing'}`);
    console.log(`  admin key:  ${!!process.env.ADMIN_KEY ? 'configured' : 'not set'}`);
    console.log(`  synthesis:  ${process.env.ANTHROPIC_API_KEY ? 'Claude (' + (process.env.ANTHROPIC_MODEL || 'claude-opus-4-7') + ') + Pattern' : 'Pattern (no API key set)'}`);
    console.log(`  stripe:     ${stripeStatus}`);
    console.log('');
  });
  globalThis.__kolmServer = httpServer;

  // Continuous re-attestation: in-process sweep so the product self-drives with
  // no external scheduler. Production runs a single long-lived API task, and
  // claim-then-run in asr-fulfillment makes a sweep idempotent even if one ever
  // overlaps. Runs every KOLM_REATTEST_INTERVAL_MIN (default 30), no-op when
  // nothing is due. Disable with KOLM_REATTEST_DISABLE=1. The external
  // POST /v1/audit/continuous/tick (cron-secret) stays as a manual backup.
  if (process.env.KOLM_REATTEST_DISABLE !== '1') {
    const everyMin = Math.max(5, parseInt(process.env.KOLM_REATTEST_INTERVAL_MIN || '30', 10) || 30);
    const sweep = () => {
      try {
        const r = runDueReattestations({});
        if (r && r.ran) console.log(`[reattest] ran ${r.ran}/${r.considered} due subscriptions`);
      } catch (e) { console.error('[reattest] sweep error:', e && e.message); }
      try {
        const p = resignPendingReports({});
        if (p && p.fixed) console.log(`[reattest] re-signed ${p.fixed}/${p.pending} pending reports`);
      } catch (e) { console.error('[reattest] resign-pending error:', e && e.message); }
      // M12 dunning: advance the failed-payment retry ladder, emailing each
      // reminder / suspension notice. Best-effort - one bad email never stalls
      // the sweep. Tenant email is resolved from the store at send time.
      try {
        const d = runDueDunning({
          sendFn: ({ dunning, final }) => {
            try {
              const t = dunning && dunning.tenant_id ? findOne('tenants', (x) => x.id === dunning.tenant_id) : null;
              const to = (t && (t.email || t.owner_email || t.billing_email)) || null;
              if (!to) return;
              const mail = tEmailDunning({
                email: to,
                attempt: dunning.attempt,
                final,
                amount_cents: dunning.amount_cents,
                currency: dunning.currency,
                next_retry_at: dunning.next_retry_at,
              });
              return sendEmail({ to, subject: mail.subject, html: mail.html, text: mail.text, tag: 'dunning' });
            } catch { /* per-send best-effort */ }
          },
        });
        if (d && d.processed) console.log(`[reattest] dunning advanced ${d.processed}/${d.due} due schedule(s)`);
      } catch (e) { console.error('[reattest] dunning error:', e && e.message); }
    };
    const everyMs = everyMin * 60 * 1000;
    const t = setInterval(sweep, everyMs);
    if (t.unref) t.unref();
    const kick = setTimeout(sweep, 60 * 1000);
    if (kick.unref) kick.unref();
    console.log(`  reattest:   in-process sweep every ${everyMin}m`);
  }

  // Durable backups: in-process snapshot scheduler so the data layer self-protects
  // with no external cron. SQLite snapshots use VACUUM INTO (consistent + online);
  // JSON snapshots copy the table files. Runs every KOLM_BACKUP_INTERVAL_H hours
  // (default 6), unref'd so it never holds the event loop open, and prunes to the
  // most recent 14 snapshots after each run. Disable with KOLM_BACKUP_DISABLE=1.
  // A best-effort snapshot also runs on SIGTERM/SIGINT (see onSig above). See
  // docs/durability.md for the restore runbook.
  if (process.env.KOLM_BACKUP_DISABLE !== '1') {
    const everyH = Math.max(1, parseInt(process.env.KOLM_BACKUP_INTERVAL_H || '6', 10) || 6);
    const snapshot = (reason) => {
      try {
        const r = backupNow();
        if (r && r.ok) {
          console.log(`[backup] ${reason} snapshot -> ${r.path}`);
          const pruned = pruneBackups();
          if (pruned && pruned.pruned && pruned.pruned.length) {
            console.log(`[backup] pruned ${pruned.pruned.length} old snapshot(s), kept ${pruned.kept}`);
          }
        } else {
          console.error(`[backup] ${reason} snapshot failed: ${r && r.error}`);
        }
      } catch (e) { console.error('[backup] sweep error:', e && e.message); }
    };
    const everyMs = everyH * 60 * 60 * 1000;
    const bt = setInterval(() => snapshot('scheduled'), everyMs);
    if (bt.unref) bt.unref();
    // Kick an initial snapshot a couple minutes after boot so a fresh deploy
    // has a recovery point immediately, without waiting a full interval.
    const bkick = setTimeout(() => snapshot('startup'), 2 * 60 * 1000);
    if (bkick.unref) bkick.unref();
    console.log(`  backup:     snapshot every ${everyH}h -> ${backupDir() || 'data/backups'}`);
  }

  // AUTH-03 - background flush of scoped-key last_used_at (no-op unless
  // KOLM_KEY_LAST_USED_TRACKING=1; cadence KOLM_KEY_LAST_USED_FLUSH_MS, default
  // 30000). Without this the coalesced queue never drains and last_used_at stays
  // a false 'never used' signal. unref'd so it never holds the event loop open.
  if (keyLastUsedTrackingEnabled()) {
    try {
      const flusher = startKeyLastUsedFlusher(console);
      if (flusher) console.log('  keyflush:   scoped-key last_used flusher armed');
    } catch (e) { console.error('[auth] key last_used flusher start error:', e && e.message); }
  }

  // AUTH-03 (email lane) - hourly GC of dead magic-link rows (cadence
  // KOLM_MAGICLINK_GC_MS, default 3600000; retention KOLM_MAGICLINK_RETENTION_DAYS,
  // default 7). Keeps the magic_link_tokens full-scan from degrading with sign-in
  // volume. unref'd, self-driving, no external cron.
  try {
    const gc = startMagicLinkGc(console);
    if (gc) console.log('  magiclink:  dead-token GC armed');
  } catch (e) { console.error('[auth] magic-link gc start error:', e && e.message); }
}

export { app };
export default app;

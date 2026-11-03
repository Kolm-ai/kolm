// src/billing-activation.js
//
// P0 - Stripe billing ACTIVATION readiness + on-the-fly Checkout for the
// PRICE-ID path. This is the module an operator "turns on" by setting price-id
// env vars in AWS/Vercel; the code here is ready the moment they do.
//
// Why this exists alongside billing-upgrade.js / stripe.js:
//   * stripe.js - webhook signature verify + amount->plan mapping.
//   * billing-upgrade.js - resolveUpgradeUrl(): a 4-path fallback that always
//                          returns *some* working URL (payment link -> checkout
//                          session -> self-hosted -> manual). It never tells
//                          the operator which env vars are missing.
//   * billing-activation - this file. The strict counterpart: a single source
//                          of truth for "is real Stripe Checkout configured?"
//                          and a checkout creator that FAILS LOUD with the exact
//                          env var names when it is not. The /v1/billing/ready
//                          probe + the release-verify `billing-tiers` gate read
//                          from here.
//
// Convention parity (do NOT diverge - these are the same env names the rest of
// the codebase already reads):
//   * Secret key:  STRIPE_SECRET_KEY (env-normalize maps stripe_api_key ->
//                  STRIPE_SECRET_KEY; KOLM_STRIPE_KEY is the legacy alias read
//                  by billing-upgrade.js). We accept all three via envSecret.
//   * Price ids:   KOLM_STRIPE_PRICE_<PLAN> - IDENTICAL to STRIPE_PRICE_ENVS in
//                  billing-upgrade.js so a single set of env vars powers BOTH
//                  the fallback path and this strict path. STRIPE_PRICE_<PLAN>
//                  is accepted as a secondary alias for operators who name them
//                  the Stripe-dashboard way.
//
// ZERO new npm deps. No Stripe SDK - Stripe's REST API over global fetch, the
// same approach billing-upgrade.js already uses (Node >= 18).

import { envSecret } from './env.js';
import { PLAN_CATALOG, PLAN_ORDER, canonicalPlanId } from './plan-catalog.js';

// Canonical price-id env var per plan. Matches billing-upgrade.js
// STRIPE_PRICE_ENVS exactly so one configuration powers both paths.
const PRICE_ENVS = {
  indie:      'KOLM_STRIPE_PRICE_INDIE',
  pro:        'KOLM_STRIPE_PRICE_PRO',
  teams:      'KOLM_STRIPE_PRICE_TEAM',
  business:   'KOLM_STRIPE_PRICE_BUSINESS',
};
// Secondary aliases tried in order if the canonical env is unset. Mirrors
// billing-upgrade.js LEGACY_STRIPE_PRICE_ENVS plus the dashboard-style
// STRIPE_PRICE_<PLAN> spelling so no valid operator config is rejected.
const PRICE_ENV_ALIASES = {
  indie:    ['STRIPE_PRICE_INDIE'],
  pro:      ['KOLM_STRIPE_PRICE_STARTER', 'STRIPE_PRICE_PRO'],
  teams:    ['KOLM_STRIPE_PRICE_TEAMS', 'STRIPE_PRICE_TEAM', 'STRIPE_PRICE_TEAMS'],
  business: ['KOLM_STRIPE_PRICE_BIZ', 'STRIPE_PRICE_BUSINESS'],
};

// The set of plans we expect to be price-id configured to "go live". Derived
// from the catalog: self-serve, fixed-price, paid (excludes free + the
// sales-led enterprise tier which never produces a self-serve Stripe charge).
export function selfServePaidPlans() {
  return PLAN_ORDER.filter((id) => {
    const p = PLAN_CATALOG[id];
    return p && p.self_serve === true && typeof p.price_usd_month === 'number' && p.price_usd_month > 0;
  });
}

// The secret key, honoring every accepted alias. envSecret returns null (never
// '') so an empty STRIPE_SECRET_KEY="" is correctly treated as unset.
export function stripeSecretKey() {
  return envSecret('STRIPE_SECRET_KEY') || envSecret('KOLM_STRIPE_KEY') || envSecret('STRIPE_API_KEY');
}

// The canonical price-id env var name for a plan (what the operator must set).
// Used in error messages + the /ready missing[] list.
export function priceEnvVar(plan) {
  const id = canonicalPlanId(plan) || String(plan || '').toLowerCase();
  return PRICE_ENVS[id] || `KOLM_STRIPE_PRICE_${id.toUpperCase()}`;
}

// The configured price id for a plan (canonical env first, then aliases), or
// null if none is set.
export function priceIdFor(plan) {
  const id = canonicalPlanId(plan) || String(plan || '').toLowerCase();
  const names = [PRICE_ENVS[id]].concat(PRICE_ENV_ALIASES[id] || []).filter(Boolean);
  for (const name of names) {
    const v = envSecret(name);
    if (v) return v;
  }
  return null;
}

// billingReady - is real Stripe Checkout fully configured to charge customers?
//
// Returns { ready, missing[], secret_key, webhook_secret, livemode, prices }.
// `missing` lists the EXACT env var names the operator must set in
// AWS/Vercel. The webhook secret is required for the activation to be
// end-to-end useful (it's what flips the tenant's plan after payment), so its
// absence is reported in missing[] too - matching the existing
// /v1/billing/tiers `stripe.ready` semantics.
export function billingReady({ plans } = {}) {
  const wanted = Array.isArray(plans) && plans.length
    ? plans.map((p) => canonicalPlanId(p) || String(p).toLowerCase())
    : selfServePaidPlans();

  const missing = [];
  const prices = {};

  const secret = stripeSecretKey();
  const haveSecret = !!secret;

  // A plan is chargeable via EITHER a Checkout-API price id (KOLM_STRIPE_PRICE_<PLAN>)
  // OR a hosted Stripe Payment Link (STRIPE_PAYMENT_LINK_<PLAN> - the path
  // src/billing-upgrade.js already uses). Payment links collect revenue with just the
  // links + webhook secret (no Stripe secret key), so the secret key is only required
  // when no payment link covers a plan.
  const PAYMENT_LINK_ENVS = { indie: 'STRIPE_PAYMENT_LINK_STARTER', pro: 'STRIPE_PAYMENT_LINK_PRO', teams: 'STRIPE_PAYMENT_LINK_TEAMS', business: 'STRIPE_PAYMENT_LINK_BUSINESS' };
  let anyLink = false;
  for (const plan of wanted) {
    const havePrice = !!priceIdFor(plan);
    const linkEnv = PAYMENT_LINK_ENVS[plan] || `STRIPE_PAYMENT_LINK_${String(plan).toUpperCase()}`;
    const haveLink = !!envSecret(linkEnv);
    if (haveLink) anyLink = true;
    const have = havePrice || haveLink;
    prices[plan] = have ? (havePrice ? 'price_id' : 'payment_link') : false;
    if (!have) missing.push(`${priceEnvVar(plan)} (or ${linkEnv})`);
  }
  if (!haveSecret && !anyLink) missing.push('STRIPE_SECRET_KEY');

  const webhookSecret = envSecret('STRIPE_WEBHOOK_SECRET');
  if (!webhookSecret) missing.push('STRIPE_WEBHOOK_SECRET');

  return {
    ready: missing.length === 0,
    missing,
    secret_key: haveSecret,
    webhook_secret: !!webhookSecret,
    // Detect test vs live key without ever exposing the key value itself.
    livemode: haveSecret ? /^sk_live_/.test(secret) : null,
    plans: wanted,
    prices,
  };
}

// Typed error for the "operator hasn't set price ids" case. Carries the exact
// env var names + an HTTP status so the route can surface them verbatim and
// never collapse into a generic 500.
export class BillingNotConfiguredError extends Error {
  constructor(missing, message) {
    super(
      message ||
        ('Stripe billing is not configured. Set these environment variables in AWS/Vercel: ' +
          (missing || []).join(', '))
    );
    this.name = 'BillingNotConfiguredError';
    this.code = 'price_ids_not_configured';
    this.statusCode = 503;
    this.missing = missing || [];
  }
  toJSON() {
    return { error: this.code, message: this.message, missing: this.missing };
  }
}

// Stripe REST: POST /v1/checkout/sessions (form-urlencoded, Bearer auth). No
// SDK - same shape as billing-upgrade.js createStripeCheckoutSession, lifted
// here so the strict path is self-contained and returns the full session
// object (id + url) rather than just a URL string.
async function postCheckoutSession({ priceId, customerEmail, tenantId, plan, successUrl, cancelUrl, apiKey, baseUrl, quantity, fetchImpl }) {
  const doFetch = fetchImpl || globalThis.fetch;
  const base = (baseUrl || 'https://api.stripe.com').replace(/\/+$/, '');
  const params = new URLSearchParams();
  params.set('mode', 'subscription');
  params.set('line_items[0][price]', priceId);
  params.set('line_items[0][quantity]', String(quantity && quantity > 0 ? Math.floor(quantity) : 1));
  params.set('success_url', successUrl);
  params.set('cancel_url', cancelUrl);
  params.set('allow_promotion_codes', 'true');
  if (customerEmail) params.set('customer_email', customerEmail);
  if (tenantId) {
    params.set('client_reference_id', tenantId);
    params.set('metadata[tenant_id]', tenantId);
    params.set('subscription_data[metadata][tenant_id]', tenantId);
  }
  if (plan) {
    params.set('metadata[plan]', plan);
    params.set('subscription_data[metadata][plan]', plan);
  }
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), 10000);
  try {
    const resp = await doFetch(base + '/v1/checkout/sessions', {
      method: 'POST',
      headers: {
        authorization: `Bearer ${apiKey}`,
        'content-type': 'application/x-www-form-urlencoded',
      },
      body: params.toString(),
      signal: ctl.signal,
    });
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok || !body || !body.url) {
      const msg = body && body.error && body.error.message ? body.error.message : `stripe_http_${resp.status}`;
      const err = new Error(msg);
      err.code = 'stripe_api_error';
      err.statusCode = 502;
      err.stripe_status = resp.status;
      throw err;
    }
    return body;
  } finally {
    clearTimeout(timer);
  }
}

// createCheckoutSession - start a real Stripe Checkout for a tenant upgrading
// to a paid plan, using the operator-configured PRICE ID for that plan.
//
// @param {object} opts
//   opts.tenant      tenant id string OR object with .id (required)
//   opts.plan        paid plan id (required; canonicalized - 'team' -> 'teams')
//   opts.email       customer email to prefill (optional)
//   opts.quantity    seat quantity (default 1)
//   opts.successUrl  override success redirect (optional)
//   opts.cancelUrl   override cancel redirect (optional)
//   opts.fetchImpl   inject fetch for tests (optional; defaults global fetch)
// @returns {Promise<{ id, url, plan, tenant, source, livemode }>}
// @throws  BillingNotConfiguredError (code 'price_ids_not_configured', 503)
//          listing exactly which env vars the operator must set; or
//          invalid_plan / tenant_required (400); or stripe_api_error (502).
export async function createCheckoutSession(opts = {}) {
  const tenantId = opts.tenant && typeof opts.tenant === 'object'
    ? (opts.tenant.id || opts.tenant.tenant)
    : opts.tenant;
  if (!tenantId) {
    const err = new Error('tenant is required');
    err.code = 'tenant_required';
    err.statusCode = 400;
    throw err;
  }

  const plan = canonicalPlanId(opts.plan);
  const meta = plan ? PLAN_CATALOG[plan] : null;
  const payable = meta && meta.self_serve === true && typeof meta.price_usd_month === 'number' && meta.price_usd_month > 0;
  if (!payable) {
    const err = new Error(
      'plan must be a self-serve paid plan (one of: ' + selfServePaidPlans().join(', ') +
      '). Enterprise is sales-led - route it to /v1/sales/demo-request.'
    );
    err.code = 'invalid_plan';
    err.statusCode = 400;
    err.plan = opts.plan;
    throw err;
  }

  // Validate exactly what THIS checkout needs and report it precisely.
  const secret = stripeSecretKey();
  const priceId = priceIdFor(plan);
  const missing = [];
  if (!secret) missing.push('STRIPE_SECRET_KEY');
  if (!priceId) missing.push(priceEnvVar(plan));
  if (missing.length) {
    throw new BillingNotConfiguredError(
      missing,
      `Cannot start Stripe Checkout for "${plan}": price id / secret key not configured. ` +
      'Operator must set these in AWS/Vercel: ' + missing.join(', ')
    );
  }

  const publicBase = (process.env.PUBLIC_BASE || 'https://kolm.ai').replace(/\/+$/, '');
  const stripeBaseUrl = envSecret('KOLM_STRIPE_BASE_URL') || undefined;
  const doFetch = opts.fetchImpl || globalThis.fetch;
  if (typeof doFetch !== 'function') {
    const err = new Error('global fetch is unavailable (requires Node >= 18)');
    err.code = 'fetch_unavailable';
    err.statusCode = 500;
    throw err;
  }

  const session = await postCheckoutSession({
    priceId,
    customerEmail: opts.email || undefined,
    tenantId: String(tenantId),
    plan,
    quantity: opts.quantity,
    successUrl: opts.successUrl || `${publicBase}/account?upgrade=success&plan=${encodeURIComponent(plan)}&session_id={CHECKOUT_SESSION_ID}`,
    cancelUrl: opts.cancelUrl || `${publicBase}/pricing?upgrade=cancelled&plan=${encodeURIComponent(plan)}`,
    apiKey: secret,
    baseUrl: stripeBaseUrl,
    fetchImpl: doFetch,
  });

  return {
    id: session.id,
    url: session.url,
    plan,
    tenant: String(tenantId),
    source: 'stripe_checkout_api',
    livemode: /^sk_live_/.test(secret),
  };
}

export default {
  selfServePaidPlans,
  stripeSecretKey,
  priceEnvVar,
  priceIdFor,
  billingReady,
  createCheckoutSession,
  BillingNotConfiguredError,
};

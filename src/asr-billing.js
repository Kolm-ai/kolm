// src/asr-billing.js
//
// Stripe billing for the Agent Security-Review (ASR) PRODUCTS - distinct from
// the gateway PLAN billing in billing-activation.js. Three products:
//
//   report   one-time   $750   Signed Readiness Report (mode:'payment')
//   starter  recurring  $299   Continuous Starter        (mode:'subscription')
//   growth   recurring  $999   Continuous Growth         (mode:'subscription')
//
// Two ways to charge, mirroring billing-activation.js so ONE operator config
// powers both:
//   1. Checkout Sessions API  - needs STRIPE_SECRET_KEY + a price id env.
//                               Carries real metadata[kolm_product] back.
//   2. Hosted Payment Link    - needs only the *_LINK env + STRIPE_WEBHOOK_SECRET
//                               (current prod has no secret key). Payment Links
//                               cannot carry arbitrary metadata via URL, so the
//                               audit / subscription binding is encoded into
//                               client_reference_id (the one field a Payment Link
//                               does round-trip): "asrrep_<audit_id>" for a
//                               one-time report, "asrsub_<product>_<tenant_id>"
//                               for a subscription. parseAsrRef() decodes it in
//                               the webhook.
//
// When NOTHING is configured for a product, createAsrCheckout throws
// BillingNotConfiguredError (503) listing the EXACT env var names - so the route
// degrades to "configure to sell" rather than 500ing. ZERO new npm deps: Stripe
// REST over global fetch, same as billing-activation.js.

import { envSecret } from './env.js';
import { stripeSecretKey, BillingNotConfiguredError } from './billing-activation.js';
import { appendCheckoutParams } from './stripe.js';

// The ASR product catalog. amount_cents is for display + a defensive webhook
// cross-check; it is NEVER what the webhook branches on (it branches on the
// client_reference_id prefix / metadata.kolm_product, never on amount).
export const ASR_PRODUCTS = Object.freeze({
  report: {
    kind: 'one_time', amount_cents: 75000, label: 'Signed Readiness Report',
    price_env: 'KOLM_STRIPE_PRICE_ASR_REPORT', link_env: 'STRIPE_PAYMENT_LINK_ASR_REPORT',
    kolm_product: 'asr_report',
  },
  starter: {
    kind: 'subscription', amount_cents: 29900, label: 'Continuous Starter',
    price_env: 'KOLM_STRIPE_PRICE_ASR_CONTINUOUS_STARTER', link_env: 'STRIPE_PAYMENT_LINK_ASR_CONTINUOUS_STARTER',
    kolm_product: 'asr_continuous',
  },
  growth: {
    kind: 'subscription', amount_cents: 99900, label: 'Continuous Growth',
    price_env: 'KOLM_STRIPE_PRICE_ASR_CONTINUOUS_GROWTH', link_env: 'STRIPE_PAYMENT_LINK_ASR_CONTINUOUS_GROWTH',
    kolm_product: 'asr_continuous',
  },
  // Full Readiness is a one-time CHARGE (mode:'payment') but a tenant-bound
  // PACKAGE entitlement, not an upgrade of one specific audit. The Stripe
  // checkout mode keys off `kind` ('one_time' -> mode:'payment'); the FULFILLMENT
  // routing keys off kolm_product:'asr_package' (-> an 'asrpkg_<tenant>' ref and
  // fulfillPackagePurchase), so it never touches the per-audit report path.
  // Deep Red-Team ($10k) and Reviewed Attestation ($25k) are one-time CHARGES
  // (mode:'payment') that grant a DURABLE, tenant-bound ENTITLEMENT (an
  // asr_entitlements row), not an upgrade of one specific audit and not a
  // recurring subscription. The checkout mode keys off `kind` ('one_time' ->
  // mode:'payment'); the FULFILLMENT routing keys off kolm_product
  // ('asr_redteam' / 'asr_reviewed' -> an 'asrent_<kind>_<tenant>' ref +
  // fulfillEntitlementPurchase). entitlement_kind is the row's durable `kind`.
  redteam: {
    kind: 'one_time', amount_cents: 1000000, label: 'Deep Red-Team',
    price_env: 'KOLM_STRIPE_PRICE_ASR_DEEP_REDTEAM', link_env: 'STRIPE_PAYMENT_LINK_ASR_DEEP_REDTEAM',
    kolm_product: 'asr_redteam', entitlement_kind: 'deep_redteam', ref_kind: 'redteam',
  },
  reviewed: {
    kind: 'one_time', amount_cents: 2500000, label: 'Reviewed Attestation',
    price_env: 'KOLM_STRIPE_PRICE_ASR_REVIEWED', link_env: 'STRIPE_PAYMENT_LINK_ASR_REVIEWED',
    kolm_product: 'asr_reviewed', entitlement_kind: 'reviewed_attestation', ref_kind: 'reviewed',
  },
  full: {
    kind: 'one_time', amount_cents: 1500000, label: 'Full Readiness',
    price_env: 'KOLM_STRIPE_PRICE_ASR_FULL_READINESS', link_env: 'STRIPE_PAYMENT_LINK_ASR_FULL_READINESS',
    kolm_product: 'asr_package',
  },
  // Continuous-Plus reuses the EXISTING subscription path end-to-end: it encodes
  // an 'asrsub_plus_<tenant>' ref, flows through the asr_continuous webhook
  // branch, and activates via activateSubscription with product_key 'plus'.
  plus: {
    kind: 'subscription', amount_cents: 350000, label: 'Continuous-Plus',
    price_env: 'KOLM_STRIPE_PRICE_ASR_CONTINUOUS_PLUS', link_env: 'STRIPE_PAYMENT_LINK_ASR_CONTINUOUS_PLUS',
    kolm_product: 'asr_continuous',
  },
});

// Products whose fulfillment is a durable tenant PACKAGE entitlement (asr_packages)
// rather than an audit upgrade or a subscription. Keyed by kolm_product so the
// ref/checkout logic stays declarative as more packages are added.
function _isPackage(def) { return !!def && def.kolm_product === 'asr_package'; }

// Products whose fulfillment is a durable tenant ENTITLEMENT (asr_entitlements):
// Deep Red-Team / Reviewed Attestation. Keyed by entitlement_kind on the def.
function _isEntitlement(def) { return !!def && !!def.entitlement_kind; }

// Map an 'asrent_' ref kind (redteam|reviewed) back to its product key.
const _ENTITLEMENT_PRODUCT_BY_REF_KIND = Object.freeze({ redteam: 'redteam', reviewed: 'reviewed' });

// Encode the binding into client_reference_id (alphanumeric + _ only - within
// Stripe's allowed charset). audit_id is "audses_<hex>"; tenant_id is
// "tenant_<hex>"; product is "starter"|"growth" (no underscore), so the
// delimiters below parse unambiguously.
export function encodeAsrRef({ product, audit_id, tenant_id }) {
  const def = ASR_PRODUCTS[product];
  if (!def) return null;
  // A tenant-bound entitlement (Deep Red-Team / Reviewed Attestation): bind to
  // the tenant via the 'asrent_<ref_kind>_<tenant>' ref.
  if (_isEntitlement(def)) return tenant_id ? `asrent_${def.ref_kind}_${tenant_id}` : null;
  // A tenant-bound package (Full Readiness): bind to the tenant, not an audit.
  if (_isPackage(def)) return tenant_id ? `asrpkg_${tenant_id}` : null;
  if (def.kind === 'one_time') return audit_id ? `asrrep_${audit_id}` : null;
  return tenant_id ? `asrsub_${product}_${tenant_id}` : null;
}

// Decode a client_reference_id back to { product, kind, audit_id?|tenant_id? }.
// Returns null for any non-ASR ref (so the webhook can fall through to gateway
// plan handling). Pure, never throws.
export function parseAsrRef(ref) {
  if (!ref || typeof ref !== 'string') return null;
  if (ref.startsWith('asrent_')) {
    // 'asrent_<ref_kind>_<tenant_id>' - a tenant-bound entitlement (Deep
    // Red-Team / Reviewed Attestation). ref_kind ('redteam'|'reviewed') has no
    // underscore, so the FIRST '_' after the prefix splits kind from tenant.
    const rest = ref.slice('asrent_'.length);
    const us = rest.indexOf('_');
    if (us <= 0) return null;
    const refKind = rest.slice(0, us);
    const tenant_id = rest.slice(us + 1);
    const product = _ENTITLEMENT_PRODUCT_BY_REF_KIND[refKind];
    if (!product || !tenant_id || !ASR_PRODUCTS[product]) return null;
    return { product, kind: 'entitlement', tenant_id, entitlement_kind: ASR_PRODUCTS[product].entitlement_kind };
  }
  if (ref.startsWith('asrpkg_')) {
    // 'asrpkg_<tenant_id>' - a tenant-bound package. There is one package
    // product (Full Readiness); the Checkout-API path also carries product_key
    // in metadata, but the ref alone resolves to 'full' so the Payment-Link path
    // (no metadata) still binds correctly.
    const tenant_id = ref.slice('asrpkg_'.length);
    return tenant_id ? { product: 'full', kind: 'package', tenant_id } : null;
  }
  if (ref.startsWith('asrrep_')) {
    const audit_id = ref.slice('asrrep_'.length);
    return audit_id ? { product: 'report', kind: 'one_time', audit_id } : null;
  }
  if (ref.startsWith('asrsub_')) {
    const rest = ref.slice('asrsub_'.length);
    const us = rest.indexOf('_');
    if (us <= 0) return null;
    const product = rest.slice(0, us);
    const tenant_id = rest.slice(us + 1);
    if (!ASR_PRODUCTS[product] || ASR_PRODUCTS[product].kind !== 'subscription' || !tenant_id) return null;
    return { product, kind: 'subscription', tenant_id };
  }
  return null;
}

// Is ASR billing configured to actually charge? Shape mirrors billingReady().
export function asrBillingReady() {
  const webhook = !!envSecret('STRIPE_WEBHOOK_SECRET');
  const secret = !!stripeSecretKey();
  const products = {};
  const missing = [];
  let anyConfigured = false;
  for (const [k, def] of Object.entries(ASR_PRODUCTS)) {
    const havePrice = !!envSecret(def.price_env);
    const haveLink = !!envSecret(def.link_env);
    const configured = (havePrice && secret) || haveLink;
    products[k] = configured ? (haveLink ? 'payment_link' : 'price_id') : false;
    if (configured) anyConfigured = true;
    else missing.push(`${def.price_env} (or ${def.link_env})`);
  }
  if (!webhook) missing.push('STRIPE_WEBHOOK_SECRET');
  return { ready: anyConfigured && webhook, webhook_secret: webhook, secret_key: secret, products, missing };
}

function _publicBase() {
  return (process.env.PUBLIC_BASE || process.env.KOLM_VERIFY_URL_BASE || 'https://kolm.ai').replace(/\/+$/, '');
}

// Stripe REST: POST /v1/checkout/sessions for an ASR product. Parameterized mode
// ('payment' for the one-time report, 'subscription' for continuous) + real
// metadata. Mirrors billing-activation.postCheckoutSession but is self-contained.
async function postAsrCheckoutSession({ priceId, mode, customerEmail, clientReferenceId, metadata, successUrl, cancelUrl, apiKey, baseUrl, fetchImpl }) {
  const doFetch = fetchImpl || globalThis.fetch;
  if (typeof doFetch !== 'function') {
    const err = new Error('global fetch is unavailable (requires Node >= 18)');
    err.code = 'fetch_unavailable'; err.statusCode = 500; throw err;
  }
  const base = (baseUrl || 'https://api.stripe.com').replace(/\/+$/, '');
  const params = new URLSearchParams();
  params.set('mode', mode);
  params.set('line_items[0][price]', priceId);
  params.set('line_items[0][quantity]', '1');
  params.set('success_url', successUrl);
  params.set('cancel_url', cancelUrl);
  if (mode === 'subscription') params.set('allow_promotion_codes', 'true');
  if (customerEmail) params.set('customer_email', customerEmail);
  if (clientReferenceId) params.set('client_reference_id', clientReferenceId);
  for (const [k, v] of Object.entries(metadata || {})) {
    if (v != null) {
      params.set(`metadata[${k}]`, String(v));
      // Mirror onto the subscription so subscription.* webhook events also carry it.
      if (mode === 'subscription') params.set(`subscription_data[metadata][${k}]`, String(v));
    }
  }
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), 10000);
  try {
    const resp = await doFetch(base + '/v1/checkout/sessions', {
      method: 'POST',
      headers: { authorization: `Bearer ${apiKey}`, 'content-type': 'application/x-www-form-urlencoded' },
      body: params.toString(),
      signal: ctl.signal,
    });
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok || !body || !body.url) {
      const msg = body && body.error && body.error.message ? body.error.message : `stripe_http_${resp.status}`;
      const err = new Error(msg); err.code = 'stripe_api_error'; err.statusCode = 502; err.stripe_status = resp.status; throw err;
    }
    return body;
  } finally {
    clearTimeout(timer);
  }
}

// createAsrCheckout - start a Checkout for an ASR product.
//   opts.product   'report' | 'starter' | 'growth'   (required)
//   opts.tenant    tenant id string OR object with .id (required)
//   opts.audit_id  the audit to upgrade (required for 'report')
//   opts.email     prefill (optional)
// Returns { url, source, product }. Throws BillingNotConfiguredError (503) with
// the exact env var names when the product is not wired.
export async function createAsrCheckout(opts = {}) {
  const product = opts.product;
  const def = ASR_PRODUCTS[product];
  if (!def) {
    const err = new Error(`invalid ASR product "${product}" (one of: ${Object.keys(ASR_PRODUCTS).join(', ')})`);
    err.code = 'invalid_product'; err.statusCode = 400; throw err;
  }
  const tenantId = opts.tenant && typeof opts.tenant === 'object' ? (opts.tenant.id || opts.tenant.tenant) : opts.tenant;
  if (!tenantId) { const err = new Error('tenant is required'); err.code = 'tenant_required'; err.statusCode = 400; throw err; }
  // The audit-bound one-time report needs an audit to upgrade; a tenant-bound
  // package (Full Readiness) does not (it is purchased ahead of any scan).
  if (def.kind === 'one_time' && !_isPackage(def) && !opts.audit_id) {
    const err = new Error('audit_id is required for the one-time report'); err.code = 'audit_required'; err.statusCode = 400; throw err;
  }

  const ref = encodeAsrRef({ product, audit_id: opts.audit_id, tenant_id: String(tenantId) });
  const successUrl = `${_publicBase()}/dashboard?asr=${encodeURIComponent(product)}&session_id={CHECKOUT_SESSION_ID}`;
  const cancelUrl = `${_publicBase()}/pricing?asr=cancelled`;

  // Path 1: Checkout Sessions API (secret key + price id present).
  const secret = stripeSecretKey();
  const priceId = envSecret(def.price_env);
  if (secret && priceId) {
    const session = await postAsrCheckoutSession({
      priceId,
      mode: def.kind === 'one_time' ? 'payment' : 'subscription',
      customerEmail: opts.email || undefined,
      clientReferenceId: ref,
      metadata: {
        kolm_product: def.kolm_product,
        product_key: product,
        tenant_id: String(tenantId),
        ...(opts.audit_id ? { audit_id: String(opts.audit_id) } : {}),
      },
      successUrl, cancelUrl, apiKey: secret,
      baseUrl: envSecret('KOLM_STRIPE_BASE_URL') || undefined,
      fetchImpl: opts.fetchImpl,
    });
    return { url: session.url, id: session.id, source: 'stripe_checkout_api', product };
  }

  // Path 2: hosted Payment Link (link env present). The binding rides in
  // client_reference_id; success/cancel URLs come from the link's own config.
  const link = envSecret(def.link_env);
  if (link) {
    return { url: appendCheckoutParams(link, { tenantId: ref, email: opts.email }), source: 'payment_link', product };
  }

  // Nothing configured - fail loud with the exact env names.
  const missing = [`${def.price_env} (or ${def.link_env})`];
  if (!envSecret('STRIPE_WEBHOOK_SECRET')) missing.push('STRIPE_WEBHOOK_SECRET');
  throw new BillingNotConfiguredError(
    missing,
    `Cannot start checkout for ASR "${product}": not configured. Set in AWS/Vercel: ${missing.join(', ')}. ` +
    'Either a hosted Stripe Payment Link, or STRIPE_SECRET_KEY plus the price id.'
  );
}

export default { ASR_PRODUCTS, encodeAsrRef, parseAsrRef, asrBillingReady, createAsrCheckout };

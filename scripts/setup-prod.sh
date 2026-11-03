#!/usr/bin/env bash
# One-shot setter for kolm production env vars on AWS.
#
# Writes each variable as a SecureString parameter in SSM Parameter Store under
# $KOLM_SSM_PREFIX (default /kolm/prod). The ECS task definition references
# these parameters by ARN in its `secrets` block, so a redeploy picks them up.
#
# Fill in the values you have, comment out the ones you don't yet, and run:
#   bash scripts/setup-prod.sh
#
# Requires the AWS CLI with credentials that can ssm:PutParameter in the
# target account/region (AWS_PROFILE / AWS_REGION as usual).
#
# Anything left as <YOURS> is skipped — re-run after you create the OAuth
# apps / Resend API key / etc. Idempotent (--overwrite).

set -euo pipefail

PREFIX="${KOLM_SSM_PREFIX:-/kolm/prod}"
REGION_FLAG=()
if [ -n "${AWS_REGION:-}" ]; then REGION_FLAG=(--region "$AWS_REGION"); fi

# Source the operator's production env file if present. This file is .gitignored
# (covered by .env.* in .gitignore) — paste the live Stripe payment-link URLs +
# secrets there once, then any future invocation of this script picks them up
# without leaving secrets in version control.
PROD_ENV_FILE="${KOLM_PROD_ENV_FILE:-scripts/.env.production}"
if [ -f "$PROD_ENV_FILE" ]; then
  set -a; . "$PROD_ENV_FILE"; set +a
  echo "loaded $PROD_ENV_FILE"
fi

set_var() {
  local name="$1"
  local value="$2"
  if [ -z "$value" ] || [ "$value" = "<YOURS>" ]; then
    echo "skip  $name (not set)"
    return
  fi
  # MSYS_NO_PATHCONV=1 prevents Git Bash for Windows from rewriting the
  # /kolm/prod/... parameter name into C:/Program Files/Git/kolm/prod/...
  MSYS_NO_PATHCONV=1 aws ssm put-parameter "${REGION_FLAG[@]}" \
    --name "$PREFIX/$name" --type SecureString --overwrite \
    --value "$value" >/dev/null
  echo "ok    $PREFIX/$name"
}

# --- Stripe ----------------------------------------------------------------
# Payment links are env-driven so the URLs never live in version control. Set
# them in scripts/.env.production (gitignored) or export before running. The
# canonical monthly amounts ($9 / $49 / $149 / $1,499 / $2,999) are checked
# server-side via planFromAmount() — every Payment Link must charge exactly
# the canonical price for plan resolution to work.
set_var STRIPE_PAYMENT_LINK_STARTER  "${STRIPE_PAYMENT_LINK_STARTER:-}"
set_var STRIPE_PAYMENT_LINK_PRO      "${STRIPE_PAYMENT_LINK_PRO:-}"
set_var STRIPE_PAYMENT_LINK_TEAMS    "${STRIPE_PAYMENT_LINK_TEAMS:-}"
set_var STRIPE_PAYMENT_LINK_BUSINESS "${STRIPE_PAYMENT_LINK_BUSINESS:-}"
set_var STRIPE_PAYMENT_LINK_ENT      "${STRIPE_PAYMENT_LINK_ENT:-}"

# Webhook secret + secret key — both env-driven via scripts/.env.production.
# Webhook signs `checkout.session.completed` etc.; secret key powers the
# subscription-cancel-on-delete path in /v1/account/delete.
set_var STRIPE_WEBHOOK_SECRET "${STRIPE_WEBHOOK_SECRET:-}"
set_var STRIPE_SECRET_KEY     "${STRIPE_SECRET_KEY:-}"

# --- OAuth -----------------------------------------------------------------
# Google: https://console.cloud.google.com/apis/credentials
#   Authorized redirect URI: https://kolm.ai/v1/oauth/google/callback
set_var GOOGLE_OAUTH_CLIENT_ID     "${GOOGLE_OAUTH_CLIENT_ID:-}"
set_var GOOGLE_OAUTH_CLIENT_SECRET "${GOOGLE_OAUTH_CLIENT_SECRET:-}"

# GitHub: https://github.com/settings/developers -> New OAuth App
#   Homepage URL:               https://kolm.ai
#   Authorization callback URL: https://kolm.ai/v1/oauth/github/callback
set_var GITHUB_OAUTH_CLIENT_ID     "${GITHUB_OAUTH_CLIENT_ID:-}"
set_var GITHUB_OAUTH_CLIENT_SECRET "${GITHUB_OAUTH_CLIENT_SECRET:-}"

set_var OAUTH_REDIRECT_BASE "${OAUTH_REDIRECT_BASE:-https://kolm.ai}"

# --- Persistence -----------------------------------------------------------
# Required for /ready to return ready in production. KOLM_DATA_DIR must point
# at the EFS mount declared in the task definition (the Dockerfile default is
# /var/lib/kolm; docker-entrypoint.sh chowns it and drops to the node user).
# Without a persistent mount this is ephemeral (wiped on every task restart).
set_var KOLM_DATA_DIR        "${KOLM_DATA_DIR:-/var/lib/kolm}"
set_var KOLM_ARTIFACT_DIR    "${KOLM_ARTIFACT_DIR:-/var/lib/kolm/artifacts}"
set_var KOLM_STORE_DRIVER    "${KOLM_STORE_DRIVER:-sqlite}"

# --- API keys --------------------------------------------------------------
# ANTHROPIC_API_KEY powers the model calls; ADMIN_KEY gates /v1/admin/*. Both
# are in the task definition secrets block, so the task will not start without
# them (scripts/aws/check-ssm-params.sh lists what is still missing).
set_var ANTHROPIC_API_KEY "${ANTHROPIC_API_KEY:-}"
set_var ADMIN_KEY         "${ADMIN_KEY:-}"

# --- Receipts --------------------------------------------------------------
# HMAC secret for /v1/compile receipts; the Ed25519 key pair lives under
# $KOLM_DATA_DIR/keys on the volume and is NOT an env var.
set_var RECIPE_RECEIPT_SECRET "${RECIPE_RECEIPT_SECRET:-}"

# --- Email (Resend) --------------------------------------------------------
# 1. Sign up at https://resend.com (Google login works)
# 2. Add the kolm.ai domain at https://resend.com/domains
# 3. Create an API key at https://resend.com/api-keys
# 4. Set RESEND_API_KEY in scripts/.env.production (re_... key)
set_var RESEND_API_KEY "${RESEND_API_KEY:-}"
set_var EMAIL_FROM     "${EMAIL_FROM:-kolm <hello@kolm.ai>}"
set_var EMAIL_REPLY_TO "${EMAIL_REPLY_TO:-dev@kolm.ai}"

echo ""
echo "done. Force a new deployment so the task picks up the parameters:"
echo "  aws ecs update-service --cluster kolm-api --service kolm-api --force-new-deployment"

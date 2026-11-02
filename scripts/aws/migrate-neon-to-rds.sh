#!/usr/bin/env bash
# Copies one Neon database into an RDS instance made by infra/aws/rds-postgres.yaml.
#
#   SOURCE_URL='postgres://...-pooler-less-host.neon.tech/db?sslmode=require' \
#   scripts/aws/migrate-neon-to-rds.sh <rds-stack> <app> [db-name]
#
# Steps:
#   1. read the RDS master credentials from the Secrets Manager secret the stack owns
#   2. create role <app> with a fresh random password and database <db-name>
#      owned by it (re-running rotates the password: re-sync the app's env after)
#   3. store the app's DATABASE_URL at SSM /<app>/prod/DATABASE_URL (SecureString)
#   4. pg_dump --format=custom from Neon (use the UNPOOLED host), pg_restore into RDS
#   5. compare per-table row counts on both sides
#
# Needs: psql, pg_dump, pg_restore (PostgreSQL 16 client tools), aws CLI,
# python3, curl. No secret is printed; the outputs are the SSM parameter name,
# the restore log path and the row-count table.
# Env: AWS_REGION (us-east-1), SSM_NAME (/<app>/prod/DATABASE_URL),
#      WORK_DIR (a scratch directory), RESTORE_JOBS (4), ALLOW_RESTORE_ERRORS (unset),
#      RECREATE_DB (unset; set to drop and recreate the database first, for the
#      final sync during the write freeze).
set -euo pipefail
export MSYS_NO_PATHCONV=1

STACK="${1:?usage: migrate-neon-to-rds.sh <rds-stack> <app> [db-name]}"
APP="${2:?usage: migrate-neon-to-rds.sh <rds-stack> <app> [db-name]}"
DB="${3:-$APP}"
: "${SOURCE_URL:?set SOURCE_URL to the Neon connection string (unpooled host, sslmode=require)}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
SSM_NAME="${SSM_NAME:-/$APP/prod/DATABASE_URL}"
WORK="${WORK_DIR:-${TMPDIR:-/tmp}/neon-to-rds-$APP}"
mkdir -p "$WORK"

for tool in psql pg_dump pg_restore aws python3 curl; do
  command -v "$tool" >/dev/null || { echo "missing $tool" >&2; exit 2; }
done
if ! [[ "$APP" =~ ^[a-z_][a-z0-9_]*$ && "$DB" =~ ^[a-z_][a-z0-9_]*$ ]]; then
  echo "app and db names must match [a-z_][a-z0-9_]*" >&2; exit 2
fi

stack_output() {
  aws cloudformation describe-stacks --stack-name "$STACK" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}
HOST="$(stack_output Endpoint)"
PORT="$(stack_output Port)"
SECRET_ARN="$(stack_output MasterSecretArn)"
[ -n "$HOST" ] && [ "$HOST" != "None" ] || { echo "stack $STACK has no Endpoint output" >&2; exit 1; }

# RDS CA bundle so every connection to RDS is verify-full. Neon's certificate
# chains to a public CA, so the Neon side uses sslmode=require without it.
CA="$WORK/global-bundle.pem"
[ -s "$CA" ] || curl -fsSL -o "$CA" https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem
export PGSSLMODE=verify-full PGSSLROOTCERT="$CA"
neon_psql() { env -u PGSSLROOTCERT PGSSLMODE=require psql -v ON_ERROR_STOP=1 -qAt "$SOURCE_URL" "$@"; }

# Master credentials stay in this shell's environment only.
master_json="$(aws secretsmanager get-secret-value --secret-id "$SECRET_ARN" --query SecretString --output text)"
MASTER_USER="$(printf '%s' "$master_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["username"])')"
MASTER_PASS="$(printf '%s' "$master_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["password"])')"
unset master_json
admin_psql() {   # SQL on stdin, against the maintenance database
  PGPASSWORD="$MASTER_PASS" psql -v ON_ERROR_STOP=1 -qAt -h "$HOST" -p "$PORT" -U "$MASTER_USER" -d postgres "$@"
}

echo "==> 1/4 role and database on $HOST"
APP_PASS="$(aws secretsmanager get-random-password --password-length 40 --exclude-punctuation \
  --query RandomPassword --output text)"
if [ "$(printf "select 1 from pg_roles where rolname = '%s'" "$APP" | admin_psql)" = "1" ]; then
  printf "alter role \"%s\" with login password '%s'" "$APP" "$APP_PASS" | admin_psql >/dev/null
  echo "role $APP exists; password rotated"
else
  printf "create role \"%s\" with login password '%s'" "$APP" "$APP_PASS" | admin_psql >/dev/null
  echo "created role $APP"
fi
db_exists() { [ "$(printf "select 1 from pg_database where datname = '%s'" "$DB" | admin_psql)" = "1" ]; }
if [ -n "${RECREATE_DB:-}" ] && db_exists; then
  # The final sync during the write freeze: pg_restore cannot load into a
  # database that already holds the tables, so start from an empty one.
  printf "select pg_terminate_backend(pid) from pg_stat_activity where datname = '%s' and pid <> pg_backend_pid()" "$DB" | admin_psql >/dev/null
  printf 'drop database "%s"' "$DB" | admin_psql >/dev/null
  echo "dropped database $DB (RECREATE_DB)"
fi
if ! db_exists; then
  printf 'create database "%s" owner "%s"' "$DB" "$APP" | admin_psql >/dev/null
  echo "created database $DB owned by $APP"
fi
# The RDS master is not a superuser; it needs membership to restore as the owner.
printf 'grant "%s" to "%s"' "$APP" "$MASTER_USER" | admin_psql >/dev/null

# sslmode=require keeps the Vercel apps working without shipping the CA bundle;
# switch the stored URL to verify-full once each app bundles global-bundle.pem.
aws ssm put-parameter --name "$SSM_NAME" --type SecureString --overwrite \
  --value "postgresql://$APP:$APP_PASS@$HOST:$PORT/$DB?sslmode=require" >/dev/null
unset APP_PASS
echo "stored the app connection string at ssm:$SSM_NAME"

echo "==> 2/4 dump from Neon"
DUMP="$WORK/$APP.dump"
env -u PGSSLROOTCERT PGSSLMODE=require pg_dump --format=custom --no-owner --no-privileges \
  --no-comments --file "$DUMP" "$SOURCE_URL"
ls -l "$DUMP"

echo "==> 3/4 restore into $DB"
TARGET_URL="postgresql://$MASTER_USER@$HOST:$PORT/$DB"
RESTORE_LOG="$WORK/restore-$(date -u +%Y%m%dT%H%M%SZ).log"
set +e
PGPASSWORD="$MASTER_PASS" pg_restore --no-owner --no-privileges --role="$APP" \
  --jobs="${RESTORE_JOBS:-4}" --dbname="$TARGET_URL" "$DUMP" 2> "$RESTORE_LOG"
set -e
errors="$(grep -c '^pg_restore: error' "$RESTORE_LOG" || true)"
echo "restore log: $RESTORE_LOG ($errors errors)"
if [ "$errors" != "0" ]; then
  grep '^pg_restore: error' "$RESTORE_LOG" | head -20
  if [ -z "${ALLOW_RESTORE_ERRORS:-}" ]; then
    echo "pg_restore reported errors; inspect the log, fix, and re-run (or ALLOW_RESTORE_ERRORS=1 to accept, e.g. Neon-only extensions)" >&2
    exit 1
  fi
fi

echo "==> 4/4 row counts"
tables="$(neon_psql -c "select quote_ident(schemaname) || '.' || quote_ident(tablename) from pg_tables where schemaname not in ('pg_catalog', 'information_schema') order by 1")"
mismatch=0
printf '%-56s %12s %12s\n' table neon rds
while IFS= read -r t; do
  [ -n "$t" ] || continue
  src="$(neon_psql -c "select count(*) from $t")"
  dst="$(PGPASSWORD="$MASTER_PASS" psql -qAt "$TARGET_URL" -c "select count(*) from $t" 2>/dev/null || echo missing)"
  printf '%-56s %12s %12s\n' "$t" "$src" "$dst"
  [ "$src" = "$dst" ] || mismatch=$((mismatch + 1))
done <<< "$tables"
if [ "$mismatch" != "0" ]; then
  echo "$mismatch table(s) differ" >&2; exit 1
fi
echo "all tables match"
echo
echo "next: give the app its new DATABASE_URL without displaying it, e.g."
echo "  aws ssm get-parameter --name $SSM_NAME --with-decryption --query Parameter.Value --output text \\"
echo "    | vercel env add DATABASE_URL production --force"

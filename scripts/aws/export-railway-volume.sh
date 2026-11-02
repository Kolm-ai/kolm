#!/usr/bin/env bash
# Pulls the kolm API data directory off the Railway volume as a tarball,
# through `railway ssh` (needs `railway login` and a running instance).
#
#   scripts/aws/export-railway-volume.sh [out.tgz]
#
# Env: RAILWAY_PROJECT (default: the kolmogorov-stack project id),
#      RAILWAY_SERVICE (default kolmogorov-stack), RAILWAY_ENVIRONMENT (production).
#
# Inside the container the sqlite store is copied with VACUUM INTO, which is a
# consistent snapshot even while the API is serving, into export/kolm.sqlite;
# the live kolm.sqlite* files and the backups/ snapshots are left out.
# events/, keys/ (the Ed25519 signing key) and artifacts are copied as they
# are, so run this during the write freeze. railway ssh is not a byte-clean
# pipe, so the tarball travels base64-encoded between marker lines and is
# checked with gzip -t after decoding. The tarball holds the signing key:
# keep it on an encrypted disk and delete it after the restore.
set -euo pipefail

OUT="${1:-kolm-railway-$(date -u +%Y%m%dT%H%M%SZ).tgz}"
RAILWAY_PROJECT="${RAILWAY_PROJECT:-9efeea73-97e9-4c7d-ab86-b3ca1e277a92}"
RAILWAY_SERVICE="${RAILWAY_SERVICE:-kolmogorov-stack}"
RAILWAY_ENVIRONMENT="${RAILWAY_ENVIRONMENT:-production}"

command -v railway >/dev/null || { echo "railway CLI not installed" >&2; exit 2; }

remote_script="$(cat <<'REMOTE'
set -e
D="${KOLM_DATA_DIR:-/var/lib/kolm}"
cd "$D"
rm -rf export && mkdir export
node -e '
  const { DatabaseSync } = require("node:sqlite");
  const [src, dst] = process.argv.slice(1);
  const db = new DatabaseSync(src, { readOnly: true });
  db.exec("VACUUM INTO \x27" + dst + "\x27");
  db.close();
' "$D/kolm.sqlite" "$D/export/kolm.sqlite"
echo "__KOLM_EXPORT_BEGIN__"
tar czf - \
  --exclude=./kolm.sqlite --exclude=./kolm.sqlite-wal --exclude=./kolm.sqlite-shm \
  --exclude=./backups --exclude=./lost+found . | base64
echo
echo "__KOLM_EXPORT_END__"
rm -rf "$D/export"
REMOTE
)"
encoded="$(printf '%s\n' "$remote_script" | base64 | tr -d '\r\n')"

raw="$OUT.b64"
echo "exporting $RAILWAY_SERVICE ($RAILWAY_ENVIRONMENT) data dir -> $OUT"
railway ssh -p "$RAILWAY_PROJECT" -s "$RAILWAY_SERVICE" -e "$RAILWAY_ENVIRONMENT" -- \
  sh -c "echo $encoded | base64 -d | sh" > "$raw"

awk '/__KOLM_EXPORT_BEGIN__/ { on = 1; next } /__KOLM_EXPORT_END__/ { on = 0 } on' "$raw" \
  | tr -cd 'A-Za-z0-9+/=' | base64 -d > "$OUT"
rm -f "$raw"

gzip -t "$OUT"
echo "contents:"
tar tzf "$OUT" | awk 'NR <= 25 { print "  " $0 } END { print "  (" NR " entries)" }'
ls -l "$OUT"
echo "next: scripts/aws/restore-volume-to-efs.sh $OUT <s3-bucket>"

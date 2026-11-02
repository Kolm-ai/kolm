#!/usr/bin/env bash
# Verifies that every SSM parameter the kolm-api task definition reads exists.
# The names are parsed out of infra/aws/kolm-api.yaml, so the template is the
# single source of truth; a missing parameter makes every task fail to start.
# Values are never read or printed.
#
#   scripts/aws/check-ssm-params.sh
# Env: AWS_REGION (us-east-1), KOLM_SSM_PREFIX (/kolm/prod).
set -euo pipefail
export MSYS_NO_PATHCONV=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$HERE/../../infra/aws/kolm-api.yaml"
PREFIX="${KOLM_SSM_PREFIX:-/kolm/prod}"
export AWS_REGION="${AWS_REGION:-us-east-1}"

mapfile -t NAMES < <(grep -oE 'parameter\$\{SsmPrefix\}/[A-Z0-9_]+' "$TEMPLATE" | sed 's#.*/##' | sort -u)
if [ "${#NAMES[@]}" -eq 0 ]; then
  echo "no SSM-backed secrets found in $TEMPLATE" >&2; exit 2
fi

missing=()
# get-parameters takes at most 10 names per call.
for ((i = 0; i < ${#NAMES[@]}; i += 10)); do
  full=()
  for n in "${NAMES[@]:i:10}"; do full+=("$PREFIX/$n"); done
  while IFS= read -r p; do
    if [ -n "$p" ] && [ "$p" != "None" ]; then missing+=("$p"); fi
  done < <(aws ssm get-parameters --names "${full[@]}" \
             --query 'InvalidParameters[]' --output text | tr '\t' '\n')
done

echo "checked ${#NAMES[@]} parameters under $PREFIX ($AWS_REGION)"
if [ "${#missing[@]}" -gt 0 ]; then
  echo "missing; set them with scripts/setup-prod.sh or:"
  echo "  aws ssm put-parameter --type SecureString --name <name> --value <value>"
  printf '  %s\n' "${missing[@]}"
  exit 1
fi
echo "all present"

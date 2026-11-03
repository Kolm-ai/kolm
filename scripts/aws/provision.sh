#!/usr/bin/env bash
# Deploys the kolm AWS stacks (infra/aws/*.yaml) in dependency order.
# Idempotent: `aws cloudformation deploy` creates or updates in place.
#
#   scripts/aws/provision.sh network                 # VPC + public subnets, once per region
#   scripts/aws/provision.sh ecr                     # image repository (us-east-1)
#   scripts/aws/provision.sh cert api.kolm.ai        # ACM certificate; prints the DNS validation CNAME
#   scripts/aws/provision.sh rds botspace-db [Param=Value ...]
#   scripts/aws/provision.sh api api.kolm.ai [Param=Value ...]
#   scripts/aws/provision.sh oidc                    # GitHub Actions deploy role
#   scripts/aws/provision.sh outputs <stack>
#
# Env: AWS_REGION (default us-east-1), AWS_PROFILE (optional),
#      NETWORK_STACK (kolm-prod-network), ECR_STACK (kolm-ecr),
#      API_STACK (kolm-api), OIDC_STACK (kolm-github-oidc).
# Order for a fresh account: network, ecr, oidc, push an image
# (deploy-aws.yml with roll=false), cert, check-ssm-params.sh, api.
set -euo pipefail
export MSYS_NO_PATHCONV=1   # Git Bash: keep /kolm/prod style values intact

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATES="$HERE/../../infra/aws"
export AWS_REGION="${AWS_REGION:-us-east-1}"
NETWORK_STACK="${NETWORK_STACK:-kolm-prod-network}"
ECR_STACK="${ECR_STACK:-kolm-ecr}"
API_STACK="${API_STACK:-kolm-api}"
OIDC_STACK="${OIDC_STACK:-kolm-github-oidc}"

usage() { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

outputs() {
  aws cloudformation describe-stacks --stack-name "$1" \
    --query 'Stacks[0].Outputs[].[OutputKey,OutputValue]' --output table
}

# deploy <stack> <template> [--capabilities CAP] [Param=Value ...]
deploy() {
  local stack="$1" template="$2"; shift 2
  local caps=() params=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --capabilities) caps=(--capabilities "$2"); shift 2 ;;
      *=*) params+=("$1"); shift ;;
      *) echo "unexpected argument: $1" >&2; exit 2 ;;
    esac
  done
  echo "==> $stack <- $template ($AWS_REGION)"
  aws cloudformation deploy \
    --stack-name "$stack" \
    --template-file "$TEMPLATES/$template" \
    --no-fail-on-empty-changeset \
    ${caps[@]+"${caps[@]}"} \
    ${params[@]+--parameter-overrides "${params[@]}"}
  outputs "$stack"
}

issued_cert_arn() {
  aws acm list-certificates --certificate-statuses ISSUED \
    --query "CertificateSummaryList[?DomainName=='$1'].CertificateArn | [0]" --output text
}

cert() {
  local host="${1:?usage: provision.sh cert <api-hostname>}" arn
  arn="$(aws acm list-certificates --certificate-statuses ISSUED PENDING_VALIDATION \
    --query "CertificateSummaryList[?DomainName=='$host'].CertificateArn | [0]" --output text)"
  if [ -z "$arn" ] || [ "$arn" = "None" ]; then
    arn="$(aws acm request-certificate --domain-name "$host" --validation-method DNS \
      --query CertificateArn --output text)"
    echo "requested $arn"
    sleep 5   # the validation record appears a few seconds after the request
  fi
  echo "certificate: $arn"
  echo "status:      $(aws acm describe-certificate --certificate-arn "$arn" --query Certificate.Status --output text)"
  echo "DNS validation record (add it at the DNS provider for $host, then re-run to see ISSUED):"
  aws acm describe-certificate --certificate-arn "$arn" \
    --query 'Certificate.DomainValidationOptions[0].ResourceRecord.[Type,Name,Value]' --output text
}

cmd="${1:-}"; [ $# -gt 0 ] && shift
case "$cmd" in
  network) deploy "$NETWORK_STACK" network.yaml "$@" ;;
  ecr)     deploy "$ECR_STACK" kolm-ecr.yaml "$@" ;;
  cert)    cert "$@" ;;
  rds)
    name="${1:?usage: provision.sh rds <stack-name> [Param=Value ...]}"; shift
    deploy "$name" rds-postgres.yaml "NetworkStackName=$NETWORK_STACK" "$@" ;;
  api)
    host="${1:?usage: provision.sh api <api-hostname> [Param=Value ...]}"; shift
    arn="$(issued_cert_arn "$host")"
    if [ -z "$arn" ] || [ "$arn" = "None" ]; then
      echo "no ISSUED ACM certificate for $host in $AWS_REGION; run: $0 cert $host" >&2; exit 1
    fi
    "$HERE/check-ssm-params.sh"
    deploy "$API_STACK" kolm-api.yaml --capabilities CAPABILITY_NAMED_IAM \
      "NetworkStackName=$NETWORK_STACK" "EcrStackName=$ECR_STACK" "CertificateArn=$arn" "$@"
    echo "point the API hostname at the LoadBalancerDnsName output (CNAME), then verify https://$host/health" ;;
  oidc)
    deploy "$OIDC_STACK" github-oidc.yaml --capabilities CAPABILITY_NAMED_IAM \
      "EcrStackName=$ECR_STACK" "ApiStackName=$API_STACK" "$@"
    echo "set the DeployRoleArn output as the AWS_DEPLOY_ROLE_ARN repository variable:"
    echo "  gh variable set AWS_DEPLOY_ROLE_ARN --body <arn>" ;;
  outputs) outputs "${1:?usage: provision.sh outputs <stack>}" ;;
  *) usage ;;
esac

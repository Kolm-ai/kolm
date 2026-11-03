#!/usr/bin/env bash
# Points the `prod` tag at an image that was already pushed (every deploy
# pushes <git sha> and prod), then rolls the ECS service onto it.
#
#   scripts/aws/rollback.sh <git-sha>
#
# Recent candidates:
#   aws ecr describe-images --repository-name kolm/api \
#     --query 'sort_by(imageDetails,&imagePushedAt)[-10:].[imagePushedAt,imageTags]' --output text
# Env: AWS_REGION (us-east-1), ECR_REPOSITORY (kolm/api), ECS_CLUSTER (kolm-api), ECS_SERVICE (kolm-api).
set -euo pipefail

SHA="${1:?usage: rollback.sh <git-sha>}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
REPO="${ECR_REPOSITORY:-kolm/api}"
CLUSTER="${ECS_CLUSTER:-kolm-api}"
SERVICE="${ECS_SERVICE:-kolm-api}"

digest_of() {
  aws ecr describe-images --repository-name "$REPO" --image-ids "imageTag=$1" \
    --query 'imageDetails[0].imageDigest' --output text 2>/dev/null || true
}

target="$(digest_of "$SHA")"
if [ -z "$target" ] || [ "$target" = "None" ]; then
  echo "no image tagged $SHA in $REPO ($AWS_REGION)" >&2; exit 1
fi
current="$(digest_of prod)"

if [ "$current" = "$target" ]; then
  echo "prod already points at $SHA ($target)"
else
  manifest="$(aws ecr batch-get-image --repository-name "$REPO" --image-ids "imageTag=$SHA" \
    --query 'images[0].imageManifest' --output text)"
  media_type="$(aws ecr batch-get-image --repository-name "$REPO" --image-ids "imageTag=$SHA" \
    --query 'images[0].imageManifestMediaType' --output text)"
  aws ecr put-image --repository-name "$REPO" --image-tag prod \
    --image-manifest "$manifest" --image-manifest-media-type "$media_type" \
    --query 'image.imageId.imageDigest' --output text
  echo "prod: $current -> $target ($SHA)"
fi

echo "rolling $CLUSTER/$SERVICE"
aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" --force-new-deployment \
  --query 'service.deployments[0].id' --output text
aws ecs wait services-stable --cluster "$CLUSTER" --services "$SERVICE"
echo "stable on $SHA"

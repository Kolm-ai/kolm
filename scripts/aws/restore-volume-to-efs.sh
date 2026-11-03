#!/usr/bin/env bash
# Seeds the kolm-api EFS volume from a tarball made by export-railway-volume.sh.
#
#   scripts/aws/restore-volume-to-efs.sh <export.tgz> <s3-bucket>
#
# Steps: upload to s3://<bucket>/kolm-seed/, presign it for an hour, scale the
# service to 0 so nothing writes to sqlite, run a one-off task from the
# service's own task definition (same image, EFS mount and network) that
# streams the tarball into /var/lib/kolm, then scale back to 1 and delete the
# upload. The task role has no S3 access, hence the presigned URL. The bucket
# must be private (block public access, default encryption).
# Env: AWS_REGION (us-east-1), ECS_CLUSTER (kolm-api), ECS_SERVICE (kolm-api).
set -euo pipefail

TGZ="${1:?usage: restore-volume-to-efs.sh <export.tgz> <s3-bucket>}"
BUCKET="${2:?usage: restore-volume-to-efs.sh <export.tgz> <s3-bucket>}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
CLUSTER="${ECS_CLUSTER:-kolm-api}"
SERVICE="${ECS_SERVICE:-kolm-api}"

command -v python3 >/dev/null || { echo "python3 is needed to build the task override" >&2; exit 2; }
gzip -t "$TGZ"

key="kolm-seed/$(basename "$TGZ")"
echo "==> upload to s3://$BUCKET/$key"
aws s3 cp --only-show-errors "$TGZ" "s3://$BUCKET/$key"
url="$(aws s3 presign "s3://$BUCKET/$key" --expires-in 3600)"
cleanup() { aws s3 rm --only-show-errors "s3://$BUCKET/$key" || true; }
trap cleanup EXIT

taskdef="$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" \
  --query 'services[0].taskDefinition' --output text)"
netcfg="$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" \
  --query 'services[0].networkConfiguration' --output json)"
[ -n "$taskdef" ] && [ "$taskdef" != "None" ] || { echo "service $CLUSTER/$SERVICE not found" >&2; exit 1; }

echo "==> scale $SERVICE to 0"
aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" --desired-count 0 \
  --query 'service.desiredCount' --output text
aws ecs wait services-stable --cluster "$CLUSTER" --services "$SERVICE"

# The container runs the image's entrypoint (drops to the node user) and then
# this command. The URL goes in as an environment variable so no quoting of it
# is needed inside the shell snippet.
overrides="$(python3 - "$url" <<'PY'
import json, sys
script = (
    "set -e; cd /var/lib/kolm; "
    "wget -qO- \"$SEED_URL\" | tar xzf -; "
    "if [ -f export/kolm.sqlite ]; then "
    "  rm -f kolm.sqlite kolm.sqlite-wal kolm.sqlite-shm; mv export/kolm.sqlite kolm.sqlite; rmdir export; "
    "fi; "
    "echo seeded; ls -la /var/lib/kolm; du -sh /var/lib/kolm"
)
print(json.dumps({"containerOverrides": [{
    "name": "api",
    "command": ["sh", "-c", script],
    "environment": [{"name": "SEED_URL", "value": sys.argv[1]}],
}]}))
PY
)"

echo "==> one-off seed task"
task="$(aws ecs run-task --cluster "$CLUSTER" --launch-type FARGATE --task-definition "$taskdef" \
  --network-configuration "$netcfg" --overrides "$overrides" --started-by volume-seed \
  --query 'tasks[0].taskArn' --output text)"
echo "task $task"
aws ecs wait tasks-stopped --cluster "$CLUSTER" --tasks "$task"
exit_code="$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$task" \
  --query 'tasks[0].containers[0].exitCode' --output text)"
task_id="${task##*/}"
echo "--- task log (/ecs/$CLUSTER api/api/$task_id) ---"
aws logs get-log-events --log-group-name "/ecs/$CLUSTER" --log-stream-name "api/api/$task_id" \
  --query 'events[].message' --output text 2>/dev/null | tail -40 || true
echo "---"
if [ "$exit_code" != "0" ]; then
  echo "seed task exited with $exit_code; the service stays at 0 until you fix and re-run" >&2
  exit 1
fi

echo "==> scale $SERVICE back to 1"
aws ecs update-service --cluster "$CLUSTER" --service "$SERVICE" --desired-count 1 \
  --query 'service.desiredCount' --output text
aws ecs wait services-stable --cluster "$CLUSTER" --services "$SERVICE"
echo "done; check /health and that /v1/whoami accepts an existing key"

# Off Neon and Railway, onto AWS

Status: inventory complete, target confirmed, templates and scripts written
and linted but not yet run against an AWS account; nothing provisioned
(2026-11-02).

Target state: **Vercel** keeps the web apps, static sites and serverless
functions. **AWS** holds everything stateful or long-running: Postgres,
container services, scheduled workers, volumes, object storage. Neon and
Railway go away entirely. Supabase, Upstash and Cloudflare are flagged below
but were not in the original ask; see "Open decisions".

What this repo carries for it:

| Path | What |
|---|---|
| `infra/aws/network.yaml` | VPC with two public subnets, one stack per region (`kolm-prod-network`) |
| `infra/aws/kolm-ecr.yaml` | ECR repository `kolm/api` (stack `kolm-ecr`) |
| `infra/aws/rds-postgres.yaml` | one RDS PostgreSQL 16 instance per stack: public endpoint, `rds.force_ssl=1`, master password in Secrets Manager, deletion protection |
| `infra/aws/kolm-api.yaml` | the kolm API: ECS Fargate service, ALB with ACM, EFS at `/var/lib/kolm`, secrets from SSM, CloudWatch logs and alarms (stack `kolm-api`) |
| `infra/aws/github-oidc.yaml` | the role GitHub Actions assumes to push images and roll the service (stack `kolm-github-oidc`) |
| `.github/workflows/deploy-aws.yml` | build, push the image as `<sha>` and `prod`, roll the service, check `/health`; inert until the `AWS_DEPLOY_ROLE_ARN` repository variable exists |
| `scripts/aws/provision.sh` | deploys the stacks above in order |
| `scripts/aws/check-ssm-params.sh` | lists the SSM parameters the task definition needs that do not exist yet |
| `scripts/aws/migrate-neon-to-rds.sh` | one Neon database into one RDS instance: role, database, dump, restore, row counts |
| `scripts/aws/export-railway-volume.sh` | the kolm data directory off the Railway volume as a tarball |
| `scripts/aws/restore-volume-to-efs.sh` | that tarball onto EFS through a one-off ECS task |
| `scripts/aws/rollback.sh` | repoint `prod` at an earlier image and roll |

All of it is driven by the AWS CLI (the only tool installed locally) and is
idempotent: re-running a `provision.sh` step updates the stack in place.

## 1. What exists today

### 1.1 Neon (via the Vercel Marketplace integration, team `mithril`)

| Neon store (Vercel name) | Neon project | Region | Neon Auth | Used by | Source repo | Driver in code |
|---|---|---|---|---|---|---|
| `botspace-db` | `calm-shape-39871418` | iad1 (us-east-1) | no | `botspace` (prod/preview/dev), app.truffle.tech | github:publu/swarm@main | `pg` Pool on `DATABASE_URL` + `DATABASE_URL_UNPOOLED` (LISTEN/NOTIFY) |
| `neon-aureolin-door` | `proud-cell-38605231` | iad1 | **yes** | `gaithub` (env prefix `NEON2_`) | github:sneaky-hippo/gaithub@omnibreak-restore | `@neondatabase/serverless` HTTP `neon()` |
| `neon-crimson-village` | `plain-meadow-59149599` | iad1 | **yes** | `gaithub` (prod/preview, unprefixed vars) | same | same |
| `hackvc-platform-staging` | `damp-unit-47259473` | iad1 | no | `hackvc-site` development env only | github:sneaky-hippo/hackvc-site@main | none found in the checkout (likely unused) |
| `truffle-authority` | `crimson-poetry-84626643` | sin1 (ap-southeast-1) | **yes** | not connected to any Vercel project. Almost certainly the `truffle-chat` database: Singapore matches its Railway region, and truffle-chat's `DATABASE_URL` / `TRUFFLE_DATABASE_*_URL` are plain env vars | github:sneaky-hippo/truffle-chat@main | `postgres` (postgres.js) with `sslmode=verify-full` and a custom CA PEM; Cloudflare Hyperdrive bindings for the web tier |

Notes

- `gaithub` and `truffle-chat` code contain **no** references to Neon Auth
  (`NEON_AUTH_*`, `@stackframe`). The auth add-on on those three stores is
  probably unused. Confirm in the Neon console before dropping it; if it is
  used, the replacement is Cognito or the app's own user table on RDS.
- `gaithub` must switch driver: `@neondatabase/serverless` in HTTP mode only
  talks to Neon. Use `pg` (what botspace already uses) or `postgres`.
- `hackvc-site` production has `DATABASE_OPS_URL`, `DATABASE_PUBLIC_URL`,
  `DATABASE_WORKER_URL` set (Railway Postgres naming) but the `main` checkout
  has no database code. Treat as stale; verify, then delete.
- Env values on Vercel are not decryptable through the CLI token, so the
  exact host behind each plain `DATABASE_URL` must be read in the Vercel
  dashboard during cutover.
- Vercel Blob stores (`truffle`, `truffle-chat`, `hackvc-site`) are
  Vercel-native and stay.

### 1.2 Railway (inventory partial: CLI token expired, `railway login` needed)

| Railway project | Id | What we know |
|---|---|---|
| `kolmogorov-stack` | `9efeea73-97e9-4c7d-ab86-b3ca1e277a92` | kolm API (`server.js`, `Dockerfile`, `railway.toml`). Public host: the `*.up.railway.app` origin that `vercel.json` rewrites `/v1/*` to. Volume holds `KOLM_DATA_DIR`: sqlite core store, `events/`, `keys/` (Ed25519 receipt signing key), artifacts. |
| `truffle-file-scanner` | `9e599c9b-ed6e-4e81-9f54-5f185abc32d4` | truffle-chat's production worker cohort, region `asia-southeast1`: 14 services (`truffle-access-mail`, `-access-retention`, `-billing-reconciliation`, `-channel-workflow-retention`, `-execution`, `-file-retention`, `-file-scanner`, `-github-worker`, `-identity-mail`, `-meeting-retention`, `-notification-delivery`, `-production-synthetic`, `-provider-broker`, `-slack-adapter`), each with its own Dockerfile, managed by `.railway/railway.ts`. |
| `mithril` | `2e230aa1-1446-4516-b3c7-e4d4ed99dd16` | unknown |
| `slopshop` | `da463871-f5dd-4f79-a356-2ec19b9585d2` | unknown |
| `disciplined-tranquility` | `9b6a5bcd-1143-4cc1-bc47-6bffe9610d49` | unknown (one service) |
| `gaithub` | `0f4d4c7b-b242-402e-9b0f-1f4952a68835` | unknown (one service) |
| `mirai-self` | `d0780021-2aa8-4475-b264-fb12aa2a5ca0` | unknown (one service) |

### 1.3 Vercel (stays)

20 projects in team `mithril`. Only the ones above touch Neon or Railway.
`hub` and `niagara` (mithril.pro) use Supabase + Upstash; `proofmarket`
(benchlist.ai) uses Upstash. `truffle-chat` also deploys a Cloudflare Worker
(`wrangler.jsonc`, `@opennextjs/cloudflare`, 14 Hyperdrive bindings).

## 2. Target on AWS (confirmed)

Regions: **us-east-1** for everything that is iad1 today (kolm, gaithub,
botspace, hackvc). **ap-southeast-1** for truffle (sin1 / asia-southeast1).

| Need | AWS service | Notes |
|---|---|---|
| Postgres for gaithub, botspace | RDS for PostgreSQL 16, one instance in us-east-1 (stack `pg-us-east-1`), one database per product (`gaithub`, `botspace`), one role per product | db.t4g.small single-AZ, gp3 20 GB growing to 100, 7-day automated backups. Multi-AZ is a parameter (`MultiAZ=true`). |
| Postgres for truffle | RDS for PostgreSQL 16 in ap-southeast-1 (stack `pg-ap-southeast-1`), database `truffle`, the 14 existing roles | Same sizing. |
| Reachability from Vercel functions | RDS **publicly accessible**, security group open on 5432 to `ClientCidr` (default `0.0.0.0/0`), `rds.force_ssl=1`; the stored app URLs use `sslmode=require`, move to `verify-full` once each app bundles the RDS CA | Vercel functions have no fixed egress IP. The alternative (RDS Proxy in a private subnet) requires the DB callers to run on AWS; only kolm fits that. |
| kolm API | ECS Fargate service, 1 vCPU / 2 GB, desired count 1, behind an ALB with an ACM cert; `api.kolm.ai` CNAME to the ALB; `/v1/*`, `/health`, `/ready` keep reaching it through the `vercel.json` rewrites | Desired count stays 1 while the core store is sqlite (single writer). Deploys stop the old task before starting the new one for the same reason, so each deploy has one to two minutes of 503 from the ALB. |
| kolm data dir | EFS mounted at `/var/lib/kolm` through an access point (uid/gid 1000) | Carries sqlite, events, keys, artifacts across task restarts. sqlite runs with a rollback journal there (`KOLM_SQLITE_JOURNAL_MODE=TRUNCATE`, set in the task definition) because WAL needs local shared memory. The Dockerfile entrypoint chowns the mount and drops to `node`. |
| truffle workers (14) | Same ECS cluster pattern in ap-southeast-1: long-running ones as services, retention/synthetic jobs as EventBridge scheduled tasks | One task definition per Dockerfile. Not templated yet. |
| Images | ECR `kolm/api`, built by GitHub Actions through an OIDC role, tagged with the git sha and `prod` | No long-lived AWS keys in GitHub. |
| Secrets / env | SSM Parameter Store (`/kolm/prod/<NAME>`), referenced from the task definition's `secrets` block | `scripts/setup-prod.sh` writes there. |
| Object storage | S3 only where a service needs it (the volume seed bucket in Phase 3, truffle R2 replacement if Cloudflare goes) | Vercel Blob stays. |
| IaC | CloudFormation under `infra/aws/`, driven by `scripts/aws/provision.sh` | Terraform under `deploy/terraform` is the BYOC wrapper for customers, not ours to reuse. |

## 3. Migration order

Every command below runs from the repo root in Git Bash with AWS credentials
in the environment (`AWS_PROFILE` or the `AWS_*` variables) and `gh` logged
in. The scripts set `MSYS_NO_PATHCONV` themselves. Nothing prints a secret.

### Phase 0: access and tooling (blocked on you)

1. AWS account plus a role I can assume: `aws configure sso` (preferred) or
   `aws configure` with an IAM user limited to this work. No credentials
   exist on this machine today.
2. `railway login` so the Railway inventory can be finished and the kolm
   volume exported.
3. Install PostgreSQL client tools (`pg_dump`, `pg_restore`, `psql`), for
   example `winget install PostgreSQL.PostgreSQL.16`. Not installed today.
4. Merge this branch: the deploy workflow only runs from `main`.

### Phase 1: provision (us-east-1 first)

Network, registry and the deploy role. `oidc` prints the role ARN; the
`gh variable set` makes the workflow live.

```bash
bash scripts/aws/provision.sh network
```

```bash
bash scripts/aws/provision.sh ecr
```

```bash
bash scripts/aws/provision.sh oidc
```

```bash
gh variable set AWS_DEPLOY_ROLE_ARN --body "$(aws cloudformation describe-stacks --stack-name kolm-github-oidc --query "Stacks[0].Outputs[?OutputKey=='DeployRoleArn'].OutputValue" --output text)"
```

Push the first image. The service does not exist yet, so `roll=false`.

```bash
gh workflow run deploy-aws.yml -f roll=false
```

```bash
gh run watch "$(gh run list --workflow deploy-aws.yml --limit 1 --json databaseId --jq '.[0].databaseId')"
```

Certificate. The command prints a CNAME; add it at the DNS provider for
`kolm.ai`, then re-run until the status is `ISSUED` (minutes).

```bash
bash scripts/aws/provision.sh cert api.kolm.ai
```

Secrets. Put the production values in `scripts/.env.production` (gitignored),
including `ANTHROPIC_API_KEY` and `ADMIN_KEY`, then write them to SSM and
check nothing the task definition references is missing:

```bash
bash scripts/setup-prod.sh
```

```bash
bash scripts/aws/check-ssm-params.sh
```

The API stack, with the service parked at 0 tasks until the volume is seeded
in Phase 3 (otherwise the first task would mint a fresh signing key on the
empty EFS). It takes a few minutes; the last line names the ALB.

```bash
bash scripts/aws/provision.sh api api.kolm.ai DesiredCount=0
```

Add a CNAME `api.kolm.ai` to the `LoadBalancerDnsName` output:

```bash
bash scripts/aws/provision.sh outputs kolm-api
```

Postgres. One instance in us-east-1 for botspace and gaithub, one in
ap-southeast-1 for truffle (that region needs its own network stack). Each
takes 10 to 15 minutes; `ClientCidr`, `InstanceClass`, `MultiAZ` are
parameters if the defaults in section 2 are wrong.

```bash
bash scripts/aws/provision.sh rds pg-us-east-1
```

```bash
AWS_REGION=ap-southeast-1 bash scripts/aws/provision.sh network
```

```bash
AWS_REGION=ap-southeast-1 bash scripts/aws/provision.sh rds pg-ap-southeast-1
```

Nothing is cut over yet; both sides run.

### Phase 2: move the Postgres data (per database, repeatable)

The script creates the role and database, stores the app's connection string
at SSM `/<app>/prod/DATABASE_URL`, dumps from Neon, restores, and compares
per-table row counts. The Neon string is the **unpooled** one: Vercel
dashboard, Storage, the store, `.env.local` tab (`DATABASE_URL_UNPOOLED`), or
the Neon console with "connection pooling" off.

```bash
read -rsp 'Neon unpooled connection string: ' SOURCE_URL; echo; export SOURCE_URL
```

```bash
bash scripts/aws/migrate-neon-to-rds.sh pg-us-east-1 botspace
```

1. Rehearse once without a freeze to time it and to surface restore errors
   (Neon-only extensions show up as `pg_restore: error`; `ALLOW_RESTORE_ERRORS=1`
   accepts them once you have read the log).
2. For the real run, stop the app's writes (pause the Vercel deployment, or
   redeploy with a maintenance flag if the app has one), then re-run with
   `RECREATE_DB=1`, which drops the rehearsal database first so the restore
   lands in an empty one.
3. Hand the app its new URL without displaying it, redeploy, smoke-test.
   For botspace `DATABASE_URL_UNPOOLED` gets the same value (there is no
   pooler in front of RDS):

```bash
aws ssm get-parameter --name /botspace/prod/DATABASE_URL --with-decryption --query Parameter.Value --output text | vercel env add DATABASE_URL production --force --scope mithrilpro
```

4. Remove the Neon store from the Vercel project (Vercel, Storage, store,
   Remove / Delete) once a day on RDS has passed.

| App | Source | Target | Before / after |
|---|---|---|---|
| botspace | `botspace-db` | `pg-us-east-1`, db `botspace` | after: `DATABASE_URL` and `DATABASE_URL_UNPOOLED`; its `scripts/migrate-network.mjs --verify` as a second check |
| gaithub | `neon-crimson-village` (prod/preview) | `pg-us-east-1`, db `gaithub` | before: the driver switch in section 4; after: `DATABASE_URL`, drop `NEON_AUTH_*` |
| gaithub | `neon-aureolin-door` (`NEON2_*`) | `pg-us-east-1`, db `gaithub2`, or nothing | find what reads `NEON2_*` first; if nothing, delete the store |
| truffle | `truffle-authority` | `pg-ap-southeast-1`, db `truffle` (`AWS_REGION=ap-southeast-1`) | the script makes one role `truffle`; truffle-chat's 14 per-worker roles are created by its own setup afterwards |
| hackvc | `hackvc-platform-staging` | none | delete the store and the stale `DATABASE_*` vars |

### Phase 3: kolm cutover

The Railway volume holds the sqlite store, `events/` and the Ed25519 signing
key; all of it moves to EFS. The freeze is the ALB itself: once Vercel sends
`/v1/*` to the ALB while the ECS service sits at 0 tasks, every call gets a
503 and nothing reaches Railway, so the export is consistent. Expect about
five minutes of 503s; Railway keeps running untouched, so the rollback inside
that window is a Vercel rollback.

Prerequisites: Phase 1 done, `api.kolm.ai` resolving to the ALB and
`https://api.kolm.ai/health` answering 503 (no tasks yet), `railway login`
done, and a private bucket for the seed tarball:

```bash
aws s3api create-bucket --bucket <private-bucket-name>
```

```bash
aws s3api put-public-access-block --bucket <private-bucket-name> --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
```

1. Rehearse the data move without a freeze: export, restore, check. The
   restore replaces whatever is on EFS, so this can run as often as needed.
   Keep the tarball outside the repo; it contains the signing key.

```bash
bash scripts/aws/export-railway-volume.sh "$HOME/kolm-railway.tgz"
```

```bash
bash scripts/aws/restore-volume-to-efs.sh "$HOME/kolm-railway.tgz" <private-bucket-name>
```

   The restore scales the service to 1 at the end. Check it, then compare the
   key directory from both origins: the `kid` values in
   `/.well-known/jwks.json` from `https://kolm.ai` (still Railway) and from
   `https://api.kolm.ai` must match, and `/v1/whoami` with an existing API
   key must answer on both.

```bash
curl -fsS https://api.kolm.ai/health && curl -fsS https://api.kolm.ai/ready
```

   Then park the service again for the real run:

```bash
aws ecs update-service --cluster kolm-api --service kolm-api --desired-count 0 --query service.desiredCount
```

2. Prepare the origin switch as one commit, not pushed yet: `vercel.json`
   rewrites for `/health`, `/ready`, `/v1/*` and the CSP `connect-src`;
   the `web/next.config.mjs` default origin (or set `KOLM_API_ORIGIN` on
   Vercel instead); the fallbacks in `api/teacher-chat.js` and
   `api/runpod.js`; the Railway entries in the subprocessor list and the
   SOC 2 control note in `src/router.js` (grep for `'Railway'`).
3. Push it. Vercel deploys and `/v1/*` now ends at the ALB with no tasks:
   503s begin, Railway goes quiet. (The push also runs `deploy-aws.yml`,
   which is harmless at 0 tasks.)
4. Export and restore, same two commands as the rehearsal. The restore ends
   with the service at 1; the 503s stop when `/health` passes (one to two
   minutes).
5. Verify as in step 1 (`jwks.json` from `https://kolm.ai` now comes from
   AWS, so compare it against the rehearsal output), then watch the error
   rate for a day. Let deploys verify themselves from now on:

```bash
gh variable set KOLM_API_HOST --body api.kolm.ai
```

   Rollback inside the window: `vercel rollback` to the previous deployment,
   which sends `/v1/*` back to Railway; nothing on Railway was changed.
   Rollback of a later bad image: `bash scripts/aws/rollback.sh <git-sha>`.
6. After the day: delete the tarball, delete the Railway service and
   project, delete `railway.toml`, update `docs/runbook-rollback.md` and
   `docs/reference/deployment-policy.md` (they still describe Railway as the
   deploy target), update the Railway expectations in
   `tests/wave890-13-deployment.test.js` (#1, #2, #5) and regenerate the
   gitignored `data/w890-13-*.json` snapshots it reads.

### Phase 4: truffle workers, then the rest of Railway

Each truffle Dockerfile becomes an ECS task definition (the `kolm-api.yaml`
service pattern minus the ALB and EFS, env from SSM) in ap-southeast-1.
Hyperdrive: either repoint the 14 configs at RDS (if Cloudflare stays) or
retire the Worker (if Cloudflare goes; see open decisions). Then inventory
and migrate or delete `mithril`, `slopshop`, `disciplined-tranquility`,
`gaithub`, `mirai-self` on Railway.

### Phase 5: teardown

Remove the Neon integration from the Vercel team, delete the Neon projects,
delete all Railway projects, remove the Railway CLI config.

## 4. Code changes per repo

kolm (done in this branch):

- Production detection no longer reads `RAILWAY_ENVIRONMENT`; it reads
  `NODE_ENV`, `VERCEL`, `AWS_EXECUTION_ENV`, `ECS_CONTAINER_METADATA_URI_V4`
  and `AWS_LAMBDA_FUNCTION_NAME` (`src/env.js`, `src/store.js`).
- Region comes from `REGION`, then `AWS_REGION`, then `VERCEL_REGION`
  (`src/router.js`); the build sha from `KOLM_GIT_COMMIT`,
  `VERCEL_GIT_COMMIT_SHA` or `GIT_COMMIT_SHA`.
- `api/teacher-chat.js` and `api/runpod.js` read `KOLM_API_ORIGIN`.
- `scripts/setup-prod.sh` writes SSM parameters instead of Railway vars, now
  including `ANTHROPIC_API_KEY` and `ADMIN_KEY`.
- `Dockerfile` takes a `GIT_COMMIT_SHA` build arg (the workflow passes
  `GIT_COMMIT_SHA=$GITHUB_SHA`) so `/health` reports the deployed commit.
- `src/store.js` honours `KOLM_SQLITE_JOURNAL_MODE` (WAL, DELETE, TRUNCATE or
  PERSIST; default WAL). The task definition sets TRUNCATE for EFS. Verified
  with a second-connection `VACUUM INTO` (what `src/store-backup.js` does)
  and `tests/sota-persistence.test.js` under that mode.
- The templates, workflow and scripts in the table at the top.
- Tests simulate production with `AWS_EXECUTION_ENV=AWS_ECS_FARGATE`.
- Still pointing at the Railway host until Phase 3, because the AWS origin
  does not exist yet: `vercel.json`, the `web/next.config.mjs` default, the
  two `api/*.js` fallbacks, `railway.toml`, and the Railway entries in the
  subprocessor list and SOC 2 control note in `src/router.js`.

gaithub: replace `@neondatabase/serverless` with `pg` in
`packages/core/src/pg-store.ts` and `api/router.js`; drop the `NEON2_*` and
`NEON_AUTH_*` vars; TLS via the RDS CA bundle.

botspace (publu/swarm): no driver change (`pg`). Keep `DATABASE_URL_UNPOOLED`
pointing at the same RDS host (there is no pooler in front of RDS unless you
add PgBouncer or RDS Proxy).

truffle-chat: `.railway/railway.ts` and `scripts/railway-production-cohort.ts`
are replaced by ECS task definitions; `TRUFFLE_POSTGRES_TLS_CA_PEM` becomes
the RDS CA; Hyperdrive decision above.

hackvc-site: delete the dev Neon store and the stale `DATABASE_*` vars.

## 5. Open decisions

1. AWS access (Phase 0). Nothing can be provisioned until then.
2. Compute: ECS Fargate + EFS (chosen) vs one EC2 instance with an EBS
   volume (cheaper, simpler, single point of failure) vs App Runner (no
   persistent volume, so not for kolm). Revisit if sqlite on EFS turns out
   slow (section 6).
3. RDS public endpoint with TLS (chosen, needed for Vercel functions) vs
   moving all DB access behind the kolm and truffle APIs on AWS.
4. Neon Auth on gaithub / truffle-authority: drop (code does not use it) or
   replace with Cognito.
5. Are Supabase + Upstash (hub, niagara, proofmarket) and Cloudflare
   (truffle-chat Worker + Hyperdrive) in scope for "just Vercel and AWS"?
6. Regions: us-east-1 everywhere, or keep truffle in ap-southeast-1.

## 6. Risks

- The kolm Ed25519 key and receipt chain live only on the Railway volume. If
  the volume export is skipped, every issued receipt stops verifying. The
  export tarball holds that key: keep it off the repo, delete it after the
  restore.
- sqlite on EFS is single-writer; never run two kolm tasks against one EFS.
  The service's deployment settings (0% minimum healthy, 100% maximum) exist
  for this, and are why every deploy and rollback has one to two minutes of
  503 from the ALB.
- sqlite on EFS uses a rollback journal instead of WAL, and EFS adds latency
  to each commit. Write-heavy endpoints will be slower than on Railway's
  local volume; watch their latency after cutover and fall back to open
  decision 2 (EC2 + EBS) if it matters.
- A task that starts against an empty EFS creates a new sqlite store and a
  new signing key. The API stack is therefore provisioned at 0 tasks and only
  the restore script scales it up.
- A public RDS endpoint needs `rds.force_ssl=1` and a tight security group;
  rotate the Neon-era passwords at cutover (the migration script mints new
  ones). The stored URLs use `sslmode=require`, which encrypts but does not
  pin the CA; move each app to `verify-full` with the RDS bundle.
- botspace uses LISTEN/NOTIFY on the unpooled URL; RDS supports it, a pooler
  would not.
- None of the templates or scripts has run against a real account yet; the
  first `provision.sh network` is the first real test.

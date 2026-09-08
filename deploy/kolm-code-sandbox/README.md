# kolm-code-sandbox

Remote worktree host for **kolm code** (build spec sections 8 and 11.2). A
Cloudflare Worker fronts one `@cloudflare/sandbox` container per ladder run:

```
/workspace/<runId>/base              shallow clone of the target repository
/workspace/<runId>/<candidateId>     detached git worktree, one per candidate
/workspace/<runId>/<candidateId>.base   the base sha for that worktree
```

The ladder talks to it through `src/ladder/worktree/cloudflare-sandbox.js`,
which implements the same worktree interface as the local git driver.

## Layout

| file | role |
| --- | --- |
| `src/worker.mjs` | entrypoint: exports `KolmCodeSandbox` and `default { fetch(request, env) }` |
| `src/api.mjs` | all routing, auth, path safety and git orchestration; no Cloudflare imports, so it is unit-testable |
| `src/sandbox.mjs` | `class KolmCodeSandbox extends Sandbox` — egress posture and container lifetime |
| `src/hosts.mjs` | the egress allowlist, split out so it can be imported by plain Node tests |
| `test/api.test.mjs` | 35 tests driving `createApi` against a fake sandbox recorder |

## API

Every route requires `Authorization: Bearer <KOLM_CODE_SANDBOX_TOKEN>`; the
comparison is constant time over equal-length buffers. Bodies and responses are
JSON.

| method | path | body / query | response |
| --- | --- | --- | --- |
| POST | `/v1/runs/:runId/worktrees` | `{ repoUrl, baseRef, candidateId, gitToken? }` | `{ id, path, baseSha, cloned, reused }` |
| GET | `/v1/runs/:runId/worktrees/:id` | – | `{ id, path, baseSha }` |
| POST | `/v1/runs/:runId/worktrees/:id/exec` | `{ cmd, cwd?, timeoutMs?, env?, maxOutputBytes? }` | `{ stdout, stderr, exitCode, durationMs, timedOut, truncated }` |
| GET | `…/file?path=` | – | `{ path, content }` or 404 |
| PUT | `…/file` | `{ path, content }` | `{ ok, path, bytes }` |
| DELETE | `…/file?path=` | – | `{ ok, path }` |
| GET | `…/files?glob=&limit=` | – | `{ files: string[] }` |
| GET | `…/search?pattern=&glob=&limit=` | – | `{ hits: [{ path, line, text }] }` |
| GET | `…/diff` | – | `{ diff, truncated }` |
| GET | `…/changed` | – | `{ files: [{ path, status, added, deleted }] }` |
| POST | `…/patch` | `{ diff }` | `{ ok, error? }` |
| POST | `…/commit` | `{ message }` | `{ sha, branch }` |
| DELETE | `/v1/runs/:runId/worktrees/:id` | – | `{ ok: true }` |
| DELETE | `/v1/runs/:runId` | – | `{ ok: true }` |

Error responses are `{ error: <code>, message }` with codes `unauthorized`,
`invalid-id`, `invalid-path`, `invalid-body`, `invalid-command`,
`invalid-repo-url`, `invalid-token`, `invalid-ref`, `worktree-not-found`,
`bad-ref`, `clone-failed`, `fetch-failed`, `worktree-failed`, `commit-failed`,
`git-failed`, `sandbox-exec-failed`, `sandbox-not-configured`, `internal`.

## Constraints kept by the implementation

- **Credentials.** `gitToken` is used only as
  `-c http.extraheader=AUTHORIZATION: Basic base64(x-access-token:<token>)` on
  the `git clone` / `git fetch` command. It is never written to a file, never
  stored in the container's git config, and never echoed in a response. A test
  asserts no other command and no written file carries the value.
- **Path safety.** Every caller-supplied path is rejected if it is absolute,
  contains a `..` segment, a backslash or a control character, and then again if
  its `realpath` resolves outside `/workspace/<runId>/<candidateId>` — so a
  symlink planted by the model cannot be used to read outside the worktree.
- **Shell quoting.** Every interpolated value goes through `shellQuote` (POSIX
  single quoting). Identifiers are additionally restricted to
  `[A-Za-z0-9][A-Za-z0-9._-]{0,63}` and `repoUrl` must be `https://…`.
- **Timeouts and caps.** Commands run under `timeout --kill-after=5s Ns`,
  default 300 s, hard cap 900 s; stdout/stderr are capped at 1 MiB (or the
  caller's smaller `maxOutputBytes`) with `truncated: true`.
- **Idempotent clone.** `/workspace/<runId>/base/.git` is checked before
  cloning, so a second candidate in the same run reuses the clone; an existing
  worktree directory is reported as `reused: true` rather than recreated.
- **`git add -A`** is used in exactly one place, `POST …/commit`, and only ever
  inside a throwaway candidate worktree — never in a developer checkout.
- **Egress.** `enableInternet = false`, `interceptHttps = true` and an
  allowlist of `github.com`, `codeload.github.com`,
  `objects.githubusercontent.com`, `raw.githubusercontent.com`,
  `api.github.com`, `registry.npmjs.org`, `pypi.org`,
  `files.pythonhosted.org`, plus anything in the
  `KOLM_CODE_SANDBOX_ALLOWED_HOSTS` var.

## Develop

```sh
npm install
npm test          # node --test test/*.test.mjs — no network, no container
npm run build     # wrangler dry-run (bundle + config validation)
```

## Deploy

```sh
npx wrangler login                              # account 5d6e3a656784a798c0d763c9eed5214d
npx wrangler secret put KOLM_CODE_SANDBOX_TOKEN # the ladder's bearer token
npx wrangler deploy                             # needs Docker to build the image
```

Then arm the ladder side:

```sh
export KOLM_CODE_SANDBOX_URL=https://kolm-code-sandbox.<subdomain>.workers.dev
export KOLM_CODE_SANDBOX_TOKEN=<the same secret>
```

`workers_dev` and `preview_urls` are disabled in `wrangler.jsonc`; bind a route
or re-enable `workers_dev` before the URL above resolves.

## Build notes

`npm run build` passes `--containers-rollout=none`. Without that flag the
dry-run tries to build the container image locally and fails on a machine
without Docker, with exactly this error:

```
✘ [ERROR] The Docker CLI is needed to build the configured image before deploying
  (even in dry-run mode) but could not be launched.
```

`--containers-rollout=none` is the flag wrangler itself recommends in that
message; it skips the image build while still bundling the worker and
validating the whole configuration (bindings, durable objects, migrations,
container class), which is what the check is for. Run `npm run build:image` on a
machine with Docker to exercise the image build as well; that path has **not**
been verified here because Docker is not installed on this box.

`instance_type` is `standard-1` (wrangler 4.128 renamed `standard`).

# kolm code - operator guide

`kolm code` is a coding agent that treats "is this patch good?" as a
measurement problem instead of a vibe. It generates several candidate patches
in isolated worktrees, pushes each one up a five-rung verification cascade that
gets more expensive at every rung, and stops spending money the moment a
candidate is confidently bad. Only the residue that the cheap rungs cannot
call goes to the expensive judge.

Everything below is reachable as `kolm code <verb>` or, equivalently, as
`node cli/kolm-code.mjs <verb>`.

---

## 1. The 60-second explanation

Two ladders run at right angles to each other.

**The verification ladder** decides how much confidence you have in a patch.
Each rung emits `good`, `bad` or `unsure`, and a rung only runs if the rung
below it did not already say `bad`.

| Rung          | What it measures on the patch                          | Typical cost   | Wall clock |
|---------------|--------------------------------------------------------|----------------|------------|
| L0 meta       | Patch hygiene: lines changed, files touched            | free           | ms         |
| L1 static     | New type errors and new lint errors vs the baseline    | free           | seconds    |
| L2 dynamic    | Newly failing tests, coverage of the changed lines     | free           | seconds+   |
| L3 semantic   | A cheap model scores the diff against a fixed rubric   | fractions of a cent | seconds |
| L4 judge      | The frontier judge, on the unsure residue only         | cents          | seconds    |

The verdict rule is deliberately blunt:

* any rung says `bad` -> **FAIL** (the candidate is dropped, no further spend)
* no `bad`, but something is still `unsure` -> **JUDGE** (escalate to L4)
* every rung that ran said `good` -> **PASS**

A rung's `good`/`bad`/`unsure` call comes from a *band*: a numeric threshold
pair per metric. The good edge is inclusive, the bad edge is exclusive. See
section 10 for reading and calibrating them.

**The generation ladder** decides who writes the patch. `KOLM_CODE_CODER_LADDER`
is an ordered list of model refs, cheapest first. Candidate 0 is written by the
first armed coder, and when a candidate fails L0 a repair pass is made
(`KOLM_CODE_REPAIR_ROUNDS` of them, each capped at `KOLM_CODE_REPAIR_TURNS`
tool-call turns) before escalating to the next rung of the generation ladder.
A repair pass only runs when the failing rung produced something actionable:
failing test names, lint or typecheck output, reviewer findings, judge
blocking items. A timeout or a provider failure is infrastructure, not
feedback, so it emits `repair.skipped` and the candidate keeps its `unsure`.
When a whole coder rung fails at the provider (a 400, an unreachable model),
the run emits `escalate` with `reason: rung-failed` and tries the next rung
instead of ending. That is why run
output shows candidate ids like `c0a0`, `c0a1`, `c1a0`: `c<coder rung>a<attempt>`.

Everything a run learns is written to a SQLite store and summarised in a
signed evidence receipt.

---

## 2. Quick start

The free path needs no API keys at all: run Ollama locally and let it be both
the coder and the reviewer.

```
ollama serve
ollama pull qwen3.8:latest

export KOLM_CODE_ORCHESTRATOR=ollama:qwen3.8:latest
export KOLM_CODE_CODER_LADDER=ollama:qwen3.8:latest
export KOLM_CODE_REVIEWER_LADDER=ollama:qwen3.8:latest

node cli/kolm.js code doctor
node cli/kolm.js code run "add input validation to parseConfig" --repo .
node cli/kolm.js code serve --open
```

`doctor` first, always. It tells you exactly which environment variables are
missing and never prints their values. A run refuses to start unless the
orchestrator, at least one coder, the worktree driver and the store are armed.

---

## 3. Your own keys

Nothing in `kolm code` ships with a key. You bring your own, and there are
three ways to hand one over:

1. **An environment variable** - `export ANTHROPIC_API_KEY=...`. Right for CI,
   for a container, and for a shell you have already set up.
2. **The keystore** - `kolm code keys set anthropic`, typed once and read by
   every later run from a file in your home directory.
3. **Neither** - the rung stays unarmed, and `doctor` names the variable that
   is missing and the `keys set` command that would fill it.

**The environment wins over the keystore.** That is the same precedence `gh`
and the AWS CLI use, and for the same reason: a stored key is a convenience for
one machine, while a job that exports a variable - CI, a container, a one-off
`ANTHROPIC_API_KEY=... kolm code run ...` - has to be able to override it
without editing a file first. `kolm code keys list` shows which one is winning.

The keystore holds exactly six names: `anthropic`, `cerebras`, `deepseek`,
`openrouter`, `github`, `sandbox`. Any other name is refused.

### Storing a key

```
kolm code keys set openrouter                  # prompts, and does not echo
printf '%s' "$KEY" | kolm code keys set openrouter
kolm code keys set openrouter --from-env MY_KEY
kolm code keys set openrouter --from-file ~/secrets/openrouter.txt
```

The value comes from `--from-env VAR`, from the first line of `--from-file`,
from stdin when stdin is a pipe (or with `--stdin`), or from a hidden terminal
prompt - in that order. **A key is never accepted as a command-line argument**:
`kolm code keys set openrouter sk-...` exits with a usage error, because
arguments land in shell history, in `ps` output and in CI logs. On success the
command prints the name, the fingerprint and the file, and nothing else:

```
set openrouter (fingerprint sha256:d9cd1f5fbd1c) at /home/you/.kolm/code/keys.json
```

The fingerprint is the first 12 hex characters of the SHA-256 of the value. It
is enough to tell two keys apart or to confirm that a rotation landed, and it
is not enough to reconstruct the key.

The other subcommands:

| Command | What it does |
|---------|--------------|
| `kolm code keys list` | one row per name: source, fingerprint, when it was set |
| `kolm code keys list --json` | the same as JSON, for scripts |
| `kolm code keys unset <name>` | removes one entry (removing an absent one is fine) |
| `kolm code keys path` | prints the keystore path and nothing else |
| `kolm code keys test [name]` | one live authenticated request per key that is set |

### Where the file lives

```
<KOLM_DATA_DIR or ~/.kolm>/code/keys.json
```

| Variable | Effect |
|----------|--------|
| `KOLM_CODE_KEYS_PATH` | use this file instead (`~` expands); handy for a test or a second profile |
| `KOLM_CODE_NO_KEYSTORE=1` | never read the file at all - CI and the Truffle worker set it |

The file is JSON, written whole through a temp file and a rename so a reader
never sees a half-written keystore:

```json
{
  "version": 1,
  "keys": {
    "openrouter": {
      "value": "...",
      "set_at": "2026-09-09T04:06:24.271Z",
      "fingerprint": "sha256:d9cd1f5fbd1c"
    }
  }
}
```

It is created with mode `0600` - owner read/write only - so another account on
a shared POSIX box cannot read it. **On Windows the mode is ignored**: NTFS
uses ACLs, and Node's `chmod` is a no-op there. What protects the file on
Windows is the ACL of your user profile directory, which is where it sits by
default. If that is not enough for your threat model, keep the key in an
environment variable your session manager supplies, or point
`KOLM_CODE_KEYS_PATH` at a location you have locked down yourself. `kolm code`
does not shell out to `icacls` on your behalf.

A keystore that is corrupt, or that is not a JSON object, is reported by path
with an instruction to fix or delete it - never as a raw parse error. A missing
file is not an error at all: it reads as an empty keystore.

### Checking a key

`kolm code keys test` makes one cheap authenticated request per key that is
actually set - `GET /v1/models` for Anthropic, `GET <base>/models` for
Cerebras, DeepSeek and OpenRouter, `GET /user` for GitHub, `GET /health` for
the sandbox - with a 10-second timeout each:

```
NAME        SOURCE    RESULT    DETAIL
----------  --------  --------  --------
anthropic   env       ok        ok
openrouter  keystore  bad-key   rejected with HTTP 401
```

`ok` means the key was accepted, `bad-key` means the provider rejected it (401
or 403), and `error <status>` means something else went wrong - a timeout, a
network failure, a provider outage - which is not a verdict on the key. The
command exits non-zero when any tested key came back `bad-key`. This is the one
verb here that reaches the network; every other `keys` subcommand is local.

### What never happens to a key

A key value is never printed, and never leaves the process except as an
`Authorization` (or `x-api-key`) header to the provider it belongs to. It does
not appear in a run receipt, in the SQLite store, in `doctor` output, in
`keys list`, in the viewer or in any log line. Both `loadConfig` and
`readKeystore` hold their values in a non-enumerable property, so
`JSON.stringify(config)` and `JSON.stringify(readKeystore(env))` are safe to
paste into an issue. What is reported anywhere is a name, a source (`env` or
`keystore`) and, at most, a fingerprint.

---

## 4. Environment variables

The ARMS column says what the variable turns on. Nothing here is read at
import time; `loadConfig(env)` reads it once per process and freezes the result,
and secrets are held non-enumerably so they cannot be logged by accident.

The six credentials marked (+) below can also come from the keystore instead of
the environment (section 3); the variable wins when both are set.

### Models

| Variable | Default | ARMS |
|----------|---------|------|
| `ANTHROPIC_API_KEY` (+) | - | every `anthropic:*` ref (orchestrator, judge, top coder rung) |
| `CEREBRAS_API_KEY` (or `KOLM_CEREBRAS_TOKEN`) (+) | - | every `cerebras:*` ref |
| `DEEPSEEK_API_KEY` (+) | - | every `deepseek:*` ref |
| `OPENROUTER_API_KEY` (+) | - | every `openrouter:*` ref |
| `KOLM_CODE_OLLAMA_URL` | `http://127.0.0.1:11434/v1` | every `ollama:*` ref, if `GET /api/tags` answers |
| `KOLM_CEREBRAS_URL` | Cerebras public API | a self-hosted or proxied Cerebras endpoint |
| `KOLM_CODE_DEEPSEEK_URL` | DeepSeek public API | a proxied DeepSeek endpoint |
| `KOLM_CODE_OPENROUTER_URL` | OpenRouter public API | a proxied OpenRouter endpoint |
| `KOLM_CODE_DEEPSEEK_OFFPEAK` | unset | off-peak price table for DeepSeek cost estimates |

### Ladders and mode

| Variable | Default | ARMS |
|----------|---------|------|
| `KOLM_CODE_ORCHESTRATOR` | `anthropic:claude-fable-5-1` | the planner and repair model |
| `KOLM_CODE_JUDGE` | same as the orchestrator | the L4 judge |
| `KOLM_CODE_CODER_LADDER` | `cerebras:qwen-3.8-27b,deepseek:deepseek-v4-flash,anthropic:claude-opus-5` | generation, cheapest rung first; unarmed refs are dropped, and `run`, `resume` and `truffle-worker` print one `skipping coder rung ...` line per dropped ref at start; with no coder armed the orchestrator writes the candidates |
| `KOLM_CODE_REVIEWER_LADDER` | `ollama:qwen3.8:latest,cerebras:gpt-oss-120b,deepseek:deepseek-v4-flash` | L3; the first armed ref is used, and L3 returns `skip` if none is |
| `KOLM_CODE_MODE` | `cascade` | `cascade` or `direct` (see section 10) |

### Coordination

How a run is sized, split, and taught. All seven are on by default, so a plain
`kolm code run` already plans, splits, self-reviews, coaches, audits its own
plan and remembers; set any of them to `0` to get the older single-shot
behaviour back.

| Variable | Default | ARMS |
|----------|---------|------|
| `KOLM_CODE_ROUTING` | `auto` | `auto` lets the plan size the run (candidates, rungs, whether to split); `fixed` obeys the flags and the defaults exactly |
| `KOLM_CODE_SUBTASKS` | `1` | splitting a large task into subtasks, each solved and merged in its own branch |
| `KOLM_CODE_SUBTASK_CONCURRENCY` | `2` | how many subtasks run at once (1 to 16) |
| `KOLM_CODE_COACH` | `1` | the coach pass that turns a failed repair round into concrete guidance for the next one |
| `KOLM_CODE_AUDIT` | `1` | the plan audit that inspects an acceptance criterion no candidate can satisfy and may retire or rewrite it (see section 14) |
| `KOLM_CODE_SELF_REVIEW` | `1` | the coder's own review of its diff before the candidate is scored |
| `KOLM_CODE_LESSONS` | `1` | cross-run memory: lessons learned in one run are offered to the next run on the same repo (see `kolm code lessons`) |

A subtask is judged only against the acceptance criteria it can satisfy: the
ones that name its own files, plus the ones that name no subtask file at all.
Criteria about other subtasks' files are left out; a criterion that names both
goes to the subtask whose file it names first and carries a note saying which
part is that subtask's. The reviewer and judge are also told the candidate is
one part of a split plan. `kolm code show` prints the
scoped list inside each subtask's plan.

### Budgets and shape

| Variable | Default | ARMS |
|----------|---------|------|
| `KOLM_CODE_CANDIDATES` | `3` | how many candidates a run generates |
| `KOLM_CODE_CODER_TURNS` | `24` | tool-call turns allowed per candidate |
| `KOLM_CODE_REPAIR_ROUNDS` | `2` | repair passes over a candidate before escalating the generation ladder |
| `KOLM_CODE_REPAIR_TURNS` | `12` | tool-call turns allowed inside one repair pass |
| `KOLM_CODE_MAX_COST_USD` | `5` | hard ceiling; the run stops when it is reached |
| `KOLM_CODE_CALIBRATION_RATE` | `0.1` | fraction of confident candidates also sent to L4, to measure the cheap rungs |
| `KOLM_CODE_CALIBRATION_MIN_DISAGREEMENTS` | `3` | disagreements a metric needs before `bands --suggest` proposes a move |
| `KOLM_CODE_STAGE_TIMEOUT_MS` | `600000` | per-stage timeout (typecheck, lint, tests) |
| `KOLM_CODE_MODEL_TIMEOUT_MS` | `600000` | per-model-call timeout (planner, coder, L3, L4) |
| `KOLM_CODE_KEEP_WORKTREES` | `losers-destroyed` | `all`, `none` or `losers-destroyed` |
| `KOLM_CODE_BANDS_PATH` | bundled `src/ladder/bands.json` | an operator-owned bands file |

### Context

What the models are allowed to see, and how much of it.

| Variable | Default | ARMS |
|----------|---------|------|
| `KOLM_CODE_CONTEXT_TOKENS` | `0` | transcript compaction: the coder's context window is capped at this many tokens, and the transcript is compacted whenever its estimate exceeds 75% of that cap minus the 8000-token reply budget, never below a 1000-token floor. `6000` therefore compacts from about 1000 tokens on. `0` means never compact |
| `KOLM_CODE_RETRIEVAL_BYTES` | `16384` | the byte budget for retrieved code snippets attached to a task prompt |

Retrieval is local: it greps the checkout for terms taken from the task text,
skips vendored directories, and never leaves the machine.

### Storage and worktrees

| Variable | Default | ARMS |
|----------|---------|------|
| `KOLM_DATA_DIR` | `~/.kolm` | the data root; the store lands at `<root>/code/ladder.db` |
| `KOLM_CODE_HOME` | `<KOLM_DATA_DIR>/code` | overrides just the kolm code directory |
| `KOLM_CODE_DB_PATH` | `<code home>/ladder.db` | an explicit store path |
| `KOLM_CODE_KEYS_PATH` | `<code home>/keys.json` | an explicit keystore path (section 3) |
| `KOLM_CODE_NO_KEYSTORE` | unset | `1` makes every verb ignore the keystore file |
| `KOLM_CODE_DRIVER` | `local` | `local` (git worktrees), `cloudflare` (sandbox), `fake` (tests) |
| `KOLM_CODE_SANDBOX_URL` | - | the Cloudflare sandbox worker; required by the `cloudflare` driver |
| `KOLM_CODE_SANDBOX_TOKEN` (+) | - | bearer token for that worker |
| `KOLM_CODE_SHELL` | Git-for-Windows `bash.exe` on win32, `/bin/sh` elsewhere | the shell the coder's `run` tool goes through |

### Runtime

Transport-level switches. These change how provider calls are made, not what
is asked.

| Variable | Default | ARMS |
|----------|---------|------|
| `KOLM_CODE_TEXT_TOOLS` | - | comma-separated refs forced onto the text tool protocol, for endpoints that advertise tool calling but ignore it (for example `ollama:qwen3-coder-30b`) |
| `KOLM_CODE_PROVIDER_CONCURRENCY` | built-in per-provider defaults | `name=n` pairs, comma separated, `1 <= n <= 64` (for example `ollama=1,anthropic=4`); names are case-insensitive |
| `KOLM_CODE_STREAM` | unset | streaming responses where the adapter supports them |

A malformed entry is a startup error, not a silent default: `ollama` with no
`=n`, `ollama=0` and `ollama=two` are all rejected by `loadConfig`.

### Benchmark

| Variable | Default | ARMS |
|----------|---------|------|
| `KOLM_CODE_BENCH_DIR` | `<code home>/bench` | where `kolm code bench` writes result files; `~` is expanded |

### GitHub pull requests

| Variable | Default | ARMS |
|----------|---------|------|
| `KOLM_GITHUB_TOKEN` (+) | - | `--pr`, highest precedence |
| `GITHUB_TOKEN` | - | `--pr`, fallback |
| `KOLM_GITHUB_APP_ID` + `KOLM_GITHUB_APP_INSTALLATION_ID` | - | `--pr` via a GitHub App installation token |
| `KOLM_GITHUB_APP_PRIVATE_KEY` or `KOLM_GITHUB_APP_PRIVATE_KEY_PATH` | `~/.truffle/truffle-pr-authority.private-key.pem` if present | the App's signing key |

### Truffle partner worker

| Variable | Default | ARMS |
|----------|---------|------|
| `TRUFFLE_PARTNER_API_URL` | - | the partner task API |
| `TRUFFLE_PARTNER_SERVICE_TOKEN` | - | authentication to that API |
| `TRUFFLE_PARTNER_WORKSPACE_ID` | - | which workspace to claim tasks from |
| `TRUFFLE_PARTNER_POLL_MS` | `5000` | claim poll interval |
| `KOLM_TRUFFLE_HARNESS_ID` | `kolm-code` | how this worker identifies itself |
| `TRUFFLE_PARTNER_PROCESSOR_URL` + `_KEY_ID` + `_SIGNING_SECRET` | - | the signed settlement callback |

### Viewer and test switches

| Variable | Default | ARMS |
|----------|---------|------|
| `KOLM_CODE_VIEWER_PUBLIC` | unset | allows `kolm code serve --host 0.0.0.0`; refused otherwise |
| `KOLM_TEST_MOCK_PROVIDER` | unset | replaces every provider adapter with a deterministic mock; used by the test suite |

---

## 5. Verbs

### `run`

```
kolm code run "<task>" [options]
```

Plans the task, generates `--candidates` patches in isolated worktrees, runs
each up the cascade, and keeps the winner's branch.

```
kolm code run "make the retry loop respect Retry-After" --repo ../service
kolm code run "port the parser to the new AST" --candidates 5 --max-cost 2
kolm code run "fix the flaky upload test" --mode direct --json
kolm code run "add the missing index" --coder ollama:qwen3.8:latest --pr
```

Options: `--repo <path>`, `--repo-url <url>`, `--base <ref>`,
`--mode cascade|direct`, `--driver local|cloudflare|fake`, `--candidates <n>`,
`--coder <ref,...>` (repeatable), `--reviewer <ref,...>`, `--judge <ref>`,
`--orchestrator <ref>`, `--max-cost <usd>`, `--keep-worktrees`, `--pr`,
`--pr-base <branch>`, `--json`, `--quiet`.

Every flag that names a model or a budget is applied by overlaying the
corresponding environment variable before the config is loaded, so a flag and
its variable can never disagree.

Stage events stream to **stderr** as one compact line each, so `--json` on
stdout stays pipeable:

```
[run.start] runId=r_20260908_bdc3b49a mode=cascade
[candidate.start c0a0] candidateId=c0a0 generatorRef=cerebras:qwen-3.8-27b rung=0
[coder.turn c0a0] turn 3 write_file,run
[l1-static c0a0] good 1.2s $0.0031
[run.end] PASS 41.8s
```

Ctrl-C aborts cleanly and prints the exact command to pick the run back up.

### `resume`

```
kolm code resume r_20260908_bdc3b49a
```

Continues an interrupted run from the store: candidates already verified are
not re-verified and not re-paid for. The task, repository path, mode and the
run's shape (candidates per rung, coder and repair turn caps) come from the
stored run row, so a resume never widens a run; only the models and the cost
cap can be changed on the resume command line, which is how you raise the
budget on a run that stopped on `budget`. In a split run, a subtask is carried
over only when it passed and left a branch behind; one that failed or was left
unresolved is run again (you see `subtask.retry` before its `subtask.start`),
which is what makes `kolm code resume <runId> --coder <stronger model>` the
intended recovery from a failed subtask. A run that already finished PASS
re-prints its stored outcome and runs nothing. A run that finished FAIL or
JUDGE is reopened instead: the resume says so (`run.resume` carries
`reopened`, `priorVerdict` and `priorStopReason`), the run goes back to
`running`, and nothing that was already coded is coded again. The coders on
the resume command line take over ABOVE the rungs the earlier attempt spent,
so a run whose one rung of two candidates failed sees the new coder write
`c2a0` and `c3a0`; naming the same model draws new candidates from it rather
than re-grading the old ones. When the reopened run ends, the receipt on disk
is replaced by the new one - it lists both attempts' candidates and the whole
run's cost - and the run's events keep both outcomes end to end.

### `status`

```
kolm code status              # every run, newest first
kolm code status <runId>      # one run's candidates and their latest stage calls
kolm code status --limit 50 --json
```

### `show`

```
kolm code show <runId>
kolm code show <runId> --json
kolm code show <runId> --json --events
```

Full detail: the candidate x stage grid, per-model and per-stage cost, the
event count, and the receipt path. Two lines describe what the run was given
and how it was routed: `context` (bytes of project instructions and how many
files they came from, how many lessons from earlier runs were attached, and the
size of the task-focused retrieval, with `(cached)` when it came from the
context cache) and `route` (the plan's complexity and risk, the starting rung,
the candidate count and repair rounds the router chose, and why). Runs recorded
before these events existed print neither line. `--events` swaps the `events` count for the
event list itself (`eventCount` still carries the number), which is how you
read back what each `coder.turn` actually called: every tool call carries its
name, the path or command it named (200 chars) and the head of what came back
(300 chars). File contents are never in there.

When the run was split into subtasks, `show` also prints a SUBTASK table -
status, verdict, winning model, branch and cost per subtask - and one
`integration` line with the merge verdict, how many branches merged and how
many conflicted. Runs that were not split print neither, and a run recorded by
an older version simply shows nothing there rather than failing.

### `funnel`

```
kolm code funnel
kolm code funnel --since 2026-09-01 --json
```

How many candidates each rung stopped, what fraction cleared cheaply, and how
many reached the judge. This is the number that says whether the cascade is
earning its complexity.

### `eval`

```
kolm code eval
kolm code eval --since 2026-09-01 --json
```

Cascade versus direct over the stored runs: verdict agreement, cost per run,
and where the cheap rungs disagreed with the judge. To generate the comparison
arm, run some tasks with `--mode direct`.

### `bench`

```
kolm code bench
kolm code bench --tasks fix-duration-units,crlf-edit
kolm code bench --concurrency 2 --json > bench-today.json
kolm code bench --compare bench-today.json
```

Runs the ten bundled tasks in `tests/fixtures/ladder-bench/` end to end: each
one is seeded into a throwaway git repository, solved by a full `run`, and then
scored by a hidden test the coder never saw. The run's own model flags
(`--coder`, `--reviewer`, `--judge`, `--orchestrator`, `--mode`, `--driver`,
`--candidates`, `--max-cost`, `--bands-path`, `--db`) all apply, which is how
you compare two ladders on the same ten tasks.

| Task | What its hidden test checks |
|------|-----------------------------|
| `fix-duration-units` | the seconds divisor is right for every duration, not only the sampled one |
| `add-cli-json-flag` | `--json` prints parseable JSON and the human output is unchanged without it |
| `crlf-edit` | the new default is present and every line ending in the touched file is still CRLF |
| `multi-file-rename` | the new name is exported, used at every call site, and the old name is gone |
| `refactor-extract-module` | the helpers moved to the new module, re-export intact, behaviour identical |
| `write-tests-for-untested` | the added tests actually exercise the edge cases, and the test script runs them |
| `flaky-baseline` | the new function works and the already-failing test is still failing, not deleted |
| `forbidden-path-trap` | the fix is in the source, not in the generated bundle that a lazy patch would edit |
| `two-subtask-feature` | both features exist and the integration test that uses them together passes |
| `python-fix-if-available` | the seconds part of the label is restored; skipped when python or pytest is missing |

A task whose requirement is missing is reported `skip`, not `fail`. Exit 0 when
every non-skipped task passed its hidden test, 2 when any did not.

`--compare <file>` reads an earlier `--json` document and prints a delta table
(task, before, after, and the cost difference), so a ladder change is a diff
rather than a memory. `--keep-work` leaves the seeded repositories on disk
under `KOLM_CODE_BENCH_DIR` instead of deleting them.

### `lessons`

```
kolm code lessons .
kolm code lessons https://github.com/acme/widget --json
kolm code lessons . --clear
```

Prints the lessons remembered for a repository: what a previous run learned
(usually from a repair round or a judge finding) and how often that lesson has
been reused. The repository argument is normalised the same way a run
normalises it, so a path and its clone URL address the same lesson set.

`--clear` deletes the lesson file for that repository. It works even with
`KOLM_CODE_LESSONS=0`, and reports whether a file was actually there. With
lessons disabled, a read prints an empty set and says so.

### `bands`

```
kolm code bands                 # print the thresholds in force
kolm code bands --suggest       # propose new ones from stored judged runs
kolm code bands --apply --yes   # write the proposal
```

`--apply` never touches the bundled `src/ladder/bands.json`. If you have not
set `KOLM_CODE_BANDS_PATH`, it writes `<code home>/bands.json` and tells you to
point that variable at it. Each changed threshold is printed as a before/after
line.

Two safety bands are `locked` and can never be moved by calibration:
`l2-dynamic.newly_failing_tests` and `l0-meta.secrets`. A band entry marked
`"locked": true` in the bands file gets the same treatment. `--suggest` prints
the keys it skipped and why (`locked`, or `n<3` when fewer than
`KOLM_CODE_CALIBRATION_MIN_DISAGREEMENTS` candidates disagreed on that key).

### `clean`

```
kolm code clean                       # every finished run
kolm code clean --run r_20260908_bdc3b49a
kolm code clean --older-than 7 --dry-run
kolm code clean --older-than 30 --json
```

Removes the worktrees finished runs left behind: `git worktree remove --force`
for every worktree registered under `<code home>/worktrees/<runId>`, then
`git worktree prune`, then the run directory itself. A run whose status is
still `running` is never touched, and no branch is ever deleted - the work on
`kolm/code/<runId>/<candidate>` survives the cleanup. `--dry-run` lists what
would go and removes nothing.

### `receipt`

```
kolm code receipt show <runId>
kolm code receipt verify <runId>
kolm code receipt verify ./receipts/r_20260908_bdc3b49a.json
```

Exit 0 if the signature verifies, 2 if it does not, 5 if the receipt is not
there.

### `keys`

```
kolm code keys set <name>      # value on stdin, --from-env, --from-file, or a hidden prompt
kolm code keys list [--json]
kolm code keys unset <name>
kolm code keys path
kolm code keys test [name] [--json]
```

The bring-your-own-key store: section 3 covers the file, the precedence and
the Windows permission caveat in full. `set`, `list`, `unset` and `path` are
local; `test` is the one subcommand that reaches the network, and it exits
non-zero when a key it tested came back `bad-key`.

### `doctor`

```
kolm code doctor
kolm code doctor --json
kolm code doctor --live
```

An ASCII table with one row per orchestrator, judge, coder rung, reviewer rung,
plus the driver, GitHub, Truffle, SQLite and git. Every row is ARMED or
NOT ARMED with the reason. The KEY column says where a provider secret came
from - `env`, `keystore` or blank. **Reasons name variables, never values.**

`--live` is opt-in and makes real network calls: an Ollama reachability probe,
`gh auth token`, a one-token `ping` completion against every armed provider
(latency and cost printed per row), and a create/destroy cycle of the local git
driver against a throwaway repository. Errors are caught per row so one dead
provider does not hide the rest.

Exit 0 when the orchestrator, at least one coder, the driver and SQLite are all
armed; exit 3 otherwise.

A second table, RUNTIME, prints the settings from the Coordination, Context,
Runtime and Benchmark groups of section 4 as they are actually in force:
routing mode and whether subtasks are on (with their concurrency), coach,
self-review and lessons on or off, the context and retrieval budgets, which
refs are forced onto text tools, the per-provider concurrency map, streaming,
and the bench directory. `--json` carries the same values under `runtime`.

### `serve`

```
kolm code serve
kolm code serve --port 8080 --host 127.0.0.1 --open
```

Starts the read-only viewer (section 12) and prints its URL. It binds
`127.0.0.1` and refuses `0.0.0.0` unless `KOLM_CODE_VIEWER_PUBLIC=1` is set,
because the viewer exposes diffs and stage output with no authentication.

### `truffle-worker`

```
kolm code truffle-worker
kolm code truffle-worker --max-tasks 5 --poll-ms 2000
```

Claims tasks from the Truffle partner API, runs each through `run`, and settles
it with the verdict, the receipt, the cost and the pull request URL. Exits 3
listing the missing variable names if the partner environment is not armed.

### `help`

`kolm code help`, `kolm code --help`, or no verb at all. Exit 0. An unrecognised
verb prints the same text and exits 64.

---

## 6. Exit codes

| Code | Meaning |
|------|---------|
| 0 | success; a `run` reached PASS |
| 1 | bad arguments or bad configuration |
| 2 | the gate said no: a run ended FAIL or JUDGE, a bench task failed its hidden test, or a receipt signature did not verify |
| 3 | a prerequisite is missing: something is not armed, or `node:sqlite` is unavailable |
| 4 | execution error, including an aborted run |
| 5 | not found: an unknown run id or a missing receipt |
| 64 | usage error: an unknown verb |

A `LadderError` never reaches you as a stack trace. It prints as
`kolm code: <code>: <message>` on stderr and maps to the code above.

---

## 7. Arming matrix

You can run the whole cascade with no paid API at all; you trade judgement
quality for it.

| Tier | Orchestrator / judge | Coders | L3 reviewer | Cost per run |
|------|----------------------|--------|-------------|--------------|
| Free | `ollama:qwen3.8:latest` | `ollama:qwen3.8:latest` | `ollama:qwen3.8:latest` | $0, local GPU time |
| Cheap | `cerebras:gpt-oss-120b` | `cerebras:qwen-3.8-27b` | `ollama:*` or `cerebras:gpt-oss-120b` | fractions of a cent |
| Mixed | `anthropic:claude-fable-5-1` | `deepseek:deepseek-v4-flash`, then `anthropic:claude-opus-5` | `deepseek:deepseek-v4-flash` | cents |
| Full | `anthropic:claude-fable-5-1` | the default three-rung ladder | `ollama:*` first | cents to low dollars |

DeepSeek can be reached directly (`DEEPSEEK_API_KEY`) or through OpenRouter
(`OPENROUTER_API_KEY` with `openrouter:deepseek/...` refs). L0 through L2 cost
nothing regardless of tier: they are your own type checker, linter and tests.

Model ref grammar is `<provider>:<model>`, split on the **first** colon only,
so `ollama:qwen3.8:latest` is provider `ollama`, model `qwen3.8:latest`.

---

## 8. Worktree drivers

`local` (default) uses real `git worktree` checkouts under
`<code home>/worktrees/<runId>/<candidateId>`. It is the right choice when the
repository is already on the machine and you trust the task.

`cloudflare` runs each candidate inside a Cloudflare sandbox container, which
is the right choice for untrusted tasks and for a worker with no local
checkout. Deploy it once with the steps in
[`deploy/kolm-code-sandbox/README.md`](../deploy/kolm-code-sandbox/README.md),
then set `KOLM_CODE_SANDBOX_URL` and `KOLM_CODE_SANDBOX_TOKEN`.

`fake` is an in-memory driver for tests. It never touches the filesystem.

---

## 9. Receipts

Every run writes a signed evidence receipt to
`<code home>/receipts/<runId>.json`. It records the task, the arming report as
it was at run time, every candidate, every stage decision with its metric
values and the band that produced the call, the model refs and their costs, the
winning diff hash, and the verdict.

The receipt names the repository by basename only (`version` 2 of the
`kolm-code-receipt-v1` schema); the full local path stays in the SQLite run row
and never travels with a receipt you hand to someone else. `verify` accepts
both version 1 and version 2 receipts.

The receipt is canonicalised with stable key ordering and signed with Ed25519,
so anyone with the public key can check that a receipt was not edited after the
fact:

```
kolm code receipt verify r_20260908_bdc3b49a
```

Verification is offline. It proves the receipt is unmodified and names the key
id that signed it. It does not re-run the cascade.

---

## 10. Cascade versus direct, and calibration

`--mode direct` skips L3 and sends every candidate to the judge. It is the
control arm: the point of the cascade is only made by measuring it against the
simpler thing.

```
kolm code run "<task>" --mode direct
kolm code eval
```

Independently, `KOLM_CODE_CALIBRATION_RATE` (default 0.1) sends a random tenth
of the candidates the cheap rungs were *confident* about to L4 anyway. That
sample is what tells you whether a band's thresholds are actually where the
judge's opinion changes.

```
kolm code bands --suggest
```

reads those judged runs and proposes thresholds that would have agreed with the
judge more often, printing a before/after line per metric. Review them, then
`kolm code bands --apply --yes`. The bundled file is never modified; set
`KOLM_CODE_BANDS_PATH` to the written file to put the new thresholds in force.

Bands cover `l0-meta` (`patch_lines`, `files_touched`), `l1-static`
(`new_type_errors`, `new_lint_errors`), `l2-dynamic` (`newly_failing_tests`,
`changed_line_coverage`) and `l3-semantic` (`rubric_score`,
`reviewer_confidence`). The good edge is inclusive and the bad edge is
exclusive.

---

## 11. Cost controls

* `--max-cost <usd>` / `KOLM_CODE_MAX_COST_USD` stops the run at a hard ceiling.
* `--candidates <n>` is the biggest single lever: cost is roughly linear in it.
* The generation ladder is ordered cheapest-first on purpose. Put a local or
  cheap model at rung 0 and the expensive model only at the last rung.
* L0 through L2 are free and stop most bad patches before any model is asked
  for an opinion. `kolm code funnel` shows what fraction that is on your repo.
* `KOLM_CODE_CALIBRATION_RATE=0` turns off calibration spend, at the price of
  losing the data that `bands --suggest` needs.
* `kolm code show <runId>` breaks cost down by model and by stage.

---

## 12. The viewer

```
kolm code serve --open
```

A single self-contained page served by `node:http` from the local store. No
external requests, no build step, no write access to anything: it reads runs,
events, the funnel, the eval table and receipts.

* left rail: runs, newest first
* funnel strip: L0 to L4 with what each rung stopped
* candidate x stage grid: click a cell for that stage's metrics and reasoning
* live event log over Server-Sent Events, so a run in progress updates as it goes
* receipt panel with the signature state

Keyboard: `j` and `k` move through the run list, `r` refreshes.

Bind address is `127.0.0.1`. `--host 0.0.0.0` is refused unless
`KOLM_CODE_VIEWER_PUBLIC=1`, because there is no authentication in front of it.

---

## 13. Resumability

A run is a row in SQLite from the moment it starts, and every candidate,
stage decision, cost line and event is written as it happens rather than at the
end. So:

* Ctrl-C is safe. The CLI aborts the in-flight work and prints
  `aborted; resume with: kolm code resume <runId>`.
* A crashed or killed process leaves a resumable run behind.
* `kolm code resume <runId>` re-verifies nothing that already has a decision.
* `KOLM_CODE_KEEP_WORKTREES=all` keeps the losing candidates' worktrees so you
  can inspect what the cascade rejected.
* A run that ends in an error destroys the worktrees it had open (unless
  `KOLM_CODE_KEEP_WORKTREES=all`); `kolm code clean` sweeps up whatever an
  earlier version, or a killed process, left behind.

---

## 14. Troubleshooting

**`sqlite-unavailable` / exit 3 from every store verb.** `node:sqlite` needs
Node 22.5 or newer. Check with `node -v`. There is no fallback driver; upgrade
Node.

**`ollama reachability unknown` in doctor.** Nothing is listening on
`KOLM_CODE_OLLAMA_URL`. Start `ollama serve` and confirm with
`curl http://127.0.0.1:11434/api/tags`. `doctor --live` probes it for real;
plain `doctor` deliberately does not make network calls.

**Worktree creation fails on Windows with a path error.** Paths under
`<code home>/worktrees/<runId>/<candidateId>` plus a deep repository can exceed
the 260-character limit. Either enable long paths
(`git config --global core.longpaths true`) or point `KOLM_DATA_DIR` at a short
root such as `C:\k`.

**The Cloudflare sandbox image will not build.** The build needs a running
Docker daemon. Without Docker, use `--driver local`.

**`--pr` says not armed.** Token order is `KOLM_GITHUB_TOKEN`, then
`GITHUB_TOKEN`, then the keystore (`kolm code keys set github`), then a GitHub
App installation token, then `gh auth token`.
`doctor` prints which of those is present. Note that the Truffle GitHub App is
a settlement identity, not a pull request author; arm the kolm App or a token
for `--pr`.

**A key I stored is not being used.** `kolm code keys list` says which
source wins. `env+keystore(env wins)` means a variable in this shell is
shadowing the stored key; unset it, or accept it. `(disabled by
KOLM_CODE_NO_KEYSTORE)` means the whole file is being ignored on purpose.
A stored key that reaches the provider but is rejected shows as `bad-key` in
`kolm code keys test`.

**A reviewer or judge "returned no JSON" on a thinking model.** The model
spent its whole token budget on reasoning and sent back an empty reply with
`finish_reason: length`. The adapter retries such a reply once with up to 4x
the budget (capped at 16000 tokens) and records the retry as
`meta.emptyRetry` on the stage result; if the retry is empty too, the reasoning
text is kept as the reply so the rung reports `unsure` with the evidence rather
than guessing. Raise `KOLM_CODE_MODEL_TIMEOUT_MS` if the larger call times out.

**A run ends FAIL immediately with no candidate diff.** The coder produced no
change, so L0 called it bad. Check the plan line in the event stream and make
the task text more specific about which file to touch.

**Every candidate fails the same acceptance criterion round after round.** The
criterion itself may be the defect - a plan can turn a false premise in the task
text ("the files use CRLF line endings", when they do not) into a criterion no
diff can ever satisfy, and the repair loop will then spend its whole budget
producing correct diffs that are rejected for the same reason. `kolm code` now
notices this. After a repair round that produced no winner, it compares the
failure reasons of the round with the failure reasons of the round before: a
criterion that every failing candidate named in both rounds, with nothing having
passed in either, is stuck, and a candidate whose reasons are byte-for-byte what
they were before made no progress. That emits one `repair.stuck` event carrying
the stuck criteria and the stalled candidates. If any criterion is stuck, the
orchestrator model is shown the task text, the plan, the best candidate's diff
and deterministic facts about the files the criterion names - byte count, line
count, how many CRLF and how many bare LF endings, BOM, trailing newline - and
asked once whether to keep, retire or rewrite each one. A retire or a rewrite
writes a `plan.criterion` event (`from`, `to`, `verdict`, `reason`) and produces
an amended plan for the rest of that subtask; the failing candidates are then
re-scored against the amended plan without being coded again, so a diff that was
already correct can win on the spot. Look for both events in
`kolm code show <runId> --events`. The audit runs at most once per subtask per
run, never on the first round, never when the budget is spent, and never when
there is no orchestrator model; if it keeps every criterion, the stalled
candidates are not repaired again and the run goes to escalation or FAIL as
before. Set `KOLM_CODE_AUDIT=0` to turn the pass off and keep the plan exactly
as written.

**The coder answers in prose and never calls a tool.** Some small models with a
native tool API describe the call instead of making it ("I will use the
`edit_file` function ... here's the command to do that"), and `coder.turn` shows
`toolCalls=[]` twice followed by `candidate.diff ... reason=no-tool-calls`.
`kolm code` handles this in two steps without being asked. On the second turn of
a session that has not called a tool yet, it moves that session - and only that
session - to the text tool protocol, where the tools and their JSON schemas are
written into the system prompt and the model answers with
`<<<tool name="edit_file">>> {...} <<<end>>>` blocks; the switch shows up as a
`coder.protocol` event and as `protocol: text` on `candidate.diff`. If the
attempt still ends with no tool call, the repair round is handed to another
model - the next rung, or the orchestrator model when it is not the same ref -
which shows up as a `coder.fallback` event, and `show` and the receipt then name
the fallback as that candidate's generator. To skip the first step and start a
model in the text protocol, name it in `KOLM_CODE_TEXT_TOOLS=<ref>` (comma
separated, wildcards allowed).

**Integration says "merge of <subtask> failed" with no conflict files.** Git
refused to start the merge at all, so no file conflicted and there is nothing a
coder could resolve; the usual cause is a merge left in progress
(`fatal: You have not concluded your merge (MERGE_HEAD exists).`), a branch that
no longer resolves, or an unrelated history. The reason git gave is in the
`integration.merge` event and in `kolm code show <runId>`. The run stops there
with `stopReason: integration-conflict` and a FAIL verdict that repeats the
reason, rather than reporting a merge that never happened. Every repaired
conflict is now committed in the integration worktree the moment the markers are
gone - even when the resolution keeps HEAD's side and leaves the tree unchanged
- so the next merge of the same level starts from a clean worktree.

**A subtask failed, its dependents show SKIPPED, and integration says FAIL
without judging anything.** That is the split run refusing to grade half a
change. A subtask that did not pass leaves no branch to merge, so every subtask
that depends on it has nothing to build on: those are skipped before their
ladder starts (a `subtask.skipped` event naming the dependencies they were
waiting on, and no cost), while independent subtasks still run and still merge.
The survivors are merged and committed, so the integration branch holds the work
that did land, but the cascade does not run on it: a partial diff is not the
change the plan describes, and judging it would report a verdict for something
nobody proposed. The run reports `stopReason: subtask-failed` and an
`integration.verdict` FAIL whose reasons name each missing subtask
("subtask alpha failed", "subtask beta skipped (depends on alpha)"). The
recovery is `kolm code resume <runId> --coder <stronger model>`: the passed
subtasks are carried over untouched, the failed one is retried with the better
model, and the dependents run once it lands.

---

## 15. Caveats

What is proven locally, on this machine, with no network:

* The CLI surface, its exit codes and its argument parsing.
* The viewer server, all of its routes, its SSE stream, its CSP headers and its
  refusal to bind a public interface.
* The full cascade end to end under `KOLM_TEST_MOCK_PROVIDER=1` with the fake
  worktree driver, including the store writes, the funnel and the receipt path.
* `doctor` never emitting a credential value.

What is environment-gated and therefore *not* exercised by the test suite:

* Every real provider call. Cerebras, DeepSeek, OpenRouter, Anthropic and
  Ollama are reached only when their keys or endpoints are present, and only
  `doctor --live` will make such a call on purpose.
* The Cloudflare sandbox driver, which needs a deployed worker.
* Pull request creation, which needs a real GitHub token or App installation.
* The Truffle partner worker, which needs the partner API and its signing
  secrets.

What the cascade does **not** claim:

* A PASS is a statement that five specific measurements came out inside their
  bands. It is not a proof of correctness, and the bands are only as good as
  the calibration sample behind them.
* Cost figures are estimates from a per-model price table, not billed amounts.
* The L3 and L4 rungs are model opinions. They are recorded with their reasoning
  so you can disagree with them, which is the point of the receipt.

## 16. Limitations

* Single repository per run. Cross-repository changes are not modelled.
* The coder works through a fixed tool set (read, write, list, run) inside its
  worktree; there is no interactive debugger and no network access from the
  coder.
* L2 relies on the repository's own test and coverage commands being
  detectable. If they are not, L2 returns `unsure` and the load moves up to L3
  and L4, which costs more.
* The viewer is read-only. There is no way to start, stop or re-run anything
  from it.
* No npm dependencies were added for any of this; where that constrained the
  implementation, it constrained it toward the standard library.
